#!/usr/bin/env python3
"""
Core Training Pipeline for MS-ASL ASL Recognition.

This module implements the complete training pipeline for ASL recognition using
Swin Transformer and CNN-LSTM models on the MS-ASL dataset.

Key Features:
- AdamW optimizer with cosine learning rate scheduling
- Mixed precision training (AMP)
- Model checkpointing and resume training
- Comprehensive logging and metrics
- Train/Val/Test split evaluation

Author: ASL Recognition Research Team
"""

import os
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Import project modules
from datasets.msasl_dataset import create_msasl_data_loaders
from models.swin_video import create_swin_asl_model
from models.cnn_lstm import create_cnn_lstm_model
from utils import (
    setup_logger, Timer, format_time, compute_classification_metrics,
    save_checkpoint, load_checkpoint, log_system_info, log_config,
    plot_training_curves, save_experiment_results
)

# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def get_optimizer_and_scheduler(
    model: nn.Module,
    num_training_steps: int,
    config: dict
) -> Tuple[optim.Optimizer, lr_scheduler._LRScheduler]:
    """
    Create optimizer and learning rate scheduler.

    Args:
        model: PyTorch model
        num_training_steps: Total number of training steps
        config: Training configuration

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Separate backbone and classifier parameters for different learning rates
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if 'backbone' in name or 'cnn' in name and not getattr(model, 'freeze_backbone', False):
            backbone_params.append(param)
        else:
            classifier_params.append(param)

    # Create optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['training']['learning_rate'] * 0.1},  # Lower LR for backbone
        {'params': classifier_params, 'lr': config['training']['learning_rate']}       # Full LR for classifier
    ], weight_decay=config['training']['weight_decay'],
       betas=config['training']['betas'], eps=config['training']['eps'])

    # Create scheduler
    if config['training']['lr_scheduler'] == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps, eta_min=config['training']['min_lr']
        )
    elif config['training']['lr_scheduler'] == 'linear':
        scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=1.0,
            end_factor=config['training']['min_lr']/config['training']['learning_rate'],
            total_iters=num_training_steps
        )
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    return optimizer, scheduler

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    scaler: GradScaler,
    epoch: int,
    device: str,
    config: dict,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        epoch: Current epoch number
        device: Device to train on
        config: Training configuration
        logger: Logger instance

    Returns:
        Dictionary of training metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    epoch_start_time = time.time()

    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos, labels = videos.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(enabled=config['training']['use_amp']):
            outputs = model(videos)
            loss = nn.CrossEntropyLoss()(outputs, labels)

        # Backward pass
        if config['training']['use_amp']:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update scheduler
        if isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log progress
        if (batch_idx + 1) % config['training']['log_frequency'] == 0 or batch_idx == len(train_loader) - 1:
            batch_acc = 100. * correct / total
            logger.info(
                f"Epoch {epoch+1}/{config['training']['epochs']} | "
                f"Batch {batch_idx+1}/{len(train_loader)} | "
                f"Loss: {running_loss/(batch_idx+1):.4f} | "
                f"Acc: {batch_acc:.2f}%"
            )

    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    metrics = {
        'train_loss': epoch_loss,
        'train_accuracy': epoch_acc,
        'epoch_time': epoch_time
    }

    logger.info(
        f"Epoch {epoch+1} Training | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_acc:.2f}% | "
        f"Time: {format_time(epoch_time)}"
    )

    return metrics

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    config: dict,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Args:
        model: PyTorch model
        val_loader: Validation data loader
        device: Device to validate on
        config: Training configuration
        logger: Logger instance

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    epoch_start_time = time.time()

    for videos, labels in val_loader:
        videos, labels = videos.to(device), labels.to(device)

        # Forward pass
        with autocast(enabled=config['training']['use_amp']):
            outputs = model(videos)
            loss = nn.CrossEntropyLoss()(outputs, labels)

        running_loss += loss.item()

        # Get predictions and probabilities
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / len(val_loader)

    # Compute comprehensive metrics
    metrics = compute_classification_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )

    metrics['val_loss'] = epoch_loss
    metrics['val_time'] = epoch_time

    logger.info(
        f"Validation | Loss: {epoch_loss:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Top-5: {metrics.get('top_5_accuracy', 'N/A')} | "
        f"Time: {format_time(epoch_time)}"
    )

    return metrics

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model(
    model_type: str = 'swin',
    experiment_name: str = 'msasl_experiment',
    resume_checkpoint: Optional[str] = None,
    use_wandb: bool = False,
    config: dict = None,
    debug: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Main training function for MS-ASL ASL recognition models.

    Args:
        model_type: Type of model ('swin' or 'cnn_lstm')
        experiment_name: Name for this experiment
        resume_checkpoint: Path to checkpoint to resume from
        use_wandb: Whether to use Weights & Biases logging
        config: Configuration dictionary
        debug: Enable debug mode
        **kwargs: Additional configuration overrides

    Returns:
        Dictionary with training results and metrics
    """
    # Load default config if not provided
    if config is None:
        import yaml
        config_path = Path(__file__).parent / "configs.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Override config with kwargs
    for key, value in kwargs.items():
        if key in config['training']:
            config['training'][key] = value

    # Set random seeds for reproducibility
    torch.manual_seed(config['reproducibility']['seed'])
    torch.cuda.manual_seed(config['reproducibility']['seed']) if torch.cuda.is_available() else None
    np.random.seed(config['reproducibility']['seed'])

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = config['reproducibility']['deterministic']
    torch.backends.cudnn.benchmark = config['reproducibility']['benchmark']

    # Setup logging
    logger = setup_logger(f"train_{experiment_name}")
    if debug:
        logger.setLevel(logging.DEBUG)

    log_system_info(logger)
    log_config(logger, config)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    # Create data loaders
    logger.info("Creating MS-ASL data loaders...")
    with Timer("Data loader creation", logger):
        cache_dir = Path(config['logging']['cache_dir']) / experiment_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        train_loader, val_loader, test_loader = create_msasl_data_loaders(
            batch_size=config['training']['batch_size'],
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory'],
            cache_dir=str(cache_dir)
        )

    # Create model
    logger.info(f"Creating {model_type} model...")
    if model_type == 'swin':
        model = create_swin_asl_model(
            model_name=config['model']['swin']['name'],
            num_classes=config['dataset']['num_classes'],
            pretrained=config['model']['swin']['pretrained'],
            temporal_pooling=config['model']['swin']['temporal_pooling'],
            use_temporal_transformer=config['model']['swin']['temporal_transformer'],
            dropout_rate=config['model']['swin']['dropout'],
            freeze_backbone=config['model']['swin']['freeze_backbone']
        )
    elif model_type == 'cnn_lstm':
        model = create_cnn_lstm_model(
            num_classes=config['dataset']['num_classes'],
            hidden_size=config['model']['cnn_lstm']['hidden_size'],
            num_layers=config['model']['cnn_lstm']['num_layers'],
            dropout=config['model']['cnn_lstm']['dropout'],
            freeze_backbone=config['model']['cnn_lstm']['freeze_backbone']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Resume from checkpoint if provided
    start_epoch = 0
    best_accuracy = 0.0
    training_history = []

    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        model, checkpoint = load_checkpoint(Path(resume_checkpoint), model)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        training_history = checkpoint.get('training_history', [])

    # Setup optimizer and scheduler
    num_training_steps = len(train_loader) * (config['training']['epochs'] - start_epoch)
    optimizer, scheduler = get_optimizer_and_scheduler(model, num_training_steps, config)

    # Resume optimizer/scheduler state if available
    if resume_checkpoint and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if resume_checkpoint and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Setup mixed precision
    scaler = GradScaler() if config['training']['use_amp'] else None

    # Training loop
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    training_start_time = time.time()

    for epoch in range(start_epoch, config['training']['epochs']):
        epoch_start_time = time.time()

        # Train epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, epoch, device, config, logger
        )

        # Validate epoch
        val_metrics = validate_epoch(model, val_loader, device, config, logger)

        # Update scheduler (for non-cosine schedulers)
        if not isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6f}")

        # Compile epoch metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['train_loss'],
            'train_accuracy': train_metrics['train_accuracy'],
            'val_loss': val_metrics['val_loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_top5_accuracy': val_metrics.get('top_5_accuracy'),
            'learning_rate': current_lr,
            'epoch_time': time.time() - epoch_start_time
        }

        training_history.append(epoch_metrics)

        # Save best model
        current_accuracy = val_metrics['accuracy']
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            models_dir = Path(config['logging']['models_dir'])
            models_dir.mkdir(exist_ok=True)
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics['val_loss'], val_metrics,
                models_dir / f"{experiment_name}_best.pth"
            )
            logger.info(f"✅ New best model saved (accuracy: {best_accuracy:.4f})")

        # Save checkpoint
        if config['training']['save_checkpoints'] and (epoch + 1) % config['training']['checkpoint_frequency'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics['val_loss'], val_metrics,
                models_dir / f"{experiment_name}_epoch_{epoch+1}.pth"
            )

        # Early stopping
        if config['training']['early_stopping']:
            recent_epochs = training_history[-config['training']['early_stopping_patience']:]
            if len(recent_epochs) >= config['training']['early_stopping_patience']:
                recent_accs = [e['val_accuracy'] for e in recent_epochs]
                if recent_accs[-1] < max(recent_accs):
                    logger.info("Early stopping triggered")
                    break

    training_time = time.time() - training_start_time
    logger.info(f"Training completed in {format_time(training_time)}")

    # Load best model for final evaluation
    best_model_path = models_dir / f"{experiment_name}_best.pth"
    if best_model_path.exists():
        model, _ = load_checkpoint(best_model_path, model)

    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    test_metrics = validate_epoch(model, test_loader, device, config, logger)

    # Save training curves
    if config['logging']['save_logs']:
        results_dir = Path(config['logging']['results_dir'])
        results_dir.mkdir(exist_ok=True)

        train_losses = [h['train_loss'] for h in training_history]
        val_losses = [h['val_loss'] for h in training_history]
        train_accs = [h['train_accuracy'] for h in training_history]
        val_accs = [h['val_accuracy'] for h in training_history]

        plot_training_curves(
            train_losses, val_losses,
            {'accuracy': train_accs}, {'accuracy': val_accs},
            save_path=results_dir / f"{experiment_name}_training_curves.png"
        )

    # Save experiment results
    experiment_results = {
        'experiment_name': experiment_name,
        'model_type': model_type,
        'config': config,
        'training_history': training_history,
        'best_accuracy': best_accuracy,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'total_epochs': len(training_history)
    }

    save_experiment_results(
        experiment_name,
        config,
        test_metrics,
        None,  # per_class_df would need to be computed
        results_dir
    )

    # Final summary
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Best validation accuracy: {best_accuracy:.4f}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1-score: {test_metrics['f1_macro']:.4f}")
    logger.info(f"Training time: {format_time(training_time)}")
    logger.info(f"Model saved to: {models_dir / f'{experiment_name}_best.pth'}")
    logger.info("=" * 60)

    return experiment_results

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MS-ASL ASL recognition models")

    parser.add_argument('--model_type', type=str, default='swin', choices=['swin', 'cnn_lstm'])
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    results = train_model(
        model_type=args.model_type,
        experiment_name=args.experiment_name,
        resume_checkpoint=args.resume,
        config=config,
        debug=args.debug
    )

    print("\n🎉 Training completed successfully!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Test accuracy: {results['test_metrics']['accuracy']:.4f}")