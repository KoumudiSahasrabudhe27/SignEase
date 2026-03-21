#!/usr/bin/env python3
"""
Utility functions for ASLLVD + Swin Transformer ASL Recognition System.

This module provides logging utilities, evaluation metrics, data visualization,
and helper functions for the vision-based ASL recognition pipeline.

Research Context:
- Dataset: ASLLVD (American Sign Language Lexicon Video Dataset)
- Model: Swin Transformer (Vision Transformer)
- Task: Sign-to-text recognition (36 classes: A-Z + 0-9)

Author: ASL Recognition Research Team
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import warnings

import yaml
from pathlib import Path

# Load basic configuration constants
config_path = Path(__file__).parent / "configs.yaml"
if config_path.exists():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    NUM_CLASSES = config['dataset']['num_classes']
    LOGS_DIR = Path(config['logging']['logs_dir'])
    RESULTS_DIR = Path(config['logging']['results_dir'])
    LOG_LEVEL = config['logging']['level']
    LOG_FORMAT = config['logging']['log_format']
    SAVE_LOGS = config['logging']['save_logs']
else:
    # Fallback values
    NUM_CLASSES = 1000
    LOGS_DIR = Path("logs")
    RESULTS_DIR = Path("results")
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    SAVE_LOGS = True

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def setup_logger(name: str = "asllvd_swin", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logger with both console and file handlers.

    Args:
        name: Logger name
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set log level
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if SAVE_LOGS:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = LOGS_DIR / f"{name}_{timestamp}.log"

        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def log_system_info(logger: logging.Logger):
    """Log system and hardware information."""
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)

    # PyTorch version and CUDA
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("Running on CPU")

    # Memory info
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            logger.info(f"GPU Memory: {free / 1024**3:.1f} GB free / {total / 1024**3:.1f} GB total")
        except Exception as e:
            logger.debug(f"Could not get GPU memory info: {e}")
    else:
        logger.info("CPU Memory: Check system resources manually")

    logger.info("=" * 60)

def log_config(logger: logging.Logger, config_dict: Dict[str, Any]):
    """Log configuration parameters."""
    logger.info("CONFIGURATION")
    logger.info("=" * 60)

    def log_dict(d: Dict[str, Any], prefix: str = ""):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                log_dict(value, prefix + "  ")
            else:
                logger.info(f"{prefix}{key}: {value}")

    log_dict(config_dict)
    logger.info("=" * 60)

# =============================================================================
# METRICS AND EVALUATION
# =============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (for AUC)
        average: Averaging method ('macro', 'micro', 'weighted')

    Returns:
        Dictionary of metrics
    """
    # Lazy imports so this module can be used in minimal environments
    # (e.g., local API + training) without requiring sklearn/pandas/seaborn.
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )
    from sklearn.preprocessing import label_binarize

    metrics: Dict[str, float] = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics[f'precision_{average}'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics[f'recall_{average}'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics[f'f1_{average}'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Top-K accuracy
    if y_prob is not None:
        for k in [1, 5]:  # Top-1 and Top-5 for MS-ASL
            top_k_pred = np.argsort(y_prob, axis=1)[:, -k:]
            top_k_correct = np.any(top_k_pred == y_true.reshape(-1, 1), axis=1)
            metrics[f'top_{k}_accuracy'] = np.mean(top_k_correct)

    # AUC (if probabilities available)
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            # Binarize labels for multi-class AUC
            y_true_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
            if y_true_bin.shape[1] == 1:
                # Binary case
                metrics['auc'] = roc_auc_score(y_true_bin, y_prob[:, 1])
            else:
                # Multi-class case
                metrics['auc_macro'] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
                metrics['auc_weighted'] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='weighted')
        except Exception as e:
            print(f"Warning: Could not compute AUC: {e}")

    return metrics

def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Any:
    """
    Compute per-class precision, recall, and F1-score.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Class names (uses class indices for MS-ASL if None)

    Returns:
        DataFrame with per-class metrics
    """
    import pandas as pd
    from sklearn.metrics import classification_report

    if class_names is None:
        # For MS-ASL, use string class indices
        class_names = [f'class_{i}' for i in range(NUM_CLASSES)]

    # Get classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(report).transpose()

    # Filter out averages
    per_class_df = df[~df.index.str.contains('macro|micro|weighted|accuracy')]

    # Add class names
    per_class_df['class'] = per_class_df.index
    per_class_df['class_name'] = [class_names[int(idx)] if idx.isdigit() else idx
                                  for idx in per_class_df.index]

    return per_class_df

def save_metrics_to_csv(
    metrics: Dict[str, float],
    per_class_df: Any,
    save_path: Path
):
    """Save evaluation metrics to CSV files."""
    import pandas as pd
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Overall metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(save_path / "overall_metrics.csv", index=False)

    # Per-class metrics
    per_class_df.to_csv(save_path / "per_class_metrics.csv", index=False)

    print(f"✅ Metrics saved to {save_path}")

# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: str = 'true',
    save_path: Optional[Path] = None
):
    """
    Plot and save confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Class names for display
        normalize: Normalization method ('true', 'pred', 'all', or None)
        save_path: Path to save the plot
    """
    # Lazy imports: seaborn + sklearn are optional for minimal environments
    from sklearn.metrics import confusion_matrix
    try:
        import seaborn as sns
    except ImportError as e:
        raise ImportError("plot_confusion_matrix requires seaborn. Install seaborn to use this feature.") from e

    if class_names is None:
        # For MS-ASL, use abbreviated class names
        class_names = [f'{i}' for i in range(NUM_CLASSES)]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.title(f'Confusion Matrix (Normalized: {normalize})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")

    plt.show()

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[Path] = None
):
    """
    Plot training curves for loss and metrics.

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_metrics: Training metrics per epoch
        val_metrics: Validation metrics per epoch
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Metrics plot (if available)
    if train_metrics and val_metrics and 'accuracy' in train_metrics:
        axes[1].plot(epochs, train_metrics['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, val_metrics['accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to {save_path}")

    plt.show()

# =============================================================================
# MODEL UTILITIES
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    save_path: Path
):
    """Save model checkpoint."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }

    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved to {save_path}")

def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """Load model checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"✅ Checkpoint loaded from {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Loss: {checkpoint.get('loss', 'Unknown'):.4f}")

    return checkpoint

# =============================================================================
# DATA UTILITIES
# =============================================================================

def get_class_distribution(labels: np.ndarray) -> Dict[str, int]:
    """Get class distribution from labels."""
    unique, counts = np.unique(labels, return_counts=True)
    distribution = {f'class_{int(i)}': int(count) for i, count in zip(unique, counts)}
    return distribution

def balance_dataset(
    X: np.ndarray,
    y: np.ndarray,
    max_samples_per_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance dataset by undersampling majority classes.

    Args:
        X: Feature array
        y: Label array
        max_samples_per_class: Maximum samples per class

    Returns:
        Balanced X and y arrays
    """
    unique_labels = np.unique(y)
    balanced_indices = []

    for label in unique_labels:
        label_indices = np.where(y == label)[0]

        if max_samples_per_class is None:
            max_samples = len(label_indices)
        else:
            max_samples = min(len(label_indices), max_samples_per_class)

        # Random sample
        selected_indices = np.random.choice(label_indices, max_samples, replace=False)
        balanced_indices.extend(selected_indices)

    balanced_indices = np.array(balanced_indices)
    np.random.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]

# =============================================================================
# TIME AND PERFORMANCE UTILITIES
# =============================================================================

class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self, name: str = "Timer", logger=None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"⏱️  Started {self.name}")
        else:
            print(f"⏱️  Started {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.logger:
            self.logger.info(f"⏱️  Completed {self.name} in {elapsed:.2f}s")
        else:
            print(f"⏱️  Completed {self.name} in {elapsed:.2f}s")
def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"

# =============================================================================
# RESEARCH PAPER UTILITIES
# =============================================================================

def save_experiment_results(
    experiment_name: str,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    per_class_df: Any,
    save_dir: Path = RESULTS_DIR
):
    """Save complete experiment results for research paper."""
    import pandas as pd
    experiment_dir = save_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(experiment_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # Save metrics
    save_metrics_to_csv(metrics, per_class_df, experiment_dir)

    # Save summary
    summary = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'config': config
    }

    with open(experiment_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"✅ Experiment results saved to {experiment_dir}")
    return experiment_dir

# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    # Test logging
    logger = setup_logger("test_utils")
    logger.info("Testing utils module")

    # Test metrics computation
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    y_prob = np.random.rand(6, 3)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    print("Test metrics:", metrics)

    # Test per-class metrics
    per_class_df = compute_per_class_metrics(y_true, y_pred, ['A', 'B', 'C'])
    print("Per-class metrics shape:", per_class_df.shape)

    print("✅ Utils module tests passed")


# =============================================================================
# LOCAL VIDEO PREPROCESSING (Swin input: 224x224)
# =============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_bgr_frames_to_swin_tensor(
    frames_bgr: List[np.ndarray],
    frame_count: int = 16,
    img_size: int = 224,
) -> torch.Tensor:
    """
    Convert a list of OpenCV BGR frames into a Swin-ready tensor.

    Output shape: (T, 3, img_size, img_size), float32, ImageNet-normalized.
    """
    if len(frames_bgr) == 0:
        raise ValueError("No frames provided")

    # Lazy import (opencv-python is only needed for this helper)
    import cv2

    # Sample/pad to T frames
    if len(frames_bgr) >= frame_count:
        idx = np.linspace(0, len(frames_bgr) - 1, frame_count).round().astype(int).tolist()
        frames_bgr = [frames_bgr[i] for i in idx]
    else:
        last = frames_bgr[-1]
        frames_bgr = frames_bgr + [last] * (frame_count - len(frames_bgr))

    frames_rgb = []
    for fr in frames_bgr:
        fr = cv2.resize(fr, (img_size, img_size), interpolation=cv2.INTER_AREA)
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(fr)

    x = np.stack(frames_rgb, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    x = (x - mean) / std
    x = np.transpose(x, (0, 3, 1, 2))  # (T, C, H, W)
    return torch.from_numpy(x)
