#!/usr/bin/env python3
"""
Evaluation Pipeline for Swin Transformer ASL Recognition.

This module provides comprehensive evaluation capabilities for ASL recognition
models trained on the ASLLVD dataset. It includes accuracy metrics, confusion
matrices, per-class analysis, and research-quality reporting.

Key Features:
- Comprehensive classification metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Per-class performance analysis
- Model comparison utilities
- Research paper-ready results export

Research Context:
- Dataset: ASLLVD (American Sign Language Lexicon Video Dataset)
- Model: Swin Transformer (Vision Transformer)
- Task: Sign-to-text recognition (36 classes: A-Z + 0-9)
- Target: Springer conference publication quality

Author: ASL Recognition Research Team
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

import yaml
from pathlib import Path

# Load configuration
config_path = Path(__file__).parent / "configs.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract commonly used config values
BATCH_SIZE = config['training']['batch_size']
NUM_WORKERS = config['hardware']['num_workers']
PIN_MEMORY = config['hardware']['pin_memory']
RESULTS_DIR = Path(config['logging']['results_dir'])
MODELS_DIR = Path(config['logging']['models_dir'])
EXPERIMENT_NAME = config['logging']['experiment_name_template']
from datasets.msasl_dataset import create_msasl_data_loaders
from models.cnn_lstm import create_cnn_lstm_model
from utils import (
    setup_logger, compute_classification_metrics, compute_per_class_metrics,
    plot_confusion_matrix, save_metrics_to_csv, Timer, save_experiment_results,
    load_checkpoint
)

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    dataset_name: str = "test",
    compute_per_class: bool = True,
    save_predictions: bool = False
) -> Tuple[Dict[str, float], pd.DataFrame, Dict[str, Any]]:
    """
    Comprehensive model evaluation.

    Args:
        model: PyTorch model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        dataset_name: Name of the dataset being evaluated
        compute_per_class: Whether to compute per-class metrics
        save_predictions: Whether to save individual predictions

    Returns:
        Tuple of (overall_metrics, per_class_df, predictions_dict)
    """
    logger = setup_logger(f"evaluate_{dataset_name}")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_video_paths = []

    logger.info(f"Evaluating model on {dataset_name} set ({len(data_loader)} batches)")

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if len(batch) == 2:
                videos, labels = batch
                video_paths = None
            else:
                videos, labels, video_paths = batch

            videos, labels = videos.to(device), labels.to(device)

            # Forward pass
            outputs = model(videos)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if video_paths:
                all_video_paths.extend(video_paths)

            # Progress logging
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(data_loader) - 1:
                logger.debug(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute overall metrics
    logger.info("Computing overall metrics...")
    overall_metrics = compute_classification_metrics(
        all_labels, all_preds, all_probs
    )

    # Add dataset info
    overall_metrics['dataset'] = dataset_name
    overall_metrics['num_samples'] = len(all_labels)
    overall_metrics['num_classes'] = len(np.unique(all_labels))

    # Compute per-class metrics
    per_class_df = None
    if compute_per_class:
        logger.info("Computing per-class metrics...")
        per_class_df = compute_per_class_metrics(all_labels, all_preds)

        # Add class names (using indices for MS-ASL)
        per_class_df['class_name'] = per_class_df.index.map(
            lambda x: f'class_{int(x)}' if str(x).isdigit() else x
        )

    # Prepare predictions dictionary
    predictions_dict = {
        'true_labels': all_labels.tolist(),
        'predicted_labels': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'video_paths': all_video_paths if all_video_paths else None
    }

    # Log results
    logger.info(f"Evaluation Results for {dataset_name}:")
    logger.info(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"  Macro F1: {overall_metrics['f1_macro']:.4f}")
    logger.info(f"  Top-5 Accuracy: {overall_metrics.get('top_5_accuracy', 'N/A')}")
    logger.info(f"  Samples: {overall_metrics['num_samples']}")

    return overall_metrics, per_class_df, predictions_dict

def evaluate_multiple_models(
    model_configs: List[Dict[str, Any]],
    data_loader: DataLoader,
    device: str,
    save_results: bool = True
) -> pd.DataFrame:
    """
    Evaluate multiple models for comparison.

    Args:
        model_configs: List of model configurations
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        save_results: Whether to save results to file

    Returns:
        DataFrame with comparison results
    """
    logger = setup_logger("model_comparison")
    results = []

    for config in model_configs:
        model_name = config['name']
        model_path = config.get('path')

        logger.info(f"Evaluating {model_name}...")

        try:
            if model_path:
                # Load pretrained model
                model, _ = load_checkpoint(Path(model_path), None)
            else:
                # Create model from config
                if 'baseline_cnn' in model_name.lower():
                    model = create_cnn_lstm_model()
                else:
                    model = config.get('model_class', lambda: None)()

            model = model.to(device)

            # Evaluate
            metrics, _, _ = evaluate_model(
                model, data_loader, device,
                dataset_name=f"{model_name}_test",
                compute_per_class=False
            )

            # Add model info
            metrics['model_name'] = model_name
            metrics['model_type'] = config.get('type', 'unknown')

            results.append(metrics)
            logger.info(f"✅ {model_name} evaluation completed")

        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            results.append({
                'model_name': model_name,
                'error': str(e)
            })

    # Create comparison DataFrame
    df = pd.DataFrame(results)

    if save_results:
        output_path = RESULTS_DIR / "model_comparison.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Model comparison saved to {output_path}")

    return df

# =============================================================================
# VISUALIZATION AND REPORTING
# =============================================================================

def generate_evaluation_report(
    overall_metrics: Dict[str, float],
    per_class_df: pd.DataFrame,
    experiment_name: str,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.

    Args:
        overall_metrics: Overall evaluation metrics
        per_class_df: Per-class metrics DataFrame
        experiment_name: Name of the experiment
        save_path: Path to save report

    Returns:
        Dictionary containing the full report
    """
    logger = setup_logger("report_generator")

    report = {
        'experiment_name': experiment_name,
        'timestamp': pd.Timestamp.now().isoformat(),
        'overall_metrics': overall_metrics,
        'summary': {}
    }

    # Generate summary statistics
    summary = {
        'accuracy': overall_metrics.get('accuracy', 0),
        'macro_f1': overall_metrics.get('f1_macro', 0),
        'macro_precision': overall_metrics.get('precision_macro', 0),
        'macro_recall': overall_metrics.get('recall_macro', 0),
        'top5_accuracy': overall_metrics.get('top_5_accuracy'),
        'num_classes': overall_metrics.get('num_classes', NUM_CLASSES),
        'num_samples': overall_metrics.get('num_samples', 0)
    }

    # Per-class analysis
    if per_class_df is not None and not per_class_df.empty:
        summary.update({
            'worst_class_f1': per_class_df['f1-score'].min(),
            'best_class_f1': per_class_df['f1-score'].max(),
            'avg_class_f1': per_class_df['f1-score'].mean(),
            'classes_below_50_f1': (per_class_df['f1-score'] < 0.5).sum(),
            'class_variance_f1': per_class_df['f1-score'].var()
        })

        # Add per-class details
        report['per_class_metrics'] = per_class_df.to_dict('records')

    report['summary'] = summary

    # Log summary
    logger.info("EVALUATION REPORT SUMMARY")
    logger.info("=" * 50)
    for key, value in summary.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 50)

    # Save report
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"✅ Evaluation report saved to {save_path}")

    return report

def create_visualizations(
    predictions_dict: Dict[str, Any],
    experiment_name: str,
    save_dir: Path = RESULTS_DIR
) -> None:
    """
    Create evaluation visualizations.

    Args:
        predictions_dict: Dictionary with predictions
        experiment_name: Name of the experiment
        save_dir: Directory to save visualizations
    """
    logger = setup_logger("visualizations")

    true_labels = np.array(predictions_dict['true_labels'])
    pred_labels = np.array(predictions_dict['predicted_labels'])

    # Confusion matrix
    # Skip confusion matrix for large numbers of classes (MS-ASL has 1000 classes)
    if len(np.unique(true_labels)) <= 50:  # Only plot for small numbers of classes
        logger.info("Generating confusion matrix...")
        class_names = [f'class_{i}' for i in range(len(np.unique(true_labels)))]
        plot_confusion_matrix(
            true_labels, pred_labels,
            class_names=class_names,
            normalize='true',
            save_path=save_dir / f"{experiment_name}_confusion_matrix.png"
        )
    else:
        logger.info("Skipping confusion matrix visualization (too many classes for MS-ASL)")

    # Per-class performance bar chart
    try:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average=None, labels=range(NUM_CLASSES)
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Precision
        axes[0].bar(range(NUM_CLASSES), precision, color='skyblue')
        axes[0].set_title('Per-Class Precision')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Precision')
        axes[0].set_xticks(range(NUM_CLASSES))
        axes[0].set_xticklabels([f'class_{i}' for i in range(NUM_CLASSES)], rotation=45)

        # Recall
        axes[1].bar(range(NUM_CLASSES), recall, color='lightgreen')
        axes[1].set_title('Per-Class Recall')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Recall')
        axes[1].set_xticks(range(NUM_CLASSES))
        axes[1].set_xticklabels([f'class_{i}' for i in range(NUM_CLASSES)], rotation=45)

        # F1-Score
        axes[2].bar(range(NUM_CLASSES), f1, color='salmon')
        axes[2].set_title('Per-Class F1-Score')
        axes[2].set_xlabel('Class')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_xticks(range(NUM_CLASSES))
        axes[2].set_xticklabels([f'class_{i}' for i in range(NUM_CLASSES)], rotation=45)

        plt.tight_layout()
        plt.savefig(save_dir / f"{experiment_name}_per_class_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✅ Per-class metrics visualization saved")

    except Exception as e:
        logger.warning(f"Could not create per-class visualizations: {e}")

# =============================================================================
# RESEARCH PAPER UTILITIES
# =============================================================================

def generate_paper_tables(
    results_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> str:
    """
    Generate LaTeX tables for research paper.

    Args:
        results_df: DataFrame with model comparison results
        save_path: Path to save LaTeX tables

    Returns:
        LaTeX table string
    """
    logger = setup_logger("paper_tables")

    # Select relevant columns
    table_cols = ['model_name', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'top_5_accuracy']
    display_names = {
        'model_name': 'Model',
        'accuracy': 'Accuracy',
        'precision_macro': 'Precision',
        'recall_macro': 'Recall',
        'f1_macro': 'F1-Score',
        'top_5_accuracy': 'Top-5 Acc'
    }

    # Format table
    table_df = results_df[table_cols].copy()
    table_df = table_df.rename(columns=display_names)

    # Format numeric columns
    for col in table_df.columns:
        if col != 'Model':
            table_df[col] = table_df[col].apply(lambda x: '.1f' if pd.notnull(x) else '-')

    # Generate LaTeX
    latex_table = table_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="ASL Recognition Model Comparison on ASLLVD Dataset",
        label="tab:model_comparison"
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(latex_table)
        logger.info(f"✅ LaTeX table saved to {save_path}")

    return latex_table

def ablation_study_report(
    ablation_results: Dict[str, Dict[str, float]],
    baseline_metrics: Dict[str, float],
    save_path: Optional[Path] = None
) -> Dict[str, float]:
    """
    Generate ablation study analysis.

    Args:
        ablation_results: Results from different ablation configurations
        baseline_metrics: Baseline model metrics
        save_path: Path to save report

    Returns:
        Dictionary with ablation analysis
    """
    logger = setup_logger("ablation_study")

    analysis = {}

    for config_name, metrics in ablation_results.items():
        analysis[config_name] = {}

        for metric_name in ['accuracy', 'f1_macro']:
            if metric_name in metrics and metric_name in baseline_metrics:
                baseline_val = baseline_metrics[metric_name]
                config_val = metrics[metric_name]
                delta = config_val - baseline_val

                analysis[config_name][f'{metric_name}_delta'] = delta
                analysis[config_name][f'{metric_name}_percent_change'] = (delta / baseline_val) * 100

                logger.info(f"{config_name} {metric_name}: {config_val:.4f} "
                          f"(Δ={delta:+.4f}, {delta/baseline_val:+.1%})")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump({
                'baseline': baseline_metrics,
                'ablations': ablation_results,
                'analysis': analysis
            }, f, indent=2)
        logger.info(f"✅ Ablation study report saved to {save_path}")

    return analysis

# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

def run_evaluation(
    model_path: Optional[str] = None,
    experiment_name: str = "evaluation",
    dataset_split: str = "test",
    save_results: bool = True,
    should_create_visualizations: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Main evaluation function.

    Args:
        model_path: Path to model checkpoint (if None, uses default)
        experiment_name: Name for this evaluation run
        dataset_split: Which dataset split to evaluate on
        save_results: Whether to save results to files
        should_create_visualizations: Whether to create visualization plots
        **kwargs: Additional configuration overrides

    Returns:
        Dictionary with evaluation results
    """
    # Override config with kwargs
    global_vars = globals()
    for key, value in kwargs.items():
        if key.upper() in global_vars:
            global_vars[key.upper()] = value

    # Setup logging
    logger = setup_logger(f"eval_{experiment_name}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    if model_path is None:
        model_path = MODELS_DIR / f"{EXPERIMENT_NAME}_best.pth"

    logger.info(f"Loading model from {model_path}")
    model, checkpoint_info = load_checkpoint(Path(model_path), None)

    # Create data loader
    logger.info("Creating data loader...")
    train_loader, val_loader, test_loader = create_msasl_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        cache_dir=None
    )

    # Select dataset
    if dataset_split == "test":
        eval_loader = test_loader
    elif dataset_split == "val":
        eval_loader = val_loader
    elif dataset_split == "train":
        eval_loader = train_loader
    else:
        raise ValueError(f"Unknown dataset split: {dataset_split}")

    # Run evaluation
    with Timer("Model evaluation", logger):
        overall_metrics, per_class_df, predictions_dict = evaluate_model(
            model, eval_loader, device, dataset_name=dataset_split
        )

    # Generate report
    report = generate_evaluation_report(
        overall_metrics, per_class_df, experiment_name,
        save_path=RESULTS_DIR / f"{experiment_name}_report.json" if save_results else None
    )

    # Create visualizations
    if should_create_visualizations:
        create_visualizations(
            predictions_dict, experiment_name,
            save_dir=RESULTS_DIR
        )

    # Save predictions
    if save_results:
        pred_save_path = RESULTS_DIR / f"{experiment_name}_predictions.json"
        with open(pred_save_path, 'w') as f:
            json.dump(predictions_dict, f, indent=2)
        logger.info(f"✅ Predictions saved to {pred_save_path}")

    # Final summary
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 50)
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {dataset_split}")
    logger.info(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"Macro F1: {overall_metrics['f1_macro']:.4f}")
    if 'top_5_accuracy' in overall_metrics:
        logger.info(f"Top-5 Accuracy: {overall_metrics['top_5_accuracy']:.4f}")
    logger.info("=" * 50)

    return {
        'overall_metrics': overall_metrics,
        'per_class_metrics': per_class_df.to_dict('records') if per_class_df is not None else None,
        'predictions': predictions_dict,
        'report': report,
        'checkpoint_info': checkpoint_info
    }

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(description="Evaluate ASL Recognition Model")

    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--experiment_name', type=str, default="evaluation",
                       help='Name for this evaluation run')
    parser.add_argument('--dataset_split', type=str, default="test",
                       choices=['train', 'val', 'test'],
                       help='Which dataset split to evaluate on')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    parser.add_argument('--no_visualizations', action='store_true',
                       help='Do not create visualization plots')


    args = parser.parse_args()

    # Run evaluation
    results = run_evaluation(
        model_path=args.model_path,
        experiment_name=args.experiment_name,
        dataset_split=args.dataset_split,
        save_results=not args.no_save,
        should_create_visualizations=not args.no_visualizations
    )

    print("\n🎯 Evaluation completed!")
    print(f"Accuracy: {results['overall_metrics']['accuracy']:.4f}")
    print(f"Macro F1: {results['overall_metrics']['f1_macro']:.4f}")

if __name__ == "__main__":
    main()
