#!/usr/bin/env python3
"""
MS-ASL Training Script

Standard training script for MS-ASL ASL recognition using Swin Transformer and CNN-LSTM models.

Primary Workflow:
    # Train Swin Transformer (default model)
    python3 train_msasl.py --experiment_name swin_msasl_run

    # Train CNN-LSTM baseline
    python3 train_msasl.py --model_type cnn_lstm --experiment_name cnn_lstm_run

    # Resume training
    python3 train_msasl.py --experiment_name swin_resume --resume models/swin_msasl_run_best.pth

Configuration:
- Loads epochs and batch_size from configs.yaml
- Override with --epochs and --batch_size flags
- Default model: Swin Transformer (Video Swin Transformer)
- Dataset: MS-ASL (1000 classes)
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def load_config(config_path: str = "configs.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def create_experiment_name(model_type: str, timestamp: str = None) -> str:
    """Create experiment name with timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_{timestamp}"

def main():
    """Main training script for MS-ASL."""
    parser = argparse.ArgumentParser(description="Train models on MS-ASL dataset")

    # Model configuration
    parser.add_argument('--model_type', type=str, default='swin',
                       choices=['swin', 'cnn_lstm'],
                       help='Type of model to train')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Specific model name (for Swin Transformer)')

    # Training configuration
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Unique name for this experiment (auto-generated if not provided)')
    parser.add_argument('--config', type=str, default='configs.yaml',
                       help='Configuration file path')

    # Override config values (optional)
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (overrides config)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Advanced options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose logging')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default=None,
                       help='W&B project name')

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1

    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.model_name is not None:
        config['model']['swin']['name'] = args.model_name
    if args.wandb_project is not None:
        config['logging']['wandb_project'] = args.wandb_project

    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = create_experiment_name(args.model_type, timestamp)

    # Print training configuration
    model_display_name = "Swin Transformer" if args.model_type == 'swin' else "CNN-LSTM"
    if args.model_name:
        model_display_name += f" ({args.model_name})"

    print("🚀 Starting MS-ASL Training")
    print("=" * 50)
    print(f"Model: {model_display_name}")
    print(f"Dataset: MS-ASL ({config['dataset']['num_classes']} classes)")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Experiment: {args.experiment_name}")
    if args.resume:
        print(f"Resume: {args.resume}")
    print("=" * 50)

    # Import training modules
    try:
        from train import train_model
    except ImportError as e:
        print(f"❌ Failed to import training module: {e}")
        return 1

    # Run training
    try:
        results = train_model(
            model_type=args.model_type,
            experiment_name=args.experiment_name,
            resume_checkpoint=args.resume,
            use_wandb=args.use_wandb,
            config=config,
            debug=args.debug
        )

        print("\n✅ Training completed!")
        print(f"Best validation accuracy: {results['best_accuracy']:.4f}")
        if 'test_metrics' in results:
            print(f"Final test accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"Model saved to: models/{args.experiment_name}_best.pth")

        return 0

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
