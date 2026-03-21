# MS-ASL ASL Recognition System

This is a complete ASL recognition system using Swin Transformer and CNN-LSTM models trained on the Microsoft American Sign Language (MS-ASL) dataset.

## 🏗️ Project Structure

```
backend2/
├── datasets/                    # Dataset implementations
│   └── msasl_dataset.py        # MS-ASL dataset loader
├── models/                     # Model architectures
│   ├── swin_video.py          # Swin Transformer model
│   └── cnn_lstm.py            # CNN-LSTM baseline model
├── configs.yaml               # Configuration file
├── train_msasl.py             # Training script
├── evaluate.py                # Evaluation script
├── train.py                   # Core training pipeline
├── utils.py                   # Utilities and metrics
├── MS-ASL/                    # Dataset files
│   ├── MSASL_train.json
│   ├── MSASL_val.json
│   ├── MSASL_test.json
│   └── MSASL_classes.json
└── README_MSASL.md           # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Swin Transformer (Default Model)
```bash
python3 train_msasl.py --experiment_name swin_msasl_run
```

### 3. Train CNN-LSTM Baseline
```bash
python3 train_msasl.py --model_type cnn_lstm --experiment_name cnn_lstm_run
```

### 4. Evaluate Models
```bash
python3 evaluate.py --experiment_name swin_msasl_run
python3 evaluate.py --experiment_name cnn_lstm_run
```

## 📊 Dataset

**MS-ASL (Microsoft American Sign Language)**
- **Classes**: 1,000 ASL signs
- **Format**: JSON annotations with YouTube video links
- **Processing**: 16-frame temporal sampling, 224x224 spatial resolution
- **Training Split**: ~67,000 videos
- **Validation Split**: ~8,000 videos
- **Test Split**: ~7,000 videos

## 🏗️ Architecture

### Swin Transformer Model
- **Backbone**: Swin-Base (patch4, window7, 224)
- **Input**: (T=16, C=3, H=224, W=224)
- **Temporal Pooling**: Mean pooling across frames
- **Classification Head**: 1000 classes

### CNN-LSTM Baseline Model
- **CNN Backbone**: ResNet-like 3D CNN
- **LSTM**: Bidirectional, 512 hidden units
- **Input**: Same as Swin Transformer
- **Classification Head**: 1000 classes

## 🛠️ Configuration

Key hyperparameters in `config.py`:
```python
NUM_CLASSES = 1000          # MS-ASL classes
FRAME_COUNT = 16            # Temporal frames
IMG_SIZE = 224              # Spatial resolution
BATCH_SIZE = 4              # Training batch size
EPOCHS = 50                 # Training epochs
LEARNING_RATE = 1e-4        # AdamW learning rate
```

## 📈 Training Commands

### Swin Transformer Training
```bash
# Full training
python3 train_msasl.py --experiment_name swin_full_training

# Custom hyperparameters
python train_msasl.py \
    --experiment_name swin_custom \
    --batch_size 2 \
    --epochs 100 \
    --learning_rate 5e-5

# Resume training
python3 train_msasl.py \
    --experiment_name swin_resume \
    --resume models/swin_msasl_run_best.pth
```

### CNN-LSTM Baseline Training
```bash
# Train baseline model
python3 train_msasl.py \
    --model_type cnn_lstm \
    --experiment_name cnn_lstm_baseline
```

## 📊 Evaluation

Evaluate trained models:
```bash
# Evaluate Swin Transformer
python3 evaluate.py --experiment_name swin_msasl_run

# Evaluate CNN-LSTM
python3 evaluate.py --experiment_name cnn_lstm_run
```

Metrics computed:
- **Accuracy**: Top-1 classification accuracy
- **Top-5 Accuracy**: Top-5 classification accuracy
- **Precision/Recall/F1**: Per-class and macro-averaged
- **Confusion Matrix**: Full 1000x1000 matrix

## 🔬 Research Features

### Reproducibility
- Fixed random seeds (`SEED = 42`)
- Deterministic operations
- Config logging and saving
- Experiment tracking with unique names

### Model Comparison
- Swin Transformer vs CNN-LSTM baseline
- Automatic metric comparison
- Training curve visualization
- Per-class performance analysis

### Paper-Ready Outputs
- Springer conference format figures
- Comprehensive logging
- Statistical analysis
- Ablation study support

## 📁 Directory Structure

```
backend2/
├── MS-ASL/                 # Dataset annotations
│   ├── MSASL_train.json
│   ├── MSASL_val.json
│   ├── MSASL_test.json
│   └── MSASL_classes.json
├── models/                 # Saved checkpoints
├── results/                # Evaluation results
├── logs/                   # Training logs
├── paper_figures/         # Publication figures
├── config.py              # Configuration
├── dataset.py             # MS-ASL dataset class
├── model.py               # Model architectures
├── train.py               # Training pipeline
├── train_msasl.py         # Training script
├── evaluate.py            # Evaluation script
└── utils.py               # Utilities
```

## 🎯 Expected Performance

### Swin Transformer (Target)
- **Top-1 Accuracy**: 40-60% (expected range)
- **Top-5 Accuracy**: 65-80%
- **Macro F1**: 35-55%

### CNN-LSTM Baseline (Comparison)
- **Top-1 Accuracy**: 20-40%
- **Top-5 Accuracy**: 45-65%
- **Macro F1**: 15-35%

*Note: Actual performance depends on dataset quality, training duration, and hyperparameters.*

## 🔧 Troubleshooting

### Memory Issues
- Reduce batch size: `--batch_size 2`
- Use gradient accumulation
- Enable mixed precision (default)

### Video Loading Issues
- Ensure PyAV is installed: `pip install av`
- Check video file accessibility
- Videos are cached after first load

### Training Slow
- Use GPU if available
- Reduce model size (Swin-Tiny)
- Increase batch size if memory allows

## 📝 Citation

If you use this code in your research:

```bibtex
@misc{msasl_asl_recognition,
    title={MS-ASL American Sign Language Recognition using Vision Transformers},
    author={ASL Recognition Research Team},
    year={2024},
    howpublished={\url{https://github.com/your-repo/msasl-asl}}
}
```

## 📞 Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review configuration in `config.py`
3. Ensure all dependencies are installed
4. Verify MS-ASL dataset accessibility
