#!/usr/bin/env python3
"""
Swin Transformer Model for Video ASL Recognition.

This module implements a Swin Transformer-based architecture for American Sign
Language (ASL) recognition using video data. The model processes sequences of
video frames and classifies them into 1000 ASL signs.

Key Features:
- Pretrained Swin Transformer backbone (swin_base_patch4_window7_224)
- Temporal aggregation for video understanding
- Classification head for 1000 ASL classes
- Support for mixed precision training

Dataset: MS-ASL (1000 classes)
Architecture: Swin Transformer + Temporal Pooling

Author: ASL Recognition Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm library required for Swin Transformer models")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model hyperparameters
NUM_CLASSES = 1000  # MS-ASL classes
IMG_SIZE = 224     # Input image size
DROP_RATE = 0.1    # Dropout rate

# =============================================================================
# TEMPORAL AGGREGATION MODULES
# =============================================================================

class TemporalPooling(nn.Module):
    """Temporal pooling for aggregating frame-level features."""

    def __init__(self, pooling_type: str = 'mean'):
        """
        Initialize temporal pooling.

        Args:
            pooling_type: Type of pooling ('mean', 'max', 'attention')
        """
        super().__init__()
        self.pooling_type = pooling_type

        if pooling_type == 'attention':
            # Simple attention mechanism for temporal aggregation
            self.attention = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Tanh(),
                nn.Linear(512, 1),
                nn.Softmax(dim=1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal pooling.

        Args:
            x: Input tensor of shape (batch, frames, features)

        Returns:
            Pooled tensor of shape (batch, features)
        """
        if self.pooling_type == 'mean':
            return torch.mean(x, dim=1)
        elif self.pooling_type == 'max':
            return torch.max(x, dim=1)[0]
        elif self.pooling_type == 'attention':
            # Compute attention weights
            attn_weights = self.attention(x)  # (batch, frames, 1)
            # Apply attention
            weighted_sum = torch.sum(x * attn_weights, dim=1)
            return weighted_sum
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

class VideoTransformer(nn.Module):
    """Video transformer for temporal modeling."""

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize video transformer.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()

        # Positional encoding for temporal dimension
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, embed_dim))  # T=16 frames

        # Transformer layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_layers
        )

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply video transformer.

        Args:
            x: Input tensor of shape (batch, frames, features)

        Returns:
            Output tensor of shape (batch, frames, features)
        """
        # Add positional encoding
        x = x + self.pos_embedding

        # Apply transformer
        x = self.transformer(x)

        # Apply layer norm
        x = self.norm(x)

        return x

# =============================================================================
# SWIN TRANSFORMER ASL MODEL
# =============================================================================

class SwinTransformerASL(nn.Module):
    """
    Swin Transformer model for ASL recognition.

    This model uses a pretrained Swin Transformer backbone to extract features
    from individual video frames, then aggregates them temporally for video-level
    ASL classification with 1000 classes.
    """

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        temporal_pooling: str = 'mean',
        use_temporal_transformer: bool = False,
        dropout_rate: float = DROP_RATE,
        freeze_backbone: bool = False
    ):
        """
        Initialize Swin Transformer ASL model.

        Args:
            model_name: Name of the timm Swin Transformer model
            num_classes: Number of output classes (1000 for MS-ASL)
            pretrained: Whether to use pretrained weights
            temporal_pooling: Type of temporal pooling ('mean', 'max', 'attention')
            use_temporal_transformer: Whether to use transformer for temporal modeling
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.temporal_pooling = temporal_pooling
        self.use_temporal_transformer = use_temporal_transformer
        self.freeze_backbone = freeze_backbone

        if not HAS_TIMM:
            raise ImportError("timm library required for Swin Transformer models. Install with: pip install timm")

        # Load pretrained Swin Transformer backbone
        print(f"Loading Swin Transformer: {model_name}")
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                drop_rate=dropout_rate
            )
            print("✅ Backbone loaded successfully")
        except Exception as e:
            print(f"Failed to load backbone: {e}")
            raise

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[-1]

        print(f"Backbone feature dimension: {self.feature_dim}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✅ Backbone frozen")

        # Temporal modeling
        if use_temporal_transformer:
            self.temporal_transformer = VideoTransformer(
                embed_dim=self.feature_dim,
                num_heads=8,
                num_layers=4,
                dropout=dropout_rate
            )

        # Temporal pooling
        self.temporal_pool = TemporalPooling(temporal_pooling)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim // 2, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Temporal pooling: {temporal_pooling}")
        print(f"Temporal transformer: {use_temporal_transformer}")

    def _initialize_weights(self):
        """Initialize model weights."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch, frames, 3, H, W)

        Returns:
            Output logits of shape (batch, num_classes)
        """
        batch_size, num_frames, channels, height, width = x.shape

        # Reshape for frame-wise processing: (batch * frames, channels, H, W)
        x = x.view(-1, channels, height, width)

        # Extract features from each frame using backbone
        frame_features = self.backbone(x)  # (batch * frames, feature_dim)

        # Reshape back to video format: (batch, frames, feature_dim)
        frame_features = frame_features.view(batch_size, num_frames, -1)

        # Apply temporal transformer if enabled
        if self.use_temporal_transformer:
            frame_features = self.temporal_transformer(frame_features)

        # Apply temporal pooling
        video_features = self.temporal_pool(frame_features)  # (batch, feature_dim)

        # Classification
        logits = self.classifier(video_features)  # (batch, num_classes)

        return logits

    def get_feature_extractor(self) -> nn.Module:
        """Get feature extractor part of the model (backbone only)."""
        return self.backbone

    def get_classifier(self) -> nn.Module:
        """Get classification head of the model."""
        return self.classifier

# =============================================================================
# MODEL FACTORY FUNCTION
# =============================================================================

def create_swin_asl_model(
    model_name: str = "swin_base_patch4_window7_224",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    temporal_pooling: str = 'mean',
    use_temporal_transformer: bool = False,
    dropout_rate: float = DROP_RATE,
    freeze_backbone: bool = False
) -> SwinTransformerASL:
    """
    Factory function to create Swin Transformer ASL model.

    Args:
        model_name: Name of the timm Swin Transformer model
        num_classes: Number of output classes (1000 for MS-ASL)
        pretrained: Whether to use pretrained weights
        temporal_pooling: Type of temporal pooling ('mean', 'max', 'attention')
        use_temporal_transformer: Whether to use transformer for temporal modeling
        dropout_rate: Dropout rate for regularization
        freeze_backbone: Whether to freeze backbone parameters

    Returns:
        Configured SwinTransformerASL model
    """
    model = SwinTransformerASL(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        temporal_pooling=temporal_pooling,
        use_temporal_transformer=use_temporal_transformer,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )

    return model

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())

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

# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    # Test model creation
    print("Testing Swin Transformer ASL model creation...")

    try:
        # Create model
        model = create_swin_asl_model(
            model_name="swin_base_patch4_window7_224",
            num_classes=NUM_CLASSES,
            pretrained=True,
            temporal_pooling='mean',
            use_temporal_transformer=False,
            dropout_rate=DROP_RATE
        )

        # Test forward pass
        print("Testing forward pass...")
        batch_size = 2
        dummy_input = torch.randn(batch_size, 16, 3, IMG_SIZE, IMG_SIZE)  # T=16 frames

        with torch.no_grad():
            output = model(dummy_input)

        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output classes: {NUM_CLASSES}")

        # Model info
        total_params = count_parameters(model)
        model_size = get_model_size_mb(model)
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {model_size:.2f} MB")

        print("✅ Swin Transformer model test completed successfully")

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
