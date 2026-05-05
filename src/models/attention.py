"""
Attention mechanisms for improving model focus on important features.
"""
import torch
from torch import nn


class AttentionLayer(nn.Module):
    """
    Self-attention layer for learning which features/timesteps are important.
    
    Allows the model to focus on relevant parts of the image or feature sequence.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
               or (batch_size, num_patches, hidden_size)
        
        Returns:
            Attention-weighted output of same shape
        """
        # Self-attention: Q, K, V all from same input
        attn_out, _ = self.attention(x, x, x)
        
        # Residual connection + layer norm
        return self.norm(x + attn_out)


class CNNWithAttention(nn.Module):
    """
    CNN with spatial attention mechanism.
    
    Learns to focus on important facial regions (eyes, head pose).
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        attention_channels: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Base CNN features
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            nn.MaxPool2d(2),
            self._conv_block(32, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 256),
            nn.AdaptiveAvgPool2d((7, 7)),  # Fixed spatial size for attention
        )
        
        # Channel attention (Squeeze-and-Excitation)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 256, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            images: Input images of shape (batch_size, 3, 224, 224)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features
        x = self.features(images)
        
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        # Classification
        return self.classifier(x)
    
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self(images)
            return torch.softmax(logits, dim=1)


class ResNetWithAttention(nn.Module):
    """
    ResNet50 with attention for transfer learning.
    
    Uses pre-trained ResNet50 backbone + attention layers + classification head.
    """
    
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        try:
            from torchvision import models
            
            # Load pre-trained ResNet50
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            
            # Remove classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            
        except ImportError:
            raise ImportError("torchvision required for ResNet. Install with: pip install torchvision")
        
        # Attention layer
        self.attention = AttentionLayer(2048)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet backbone.
        
        Args:
            images: Input images of shape (batch_size, 3, 224, 224)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features from ResNet
        features = self.backbone(images)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 2048)
        
        # Apply attention
        features = features.unsqueeze(1)  # (batch_size, 1, 2048)
        features = self.attention(features)  # (batch_size, 1, 2048)
        features = features.squeeze(1)  # (batch_size, 2048)
        
        # Classify
        return self.classifier(features)
    
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self(images)
            return torch.softmax(logits, dim=1)
