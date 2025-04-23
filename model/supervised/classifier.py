import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.vit import ViTBackbone
from ..backbone.cnn import CNNBackbone
from ..backbone.hybrid import HybridBackbone
from ..base.heads import ClassificationHead

class BaseClassifier(nn.Module):
    """
    Base classifier model for supervised learning
    
    Implements a common structure for CSI and ACF classifiers
    """
    
    def __init__(self, 
                 data_type='csi',
                 backbone_type='vit',  # 'vit', 'cnn', or 'hybrid'
                 win_len=250,
                 feature_size=98,
                 in_channels=1,
                 emb_dim=128,
                 num_classes=3,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        
        self.data_type = data_type
        self.backbone_type = backbone_type
        
        # Initialize backbone based on type
        if backbone_type == 'vit':
            self.backbone = ViTBackbone(
                data_type=data_type,
                win_len=win_len,
                feature_size=feature_size,
                in_channels=in_channels,
                emb_dim=emb_dim,
                dropout=dropout,
                **kwargs
            )
        elif backbone_type == 'cnn':
            self.backbone = CNNBackbone(
                data_type=data_type,
                in_channels=in_channels,
                feature_dim=emb_dim,
                img_size=(feature_size, win_len),
                dropout=dropout,
                **kwargs
            )
        elif backbone_type == 'hybrid':
            self.backbone = HybridBackbone(
                data_type=data_type,
                in_channels=in_channels,
                emb_dim=emb_dim,
                img_size=(feature_size, win_len),
                dropout=dropout,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
            
        # Classification head
        self.classifier = ClassificationHead(
            in_features=emb_dim,
            num_classes=num_classes,
            hidden_dim=emb_dim*2,
            dropout=dropout
        )
        
    def forward(self, x):
        """
        Forward pass for classifier
        
        Args:
            x: Input data [B, C, H, W]
            
        Returns:
            Classification logits [B, num_classes]
        """
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_representation(self, x):
        """
        Get representation before classification head
        
        Args:
            x: Input data [B, C, H, W]
            
        Returns:
            Feature representation [B, emb_dim]
        """
        return self.backbone(x)
    
    def load_from_ssl(self, state_dict, strict=False):
        """
        Load weights from a self-supervised pretrained model
        
        Args:
            state_dict: State dict from SSL model
            strict: Whether to strictly enforce that the keys in state_dict match
        """
        # Filter weights to only include backbone
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.') or k.startswith('encoder.') or k.startswith('input_embed.'):
                backbone_dict[k] = v
        
        # Load filtered weights
        missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_dict, strict=strict)
        
        return missing_keys, unexpected_keys


class CSIClassifier(BaseClassifier):
    """Specialized classifier for CSI data"""
    
    def __init__(self, **kwargs):
        super().__init__(data_type='csi', **kwargs)


class ACFClassifier(BaseClassifier):
    """Specialized classifier for ACF data"""
    
    def __init__(self, **kwargs):
        super().__init__(data_type='acf', **kwargs)
