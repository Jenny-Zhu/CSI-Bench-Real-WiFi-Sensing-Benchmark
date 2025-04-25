import torch
import torch.nn as nn
from ..backbone.vit import ViTBackbone

class ViT_Parallel(nn.Module):
    """
    Vision Transformer for supervised learning
    
    This is a simplified version that only supports supervised learning mode
    """
    
    def __init__(self, 
                 win_len=250,
                 feature_size=98,
                 in_channels=1,
                 emb_dim=128,
                 depth=6,
                 num_heads=4,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 num_classes=2,
                 **kwargs):
        super().__init__()
        
        # Backbone for feature extraction
        self.backbone = ViTBackbone(
            data_type='csi',  # Default to CSI data
            win_len=win_len,
            feature_size=feature_size,
            in_channels=in_channels,
            emb_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Linear(emb_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass for supervised learning
        
        Args:
            x: Input data [B, C, H, W]
            
        Returns:
            Classification logits [B, num_classes]
        """
        # Get features from backbone
        features = self.backbone(x)
        
        # Apply classifier
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