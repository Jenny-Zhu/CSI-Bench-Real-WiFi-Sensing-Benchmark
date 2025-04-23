import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.cnn import CNNBackbone
from ..base.heads import ClassificationHead, ContrastiveHead, ReconstructionHead

class CNNUnified(nn.Module):
    """
    Unified CNN model that can be used for all three pipelines
    
    Supports both CSI and ACF data
    Supports pretraining, supervised learning, and meta-learning
    """
    
    def __init__(self, 
                 task_type='supervised',   # 'pretraining', 'supervised', 'meta_learning'
                 data_type='csi',          # 'csi' or 'acf'
                 img_size=(64, 100),       # Image size (H, W)
                 in_channels=1,            # Input channels
                 base_channels=32,         # Base channel count
                 feature_dim=128,          # Feature dimension
                 num_classes=3,            # Number of classes for classification
                 dropout=0.1,              # Dropout probability
                 contrastive_dim=128,      # Projection dimension for contrastive learning
                 joint=False,              # Whether to use joint learning
                 **kwargs):
        super().__init__()
        
        self.task_type = task_type
        self.data_type = data_type
        self.joint = joint
        
        # CNN backbone
        self.backbone = CNNBackbone(
            data_type=data_type,
            in_channels=in_channels,
            base_channels=base_channels,
            feature_dim=feature_dim,
            img_size=img_size,
            dropout=dropout,
            **kwargs
        )
        
        # Task-specific heads
        
        # 1. Classification head - for supervised learning
        self.classifier = ClassificationHead(
            in_features=feature_dim,
            num_classes=num_classes,
            hidden_dim=feature_dim//2,
            dropout=dropout
        )
        
        # 2. Contrastive learning head - for self-supervised pretraining
        self.contrastive_head = ContrastiveHead(
            in_features=feature_dim,
            out_features=contrastive_dim,
            hidden_dim=feature_dim//2
        )
        
        # 3. Reconstruction head - for joint learning
        if joint:
            self.reconstruction_head = ReconstructionHead(
                in_features=feature_dim,
                img_size=img_size,
                channels=in_channels
            )
            
    def forward(self, x1, x2=None, mask=None, flag=None):
        """
        Unified forward function with flag parameter to handle different pipeline interfaces
        
        Args:
            x1: First input, required for all pipelines
            x2: Second input, required for pretraining pipeline for contrastive learning
            mask: Mask, used for masked modeling in joint learning
            flag: Indicator for which pipeline interface to use
                  - 'pretraining': Contrastive learning
                  - 'supervised': Classification
                  - 'supervised_acf': ACF classification
                  - 'meta': Meta-learning
                  - 'joint': Joint learning
                  
        Returns:
            Appropriate outputs based on flag
        """
        # If flag not specified, infer from task_type
        if flag is None:
            flag = self.task_type
            
        # Contrastive learning - self-supervised pretraining
        if flag == 'pretraining' or flag == 'ssl':
            h1 = self.backbone(x1)
            h2 = self.backbone(x2)
            
            z1 = self.contrastive_head(h1)
            z2 = self.contrastive_head(h2)
            
            return z1, z2
        
        # Supervised learning - classification
        elif flag == 'supervised' or flag == 'supervised_acf':
            features = self.backbone(x1)
            logits = self.classifier(features)
            return logits
        
        # Meta-learning
        elif flag == 'meta':
            features = self.backbone(x1)
            logits = self.classifier(features)
            return logits
        
        # Joint learning - contrastive + reconstruction
        elif flag == 'joint':
            h1 = self.backbone(x1)
            h2 = self.backbone(x2)
            
            z1 = self.contrastive_head(h1)
            z2 = self.contrastive_head(h2)
            
            # If mask provided, do masked modeling reconstruction
            if mask is not None and hasattr(self, 'reconstruction_head'):
                reconstructed = self.reconstruction_head(h1)
                return z1, z2, reconstructed
            
            return z1, z2
        
        # Default to image classifier
        else:
            features = self.backbone(x1)
            logits = self.classifier(features)
            return logits
    
    def get_params_count(self):
        """Get parameter count"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_embeddings(self, x):
        """Get embeddings for fine-tuning and feature extraction"""
        return self.backbone(x)
