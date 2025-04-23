import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.embedding import CSIEmbedding, ACFEmbedding
from ..base.transformer import TransformerEncoder
from ..base.heads import ClassificationHead, ContrastiveHead, ReconstructionHead

class ViTUnified(nn.Module):
    """
    Unified Vision Transformer model that can be used for all three pipelines
    
    Supports both CSI and ACF data
    Supports pretraining, supervised learning, and meta-learning
    """
    
    def __init__(self, 
                 task_type='pretraining',  # 'pretraining', 'supervised', 'meta_learning'
                 data_type='csi',          # 'csi', 'acf'
                 win_len=250,              # Time window length
                 feature_size=98,          # Feature dimension
                 in_channels=1,            # Input channels
                 emb_dim=128,              # Embedding dimension
                 depth=6,                  # Transformer depth
                 num_heads=4,              # Number of attention heads
                 mlp_ratio=4.0,            # MLP expansion ratio
                 dropout=0.1,              # Dropout probability
                 num_classes=3,            # Number of classes for classification
                 contrastive_dim=128,      # Projection dimension for contrastive learning
                 joint=False,              # Whether to use joint learning
                 **kwargs):                # Other parameters
        super().__init__()
        
        self.task_type = task_type
        self.data_type = data_type
        self.joint = joint
        
        # Choose embedding layer based on data type
        if data_type == 'csi':
            self.input_embed = CSIEmbedding(
                win_len=win_len,
                feature_size=feature_size,
                emb_dim=emb_dim,
                in_channels=in_channels
            )
        else:  # 'acf'
            self.input_embed = ACFEmbedding(
                win_len=win_len,
                feature_size=feature_size,
                emb_dim=emb_dim,
                in_channels=in_channels
            )
        
        # Backbone network - Transformer encoder
        self.encoder = TransformerEncoder(
            emb_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Task-specific heads
        
        # 1. Classification head - for supervised learning
        self.classifier = ClassificationHead(
            in_features=emb_dim,
            num_classes=num_classes,
            hidden_dim=emb_dim*2,
            dropout=dropout
        )
        
        # 2. Contrastive learning head - for self-supervised pretraining
        self.contrastive_head = ContrastiveHead(
            in_features=emb_dim,
            out_features=contrastive_dim,
            hidden_dim=emb_dim*2
        )
        
        # 3. Reconstruction head - for joint learning
        if joint:
            self.reconstruction_head = ReconstructionHead(
                in_features=emb_dim,
                img_size=(feature_size, win_len),
                channels=in_channels
            )
    
    def encode(self, x):
        """
        Encode input data
        
        Args:
            x: Input data [B, C, H, W]
            
        Returns:
            Feature vector [B, emb_dim]
        """
        # Ensure input is 4D tensor
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Embedding
        x = self.input_embed(x)
        
        # Encoding
        x = self.encoder(x)
        
        # Use first token as global feature
        x = x[:, 0]
        
        return x
    
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
            z1 = self.encode(x1)
            z2 = self.encode(x2)
            
            z1_proj = self.contrastive_head(z1)
            z2_proj = self.contrastive_head(z2)
            
            return z1_proj, z2_proj
        
        # Supervised learning - classification
        elif flag == 'supervised' or flag == 'supervised_acf':
            features = self.encode(x1)
            logits = self.classifier(features)
            return logits
        
        # Meta-learning
        elif flag == 'meta':
            features = self.encode(x1)
            logits = self.classifier(features)
            return logits
        
        # Joint learning - contrastive + reconstruction
        elif flag == 'joint':
            z1 = self.encode(x1)
            z2 = self.encode(x2)
            
            z1_proj = self.contrastive_head(z1)
            z2_proj = self.contrastive_head(z2)
            
            # If mask provided, do masked modeling reconstruction
            if mask is not None and hasattr(self, 'reconstruction_head'):
                masked_input = x1 * mask
                reconstructed = self.reconstruction_head(z1)
                return z1_proj, z2_proj, reconstructed
            
            return z1_proj, z2_proj
        
        # Default to image classifier
        else:
            features = self.encode(x1)
            logits = self.classifier(features)
            return logits

    def get_params_count(self):
        """Get parameter count"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_embeddings(self, x):
        """Get embeddings for fine-tuning and feature extraction"""
        return self.encode(x)
