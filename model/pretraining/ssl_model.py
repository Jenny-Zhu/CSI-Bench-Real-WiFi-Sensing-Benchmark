import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.vit import ViTBackbone
from ..backbone.cnn import CNNBackbone
from ..backbone.hybrid import HybridBackbone
from ..base.heads import ContrastiveHead, ReconstructionHead

class SSLModel(nn.Module):
    """
    Self-supervised learning model for CSI/ACF data
    
    Implements contrastive learning approach using different backbones
    """
    
    def __init__(self, 
                 data_type='csi',
                 backbone_type='vit',  # 'vit', 'cnn', or 'hybrid'
                 win_len=250,
                 feature_size=98,
                 in_channels=1,
                 emb_dim=128,
                 proj_dim=128,
                 temperature=0.1,
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
                **kwargs
            )
        elif backbone_type == 'cnn':
            self.backbone = CNNBackbone(
                data_type=data_type,
                in_channels=in_channels,
                feature_dim=emb_dim,
                img_size=(feature_size, win_len),
                **kwargs
            )
        elif backbone_type == 'hybrid':
            self.backbone = HybridBackbone(
                data_type=data_type,
                in_channels=in_channels,
                emb_dim=emb_dim,
                img_size=(feature_size, win_len),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
            
        # Projection head for contrastive learning
        self.projection_head = ContrastiveHead(
            in_features=emb_dim,
            out_features=proj_dim,
            hidden_dim=emb_dim*2,
            temperature=temperature
        )
        
    def forward(self, x1, x2=None):
        """
        Forward pass for SSL model
        
        Args:
            x1: First augmented view [B, C, H, W]
            x2: Second augmented view [B, C, H, W], optional
            
        Returns:
            If x2 is None, returns projections for x1
            Otherwise, returns (loss, z1, z2)
        """
        # Extract features from first view
        h1 = self.backbone(x1)
        z1 = self.projection_head(h1)
        
        # If no second view, just return projections
        if x2 is None:
            return z1
            
        # Extract features from second view
        h2 = self.backbone(x2)
        
        # Compute contrastive loss and return projections
        loss, z1, z2 = self.projection_head(h1, h2)
        
        return loss, z1, z2
    
    def get_representation(self, x):
        """
        Get representation before projection head
        
        Args:
            x: Input data [B, C, H, W]
            
        Returns:
            Feature representation [B, emb_dim]
        """
        return self.backbone(x)


class JointSSLModel(nn.Module):
    """
    Joint self-supervised learning model that combines contrastive learning with other pretext tasks
    
    Can combine contrastive learning with reconstruction, rotation prediction, etc.
    """
    
    def __init__(self, 
                 data_type='csi',
                 backbone_type='vit',  # 'vit', 'cnn', or 'hybrid'
                 win_len=250,
                 feature_size=98,
                 in_channels=1,
                 emb_dim=128,
                 proj_dim=128,
                 pretext_tasks=['contrastive', 'reconstruction'],
                 temperature=0.1,
                 **kwargs):
        super().__init__()
        
        self.data_type = data_type
        self.backbone_type = backbone_type
        self.pretext_tasks = pretext_tasks
        
        # Initialize backbone based on type
        if backbone_type == 'vit':
            self.backbone = ViTBackbone(
                data_type=data_type,
                win_len=win_len,
                feature_size=feature_size,
                in_channels=in_channels,
                emb_dim=emb_dim,
                **kwargs
            )
        elif backbone_type == 'cnn':
            self.backbone = CNNBackbone(
                data_type=data_type,
                in_channels=in_channels,
                feature_dim=emb_dim,
                img_size=(feature_size, win_len),
                **kwargs
            )
        elif backbone_type == 'hybrid':
            self.backbone = HybridBackbone(
                data_type=data_type,
                in_channels=in_channels,
                emb_dim=emb_dim,
                img_size=(feature_size, win_len),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
            
        # Add task-specific heads
        if 'contrastive' in pretext_tasks:
            self.contrastive_head = ContrastiveHead(
                in_features=emb_dim,
                out_features=proj_dim,
                hidden_dim=emb_dim*2,
                temperature=temperature
            )
            
        if 'reconstruction' in pretext_tasks:
            self.reconstruction_head = ReconstructionHead(
                in_features=emb_dim,
                img_size=(feature_size, win_len),
                channels=in_channels
            )
            
        if 'rotation' in pretext_tasks:
            self.rotation_head = nn.Linear(emb_dim, 4)  # 4 rotation angles (0°, 90°, 180°, 270°)
        
    def forward(self, x1, x2=None, mask=None, rotation=None):
        """
        Forward pass for joint SSL model
        
        Args:
            x1: First view [B, C, H, W]
            x2: Second view for contrastive learning [B, C, H, W], optional
            mask: Mask for masked autoencoding [B, C, H, W], optional
            rotation: Rotation labels [B], optional
            
        Returns:
            Dictionary of outputs for each active pretext task
        """
        outputs = {}
        
        # Extract features from first view
        h1 = self.backbone(x1)
        
        # Apply contrastive learning if enabled
        if 'contrastive' in self.pretext_tasks and x2 is not None:
            h2 = self.backbone(x2)
            contrastive_loss, z1, z2 = self.contrastive_head(h1, h2)
            outputs['contrastive'] = {
                'loss': contrastive_loss,
                'z1': z1,
                'z2': z2
            }
            
        # Apply reconstruction if enabled
        if 'reconstruction' in self.pretext_tasks and mask is not None:
            reconstruction = self.reconstruction_head(h1)
            # If mask is provided, we focus on reconstructing masked regions
            if mask is not None:
                masked_input = x1 * (1 - mask)  # Zero out masked regions
                masked_target = x1 * mask  # Keep only masked regions as target
                # MSE on masked regions
                recon_loss = F.mse_loss(reconstruction * mask, masked_target)
            else:
                recon_loss = F.mse_loss(reconstruction, x1)
                
            outputs['reconstruction'] = {
                'loss': recon_loss,
                'reconstruction': reconstruction
            }
            
        # Apply rotation prediction if enabled
        if 'rotation' in self.pretext_tasks and rotation is not None:
            rot_logits = self.rotation_head(h1)
            rot_loss = F.cross_entropy(rot_logits, rotation)
            outputs['rotation'] = {
                'loss': rot_loss,
                'logits': rot_logits
            }
            
        return outputs
    
    def get_representation(self, x):
        """
        Get representation before any task-specific head
        
        Args:
            x: Input data [B, C, H, W]
            
        Returns:
            Feature representation [B, emb_dim]
        """
        return self.backbone(x)


# For backward compatibility
class ViT_Parallel(nn.Module):
    """Legacy class for backward compatibility"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Map to new implementation
        self.model = SSLModel(backbone_type='vit', **kwargs)
        self.num_classes = kwargs.get('num_classes', 2)
        self.classifier = nn.Linear(kwargs.get('emb_dim', 128), self.num_classes)
        
    def forward(self, x1, x2=None):
        # 如果只有一个输入，视为监督学习模式
        if x2 is None:
            # 获取特征表示
            features = self.model.get_representation(x1)
            # 应用分类器
            return self.classifier(features)
        # 否则，传递给SSL模型
        else:
            return self.model(x1, x2)


class ViT_MultiTask(nn.Module):
    """Legacy class for backward compatibility"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Map to new implementation
        self.model = JointSSLModel(backbone_type='vit', **kwargs)
        self.num_classes = kwargs.get('num_classes', 2)
        self.classifier = nn.Linear(kwargs.get('emb_dim', 128), self.num_classes)
        
    def forward(self, x1, x2=None, mask=None):
        # 如果没有第二个输入和mask，视为监督学习模式
        if x2 is None and mask is None:
            # 获取特征表示
            features = self.model.get_representation(x1)
            # 应用分类器
            return self.classifier(features)
        # 否则，传递给SSL模型
        else:
            return self.model(x1, x2, mask)
