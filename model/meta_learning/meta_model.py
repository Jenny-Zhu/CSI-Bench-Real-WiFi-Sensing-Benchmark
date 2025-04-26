import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ..backbone.vit import ViTBackbone
from ..backbone.cnn import CNNBackbone
from ..backbone.hybrid import HybridBackbone
from ..base.heads import ClassificationHead
from .models import (
    MLPClassifier, LSTMClassifier, ResNet18Classifier, TransformerClassifier, ViTClassifier
)

class BaseMetaModel(nn.Module):
    def __init__(self, 
                 model_type='vit',  # 'mlp', 'lstm', 'resnet18', 'transformer', 'vit'
                 win_len=250,
                 feature_size=98,
                 in_channels=1,
                 emb_dim=128,
                 num_classes=5,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        if model_type == 'mlp':
            self.model = MLPClassifier(win_len=win_len, feature_size=feature_size, num_classes=num_classes, emb_dim=emb_dim)
        elif model_type == 'lstm':
            self.model = LSTMClassifier(feature_size=feature_size, num_classes=num_classes, emb_dim=emb_dim)
        elif model_type == 'resnet18':
            self.model = ResNet18Classifier(win_len=win_len, feature_size=feature_size, num_classes=num_classes, emb_dim=emb_dim)
        elif model_type == 'transformer':
            self.model = TransformerClassifier(feature_size=feature_size, num_classes=num_classes, emb_dim=emb_dim)
        elif model_type == 'vit':
            self.model = ViTClassifier(win_len=win_len, feature_size=feature_size, in_channels=in_channels, emb_dim=emb_dim, num_classes=num_classes, dropout=dropout, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x):
        return self.model(x)
    
    def adapt(self, support_x, support_y, num_inner_steps=1):
        """
        Adapt model to support set (inner loop of MAML)
        
        Args:
            support_x: Support set inputs [N*K, C, H, W] where N is num_classes and K is shots
            support_y: Support set labels [N*K]
            num_inner_steps: Number of inner loop update steps
            
        Returns:
            Adapted model parameters
        """
        # Create a copy of the model for adaptation
        fast_weights = {}
        for name, param in self.named_parameters():
            fast_weights[name] = param.clone()
            
        # Inner loop adaptation
        for step in range(num_inner_steps):
            # Forward pass with current fast weights
            features = self.backbone_with_params(support_x, fast_weights)
            logits = self.classifier_with_params(features, fast_weights)
            
            # Compute loss
            loss = F.cross_entropy(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
            # Update fast weights
            for (name, param), grad in zip(fast_weights.items(), grads):
                fast_weights[name] = param - self.inner_lr * grad
                
        return fast_weights
    
    def backbone_with_params(self, x, params):
        """
        Forward pass through backbone with specified parameters
        
        Args:
            x: Input data
            params: Dictionary of model parameters
            
        Returns:
            Features from backbone
        """
        # Extract backbone parameters
        backbone_params = {k: v for k, v in params.items() if k.startswith('backbone.')}
        
        # Create a temporary model for forward pass
        # This is a simplified implementation and may need to be customized
        # based on the specific backbone architecture
        
        # Here we assume x is already properly formatted
        if self.backbone_type == 'vit':
            # Apply ViT specific forward pass
            # (simplified for demonstration purposes)
            emb_params = {k.replace('backbone.input_embed.', ''): v for k, v in backbone_params.items() 
                          if k.startswith('backbone.input_embed.')}
            encoder_params = {k.replace('backbone.encoder.', ''): v for k, v in backbone_params.items() 
                              if k.startswith('backbone.encoder.')}
            
            # Manual forward pass with custom parameters
            # (This is a simplified example; actual implementation depends on model architecture)
            x = self.backbone.input_embed(x)  # Assume input_embed doesn't need custom params for simplicity
            # Forward through encoder with custom params would go here
            x = self.backbone.encoder(x)  # Simplified, ignoring custom params
            return x[:, 0]  # Return class token
        else:
            # For CNN and hybrid, we simply use the existing backbone
            # This is a fallback and might not correctly use the adapted parameters
            return self.backbone(x)
    
    def classifier_with_params(self, x, params):
        """
        Forward pass through classifier with specified parameters
        
        Args:
            x: Features from backbone
            params: Dictionary of model parameters
            
        Returns:
            Classification logits
        """
        # Extract classifier parameters
        classifier_params = {k.replace('classifier.', ''): v for k, v in params.items() 
                             if k.startswith('classifier.')}
        
        # Apply classifier weights directly
        if 'classifier.fc.weight' in params and 'classifier.fc.bias' in params:
            return F.linear(x, params['classifier.fc.weight'], params['classifier.fc.bias'])
        else:
            # Fallback to regular classifier
            return self.classifier(x)
            
    def meta_learning_forward(self, support_x, support_y, query_x):
        """
        MAML-style meta-learning forward pass
        
        Args:
            support_x: Support set inputs [N*K, C, H, W]
            support_y: Support set labels [N*K]
            query_x: Query set inputs [N*Q, C, H, W]
            
        Returns:
            Query set logits [N*Q, num_classes]
        """
        # Adapt to support set
        fast_weights = self.adapt(support_x, support_y)
        
        # Evaluate on query set using adapted weights
        features = self.backbone_with_params(query_x, fast_weights)
        logits = self.classifier_with_params(features, fast_weights)
        
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


class CSIMetaModel(BaseMetaModel):
    """Specialized meta-learning model for CSI data"""
    
    def __init__(self, **kwargs):
        super().__init__(data_type='csi', **kwargs)


class ACFMetaModel(BaseMetaModel):
    """Specialized meta-learning model for ACF data"""
    
    def __init__(self, **kwargs):
        super().__init__(data_type='acf', **kwargs)


# Legacy models for backward compatibility

class CSI2DCNN(nn.Module):
    """Legacy 2D CNN model for CSI data, for backward compatibility"""
    
    def __init__(self, in_channels=1, num_classes=5, **kwargs):
        super().__init__()
        # Map to new implementation but with CNN backbone
        self.model = CSIMetaModel(
            backbone_type='cnn',
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )
        
    def forward(self, x):
        return self.model(x)
    
    def meta_learning_forward(self, support_x, support_y, query_x):
        return self.model.meta_learning_forward(support_x, support_y, query_x)


class CSITransformer(nn.Module):
    """Legacy Transformer model for CSI data, for backward compatibility"""
    
    def __init__(self, in_channels=1, num_classes=5, **kwargs):
        super().__init__()
        # Map to new implementation with ViT backbone
        self.model = CSIMetaModel(
            backbone_type='vit',
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )
        
    def forward(self, x):
        return self.model(x)
    
    def meta_learning_forward(self, support_x, support_y, query_x):
        return self.model.meta_learning_forward(support_x, support_y, query_x)
