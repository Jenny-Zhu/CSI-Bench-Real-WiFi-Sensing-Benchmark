import torch
import torch.nn as nn
from model import CSI2DCNN, CSITransformer
from model.meta_learning.meta_model import BaseMetaModel

def load_csi_model_benchmark(H, W, device):
    """
    Load CSI model for benchmark meta-learning tasks.
    
    Args:
        H (int): Height of input images
        W (int): Width of input images
        device (torch.device): Device to load model on
        
    Returns:
        model (nn.Module): CSI2DCNN model
    """
    print(f"Loading CSI 2D CNN for meta-learning (H={H}, W={W})")
    
    # Simple ConvNet configuration
    model = CSI2DCNN(
        input_channels=1,
        img_size=(H, W),
        num_filters=32,
        filter_size=(3, 3),
        hidden_dims=[64, 32],
        num_classes=2
    ).to(device)
    
    # Initialize model (optional)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    print(f"Model loaded and initialized with parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
    return model

def load_meta_model(model_type, win_len, feature_size, n_way, device, **kwargs):
    """
    Load model for meta-learning tasks.
    
    Args:
        model_type (str): Model type ('mlp', 'lstm', 'resnet18', 'transformer', 'vit')
        win_len (int): Window length
        feature_size (int): Feature size
        n_way (int): N-way classification (number of classes)
        device (torch.device): Device to load model on
        **kwargs: Additional model parameters
        
    Returns:
        model (nn.Module): Meta-learning model
    """
    print(f"Loading {model_type.upper()} for meta-learning (win_len={win_len}, feature_size={feature_size})")
    
    # Default parameters
    params = {
        'in_channels': 1,
        'emb_dim': 128,
        'dropout': 0.1
    }
    
    # Update with provided kwargs
    params.update(kwargs)
    
    # Create meta-learning model
    model = BaseMetaModel(
        model_type=model_type,
        win_len=win_len,
        feature_size=feature_size,
        in_channels=params['in_channels'],
        emb_dim=params['emb_dim'],
        num_classes=n_way,
        dropout=params['dropout']
    ).to(device)
    
    print(f"Model loaded and initialized with parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            
    return model