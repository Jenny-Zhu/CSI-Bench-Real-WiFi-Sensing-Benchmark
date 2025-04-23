import torch
import torch.nn as nn
from model import CSI2DCNN, CSITransformer

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
