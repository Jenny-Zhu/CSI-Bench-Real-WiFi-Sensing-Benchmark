import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConvBlock(nn.Module):
    """Basic convolution block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block for CNN"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CSIConvEncoder(nn.Module):
    """Convolutional encoder for CSI data"""
    
    def __init__(self, in_channels=1, base_channels=16, num_blocks=3, use_residual=False):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        # First convolution block
        self.blocks.append(BasicConvBlock(in_channels, base_channels))
        
        # Add more convolution blocks, each doubling the number of channels
        for i in range(1, num_blocks):
            in_ch = base_channels * (2 ** (i-1))
            out_ch = base_channels * (2 ** i)
            
            if use_residual:
                self.blocks.append(ResidualBlock(in_ch, out_ch, stride=2))
            else:
                self.blocks.append(nn.Sequential(
                    BasicConvBlock(in_ch, out_ch, stride=2),  # Downsample
                    BasicConvBlock(out_ch, out_ch)
                ))
        
    def forward(self, x):
        """
        Args:
            x: Input data [B, C, H, W]
            
        Returns:
            List of feature maps, from low to high level
        """
        features = []
        
        for block in self.blocks:
            x = block(x)
            features.append(x)
            
        return features
    
class ACFConvEncoder(nn.Module):
    """Convolutional encoder for ACF data"""
    
    def __init__(self, in_channels=1, base_channels=16, num_blocks=3, use_residual=False):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        # First convolution block
        self.blocks.append(BasicConvBlock(in_channels, base_channels))
        
        # Add more convolution blocks, each doubling the number of channels
        for i in range(1, num_blocks):
            in_ch = base_channels * (2 ** (i-1))
            out_ch = base_channels * (2 ** i)
            
            if use_residual:
                self.blocks.append(ResidualBlock(in_ch, out_ch, stride=2))
            else:
                self.blocks.append(nn.Sequential(
                    BasicConvBlock(in_ch, out_ch, stride=2),  # Downsample
                    BasicConvBlock(out_ch, out_ch)
                ))
    
    def forward(self, x):
        """
        Args:
            x: Input data [B, C, H, W]
            
        Returns:
            List of feature maps, from low to high level
        """
        features = []
        
        for block in self.blocks:
            x = block(x)
            features.append(x)
            
        return features
