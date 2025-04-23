import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.cnn import CSIConvEncoder, ACFConvEncoder

class CNNBackbone(nn.Module):
    """
    CNN backbone for feature extraction
    
    Can process both CSI and ACF data, outputting feature vectors
    """
    
    def __init__(self, 
                 data_type='csi',
                 in_channels=1,
                 base_channels=32,
                 feature_dim=128,
                 img_size=(64, 100),  # (H, W) or (T, F)
                 dropout=0.0,
                 use_residual=False,
                 **kwargs):
        super().__init__()
        
        self.data_type = data_type
        
        # Choose encoder based on data type
        if data_type == 'csi':
            self.encoder = CSIConvEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_blocks=3,
                use_residual=use_residual
            )
        else:  # 'acf'
            self.encoder = ACFConvEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_blocks=3,
                use_residual=use_residual
            )
            
        # Calculate feature size after CNN encoding
        # Assume 3 blocks with stride 2 in two of them -> reduction by factor of 4
        feature_h = img_size[0] // 4
        feature_w = img_size[1] // 4
        # Final channels is base_channels * 4 (doubles twice)
        conv_out_dim = base_channels * 4 * feature_h * feature_w
        
        # Feature projection to desired dimension
        self.feature_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, return_features_maps=False):
        """
        Extract features from input data
        
        Args:
            x: Input data [B, C, H, W] or [B, C, T, F]
            return_features_maps: Whether to return intermediate feature maps
            
        Returns:
            Features [B, feature_dim] if return_features_maps=False
            else list of feature maps from low to high level
        """
        # Ensure input is 4D tensor
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        # Get feature maps from encoder
        feature_maps = self.encoder(x)
        
        if return_features_maps:
            return feature_maps
        
        # Use final feature map for projection
        x = feature_maps[-1]
        
        # Project to feature vector
        x = self.feature_proj(x)
        
        return x
