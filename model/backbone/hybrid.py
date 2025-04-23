import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base.cnn import CSIConvEncoder, ACFConvEncoder
from ..base.transformer import TransformerEncoder

class HybridBackbone(nn.Module):
    """
    Hybrid CNN+Transformer backbone for feature extraction
    
    Uses CNN for local feature extraction and Transformer for global modeling
    """
    
    def __init__(self, 
                 data_type='csi',
                 in_channels=1,
                 base_channels=32,
                 emb_dim=128,
                 img_size=(64, 100),  # (H, W) or (T, F)
                 transformer_depth=4,
                 num_heads=4,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        
        self.data_type = data_type
        
        # CNN feature extractor
        if data_type == 'csi':
            self.conv_encoder = CSIConvEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_blocks=2  # Reduced number of blocks
            )
        else:  # 'acf'
            self.conv_encoder = ACFConvEncoder(
                in_channels=in_channels,
                base_channels=base_channels,
                num_blocks=2  # Reduced number of blocks
            )
            
        # Calculate feature dimensions after CNN
        # Assume 2 blocks with stride 2 in one of them -> reduction by factor of 2
        feature_h = img_size[0] // 2
        feature_w = img_size[1] // 2
        # Final channels is base_channels * 2 (doubles once)
        self.conv_channels = base_channels * 2
        
        # Project CNN features to embedding dimension
        self.proj = nn.Conv2d(self.conv_channels, emb_dim, kernel_size=1)
        
        # Create class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        # Position embedding for transformer
        num_patches = feature_h * feature_w + 1  # +1 for class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        
        # Transformer for global modeling
        self.transformer = TransformerEncoder(
            emb_dim=emb_dim,
            depth=transformer_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Initialize parameters
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x, return_all_tokens=False):
        """
        Extract features using hybrid CNN+Transformer approach
        
        Args:
            x: Input data [B, C, H, W] or [B, C, T, F]
            return_all_tokens: Whether to return all transformer tokens
            
        Returns:
            Features [B, emb_dim] if return_all_tokens=False
            else [B, N, emb_dim] where N is the number of tokens
        """
        # Ensure input is 4D tensor
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        B = x.shape[0]
        
        # Extract CNN features
        features = self.conv_encoder(x)
        x = features[-1]  # Use final feature map
        
        # Project to embedding dimension
        x = self.proj(x)  # [B, emb_dim, H, W]
        
        # Reshape to sequence of tokens
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, emb_dim]
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        if return_all_tokens:
            return x
        else:
            # Use class token as global representation
            return x[:, 0]
