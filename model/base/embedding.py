import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    """
    Split input data into patches and embed them
    
    Works for both CSI and ACF data, automatically handling different input shapes
    """
    
    def __init__(self, 
                 data_type='csi',
                 in_channels=1, 
                 patch_size=(4, 4), 
                 emb_dim=128, 
                 norm_layer=nn.LayerNorm):
        """
        Initialize patch embedding
        
        Args:
            data_type (str): 'csi' or 'acf'
            in_channels (int): Number of input channels
            patch_size (tuple): Size of patches (height, width)
            emb_dim (int): Embedding dimension
            norm_layer: Normalization layer
        """
        super().__init__()
        self.data_type = data_type
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        
        # Projection layer - map patches to embedding space
        self.proj = nn.Conv2d(
            in_channels, 
            emb_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.norm = norm_layer(emb_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input data of shape [B, C, H, W] 
               or [B, C, T, F] for time-frequency representation
        
        Returns:
            tokens: Tokens of shape [B, N, E] where N is number of patches and E is embedding dimension
        """
        # Ensure input is 4D tensor
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        B, C, H, W = x.shape
        
        # Apply patch embedding
        x = self.proj(x)  # [B, E, H//patch_h, W//patch_w]
        
        # Reshape to sequence of tokens
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        
        # Apply normalization
        x = self.norm(x)
        
        return x

class CSIEmbedding(nn.Module):
    """Specialized embedding layer for CSI data"""
    
    def __init__(self, win_len, feature_size, emb_dim=128, in_channels=1):
        super().__init__()
        
        # Determine appropriate patch size for CSI data
        self.patch_h = max(1, feature_size // 10)  # Freq dimension
        self.patch_w = max(1, win_len // 10)       # Time dimension
        
        self.embedding = PatchEmbedding(
            data_type='csi',
            in_channels=in_channels,
            patch_size=(self.patch_h, self.patch_w),
            emb_dim=emb_dim
        )
        
        # Calculate output sequence length
        self.num_patches = (win_len // self.patch_w) * (feature_size // self.patch_h)
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: CSI data of shape [B, C, T, F]
        
        Returns:
            Embedded sequence [B, N, E]
        """
        x = self.embedding(x)  # [B, N, E]
        
        # 动态调整位置嵌入大小以匹配序列长度
        seq_len = x.size(1)
        if seq_len != self.pos_embedding.size(1):
            # 使用插值来调整位置嵌入大小
            pos_embed = self.pos_embedding
            pos_embed = pos_embed.transpose(1, 2)  # [1, dim, seq_len]
            pos_embed = F.interpolate(pos_embed, size=seq_len, mode='linear')
            pos_embed = pos_embed.transpose(1, 2)  # [1, seq_len, dim]
            x = x + pos_embed
        else:
            x = x + self.pos_embedding
            
        return x

class ACFEmbedding(nn.Module):
    """Specialized embedding layer for ACF data"""
    
    def __init__(self, win_len, feature_size, emb_dim=128, in_channels=1):
        super().__init__()
        
        # Determine appropriate patch size for ACF data
        self.patch_h = max(1, feature_size // 10)  # Freq dimension 
        self.patch_w = max(1, win_len // 10)       # Time dimension
        
        self.embedding = PatchEmbedding(
            data_type='acf',
            in_channels=in_channels,
            patch_size=(self.patch_h, self.patch_w),
            emb_dim=emb_dim
        )
        
        # Calculate output sequence length
        self.num_patches = (win_len // self.patch_w) * (feature_size // self.patch_h)
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: ACF data of shape [B, C, T, F]
        
        Returns:
            Embedded sequence [B, N, E]
        """
        x = self.embedding(x)  # [B, N, E]
        
        # 动态调整位置嵌入大小以匹配序列长度
        seq_len = x.size(1)
        if seq_len != self.pos_embedding.size(1):
            # 使用插值来调整位置嵌入大小
            pos_embed = self.pos_embedding
            pos_embed = pos_embed.transpose(1, 2)  # [1, dim, seq_len]
            pos_embed = F.interpolate(pos_embed, size=seq_len, mode='linear')
            pos_embed = pos_embed.transpose(1, 2)  # [1, seq_len, dim]
            x = x + pos_embed
        else:
            x = x + self.pos_embedding
            
        return x
