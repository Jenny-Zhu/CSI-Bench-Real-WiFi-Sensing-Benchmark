import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, emb_dim, num_heads, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim, "emb_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input sequence [B, N, E]
            
        Returns:
            Attention output [B, N, E]
        """
        B, N, E = x.shape
        
        # Linear projections
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, D]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Attention output
        out = (attn @ v).transpose(1, 2).reshape(B, N, E)  # [B, N, E]
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

class TransformerBlock(nn.Module):
    """Transformer block"""
    
    def __init__(self, emb_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
        mlp_hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input sequence [B, N, E]
            
        Returns:
            Transformer block output [B, N, E]
        """
        # First residual connection
        x = x + self.attn(self.norm1(x))
        # Second residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    """Transformer encoder"""
    
    def __init__(self, emb_dim, depth, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input sequence [B, N, E]
            
        Returns:
            Encoder output [B, N, E]
        """
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x
