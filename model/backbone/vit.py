import torch
import torch.nn as nn
from ..base.embedding import CSIEmbedding, ACFEmbedding
from ..base.transformer import TransformerEncoder

class ViTBackbone(nn.Module):
    """
    Vision Transformer backbone for feature extraction
    
    Can process both CSI and ACF data, outputting feature vectors
    """
    
    def __init__(self, 
                 data_type='csi',
                 win_len=250,
                 feature_size=98,
                 in_channels=1,
                 emb_dim=128,
                 depth=6,
                 num_heads=4,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 **kwargs):
        super().__init__()
        
        self.data_type = data_type
        self.emb_dim = emb_dim
        
        # Choose embedding based on data type
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
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.normal_(self.cls_token, std=0.02)
            
        # Transformer encoder backbone
        self.encoder = TransformerEncoder(
            emb_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Final normalization
        self.norm = nn.LayerNorm(emb_dim)
        
    def forward(self, x, return_all_tokens=False):
        """
        Extract features from input data
        
        Args:
            x: Input data [B, C, H, W] or [B, C, T, F]
            return_all_tokens: Whether to return all tokens or just the class token
            
        Returns:
            Features [B, emb_dim] if return_all_tokens=False
            else [B, N, emb_dim] where N is the number of tokens
        """
        # Ensure input is 4D tensor
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        # Embedding
        x = self.input_embed(x)
        
        # 添加class token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Transformer encoding
        x = self.encoder(x)
        
        # 应用最终的Layer Normalization
        x = self.norm(x)
        
        if return_all_tokens:
            return x
        else:
            # Use first token (class token) as global representation
            return x[:, 0]
