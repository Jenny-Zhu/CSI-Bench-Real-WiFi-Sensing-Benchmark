import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class MLPClassifier(nn.Module):
    """Multi-layer Perceptron for WiFi sensing"""
    def __init__(self, win_len=250, feature_size=98, num_classes=2):
        super(MLPClassifier, self).__init__()
        # Calculate input size but limit it to prevent memory issues
        input_size = min(win_len * feature_size, 10000)
        
        self.win_len = win_len
        self.feature_size = feature_size
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Flatten input: [batch, channels, win_len, feature_size] -> [batch, win_len*feature_size]
        x = x.view(x.size(0), -1)
        # Limit input size if needed
        if x.size(1) > 10000:
            x = x[:, :10000]
        return self.fc(x)

class LSTMClassifier(nn.Module):
    """LSTM model for WiFi sensing"""
    def __init__(self, feature_size=98, hidden_size=256, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # Input shape: [batch, channels, win_len, feature_size]
        # LSTM expects: [batch, win_len, feature_size]
        x = x.squeeze(1)  # Remove channel dimension
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the final hidden state from both directions
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Classification
        out = self.fc(hidden_cat)
        return out

class ResNet18Classifier(nn.Module):
    """Modified ResNet-18 for WiFi sensing"""
    def __init__(self, win_len=250, feature_size=98, num_classes=2):
        super(ResNet18Classifier, self).__init__()
        
        # Load pretrained ResNet-18
        self.resnet = resnet18(pretrained=False)
        
        # Modify first conv layer to accept single channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final fc layer
        self.resnet.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # ResNet forward pass
        return self.resnet(x)

class TransformerClassifier(nn.Module):
    """Transformer model for WiFi sensing"""
    def __init__(self, feature_size=98, d_model=256, nhead=8, 
                 num_layers=4, dropout=0.1, num_classes=2):
        super(TransformerClassifier, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(feature_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes)
        )
        
    def forward(self, x):
        # Input shape: [batch, channels, win_len, feature_size]
        # Transform to: [batch, win_len, feature_size]
        x = x.squeeze(1)
        
        # Project to d_model dimensions
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Global average pooling over sequence length
        x = x.mean(dim=1)
        
        # Classification
        return self.classifier(x)

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



# --- Patch Embedding and Position Embedding ---
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=(4, 4), emb_dim=128, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(emb_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class ViTEmbedding(nn.Module):
    def __init__(self, win_len, feature_size, emb_dim=128, in_channels=1):
        super().__init__()
        patch_h = max(1, feature_size // 10)
        patch_w = max(1, win_len // 10)
        self.embedding = PatchEmbedding(
            in_channels=in_channels,
            patch_size=(patch_h, patch_w),
            emb_dim=emb_dim
        )
        self.num_patches = (win_len // patch_w) * (feature_size // patch_h)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        x = self.embedding(x)
        seq_len = x.size(1)
        if seq_len != self.pos_embedding.size(1):
            pos_embed = self.pos_embedding.transpose(1, 2)
            pos_embed = F.interpolate(pos_embed, size=seq_len, mode='linear')
            pos_embed = pos_embed.transpose(1, 2)
            x = x + pos_embed
        else:
            x = x + self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        assert self.head_dim * num_heads == emb_dim
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, E)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class TransformerBlock(nn.Module):
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
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, depth, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class ViTClassifier(nn.Module):
    def __init__(self, win_len=250, feature_size=98, in_channels=1, emb_dim=128, depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1, num_classes=2):
        super().__init__()
        self.embedding = ViTEmbedding(win_len, feature_size, emb_dim, in_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.encoder = TransformerEncoder(
            emb_dim=emb_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.embedding(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.encoder(x)
        x = self.norm(x)
        return self.classifier(x[:, 0])