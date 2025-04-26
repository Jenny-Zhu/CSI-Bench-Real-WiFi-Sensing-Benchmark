import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet18

# --- MLP ---
class MLPClassifier(nn.Module):
    def __init__(self, win_len=250, feature_size=98, num_classes=2, emb_dim=128):
        super().__init__()
        input_size = win_len * feature_size
        self.feature = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, emb_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        feat = self.feature(x)
        return self.classifier(feat)

# --- LSTM ---
class LSTMClassifier(nn.Module):
    def __init__(self, feature_size=98, hidden_size=128, num_layers=2, num_classes=2, emb_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        feat = self.fc(hidden_cat)
        return self.classifier(feat)

# --- ResNet-18 ---
class ResNet18Classifier(nn.Module):
    def __init__(self, win_len=250, feature_size=98, num_classes=2, emb_dim=128):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()
        self.proj = nn.Linear(512, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        feat = self.resnet(x)
        feat = self.proj(feat)
        return self.classifier(feat)

# --- Transformer ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
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

class TransformerClassifier(nn.Module):
    def __init__(self, feature_size=98, d_model=128, nhead=8, num_layers=4, num_classes=2, emb_dim=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
    def forward(self, x):
        x = x.squeeze(1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        feat = self.proj(x.mean(dim=1))
        return self.classifier(feat)

# --- ViT ---
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