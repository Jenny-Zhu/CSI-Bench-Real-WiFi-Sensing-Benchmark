import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce as Reduce
from .ViT import ResidualAdd
from einops.layers.torch import Rearrange, Reduce
patch_size_w_scale = 50
patch_size_h_scale = 8

class PatchEmbedding(nn.Module):
    def __init__(self, win_len, feature_size, emb_size, in_channels=1):
        self.win_len = win_len
        self.feature_size = feature_size
        patch_size_w = int(win_len / patch_size_w_scale)
        patch_size_h = int(feature_size / patch_size_h_scale)
        img_size = win_len * feature_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=(patch_size_w, patch_size_h),
                      stride=(patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        size_ratio = int(img_size / emb_size)
        # print(size_ratio)
        self.position = nn.Parameter(torch.randn(size_ratio + 1, emb_size))

    def forward(self, x):
        x = x.view(-1, 2, self.win_len, self.feature_size)
        b, _, _, _ = x.shape
        # print(x.shape)
        x = self.projection(x)
        # print(x.shape)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        # print(x.shape)
        x += self.position
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads=5, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        # print(x.shape)
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.emb_size // self.num_heads).permute(2,
                                                                                                                      0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        att = F.softmax(energy, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhqk, bhkd -> bhqd', att, values).reshape(x.shape[0], x.shape[1], -1)
        return self.projection(out)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, dropout=0.1):
        super(FeedForwardBlock, self).__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads=5, dropout=0.1, forward_expansion=4):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(emb_size, num_heads, dropout),
            nn.Dropout(dropout)
        ))
        self.feed_forward = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(emb_size, forward_expansion, dropout),
            nn.Dropout(dropout)
        ))

    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=1, **kwargs):
        super(TransformerEncoder, self).__init__(*[
            TransformerEncoderBlock(**kwargs) for _ in range(depth)
        ])


class ViTEncoder(nn.Module):
    def __init__(self, win_len, feature_size, emb_size, depth=1, in_channels=1):
        super(ViTEncoder, self).__init__()
        self.patch_embedding = PatchEmbedding(win_len, feature_size, emb_size, in_channels)
        self.encoder = TransformerEncoder(depth=depth, emb_size=emb_size)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        return x.mean(dim=1)  # Pooling the sequence to one vector

class MLPHead(nn.Module):
    def __init__(self, input_dim,num_classes, hidden_dim=512 ):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, emb_size, num_classes):
        super(AttentionHead, self).__init__()
        self.query = nn.Parameter(torch.randn(num_classes, emb_size))
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        # Batch size (b), Number of patches + 1 (n), Embedding size (e)
        b, n, e = x.shape

        # Generate keys and values from input
        keys = self.key(x)  # Shape: (b, n, e)
        values = self.value(x)  # Shape: (b, n, num_classes)

        # Broadcast query to batch size
        query = self.query.unsqueeze(0).expand(b, -1, -1)  # Shape: (b, num_classes, e)

        # Attention mechanism
        attention_scores = torch.einsum('bce,bne->bcn', query, keys)  # Shape: (b, num_classes, n)
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (b, num_classes, n)

        # Weighted sum of values based on attention weights
        attention_output = torch.einsum('bcn,bnv->bcv', attention_weights, values)  # Shape: (b, num_classes, num_classes)

        # For classification, you typically return the diagonal or use another method to reduce to class scores
        output = attention_output.mean(dim=1)  # Reduce to (b, num_classes)
        return output


class ViT_Parallel(nn.Module):
    def __init__(self, win_len, feature_size, emb_size,classification_head_type, depth=1, in_channels=1, num_classes=4 ):
        super(ViT_Parallel, self).__init__()
        self.encoder = ViTEncoder(win_len, feature_size, emb_size, depth, in_channels)
        if classification_head_type =="single_layer":
            self.classifier = nn.Linear(emb_size,  num_classes)
        elif classification_head_type =="MLP":
            self.classifier = nn.MLPHead(emb_size,  num_classes)
        elif classification_head_type =="Attention":
            self.classifier = nn.AttentionHead(emb_size,  num_classes)
        else:
            raise ValueError("Classifcation Head Type not Found!!!")



    def forward(self, x1, x2=None, flag='unsupervised'):
        if flag == 'supervised':
            x1 = self.encoder(x1)
            # x1 = x1.mean(dim=1)  # Aggregate over the sequence length
            return self.classifier(x1)
        else:
            z1 = self.encoder(x1)
            if x2 is not None:
                z2 = self.encoder(x2)
                return z1, z2
            return z1
