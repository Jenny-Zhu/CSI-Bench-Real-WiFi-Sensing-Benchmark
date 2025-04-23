import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce as Reduce
from einops.layers.torch import Rearrange, Reduce
patch_size_w_scale = 50
patch_size_h_scale = 2

class PatchEmbedding(nn.Module):
    def __init__(self, win_len, feature_size, emb_size, in_channels = 1):
        self.win_len = win_len
        self.feature_size = feature_size
        patch_size_w = int(win_len/patch_size_w_scale)
        patch_size_h = int(feature_size/patch_size_h_scale)
        img_size = win_len*feature_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size = (patch_size_w, patch_size_h), stride = (patch_size_w, patch_size_h)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        num_patches = (win_len // patch_size_w) * (feature_size // patch_size_h)
        print(num_patches)
        self.position = nn.Parameter(torch.randn(num_patches + 1, emb_size))
    def forward(self, x):
        x = x.view(-1,1,self.win_len,self.feature_size)
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
    def __init__(self, emb_size, num_heads = 5, dropout = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size*3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
    def forward(self, x, mask = None):
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion = 4, drop_p = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 drop_p = 0.5,
                 forward_expansion = 4,
                 forward_drop_p = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 depth = 1,
                 **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, num_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes))

class ViT(nn.Sequential):
    def __init__(self,
                win_len,
                feature_size,
                emb_size,
                depth = 1,
                in_channels = 1,
                *,
                num_classes=2,
                **kwargs):
        super().__init__(
            PatchEmbedding(win_len,
                           feature_size,
                           emb_size,
                           in_channels),
            TransformerEncoder(depth,
                               emb_size=emb_size,
                               **kwargs),
            ClassificationHead(emb_size,
                               num_classes)
        )
