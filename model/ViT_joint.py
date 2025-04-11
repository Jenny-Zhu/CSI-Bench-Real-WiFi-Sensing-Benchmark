import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce as Reduce
from .ViT import ResidualAdd
from einops.layers.torch import Rearrange, Reduce
import matplotlib.pyplot as plt
import math
patch_size_w_scale = 10
patch_size_h_scale = 2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeRelativePositionBias(nn.Module):
    """
    Learns a scalar bias for each possible relative distance in
    range [-max_len+1, max_len-1].

    Then, for a given actual seq_len <= max_len, returns a (seq_len, seq_len)
    float matrix of biases.
    """

    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len
        # Number of possible distances = 2*max_len - 1
        # Embedding dimension = 1 => each distance maps to a single float
        self.relative_bias = nn.Embedding(2 * max_len - 1, 1)

    def forward(self, seq_len, device=None):
        """
        Returns a (seq_len, seq_len) matrix of biases for the given seq_len.

        For positions i, j in [0..seq_len-1],
        relative distance = j - i,
        which is clamped to [-max_len+1, max_len-1].
        """
        if device is None:
            device = torch.device("cpu")

        # positions = 0..seq_len-1
        positions = torch.arange(seq_len, device=device)
        # distance_mat[i, j] = j - i, in range [-(seq_len-1) .. (seq_len-1)]
        distance_mat = positions.unsqueeze(0) - positions.unsqueeze(1)

        # clamp to [-max_len+1, max_len-1]
        distance_mat_clamped = distance_mat.clamp(-self.max_len + 1, self.max_len - 1)
        # shift into [0 .. 2*max_len-2]
        distances = distance_mat_clamped + (self.max_len - 1)

        # Lookup the learned bias => shape: (seq_len, seq_len, 1)
        bias = self.relative_bias(distances)
        # remove extra dimension => shape: (seq_len, seq_len)
        bias = bias.squeeze(-1)

        # If you truly want to scale by sqrt(d_model), do it here:
        # e.g., if you have a model dimension d_model:
        # d_model = 512
        # bias = bias / (d_model ** 0.5)

        return bias

class SmallFreqCNN(nn.Module):
    def __init__(self, in_channels=1, c_mid=8, c_out=16, freq_out=6):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, c_mid, kernel_size=(1, 7),
            stride=(1, 4), padding=(0, 3)
        )
        self.conv2 = nn.Conv2d(
            c_mid, c_out, kernel_size=(1, 7),
            stride=(1, 4), padding=(0, 3)
        )
        self.freq_out = freq_out

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # shape => (B, c_out, T_out, F2)
        T_curr = x.shape[2]
        x = F.adaptive_avg_pool2d(x, (T_curr, self.freq_out))
        return x

class FreqDecoder(nn.Module):
    def __init__(self, c_in=16, c_mid=8, out_channels=1):
        super().__init__()
        self.tconv1 = nn.ConvTranspose2d(
            in_channels=c_in, out_channels=c_mid,
            kernel_size=(1, 7), stride=(1, 4), padding=(0, 3), output_padding=(0, 3)
        )
        self.tconv2 = nn.ConvTranspose2d(
            in_channels=c_mid, out_channels=out_channels,
            kernel_size=(1, 11), stride=(1, 4), padding=(0, 4), output_padding=(0, 3)
        )

    def forward(self, x):
        x = self.tconv1(x)
        x = F.relu(x)
        x = self.tconv2(x)
        x = F.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, c_out=16, c_mid=8, in_channels=1):
        super().__init__()
        # Transposed convolutions (same as before)
        self.tconv2 = nn.ConvTranspose2d(
            c_out, c_mid, kernel_size=(1, 7),
            stride=(1, 4), padding=(0, 3),
            output_padding=(0, 3)
        )
        self.tconv1 = nn.ConvTranspose2d(
            c_mid, in_channels, kernel_size=(1, 7),
            stride=(1, 4), padding=(0, 3),
            output_padding=(0, 3)
        )

    def forward(self, x, target_freq):
        # Input shape: (B, c_out, T, 6)
        x = self.tconv2(x)
        x = F.relu(x)
        x = self.tconv1(x)
        # Output shape: (B, in_channels, T, ~96)
        # Interpolate to the original input's frequency dimension (F_in)
        x = F.interpolate(
            x,
            size=(x.size(2), target_freq),  # (T, target_freq)
            mode="bilinear",
            align_corners=False
        )
        return x  # (B, in_channels, T, F_in)

###################################################### Cross-attention
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, seq_len_or_x, device=None):
        """
        Args:
            seq_len_or_x: Either an integer (sequence length) or a tensor (to infer device).
            device: Optional device (required if `seq_len_or_x` is an integer).
        """
        if isinstance(seq_len_or_x, torch.Tensor):
            # Infer sequence length and device from input tensor
            seq_len = seq_len_or_x.size(-1)
            device = seq_len_or_x.device
        else:
            # Use provided sequence length and device
            seq_len = seq_len_or_x
            if device is None:
                raise ValueError("Device must be specified for integer sequence lengths.")

        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(seq_len, device=device).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float() *
            (-math.log(10000.0) / self.d_model)
             )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _get_device(self):
        return next(self.parameters()).device

class AdaptiveDownsampleEncoder(nn.Module):
    def __init__(self, in_channels=1, c_mid=8, c_out=16, freq_out=6):
        super().__init__()
        self.freq_out = freq_out

        # Project input channels to mid-dimension
        self.proj = nn.Linear(in_channels, c_mid)

        # Learnable queries (one per target frequency bin)
        self.queries = nn.Parameter(torch.randn(freq_out, c_mid))

        # Final projection to output channels
        self.fc_out = nn.Linear(c_mid, c_out)

        self.pos_encoder = PositionalEncoding(d_model=c_mid)
    def forward(self, x):
        # Input shape: (B, in_channels, T, F_in)
        B, C, T, F_in = x.shape

        # Treat frequency as a sequence: (B, T, F_in, C)
        x = x.permute(0, 2, 3, 1)

        # Project each frequency bin: (B, T, F_in, c_mid)
        x_proj = self.proj(x)  # Shape: [B, T, F_in, c_mid]
        pe = self.pos_encoder(x,x.device)  # (F_in, c_mid)
        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, F_in, c_mid)
        x_proj += pe
        # Compute attention scores between queries and frequency bins
        attn_scores = torch.einsum('btfc,qc->btqf', x_proj, self.queries)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, freq_out, F_in]

        # Aggregate features using attention: (B, T, freq_out, c_mid)
        x_agg = torch.einsum('btqf,btfc->btqc', attn_weights, x_proj)

        # Project to output channels: (B, T, freq_out, c_out)
        x_out = self.fc_out(x_agg)

        # Reshape to (B, c_out, T, freq_out)
        x_out = x_out.permute(0, 3, 1, 2)

        return x_out  # Fixed shape: (B, c_out, T, freq_out)


class AdaptiveUpsampleDecoder(nn.Module):
    def __init__(self, in_channels=1, c_mid=8, c_out=16):
        super().__init__()
        self.in_channels = in_channels

        # Project encoder output to mid-dimension (keys/values)
        self.proj_enc = nn.Linear(c_out, c_mid)

        # Learnable parameters for query generation
        self.query_proj = nn.Linear(c_mid, c_mid)  # Optional

        # Final projection to input channels
        self.fc_out = nn.Linear(c_mid, in_channels)

        # Positional encoding module (same as encoder)
        self.pos_encoder = PositionalEncoding(d_model=c_mid)

    def forward(self, x_enc, target_freq: int):
        """
        Args:
            x_enc: Encoder output (B, c_out, T, freq_out=6)
            target_freq: Original input frequency dimension (F_in)
        Returns:
            x_recon: (B, in_channels, T, target_freq)
        """
        B, c_out, T, freq_out = x_enc.shape

        # 1. Prepare encoder output (keys/values)
        x_enc = x_enc.permute(0, 2, 3, 1)  # (B, T, 6, c_out)
        keys_values = self.proj_enc(x_enc)  # (B, T, 6, c_mid)

        # 2. Generate positional queries for target frequencies (F_in)
        pos_queries = self.pos_encoder(
            target_freq,  # Integer sequence length
            device=x_enc.device  # Device from encoder output
        )  # (F_in, c_mid)        pos_queries = pos_queries.unsqueeze(0).unsqueeze(0)  # (1, 1, F_in, c_mid)
        pos_queries = pos_queries.expand(B, T, -1, -1)  # (B, T, F_in, c_mid)

        # Optional: Add learned transformations to queries
        # pos_queries = self.query_proj(pos_queries)

        # 3. Cross-attention: Reconstruct F_in from encoder output
        attn_scores = torch.einsum('btfc,btqc->btqf', keys_values, pos_queries)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, F_in, 6)

        # Aggregate features using attention
        x_agg = torch.einsum('btqf,btfc->btqc', attn_weights, keys_values)  # (B, T, F_in, c_mid)

        # 4. Project to input channels
        x_recon = self.fc_out(x_agg)  # (B, T, F_in, in_channels)
        x_recon = x_recon.permute(0, 3, 1, 2)  # (B, in_channels, T, F_in)

        return x_recon

class RelativeTransformerEncoderLayer(nn.Module):
    """
    A re-implementation of TransformerEncoderLayer that adds
    relative position bias to the self-attention, handling both
    float masks (with -inf for blocked) and boolean masks (True => blocked).
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0,
        max_len=512,
        batch_first=True
    ):
        super().__init__()
        self.batch_first = batch_first

        # (A) Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # (B) Two-layer MLP
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # (C) Relative position bias module -> returns (T, T)
        self.rel_pos_bias = TimeRelativePositionBias(max_len=max_len)

    def forward(
        self,
        src,
        src_mask=None,         # shape could be (T, T) or (B, T, T)
        is_causal=False,       # you won’t use it, but we keep the arg
        src_key_padding_mask=None  # shape (B, T) for padded tokens
    ):
        """
        Args:
            src: (B, T, d_model) if batch_first=True
            src_mask: optional mask.
                - If float, should contain 0.0 for allowed and -inf for blocked positions.
                - If bool, True means "block".
                - Shape can be (T, T) for a global mask, or (B, T, T) for per-batch.
            src_key_padding_mask: (B, T) boolean mask for padding. True => ignore.
        """
        # Identify B and T
        if self.batch_first:
            B, T, _ = src.shape
        else:
            T, B, _ = src.shape

        # 1) Compute the (T, T) relative bias
        base_bias = self.rel_pos_bias(T, device=src.device)  # shape (T, T)

        rel_bias = self.rel_pos_bias(T,device=src.device)  # (250, 250) float
        attn_output, _ = self.self_attn(
            query=src.clone(), key=src, value=src,
            attn_mask=rel_bias,               # float or 2D/3D bool for attention
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        src2 = attn_output

        # 3) First residual & norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 4) Feedforward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        # 5) Second residual & norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


##############################################################################
# 2) A generic Transformer (can be used for either encoder or reconstruction)
##############################################################################
def make_relative_transformer_encoder(
    emb_dim, num_heads, ff_dim, dropout, num_layers, max_len=250
):
    """
    Constructs a stack of RelativeTransformerEncoderLayer.
    """
    # We create a single layer definition and let nn.TransformerEncoder
    # clone it 'num_layers' times. (This is how PyTorch's API works.)
    custom_layer = RelativeTransformerEncoderLayer(
        d_model=emb_dim,
        nhead=num_heads,
        dim_feedforward=ff_dim,
        dropout=dropout,
        max_len=max_len,       # <= so the relative bias can handle up to T=250
        batch_first=True
    )
    return nn.TransformerEncoder(custom_layer, num_layers=num_layers)


##############################################################################
# 3) Multi-task model
##############################################################################
class ViT_MultiTask(nn.Module):
    def __init__(
        self,
        emb_dim=128,
        encoder_heads=4,
        encoder_layers=6,
        encoder_ff_dim=512,
        encoder_dropout=0.1,
        recon_heads=4,
        recon_layers=3,
        recon_ff_dim=512,
        recon_dropout=0.1,
        num_classes=3,
        c_out=16,
        freq_out=10,  # final freq dimension from the CNN
        max_len=512
    ):
        super().__init__()
        # 1) Encoder CNN
        self.freq_cnn = SmallFreqCNN(
            in_channels=1,
            c_mid=8,
            c_out=c_out,
            freq_out=freq_out
        )
        self.freq_encoder = AdaptiveDownsampleEncoder(in_channels=1, c_out=c_out, freq_out=freq_out)

        # 2) Flatten dim
        self.flat_dim = c_out * freq_out
        self.emb_dim = emb_dim

        # 3) Input embedding
        self.input_embed = nn.Linear(self.flat_dim, emb_dim)

        # 4) Transformer(s)
        self.encoder = make_relative_transformer_encoder(
            emb_dim=emb_dim,
            num_heads=encoder_heads,
            ff_dim=encoder_ff_dim,
            dropout=encoder_dropout,
            num_layers=encoder_layers,
            max_len=max_len
        )
        self.reconstruction_transformer = make_relative_transformer_encoder(
            emb_dim=emb_dim,
            num_heads=recon_heads,
            ff_dim=recon_ff_dim,
            dropout=recon_dropout,
            num_layers=recon_layers,
            max_len=max_len
        )

        # 5) Heads: contrastive, classification, etc.
        self.contrastive_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.norm = nn.LayerNorm(emb_dim)

        # 6) For Reconstruction to original shape:
        #    (B, T_out, emb_dim) -> (B, T_out, c_out*freq_out) -> (B, c_out, T_out, freq_out)
        #    -> decode to (B, 1, T_out, F_in)
        self.decoder_input = nn.Linear(emb_dim, self.flat_dim)  # c_out*freq_out
        self.decoder_cnn = FreqDecoder(c_in=c_out, c_mid=8, out_channels=1)
        # self.decoder = Decoder(c_out=c_out)  # F_in is fixed for your task

        self.freq_decoder = AdaptiveUpsampleDecoder(in_channels=1)
    def forward(self, x1, x2=None, mask=None, flag='joint'):
        """
        Now we can do a full reconstruction back to (B,1,T_in,F_in)
        in the 'reconstruction' or 'joint' branches, via decoder_cnn.
        """
        if mask is not None:
            x1 = x1 * (1 - mask.unsqueeze(1))
        F_in = x1.shape[-1]  # Original frequency dimension
        # --- CNN encoder ---
        x1_cnn = self.freq_encoder(x1)  # => (B, c_out, T_out, freq_out=10)
        B, c_out, T_out, F_out = x1_cnn.shape  # F_out = 6

        # Flatten => (B, T_out, c_out*6)
        x1_cnn = x1_cnn.permute(0, 2, 1, 3).contiguous()
        x1_cnn = x1_cnn.view(B, T_out, c_out * F_out)

        # Embed => (B, T_out, emb_dim)
        x1_emb = self.input_embed(x1_cnn)

        # Transformer encode => (B, T_out, emb_dim)
        z1_enc = self.encoder(x1_emb)

        # Check flags
        if flag == 'contrastive':
            z1_pooled = z1_enc.mean(dim=1)
            z1_proj = self.contrastive_head(z1_pooled)
            if x2 is not None:
                if mask is not None:
                    x2 = x2 * (1 - mask.unsqueeze(1))
                x2_cnn = self.freq_encoder(x2)
                B2, c_out2, T_out2, F_out2 = x2_cnn.shape
                x2_cnn = x2_cnn.permute(0, 2, 1, 3).contiguous()
                x2_cnn = x2_cnn.view(B2, T_out2, c_out2 * F_out2)
                x2_emb = self.input_embed(x2_cnn)
                z2_enc = self.encoder(x2_emb)
                z2_pooled = z2_enc.mean(dim=1)
                z2_proj = self.contrastive_head(z2_pooled)
                return z1_proj, z2_proj
            else:
                return z1_proj

        elif flag == 'reconstruction':
            # 1) Reconstruction transformer
            z1_recon = self.reconstruction_transformer(z1_enc)  # (B, T_out, emb_dim)

            # 2) Project back to c_out*F_out (c_out*6)
            dec_in = self.decoder_input(z1_recon)  # (B, T_out, c_out*6)

            # 3) Reshape to (B, c_out, T_out, 6) for decoder input
            dec_in = dec_in.view(B, T_out, c_out, F_out)
            dec_in = dec_in.permute(0, 2, 1, 3).contiguous()  # (B, c_out, T_out, 6)

            # 4) Pass through the new decoder
            reconstructed = self.freq_decoder(dec_in, F_in)  # Output: (B, 1, T_out, F_in)

            return reconstructed  # No need for interpolation/unsqueeze

        elif flag == 'joint':
            # 1) Do contrastive
            z1_pooled = z1_enc.mean(dim=1)
            z1_proj = self.contrastive_head(z1_pooled)

            # 2) Reconstruction
            z1_recon = self.reconstruction_transformer(z1_enc)
            dec_in = self.decoder_input(z1_recon)
            dec_in = dec_in.view(B, T_out, c_out, F_out).permute(0, 2, 1, 3)
            reconstructed = self.freq_decoder(dec_in, F_in)  # (B, 1, T_out, F_in)

            if x2 is not None:
                if mask is not None:
                    x2 = x2 * (1 - mask.unsqueeze(1))
                x2_cnn = self.freq_cnn(x2)
                B2, c_out2, T_out2, F_out2 = x2_cnn.shape
                x2_cnn = x2_cnn.permute(0, 2, 1, 3).contiguous()
                x2_cnn = x2_cnn.view(B2, T_out2, c_out2*F_out2)
                x2_emb = self.input_embed(x2_cnn)
                z2_enc = self.encoder(x2_emb)
                z2_pooled = z2_enc.mean(dim=1)
                z2_proj = self.contrastive_head(z2_pooled)
            else:
                z2_proj = None

            return z1_proj, z2_proj, reconstructed

        elif flag == 'supervised':
            # Classification
            z1_enc = self.norm(z1_enc)
            z1_pooled = z1_enc.mean(dim=1)
            logits = self.classifier(z1_pooled)
            return logits, z1_pooled


