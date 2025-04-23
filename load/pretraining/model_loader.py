import torch
import torch.nn as nn
from model import ViT_Parallel, ViT_MultiTask

# Constants or default scaling factors for patch sizes
PATCH_W_SCALE = 10
PATCH_H_SCALE = 2


def load_model_unsupervised(win_len, feature_size, depth=1, in_channels=2):
    """
    Loads a basic ViT_Parallel-based unsupervised model with
    patch-size scaling based on 'win_len' and 'feature_size'.

    Args:
        win_len (int): Window length
        feature_size (int): Feature size
        depth (int): Number of Transformer layers
        in_channels (int): Input channels (often 2 for amplitude+phase)

    Returns:
        model (nn.Module): Instance of ViT_Parallel
    """
    emb_size = int((win_len / PATCH_W_SCALE) * (feature_size / PATCH_H_SCALE))
    print(f"[load_model_unsupervised] Using ViT_Parallel with emb_size={emb_size}")
    model = ViT_Parallel(
        win_len=win_len,
        feature_size=feature_size,
        emb_size=emb_size,
        depth=depth,
        in_channels=in_channels
    )
    return model


def load_model_unsupervised_joint(win_len, feature_size, depth=1, in_channels=2):
    """
    Loads a basic ViT_MultiTask-based unsupervised model
    for 'joint' tasks.

    Args:
        win_len (int): Window length
        feature_size (int): Feature size
        depth (int): Number of Transformer layers
        in_channels (int): Input channels (often 2 for amplitude+phase)

    Returns:
        model (nn.Module): Instance of ViT_MultiTask
    """
    emb_size = int((win_len / PATCH_W_SCALE) * (feature_size / PATCH_H_SCALE))
    print(f"[load_model_unsupervised_joint] Using ViT_MultiTask with emb_size={emb_size}")
    model = ViT_MultiTask(
        emb_dim=128,
        encoder_heads=4,
        encoder_layers=6,
        encoder_ff_dim=512,
        encoder_dropout=0.1,
        recon_heads=4,
        recon_layers=3,
        recon_ff_dim=512,
        recon_dropout=0.1,
        num_classes=3,   # Default or can be parameterized
        c_out=60,
        freq_out=6,
        max_len=512
    )
    return model


def load_model_unsupervised_joint_csi_var(emb_size=128, depth=6, freq_out=10, in_channels=1):
    """
    Loads a ViT_MultiTask with variable input CSI shape.

    Args:
        emb_size (int): Embedding dimension
        depth (int): Number of Transformer layers
        freq_out (int): Final frequency dimension
        in_channels (int): Input channels

    Returns:
        model (nn.Module): Instance of ViT_MultiTask
    """
    print("[load_model_unsupervised_joint_csi_var] Using ViT_MultiTask with variable CSI shapes")
    model = ViT_MultiTask(
        emb_dim=emb_size,
        encoder_heads=4,
        encoder_layers=depth,
        encoder_ff_dim=512,
        encoder_dropout=0.1,
        recon_heads=4,
        recon_layers=3,
        recon_ff_dim=512,
        recon_dropout=0.1,
        num_classes=3,   # or param
        c_out=16,
        freq_out=freq_out,
        max_len=512
    )
    return model


def load_model_unsupervised_joint_fix_length(win_len, feature_size):
    """
    Similar to load_model_unsupervised_joint but uses a fixed time/freq size (250/98).
    
    Args:
        win_len (int): Window length
        feature_size (int): Feature size
        
    Returns:
        model (nn.Module): Instance of ViT_MultiTask
    """
    emb_size = int((win_len / PATCH_W_SCALE) * (feature_size / PATCH_H_SCALE))
    print(f"[load_model_unsupervised_joint_fix_length] Using ViT_MultiTask with emb_size={emb_size}")
    model = ViT_MultiTask(
        time_len=250,
        freq_size=98,
        emb_dim=128,
        encoder_heads=4,
        encoder_layers=6,
        encoder_ff_dim=512,
        encoder_dropout=0.1,
        recon_heads=4,
        recon_layers=3,
        recon_ff_dim=512,
        recon_dropout=0.1,
        num_classes=3
    )
    return model
