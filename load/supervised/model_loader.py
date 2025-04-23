import torch
import torch.nn as nn
from model import ViT_Parallel, ViT_MultiTask

# Constants or default scaling factors for patch sizes
PATCH_W_SCALE = 10
PATCH_H_SCALE = 2


def load_model_pretrained(checkpoint_path, win_len, feature_size, emb_size, depth, in_channels, num_classes):
    """
    Loads a ViT_Parallel and partially updates from a checkpoint
    where only 'encoder' weights are used strictly.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        win_len (int): Window length
        feature_size (int): Feature size
        emb_size (int): Embedding size
        depth (int): Number of Transformer layers
        in_channels (int): Input channels
        num_classes (int): Number of output classes

    Returns:
        model (nn.Module): Updated model with encoder weights from checkpoint
    """
    model = ViT_Parallel(win_len, feature_size, emb_size, depth, in_channels, num_classes)
    state_dict = torch.load(checkpoint_path)
    encoder_state = {k: v for k, v in state_dict.items() if 'encoder' in k}
    model.load_state_dict(encoder_state, strict=False)
    return model


def fine_tune_model(model, freeze_up_to_layer=2):
    """
    Freezes the first 'freeze_up_to_layer' blocks in model.encoder.
    For deeper Transformer with .layers,
    each block is enumerated and param.require_grad is set accordingly.
    
    Args:
        model (nn.Module): Model to freeze layers
        freeze_up_to_layer (int): Number of layers to freeze
        
    Returns:
        model (nn.Module): Model with frozen layers
    """
    # Example usage: freeze_up_to_layer=2 => freeze layer0, layer1, unfreeze rest
    for i, block in enumerate(model.encoder.layers):
        if i < freeze_up_to_layer:
            for param in block.parameters():
                param.requires_grad = False
    return model


def load_model_trained(checkpoint_path, win_len, feature_size, emb_size, depth, in_channels, num_classes):
    """
    Loads the entire model state from a checkpoint.
    Great for fully trained models (not partial).
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        win_len (int): Window length
        feature_size (int): Feature size
        emb_size (int): Embedding size
        depth (int): Number of Transformer layers
        in_channels (int): Input channels
        num_classes (int): Number of output classes
        
    Returns:
        model (nn.Module): Model with weights from checkpoint
    """
    model = ViT_Parallel(win_len, feature_size, emb_size, depth, in_channels, num_classes)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    return model


def load_model_scratch(win_len, feature_size, depth=6, in_channels=2, num_classes=3, flag='joint'):
    """
    Creates a new model from scratch (ViT_MultiTask if flag='joint', or ViT_Parallel otherwise).
    
    Args:
        win_len (int): Window length
        feature_size (int): Feature size
        depth (int): Number of Transformer layers
        in_channels (int): Input channels
        num_classes (int): Number of output classes
        flag (str): Whether to use ViT_MultiTask ('joint') or ViT_Parallel
        
    Returns:
        model (nn.Module): New model instance
    """
    emb_size = int((win_len / PATCH_W_SCALE) * (feature_size / PATCH_H_SCALE))
    print(f"[load_model_scratch] emb_size={emb_size}, flag={flag}")
    
    if flag == 'joint':
        model = ViT_MultiTask(
            emb_dim=128,
            encoder_heads=4,
            encoder_layers=depth,
            encoder_ff_dim=512,
            encoder_dropout=0.1,
            recon_heads=4,
            recon_layers=3,
            recon_ff_dim=512,
            recon_dropout=0.1,
            num_classes=num_classes,
            c_out=60,
            freq_out=6,
            max_len=512
        )
    else:
        model = ViT_Parallel(
            win_len=win_len,
            feature_size=feature_size,
            emb_size=emb_size,
            depth=depth,
            in_channels=in_channels,
            num_classes=num_classes
        )
    
    return model
