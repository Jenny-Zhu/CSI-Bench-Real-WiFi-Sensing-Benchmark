import torch
import torch.nn as nn
from model import ViT_Parallel, ViT_MultiTask

# Constants or default scaling factors for patch sizes
PATCH_W_SCALE = 10
PATCH_H_SCALE = 2


def load_model_pretrained(checkpoint_path, num_classes, win_len=250, feature_size=98, emb_size=None, depth=6, in_channels=2):
    """
    Loads a ViT_Parallel and partially updates from a checkpoint
    where only 'encoder' weights are used strictly.

    Args:
        checkpoint_path (str): Path to the checkpoint file
        num_classes (int): Number of output classes
        win_len (int, optional): Window length. Defaults to 250.
        feature_size (int, optional): Feature size. Defaults to 98.
        emb_size (int, optional): Embedding size. If None, will be calculated based on win_len and feature_size.
        depth (int, optional): Number of Transformer layers. Defaults to 6.
        in_channels (int, optional): Input channels. Defaults to 2.

    Returns:
        model (nn.Module): Updated model with encoder weights from checkpoint
    """
    # 计算embedding size如果没有提供
    if emb_size is None:
        emb_size = int((win_len / PATCH_W_SCALE) * (feature_size / PATCH_H_SCALE))
    
    # 创建新版本的ViT_Parallel
    model = ViT_Parallel(
        win_len=win_len,
        feature_size=feature_size,
        emb_dim=emb_size,
        in_channels=in_channels,
        proj_dim=emb_size,
        num_classes=num_classes
    )
    
    # 加载预训练权重
    state_dict = torch.load(checkpoint_path)
    
    # 检查状态字典中的键，以确定是否为旧版本或新版本的模型
    if any('encoder' in k for k in state_dict.keys()):
        # 旧版本的模型
        encoder_state = {k: v for k, v in state_dict.items() if 'encoder' in k}
        model.load_state_dict(encoder_state, strict=False)
    else:
        # 新版本的模型
        model.load_state_dict(state_dict, strict=False)
    
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
    # 检查是否为新版本的模型结构
    if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
        # 新版本的ViT_Parallel
        for i, block in enumerate(model.model.backbone.layers):
            if i < freeze_up_to_layer:
                for param in block.parameters():
                    param.requires_grad = False
    else:
        # 旧版本的模型
        for i, block in enumerate(model.encoder.layers):
            if i < freeze_up_to_layer:
                for param in block.parameters():
                    param.requires_grad = False
    return model


def load_model_trained(checkpoint_path, num_classes, win_len=250, feature_size=98, emb_size=None, depth=6, in_channels=2):
    """
    Loads the entire model state from a checkpoint.
    Great for fully trained models (not partial).
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        num_classes (int): Number of output classes
        win_len (int, optional): Window length. Defaults to 250.
        feature_size (int, optional): Feature size. Defaults to 98.
        emb_size (int, optional): Embedding size. If None, will be calculated based on win_len and feature_size.
        depth (int, optional): Number of Transformer layers. Defaults to 6.
        in_channels (int, optional): Input channels. Defaults to 2.
        
    Returns:
        model (nn.Module): Model with weights from checkpoint
    """
    # 计算embedding size如果没有提供
    if emb_size is None:
        emb_size = int((win_len / PATCH_W_SCALE) * (feature_size / PATCH_H_SCALE))
    
    # 创建新版本的ViT_Parallel
    model = ViT_Parallel(
        win_len=win_len,
        feature_size=feature_size,
        emb_dim=emb_size,
        in_channels=in_channels,
        proj_dim=emb_size,
        num_classes=num_classes
    )
    
    # 加载预训练权重
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    
    return model


def load_model_scratch(num_classes=3, win_len=250, feature_size=98, depth=6, in_channels=2, flag='supervised'):
    """
    Creates a new model from scratch (ViT_MultiTask if flag='joint', or ViT_Parallel otherwise).
    
    Args:
        num_classes (int): Number of output classes
        win_len (int): Window length (defaults to 250 for CSI data)
        feature_size (int): Feature size (defaults to 98 for CSI data)
        depth (int): Number of Transformer layers
        in_channels (int): Input channels
        flag (str): Whether to use ViT_MultiTask ('joint') or ViT_Parallel
        
    Returns:
        model (nn.Module): New model instance
    """
    emb_size = int((win_len / PATCH_W_SCALE) * (feature_size / PATCH_H_SCALE))
    print(f"[load_model_scratch] emb_size={emb_size}, flag={flag}, win_len={win_len}, feature_size={feature_size}")
    
    if flag == 'joint':
        model = ViT_MultiTask(
            win_len=win_len,
            feature_size=feature_size,
            emb_dim=emb_size,
            in_channels=in_channels,
            proj_dim=emb_size,
            pretext_tasks=['contrastive', 'reconstruction'],
            num_classes=num_classes
        )
    else:
        model = ViT_Parallel(
            win_len=win_len,
            feature_size=feature_size,
            emb_dim=emb_size,
            in_channels=in_channels,
            proj_dim=emb_size,
            num_classes=num_classes
        )
    
    return model
