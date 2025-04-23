import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ViT_Parallel, ViT_MultiTask, CSI2DCNN, CSITransformer
from util.checkpoints import load_checkpoint


# Constants or default scaling factors for patch sizes
PATCH_W_SCALE = 10
PATCH_H_SCALE = 2


def freeze_encoder(model):
    """
    Freezes all parameters in model.encoder, preventing backprop updates.
    """
    for param in model.encoder.parameters():
        param.requires_grad = False


# -----------------------------------------------------------------------------
#                          MODEL LOADING (Unsupervised)
# -----------------------------------------------------------------------------

def load_model_unsupervised(win_len, feature_size, depth=1, in_channels=2):
    """
    Loads a basic ViT_Parallel-based unsupervised model with
    patch-size scaling based on 'win_len' and 'feature_size'.

    Args:
        win_len (int)
        feature_size (int)
        depth (int): number of Transformer layers
        in_channels (int): input channels (often 2 for amplitude+phase)

    Returns:
        model (nn.Module): instance of ViT_Parallel
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
        win_len (int)
        feature_size (int)
        depth (int)
        in_channels (int)

    Returns:
        model (nn.Module): instance of ViT_MultiTask
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
        emb_size (int): embedding dim
        depth (int): number of layers
        freq_out (int): final frequency dimension
        in_channels (int): input channels

    Returns:
        model (nn.Module): instance of ViT_MultiTask
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


# -----------------------------------------------------------------------------
#                           Fine-tuning / Pretrained
# -----------------------------------------------------------------------------

def load_model_pretrained(checkpoint_path, win_len, feature_size, emb_size, depth, in_channels, num_classes):
    """
    Loads a ViT_Parallel and partially updates from a checkpoint
    where only 'encoder' weights are used strictly.

    Args:
        checkpoint_path (str)
        win_len, feature_size, emb_size, depth, in_channels, num_classes: model params

    Returns:
        model (nn.Module) updated with the encoder part of checkpoint
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
    """
    model = ViT_Parallel(win_len, feature_size, emb_size, depth, in_channels, num_classes)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    return model


def load_model_scratch(
    win_len,
    feature_size,
    depth=6,
    in_channels=2,
    num_classes=3,
    flag='joint'
):
    """
    Creates a new model from scratch (ViT_MultiTask if flag='joint', or ViT_Parallel otherwise).
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
        print("Using model: ViT_MultiTask (joint)")
    else:
        model = ViT_Parallel(
            win_len=win_len,
            feature_size=feature_size,
            emb_size=emb_size,
            depth=depth,
            in_channels=in_channels,
            num_classes=num_classes
        )
        print("Using model: ViT_Parallel")

    return model

# --------------------------------------
# ------------------------------------- yuqian benchmark model loading
def load_csi_model_benchmark(H, W, device):
    """
    Load CSI2DCNN model for benchmarking.
    :param H: Height of CSI input
    :param W: Width of CSI input
    :param device: torch.device (cuda or cpu)
    :return: Model moved to device
    """
    model = CSI2DCNN(input_size=(H, W))
    model = model.to(device)

    return model
