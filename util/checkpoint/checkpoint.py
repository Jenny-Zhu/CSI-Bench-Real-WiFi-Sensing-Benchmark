import torch
import os

def checkpoint(model, filename):
    """
    Save only the model weights (state_dict).
    
    Args:
        model: Model to save
        filename: Path to save the model weights
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)

def resume(model, filename, map_location=None):
    """
    Load model weights from file.
    
    Args:
        model: Model to load weights into
        filename: Path to the weights file
        map_location: Optional device mapping for loading on a different device
    """
    checkpoint = torch.load(filename, map_location=map_location or torch.device('cpu'))
    model.load_state_dict(checkpoint)

def warmup_schedule(epoch, warmup_epochs):
    """
    Warm-up learning rate scheduler.
    
    Args:
        epoch: Current epoch
        warmup_epochs: Number of warm-up epochs
        
    Returns:
        float: Scaling factor for learning rate
    """
    return min(1.0, epoch / warmup_epochs)

def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    """
    Save full training state including model, optimizer, epoch, etc.
    
    Args:
        state: Dictionary containing state to save
        filename: Path to save the checkpoint
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer=None, map_location=None):
    """
    Load full training checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        map_location: Optional device mapping
        
    Returns:
        tuple: (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(filepath, map_location=map_location or torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, epoch, loss
