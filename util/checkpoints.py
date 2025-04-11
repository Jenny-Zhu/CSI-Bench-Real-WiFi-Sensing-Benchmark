import torch
import os

def checkpoint(model, filename):
    """Save only the model weights (state_dict)."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)

def resume(model, filename, map_location=None):
    """Load model weights from file."""
    checkpoint = torch.load(filename, map_location=map_location or torch.device('cpu'))
    model.load_state_dict(checkpoint)

def warmup_schedule(epoch, warmup_epochs):
    """Warm-up learning rate scheduler."""
    return min(1.0, epoch / warmup_epochs)

def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    """Save full training state including model, optimizer, epoch, etc."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filepath, model, optimizer=None, map_location=None):
    """Load full training checkpoint."""
    checkpoint = torch.load(filepath, map_location=map_location or torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, epoch, loss
