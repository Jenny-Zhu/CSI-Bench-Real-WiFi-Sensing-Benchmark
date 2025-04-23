import torch
import numpy as np

def warmup_schedule(epoch, warmup_epochs):
    """
    Warmup learning rate scheduler function
    
    Args:
        epoch: Current training epoch
        warmup_epochs: Number of epochs for warmup
        
    Returns:
        Learning rate factor
    """
    if epoch < warmup_epochs:
        return max(0.1, epoch / warmup_epochs)
    else:
        # Linear decay
        decay_rate = 0.95
        return max(0.1, decay_rate ** (epoch - warmup_epochs))

def freeze_cnn_layers(model, unfreeze_classifier=True):
    """
    Freeze CNN layers of a model, only train classifier part
    
    Args:
        model: Model to freeze layers
        unfreeze_classifier: Whether to unfreeze classifier layers
    
    Returns:
        Model with frozen layers
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier layers if needed
    if unfreeze_classifier:
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
    
    return model

def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy
    
    Args:
        outputs: Model outputs [batch_size, num_classes]
        labels: True labels [batch_size]
        
    Returns:
        Accuracy
    """
    if outputs.shape[1] > 1:  # Multi-class
        _, predicted = torch.max(outputs.data, 1)
    else:  # Binary
        predicted = (outputs.data > 0.5).float()
    
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    
    return correct / total

def setup_mixup(alpha=0.2):
    """
    Setup Mixup data augmentation
    
    Args:
        alpha: Beta distribution alpha parameter
        
    Returns:
        mixup_fn: mixup function
    """
    def mixup_data(x, y, device):
        """Perform mixup operation"""
        batch_size = x.size(0)
        lam = np.random.beta(alpha, alpha, batch_size)
        # Ensure correct shape
        lam = torch.from_numpy(lam).float().to(device)
        lam = lam.view(-1, 1, 1, 1)  # For 2D data

        # Get random indices
        index = torch.randperm(batch_size).to(device)
        
        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[index, :]
        lam = lam.view(-1)  # For loss calculation
        
        return mixed_x, y, y[index], lam
    
    return mixup_data
