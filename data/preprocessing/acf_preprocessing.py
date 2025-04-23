import numpy as np
import torch

def normalize_acf(acf):
    """Normalize ACF data.
    
    Args:
        acf: The ACF data to normalize.
        
    Returns:
        The normalized ACF data.
    """
    if isinstance(acf, np.ndarray):
        mean = np.mean(acf, axis=-1, keepdims=True)
        std = np.std(acf, axis=-1, keepdims=True)
        return (acf - mean) / (std + 1e-8)
    elif isinstance(acf, torch.Tensor):
        mean = torch.mean(acf, dim=-1, keepdim=True)
        std = torch.std(acf, dim=-1, keepdim=True)
        return (acf - mean) / (std + 1e-8)
    else:
        raise TypeError("acf should be a numpy array or a torch tensor.")
