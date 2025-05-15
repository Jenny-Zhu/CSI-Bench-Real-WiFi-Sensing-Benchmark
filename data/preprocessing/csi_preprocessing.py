import numpy as np
import torch

def normalize_csi(csi):
    """Normalize CSI data.
    
    Args:
        csi: The CSI data to normalize.
        
    Returns:
        The normalized CSI data.
    """
    chnnorm = np.sqrt(np.sum(np.abs(csi) ** 2, axis=2, keepdims=True))
    csi_normalized = csi / (chnnorm + np.finfo(float).eps)
    
    # Replace NaNs and Infs with 0
    csi_normalized[np.isnan(csi_normalized)] = 0
    csi_normalized[np.isinf(csi_normalized)] = 0
    
    return csi_normalized

def rescale_csi(csi, scale=1.0):
    """Rescale CSI data.
    
    Args:
        csi: The CSI data to rescale.
        scale: The scale factor.
        
    Returns:
        The rescaled CSI data.
    """
    return csi * scale

def transform_csi_to_real(csi):
    """Transform complex CSI data to real-valued representation.
    
    Args:
        csi: The complex CSI data.
        
    Returns:
        The real-valued representation of the CSI data.
    """
    if isinstance(csi, np.ndarray):
        real_part = np.real(csi)
        imag_part = np.imag(csi)
        return np.stack([real_part, imag_part], axis=1)
    elif isinstance(csi, torch.Tensor):
        real_part = torch.real(csi)
        imag_part = torch.imag(csi)
        return torch.stack([real_part, imag_part], dim=1)
    else:
        raise TypeError("csi should be a numpy array or a torch tensor.")
