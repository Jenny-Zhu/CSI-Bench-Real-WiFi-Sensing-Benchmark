import torch
import numpy as np

class BaseAugmentation:
    """Base class for all data augmentation techniques."""
    
    def __init__(self, device=None):
        """Initialize the base augmentation class.
        
        Args:
            device: The device to use for tensor operations.
                   If None, will use CPU.
        """
        self.device = device if device is not None else torch.device('cpu')
    
    def apply_augmentations(self, data):
        """Apply a sequence of augmentations to the data.
        
        This method should be implemented by subclasses.
        
        Args:
            data: The data to augment.
            
        Returns:
            The augmented data.
        """
        raise NotImplementedError("Subclasses must implement apply_augmentations")
