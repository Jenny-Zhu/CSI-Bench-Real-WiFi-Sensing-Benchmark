import torch
import numpy as np
import random
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomApply, ColorJitter, RandomGrayscale, ToTensor, Normalize
from torchvision.transforms.functional import to_pil_image, to_tensor
from data.augmentation.base_augmentation import BaseAugmentation

class TensorToPILTransform:
    """Convert a tensor to a PIL Image."""
    def __call__(self, tensor):
        return to_pil_image(tensor)

class GaussianNoiseTransform:
    """Add Gaussian noise to a tensor."""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        return tensor + noise

class NormalizeSample:
    """Normalize each sample in a batch."""
    def __call__(self, tensor):
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True)
        return (tensor - mean) / (std + 1e-6)

class RandomMaskTransform:
    """Apply random masking to a tensor."""
    def __init__(self, block_size, probability=0.5):
        self.block_size = block_size
        self.probability = probability

    def __call__(self, tensor):
        if random.random() < self.probability:
            x = random.randint(0, tensor.size(1) - self.block_size)
            y = random.randint(0, tensor.size(2) - self.block_size)
            tensor[:, x:x+self.block_size, y:y+self.block_size] = 0
        return tensor

class AddRandomDCTermPerFeature:
    """Add a random DC term to each feature."""
    def __init__(self, min_value, max_value, num_features):
        self.min_value = min_value
        self.max_value = max_value
        self.num_features = num_features

    def __call__(self, tensor):
        dc_values = torch.FloatTensor(self.num_features).uniform_(self.min_value, self.max_value)
        dc_values = dc_values.view(1, 1, self.num_features).expand_as(tensor)
        return tensor + dc_values

class ShuffleFeatures:
    """Shuffle the features of a tensor."""
    def __init__(self, num_features):
        self.num_features = num_features

    def __call__(self, tensor):
        indices = torch.randperm(self.num_features)
        return tensor[:, :, indices]

class DataAugmentACF(BaseAugmentation):
    """Data augmentation techniques for ACF data."""
    
    def __init__(self, device=None):
        """Initialize the ACF data augmentation.
        
        Args:
            device: The device to use for tensor operations.
        """
        super().__init__(device)
    
    def get_augmentation(self, feature_size):
        """Create an augmentation pipeline for the given feature size.
        
        Args:
            feature_size: The size of the features.
            
        Returns:
            The augmentation pipeline.
        """
        return Compose([
            TensorToPILTransform(),
            RandomHorizontalFlip(),
            RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            RandomGrayscale(p=0.2),
            ToTensor(),
            ShuffleFeatures(num_features=feature_size),
            RandomApply([GaussianNoiseTransform(std=0.1)], p=0.3),
            AddRandomDCTermPerFeature(min_value=-10, max_value=10, num_features=feature_size),
            RandomMaskTransform(block_size=10, probability=0.5),
            NormalizeSample()
        ])
    
    def __call__(self, data):
        """Apply augmentations to the data.
        
        Args:
            data: The data to augment.
            
        Returns:
            The augmented data.
        """
        feature_size = data.shape[-1]
        augmentation_pipeline = self.get_augmentation(feature_size)
        return augmentation_pipeline(data)
    
    def apply_augmentations(self, data):
        """Apply all augmentations to the data.
        
        Args:
            data: The data to augment.
            
        Returns:
            The augmented data.
        """
        return self.__call__(data)
