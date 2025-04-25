import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BaseDataset(Dataset):
    """Base class for all datasets."""
    
    def __init__(self, data_dir, transform=None):
        """Initialize base dataset.
        
        Args:
            data_dir: Directory or list of directories containing data
            transform: Transform to apply to the data
        """
        self.transform = transform
        
        # Convert single directory to list
        if isinstance(data_dir, str):
            self.data_dir = [data_dir]
        else:
            self.data_dir = data_dir
        
        # Storage for data and labels
        self.data = []
        self.labels = []
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, index):
        """Return a sample from the dataset."""
        sample = self.data[index]
        label = self.labels[index]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
        
        
class TrainDataset(BaseDataset):
    """Base class for training datasets."""
    
    def __init__(self, data_dir, transform=None):
        """Initialize training dataset.
        
        Args:
            data_dir: Directory or list of directories containing training data
            transform: Transform to apply to the data
        """
        super().__init__(data_dir, transform)
        
    def apply_augmentation(self, sample, augmentor):
        """Apply data augmentation if provided.
        
        Args:
            sample: Input sample
            augmentor: Data augmentation object
            
        Returns:
            Augmented sample
        """
        if augmentor is not None:
            return augmentor(sample)
        return sample
        
        
class ValidationDataset(BaseDataset):
    """Base class for validation datasets."""
    
    def __init__(self, data_dir, transform=None):
        """Initialize validation dataset.
        
        Args:
            data_dir: Directory or list of directories containing validation data
            transform: Transform to apply to the data
        """
        super().__init__(data_dir, transform)
        
        
class TestDataset(BaseDataset):
    """Base class for test datasets."""
    
    def __init__(self, data_dir, transform=None):
        """Initialize test dataset.
        
        Args:
            data_dir: Directory or list of directories containing test data
            transform: Transform to apply to the data
        """
        super().__init__(data_dir, transform)
        
    def get_metadata(self):
        """Get metadata about the test set if available.
        
        Returns:
            Dictionary of metadata
        """
        return {
            "num_samples": len(self.labels),
            "class_distribution": self._get_class_distribution()
        }
        
    def _get_class_distribution(self):
        """Get the distribution of classes in the dataset.
        
        Returns:
            Dictionary with class counts
        """
        if not self.labels:
            return {}
            
        class_counts = {}
        for label in self.labels:
            label_item = label.item() if torch.is_tensor(label) else label
            class_counts[label_item] = class_counts.get(label_item, 0) + 1
        
        return class_counts
