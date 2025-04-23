import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BaseDataset(Dataset):
    """Base class for all datasets."""
    
    def __init__(self, data_dir, transform=None):
        """Initialize the base dataset.
        
        Args:
            data_dir: The directory containing the data.
            transform: The transform to apply to the data.
        """
        if isinstance(data_dir, str):
            self.data_dir = [data_dir]
        elif isinstance(data_dir, list):
            self.data_dir = data_dir
        else:
            raise TypeError("data_dir should be a string or a list of strings.")
        
        self.transform = transform
        self.data = []
        self.labels = []
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset.
        
        Args:
            idx: The index of the sample to return.
            
        Returns:
            The sample at the given index.
        """
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if len(self.labels) > 0:
            label = self.labels[idx]
            return sample, label
        else:
            return sample
