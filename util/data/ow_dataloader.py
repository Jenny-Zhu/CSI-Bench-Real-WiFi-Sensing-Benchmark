import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import h5py
import random

class OWDataloader(Dataset):
    """
    Dataset class for OpenWiFi data loading.
    
    Handles both CSI and ACF data formats, with options for different preprocessing
    and augmentation strategies.
    """
    def __init__(self, data_dir, transform=None, label_keywords=None, is_inference=False):
        """
        Initialize the OpenWiFi dataset.
        
        Args:
            data_dir (str): Directory containing the data files
            transform (callable, optional): Optional transform to be applied on a sample
            label_keywords (dict, optional): Dictionary mapping label keywords to class indices
            is_inference (bool): Whether this dataset is for inference only
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_inference = is_inference
        
        # Setup label keywords if not provided
        if label_keywords is None:
            self.label_keywords = {
                "empty": 0,
                "human": 1,
                "multiple": 2
            }
        else:
            self.label_keywords = label_keywords
        
        # Get all data files
        self.files = self._get_files()
        
    def _get_files(self):
        """Get all valid data files from the data directory."""
        files = []
        
        if isinstance(self.data_dir, list):
            # Handle multiple data directories
            for dir_path in self.data_dir:
                files.extend(self._get_files_from_dir(dir_path))
        else:
            # Single data directory
            files = self._get_files_from_dir(self.data_dir)
            
        return files
        
    def _get_files_from_dir(self, dir_path):
        """Get valid data files from a single directory."""
        files = []
        
        # Check if directory exists
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist.")
            return files
            
        # Get all files with supported extensions
        for filename in os.listdir(dir_path):
            if filename.endswith(('.h5', '.npy')):
                file_path = os.path.join(dir_path, filename)
                label = self._extract_label(filename)
                
                if label is not None:
                    files.append((file_path, label))
                    
        return files
        
    def _extract_label(self, filename):
        """Extract class label from filename based on label keywords."""
        if self.is_inference:
            # For inference, assign dummy label
            return 0
            
        # Check each label keyword in the filename
        for keyword, label_idx in self.label_keywords.items():
            if keyword.lower() in filename.lower():
                return label_idx
                
        # If no keyword found, return None
        print(f"Warning: No valid label found for {filename}")
        return None
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.files)
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            tuple: (data, label) where data is a tensor and label is an integer
        """
        file_path, label = self.files[idx]
        
        # Load data based on file extension
        if file_path.endswith('.h5'):
            data = self._load_h5(file_path)
        else:  # .npy
            data = self._load_npy(file_path)
            
        # Apply transformations if any
        if self.transform:
            data = self.transform(data)
            
        # Convert to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
            
        return data, label
        
    def _load_h5(self, file_path):
        """Load data from H5 file."""
        with h5py.File(file_path, 'r') as f:
            # Assuming 'csi' is the dataset name
            if 'csi' in f:
                data = f['csi'][:]
            elif 'acf' in f:
                data = f['acf'][:]
            else:
                raise ValueError(f"No recognized dataset in {file_path}")
                
        return data
        
    def _load_npy(self, file_path):
        """Load data from NPY file."""
        data = np.load(file_path)
        return data

def create_dataloader(data_dir, batch_size=32, transform=None, label_keywords=None, 
                     shuffle=True, num_workers=4, is_inference=False):
    """
    Create a DataLoader for OpenWiFi data.
    
    Args:
        data_dir: Directory containing data files
        batch_size: Batch size for DataLoader
        transform: Optional transform to apply to data
        label_keywords: Dictionary mapping label keywords to class indices
        shuffle: Whether to shuffle the data
        num_workers: Number of worker threads for loading
        is_inference: Whether this is for inference mode
        
    Returns:
        DataLoader object
    """
    dataset = OWDataloader(
        data_dir=data_dir,
        transform=transform,
        label_keywords=label_keywords,
        is_inference=is_inference
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
