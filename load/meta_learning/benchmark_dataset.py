import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from scipy.io import loadmat

class BenchmarkCSIDataset(Dataset):
    """
    Dataset for WiFi CSI benchmark with split handling and metadata loading.
    Supports MAT, NPY, and HDF5 formats.
    """
    def __init__(self, 
                 dataset_root,  # Root directory of the dataset
                 task_name,     # Name of the task (e.g., 'motion_source_recognition')
                 split_name,    # Split name (e.g., 'train_id', 'test_cross_user')
                 transform=None,
                 target_transform=None,
                 file_format="mat",  # "mat", "npy", or "h5"
                 data_column="filepath",  # Column in metadata that points to data
                 label_column="label"):   # Column in metadata for label
        self.dataset_root = dataset_root
        self.task_name = task_name
        self.transform = transform
        self.target_transform = target_transform
        self.file_format = file_format.lower()
        self.data_column = data_column
        self.label_column = label_column
        
        # Build paths
        task_dir = os.path.join(dataset_root, "tasks", task_name)
        split_path = os.path.join(task_dir, "splits", f"{split_name}.json")
        metadata_path = os.path.join(task_dir, "metadata", "sample_metadata.csv")
        
        # Load split IDs
        with open(split_path, 'r') as f:
            self.split_ids = set(json.load(f))
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter metadata for this split
        self.split_metadata = self.metadata[self.metadata['sample_id'].isin(self.split_ids)].reset_index(drop=True)
        
        # Check if we have the required columns
        if data_column not in self.split_metadata.columns:
            raise ValueError(f"Data column '{data_column}' not found in metadata")
        if label_column not in self.split_metadata.columns:
            raise ValueError(f"Label column '{label_column}' not found in metadata")
            
        print(f"Loaded {len(self.split_metadata)} samples for {task_name} - {split_name}")
    
    def __len__(self):
        return len(self.split_metadata)
    
    def __getitem__(self, idx):
        """Get sample by index"""
        row = self.split_metadata.iloc[idx]
        
        # Get file path (might be relative to dataset_root)
        filepath = row[self.data_column]
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.dataset_root, filepath)
        
        # Load data based on file format
        if self.file_format == "npy":
            csi_data = np.load(filepath)
        elif self.file_format == "mat":
            mat_dict = loadmat(filepath)
            csi_data = self._extract_csi_from_mat(mat_dict)
        elif self.file_format == "h5":
            with h5py.File(filepath, 'r') as f:
                csi_data = np.array(f['csi'])  # Adjust key as needed
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
        
        # Convert to tensor
        csi_data = torch.from_numpy(csi_data).float()
        
        # Apply transforms
        if self.transform:
            csi_data = self.transform(csi_data)
        
        # Get label
        label = row[self.label_column]
        if self.target_transform:
            label = self.target_transform(label)
        
        return csi_data, label
    
    def _extract_csi_from_mat(self, mat_dict):
        """Extract CSI data from MAT file dict"""
        # Adjust this based on your MAT file structure
        if 'csi_data' in mat_dict:
            return mat_dict['csi_data']
        elif 'csi' in mat_dict:
            return mat_dict['csi']
        else:
            # Try to find a likely CSI array
            largest_array = None
            largest_size = 0
            for key, value in mat_dict.items():
                if isinstance(value, np.ndarray) and value.size > largest_size:
                    largest_array = value
                    largest_size = value.size
            return largest_array

    def get_label_counts(self):
        """Return a dictionary of label counts"""
        return self.split_metadata[self.label_column].value_counts().to_dict()
