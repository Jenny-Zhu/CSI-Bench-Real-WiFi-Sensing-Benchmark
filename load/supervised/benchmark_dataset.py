import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from scipy.io import loadmat
from pathlib import Path
import scipy.io as sio
from ..supervised.label_utils import LabelMapper, create_label_mapper_from_metadata

class BenchmarkCSIDataset(Dataset):
    """
    Dataset for WiFi CSI benchmark with split handling and metadata loading.
    Supports MAT, NPY, and HDF5 formats.
    Each sample has shape (time_index, feature_size, 1) in H5 files.
    """
    def __init__(self, 
                 dataset_root,  # Root directory of the dataset
                 task_name,     # Name of the task (e.g., 'motion_source_recognition')
                 split_name,    # Split name (e.g., 'train_id', 'test_cross_user')
                 transform=None,
                 target_transform=None,
                 file_format="h5",  # "mat", "npy", or "h5"
                 data_column="file_path",  # Column in metadata that points to data
                 label_column="label",   # Column in metadata for label
                 data_key="CSI_amps",   # Key in h5 file for data
                 label_mapper=None):   # Optional label mapper for converting string labels to indices
        self.dataset_root = dataset_root
        self.task_name = task_name
        self.transform = transform
        self.target_transform = target_transform
        self.file_format = file_format.lower()
        self.data_column = data_column
        self.label_column = label_column
        self.data_key = data_key
        
        # Build paths
        if "tasks" in dataset_root and os.path.isdir(os.path.join(dataset_root, task_name)):
            task_dir = os.path.join(dataset_root, task_name)
        else:
            task_dir = os.path.join(dataset_root, "tasks", task_name)
        split_path = os.path.join(task_dir, "splits", f"{split_name}.json")
        metadata_path = os.path.join(task_dir, "metadata", "subset_metadata.csv")
        
        # Load split IDs
        with open(split_path, 'r') as f:
            self.split_ids = set(json.load(f))
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        
        # Filter metadata for this split
        id_column = 'id' if 'id' in self.metadata.columns else 'sample_id'
        self.split_metadata = self.metadata[self.metadata[id_column].isin(self.split_ids)].reset_index(drop=True)
        
        # Check if we have the required columns
        if data_column not in self.split_metadata.columns:
            raise ValueError(f"Data column '{data_column}' not found in metadata")
        if label_column not in self.split_metadata.columns:
            raise ValueError(f"Label column '{label_column}' not found in metadata")
            
        print(f"Loaded {len(self.split_metadata)} samples for {task_name} - {split_name}")
        
        # Initialize label mapper if not provided
        if label_mapper is None:
            if "tasks" in dataset_root and os.path.isdir(os.path.join(dataset_root, task_name)):
                task_dir = os.path.join(dataset_root, task_name)
            else:
                task_dir = os.path.join(dataset_root, "tasks", task_name)
            metadata_path = os.path.join(task_dir, 'metadata', 'subset_metadata.csv')
            mapping_path = os.path.join(task_dir, 'metadata', 'label_mapping.json')
            
            # Create mapper directory if it doesn't exist
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            
            if os.path.exists(mapping_path):
                # Load existing mapping
                self.label_mapper = LabelMapper.load(mapping_path)
                print(f"Loaded existing label mapping with {self.label_mapper.num_classes} classes")
            else:
                # Create new mapping
                self.label_mapper, _ = create_label_mapper_from_metadata(
                    metadata_path, 
                    label_column=label_column,
                    save_path=mapping_path
                )
        else:
            self.label_mapper = label_mapper
        
        # Set number of classes
        self.num_classes = self.label_mapper.num_classes
    
    def __len__(self):
        return len(self.split_metadata)
    
    def __getitem__(self, idx):
        """Get sample by index"""
        row = self.split_metadata.iloc[idx]
        
        # Get file path (might be relative to dataset_root)
        original_filepath = row[self.data_column]
        filepath = original_filepath
        
   
        
        # # Debug print - original path from CSV
        # print(f"Original path from CSV: {original_filepath}")
        # print(f"Dataset root: {self.dataset_root}")
        
        # Handle different path formats
        if not os.path.isabs(filepath):
            # Case 1: Path includes 'tasks/TaskName/...'
            if filepath.startswith('/tasks/') or filepath.startswith('tasks/') or filepath.startswith('tasks\\'):
                filepath = os.path.join(self.dataset_root, filepath)
            
            # Case 2: Path is just 'TaskName/...'
            elif filepath.startswith(f"{self.task_name}/") or filepath.startswith(f"{self.task_name}\\"):
                filepath = os.path.join(self.dataset_root, 'tasks', filepath)
            
            # Case 3: Path is relative to task directory
            else:
                filepath = os.path.join(self.dataset_root, 'tasks', self.task_name, filepath)
        
        # # Debug print - constructed path
        # print(f"Final path constructed: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            # Try alternative constructions if file not found
            alt_paths = []
            
            # Alternative 1: Try joining directly
            alt1 = os.path.join(self.dataset_root, original_filepath)
            alt_paths.append(("Direct join", alt1))
            
            # Alternative 2: Try with 'tasks' prefix
            if 'tasks' not in original_filepath:
                alt2 = os.path.join(self.dataset_root, 'tasks', original_filepath)
                alt_paths.append(("With tasks prefix", alt2))
            
            # Alternative 3: Try with task name
            if self.task_name not in original_filepath:
                alt3 = os.path.join(self.dataset_root, 'tasks', self.task_name, original_filepath)
                alt_paths.append(("With task name", alt3))
            
            # Check alternatives
            for desc, alt_path in alt_paths:
                print(f"Trying alternative path ({desc}): {alt_path}")
                if os.path.exists(alt_path):
                    print(f"Found file using alternative path: {alt_path}")
                    filepath = alt_path
                    break
            
            # If still not found, raise error with detailed information
            if not os.path.exists(filepath):
                error_msg = f"File not found: {filepath}\n"
                error_msg += f"Original path from CSV: {original_filepath}\n"
                error_msg += f"Dataset root: {self.dataset_root}\n"
                error_msg += "Tried alternative paths:\n"
                for desc, alt_path in alt_paths:
                    error_msg += f"  - {desc}: {alt_path}\n"
                raise FileNotFoundError(error_msg)
        
        # Load data based on file format
        if self.file_format == "npy":
            csi_data = np.load(filepath)
        elif self.file_format == "mat":
            mat_dict = loadmat(filepath)
            csi_data = self._extract_csi_from_mat(mat_dict)
        elif self.file_format == "h5":
            with h5py.File(filepath, 'r') as f:
                # Use the data_key (default is 'CSI_amps') instead of hardcoded 'csi'
                if self.data_key in f:
                    csi_data = np.array(f[self.data_key])
                else:
                    # Fallback to checking other common keys
                    if 'csi' in f:
                        csi_data = np.array(f['csi'])
                    elif 'CSI' in f:
                        csi_data = np.array(f['CSI'])
                    else:
                        raise KeyError(f"Could not find data in H5 file. Available keys: {list(f.keys())}")
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
        
        # Convert to tensor
        csi_data = torch.from_numpy(csi_data).float()
        
        # Reshape to (1, time_index, feature_size)
        if len(csi_data.shape) == 3:  # (time_index, feature_size, 1)
            # Permute to get (1, time_index, feature_size)
            csi_data = csi_data.permute(2, 0, 1)
        
        # Apply transforms
        if self.transform:
            csi_data = self.transform(csi_data)
        
        # Get label
        label = row[self.label_column]
        if self.target_transform:
            label = self.target_transform(label)
        
        # Convert string label to integer using mapper
        label_idx = self.label_mapper.transform(label)
        
        return csi_data, label_idx
    
    def _extract_csi_from_mat(self, mat_dict):
        """Extract CSI data from MAT file dict"""
        # Adjust this based on your MAT file structure
        if 'csi_data' in mat_dict:
            return mat_dict['csi_data']
        elif 'csi' in mat_dict:
            return mat_dict['csi']
        elif 'CSI_amps' in mat_dict:
            return mat_dict['CSI_amps']
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
        
    def get_label_names(self):
        """Return the unique label names"""
        return self.split_metadata[self.label_column].unique().tolist()

# Add a helper function to load multiple datasets
def load_benchmark_datasets(dataset_root, task_name, splits=None, **kwargs):
    """
    Load datasets for specified splits
    
    Args:
        dataset_root: Root directory of the dataset
        task_name: Name of the task (e.g., 'motion_source_recognition')
        splits: List of split names to load (default: ['train_id', 'val_id', 'test_id'])
        **kwargs: Additional arguments passed to BenchmarkCSIDataset
        
    Returns:
        Dict of datasets, with split names as keys
    """
    if splits is None:
        splits = ['train_id', 'val_id', 'test_id']
    
    datasets = {}
    
    for split_name in splits:
        if "tasks" in dataset_root and os.path.isdir(os.path.join(dataset_root, task_name)):
            task_dir = os.path.join(dataset_root, task_name)
        else:
            task_dir = os.path.join(dataset_root, "tasks", task_name)
        split_path = os.path.join(task_dir, "splits", f"{split_name}.json")
        if os.path.exists(split_path):
            datasets[split_name] = BenchmarkCSIDataset(
                dataset_root=dataset_root,
                task_name=task_name,
                split_name=split_name,
                **kwargs
            )
    
    return datasets
