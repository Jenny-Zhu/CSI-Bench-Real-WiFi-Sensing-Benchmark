import os
import torch
import numpy as np
import scipy.io as sio
import h5py
from torch.utils.data import random_split
from data.datasets.base_dataset import BaseDataset
from data.preprocessing.acf_preprocessing import normalize_acf

class ACFDatasetOW_HM3_MAT(BaseDataset):
    """Dataset for supervised learning with ACF data from .mat files."""
    
    def __init__(self, data_dir, task='HumanNonhuman', transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing ACF data.
            task: The task for which to load data (e.g., 'HumanNonhuman', 'FourClass').
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        self.task = task
        
        # Define class mappings for different tasks
        self.class_mappings = {
            'HumanNonhuman': {'human': 1, 'nonhuman': 0},
            'FourClass': {'empty': 0, 'human': 1, 'animal': 2, 'object': 3},
            'HumanID': {'person1': 0, 'person2': 1, 'person3': 2, 'person4': 3},
            'HumanMotion': {'static': 0, 'walking': 1, 'running': 2},
            'ThreeClass': {'empty': 0, 'human': 1, 'nonhuman': 2},
            'DetectionandClassification': {'empty': 0, 'human': 1, 'animal': 2, 'object': 3, 'multiple': 4},
            'Detection': {'empty': 0, 'nonempty': 1}
        }
        
        # Load data
        self.load_data()
        
        # Convert lists to torch tensors or numpy arrays
        self.samples = np.array(self.data)
        self.labels = self.labels
    
    def load_data(self):
        """Load ACF data from files."""
        mat_files = []
        for dir_path in self.data_dir:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.mat'):
                        mat_files.append(os.path.join(root, file))
        
        for file_path in mat_files:
            try:
                data, label = self.load_acf_from_mat(file_path)
                if data is not None and label is not None:
                    self.data.append(data)
                    self.labels.append(label)
            except Exception as e:
                print(f"Error loading ACF from {file_path}: {e}")
    
    def load_acf_from_mat(self, file_path):
        """Load ACF data and label from a .mat file.
        
        Args:
            file_path: Path to the .mat file.
            
        Returns:
            A tuple of (data, label).
        """
        try:
            # Try loading with scipy.io
            mat_data = sio.loadmat(file_path)
            
            # Extract ACF data - this is a placeholder, modify based on actual structure
            acf_data = None
            for key in mat_data.keys():
                if 'acf' in key.lower() or 'data' in key.lower():
                    acf_data = mat_data[key]
                    break
            
            if acf_data is None:
                return None, None
            
            # Determine the label based on the file path and task
            label = self.determine_label(file_path)
            
            if label is None:
                return None, None
            
            # Normalize data
            acf_data = normalize_acf(acf_data)
            
            return acf_data, label
            
        except NotImplementedError:
            # Fall back to h5py for MATLAB v7.3 format
            with h5py.File(file_path, 'r') as f:
                # Extract ACF data - this is a placeholder, modify based on actual structure
                acf_data = None
                for key in f.keys():
                    if 'acf' in key.lower() or 'data' in key.lower():
                        acf_data = np.array(f[key])
                        break
                
                if acf_data is None:
                    return None, None
                
                # Determine the label based on the file path and task
                label = self.determine_label(file_path)
                
                if label is None:
                    return None, None
                
                # Normalize data
                acf_data = normalize_acf(acf_data)
                
                return acf_data, label
    
    def determine_label(self, file_path):
        """Determine the label for a file based on its path and the current task.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            The label for the file.
        """
        file_name = os.path.basename(file_path).lower()
        
        # Get the class mapping for the current task
        mapping = self.class_mappings.get(self.task, {})
        
        # Determine the label based on the file name
        for class_name, class_label in mapping.items():
            if class_name.lower() in file_name:
                return class_label
        
        # Try determining label from parent directory name
        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
        for class_name, class_label in mapping.items():
            if class_name.lower() in parent_dir:
                return class_label
        
        return None
    
    def __getitem__(self, idx):
        """Get an item from the dataset.
        
        Args:
            idx: Index of the item.
            
        Returns:
            A tuple of (sample, label).
        """
        sample = self.samples[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

class DatasetNTU_MAT(BaseDataset):
    """Dataset for supervised learning with NTU data from .mat files."""
    
    def __init__(self, data_dir, task='NTUHumanID', transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing NTU data.
            task: The task for which to load data (e.g., 'NTUHumanID', 'NTUHAR').
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        self.task = task
        
        # Define class mappings for different tasks
        self.class_mappings = {
            'NTUHumanID': {f'person{i}': i for i in range(15)},
            'NTUHAR': {
                'walking': 0, 'sitting': 1, 'standing': 2,
                'jumping': 3, 'falling': 4, 'lying': 5
            },
            'Widar': {f'activity{i}': i for i in range(22)}
        }
        
        # Load data
        self.load_data()
        
        # Convert lists to torch tensors or numpy arrays
        self.samples = np.array(self.data)
        self.labels = self.labels
    
    def load_data(self):
        """Load NTU data from files."""
        mat_files = []
        for dir_path in self.data_dir:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.mat'):
                        mat_files.append(os.path.join(root, file))
        
        for file_path in mat_files:
            try:
                data, label = self.load_ntu_from_mat(file_path)
                if data is not None and label is not None:
                    self.data.append(data)
                    self.labels.append(label)
            except Exception as e:
                print(f"Error loading NTU data from {file_path}: {e}")
    
    def load_ntu_from_mat(self, file_path):
        """Load NTU data and label from a .mat file.
        
        Args:
            file_path: Path to the .mat file.
            
        Returns:
            A tuple of (data, label).
        """
        try:
            # Try loading with scipy.io
            mat_data = sio.loadmat(file_path)
            
            # Extract data - this is a placeholder, modify based on actual structure
            ntu_data = None
            for key in mat_data.keys():
                if 'data' in key.lower() or 'features' in key.lower():
                    ntu_data = mat_data[key]
                    break
            
            if ntu_data is None:
                return None, None
            
            # Determine the label based on the file path and task
            label = self.determine_label(file_path)
            
            if label is None:
                return None, None
            
            return ntu_data, label
            
        except NotImplementedError:
            # Fall back to h5py for MATLAB v7.3 format
            with h5py.File(file_path, 'r') as f:
                # Extract data - this is a placeholder, modify based on actual structure
                ntu_data = None
                for key in f.keys():
                    if 'data' in key.lower() or 'features' in key.lower():
                        ntu_data = np.array(f[key])
                        break
                
                if ntu_data is None:
                    return None, None
                
                # Determine the label based on the file path and task
                label = self.determine_label(file_path)
                
                if label is None:
                    return None, None
                
                return ntu_data, label
    
    def determine_label(self, file_path):
        """Determine the label for a file based on its path and the current task.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            The label for the file.
        """
        file_name = os.path.basename(file_path).lower()
        
        # Get the class mapping for the current task
        mapping = self.class_mappings.get(self.task, {})
        
        # Determine the label based on the file name
        for class_name, class_label in mapping.items():
            if class_name.lower() in file_name:
                return class_label
        
        # Try determining label from parent directory name
        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
        for class_name, class_label in mapping.items():
            if class_name.lower() in parent_dir:
                return class_label
        
        return None
