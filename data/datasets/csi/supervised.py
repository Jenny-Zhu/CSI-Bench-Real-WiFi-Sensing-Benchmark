import os
import torch
import numpy as np
import h5py
import scipy.io as sio
from torch.utils.data import random_split
from data.datasets.base_dataset import BaseDataset
from data.preprocessing.csi_preprocessing import normalize_csi

class CSIDatasetOW_HM3(BaseDataset):
    """Dataset for supervised learning with CSI data."""
    
    def __init__(self, data_dir, win_len=250, sample_rate=100, if_test=0, transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing CSI data.
            win_len: Window length for segmentation.
            sample_rate: Sampling rate of the data.
            if_test: Whether to use test data (0 for train, 1 for test).
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        self.win_len = win_len
        self.sample_rate = sample_rate
        self.if_test = if_test
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load CSI data from files."""
        for dir_path in self.data_dir:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.h5') or file.endswith('.mat'):
                        file_path = os.path.join(root, file)
                        self.process_file(file_path)
    
    def process_file(self, file_path):
        """Process a single CSI file.
        
        Args:
            file_path: Path to the CSI file.
        """
        try:
            if file_path.endswith('.mat'):
                data, label = self.load_mat_file(file_path)
            elif file_path.endswith('.h5'):
                data, label = self.load_h5_file(file_path)
            else:
                return
            
            # Skip if no valid data or label
            if data is None or label is None:
                return
            
            # Add to dataset
            self.data.append(data)
            self.labels.append(label)
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    def load_mat_file(self, file_path):
        """Load data from a .mat file.
        
        Args:
            file_path: Path to the .mat file.
            
        Returns:
            A tuple of (data, label).
        """
        try:
            mat_data = sio.loadmat(file_path)
            # Extract data and label based on your file structure
            # This is a placeholder
            data = mat_data.get('csi_data', None)
            label = mat_data.get('label', None)
            
            if data is not None:
                data = normalize_csi(data)
                data = torch.from_numpy(data).float()
            
            if label is not None:
                label = torch.from_numpy(label).long()
            
            return data, label
            
        except Exception as e:
            print(f"Error loading MAT file {file_path}: {e}")
            return None, None
    
    def load_h5_file(self, file_path):
        """Load data from an .h5 file.
        
        Args:
            file_path: Path to the .h5 file.
            
        Returns:
            A tuple of (data, label).
        """
        try:
            with h5py.File(file_path, 'r') as f:
                # Extract data and label based on your file structure
                # This is a placeholder
                data = np.array(f.get('csi_data', None))
                label = np.array(f.get('label', None))
                
                if data is not None:
                    data = normalize_csi(data)
                    data = torch.from_numpy(data).float()
                
                if label is not None:
                    label = torch.from_numpy(label).long()
                
                return data, label
                
        except Exception as e:
            print(f"Error loading HDF5 file {file_path}: {e}")
            return None, None

class CSIDatasetOW_HM3_H5(BaseDataset):
    """Dataset for supervised learning with CSI data from HDF5 files."""
    
    def __init__(self, data_dir, win_len=250, sample_rate=100, if_test=0, transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing CSI data.
            win_len: Window length for segmentation.
            sample_rate: Sampling rate of the data.
            if_test: Whether to use test data (0 for train, 1 for test).
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        self.win_len = win_len
        self.sample_rate = sample_rate
        self.if_test = if_test
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load CSI data from HDF5 files."""
        for dir_path in self.data_dir:
            h5_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.h5')]
            
            for file_path in h5_files:
                try:
                    with h5py.File(file_path, 'r') as f:
                        # Determine if this is a test or train file based on naming convention
                        is_test_file = 'test' in os.path.basename(file_path).lower()
                        
                        # Only process files matching the requested split
                        if (self.if_test == 1 and is_test_file) or (self.if_test == 0 and not is_test_file):
                            # Extract data and labels - customize based on actual structure
                            csi_data = np.array(f.get('csi_data', None))
                            labels = np.array(f.get('labels', None))
                            
                            if csi_data is not None and labels is not None:
                                # Process and add each example
                                for i in range(len(labels)):
                                    sample = csi_data[i]
                                    label = labels[i]
                                    
                                    # Normalize and convert to tensor
                                    sample = normalize_csi(sample)
                                    sample_tensor = torch.from_numpy(sample).float()
                                    label_tensor = torch.tensor(label).long()
                                    
                                    self.data.append(sample_tensor)
                                    self.labels.append(label_tensor)
                                    
                except Exception as e:
                    print(f"Error loading HDF5 file {file_path}: {e}")
