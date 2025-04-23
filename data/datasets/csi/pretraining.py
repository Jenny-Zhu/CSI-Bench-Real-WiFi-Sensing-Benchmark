import os
import torch
import numpy as np
import h5py
import scipy.io as sio
from data.datasets.base_dataset import BaseDataset
from data.preprocessing.csi_preprocessing import normalize_csi, transform_csi_to_real

class SSLCSIDataset(BaseDataset):
    """Dataset for self-supervised learning with CSI data."""
    
    def __init__(self, data_dir, win_len=250, sample_rate=100, transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing CSI data.
            win_len: Window length for segmentation.
            sample_rate: Sampling rate of the data.
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        self.win_len = win_len
        self.sample_rate = sample_rate
        
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
                data = self.load_mat_file(file_path)
            elif file_path.endswith('.h5'):
                data = self.load_h5_file(file_path)
            else:
                return
            
            # Apply preprocessing
            data = self.preprocess_data(data)
            
            # Add to dataset
            self.data.append(data)
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    def load_mat_file(self, file_path):
        """Load data from a .mat file.
        
        Args:
            file_path: Path to the .mat file.
            
        Returns:
            The loaded data.
        """
        try:
            return sio.loadmat(file_path)
        except NotImplementedError:
            with h5py.File(file_path, 'r') as f:
                return {key: np.array(f[key]) for key in f.keys()}
    
    def load_h5_file(self, file_path):
        """Load data from an .h5 file.
        
        Args:
            file_path: Path to the .h5 file.
            
        Returns:
            The loaded data.
        """
        with h5py.File(file_path, 'r') as f:
            return {key: np.array(f[key]) for key in f.keys()}
    
    def preprocess_data(self, data):
        """Preprocess the loaded data.
        
        Args:
            data: The data to preprocess.
            
        Returns:
            The preprocessed data.
        """
        # This is a placeholder. Implement the actual preprocessing logic.
        # For example, extracting CSI data, normalizing, etc.
        return data

class SSLCSIDatasetMAT(BaseDataset):
    """Dataset for self-supervised learning with CSI data from .mat files."""
    
    def __init__(self, data_dir, transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing CSI data.
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load CSI data from files."""
        mat_files = []
        for dir_path in self.data_dir:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.mat'):
                        mat_files.append(os.path.join(root, file))
        
        for file_path in mat_files:
            try:
                csi_data = self.load_csi_from_mat(file_path)
                if csi_data is not None:
                    self.data.append(csi_data)
            except Exception as e:
                print(f"Error loading CSI from {file_path}: {e}")
    
    def load_csi_from_mat(self, file_path):
        """Load CSI data from a .mat file.
        
        Args:
            file_path: Path to the .mat file.
            
        Returns:
            The loaded CSI data.
        """
        try:
            # Try loading with scipy.io
            mat_data = sio.loadmat(file_path)
            csi_trace = mat_data.get('csi_trace', None)
            if csi_trace is None:
                return None
            
            csi = csi_trace['csi'][0][0]
            csi = normalize_csi(csi)
            
            # Convert to tensor
            csi_tensor = torch.from_numpy(csi).float()
            
            return csi_tensor
            
        except NotImplementedError:
            # Fall back to h5py for MATLAB v7.3 format
            with h5py.File(file_path, 'r') as f:
                if 'csi_trace' not in f:
                    return None
                
                csi_trace = f['csi_trace']
                if 'csi' not in csi_trace:
                    return None
                
                csi = np.array(csi_trace['csi'])
                csi = normalize_csi(csi)
                
                # Convert to tensor
                csi_tensor = torch.from_numpy(csi).float()
                
                return csi_tensor

class SSLCSIDatasetHDF5(BaseDataset):
    """Dataset for self-supervised learning with CSI data from HDF5 files."""
    
    def __init__(self, data_dir, win_len=250, sample_rate=100, transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing CSI data.
            win_len: Window length for segmentation.
            sample_rate: Sampling rate of the data.
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        self.win_len = win_len
        self.sample_rate = sample_rate
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load CSI data from files."""
        h5_files = []
        for dir_path in self.data_dir:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.h5'):
                        h5_files.append(os.path.join(root, file))
        
        for file_path in h5_files:
            try:
                with h5py.File(file_path, 'r') as f:
                    # Extract the CSI data - customize based on actual structure
                    if 'csi' in f:
                        csi_data = np.array(f['csi'])
                        csi_data = normalize_csi(csi_data)
                        self.data.append(torch.from_numpy(csi_data).float())
            except Exception as e:
                print(f"Error loading CSI from {file_path}: {e}")
