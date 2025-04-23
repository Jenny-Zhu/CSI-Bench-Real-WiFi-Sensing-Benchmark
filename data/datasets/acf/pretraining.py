import os
import torch
import numpy as np
import scipy.io as sio
import h5py
from data.datasets.base_dataset import BaseDataset
from data.preprocessing.acf_preprocessing import normalize_acf

class SSLACFDatasetMAT(BaseDataset):
    """Dataset for self-supervised learning with ACF data from .mat files."""
    
    def __init__(self, data_dir, transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing ACF data.
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        # Load data
        self.load_data()
    
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
                acf_data = self.load_acf_from_mat(file_path)
                if acf_data is not None:
                    self.data.append(acf_data)
            except Exception as e:
                print(f"Error loading ACF from {file_path}: {e}")
    
    def load_acf_from_mat(self, file_path):
        """Load ACF data from a .mat file.
        
        Args:
            file_path: Path to the .mat file.
            
        Returns:
            The loaded ACF data.
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
                return None
            
            # Normalize and convert to tensor
            acf_data = normalize_acf(acf_data)
            acf_tensor = torch.from_numpy(acf_data).float()
            
            return acf_tensor
            
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
                    return None
                
                # Normalize and convert to tensor
                acf_data = normalize_acf(acf_data)
                acf_tensor = torch.from_numpy(acf_data).float()
                
                return acf_tensor
