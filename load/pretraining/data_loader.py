import os
import torch
import torch.utils.data
from data.datasets.csi.pretraining import SSLCSIDatasetMAT, SSLCSIDataset, SSLCSIDatasetHDF5
from data.datasets.acf.pretraining import SSLACFDatasetMAT
from torch.utils.data import DataLoader
from util.data.bucket_sampler import FeatureBucketBatchSampler
from ..base import variable_shape_collate_fn

def load_acf_data_unsupervised(data_dir, BATCH_SIZE):
    """
    Load ACF data for unsupervised learning.
    
    Args:
        data_dir: Directory containing ACF data
        BATCH_SIZE: Batch size for data loader
        
    Returns:
        DataLoader for ACF data
    """
    ssl_set = SSLACFDatasetMAT(data_dir)
    ssl_loader = torch.utils.data.DataLoader(
        ssl_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    return ssl_loader


def load_csi_data_unsupervised(data_dir, BATCH_SIZE):
    """
    Load CSI data for unsupervised learning.
    
    Args:
        data_dir: Directory containing CSI data
        BATCH_SIZE: Batch size for data loader
        
    Returns:
        DataLoader for CSI data
    """
    ssl_set = SSLCSIDatasetMAT(data_dir)
    sampler = FeatureBucketBatchSampler(ssl_set, batch_size=BATCH_SIZE, shuffle=True)
    ssl_loader = DataLoader(
        ssl_set, 
        batch_sampler=sampler, 
        num_workers=4
    )
    
    return ssl_loader


def load_data_unsupervised(data_dir, BATCH_SIZE, win_len, sample_rate):
    """
    Load generic data for unsupervised learning.
    
    Args:
        data_dir: Directory containing data
        BATCH_SIZE: Batch size for data loader
        win_len: Window length for CSI data
        sample_rate: Sample rate for CSI data
        
    Returns:
        DataLoader for data
    """
    ssl_set = SSLCSIDataset(data_dir, win_len, sample_rate)
    ssl_loader = torch.utils.data.DataLoader(
        ssl_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    return ssl_loader


def load_preprocessed_data_unsupervised(data_dir, BATCH_SIZE, win_len, sample_rate):
    """
    Load preprocessed HDF5 data for unsupervised learning.
    
    Args:
        data_dir: Directory or list of directories containing data
        BATCH_SIZE: Batch size for data loader
        win_len: Window length for CSI data
        sample_rate: Sample rate for CSI data
        
    Returns:
        DataLoader for data
    """
    # Ensure data_dir is a list, even if it's a single path
    if isinstance(data_dir, str):
        data_dir = [data_dir]  # Convert single directory string to a list
    
    # Initialize the dataset with the list of directories
    ssl_set = SSLCSIDatasetHDF5(data_dir, win_len, sample_rate)
    
    # Create a DataLoader to handle batching and shuffling
    ssl_loader = torch.utils.data.DataLoader(
        ssl_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    return ssl_loader
