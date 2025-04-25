import os
import torch
import torch.utils.data
from data.datasets.csi.supervised import CSIDatasetMAT
from data.datasets.acf.supervised import ACFDatasetMAT
from typing import List, Dict, Tuple, Optional

def load_csi_supervised(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    task: str = 'ThreeClass',
    test_dirs: Optional[List[str]] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, torch.utils.data.DataLoader]]:
    """
    Load CSI data for supervised learning with explicit directory structure.
    
    Args:
        train_dir (str): Directory containing training data
        val_dir (str): Directory containing validation data
        batch_size (int): Batch size for data loaders
        task (str): Task type, default to 'ThreeClass'
        test_dirs (List[str], optional): List of test directories
        
    Returns:
        Tuple of (train_loader, val_loader, test_loaders_dict)
        where test_loaders_dict is a dictionary mapping test directory names to test loaders
    """
    # Create datasets from specified directories
    train_dataset = CSIDatasetMAT([train_dir], task)
    val_dataset = CSIDatasetMAT([val_dir], task)
    
    print(f"Loading training data from: {train_dir}")
    print(f"Loading validation data from: {val_dir}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Handle test loaders
    test_loaders = {}
    
    # If no test directories specified, create an empty dictionary
    if test_dirs is None or len(test_dirs) == 0:
        print("No test directories specified.")
        return train_loader, val_loader, test_loaders
    
    # Create test loaders for each specified test directory
    for test_dir in test_dirs:
        # Extract directory name for identification
        dir_name = os.path.basename(os.path.normpath(test_dir))
        
        # Create test dataset
        test_dataset = CSIDatasetMAT([test_dir], task)
        print(f"Loading test data from: {test_dir} (ID: {dir_name})")
        
        # Create test loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        test_loaders[dir_name] = test_loader
    
    return train_loader, val_loader, test_loaders


def load_acf_supervised(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    task: str = 'ThreeClass',
    test_dirs: Optional[List[str]] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, torch.utils.data.DataLoader]]:
    """
    Load ACF data for supervised learning with explicit directory structure.
    
    Args:
        train_dir (str): Directory containing training data
        val_dir (str): Directory containing validation data
        batch_size (int): Batch size for data loaders
        task (str): Task type, default to 'ThreeClass'
        test_dirs (List[str], optional): List of test directories
        
    Returns:
        Tuple of (train_loader, val_loader, test_loaders_dict)
        where test_loaders_dict is a dictionary mapping test directory names to test loaders
    """
    # Create datasets from specified directories
    train_dataset = ACFDatasetMAT([train_dir], task)
    val_dataset = ACFDatasetMAT([val_dir], task)
    
    print(f"Loading training data from: {train_dir}")
    print(f"Loading validation data from: {val_dir}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Handle test loaders
    test_loaders = {}
    
    # If no test directories specified, create an empty dictionary
    if test_dirs is None or len(test_dirs) == 0:
        print("No test directories specified.")
        return train_loader, val_loader, test_loaders
    
    # Create test loaders for each specified test directory
    for test_dir in test_dirs:
        # Extract directory name for identification
        dir_name = os.path.basename(os.path.normpath(test_dir))
        
        # Create test dataset
        test_dataset = ACFDatasetMAT([test_dir], task)
        print(f"Loading test data from: {test_dir} (ID: {dir_name})")
        
        # Create test loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        test_loaders[dir_name] = test_loader
    
    return train_loader, val_loader, test_loaders
