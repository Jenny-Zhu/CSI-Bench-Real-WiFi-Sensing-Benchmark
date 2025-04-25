import os
import torch
import torch.utils.data
from data.datasets.csi.supervised import CSIDatasetMAT
from data.datasets.acf.supervised import ACFDatasetMAT
from torch.utils.data import random_split

def load_csi_supervised(data_dir, task='ThreeClass', batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Load CSI data using the integrated CSIDatasetMAT class.
    Supports directory structures:
    1. Single directory with all data (will be split according to ratios)
    2. data_dir/train, data_dir/validation, data_dir/test directories
    3. data_dir/train, data_dir/test directories (validation will be split from train)
    
    Args:
        data_dir (str): Directory containing CSI data or parent directory with train/test subdirs
        task (str): Task type, default to 'ThreeClass'
        batch_size (int): Batch size for data loaders
        train_ratio (float): Ratio of data to use for training (used only if no train/test subdirs)
        val_ratio (float): Ratio of data to use for validation (used only if no train/test subdirs)
        test_ratio (float): Ratio of data to use for testing (used only if no train/test subdirs)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Check if data_dir has train/validation/test subdirectories
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    # Create datasets based on directory structure
    if os.path.exists(train_dir):
        # Using directory structure with separate folders
        train_dataset = CSIDatasetMAT([train_dir], task)
        
        # Check for validation directory
        if os.path.exists(validation_dir):
            # Use dedicated validation directory
            val_dataset = CSIDatasetMAT([validation_dir], task)
            print(f"Loading validation data from: {validation_dir}")
            val_set = val_dataset
        else:
            # Split train dataset for validation
            print("Validation directory not found, splitting from training data")
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_set_temp, val_set = random_split(train_dataset, [train_size, val_size])
            # Recreate train_dataset with only training samples to avoid memory issues
            train_dataset = train_set_temp
        
        # Check for test directory
        if os.path.exists(test_dir):
            # Use dedicated test directory
            test_dataset = CSIDatasetMAT([test_dir], task)
            test_set = test_dataset
        else:
            # If no test directory, use validation set as test set (not ideal but fallback)
            print("Test directory not found, using validation set for testing")
            test_set = val_set
    else:
        # Standard mode - single directory with all data
        print(f"Using single directory mode with: {data_dir}")
        dataset = CSIDatasetMAT(data_dir, task)
        
        # Split into train, validation, and test sets
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        train_dataset = train_set
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader

def load_acf_supervised(data_dir, task='ThreeClass', batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Load ACF data using the integrated ACFDatasetMAT class.
    Supports directory structures:
    1. Single directory with all data (will be split according to ratios)
    2. data_dir/train, data_dir/validation, data_dir/test directories
    3. data_dir/train, data_dir/test directories (validation will be split from train)
    
    Args:
        data_dir (str): Directory containing ACF data or parent directory with train/test subdirs
        task (str): Task type, default to 'ThreeClass'
        batch_size (int): Batch size for data loaders
        train_ratio (float): Ratio of data to use for training (used only if no train/test subdirs)
        val_ratio (float): Ratio of data to use for validation (used only if no train/test subdirs)
        test_ratio (float): Ratio of data to use for testing (used only if no train/test subdirs)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Check if data_dir has train/validation/test subdirectories
    train_dir = os.path.join(data_dir, 'train')
    validation_dir = os.path.join(data_dir, 'validation')
    test_dir = os.path.join(data_dir, 'test')
    
    # Create datasets based on directory structure
    if os.path.exists(train_dir):
        # Using directory structure with separate folders
        train_dataset = ACFDatasetMAT([train_dir], task)
        
        # Check for validation directory
        if os.path.exists(validation_dir):
            # Use dedicated validation directory
            val_dataset = ACFDatasetMAT([validation_dir], task)
            print(f"Loading validation data from: {validation_dir}")
            val_set = val_dataset
        else:
            # Split train dataset for validation
            print("Validation directory not found, splitting from training data")
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_set_temp, val_set = random_split(train_dataset, [train_size, val_size])
            # Recreate train_dataset with only training samples to avoid memory issues
            train_dataset = train_set_temp
        
        # Check for test directory
        if os.path.exists(test_dir):
            # Use dedicated test directory
            test_dataset = ACFDatasetMAT([test_dir], task)
            test_set = test_dataset
        else:
            # If no test directory, use validation set as test set (not ideal but fallback)
            print("Test directory not found, using validation set for testing")
            test_set = val_set
    else:
        # Standard mode - single directory with all data
        print(f"Using single directory mode with: {data_dir}")
        dataset = ACFDatasetMAT(data_dir, task)
        
        # Split into train, validation, and test sets
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        train_dataset = train_set
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader
