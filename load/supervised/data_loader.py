import os
import torch
import torch.utils.data
from data.datasets.csi.supervised import CSIDatasetOW_HM3, CSIDatasetOW_HM3_H5, CSIDatasetMAT
from data.datasets.acf.supervised import ACFDatasetOW_HM3_MAT, DatasetNTU_MAT
from torch.utils.data import random_split

def load_data_supervised(task, BATCH_SIZE, win_len, sample_rate):
    """
    Load CSI data for supervised learning tasks.
    
    Args:
        task (str): Task name, e.g., 'OW_HM3'
        BATCH_SIZE (int): Batch size
        win_len (int): Window length
        sample_rate (int): Sample rate
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if task == 'OW_HM3':
        data_dir = 'C:\\Guozhen\\Code\\Github\\WiFiSSL\\dataset\\metadata\\HM3_sr100_wl200'
        support_set = CSIDatasetOW_HM3_H5(data_dir, win_len, sample_rate, if_test=0)
        test_set = CSIDatasetOW_HM3_H5(data_dir, win_len, sample_rate, if_test=1)
    else:
        print(f"Task unknown: {task}")
        return None, None
    
    support_loader = torch.utils.data.DataLoader(support_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    return support_loader, test_loader


def save_data_supervised(task, BATCH_SIZE, win_len, sample_rate):
    """
    Create and return dataset instances for supervised learning tasks.
    
    Args:
        task (str): Task name, e.g., 'OW_HM3'
        BATCH_SIZE (int): Not used, kept for API compatibility
        win_len (int): Window length
        sample_rate (int): Sample rate
        
    Returns:
        Tuple of (support_set, test_set)
    """
    if task == 'OW_HM3':
        support_set = CSIDatasetOW_HM3(win_len, sample_rate, if_test=0)
        test_set = CSIDatasetOW_HM3(win_len, sample_rate, if_test=1)
        return support_set, test_set
    else:
        print(f"Task unknown: {task}")
        return None, None


def load_acf_supervised(data_dir, task, batch_size):
    """
    Load ACF data for supervised learning.
    Supports both direct data directory and directory with train/test subdirectories.
    
    Args:
        data_dir (str): Directory containing ACF data or parent directory with train/test subdirs
        task (str): Task name
        batch_size (int): Batch size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Check if data_dir has train/test subdirectories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Create datasets based on directory structure
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Legacy mode with separate directories
        train_dataset = ACFDatasetOW_HM3_MAT(train_dir, task)
        test_dataset = ACFDatasetOW_HM3_MAT(test_dir, task)
        
        # Split train dataset for train and validation (using 80/20 split)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_set, val_set = random_split(train_dataset, [train_size, val_size])
        test_set = test_dataset
    else:
        # Standard mode - single directory
        dataset = ACFDatasetOW_HM3_MAT(data_dir, task)
        
        # Split into train, validation, and test sets
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, 
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


def load_acf_unseen_environ(data_dir, task):
    """
    Load ACF data for testing on unseen environments.
    Supports both direct data directory and directory with test subdirectory.
    
    Args:
        data_dir (str): Directory containing ACF test data or parent directory with test subdir
        task (str): Task name
        
    Returns:
        DataLoader for test data
    """
    # Check if test subdirectory exists
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        test_path = test_dir
    else:
        test_path = data_dir
    
    # Create dataset and loader
    test_set = ACFDatasetOW_HM3_MAT(test_path, task)
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=32, 
        shuffle=False
    )
    
    return test_loader


def load_acf_supervised_NTUHumanID(data_dir, task, batch_size):
    """
    Load NTU ACF data for supervised learning tasks.
    
    Args:
        data_dir (str): Directory containing NTU ACF data
        task (str): Task name
        batch_size (int): Batch size
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Get the class number of different tasks
    classes = {'NTUHumanID': 15, 'NTUHAR': 6, 'Widar': 22}
    
    # Retrieve the data
    train_set = DatasetNTU_MAT(data_dir, task)
    num_human = train_set.labels.count(1)
    print(f"Labels count: {len(train_set.labels)}")
    print(f"Sample shape: {train_set.samples.shape}")
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(train_set))
    valid_size = len(train_set) - train_size
    train_set, valid_set = random_split(train_set, [train_size, valid_size])
    
    # Create data loaders for train and validation sets
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    return train_loader, test_loader


def load_acf_supervised_NTUHumanID_fewshot(data_dir, task, batch_size):
    """
    Load NTU ACF data for few-shot supervised learning tasks.
    
    Args:
        data_dir (str): Directory containing NTU ACF data
        task (str): Task name
        batch_size (int): Batch size
        
    Returns:
        DataLoader for train data
    """
    # Get the class number of different tasks
    classes = {'NTUHumanID': 15, 'NTUHAR': 6, 'Widar': 22}
    
    # Retrieve the data
    train_set = DatasetNTU_MAT(data_dir, task)
    
    # Create data loader for train set
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False
    )
    
    return train_loader


def load_acf_unseen_environ_NTU(data_dir, task):
    """
    Load NTU ACF data for testing on unseen environments.
    
    Args:
        data_dir (str): Directory containing NTU ACF test data
        task (str): Task name
        
    Returns:
        DataLoader for test data
    """
    # Get the class number of different tasks
    classes = {
        'HumanNonhuman': 2, 
        'FourClass': 4, 
        'NTUHumanID': 15, 
        'NTUHAR': 6, 
        'HumanID': 4, 
        'Widar': 22,
        'HumanMotion': 3, 
        'ThreeClass': 3, 
        'DetectionandClassification': 5, 
        'Detection': 2
    }
    
    test_set = DatasetNTU_MAT(data_dir, task)
    unseen_test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    return unseen_test_loader


def load_csi_supervised_integrated(data_dir, task='ThreeClass', batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Load CSI data using the integrated CSIDatasetMAT class.
    Supports both direct data directory and directory with train/test subdirectories.
    
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
    # Check if data_dir has train/test subdirectories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Create datasets based on directory structure
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Legacy mode with separate directories
        train_dataset = CSIDatasetMAT([train_dir], task)
        test_dataset = CSIDatasetMAT([test_dir], task)
        
        # Split train dataset for train and validation (using 80/20 split)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_set, val_set = random_split(train_dataset, [train_size, val_size])
        test_set = test_dataset
    else:
        # Standard mode - single directory
        dataset = CSIDatasetMAT(data_dir, task)
        
        # Split into train, validation, and test sets
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, 
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


def load_csi_unseen_integrated(data_dir, task='ThreeClass', batch_size=32):
    """
    Load CSI data for unseen environment testing.
    Supports both direct data directory and directory with test subdirectory.
    
    Args:
        data_dir (str): Directory containing test CSI data or parent directory with test subdir
        task (str): Task type, default to 'ThreeClass'
        batch_size (int): Batch size for data loader
        
    Returns:
        Test DataLoader
    """
    # Check if test subdirectory exists
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        test_path = test_dir
    else:
        test_path = data_dir
    
    # Create dataset and loader
    test_set = CSIDatasetMAT([test_path], task)
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return test_loader


def load_csi_supervised_with_train_test_dirs(data_dir, task='ThreeClass', batch_size=32):
    """
    Load CSI data from separate train and test directories.
    This function supports the old file structure where training and test data
    are stored in separate directories (data_dir/train and data_dir/test).
    
    Args:
        data_dir (str): Base directory containing 'train' and 'test' subdirectories
        task (str): Task type, default to 'ThreeClass'
        batch_size (int): Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    print(f"Loading CSI data from train directory: {train_dir}")
    print(f"Loading CSI data from test directory: {test_dir}")
    print(f"Task: {task}, Batch size: {batch_size}")
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory does not exist: {train_dir}")
    
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory does not exist: {test_dir}")
        print("Will split training data for validation and testing")
        test_dir = None
    
    # Create training dataset
    train_dataset = CSIDatasetMAT([train_dir], task)
    print(f"Training dataset loaded: {len(train_dataset)} samples")
    
    if hasattr(train_dataset, 'samples') and train_dataset.samples is not None:
        print(f"Training sample shape: {train_dataset.samples.shape}")
    
    # Create test dataset if test directory exists
    if test_dir and os.path.exists(test_dir):
        test_dataset = CSIDatasetMAT([test_dir], task)
        print(f"Test dataset loaded: {len(test_dataset)} samples")
        
        # Split training data for train and validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_set, val_set = random_split(train_dataset, [train_size, val_size])
        
        # Use full test set
        test_set = test_dataset
    else:
        # Split training data for train, validation, and test
        train_size = int(0.8 * len(train_dataset))
        val_size = int(0.1 * len(train_dataset))
        test_size = len(train_dataset) - train_size - val_size
        
        print(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")
        train_set, val_set, test_set = random_split(train_dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def load_acf_supervised_with_train_test_dirs(data_dir, task='ThreeClass', batch_size=32):
    """
    Load ACF data from separate train and test directories.
    This function supports the old file structure where training and test data
    are stored in separate directories (data_dir/train and data_dir/test).
    
    Args:
        data_dir (str): Base directory containing 'train' and 'test' subdirectories
        task (str): Task type, default to 'ThreeClass'
        batch_size (int): Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    print(f"Loading ACF data from train directory: {train_dir}")
    print(f"Loading ACF data from test directory: {test_dir}")
    print(f"Task: {task}, Batch size: {batch_size}")
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory does not exist: {train_dir}")
    
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory does not exist: {test_dir}")
        print("Will split training data for validation and testing")
        test_dir = None
    
    # Create training dataset
    train_dataset = ACFDatasetOW_HM3_MAT(train_dir, task)
    print(f"Training dataset loaded: {len(train_dataset)} samples")
    
    if hasattr(train_dataset, 'samples') and train_dataset.samples is not None:
        print(f"Training sample shape: {train_dataset.samples.shape}")
    
    # Create test dataset if test directory exists
    if test_dir and os.path.exists(test_dir):
        test_dataset = ACFDatasetOW_HM3_MAT(test_dir, task)
        print(f"Test dataset loaded: {len(test_dataset)} samples")
        
        # Split training data for train and validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_set, val_set = random_split(train_dataset, [train_size, val_size])
        
        # Use full test set
        test_set = test_dataset
    else:
        # Split training data for train, validation, and test
        train_size = int(0.8 * len(train_dataset))
        val_size = int(0.1 * len(train_dataset))
        test_size = len(train_dataset) - train_size - val_size
        
        print(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")
        train_set, val_set, test_set = random_split(train_dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
