import os
import torch
import torch.utils.data
from data.datasets.csi.supervised import CSIDatasetOW_HM3, CSIDatasetOW_HM3_H5
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
    Load ACF data for supervised learning tasks.
    
    Args:
        data_dir (str): Directory containing ACF data
        task (str): Task name
        batch_size (int): Batch size
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Get the class number of different tasks
    classes = {
        'HumanNonhuman': 2, 
        'FourClass': 4, 
        'NTUHumanID': 15, 
        'HumanID': 4, 
        'HumanMotion': 3, 
        'ThreeClass': 3, 
        'DetectionandClassification': 5, 
        'Detection': 2
    }
    
    # Retrieve the data
    train_set = ACFDatasetOW_HM3_MAT(data_dir, task)
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


def load_acf_unseen_environ(data_dir, task):
    """
    Load ACF data for testing on unseen environments.
    
    Args:
        data_dir (str): Directory containing ACF test data
        task (str): Task name
        
    Returns:
        DataLoader for test data
    """
    # Get the class number of different tasks
    classes = {
        'HumanNonhuman': 2, 
        'FourClass': 4, 
        'NTUHumanID': 15, 
        'HumanID': 4, 
        'HumanMotion': 3, 
        'ThreeClass': 3, 
        'DetectionandClassification': 5, 
        'Detection': 2
    }
    
    test_set = ACFDatasetOW_HM3_MAT(data_dir, task)
    unseen_test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)
    return unseen_test_loader


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
