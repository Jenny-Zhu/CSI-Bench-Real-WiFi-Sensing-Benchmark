import os
import torch
import torch.utils.data
from data.datasets.csi.supervised import CSIDatasetOW_HM3, CSIDatasetOW_HM3_H5, CSIDatasetMAT
from data.datasets.acf.supervised import ACFDatasetOW_HM3_MAT, DatasetNTU_MAT
from torch.utils.data import random_split

def load_data_supervised(task, BATCH_SIZE, win_len, sample_rate, data_dir=None):
    """
    Load CSI data for supervised learning tasks.
    
    Args:
        task (str): Task name, e.g., 'OW_HM3'
        BATCH_SIZE (int): Batch size
        win_len (int): Window length
        sample_rate (int): Sample rate
        data_dir (str): Directory containing data (optional)
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Print start loading data information
    print(f"[Info] Starting to load CSI data, task: {task}, batch size: {BATCH_SIZE}, window length: {win_len}, sample rate: {sample_rate}")
    
    if task == 'OW_HM3':
        # Use provided data_dir instead of hardcoded path
        if data_dir is None:
            data_dir = 'C:\\Guozhen\\Code\\Github\\WiFiSSL\\dataset\\metadata\\HM3_sr100_wl200'
            print(f"[Warning] No data directory provided, using default path: {data_dir}")
        
        print(f"[Info] Loading training data from {data_dir}...")
        try:
            support_set = CSIDatasetOW_HM3_H5(data_dir, win_len, sample_rate, if_test=0)
            print(f"[Success] Training dataset loaded successfully with {len(support_set)} samples")
            
            print(f"[Info] Loading testing data from {data_dir}...")
            test_set = CSIDatasetOW_HM3_H5(data_dir, win_len, sample_rate, if_test=1)
            print(f"[Success] Testing dataset loaded successfully with {len(test_set)} samples")
        except Exception as e:
            print(f"[Error] Data loading failed: {str(e)}")
            return None, None
    else:
        print(f"[Error] Unknown task: {task}")
        return None, None
    
    print(f"[Info] Creating data loaders...")
    support_loader = torch.utils.data.DataLoader(support_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    print(f"[Success] Data loaders created successfully")
    
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


def load_csi_supervised_integrated(data_dir, task='ThreeClass', batch_size=16, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, max_samples=5000):
    """
    Load CSI data using the integrated CSIDatasetMAT class.
    
    Args:
        data_dir (str): Directory containing CSI data
        task (str): Task type, default to 'ThreeClass'
        batch_size (int): Batch size for data loaders, defaults to 16 (decreased from 32)
        train_ratio (float): Ratio of data to use for training
        val_ratio (float): Ratio of data to use for validation
        test_ratio (float): Ratio of data to use for testing
        max_samples (int): Maximum number of samples to load, defaults to 5000
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print(f"Loading CSI data using integrated CSIDatasetMAT from: {data_dir}")
    print(f"Task: {task}, Batch size: {batch_size}, Max samples: {max_samples}")
    
    try:
        # Create dataset
        dataset = CSIDatasetMAT(data_dir, task)
        
        # Print dataset info
        print(f"Dataset loaded: {len(dataset)} samples")
        if hasattr(dataset, 'samples') and dataset.samples is not None:
            print(f"Sample shape: {dataset.samples.shape}")
            
            # 限制样本数量以防止内存问题
            if max_samples and len(dataset) > max_samples:
                print(f"Limiting dataset to {max_samples} samples to avoid memory issues")
                # 创建随机索引，选择max_samples个样本
                indices = torch.randperm(len(dataset))[:max_samples]
                dataset.samples = dataset.samples[indices]
                dataset.labels = [dataset.labels[i] for i in indices]
                print(f"Dataset reduced to {len(dataset.labels)} samples with shape {dataset.samples.shape}")
        
        # Split dataset into train, validation, and test sets
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Check if split sizes are valid
        if train_size <= 0 or val_size <= 0 or test_size <= 0:
            raise ValueError(f"Invalid data split ratios. Got train_size={train_size}, val_size={val_size}, test_size={test_size}")
        
        print(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")
        
        # Perform splits
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
        
        # Create data loaders with num_workers=0 to avoid potential issues with multiprocessing
        train_loader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=0  # 使用单线程加载数据
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_set, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False,
            num_workers=0  # 使用单线程加载数据
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_set, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False,
            num_workers=0  # 使用单线程加载数据
        )
        
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        print(f"Error loading CSI data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


def load_csi_unseen_integrated(data_dir, task='ThreeClass', batch_size=16):
    """
    Load CSI data for unseen environment testing using the integrated CSIDatasetMAT class.
    
    Args:
        data_dir (str): Directory containing test CSI data
        task (str): Task type, default to 'ThreeClass'
        batch_size (int): Batch size for data loader, defaults to 16
        
    Returns:
        Test DataLoader
    """
    print(f"Loading unseen environment CSI data using integrated loader from: {data_dir}")
    
    try:
        # Create dataset
        test_set = CSIDatasetMAT(data_dir, task)
        print(f"Test set loaded: {len(test_set)} samples")
        
        if hasattr(test_set, 'samples') and test_set.samples is not None:
            print(f"Sample shape: {test_set.samples.shape}")
        
        # Create test data loader
        test_loader = torch.utils.data.DataLoader(
            test_set, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False,
            num_workers=0  # 使用单线程加载数据
        )
        
        return test_loader
    
    except Exception as e:
        print(f"Error loading unseen CSI data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
