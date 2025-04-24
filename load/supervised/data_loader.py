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
            
            # Check if the dataset is empty
            if len(support_set) == 0:
                print(f"[Error] Training dataset is empty. No data was loaded from {data_dir}")
                return None, None
                
            print(f"[Success] Training dataset loaded successfully with {len(support_set)} samples")
            
            print(f"[Info] Loading testing data from {data_dir}...")
            test_set = CSIDatasetOW_HM3_H5(data_dir, win_len, sample_rate, if_test=1)
            
            # Check if the test dataset is empty
            if len(test_set) == 0:
                print(f"[Error] Test dataset is empty. No data was loaded from {data_dir}")
                return None, None
                
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
    Load CSI data with integrated loader for supervised learning.
    
    Args:
        data_dir: Directory containing the dataset
        task: Task type (e.g., ThreeClass, HumanNonhuman)
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of training set (will include validation data)
        test_ratio: Ratio of test set
        max_samples: Maximum number of samples to load
        
    Returns:
        Tuple of (train_loader, test_loader) - Changed to match old implementation
    """
    try:
        print(f"[Info] Loading CSI data with integrated loader from {data_dir}")
        dataset = CSIDatasetMAT(data_dir, task, max_samples=max_samples)
        
        # Check if the dataset is empty
        if len(dataset) == 0:
            print(f"[Error] Integrated dataset is empty. No data was loaded from {data_dir}")
            raise ValueError(f"No data found in {data_dir} for task {task}")
        
        print(f"[Success] Loaded {len(dataset)} samples for task {task}")
        
        # Calculate split sizes - combine train and validation
        dataset_size = len(dataset)
        train_size = int((train_ratio + val_ratio) * dataset_size)  # Combined train and validation
        test_size = dataset_size - train_size
        
        print(f"[Info] Splitting dataset: train={train_size}, test={test_size}")
        
        # Split the dataset
        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size]
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        print(f"[Success] Data loaders created successfully")
        return train_loader, test_loader
    
    except Exception as e:
        print(f"[Error] Error loading integrated CSI data: {str(e)}")
        raise


def load_csi_unseen_integrated(data_dir, task='ThreeClass', batch_size=16, max_samples=None):
    """
    Load CSI data for unseen environments using integrated loader.
    
    Args:
        data_dir: Directory containing test data
        task: Task type (e.g., ThreeClass, HumanNonhuman)
        batch_size: Batch size for dataloaders
        max_samples: Maximum number of samples to load per dataset
        
    Returns:
        Tuple of (train_loader, test_loader) - Changed to match old implementation
    """
    try:
        print(f"[Info] Loading unseen environment CSI data from {data_dir}")
        # Load training data
        train_path = os.path.join(data_dir, 'train')
        print(f"[Info] Loading training data from {train_path}")
        train_dataset = CSIDatasetMAT(train_path, task, max_samples=max_samples)
        
        # Check if the training dataset is empty
        if len(train_dataset) == 0:
            print(f"[Error] Training dataset is empty. No data was loaded from {train_path}")
            raise ValueError(f"No training data found in {train_path} for task {task}")
        
        print(f"[Success] Loaded {len(train_dataset)} training samples")
        
        # Load test data
        test_path = os.path.join(data_dir, 'test')
        print(f"[Info] Loading test data from {test_path}")
        test_dataset = CSIDatasetMAT(test_path, task, max_samples=max_samples)
        
        # Check if the test dataset is empty
        if len(test_dataset) == 0:
            print(f"[Error] Test dataset is empty. No data was loaded from {test_path}")
            raise ValueError(f"No test data found in {test_path} for task {task}")
        
        print(f"[Success] Loaded {len(test_dataset)} test samples")
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        print(f"[Success] Data loaders created successfully")
        return train_loader, test_loader
    
    except Exception as e:
        print(f"[Error] Error loading unseen CSI data: {str(e)}")
        raise
