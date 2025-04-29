from torch.utils.data import DataLoader
from .benchmark_dataset import BenchmarkCSIDataset
import os

def load_benchmark_supervised(
    dataset_root,
    task_name,
    batch_size=32,
    transform=None,
    target_transform=None,
    file_format="mat",
    data_column="filepath",
    label_column="label",
    num_workers=4,
    shuffle_train=True,
    train_split="train_id",
    val_split="val_id",
    test_splits=None
):
    """
    Load supervised learning data from the WiFi benchmark dataset.
    
    Args:
        dataset_root: Root directory of the dataset
        task_name: Name of the task (e.g., 'motion_source_recognition')
        batch_size: Batch size for DataLoader
        transform: Transform for data
        target_transform: Transform for labels
        file_format: Data file format ("mat", "npy", or "h5")
        data_column: Column in metadata that points to data
        label_column: Column in metadata for label
        num_workers: Number of workers for DataLoader
        shuffle_train: Whether to shuffle training data
        train_split: Name of training split
        val_split: Name of validation split
        test_splits: List of test split names or None
        
    Returns:
        Dict of DataLoaders
    """
    loaders = {}
    
    # Training data
    train_dataset = BenchmarkCSIDataset(
        dataset_root=dataset_root,
        task_name=task_name,
        split_name=train_split,
        transform=transform,
        target_transform=target_transform,
        file_format=file_format,
        data_column=data_column,
        label_column=label_column
    )
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    
    # Validation data (if available)
    if val_split:
        val_path = os.path.join(dataset_root, "tasks", task_name, "splits", f"{val_split}.json")
        if os.path.exists(val_path):
            val_dataset = BenchmarkCSIDataset(
                dataset_root=dataset_root,
                task_name=task_name,
                split_name=val_split,
                transform=transform,
                target_transform=target_transform,
                file_format=file_format,
                data_column=data_column,
                label_column=label_column
            )
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
    
    # Test data
    if test_splits:
        if isinstance(test_splits, str):
            test_splits = [test_splits]
        
        for test_split in test_splits:
            test_path = os.path.join(dataset_root, "tasks", task_name, "splits", f"{test_split}.json")
            if os.path.exists(test_path):
                test_dataset = BenchmarkCSIDataset(
                    dataset_root=dataset_root,
                    task_name=task_name,
                    split_name=test_split,
                    transform=transform,
                    target_transform=target_transform,
                    file_format=file_format,
                    data_column=data_column,
                    label_column=label_column
                )
                loaders[f'test_{test_split}'] = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers
                )
    
    return loaders
