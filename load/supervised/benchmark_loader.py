import os
from torch.utils.data import DataLoader
from .benchmark_dataset import BenchmarkCSIDataset, load_benchmark_datasets
from ..supervised.label_utils import LabelMapper, create_label_mapper_from_metadata

def load_benchmark_supervised(
    dataset_root,
    task_name,
    batch_size=32,
    transform=None,
    target_transform=None,
    file_format="h5",
    data_column="file_path",
    label_column="label",
    data_key="CSI_amps",
    num_workers=4,
    shuffle_train=True,
    train_split="train_id",
    val_split="val_id",
    test_splits=None
):
    """
    Load benchmark dataset for supervised learning.
    
    Args:
        dataset_root: Root directory of the dataset.
        task_name: Name of the task (e.g., 'motion_source_recognition')
        batch_size: Batch size for DataLoader.
        transform: Optional transform to apply to data.
        target_transform: Optional transform to apply to labels.
        file_format: File format for data files ("h5", "mat", or "npy").
        data_column: Column in metadata that contains file paths.
        label_column: Column in metadata that contains labels.
        data_key: Key in h5 file for CSI data.
        num_workers: Number of worker processes for DataLoader.
        shuffle_train: Whether to shuffle training data.
        train_split: Name of training split.
        val_split: Name of validation split.
        test_splits: List of test split names.
        
    Returns:
        Dictionary with data loaders and number of classes.
    """
    # Set default test splits if not provided
    if test_splits is None:
        test_splits = ["test_id"]
    elif isinstance(test_splits, str):
        test_splits = [test_splits]
    
    # Create all split names
    all_splits = [train_split, val_split] + test_splits
    
    # Create metadata path
    if "tasks" in dataset_root and os.path.isdir(os.path.join(dataset_root, task_name)):
        task_dir = os.path.join(dataset_root, task_name)
    else:
        task_dir = os.path.join(dataset_root, "tasks", task_name)
    metadata_path = os.path.join(task_dir, 'metadata', 'sample_metadata.csv')
    mapping_path = os.path.join(task_dir, 'metadata', 'label_mapping.json')
    
    # Create or load label mapper
    if os.path.exists(mapping_path):
        label_mapper = LabelMapper.load(mapping_path)
    else:
        label_mapper, _ = create_label_mapper_from_metadata(
            metadata_path, 
            label_column=label_column,
            save_path=mapping_path
        )
    
    # Load datasets
    datasets = {}
    for split_name in all_splits:
        dataset = BenchmarkCSIDataset(
            dataset_root=dataset_root,
            task_name=task_name,
            split_name=split_name,
            transform=transform,
            target_transform=target_transform,
            file_format=file_format,
            data_column=data_column,
            label_column=label_column,
            data_key=data_key,
            label_mapper=label_mapper
        )
        datasets[split_name] = dataset
    
    # Create data loaders
    loaders = {}
    
    # Training loader
    loaders['train'] = DataLoader(
        datasets[train_split],
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation loader
    loaders['val'] = DataLoader(
        datasets[val_split],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test loaders
    for test_split in test_splits:
        loader_name = f'test_{test_split}' if test_split != 'test_id' else 'test'
        loaders[loader_name] = DataLoader(
            datasets[test_split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Get number of classes from the label mapper
    num_classes = label_mapper.num_classes
    
    # Return everything in a dictionary
    return {
        'loaders': loaders,
        'datasets': datasets,
        'num_classes': num_classes,
        'label_mapper': label_mapper
    }

def load_all_benchmarks(
    dataset_root,
    task_names=None,
    batch_size=32,
    transform=None,
    target_transform=None,
    file_format="h5",
    data_column="file_path",
    label_column="label",
    data_key="CSI_amps",
    num_workers=4,
    shuffle_train=True,
):
    """
    Load all benchmark datasets.
    
    Args:
        dataset_root: Root directory for all benchmarks.
        task_names: List of task names to load. If None, loads all available tasks.
        Other args: Same as load_benchmark_supervised.
        
    Returns:
        Dictionary with data loaders for each task.
    """
    # If no task names provided, discover available tasks
    if task_names is None:
        task_names = []
        for item in os.listdir(dataset_root):
            task_dir = os.path.join(dataset_root, item)
            if os.path.isdir(task_dir) and os.path.exists(os.path.join(task_dir, 'splits')):
                task_names.append(item)
    
    # Load each benchmark
    benchmarks = {}
    for task_name in task_names:
        try:
            benchmark_data = load_benchmark_supervised(
                dataset_root=dataset_root,
                task_name=task_name,
                batch_size=batch_size,
                transform=transform,
                target_transform=target_transform,
                file_format=file_format,
                data_column=data_column,
                label_column=label_column,
                data_key=data_key,
                num_workers=num_workers,
                shuffle_train=shuffle_train
            )
            benchmarks[task_name] = benchmark_data
            print(f"Loaded benchmark '{task_name}' with {benchmark_data['num_classes']} classes")
        except Exception as e:
            print(f"Error loading benchmark '{task_name}': {str(e)}")
    
    return benchmarks
