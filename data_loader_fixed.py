import os
import torch
import torch.utils.data

from torch.utils.data import DataLoader
from .meta_dataset import MetaTaskDataset
from .meta_data_loader import load_meta_learning_tasks as load_meta_tasks_json

def load_csi_data_benchmark(
    data_dir,
    n_way=3,
    k_shot=5,
    q_query=5,
    batch_size=4,
    file_ext='.h5',
    num_workers=4,
    data_key='CSI_amps',
    **kwargs
):
    """
    Returns a dict of DataLoaders for meta-learning (train/val/test if available).
    
    Args:
        data_dir: Root directory containing the dataset or tasks
        n_way: Number of classes per task
        k_shot: Number of support examples per class
        q_query: Number of query examples per class
        batch_size: Number of tasks per batch
        file_ext: File extension of data files (default: '.h5')
        num_workers: Number of workers for DataLoader
        data_key: Key in h5 file for CSI data (default: 'CSI_amps')
        
    Returns:
        Dict of DataLoaders for each split
    """
    # Check if we're using the meta_splits directory structure
    if 'meta_splits' in os.listdir(os.path.join(data_dir)) or \
       (os.path.basename(data_dir) in ['tasks', 'motion_source_recognition'] and 
        os.path.exists(os.path.join(data_dir, 'meta_splits'))):
        # Use the json-based meta-learning task definition
        return load_meta_tasks_json(
            dataset_root=data_dir,
            task_name="MotionSourceRecognition" if os.path.basename(data_dir) == "tasks" else os.path.basename(data_dir),
            split_types=['train', 'val', 'test'],
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            batch_size=batch_size,
            file_format=file_ext.lstrip('.'),
            data_key=data_key,
            num_workers=num_workers,
            **kwargs
        )
    
    # Otherwise, use the original directory-based approach
    loaders = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            # Check directory structure to determine dataset type
            if any(os.path.isdir(os.path.join(split_dir, d)) for d in os.listdir(split_dir)):
                # Directory contains subdirectories (classes), use MetaTaskDataset
                dataset = MetaTaskDataset(
                    split_dir,
                    n_way=n_way,
                    k_shot=k_shot,
                    q_query=q_query,
                    file_ext=file_ext,
                    data_key=data_key
                )
            else:
                # Try to import CSITaskDataset (optional)
                try:
                    from data.datasets.csi.meta_learning import CSITaskDataset
                    # Directory does not contain subdirectories, use CSITaskDataset
                    # Define label keywords based on the directory structure
                    label_keywords = kwargs.get('label_keywords', {})
                    resize_height = kwargs.get('resize_height', 64)
                    resize_width = kwargs.get('resize_width', 100)
                    
                    dataset = CSITaskDataset(
                        split_dir,
                        k_shot=k_shot,
                        q_query=q_query,
                        resize_height=resize_height,
                        resize_width=resize_width,
                        label_keywords=label_keywords,
                        data_key=data_key
                    )
                except ImportError:
                    print(f"Warning: CSITaskDataset not available, skipping {split_dir}")
                    continue
            
            # Check if we should use a custom collate function
            use_collate_fn = not isinstance(dataset, MetaTaskDataset)
            
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=task_collate_fn if use_collate_fn else None
            )
    
    return loaders

def task_collate_fn(batch):
    """
    Custom collate function for CSITaskDataset that samples a task for each batch item.
    
    Args:
        batch: List of dataset items (ignored since CSITaskDataset samples tasks directly)
        
    Returns:
        Dict with 'support' and 'query' tuples
    """
    # For CSITaskDataset, we need to manually sample tasks
    try:
        support_x, support_y, query_x, query_y = batch[0].sample_task()
        
        return {
            'support': (support_x, support_y),
            'query': (query_x, query_y)
        }
    except AttributeError:
        # If sample_task is not available, return the first item
        return batch[0]

def load_multi_source_data(
    data_dirs,
    n_way=3,
    k_shot=5,
    q_query=5,
    batch_size=4,
    file_ext='.h5',
    num_workers=4,
    data_key='CSI_amps',
    **kwargs
):
    """
    Returns a dict of DataLoaders for meta-learning from multiple data sources.
    
    Args:
        data_dirs: List of root directories
        n_way: Number of classes per task
        k_shot: Number of support examples per class
        q_query: Number of query examples per class
        batch_size: Number of tasks per batch
        file_ext: File extension of data files
        num_workers: Number of workers for DataLoader
        data_key: Key in h5 file for CSI data (default: 'CSI_amps')
        
    Returns:
        Dict of DataLoaders for each split
    """
    # Import here to avoid circular imports
    try:
        from data.datasets.csi.meta_learning import MultiSourceTaskDataset
        
        loaders = {}
        
        for split in ['train', 'val', 'test']:
            datasets = []
            
            for data_dir in data_dirs:
                split_dir = os.path.join(data_dir, split)
                if os.path.exists(split_dir):
                    # Handle different dataset types based on directory structure
                    if any(os.path.isdir(os.path.join(split_dir, d)) for d in os.listdir(split_dir)):
                        datasets.append(
                            MetaTaskDataset(
                                split_dir,
                                n_way=n_way,
                                k_shot=k_shot,
                                q_query=q_query,
                                file_ext=file_ext,
                                data_key=data_key
                            )
                        )
                    else:
                        label_keywords = kwargs.get('label_keywords', {})
                        resize_height = kwargs.get('resize_height', 64)
                        resize_width = kwargs.get('resize_width', 100)
                        
                        from data.datasets.csi.meta_learning import CSITaskDataset
                        datasets.append(
                            CSITaskDataset(
                                split_dir,
                                k_shot=k_shot,
                                q_query=q_query,
                                resize_height=resize_height,
                                resize_width=resize_width,
                                label_keywords=label_keywords,
                                data_key=data_key
                            )
                        )
            
            if datasets:
                # If we have only one dataset, use it directly
                if len(datasets) == 1:
                    dataset = datasets[0]
                    collate_fn = task_collate_fn if not isinstance(dataset, MetaTaskDataset) else None
                else:
                    # Otherwise, use MultiSourceTaskDataset
                    dataset = MultiSourceTaskDataset(datasets)
                    collate_fn = task_collate_fn
                
                loaders[split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    collate_fn=collate_fn
                )
        
        return loaders
    except ImportError:
        print("Warning: MultiSourceTaskDataset not available. Loading from primary data directory only.")
        return load_csi_data_benchmark(
            data_dir=data_dirs[0] if data_dirs else ".",
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            batch_size=batch_size,
            file_ext=file_ext,
            num_workers=num_workers,
            data_key=data_key,
            **kwargs
        )
