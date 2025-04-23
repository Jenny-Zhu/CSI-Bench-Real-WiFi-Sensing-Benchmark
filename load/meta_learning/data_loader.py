import os
import torch
import torch.utils.data
from data.datasets.csi.meta_learning import CSITaskDataset, MultiSourceTaskDataset

def load_csi_data_benchmark(data_dir, resize_height, resize_width, label_keywords, k_shot, q_query):
    """
    Load CSI data for benchmark meta-learning tasks.
    
    Args:
        data_dir (str or list): Directory or directories containing CSI data
        resize_height (int): Height to resize images to
        resize_width (int): Width to resize images to
        label_keywords (dict): Dictionary mapping label keywords to class indices
        k_shot (int): Number of examples per class in support set
        q_query (int): Number of examples per class in query set
        
    Returns:
        Tuple of (train_datasets, task_dataset) where train_datasets is a list of
        CSITaskDataset instances and task_dataset is a MultiSourceTaskDataset
        instance that combines all train_datasets.
    """
    train_folderpaths = data_dir
    if isinstance(train_folderpaths, str):
        train_folderpaths = [train_folderpaths]  # Convert to list if string
    
    train_datasets = []
    
    print(f"\n[Info] Starting to load CSI datasets from {len(train_folderpaths)} folders...\n")
    
    for idx, path in enumerate(train_folderpaths):
        print(f"[{idx+1}/{len(train_folderpaths)}] Loading folder: {path}...")
        
        try:
            dataset = CSITaskDataset(
                folder_path=path,
                k_shot=k_shot,
                q_query=q_query,
                resize_height=resize_height,
                resize_width=resize_width,
                label_keywords=label_keywords
            )
            train_datasets.append(dataset)
            
            print(f"    -> Loaded {len(dataset.dataset)} samples.")
        except AssertionError as e:
            print(f"[WARNING] Skipping folder {path}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed loading folder {path}: {e}")
    
    print(f"\n[Info] Finished loading. {len(train_datasets)} datasets loaded.\n")
    
    task_dataset = MultiSourceTaskDataset(train_datasets)
    
    return train_datasets, task_dataset
