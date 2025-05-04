import os
import json
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from collections import defaultdict

# Label mapper class for handling label conversions
class LabelMapper:
    def __init__(self, label_column='label', save_path=None):
        self.label_column = label_column
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.num_classes = 0
        self.save_path = save_path
    
    def fit(self, metadata_path=None, metadata_df=None, labels=None):
        if metadata_path is not None:
            metadata_df = pd.read_csv(metadata_path)
            unique_labels = metadata_df[self.label_column].unique()
        elif metadata_df is not None:
            unique_labels = metadata_df[self.label_column].unique()
        elif labels is not None:
            unique_labels = np.unique(labels)
        else:
            raise ValueError("Either metadata_path, metadata_df, or labels must be provided")
        
        # Sort labels for deterministic mapping
        unique_labels = sorted(unique_labels)
        
        # Create mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        
        print(f"Created label mapping with {self.num_classes} classes:")
        for label, idx in self.label_to_idx.items():
            print(f"  {label} -> {idx}")
        
        return self
    
    def transform(self, labels):
        if isinstance(labels, (list, np.ndarray, pd.Series)):
            return np.array([self.label_to_idx.get(label, 0) for label in labels])
        else:
            # Single label
            return self.label_to_idx.get(labels, 0)
    
    def inverse_transform(self, indices):
        if isinstance(indices, (list, np.ndarray)):
            return np.array([self.idx_to_label.get(idx, "unknown") for idx in indices])
        else:
            # Single index
            return self.idx_to_label.get(indices, "unknown")
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            mapping = json.load(f)
        
        mapper = cls()
        mapper.label_to_idx = mapping['label_to_idx']
        mapper.idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}
        mapper.num_classes = mapping['num_classes']
        
        return mapper

class MetaTaskSampler:
    """
    Sampler for meta-learning tasks from json files.
    Each task consists of a support set and a query set.
    """
    def __init__(self, 
                 dataset_root, 
                 task_name,
                 task_file_path,
                 file_ext='.h5',
                 data_key='CSI_amps',
                 label_column='label',
                 metadata_file=None,
                 n_way=5,
                 k_shot=5,
                 q_query=5,
                 device='cpu'):
        
        self.dataset_root = dataset_root
        self.task_name = task_name
        self.file_ext = file_ext
        self.data_key = data_key
        self.label_column = label_column
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.device = device
        
        # Load task file
        with open(task_file_path, 'r') as f:
            self.tasks = json.load(f)
        
        # Load metadata
        if metadata_file is None:
            if "tasks" in dataset_root and os.path.isdir(os.path.join(dataset_root, task_name)):
                task_dir = os.path.join(dataset_root, task_name)
            else:
                task_dir = os.path.join(dataset_root, "tasks", task_name)
            metadata_file = os.path.join(task_dir, "metadata", "subset_metadata.csv")
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
        # Create better id mappings that handle different ID formats
        self.id_mapping = {}
        
        # Try both 'id' and 'sample_id' columns
        id_columns = ['id', 'sample_id']
        for id_col in id_columns:
            if id_col in self.metadata.columns:
                # Clean IDs: Convert to string, strip whitespace, and create mapping
                self.metadata[id_col] = self.metadata[id_col].astype(str).str.strip()
                
                # Create mappings for different formats of the same ID
                for idx, row in self.metadata.iterrows():
                    sample_id = row[id_col]
                    
                    # Add original ID
                    self.id_mapping[sample_id] = idx
                    
                    # Add without spaces
                    self.id_mapping[sample_id.replace(" ", "")] = idx
                    
                    # Add without underscores
                    self.id_mapping[sample_id.replace("_", "")] = idx
                    
                    # Add trimmed version
                    self.id_mapping[sample_id.strip()] = idx
        
        print(f"Loaded {len(self.tasks)} tasks from {task_file_path}")
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        """Get a meta-learning task by index"""
        task = self.tasks[idx]
        
        # Extract support and query sample IDs
        support_ids = task.get('support', [])
        query_ids = task.get('query', [])
        
        # Load support set
        support_data, support_labels = self._load_samples(support_ids)
        
        # Load query set
        query_data, query_labels = self._load_samples(query_ids)
        
        # Skip empty tasks
        if support_data.shape[0] == 0 or query_data.shape[0] == 0:
            # Use a default task with random data for debugging
            support_data = torch.randn(2, 1, 232, 500)
            support_labels = torch.tensor([0, 1], dtype=torch.long)
            query_data = torch.randn(2, 1, 232, 500)
            query_labels = torch.tensor([0, 1], dtype=torch.long)
        
        # Package the task
        meta_task = {
            'task_id': task.get('task_id', f"task_{idx}"),
            'subject': task.get('subject', "unknown"),
            'user': task.get('user', "unknown"),
            'support': (support_data, support_labels),
            'query': (query_data, query_labels)
        }
        
        return meta_task
    
    def _load_samples(self, sample_ids):
        """Load a set of samples by their IDs"""
        if not sample_ids:
            # Return a non-empty tensor with zero samples
            return torch.zeros((0, 1, 232, 500)), torch.zeros(0, dtype=torch.long)
        
        data_list = []
        label_list = []
        
        for sample_id in sample_ids:
            # Try to find the row in metadata
            row_idx = self._get_row_index(sample_id)
            
            if row_idx is not None:
                row = self.metadata.iloc[row_idx]
                
                # Get file path and label
                filepath = row['file_path'] if 'file_path' in row else row.get('path', None)
                label = row[self.label_column]
                
                # Check if filepath exists
                if filepath is None:
                    print(f"Warning: No file path found for sample {sample_id}")
                    continue
                
                # Construct full path if needed
                if filepath.startswith('./'):
                    filepath = filepath[2:]
                if not os.path.isabs(filepath):
                    filepath = os.path.join(self.dataset_root, filepath)
                
                # Load the data file
                try:
                    if self.file_ext == '.h5':
                        with h5py.File(filepath, 'r') as f:
                            # Try different keys
                            if self.data_key in f:
                                csi_data = np.array(f[self.data_key])
                            elif 'csi' in f:
                                csi_data = np.array(f['csi'])
                            elif 'CSI' in f:
                                csi_data = np.array(f['CSI'])
                            else:
                                print(f"Warning: No valid data key found in {filepath}. Keys: {list(f.keys())}")
                                continue
                    elif self.file_ext == '.npy':
                        csi_data = np.load(filepath)
                    else:
                        print(f"Warning: Unsupported file extension {self.file_ext}")
                        continue
                    
                    # Convert to tensor
                    csi_tensor = torch.from_numpy(csi_data).float()
                    
                    # Reshape to (1, time_index, feature_size) if needed
                    if len(csi_tensor.shape) == 3:  # (time_index, feature_size, 1)
                        csi_tensor = csi_tensor.permute(2, 0, 1)
                    
                    # Add to lists
                    data_list.append(csi_tensor)
                    label_list.append(label)
                    
                except Exception as e:
                    print(f"Error loading sample {sample_id} from {filepath}: {e}")
            else:
                # Skip this warning as it's noisy in the output
                # print(f"Warning: Sample ID {sample_id} not found in metadata")
                pass
        
        # Return placeholders if no valid samples found
        if not data_list:
            # Use a random tensor with correct shape for debugging
            return torch.zeros((0, 1, 232, 500)), torch.zeros(0, dtype=torch.long)
        
        # Convert to tensors
        data_tensor = torch.stack(data_list)
        
        # Convert string labels to integers
        unique_labels = sorted(set(label_list))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        label_indices = [label_to_idx[label] for label in label_list]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        return data_tensor, label_tensor
    
    def _get_row_index(self, sample_id):
        """Try to find the row index for a sample ID using multiple formats"""
        # Try the original ID first
        if sample_id in self.id_mapping:
            return self.id_mapping[sample_id]
        
        # Try alternative formats
        # Without spaces
        alt_id = sample_id.replace(" ", "")
        if alt_id in self.id_mapping:
            return self.id_mapping[alt_id]
        
        # Without underscores
        alt_id = sample_id.replace("_", "")
        if alt_id in self.id_mapping:
            return self.id_mapping[alt_id]
        
        # Trimmed
        alt_id = sample_id.strip()
        if alt_id in self.id_mapping:
            return self.id_mapping[alt_id]
        
        # If we still can't find it, return None
        return None

def load_meta_learning_tasks(dataset_root, task_name, split_types=['train', 'val', 'test'], 
                             n_way=5, k_shot=5, q_query=5, batch_size=4, num_workers=0,
                             collate_fn=None, device='cpu'):
    """
    Load meta-learning tasks for specified split types.
    
    Args:
        dataset_root: Root directory of the dataset
        task_name: Name of the task (e.g., 'MotionSourceRecognition')
        split_types: List of split types to load (e.g., ['train', 'val', 'test'])
        n_way: Number of classes per task
        k_shot: Number of support examples per class
        q_query: Number of query examples per class
        batch_size: Number of tasks per batch
        num_workers: Number of worker processes for DataLoader
        collate_fn: Custom collate function
        device: Device to load tensors to
        
    Returns:
        Dict of DataLoaders, with split names as keys
    """
    loaders = {}
    
    # Determine paths
    if "tasks" in dataset_root and os.path.isdir(os.path.join(dataset_root, task_name)):
        task_dir = os.path.join(dataset_root, task_name)
    else:
        task_dir = os.path.join(dataset_root, "tasks", task_name)
    
    meta_splits_dir = os.path.join(task_dir, "meta_splits")
    metadata_file = os.path.join(task_dir, "metadata", "subset_metadata.csv")
    
    # Load tasks for each split type
    for split_type in split_types:
        # Check for basic split types
        task_file = f"{split_type}_tasks.json"
        task_path = os.path.join(meta_splits_dir, task_file)
        
        # If not found, check for specific split types
        if not os.path.exists(task_path):
            # Check other possible filenames
            if split_type == 'cross_env':
                task_file = "test_cross_env_tasks.json"
            elif split_type == 'cross_user':
                task_file = "test_cross_user_tasks.json"
            elif split_type == 'cross_device':
                task_file = "test_cross_device_tasks.json"
            elif split_type.startswith('adapt'):
                task_file = f"{split_type}_tasks.json"
            else:
                print(f"Warning: No task file found for split type {split_type}")
                continue
            
            task_path = os.path.join(meta_splits_dir, task_file)
            
            if not os.path.exists(task_path):
                print(f"Warning: Task file {task_path} not found")
                continue
        
        # Create task sampler
        sampler = MetaTaskSampler(
            dataset_root=dataset_root,
            task_name=task_name,
            task_file_path=task_path,
            metadata_file=metadata_file,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            device=device
        )
        
        # Create DataLoader
        loaders[split_type] = DataLoader(
            sampler,
            batch_size=batch_size,
            shuffle=(split_type == 'train'),
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        
        print(f"Created loader for {split_type} split with {len(sampler)} tasks")
    
    return loaders

def load_csi_data_benchmark(data_dir, n_way=5, k_shot=5, q_query=5, batch_size=4, 
                            file_ext='.h5', data_key='CSI_amps', num_workers=0):
    """
    Load the CSI data for meta-learning.
    Wrapper around load_meta_learning_tasks.
    """
    # Determine task name from data_dir
    task_name = os.path.basename(data_dir)
    if task_name in ['wifi_benchmark_dataset', 'tasks', 'data']:
        task_name = 'MotionSourceRecognition'  # Default task name
    
    # Determine dataset root
    if "tasks" in data_dir and os.path.isdir(os.path.join(data_dir, task_name)):
        dataset_root = os.path.dirname(data_dir)
    else:
        dataset_root = data_dir
    
    # Define a custom collate function for meta-learning tasks
    def custom_task_collate(batch):
        """Custom collate function for tasks with variable sizes"""
        # Handle a batch of size 1 special case
        if len(batch) == 1:
            return batch[0]
        
        # Extract all keys from the first batch element
        keys = batch[0].keys()
        
        result = {}
        for key in keys:
            if key in ['support', 'query']:
                # Handle tuple of (data, labels)
                x_list, y_list = [], []
                for item in batch:
                    x, y = item[key]
                    x_list.append(x)
                    y_list.append(y)
                
                # Store as a tuple
                result[key] = (x_list, y_list)
            elif isinstance(batch[0][key], str):
                # Handle string values like 'task_id', 'subject', 'user'
                result[key] = [item[key] for item in batch]
            else:
                # Handle other data types if needed
                try:
                    result[key] = torch.stack([item[key] for item in batch])
                except:
                    # If can't stack, just keep as list
                    result[key] = [item[key] for item in batch]
        
        return result
    
    # Load meta-learning tasks
    return load_meta_learning_tasks(
        dataset_root=dataset_root,
        task_name=task_name,
        split_types=['train', 'val', 'test', 'cross_env', 'cross_user', 'cross_device',
                    'adapt_1shot', 'adapt_5shot', 'cross_env_adapt_1shot',
                    'cross_env_adapt_5shot', 'cross_user_adapt_1shot',
                    'cross_user_adapt_5shot', 'cross_device_adapt_1shot', 
                    'cross_device_adapt_5shot'],
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=custom_task_collate
    )

# Test function
if __name__ == "__main__":
    import argparse
    
    def main():
        parser = argparse.ArgumentParser(description='Test meta-learning data loader')
        parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                            help='Root directory of the dataset')
        parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                            help='Name of the task')
        parser.add_argument('--n_way', type=int, default=2,
                            help='Number of classes per task')
        parser.add_argument('--k_shot', type=int, default=5,
                            help='Number of support examples per class')
        parser.add_argument('--q_query', type=int, default=5,
                            help='Number of query examples per class')
        parser.add_argument('--data_key', type=str, default='CSI_amps',
                            help='Key in h5 file for CSI data')
        parser.add_argument('--split_type', type=str, default='train',
                            choices=['train', 'val', 'test', 'test_cross_env', 'test_cross_user', 
                                    'adapt_1shot', 'adapt_5shot'],
                            help='Split type to test')
        args = parser.parse_args()
        
        print(f"Loading meta-learning tasks from {args.data_dir}...")
        
        # Load meta-learning tasks
        loaders = load_meta_learning_tasks(
            dataset_root=args.data_dir,
            task_name=args.task_name,
            split_types=[args.split_type],
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            batch_size=1,  # For testing, use batch size 1
            file_ext=args.data_key.lstrip('.'),
            data_key=args.data_key,
            num_workers=0  # For testing, use 0 workers
        )
        
        # Check if requested split exists
        if args.split_type not in loaders:
            print(f"Split {args.split_type} not found in loaders. Available splits: {list(loaders.keys())}")
            return
        
        # Get the loader for the specified split
        loader = loaders[args.split_type]
        print(f"Loaded {args.split_type} loader with {len(loader.dataset)} tasks")
        
        # Sample a task and print its properties
        for batch_idx, batch in enumerate(loader):
            print(f"\nTask {batch_idx+1}:")
            
            # Print task information
            if 'task_id' in batch:
                print(f"  Task ID: {batch['task_id'][0]}")
            if 'subject' in batch:
                print(f"  Subject: {batch['subject'][0]}")
            if 'user' in batch:
                print(f"  User: {batch['user'][0]}")
            
            # Print support and query set information
            support_x, support_y = batch['support']
            query_x, query_y = batch['query']
            
            print(f"  Support set: {support_x.shape}, labels: {support_y.shape}")
            print(f"  Query set: {query_x.shape}, labels: {query_y.shape}")
            
            # Print unique labels
            support_labels = support_y.unique().tolist()
            query_labels = query_y.unique().tolist()
            print(f"  Support labels: {support_labels}")
            print(f"  Query labels: {query_labels}")
            
            # Print data statistics
            print(f"  Support set min/max/mean: {support_x.min():.4f}/{support_x.max():.4f}/{support_x.mean():.4f}")
            print(f"  Query set min/max/mean: {query_x.min():.4f}/{query_x.max():.4f}/{query_x.mean():.4f}")
            
            # Only show the first task
            if batch_idx == 0:
                break
    
    main() 