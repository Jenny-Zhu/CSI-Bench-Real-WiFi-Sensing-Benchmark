import os
import argparse
import torch
import json
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import h5py

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

# Meta-task sampler for loading from JSON task definitions
class MetaTaskSampler(Dataset):
    def __init__(
        self,
        dataset_root,
        task_file_path,
        n_way=None,
        k_shot=None,
        q_query=None,
        transform=None,
        target_transform=None,
        file_format="h5",
        data_column="file_path",
        label_column="label",
        data_key="CSI_amps",
        id_column="id"
    ):
        self.dataset_root = dataset_root
        self.task_file_path = task_file_path
        self.transform = transform
        self.target_transform = target_transform
        self.file_format = file_format.lower()
        self.data_column = data_column
        self.label_column = label_column
        self.data_key = data_key
        self.id_column = id_column
        
        # Override task parameters if provided
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        
        # Load task definitions from JSON
        with open(task_file_path, 'r') as f:
            self.tasks = json.load(f)
        
        # Load metadata to find file paths for sample IDs
        task_name = os.path.basename(os.path.dirname(os.path.dirname(task_file_path)))
        metadata_path = os.path.join(dataset_root, "tasks", task_name, "metadata", "subset_metadata.csv")
        self.metadata = pd.read_csv(metadata_path)
        
        # Clean up metadata index - some sample IDs have spaces, so we need to standardize
        self.metadata[id_column] = self.metadata[id_column].str.strip()
        self.metadata.set_index(id_column, inplace=True)
        
        # Load label mapper
        mapping_path = os.path.join(dataset_root, "tasks", task_name, "metadata", "label_mapping.json")
        if os.path.exists(mapping_path):
            self.label_mapper = LabelMapper.load(mapping_path)
        else:
            # Create mapper on the fly if needed
            self.label_mapper = LabelMapper(label_column=label_column)
            self.label_mapper.fit(metadata_df=self.metadata.reset_index())
        
        # Filter out tasks that don't have enough valid samples
        self.valid_tasks = []
        for task in self.tasks:
            support_ids = [sid for sid in task["support"] if self._is_valid_sample(sid)]
            query_ids = [qid for qid in task["query"] if self._is_valid_sample(qid)]
            
            if support_ids and query_ids:  # Only keep tasks with at least some valid samples
                task["support"] = support_ids
                task["query"] = query_ids
                self.valid_tasks.append(task)
        
        print(f"Loaded {len(self.valid_tasks)} valid meta-learning tasks from {task_file_path} (out of {len(self.tasks)} total)")
        self.tasks = self.valid_tasks
        
    def _is_valid_sample(self, sample_id):
        """Check if a sample ID exists in the metadata"""
        return sample_id in self.metadata.index
        
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        """Get a task by index"""
        task = self.tasks[idx]
        
        # Get support and query sample IDs
        support_ids = task["support"]
        query_ids = task["query"]
        
        # If n_way/k_shot/q_query are specified, subsample accordingly
        if self.n_way is not None and self.k_shot is not None:
            # Group samples by label
            support_by_label = {}
            query_by_label = {}
            
            for sample_id in support_ids:
                try:
                    label = self.metadata.loc[sample_id, self.label_column]
                    if label not in support_by_label:
                        support_by_label[label] = []
                    support_by_label[label].append(sample_id)
                except KeyError:
                    # Skip samples that don't exist in metadata
                    continue
                
            for sample_id in query_ids:
                try:
                    label = self.metadata.loc[sample_id, self.label_column]
                    if label not in query_by_label:
                        query_by_label[label] = []
                    query_by_label[label].append(sample_id)
                except KeyError:
                    # Skip samples that don't exist in metadata
                    continue
            
            # Select n_way labels
            available_labels = list(set(support_by_label.keys()) & set(query_by_label.keys()))
            if len(available_labels) < self.n_way:
                selected_labels = available_labels
            else:
                selected_labels = random.sample(available_labels, self.n_way)
            
            # Select k_shot support samples and q_query query samples for each label
            new_support_ids = []
            new_query_ids = []
            
            for label in selected_labels:
                if len(support_by_label[label]) >= self.k_shot:
                    new_support_ids.extend(random.sample(support_by_label[label], self.k_shot))
                else:
                    new_support_ids.extend(support_by_label[label])
                
                if self.q_query is not None:
                    if len(query_by_label[label]) >= self.q_query:
                        new_query_ids.extend(random.sample(query_by_label[label], self.q_query))
                    else:
                        new_query_ids.extend(query_by_label[label])
                else:
                    new_query_ids.extend(query_by_label[label])
            
            support_ids = new_support_ids
            query_ids = new_query_ids
        
        # Load support samples
        support_x = []
        support_y = []
        
        for sample_id in support_ids:
            try:
                # Get file path from metadata
                file_path = self.metadata.loc[sample_id, self.data_column]
                label = self.metadata.loc[sample_id, self.label_column]
                
                # Convert to absolute path if needed
                if not os.path.isabs(file_path):
                    file_path = os.path.join(self.dataset_root, file_path)
                
                # Load data
                csi_data = self._load_file(file_path)
                
                # Transform if needed
                if self.transform:
                    csi_data = self.transform(csi_data)
                
                support_x.append(csi_data)
                
                # Map label to index
                label_idx = self.label_mapper.transform(label)
                support_y.append(label_idx)
            except Exception as e:
                print(f"Error loading support sample {sample_id}: {e}")
        
        # Load query samples
        query_x = []
        query_y = []
        
        for sample_id in query_ids:
            try:
                # Get file path from metadata
                file_path = self.metadata.loc[sample_id, self.data_column]
                label = self.metadata.loc[sample_id, self.label_column]
                
                # Convert to absolute path if needed
                if not os.path.isabs(file_path):
                    file_path = os.path.join(self.dataset_root, file_path)
                
                # Load data
                csi_data = self._load_file(file_path)
                
                # Transform if needed
                if self.transform:
                    csi_data = self.transform(csi_data)
                
                query_x.append(csi_data)
                
                # Map label to index
                label_idx = self.label_mapper.transform(label)
                query_y.append(label_idx)
            except Exception as e:
                print(f"Error loading query sample {sample_id}: {e}")
        
        # Convert to tensors
        if support_x:
            support_x = torch.stack(support_x)
            support_y = torch.tensor(support_y, dtype=torch.long)
        else:
            support_x = torch.empty((0, 1, 232, 500))  # Default shape, adjust as needed
            support_y = torch.empty(0, dtype=torch.long)
        
        if query_x:
            query_x = torch.stack(query_x)
            query_y = torch.tensor(query_y, dtype=torch.long)
        else:
            query_x = torch.empty((0, 1, 232, 500))  # Default shape, adjust as needed
            query_y = torch.empty(0, dtype=torch.long)
        
        return {
            'task_id': task['task_id'],
            'subject': task.get('subject', ''),
            'user': task.get('user', ''),
            'support': (support_x, support_y),
            'query': (query_x, query_y)
        }
    
    def _load_file(self, filepath):
        """Load a single data file"""
        # Handle different file formats
        if self.file_format == "npy":
            csi_data = np.load(filepath)
        elif self.file_format == "h5":
            with h5py.File(filepath, 'r') as f:
                # Try to use the specified data_key
                if self.data_key in f:
                    csi_data = np.array(f[self.data_key])
                else:
                    # Fallback to checking other common keys
                    if 'csi' in f:
                        csi_data = np.array(f['csi'])
                    elif 'CSI' in f:
                        csi_data = np.array(f['CSI'])
                    else:
                        # If all else fails, use the first dataset in the file
                        for key in f.keys():
                            if isinstance(f[key], h5py.Dataset):
                                csi_data = np.array(f[key])
                                break
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
        
        # Convert to tensor
        csi_data = torch.from_numpy(csi_data).float()
        
        # Reshape to (1, time_index, feature_size) if needed
        if len(csi_data.shape) == 3 and csi_data.shape[2] == 1:  # (time_index, feature_size, 1)
            # Permute to get (1, time_index, feature_size)
            csi_data = csi_data.permute(2, 0, 1)
        
        return csi_data

def load_meta_learning_tasks(
    dataset_root,
    task_name="MotionSourceRecognition",
    split_types=None,
    n_way=None,
    k_shot=None,
    q_query=None,
    batch_size=4,
    transform=None,
    target_transform=None,
    file_format="h5",
    data_column="file_path",
    label_column="label",
    data_key="CSI_amps",
    num_workers=4,
    id_column="id"
):
    """Load meta-learning tasks from JSON files"""
    if split_types is None:
        split_types = ['train', 'val', 'test']
    
    # Build path to meta_splits directory
    meta_splits_dir = os.path.join(dataset_root, "tasks", task_name, "meta_splits")
    
    # Check if directory exists
    if not os.path.exists(meta_splits_dir):
        raise ValueError(f"Meta-learning splits directory not found: {meta_splits_dir}")
    
    # Find available split files
    available_files = os.listdir(meta_splits_dir)
    
    # Initialize loaders dictionary
    loaders = {}
    
    # Mapping from split_type to file pattern
    split_patterns = {
        'train': 'train_tasks.json',
        'val': 'val_tasks.json',
        'test': 'test_tasks.json',
        'test_cross_env': 'test_cross_env_tasks.json',
        'test_cross_user': 'test_cross_user_tasks.json',
        'test_cross_device': 'test_cross_device_tasks.json',
        'adapt_1shot': 'adapt_1shot_tasks.json',
        'adapt_5shot': 'adapt_5shot_tasks.json',
        'cross_env_adapt_1shot': 'cross_env_adapt_1shot_tasks.json',
        'cross_env_adapt_5shot': 'cross_env_adapt_5shot_tasks.json',
        'cross_user_adapt_1shot': 'cross_user_adapt_1shot_tasks.json',
        'cross_user_adapt_5shot': 'cross_user_adapt_5shot_tasks.json',
        'cross_device_adapt_1shot': 'cross_device_adapt_1shot_tasks.json',
        'cross_device_adapt_5shot': 'cross_device_adapt_5shot_tasks.json'
    }
    
    # Load each requested split
    for split_type in split_types:
        # Find matching file
        if split_type in split_patterns:
            filename = split_patterns[split_type]
            if filename in available_files:
                file_path = os.path.join(meta_splits_dir, filename)
                
                # Create task sampler for this split
                try:
                    dataset = MetaTaskSampler(
                        dataset_root=dataset_root,
                        task_file_path=file_path,
                        n_way=n_way,
                        k_shot=k_shot,
                        q_query=q_query,
                        transform=transform,
                        target_transform=target_transform,
                        file_format=file_format,
                        data_column=data_column,
                        label_column=label_column,
                        data_key=data_key,
                        id_column=id_column
                    )
                    
                    # Only create a loader if we have at least one valid task
                    if len(dataset) > 0:
                        # Create data loader
                        loaders[split_type] = DataLoader(
                            dataset,
                            batch_size=batch_size,
                            shuffle=(split_type == 'train'),
                            num_workers=num_workers,
                            collate_fn=None  # No custom collate function needed
                        )
                        
                        print(f"Created data loader for {split_type} with {len(dataset)} tasks")
                    else:
                        print(f"Warning: No valid tasks in {filename}, skipping")
                except Exception as e:
                    print(f"Error creating dataset for {split_type}: {e}")
            else:
                print(f"Warning: Split file {filename} not found in {meta_splits_dir}")
        else:
            print(f"Warning: Unknown split type: {split_type}")
    
    return loaders

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
        file_format="h5",
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

if __name__ == "__main__":
    main() 