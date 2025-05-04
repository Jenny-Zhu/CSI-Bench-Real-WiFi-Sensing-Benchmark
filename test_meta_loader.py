import os
import argparse
import torch
from load.meta_learning.meta_data_loader import load_meta_learning_tasks

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
    
    # Load meta-learning tasks directly using the meta_data_loader module
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