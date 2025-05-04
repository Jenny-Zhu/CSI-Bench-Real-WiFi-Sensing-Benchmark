import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Import our meta_model with all model types
from meta_model import BaseMetaModel

# Import from our standalone module for data loading
from meta_learning_data import load_meta_learning_tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Custom collate function (same as in training script)
def custom_task_collate(batch):
    """
    Custom collate function for meta-learning tasks.
    Each batch element is a dict with 'support' and 'query' tuples.
    """
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
            result[key] = torch.stack([item[key] for item in batch])
    
    return result

# Define the meta-testing function
def meta_test(model, data_loader, device, inner_lr=0.01, num_adaptation_steps=5):
    """
    Perform meta-testing to evaluate few-shot adaptation
    
    Args:
        model: The meta-learning model
        data_loader: Data loader with test tasks
        device: Device to run evaluation on
        inner_lr: Inner loop learning rate
        num_adaptation_steps: Number of adaptation steps
        
    Returns:
        Dictionary with test results
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_accuracies = []
    all_losses = []
    
    # For tracking adaptation progress
    adaptation_accuracies = [[] for _ in range(num_adaptation_steps)]
    
    # Process each batch
    for batch_idx, batch in enumerate(data_loader):
        support_x_list, support_y_list = batch['support']
        query_x_list, query_y_list = batch['query']
        
        # Move to device
        support_x_list = [x.to(device) for x in support_x_list]
        support_y_list = [y.to(device) for y in support_y_list]
        query_x_list = [x.to(device) for x in query_x_list]
        query_y_list = [y.to(device) for y in query_y_list]
        
        # Process each task in the batch
        batch_size = len(support_x_list)
        
        for task_idx in range(batch_size):
            support_x = support_x_list[task_idx]
            support_y = support_y_list[task_idx]
            query_x = query_x_list[task_idx]
            query_y = query_y_list[task_idx]
            
            # Determine number of unique classes
            all_labels = torch.cat([support_y, query_y])
            unique_labels = torch.unique(all_labels)
            num_unique_classes = len(unique_labels)
            
            # Create label mapping
            label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
            
            # Remap labels to ensure they're in range
            remapped_support_y = torch.tensor([label_map[y.item()] for y in support_y], 
                                             device=support_y.device, dtype=support_y.dtype)
            remapped_query_y = torch.tensor([label_map[y.item()] for y in query_y], 
                                           device=query_y.device, dtype=query_y.dtype)
            
            # Store the original weights
            with torch.no_grad():
                original_weights = {name: param.clone() for name, param in model.named_parameters()}
            
            # Track accuracy through adaptation
            step_accuracies = []
            
            # Inner loop adaptation
            for step in range(num_adaptation_steps):
                # Forward pass
                support_logits = model(support_x)
                
                # Ensure output size matches number of classes
                if support_logits.size(1) > num_unique_classes:
                    support_logits = support_logits[:, :num_unique_classes]
                
                # Compute loss
                inner_loss = criterion(support_logits, remapped_support_y)
                
                # Compute gradients
                model.zero_grad()
                inner_loss.backward()
                
                # Update parameters manually (SGD)
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.data.sub_(inner_lr * param.grad.data)
                
                # Evaluate on query set after this step
                with torch.no_grad():
                    query_logits = model(query_x)
                    
                    # Ensure output size matches number of classes
                    if query_logits.size(1) > num_unique_classes:
                        query_logits = query_logits[:, :num_unique_classes]
                    
                    # Calculate metrics
                    query_loss = criterion(query_logits, remapped_query_y).item()
                    pred_class = torch.argmax(query_logits, dim=1)
                    query_acc = (pred_class == remapped_query_y).float().mean().item()
                    
                    # Store accuracy for this adaptation step
                    adaptation_accuracies[step].append(query_acc)
                    
                    # Store final accuracy and loss
                    if step == num_adaptation_steps - 1:
                        all_accuracies.append(query_acc)
                        all_losses.append(query_loss)
            
            # Restore original weights
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(original_weights[name])
                    
            # Report progress
            if (task_idx + 1) % 10 == 0:
                logging.info(f"Processed {task_idx + 1}/{batch_size} tasks in batch {batch_idx + 1}")
    
    # Calculate average results
    avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0
    avg_loss = np.mean(all_losses) if all_losses else 0
    std_accuracy = np.std(all_accuracies) if all_accuracies else 0
    
    # Calculate adaptation curve
    adaptation_curve = [np.mean(accs) if accs else 0 for accs in adaptation_accuracies]
    
    # Results dictionary
    results = {
        'accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'loss': avg_loss,
        'adaptation_curve': adaptation_curve,
        'n_tasks': len(all_accuracies)
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test meta-learning models on WiFi benchmark')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                        help='Name of the task')
    parser.add_argument('--test_type', type=str, default='test',
                        choices=['test', 'cross_env', 'cross_user', 'cross_device',
                                 'adapt_1shot', 'adapt_5shot', 'cross_env_adapt_1shot',
                                 'cross_env_adapt_5shot', 'cross_user_adapt_1shot',
                                 'cross_user_adapt_5shot', 'cross_device_adapt_1shot',
                                 'cross_device_adapt_5shot'],
                        help='Type of test to perform')
    parser.add_argument('--n_way', type=int, default=3,
                        help='Number of classes per task')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of support examples per class')
    parser.add_argument('--q_query', type=int, default=5,
                        help='Number of query examples per class')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of tasks per batch')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner loop learning rate')
    parser.add_argument('--adaptation_steps', type=int, default=5,
                        help='Number of adaptation steps')
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                        help='Key in h5 file for CSI data')
    parser.add_argument('--output_dir', type=str, default='results/meta_testing',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get('args', {})
    
    # Create model using checkpoint configuration
    model_type = checkpoint_args.get('model_type', 'vit')
    win_len = checkpoint_args.get('win_len', 232)
    feature_size = checkpoint_args.get('feature_size', 500)
    in_channels = checkpoint_args.get('in_channels', 1)
    emb_dim = checkpoint_args.get('emb_dim', 128)
    num_classes = checkpoint_args.get('n_way', args.n_way)
    dropout = checkpoint_args.get('dropout', 0.1)
    
    logging.info(f"Creating {model_type.upper()} model...")
    model = BaseMetaModel(
        model_type=model_type,
        win_len=win_len,
        feature_size=feature_size,
        in_channels=in_channels,
        emb_dim=emb_dim,
        num_classes=num_classes,
        dropout=dropout,
        inner_lr=args.inner_lr
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Log model info
    logging.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Determine test split based on test_type
    split_map = {
        'test': 'test_tasks.json',
        'cross_env': 'test_cross_env_tasks.json',
        'cross_user': 'test_cross_user_tasks.json',
        'cross_device': 'test_cross_device_tasks.json',
        'adapt_1shot': 'adapt_1shot_tasks.json',
        'adapt_5shot': 'adapt_5shot_tasks.json',
        'cross_env_adapt_1shot': 'cross_env_adapt_1shot_tasks.json',
        'cross_env_adapt_5shot': 'cross_env_adapt_5shot_tasks.json',
        'cross_user_adapt_1shot': 'cross_user_adapt_1shot_tasks.json',
        'cross_user_adapt_5shot': 'cross_user_adapt_5shot_tasks.json',
        'cross_device_adapt_1shot': 'cross_device_adapt_1shot_tasks.json',
        'cross_device_adapt_5shot': 'cross_device_adapt_5shot_tasks.json'
    }
    
    split_file = split_map.get(args.test_type)
    if not split_file:
        logging.error(f"Invalid test_type: {args.test_type}")
        return
    
    # Set correct shot value for adaptation tests
    if 'adapt_1shot' in args.test_type:
        args.k_shot = 1
    elif 'adapt_5shot' in args.test_type:
        args.k_shot = 5
    
    # Load data
    logging.info(f"Loading {args.test_type} tasks from {args.data_dir}...")
    test_loader = load_meta_learning_tasks(
        dataset_root=args.data_dir,
        task_name=args.task_name,
        split_types=[args.test_type],
        task_file_paths=[os.path.join(args.data_dir, 'tasks', args.task_name, 'meta_splits', split_file)],
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        batch_size=args.batch_size,
        file_format="h5",
        data_key=args.data_key,
        num_workers=4,
        collate_fn=custom_task_collate
    )[args.test_type]
    
    # Run evaluation
    logging.info(f"Running meta-testing with {args.adaptation_steps} adaptation steps...")
    results = meta_test(
        model=model,
        data_loader=test_loader,
        device=device,
        inner_lr=args.inner_lr,
        num_adaptation_steps=args.adaptation_steps
    )
    
    # Print results
    logging.info(f"Results for {args.test_type}:")
    logging.info(f"Average accuracy: {results['accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    logging.info(f"Average loss: {results['loss']:.4f}")
    logging.info(f"Number of tasks: {results['n_tasks']}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{model_type}_{args.task_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_file = output_dir / f"{args.test_type}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    # Plot adaptation curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.adaptation_steps + 1), results['adaptation_curve'])
    plt.xlabel('Adaptation Steps')
    plt.ylabel('Query Set Accuracy')
    plt.title(f'Meta-Test Adaptation Curve for {args.test_type}\n'
              f'(Avg Accuracy: {results["accuracy"]:.4f}±{results["std_accuracy"]:.4f})')
    plt.grid(True)
    plt.savefig(output_dir / f"{args.test_type}_adaptation_curve.png")
    plt.close()
    
    logging.info(f"Results saved to {output_dir}")

# Helper class for serializing numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main() 