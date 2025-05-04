import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import logging
from pathlib import Path
import datetime

# Import from our standalone module
from meta_learning_data import load_meta_learning_tasks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Custom collate function for tasks with varying sizes
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

# Define a simple meta-learning model based on MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size=232*500, hidden_size=256, output_size=3):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Reshape input if needed
        if isinstance(x, list):
            # Process a list of tensors (from custom collate)
            results = []
            for item in x:
                if len(item.shape) > 3:  # [batch_within_task, channels, height, width]
                    batch_size = item.shape[0]
                    item = item.view(batch_size, -1)  # Flatten each tensor
                else:
                    item = self.flatten(item)
                
                item = self.fc1(item)
                item = self.relu(item)
                item = self.fc2(item)
                results.append(item)
            return results
        else:
            # Regular processing for a single tensor
            if len(x.shape) > 3:  # [batch, k_shot, channels, height, width]
                batch_size, k_shot = x.shape[0], x.shape[1]
                x = x.view(batch_size * k_shot, *x.shape[2:])
            
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

# Simple meta-learning trainer
class SimpleMetaTrainer:
    def __init__(self, model, device, lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, task_batch):
        self.model.train()
        
        # Get support and query sets (now as lists of tensors)
        support_x_list, support_y_list = task_batch['support']
        query_x_list, query_y_list = task_batch['query']
        
        # Move to device
        support_x_list = [x.to(self.device) for x in support_x_list]
        support_y_list = [y.to(self.device) for y in support_y_list]
        query_x_list = [x.to(self.device) for x in query_x_list]
        query_y_list = [y.to(self.device) for y in query_y_list]
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        support_pred_list = self.model(support_x_list)
        
        # Calculate loss (average across all tasks in batch)
        total_loss = 0
        batch_size = len(support_x_list)
        
        for i in range(batch_size):
            pred = support_pred_list[i]
            target = support_y_list[i].view(-1)
            
            # Ensure that the target labels are within range (0 to num_classes-1)
            num_classes = pred.size(1)
            if target.max() >= num_classes:
                # Remap labels to be within range
                unique_labels = torch.unique(target)
                label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
                new_target = torch.tensor([label_map[label.item()] for label in target], 
                                          device=target.device, dtype=target.dtype)
                target = new_target
            
            loss = self.criterion(pred, target)
            total_loss += loss
        
        avg_loss = total_loss / batch_size
        
        # Backward pass
        avg_loss.backward()
        
        # Update parameters
        self.optimizer.step()
        
        # Evaluate on query set
        total_query_loss = 0
        total_query_acc = 0
        
        with torch.no_grad():
            query_pred_list = self.model(query_x_list)
            
            for i in range(batch_size):
                pred = query_pred_list[i]
                target = query_y_list[i].view(-1)
                
                # Ensure target labels are within range (using the same remapping)
                num_classes = pred.size(1)
                if target.max() >= num_classes:
                    # Remap labels to be within range (same mapping as above)
                    unique_labels = torch.unique(target)
                    label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
                    new_target = torch.tensor([label_map[label.item()] for label in target], 
                                              device=target.device, dtype=target.dtype)
                    target = new_target
                
                # Calculate loss
                loss = self.criterion(pred, target)
                total_query_loss += loss.item()
                
                # Calculate accuracy
                pred_class = torch.argmax(pred, dim=1)
                accuracy = (pred_class == target).float().mean().item()
                total_query_acc += accuracy
        
        avg_query_loss = total_query_loss / batch_size
        avg_query_acc = total_query_acc / batch_size
        
        return avg_loss.item(), avg_query_loss, avg_query_acc
    
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, task_batch in enumerate(loader):
                # Get query set
                query_x_list, query_y_list = task_batch['query']
                
                # Move to device
                query_x_list = [x.to(self.device) for x in query_x_list]
                query_y_list = [y.to(self.device) for y in query_y_list]
                
                # Forward pass
                query_pred_list = self.model(query_x_list)
                
                # Calculate batch metrics
                batch_size = len(query_x_list)
                batch_loss = 0
                batch_acc = 0
                
                for i in range(batch_size):
                    pred = query_pred_list[i]
                    target = query_y_list[i].view(-1)
                    
                    # Ensure target labels are within range
                    num_classes = pred.size(1)
                    if target.max() >= num_classes:
                        # Remap labels to be within range
                        unique_labels = torch.unique(target)
                        label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
                        new_target = torch.tensor([label_map[label.item()] for label in target], 
                                                  device=target.device, dtype=target.dtype)
                        target = new_target
                    
                    # Calculate loss
                    loss = self.criterion(pred, target)
                    batch_loss += loss.item()
                    
                    # Calculate accuracy
                    pred_class = torch.argmax(pred, dim=1)
                    accuracy = (pred_class == target).float().mean().item()
                    batch_acc += accuracy
                
                total_loss += batch_loss / batch_size
                total_accuracy += batch_acc / batch_size
                num_batches += 1
        
        if num_batches == 0:
            return 0, 0
        
        return total_loss / num_batches, total_accuracy / num_batches

# Function to get maximum number of unique classes in a task set
def get_max_classes(data_loader):
    max_classes = 0
    
    for batch in data_loader:
        _, support_y = batch['support']
        _, query_y = batch['query']
        
        if isinstance(support_y, list):
            # For collated batches
            for y in support_y:
                num_unique = len(torch.unique(y))
                max_classes = max(max_classes, num_unique)
            
            for y in query_y:
                num_unique = len(torch.unique(y))
                max_classes = max(max_classes, num_unique)
        else:
            # For normal batches
            num_support_unique = len(torch.unique(support_y))
            num_query_unique = len(torch.unique(query_y))
            max_batch_classes = max(num_support_unique, num_query_unique)
            max_classes = max(max_classes, max_batch_classes)
    
    return max_classes

def main():
    parser = argparse.ArgumentParser(description='Train meta-learning model on WiFi sensing data')
    parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                        help='Name of the task')
    parser.add_argument('--n_way', type=int, default=3,
                        help='Number of classes per task')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of support examples per class')
    parser.add_argument('--q_query', type=int, default=5,
                        help='Number of query examples per class')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of tasks per batch')
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='Number of training iterations')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Hidden layer size for MLP')
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                        help='Key in h5 file for CSI data')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Iteration interval for logging')
    parser.add_argument('--save_dir', type=str, default='results/meta_standalone',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(42)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_dir / f"meta_training_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    # Load meta-learning tasks
    logging.info(f"Loading meta-learning tasks from {args.data_dir}...")
    loaders = load_meta_learning_tasks(
        dataset_root=args.data_dir,
        task_name=args.task_name,
        split_types=['train', 'val', 'test'],
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        batch_size=1,  # Use batch size 1 for initial inspection to determine num_classes
        file_format="h5",
        data_key=args.data_key,
        num_workers=0,  # Avoid worker issues for now
        collate_fn=custom_task_collate  # Use our custom collate function
    )
    
    # Determine the maximum number of classes from the data
    num_classes = 0
    train_loader = loaders.get('train')
    if train_loader:
        logging.info("Determining the maximum number of classes from the training data...")
        try:
            # Load a single batch
            sample_batch = next(iter(train_loader))
            # Get all unique labels from support and query sets
            _, support_y_list = sample_batch['support']
            _, query_y_list = sample_batch['query']
            
            # Find the maximum class index in all tensors
            max_class = 0
            for y in support_y_list + query_y_list:
                if y.numel() > 0:  # Check if tensor is not empty
                    max_class = max(max_class, y.max().item())
            
            # num_classes should be max_class + 1 (since we start counting from 0)
            num_classes = max_class + 1
            logging.info(f"Detected {num_classes} unique classes in the dataset")
        except:
            logging.warning("Could not determine number of classes automatically, using n_way instead")
            num_classes = args.n_way
    
    if num_classes == 0:
        logging.warning(f"No classes detected, using n_way={args.n_way} as output size")
        num_classes = args.n_way
    
    # Create new loaders with actual batch size
    if args.batch_size > 1:
        logging.info(f"Reloading data loaders with batch size {args.batch_size}...")
        loaders = load_meta_learning_tasks(
            dataset_root=args.data_dir,
            task_name=args.task_name,
            split_types=['train', 'val', 'test'],
            n_way=args.n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            batch_size=args.batch_size,
            file_format="h5",
            data_key=args.data_key,
            num_workers=0,  # Avoid worker issues for now
            collate_fn=custom_task_collate  # Use our custom collate function
        )
    
    # Get training loader
    train_loader = loaders.get('train')
    if train_loader is None:
        logging.error("No training loader found. Check your data directory and task name.")
        return
    
    # Create model with the correct output size
    logging.info(f"Creating model with {num_classes} output classes...")
    model = SimpleMLP(
        input_size=232*500,  # Adjust based on your data
        hidden_size=args.hidden_size,
        output_size=num_classes
    )
    
    # Create trainer
    trainer = SimpleMetaTrainer(
        model=model,
        device=args.device,
        lr=args.lr
    )
    
    # Training loop
    logging.info(f"Starting training for {args.num_iterations} iterations...")
    
    # Get validation loader
    val_loader = loaders.get('val')
    
    # Train
    best_val_acc = 0
    train_iter = iter(train_loader)
    
    for i in range(args.num_iterations):
        # Get a batch of tasks
        try:
            task_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            task_batch = next(train_iter)
        
        # Train on the batch
        train_loss, query_loss, query_acc = trainer.train_step(task_batch)
        
        # Log training progress
        if (i + 1) % args.log_interval == 0:
            logging.info(f"Iteration {i+1}/{args.num_iterations}: "
                         f"Train Loss: {train_loss:.4f}, "
                         f"Query Loss: {query_loss:.4f}, "
                         f"Query Acc: {query_acc:.4f}")
        
        # Validate
        if val_loader and (i + 1) % (args.log_interval * 5) == 0:
            val_loss, val_acc = trainer.evaluate(val_loader)
            logging.info(f"Validation: Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Save model if better
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), save_dir / "best_model.pth")
                logging.info(f"Saved new best model with val acc: {val_acc:.4f}")
    
    # Final evaluation
    logging.info("Training completed. Running final evaluation...")
    
    # Load best model
    best_model_path = save_dir / "best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        logging.info(f"Loaded best model from {best_model_path}")
    
    # Evaluate on test set
    test_loader = loaders.get('test')
    if test_loader:
        test_loss, test_acc = trainer.evaluate(test_loader)
        logging.info(f"Test: Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    
    # Evaluate on cross-environment test set
    cross_env_loader = loaders.get('test_cross_env')
    if cross_env_loader:
        cross_env_loss, cross_env_acc = trainer.evaluate(cross_env_loader)
        logging.info(f"Cross-Environment Test: Loss: {cross_env_loss:.4f}, Acc: {cross_env_acc:.4f}")
    
    # Evaluate on cross-user test set
    cross_user_loader = loaders.get('test_cross_user')
    if cross_user_loader:
        cross_user_loss, cross_user_acc = trainer.evaluate(cross_user_loader)
        logging.info(f"Cross-User Test: Loss: {cross_user_loss:.4f}, Acc: {cross_user_acc:.4f}")
    
    logging.info("Done!")

if __name__ == "__main__":
    main() 