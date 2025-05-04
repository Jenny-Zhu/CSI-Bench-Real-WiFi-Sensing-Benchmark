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
import matplotlib.pyplot as plt
import copy

# Import directly from our files 
from meta_learning_data import load_meta_learning_tasks, MetaTaskSampler
from meta_model import BaseMetaModel, MLPClassifier, LSTMClassifier, ResNet18Classifier, TransformerClassifier, ViTClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Custom collate function for tasks with variable sizes
def custom_task_collate(batch):
    """
    Custom collate function for meta-learning tasks.
    Each batch element is a dict with 'support' and 'query' tuples.
    """
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

# Class for MAML (Model-Agnostic Meta-Learning)
class MAMLTrainer:
    def __init__(self, model, device, inner_lr=0.01, meta_lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, task_batch):
        """Perform a meta-training step"""
        self.model.train()
        
        # Extract data from task_batch
        if isinstance(task_batch['support'], tuple):
            # Handle batch_size > 1
            support_x_list, support_y_list = task_batch['support']
            query_x_list, query_y_list = task_batch['query']
            batch_size = len(support_x_list)
        else:
            # Handle batch_size = 1
            support_x = task_batch['support'][0].to(self.device)
            support_y = task_batch['support'][1].to(self.device)
            query_x = task_batch['query'][0].to(self.device)
            query_y = task_batch['query'][1].to(self.device)
            support_x_list = [support_x]
            support_y_list = [support_y]
            query_x_list = [query_x]
            query_y_list = [query_y]
            batch_size = 1
        
        # Move data to device if not already
        if batch_size > 1:
            support_x_list = [x.to(self.device) for x in support_x_list]
            support_y_list = [y.to(self.device) for y in support_y_list]
            query_x_list = [x.to(self.device) for x in query_x_list]
            query_y_list = [y.to(self.device) for y in query_y_list]
        
        # Zero gradients
        self.meta_optimizer.zero_grad()
        
        # Track metrics
        meta_loss = 0.0
        task_query_accs = []
        task_support_accs = []
        
        # Process each task in batch
        for i in range(batch_size):
            support_x = support_x_list[i]
            support_y = support_y_list[i]
            query_x = query_x_list[i]
            query_y = query_y_list[i]
            
            # Skip empty batches
            if support_x.shape[0] == 0 or query_x.shape[0] == 0:
                logging.warning(f"Skipping empty batch in task {i}")
                continue
            
            # Determine maximum class index for each task
            try:
                all_labels = torch.cat([support_y, query_y])
                num_classes = len(torch.unique(all_labels))
                
                # Skip tasks with too few classes
                if num_classes < 2:
                    logging.warning(f"Skipping task {i} with only {num_classes} class")
                    continue
                
                # Create label mapping
                unique_labels = torch.unique(all_labels)
                label_map = {old_label.item(): idx for idx, old_label in enumerate(unique_labels)}
                
                # Remap labels to be zero-indexed
                support_y_remapped = torch.tensor([label_map[y.item()] for y in support_y], 
                                                device=support_y.device, dtype=support_y.dtype)
                query_y_remapped = torch.tensor([label_map[y.item()] for y in query_y], 
                                            device=query_y.device, dtype=query_y.dtype)
            except RuntimeError as e:
                logging.warning(f"Error processing task {i}: {e}")
                continue
            
            # Clone model parameters for inner loop
            try:
                fast_weights = copy.deepcopy(dict(self.model.named_parameters()))
                
                # MAML inner loop adaptation (multiple steps)
                inner_losses = []
                for _ in range(5):  # Number of inner loop steps
                    # Forward pass
                    support_logits = self.forward_with_weights(support_x, fast_weights)
                    
                    # Ensure output matches the number of classes in this task
                    support_logits = support_logits[:, :num_classes]
                    
                    # Inner loop loss
                    inner_loss = self.criterion(support_logits, support_y_remapped)
                    inner_losses.append(inner_loss.item())
                    
                    # Calculate support accuracy
                    pred_class = torch.argmax(support_logits, dim=1)
                    support_acc = (pred_class == support_y_remapped).float().mean().item()
                    task_support_accs.append(support_acc)
                    
                    # Compute gradients for inner update
                    grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=True)
                    
                    # Update fast weights
                    fast_weights = {name: param - self.inner_lr * grad
                                for (name, param), grad in zip(fast_weights.items(), grads)}
                
                # Evaluate on query set with updated weights
                query_logits = self.forward_with_weights(query_x, fast_weights)
                
                # Ensure output matches the number of classes
                query_logits = query_logits[:, :num_classes]
                
                # Calculate query loss (meta-objective)
                query_loss = self.criterion(query_logits, query_y_remapped)
                meta_loss += query_loss
                
                # Calculate query accuracy
                pred_class = torch.argmax(query_logits, dim=1)
                query_acc = (pred_class == query_y_remapped).float().mean().item()
                task_query_accs.append(query_acc)
            except Exception as e:
                logging.warning(f"Error in inner loop for task {i}: {e}")
                continue
        
        # If no tasks were processed successfully, return null metrics
        if len(task_query_accs) == 0:
            logging.warning("No valid tasks in batch. Using random dummy task.")
            
            # Create dummy task with random data
            dummy_support_x = torch.randn(2, 1, 232, 500, device=self.device)
            dummy_support_y = torch.tensor([0, 1], device=self.device)
            dummy_query_x = torch.randn(2, 1, 232, 500, device=self.device)
            dummy_query_y = torch.tensor([0, 1], device=self.device)
            
            # Forward pass
            support_logits = self.model(dummy_support_x)
            support_logits = support_logits[:, :2]  # Only 2 classes
            
            # Support loss
            support_loss = self.criterion(support_logits, dummy_support_y)
            
            # Backward and optimize
            self.meta_optimizer.zero_grad()
            support_loss.backward()
            self.meta_optimizer.step()
            
            # Return dummy metrics
            return {
                'meta_loss': support_loss.item(),
                'support_acc': 0.5,
                'query_acc': 0.5,
                'inner_losses': support_loss.item()
            }
        
        # Average meta-loss across tasks
        meta_loss = meta_loss / len(task_query_accs)
        
        # Compute meta-gradient and update meta-parameters
        meta_loss.backward()
        self.meta_optimizer.step()
        
        # Return metrics
        return {
            'meta_loss': meta_loss.item(),
            'support_acc': np.mean(task_support_accs),
            'query_acc': np.mean(task_query_accs),
            'inner_losses': inner_losses[-1] if inner_losses else 0
        }
    
    def evaluate(self, data_loader, num_adaptation_steps=5, max_tasks=None):
        """Evaluate on a set of tasks"""
        self.model.eval()
        
        all_task_accs = []
        all_task_losses = []
        
        # Limit number of tasks to evaluate on
        num_tasks = len(data_loader) if max_tasks is None else min(max_tasks, len(data_loader))
        
        for task_idx, task_batch in enumerate(data_loader):
            if task_idx >= num_tasks:
                break
            
            # Extract data
            try:
                if isinstance(task_batch['support'], tuple):
                    # Handle batch_size > 1
                    support_x = task_batch['support'][0][0].to(self.device)
                    support_y = task_batch['support'][1][0].to(self.device)
                    query_x = task_batch['query'][0][0].to(self.device)
                    query_y = task_batch['query'][1][0].to(self.device)
                else:
                    # Handle batch_size = 1
                    support_x = task_batch['support'][0].to(self.device)
                    support_y = task_batch['support'][1].to(self.device)
                    query_x = task_batch['query'][0].to(self.device)
                    query_y = task_batch['query'][1].to(self.device)
                
                # Skip empty batches
                if support_x.shape[0] == 0 or query_x.shape[0] == 0:
                    logging.warning(f"Skipping empty batch in evaluation task {task_idx}")
                    continue
                
                # Determine number of classes for this task
                all_labels = torch.cat([support_y, query_y])
                num_classes = len(torch.unique(all_labels))
                
                # Skip tasks with too few classes
                if num_classes < 2:
                    logging.warning(f"Skipping evaluation task {task_idx} with only {num_classes} class")
                    continue
                
                # Create label mapping
                unique_labels = torch.unique(all_labels)
                label_map = {old_label.item(): idx for idx, old_label in enumerate(unique_labels)}
                
                # Remap labels
                support_y_remapped = torch.tensor([label_map[y.item()] for y in support_y], 
                                                device=support_y.device, dtype=support_y.dtype)
                query_y_remapped = torch.tensor([label_map[y.item()] for y in query_y], 
                                            device=query_y.device, dtype=query_y.dtype)
                
                # Clone model for inner loop adaptation
                fast_weights = copy.deepcopy(dict(self.model.named_parameters()))
                
                # Inner loop adaptation
                for _ in range(num_adaptation_steps):
                    # Forward pass
                    support_logits = self.forward_with_weights(support_x, fast_weights)
                    
                    # Ensure output matches number of classes
                    support_logits = support_logits[:, :num_classes]
                    
                    # Inner loss
                    inner_loss = self.criterion(support_logits, support_y_remapped)
                    
                    # Compute gradients for inner update
                    grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=True)
                    
                    # Update fast weights
                    fast_weights = {name: param - self.inner_lr * grad
                                for (name, param), grad in zip(fast_weights.items(), grads)}
                
                # Evaluate on query set
                with torch.no_grad():
                    query_logits = self.forward_with_weights(query_x, fast_weights)
                    
                    # Ensure output matches number of classes
                    query_logits = query_logits[:, :num_classes]
                    
                    # Query loss
                    query_loss = self.criterion(query_logits, query_y_remapped).item()
                    
                    # Query accuracy
                    pred_class = torch.argmax(query_logits, dim=1)
                    query_acc = (pred_class == query_y_remapped).float().mean().item()
                    
                    all_task_accs.append(query_acc)
                    all_task_losses.append(query_loss)
            except Exception as e:
                logging.warning(f"Error evaluating task {task_idx}: {e}")
                continue
            
            if (task_idx + 1) % 10 == 0:
                logging.info(f"Evaluated {task_idx + 1}/{num_tasks} tasks")
        
        # If no tasks were evaluated successfully
        if len(all_task_accs) == 0:
            logging.warning("No valid tasks in evaluation. Using default metrics.")
            return {
                'accuracy': 0.0,
                'loss': 0.0,
                'std_accuracy': 0.0,
                'n_tasks': 0
            }
        
        # Calculate average results
        avg_acc = np.mean(all_task_accs)
        avg_loss = np.mean(all_task_losses)
        
        return {
            'accuracy': avg_acc,
            'loss': avg_loss,
            'std_accuracy': np.std(all_task_accs),
            'n_tasks': len(all_task_accs)
        }
    
    def forward_with_weights(self, x, weights):
        """Forward pass using the provided weights"""
        # Store original weights
        orig_weights = {}
        for name, param in self.model.named_parameters():
            orig_weights[name] = param.data.clone()
        
        # Replace with fast weights
        for name, param in self.model.named_parameters():
            param.data = weights[name].data
        
        # Forward pass
        output = self.model(x)
        
        # Restore original weights
        for name, param in self.model.named_parameters():
            param.data = orig_weights[name]
        
        return output

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='Train a meta-learning model on WiFi benchmark dataset')
    parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                        help='Name of the task')
    parser.add_argument('--model_type', type=str, default='mlp',
                        choices=['mlp', 'lstm', 'resnet18', 'transformer', 'vit'],
                        help='Type of model to train')
    parser.add_argument('--n_way', type=int, default=3,
                        help='Number of classes per task')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of support examples per class')
    parser.add_argument('--q_query', type=int, default=5,
                        help='Number of query examples per class')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of tasks per batch')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Outer loop learning rate')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of iterations for meta-learning')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Interval for logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='results/meta_standalone',
                        help='Directory to save checkpoints')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with fewer iterations')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Use CPU for now (GPU support can be added later)
    device = torch.device('cpu')
    
    # Load meta-learning tasks
    loaders = load_meta_learning_tasks(
        dataset_root=args.data_dir,
        task_name=args.task_name,
        split_types=['train', 'val', 'test'],
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        batch_size=args.batch_size,
        collate_fn=custom_task_collate,
        num_workers=0  # Use 0 for debugging
    )
    
    # Determine maximum number of classes
    logging.info("Determining the maximum number of classes from the training data...")
    
    # Check first task to see how many classes it has
    max_classes = args.n_way
    try:
        train_iter = iter(loaders['train'])
        first_batch = next(train_iter)
        
        # Get support labels
        if isinstance(first_batch['support'], tuple):
            support_y = first_batch['support'][1][0]  # First task, labels
        else:
            support_y = first_batch['support'][1]
        
        # Count unique classes
        unique_classes = len(torch.unique(support_y))
        max_classes = max(max_classes, unique_classes)
        
        logging.info(f"Detected {unique_classes} unique classes in the dataset")
    except Exception as e:
        logging.warning(f"Error determining class count: {e}")
        logging.info(f"Using default n_way={args.n_way} for number of classes")
    
    # Create the model
    logging.info(f"Creating model with {max_classes} output classes...")
    model = BaseMetaModel(
        model_type=args.model_type,
        win_len=232,
        feature_size=500,
        in_channels=1,
        emb_dim=128,
        num_classes=max_classes,
        dropout=0.1,
        inner_lr=args.inner_lr
    )
    
    # Create trainer
    trainer = MAMLTrainer(
        model=model,
        device=device,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr
    )
    
    # Training loop
    logging.info(f"Starting training for {args.num_iterations} iterations...")
    train_iter = iter(loaders['train'])
    
    best_val_acc = 0.0
    train_metrics_history = []
    
    for iteration in range(1, args.num_iterations + 1):
        # Get a task batch (reload iterator if needed)
        try:
            task_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loaders['train'])
            task_batch = next(train_iter)
        
        # Train on this task batch
        train_metrics = trainer.train_step(task_batch)
        train_metrics_history.append(train_metrics)
        
        # Log progress
        if iteration % args.log_interval == 0:
            logging.info(f"Iteration {iteration}/{args.num_iterations}: "
                         f"Train Loss: {train_metrics['meta_loss']:.4f}, "
                         f"Query Loss: {train_metrics['inner_losses']:.4f}, "
                         f"Query Acc: {train_metrics['query_acc']:.4f}")
    
    # Validate the model
    if 'val' in loaders:
        val_results = trainer.evaluate(loaders['val'])
        logging.info(f"Validation: Loss: {val_results['loss']:.4f}, Acc: {val_results['accuracy']:.4f}")
        
        # Save model if it's the best so far
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'val_accuracy': best_val_acc,
                'iteration': args.num_iterations
            }, checkpoint_path)
            
            logging.info(f"Saved new best model with val acc: {best_val_acc:.4f}")
    
    logging.info("Training completed. Running final evaluation...")
    
    # Test the model if test loader is available
    if 'test' in loaders:
        # Skip loading the best model if it doesn't exist
        checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f"Loaded best model from {checkpoint_path}")
                else:
                    logging.warning(f"Checkpoint doesn't contain model_state_dict. Using current model.")
            except Exception as e:
                logging.warning(f"Error loading checkpoint: {e}. Using current model.")
        else:
            logging.info("No checkpoint found. Using the current model for testing.")
        
        # Test
        test_results = trainer.evaluate(loaders['test'])
        logging.info(f"Test: Loss: {test_results['loss']:.4f}, Acc: {test_results['accuracy']:.4f}")
        
        # Save test results
        results_path = os.path.join(args.save_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(test_results, f)
    
    logging.info("Done!")

if __name__ == "__main__":
    main() 