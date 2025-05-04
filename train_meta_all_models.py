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

# Import our meta_model with all model types
from meta_model import BaseMetaModel

# Import from our standalone module for data loading
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

# Meta-learning trainer with MAML algorithm
class MAMLTrainer:
    def __init__(self, model, device, inner_lr=0.01, meta_lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, task_batch):
        self.model.train()
        
        # Get support and query sets (lists of tensors in our custom collate function)
        support_x_list, support_y_list = task_batch['support']
        query_x_list, query_y_list = task_batch['query']
        
        # Move to device
        support_x_list = [x.to(self.device) for x in support_x_list]
        support_y_list = [y.to(self.device) for y in support_y_list]
        query_x_list = [x.to(self.device) for x in query_x_list]
        query_y_list = [y.to(self.device) for y in query_y_list]
        
        # Zero gradients
        self.meta_optimizer.zero_grad()
        
        # Accumulate meta-gradient across tasks
        total_meta_loss = 0
        batch_size = len(support_x_list)
        
        # Support set accuracy tracking
        support_accuracies = []
        query_accuracies = []
        
        for i in range(batch_size):
            support_x = support_x_list[i]
            support_y = support_y_list[i]
            query_x = query_x_list[i]
            query_y = query_y_list[i]
            
            # Ensure labels are in the correct range
            num_unique_classes = len(torch.unique(torch.cat([support_y, query_y])))
            
            # MAML inner loop adaptation (create fast weights)
            fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
            
            # Multiple inner loop steps
            for _ in range(5):  # Number of inner loop steps
                # Forward pass with current fast weights
                support_logits = self.forward_with_weights(support_x, fast_weights)
                
                # Ensure model output size matches the number of classes
                if support_logits.size(1) != num_unique_classes:
                    # Remap labels and trim output logits if needed
                    unique_labels = torch.unique(support_y)
                    label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
                    new_support_y = torch.tensor([label_map[y.item()] for y in support_y], 
                                               device=support_y.device, dtype=support_y.dtype)
                    support_y_remapped = new_support_y
                    
                    # Calculate support accuracy with remapped labels
                    pred_class = torch.argmax(support_logits[:, :num_unique_classes], dim=1)
                    support_acc = (pred_class == support_y_remapped).float().mean().item()
                    support_accuracies.append(support_acc)
                    
                    # Inner loop loss
                    inner_loss = self.criterion(support_logits[:, :num_unique_classes], support_y_remapped)
                else:
                    # Calculate support accuracy
                    pred_class = torch.argmax(support_logits, dim=1)
                    support_acc = (pred_class == support_y).float().mean().item()
                    support_accuracies.append(support_acc)
                    
                    # Inner loop loss
                    inner_loss = self.criterion(support_logits, support_y)
                
                # Calculate gradients w.r.t. fast weights
                grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=True)
                
                # Update fast weights
                fast_weights = {name: param - self.inner_lr * grad
                               for (name, param), grad in zip(fast_weights.items(), grads)}
            
            # MAML meta-update (outer loop)
            # Forward pass on query set using adapted weights
            query_logits = self.forward_with_weights(query_x, fast_weights)
            
            # Handle different number of classes in query set
            if query_logits.size(1) != num_unique_classes:
                # Remap query labels to match logits
                unique_query_labels = torch.unique(query_y)
                query_label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_query_labels)}
                new_query_y = torch.tensor([query_label_map[y.item()] for y in query_y], 
                                          device=query_y.device, dtype=query_y.dtype)
                query_y_remapped = new_query_y
                
                # Calculate query accuracy
                pred_class = torch.argmax(query_logits[:, :num_unique_classes], dim=1)
                query_acc = (pred_class == query_y_remapped).float().mean().item()
                query_accuracies.append(query_acc)
                
                # Meta loss
                meta_loss = self.criterion(query_logits[:, :num_unique_classes], query_y_remapped)
            else:
                # Calculate query accuracy
                pred_class = torch.argmax(query_logits, dim=1)
                query_acc = (pred_class == query_y).float().mean().item()
                query_accuracies.append(query_acc)
                
                # Meta loss
                meta_loss = self.criterion(query_logits, query_y)
            
            # Accumulate meta loss
            total_meta_loss += meta_loss
        
        # Average meta loss across tasks
        avg_meta_loss = total_meta_loss / batch_size
        
        # Backward pass on meta loss
        avg_meta_loss.backward()
        
        # Update model parameters
        self.meta_optimizer.step()
        
        # Average accuracies
        avg_support_acc = sum(support_accuracies) / len(support_accuracies) if support_accuracies else 0
        avg_query_acc = sum(query_accuracies) / len(query_accuracies) if query_accuracies else 0
        
        return {
            'meta_loss': avg_meta_loss.item(),
            'support_acc': avg_support_acc,
            'query_acc': avg_query_acc
        }
    
    def forward_with_weights(self, x, weights):
        """Custom forward pass using provided weights"""
        model_copy = self.create_model_copy_with_weights(weights)
        return model_copy(x)
    
    def create_model_copy_with_weights(self, weights):
        """Create a model copy with the specified weights"""
        model_copy = type(self.model)(**self.model.__dict__).to(self.device)
        
        # Copy weights
        with torch.no_grad():
            for name, param in model_copy.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
        
        return model_copy
    
    def evaluate(self, data_loader, num_adaptation_steps=5):
        """Evaluate model on data loader"""
        self.model.eval()
        
        all_task_accuracies = []
        all_task_losses = []
        
        # Process each batch (each batch has multiple tasks)
        for batch_idx, batch in enumerate(data_loader):
            support_x_list, support_y_list = batch['support']
            query_x_list, query_y_list = batch['query']
            
            # Move to device
            support_x_list = [x.to(self.device) for x in support_x_list]
            support_y_list = [y.to(self.device) for y in support_y_list]
            query_x_list = [x.to(self.device) for x in query_x_list]
            query_y_list = [y.to(self.device) for y in query_y_list]
            
            # Process each task in the batch
            batch_size = len(support_x_list)
            batch_accuracies = []
            batch_losses = []
            
            for i in range(batch_size):
                support_x = support_x_list[i]
                support_y = support_y_list[i]
                query_x = query_x_list[i]
                query_y = query_y_list[i]
                
                # Determine number of unique classes
                num_unique_classes = len(torch.unique(torch.cat([support_y, query_y])))
                
                # Set up fast weights
                fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
                
                # Adapt using support set (inner loop)
                for _ in range(num_adaptation_steps):
                    # Forward pass
                    support_logits = self.forward_with_weights(support_x, fast_weights)
                    
                    # Ensure model output size matches number of classes
                    if support_logits.size(1) != num_unique_classes:
                        # Remap labels
                        unique_labels = torch.unique(support_y)
                        label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
                        new_support_y = torch.tensor([label_map[y.item()] for y in support_y], 
                                                   device=support_y.device, dtype=support_y.dtype)
                        
                        # Compute loss with remapped labels
                        inner_loss = self.criterion(support_logits[:, :num_unique_classes], new_support_y)
                    else:
                        # Normal loss
                        inner_loss = self.criterion(support_logits, support_y)
                    
                    # Compute gradients
                    grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=False)
                    
                    # Update fast weights
                    fast_weights = {name: param - self.inner_lr * grad
                                   for (name, param), grad in zip(fast_weights.items(), grads)}
                
                # Evaluate on query set (outer loop)
                with torch.no_grad():
                    query_logits = self.forward_with_weights(query_x, fast_weights)
                    
                    # Ensure model output size matches number of classes
                    if query_logits.size(1) != num_unique_classes:
                        # Remap labels
                        unique_query_labels = torch.unique(query_y)
                        query_label_map = {old_label.item(): new_label for new_label, old_label in enumerate(unique_query_labels)}
                        new_query_y = torch.tensor([query_label_map[y.item()] for y in query_y], 
                                                  device=query_y.device, dtype=query_y.dtype)
                        
                        # Calculate accuracy and loss with remapped labels
                        pred_class = torch.argmax(query_logits[:, :num_unique_classes], dim=1)
                        accuracy = (pred_class == new_query_y).float().mean().item()
                        loss = self.criterion(query_logits[:, :num_unique_classes], new_query_y).item()
                    else:
                        # Normal accuracy and loss
                        pred_class = torch.argmax(query_logits, dim=1)
                        accuracy = (pred_class == query_y).float().mean().item()
                        loss = self.criterion(query_logits, query_y).item()
                    
                    batch_accuracies.append(accuracy)
                    batch_losses.append(loss)
            
            # Add batch results to all results
            all_task_accuracies.extend(batch_accuracies)
            all_task_losses.extend(batch_losses)
        
        # Calculate average metrics
        avg_accuracy = sum(all_task_accuracies) / len(all_task_accuracies) if all_task_accuracies else 0
        avg_loss = sum(all_task_losses) / len(all_task_losses) if all_task_losses else 0
        
        return {
            'accuracy': avg_accuracy,
            'loss': avg_loss
        }

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Meta (outer loop) learning rate')
    parser.add_argument('--model_type', type=str, default='vit',
                        choices=['mlp', 'lstm', 'resnet18', 'transformer', 'vit'],
                        help='Type of model to use')
    parser.add_argument('--win_len', type=int, default=232,
                        help='Window length for CSI data')
    parser.add_argument('--feature_size', type=int, default=500,
                        help='Feature size for CSI data')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='Interval for evaluation')
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                        help='Key in h5 file for CSI data')
    parser.add_argument('--save_dir', type=str, default='results/meta_learning',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir) / f"{args.model_type}_{args.task_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_dir / f"meta_training_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    # Set device
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # Load meta-learning tasks
    logging.info(f"Loading meta-learning tasks from {args.data_dir}...")
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
        num_workers=4,
        collate_fn=custom_task_collate
    )
    
    # Create model
    logging.info(f"Creating {args.model_type.upper()} model...")
    model = BaseMetaModel(
        model_type=args.model_type,
        win_len=args.win_len,
        feature_size=args.feature_size,
        in_channels=args.in_channels,
        emb_dim=args.emb_dim,
        num_classes=args.n_way,
        dropout=args.dropout,
        inner_lr=args.inner_lr
    )
    
    # Log model parameters
    logging.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = MAMLTrainer(
        model=model,
        device=device,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr
    )
    
    # Training loop
    logging.info(f"Starting training for {args.num_iterations} iterations...")
    
    # Get training and validation loaders
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    
    if not train_loader:
        logging.error("No training loader found. Check your data directory and task name.")
        return
    
    # Set up metrics tracking
    train_losses = []
    train_support_accs = []
    train_query_accs = []
    val_accs = []
    val_losses = []
    best_val_acc = 0
    
    # Training loop
    train_iter = iter(train_loader)
    
    for iteration in range(args.num_iterations):
        # Get a batch of tasks
        try:
            task_batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            task_batch = next(train_iter)
        
        # Train on the batch
        train_metrics = trainer.train_step(task_batch)
        
        # Record metrics
        train_losses.append(train_metrics['meta_loss'])
        train_support_accs.append(train_metrics['support_acc'])
        train_query_accs.append(train_metrics['query_acc'])
        
        # Log progress
        if (iteration + 1) % 10 == 0:
            logging.info(f"Iteration {iteration+1}/{args.num_iterations}: "
                        f"Meta Loss: {train_metrics['meta_loss']:.4f}, "
                        f"Support Acc: {train_metrics['support_acc']:.4f}, "
                        f"Query Acc: {train_metrics['query_acc']:.4f}")
        
        # Evaluate on validation set
        if val_loader and (iteration + 1) % args.eval_interval == 0:
            val_metrics = trainer.evaluate(val_loader)
            val_accs.append(val_metrics['accuracy'])
            val_losses.append(val_metrics['loss'])
            
            logging.info(f"Validation: Acc: {val_metrics['accuracy']:.4f}, Loss: {val_metrics['loss']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'iteration': iteration + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.meta_optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'args': vars(args)
                }, save_dir / "best_model.pth")
                logging.info(f"Saved new best model with val acc: {best_val_acc:.4f}")
    
    # Save final model
    torch.save({
        'iteration': args.num_iterations,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.meta_optimizer.state_dict(),
        'val_acc': val_accs[-1] if val_accs else 0,
        'args': vars(args)
    }, save_dir / "final_model.pth")
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    # Plot training metrics
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Meta Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(train_support_accs, label='Support Accuracy')
    plt.plot(train_query_accs, label='Query Accuracy')
    if val_accs:
        # Plot validation points
        val_iters = [(i+1)*args.eval_interval for i in range(len(val_accs))]
        plt.plot(val_iters, val_accs, 'ro-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png")
    
    # Test on test set
    test_loader = loaders.get('test')
    if test_loader:
        logging.info("Evaluating on test set...")
        test_metrics = trainer.evaluate(test_loader)
        logging.info(f"Test: Acc: {test_metrics['accuracy']:.4f}, Loss: {test_metrics['loss']:.4f}")
    
    # Test on cross-domain data if available
    cross_env_loader = loaders.get('test_cross_env')
    if cross_env_loader:
        logging.info("Evaluating on cross-environment test set...")
        cross_env_metrics = trainer.evaluate(cross_env_loader)
        logging.info(f"Cross-Env: Acc: {cross_env_metrics['accuracy']:.4f}, Loss: {cross_env_metrics['loss']:.4f}")
    
    cross_user_loader = loaders.get('test_cross_user')
    if cross_user_loader:
        logging.info("Evaluating on cross-user test set...")
        cross_user_metrics = trainer.evaluate(cross_user_loader)
        logging.info(f"Cross-User: Acc: {cross_user_metrics['accuracy']:.4f}, Loss: {cross_user_metrics['loss']:.4f}")
    
    cross_device_loader = loaders.get('test_cross_device')
    if cross_device_loader:
        logging.info("Evaluating on cross-device test set...")
        cross_device_metrics = trainer.evaluate(cross_device_loader)
        logging.info(f"Cross-Device: Acc: {cross_device_metrics['accuracy']:.4f}, Loss: {cross_device_metrics['loss']:.4f}")
    
    logging.info("Training completed!")

if __name__ == "__main__":
    main() 