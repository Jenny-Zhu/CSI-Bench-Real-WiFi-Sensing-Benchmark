import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import time
from pathlib import Path

from model.meta_learning.models import MLPClassifier
from load.meta_learning.meta_dataset import MetaTaskDataset
from torch.utils.data import DataLoader

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_meta_datasets(data_dir, n_way, k_shot, q_query, data_key='CSI_amps'):
    """Create meta-learning datasets for training and testing"""
    # Create directories for each split
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_env_dir = os.path.join(data_dir, 'test_cross_env')
    
    # Print message if they don't exist yet
    if not os.path.exists(train_dir):
        print(f"Meta-learning data directories not found. Please organize your data or use data_loader.py to create the meta-learning dataset structure.")
        print(f"Expected directory structure:")
        print(f"  {train_dir}/class_name/samples...")
        print(f"  {val_dir}/class_name/samples...")
        print(f"  {test_env_dir}/class_name/samples...")
        return None, None, None
    
    # Create datasets
    train_dataset = MetaTaskDataset(
        data_dir=train_dir,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        file_ext='.h5',
        data_key=data_key
    )
    
    val_dataset = MetaTaskDataset(
        data_dir=val_dir,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        file_ext='.h5',
        data_key=data_key
    ) if os.path.exists(val_dir) else None
    
    test_env_dataset = MetaTaskDataset(
        data_dir=test_env_dir,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        file_ext='.h5',
        data_key=data_key
    ) if os.path.exists(test_env_dir) else None
    
    return train_dataset, val_dataset, test_env_dataset

def create_meta_data_loaders(train_dataset, val_dataset, test_env_dataset, batch_size, num_workers=4):
    """Create data loaders for meta-learning"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    ) if train_dataset else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if val_dataset else None
    
    test_env_loader = DataLoader(
        test_env_dataset,
        batch_size=1,  # For testing, we evaluate one task at a time
        shuffle=False,
        num_workers=num_workers
    ) if test_env_dataset else None
    
    return train_loader, val_loader, test_env_loader

class SimplifiedMAMLTrainer:
    def __init__(self, model, device, inner_lr=0.01, meta_lr=0.001):
        self.model = model
        self.device = device
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Move model to device
        self.model.to(device)
    
    def train_step(self, task_batch):
        """Perform a single meta-training step"""
        meta_loss = 0.0
        task_accuracies = []
        
        self.meta_optimizer.zero_grad()
        
        for task in task_batch:
            support_x, support_y = task['support']
            query_x, query_y = task['query']
            
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Save original parameters
            original_params = {name: param.clone() for name, param in self.model.named_parameters()}
            
            # Inner loop - adapt to task
            for _ in range(5):  # 5 adaptation steps
                # Forward pass
                support_logits = self.model(support_x)
                support_loss = self.criterion(support_logits, support_y)
                
                # Compute gradients for inner update
                grads = torch.autograd.grad(
                    support_loss, 
                    self.model.parameters(),
                    create_graph=True
                )
                
                # Inner update
                for param, grad in zip(self.model.parameters(), grads):
                    param.data = param.data - self.inner_lr * grad
            
            # Evaluate on query set with adapted parameters
            query_logits = self.model(query_x)
            task_loss = self.criterion(query_logits, query_y)
            
            # Add to meta loss
            meta_loss += task_loss
            
            # Calculate accuracy
            _, predicted = torch.max(query_logits.data, 1)
            correct = (predicted == query_y).sum().item()
            task_accuracies.append(correct / query_y.size(0))
            
            # Restore original parameters
            for name, param in self.model.named_parameters():
                param.data = original_params[name]
        
        # Backward pass on meta loss
        meta_loss = meta_loss / len(task_batch)
        meta_loss.backward()
        
        # Update meta parameters
        self.meta_optimizer.step()
        
        # Calculate average accuracy
        avg_accuracy = sum(task_accuracies) / len(task_accuracies)
        
        return meta_loss.item(), avg_accuracy
    
    def evaluate(self, data_loader, num_adaptation_steps=5):
        """Evaluate the model on a validation/test set"""
        self.model.eval()
        all_task_accuracies = []
        all_task_losses = []
        adaptation_curves = []
        
        for task in data_loader:
            support_x, support_y = task['support']
            query_x, query_y = task['query']
            
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Save original parameters for later restoration
            original_params = {name: param.clone() for name, param in self.model.named_parameters()}
            
            # Track accuracy during adaptation
            step_accuracies = []
            
            # Evaluate before adaptation (0-shot)
            with torch.no_grad():
                query_logits = self.model(query_x)
                query_loss = self.criterion(query_logits, query_y)
                _, predicted = torch.max(query_logits.data, 1)
                correct = (predicted == query_y).sum().item()
                accuracy = correct / query_y.size(0)
                step_accuracies.append(accuracy)
            
            # Adapt to task (inner loop)
            for step in range(num_adaptation_steps):
                # Forward pass
                support_logits = self.model(support_x)
                support_loss = self.criterion(support_logits, support_y)
                
                # Compute gradients for inner update
                grads = torch.autograd.grad(
                    support_loss, 
                    self.model.parameters(),
                    create_graph=False
                )
                
                # Inner update
                for param, grad in zip(self.model.parameters(), grads):
                    param.data = param.data - self.inner_lr * grad
                
                # Evaluate after each adaptation step
                with torch.no_grad():
                    query_logits = self.model(query_x)
                    _, predicted = torch.max(query_logits.data, 1)
                    correct = (predicted == query_y).sum().item()
                    accuracy = correct / query_y.size(0)
                    step_accuracies.append(accuracy)
            
            # Final evaluation after adaptation
            with torch.no_grad():
                query_logits = self.model(query_x)
                query_loss = self.criterion(query_logits, query_y)
                
            # Store results
            all_task_losses.append(query_loss.item())
            all_task_accuracies.append(step_accuracies[-1])
            adaptation_curves.append(step_accuracies)
            
            # Restore original parameters
            for name, param in self.model.named_parameters():
                param.data = original_params[name]
        
        # Calculate overall results
        avg_accuracy = sum(all_task_accuracies) / len(all_task_accuracies)
        avg_loss = sum(all_task_losses) / len(all_task_losses)
        avg_adaptation_curve = np.mean(adaptation_curves, axis=0)
        
        self.model.train()
        
        return {
            'accuracy': avg_accuracy,
            'loss': avg_loss,
            'adaptation_curve': avg_adaptation_curve,
            'std_accuracy': np.std(all_task_accuracies)
        }
    
    def train(self, train_loader, val_loader=None, num_iterations=10000, 
              validation_interval=1000, save_path=None, verbose=True):
        """Train the model with MAML"""
        # Create save directory if needed
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        # Training records
        train_losses = []
        train_accuracies = []
        val_results = []
        best_val_accuracy = 0.0
        
        # Time tracking
        start_time = time.time()
        
        # Training loop
        iteration = 0
        while iteration < num_iterations:
            for task_batch in train_loader:
                # Train step
                loss, accuracy = self.train_step(task_batch)
                
                # Record results
                train_losses.append(loss)
                train_accuracies.append(accuracy)
                
                # Print progress
                if verbose and iteration % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Iteration {iteration}/{num_iterations} - "
                          f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
                          f"Time: {elapsed:.2f}s")
                
                # Validate periodically
                if val_loader and iteration % validation_interval == 0:
                    val_result = self.evaluate(val_loader)
                    val_results.append(val_result)
                    
                    if verbose:
                        print(f"Validation - Accuracy: {val_result['accuracy']:.4f} ± "
                              f"{val_result['std_accuracy']:.4f}, Loss: {val_result['loss']:.4f}")
                    
                    # Save best model
                    if save_path and val_result['accuracy'] > best_val_accuracy:
                        best_val_accuracy = val_result['accuracy']
                        torch.save(self.model.state_dict(), 
                                  os.path.join(save_path, 'best_model.pt'))
                        print(f"Saved best model with accuracy: {best_val_accuracy:.4f}")
                
                iteration += 1
                if iteration >= num_iterations:
                    break
        
        # Save final model
        if save_path:
            torch.save(self.model.state_dict(), 
                      os.path.join(save_path, 'final_model.pt'))
        
        # Plot training results
        if save_path:
            self._plot_training_results(train_losses, train_accuracies, val_results, 
                                      validation_interval, save_path)
        
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_results': val_results
        }
    
    def _plot_training_results(self, train_losses, train_accuracies, val_results, 
                             validation_interval, save_path):
        """Plot and save training results"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training loss
        axes[0, 0].plot(train_losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        
        # Plot training accuracy
        axes[0, 1].plot(train_accuracies)
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Accuracy')
        
        # Plot validation metrics if available
        if val_results:
            val_iterations = list(range(0, len(train_losses), validation_interval))[:len(val_results)]
            
            # Validation accuracy
            val_accuracies = [result['accuracy'] for result in val_results]
            axes[1, 0].plot(val_iterations, val_accuracies)
            axes[1, 0].set_title('Validation Accuracy')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Accuracy')
            
            # Validation loss
            val_losses = [result['loss'] for result in val_results]
            axes[1, 1].plot(val_iterations, val_losses)
            axes[1, 1].set_title('Validation Loss')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_results.png'))
        plt.close()

def test_on_cross_env(model, test_loader, device, inner_lr=0.01, 
                     num_adaptation_steps=5, save_path=None):
    """Test the model on cross-environment data"""
    model.to(device)
    model.eval()
    
    trainer = SimplifiedMAMLTrainer(model, device, inner_lr=inner_lr)
    test_results = trainer.evaluate(test_loader, num_adaptation_steps=num_adaptation_steps)
    
    print(f"\nCross-Environment Test Results:")
    print(f"  Final Accuracy: {test_results['accuracy']:.4f} ± {test_results['std_accuracy']:.4f}")
    print(f"  Loss: {test_results['loss']:.4f}")
    
    # Plot adaptation curve
    if save_path:
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_adaptation_steps + 1), test_results['adaptation_curve'])
        plt.title('Cross-Environment Adaptation Curve')
        plt.xlabel('Adaptation Steps')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'cross_env_adaptation.png'))
        plt.close()
        
        # Save detailed results
        np.save(os.path.join(save_path, 'cross_env_results.npy'), test_results)
    
    return test_results

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate MLP with meta-learning for cross-environment generalization')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset/tasks/MotionSourceRecognition',
                       help='Root directory of the dataset')
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                       help='Key in h5 file for CSI data')
    
    # Task parameters
    parser.add_argument('--n_way', type=int, default=2,
                       help='Number of classes per task')
    parser.add_argument('--k_shot', type=int, default=5,
                       help='Number of support examples per class')
    parser.add_argument('--q_query', type=int, default=5,
                       help='Number of query examples per class')
    
    # Model parameters
    parser.add_argument('--win_len', type=int, default=250,
                       help='Window length for WiFi CSI data')
    parser.add_argument('--feature_size', type=int, default=98,
                       help='Feature size for WiFi CSI data')
    parser.add_argument('--emb_dim', type=int, default=128,
                       help='Embedding dimension')
    
    # Training parameters
    parser.add_argument('--meta_batch_size', type=int, default=4,
                       help='Number of tasks per batch')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                       help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                       help='Outer loop learning rate')
    parser.add_argument('--num_iterations', type=int, default=10000,
                       help='Number of iterations for meta-learning')
    parser.add_argument('--validation_interval', type=int, default=1000,
                       help='Interval for validation')
    parser.add_argument('--adaptation_steps', type=int, default=5,
                       help='Number of adaptation steps for testing')
    
    # Other parameters
    parser.add_argument('--save_dir', type=str, default='results/meta_learning/mlp',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_train', action='store_true',
                       help='Skip training and only test')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_path / 'config.json', 'w') as f:
        config = {k: v for k, v in vars(args).items()}
        json.dump(config, f, indent=4)
    
    # Create datasets and loaders
    train_dataset, val_dataset, test_env_dataset = create_meta_datasets(
        args.data_dir, args.n_way, args.k_shot, args.q_query, args.data_key
    )
    
    if train_dataset is None:
        print("No datasets found. Please prepare the data first.")
        return
    
    train_loader, val_loader, test_env_loader = create_meta_data_loaders(
        train_dataset, val_dataset, test_env_dataset, args.meta_batch_size
    )
    
    # Create model
    print(f"Creating MLP meta-learning model...")
    model = MLPClassifier(
        win_len=args.win_len,
        feature_size=args.feature_size,
        num_classes=args.n_way,
        emb_dim=args.emb_dim
    )
    
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = SimplifiedMAMLTrainer(
        model=model,
        device=device,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr
    )
    
    # Train if not skipped
    if not args.no_train:
        print(f"Starting meta-learning training...")
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_iterations=args.num_iterations,
            validation_interval=args.validation_interval,
            save_path=save_path,
            verbose=True
        )
    else:
        # Load best model if available
        model_path = save_path / 'best_model.pt'
        if model_path.exists():
            print(f"Loading best model from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"No pre-trained model found at {model_path}. Using random initialization.")
    
    # Test on cross-environment data
    if test_env_loader:
        print(f"\nEvaluating on cross-environment data...")
        test_results = test_on_cross_env(
            model=model,
            test_loader=test_env_loader,
            device=device,
            inner_lr=args.inner_lr,
            num_adaptation_steps=args.adaptation_steps,
            save_path=save_path
        )
    else:
        print("No cross-environment test data available.")
    
    print(f"\nExperiment completed! Results saved to {save_path}")

if __name__ == "__main__":
    main() 