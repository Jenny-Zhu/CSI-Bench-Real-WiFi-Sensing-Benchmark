import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from engine.base_trainer import BaseTrainer

class MetaTrainer(BaseTrainer):
    """Base Meta-Learning Trainer"""
    
    def __init__(self, model, data_loader, config, meta_optimizer=None):
        """Initialize the meta trainer
        
        Args:
            model: The model to train
            data_loader: Data loader
            config: Configuration object
            meta_optimizer: Meta optimizer (optional)
        """
        super().__init__(model, data_loader, config)
        
        # Meta-learning parameters
        self.inner_lr = getattr(config, 'inner_lr', 0.01)  # Inner loop learning rate
        self.meta_lr = getattr(config, 'meta_lr', 0.001)   # Meta learning rate
        self.n_way = getattr(config, 'n_way', 2)           # N-way classification
        self.k_shot = getattr(config, 'k_shot', 5)         # K-shot support set
        self.q_query = getattr(config, 'q_query', 15)      # Query set size
        self.meta_batch_size = getattr(config, 'meta_batch_size', 4)  # Meta batch size
        
        # Setup meta optimizer
        if meta_optimizer is None:
            self.meta_optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.meta_lr
            )
        else:
            self.meta_optimizer = meta_optimizer
            
        # Setup inner loop criterion
        self.inner_criterion = nn.CrossEntropyLoss()
        
        # Meta training records
        self.meta_train_losses = []
        self.meta_train_accuracies = []
        self.meta_val_losses = []
        self.meta_val_accuracies = []
    
    def train(self):
        """Train the model"""
        print('Starting Meta-Learning training phase...')
        
        # Use specific meta-learning algorithm
        if self.config.meta_method == 'maml':
            from engine.meta_learning.maml_training import maml_train
            return maml_train(
                model=self.model,
                meta_optimizer=self.meta_optimizer,
                train_loader=self.data_loader['train'],
                val_loader=self.data_loader['val'] if 'val' in self.data_loader else None,
                inner_lr=self.inner_lr,
                meta_batch_size=self.meta_batch_size,
                n_way=self.n_way,
                k_shot=self.k_shot,
                q_query=self.q_query,
                num_iterations=getattr(self.config, 'num_iterations', 60000),
                meta_validation_interval=getattr(self.config, 'meta_validation_interval', 1000),
                device=self.device,
                save_path=self.save_path
            )
        elif self.config.meta_method == 'lstm':
            from engine.meta_learning.lstm_training import lstm_meta_train
            return lstm_meta_train(
                model=self.model,
                meta_optimizer=self.meta_optimizer,
                train_loader=self.data_loader['train'],
                val_loader=self.data_loader['val'] if 'val' in self.data_loader else None,
                lstm_optimizer=None,  # Will be created inside the function
                meta_batch_size=self.meta_batch_size,
                n_way=self.n_way,
                k_shot=self.k_shot,
                q_query=self.q_query,
                num_iterations=getattr(self.config, 'num_iterations', 60000),
                meta_validation_interval=getattr(self.config, 'meta_validation_interval', 1000),
                device=self.device,
                save_path=self.save_path
            )
        else:
            raise ValueError(f"Unsupported meta-learning method: {self.config.meta_method}")
    
    def evaluate(self, data_loader):
        """Evaluate the model
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Evaluation metrics
        """
        # Meta-learning evaluation is different from regular evaluation, test few-shot learning
        meta_test_results = self.meta_test(data_loader)
        return meta_test_results
    
    def meta_test(self, data_loader, num_tasks=10, num_adaptation_steps=5):
        """Perform meta-testing to test few-shot learning capability
        
        Args:
            data_loader: Test data loader
            num_tasks: Number of tasks to test
            num_adaptation_steps: Number of adaptation steps
            
        Returns:
            Dictionary with test results
        """
        self.model.eval()  # Evaluation mode
        
        accuracies = []
        losses = []
        adaptation_accuracies = []  # Track accuracy during adaptation
        
        # Create an SGD optimizer for each task
        inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        criterion = nn.CrossEntropyLoss()
        
        # Sample num_tasks tasks
        for task_idx in range(num_tasks):
            # Sample support and query set
            try:
                support_x, support_y, query_x, query_y = next(iter(data_loader))
            except:
                # If data loader format is different
                task = next(iter(data_loader))
                support_x, support_y = task['support']
                query_x, query_y = task['query']
            
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Clone model weights
            model_state = {name: param.clone() for name, param in self.model.named_parameters()}
            
            # Accuracies during adaptation
            task_adaptation_accuracies = []
            
            # Adaptation phase (several gradient steps)
            for step in range(num_adaptation_steps):
                # Forward pass
                support_logits = self.model(support_x)
                support_loss = criterion(support_logits, support_y)
                
                # Backward pass
                inner_optimizer.zero_grad()
                support_loss.backward()
                inner_optimizer.step()
                
                # Calculate accuracy on query set after adaptation
                with torch.no_grad():
                    query_logits = self.model(query_x)
                    query_loss = criterion(query_logits, query_y)
                    query_acc = (query_logits.argmax(dim=1) == query_y).float().mean().item()
                    task_adaptation_accuracies.append(query_acc)
            
            # Record final results
            accuracies.append(query_acc)
            losses.append(query_loss.item())
            adaptation_accuracies.append(task_adaptation_accuracies)
            
            # Restore original model weights
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(model_state[name])
        
        # Calculate average results
        mean_accuracy = np.mean(accuracies)
        mean_loss = np.mean(losses)
        std_accuracy = np.std(accuracies)
        
        # Generate adaptation curve
        mean_adaptation_curve = np.mean(adaptation_accuracies, axis=0)
        
        # Plot adaptation curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_adaptation_steps + 1), mean_adaptation_curve)
        plt.xlabel('Adaptation Steps')
        plt.ylabel('Query Set Accuracy')
        plt.title(f'Meta-Test Adaptation Curve (Avg Accuracy: {mean_accuracy:.4f}±{std_accuracy:.4f})')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, 'meta_test_adaptation.png'))
        plt.close()
        
        print(f"Meta-Test Results - Mean Accuracy: {mean_accuracy:.4f}±{std_accuracy:.4f}, Mean Loss: {mean_loss:.4f}")
        
        return {
            'accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'loss': mean_loss,
            'adaptation_curve': mean_adaptation_curve
        }
