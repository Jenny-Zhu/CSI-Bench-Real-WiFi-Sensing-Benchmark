import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """Base class for all trainers."""
    
    def __init__(self, model, data_loader, config, device=None):
        """Initialize the base trainer.
        
        Args:
            model: The model to train.
            data_loader: The data loader to use.
            config: The configuration object.
            device: The device to use.
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Set up paths
        self.save_path = os.path.join(config.output_dir, config.results_subdir, config.model_name)
        os.makedirs(self.save_path, exist_ok=True)
        
        # Initialize training records
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_model_state = None
    
    @abstractmethod
    def train(self):
        """Train the model."""
        pass
    
    @abstractmethod
    def evaluate(self, data_loader):
        """Evaluate the model.
        
        Args:
            data_loader: The data loader to use for evaluation.
            
        Returns:
            The evaluation metrics.
        """
        pass
    
    def save_model(self, path=None, name=None):
        """Save the model.
        
        Args:
            path: The path to save the model to.
            name: The name of the model.
        """
        if path is None:
            path = self.save_path
        if name is None:
            name = "model.pt"
        
        full_path = os.path.join(path, name)
        
        save_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'epoch': self.current_epoch,
            'loss': self.best_loss
        }
        
        torch.save(save_state, full_path)
        print(f"Model saved to {full_path}")
    
    def load_model(self, path, optimizer=None):
        """Load a model.
        
        Args:
            path: The path to load the model from.
            optimizer: Optional optimizer to load state.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        elif hasattr(self, 'optimizer') and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
        if 'loss' in checkpoint:
            self.best_loss = checkpoint['loss']
            
        print(f"Model loaded from {path} (epoch {self.current_epoch})")
        
        return self.model
    
    def plot_losses(self, save=True):
        """Plot the training and validation losses.
        
        Args:
            save: Whether to save the plot.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        if len(self.val_losses) > 0:
            plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        if save:
            plt.savefig(os.path.join(self.save_path, "loss_plot.png"))
        
        plt.close()
    
    def setup_optimizer(self, learning_rate=None, weight_decay=None):
        """Set up the optimizer.
        
        Args:
            learning_rate: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
        """
        if learning_rate is None:
            learning_rate = getattr(self.config, 'learning_rate', 1e-4)
        if weight_decay is None:
            weight_decay = getattr(self.config, 'weight_decay', 1e-5)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        return self.optimizer
