import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from engine.base_trainer import BaseTrainer

class SSLTrainer(BaseTrainer):
    """Trainer for self-supervised learning."""
    
    def __init__(self, model, data_loader, config, criterion=None, augmentor=None):
        """Initialize the trainer.
        
        Args:
            model: The model to train.
            data_loader: The data loader to use.
            config: The configuration object.
            criterion: The loss function to use.
            augmentor: The data augmentation to use.
        """
        super().__init__(model, data_loader, config)
        
        self.criterion = criterion
        self.augmentor = augmentor
        
        # Setup optimizer
        self.setup_optimizer(
            learning_rate=getattr(config, 'learning_rate', 1e-4),
            weight_decay=getattr(config, 'weight_decay', 1e-5)
        )
        
        # Setup schedulers
        self.warmup_epochs = getattr(config, 'warmup_epochs', 5)
        self.setup_schedulers()
        
        # Best model tracking
        self.epochs_no_improve = 0
    
    def setup_schedulers(self):
        """Set up learning rate schedulers."""
        # Warmup scheduler
        warmup_lambda = lambda epoch: min((epoch + 1) / self.warmup_epochs, 1.0)
        self.scheduler_warmup = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
        
        # Cosine annealing scheduler
        self.scheduler_cosine = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=1e-4
        )
    
    def train(self):
        """Train the model."""
        print('Starting self-supervised training phase...')
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Pick scheduler
            scheduler = self.scheduler_warmup if epoch < self.warmup_epochs else self.scheduler_cosine
            
            current_lr = scheduler.get_last_lr()[0]
            self.learning_rates.append(current_lr)
            print(f'[Epoch {epoch+1}/{self.config.num_epochs}] Current LR: {current_lr:.2e}')
            
            # Training step
            epoch_loss = self.train_epoch()
            self.train_losses.append(epoch_loss)
            
            print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}')
            
            # Step the chosen scheduler
            scheduler.step()
            
            # Early stopping check
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.epochs_no_improve = 0
                self.best_model_state = self.model.state_dict().copy()
                self.save_model(name=f"best_model_epoch_{epoch+1}.pt")
            else:
                self.epochs_no_improve += 1
            
            print(f'Epochs without improvement: {self.epochs_no_improve}, Best Loss: {self.best_loss:.4f}')
            
            if self.epochs_no_improve >= getattr(self.config, 'patience', 20):
                print(f'Early stopping triggered after {self.config.patience} epochs without improvement.')
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Plot training curve
        self.plot_losses()
        
        return self.model, pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'learning_rate': self.learning_rates
        })
    
    def train_epoch(self):
        """Train for one epoch.
        
        Returns:
            The average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        
        for data in self.data_loader:
            x = data.to(self.device)
            
            # Two augmented views
            x1 = self.augmentor.apply_augmentations(x)
            x2 = self.augmentor.apply_augmentations(x)
            
            # Forward pass
            z1, z2 = self.model(x1, x2)
            
            # Loss
            loss = self.criterion(z1, z2)
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.data_loader)
    
    def evaluate(self, data_loader):
        """Evaluate the model.
        
        Args:
            data_loader: The data loader to use for evaluation.
            
        Returns:
            The evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data in data_loader:
                x = data.to(self.device)
                
                # Two augmented views
                x1 = self.augmentor.apply_augmentations(x)
                x2 = self.augmentor.apply_augmentations(x)
                
                # Forward pass
                z1, z2 = self.model(x1, x2)
                
                # Loss
                loss = self.criterion(z1, z2)
                
                total_loss += loss.item()
        
        return {'loss': total_loss / len(data_loader)}
