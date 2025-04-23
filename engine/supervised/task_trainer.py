import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import copy
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from engine.base_trainer import BaseTrainer
from engine.supervised.utils import warmup_schedule

class TaskTrainer(BaseTrainer):
    """Trainer for supervised learning tasks with CSI data."""
    
    def __init__(self, model, data_loader, config, criterion=None):
        """Initialize the trainer.
        
        Args:
            model: The model to train.
            data_loader: A tuple of (train_loader, val_loader).
            config: The configuration object.
            criterion: The loss function to use.
        """
        # Unpack data loaders
        self.train_loader, self.val_loader = data_loader
        
        super().__init__(model, self.train_loader, config)
        
        # Set criterion
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        
        # Setup optimizer
        self.setup_optimizer(
            learning_rate=getattr(config, 'learning_rate', 1e-4),
            weight_decay=getattr(config, 'weight_decay', 1e-5)
        )
        
        # Setup scheduler
        self.setup_scheduler()
        
        # Training tracking
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.epochs_no_improve = 0
        
        # Number of classes
        self.num_classes = getattr(config, 'num_classes', 2)
    
    def setup_scheduler(self):
        """Set up learning rate scheduler."""
        warmup_epochs = getattr(self.config, 'warmup_epochs', 5)
        lr_lambda = lambda epoch: warmup_schedule(epoch, warmup_epochs)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Train the model."""
        print('Starting supervised training phase...')
        
        # Records for tracking progress
        records = []
        
        # Number of epochs
        num_epochs = getattr(self.config, 'num_epochs', 100)
        patience = getattr(self.config, 'patience', 15)
        
        # Best model state
        best_model = None
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train one epoch
            train_loss, train_acc, train_time = self.train_epoch()
            
            # Evaluate
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            # Update records
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Step scheduler
            self.scheduler.step()
            
            # Record for this epoch
            record = {
                'Epoch': epoch + 1,
                'Train Loss': train_loss,
                'Val Loss': val_loss,
                'Train Accuracy': train_acc,
                'Val Accuracy': val_acc,
                'Time per sample': train_time
            }
            records.append(record)
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Time per sample: {train_time:.6f} seconds')
            
            # Early stopping check
            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())
                self.save_model(name="best_model.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping triggered after {patience} epochs without improvement.')
                    self.model.load_state_dict(best_model)
                    break
        
        # Create results DataFrame
        results_df = pd.DataFrame(records)
        
        # Save results
        results_df.to_csv(os.path.join(self.save_path, 'training_results.csv'), index=False)
        
        # Plot results
        self.plot_training_results()
        
        return self.model, results_df
    
    def train_epoch(self):
        """Train the model for a single epoch.
        
        Returns:
            A tuple of (loss, accuracy, time_per_sample).
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        total_samples = 0
        total_time = 0.0
        
        for inputs, labels in self.train_loader:
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Transfer to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # One-hot encoding for labels if needed
            if self.criterion.__class__.__name__ in ['BCELoss', 'BCEWithLogitsLoss']:
                labels_one_hot = F.one_hot(labels, self.num_classes).float()
            else:
                labels_one_hot = labels
            
            start_time = time.time()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels_one_hot)
            
            # Backward pass
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Measure elapsed time
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # Accumulate loss and accuracy
            epoch_loss += loss.item() * batch_size
            
            # Calculate accuracy
            if outputs.shape[1] > 1:  # Multi-class
                predicted = torch.argmax(outputs, dim=1)
                correct = (predicted == labels).sum().item()
            else:  # Binary
                predicted = (outputs > 0.5).float()
                correct = (predicted == labels).sum().item()
                
            epoch_accuracy += correct
        
        # Calculate averages
        epoch_loss /= total_samples
        epoch_accuracy /= total_samples
        time_per_sample = total_time / total_samples
        
        return epoch_loss, epoch_accuracy, time_per_sample
    
    def evaluate(self, data_loader):
        """Evaluate the model.
        
        Args:
            data_loader: The data loader to use for evaluation.
            
        Returns:
            A tuple of (loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                # Transfer to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # One-hot encoding for labels if needed
                if self.criterion.__class__.__name__ in ['BCELoss', 'BCEWithLogitsLoss']:
                    labels_one_hot = F.one_hot(labels, self.num_classes).float()
                else:
                    labels_one_hot = labels
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels_one_hot)
                
                # Accumulate loss
                total_loss += loss.item() * batch_size
                
                # Calculate accuracy
                if outputs.shape[1] > 1:  # Multi-class
                    predicted = torch.argmax(outputs, dim=1)
                    correct = (predicted == labels).sum().item()
                else:  # Binary
                    predicted = (outputs > 0.5).float()
                    correct = (predicted == labels).sum().item()
                    
                total_correct += correct
        
        # Calculate averages
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def plot_training_results(self):
        """Plot the training results."""
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training loss
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].grid(True)
        
        # Plot validation loss
        axs[0, 1].plot(self.val_losses)
        axs[0, 1].set_title('Validation Loss')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Loss')
        axs[0, 1].grid(True)
        
        # Plot training accuracy
        axs[1, 0].plot(self.train_accuracies)
        axs[1, 0].set_title('Training Accuracy')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Accuracy')
        axs[1, 0].grid(True)
        
        # Plot validation accuracy
        axs[1, 1].plot(self.val_accuracies)
        axs[1, 1].set_title('Validation Accuracy')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'training_results.png'))
        plt.close()
        
        # Also plot confusion matrix if validation loader is available
        if hasattr(self, 'val_loader'):
            self.plot_confusion_matrix()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix for validation data."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                if outputs.shape[1] > 1:  # Multi-class
                    preds = torch.argmax(outputs, dim=1)
                else:  # Binary
                    preds = (outputs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.save_path, 'confusion_matrix.png'))
        plt.close()
        
        # Also save classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        pd.DataFrame(report).to_csv(os.path.join(self.save_path, 'classification_report.csv'), index=False)
