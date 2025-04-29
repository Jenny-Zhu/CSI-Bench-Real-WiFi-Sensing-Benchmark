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
import json
import socket
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import LambdaLR
from engine.base_trainer import BaseTrainer
from engine.supervised.utils import warmup_schedule

class TaskTrainerACF(BaseTrainer):
    """Trainer for supervised learning tasks with ACF data."""
    
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
        
        # Detailed metrics tracking
        self.batch_train_losses = []
        self.batch_train_accuracies = []
        self.learning_rates = []
        
        # Number of classes
        self.num_classes = getattr(config, 'num_classes', 2)
        
        # Experiment tracking
        self.experiment_metadata = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'hostname': socket.gethostname(),
            'model_name': getattr(config, 'model_name', 'unknown'),
            'mode': 'acf',
            'num_classes': self.num_classes,
            'learning_rate': getattr(config, 'learning_rate', 1e-4),
            'weight_decay': getattr(config, 'weight_decay', 1e-5),
            'num_epochs': getattr(config, 'num_epochs', 100),
            'patience': getattr(config, 'patience', 15),
            'batch_size': self.train_loader.batch_size if hasattr(self.train_loader, 'batch_size') else 'unknown',
            'optimizer': self.optimizer.__class__.__name__,
            'scheduler': self.scheduler.__class__.__name__,
        }
        
        # ACF specific parameters
        self.freeze_cnn = getattr(config, 'freeze_backbone', False)
        if self.freeze_cnn:
            self.freeze_cnn_layers()
    
    def freeze_cnn_layers(self):
        """Freeze CNN layers to only train the classifier part"""
        # Find all convolutional layers and related batch normalization layers
        for name, param in self.model.named_parameters():
            if any(layer_type in name for layer_type in ['conv', 'bn', 'backbone']):
                param.requires_grad = False
        
        # Only train the classifier (fully connected layers)
        for name, param in self.model.named_parameters():
            if any(layer_type in name for layer_type in ['fc', 'classifier', 'head']):
                param.requires_grad = True
                
        print("CNN layers frozen, only training classifier layers")
        
        # Add to metadata
        self.experiment_metadata['freeze_backbone'] = True
    
    def setup_scheduler(self):
        """Set up learning rate scheduler"""
        warmup_epochs = getattr(self.config, 'warmup_epochs', 5)
        lr_lambda = lambda epoch: warmup_schedule(epoch, warmup_epochs)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
    
    def train(self):
        """Train the model"""
        print('Starting ACF supervised training phase...')
        
        # Save experiment metadata
        metadata_path = os.path.join(self.save_path, 'experiment_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
        
        # Records for tracking progress
        records = []
        
        # Training parameters
        num_epochs = getattr(self.config, 'num_epochs', 100)
        patience = getattr(self.config, 'patience', 15)
        
        # Best model state
        best_model = None
        best_val_loss = float('inf')
        best_val_acc = 0.0
        epochs_no_improve = 0
        
        # Record start time
        train_start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            print(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train one epoch
            train_loss, train_acc, train_time, batch_losses, batch_accs = self.train_epoch()
            
            # Record batch-level metrics
            self.batch_train_losses.extend(batch_losses)
            self.batch_train_accuracies.extend(batch_accs)
            
            # Evaluate
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            # Record current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
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
                'Time per sample': train_time,
                'Learning Rate': current_lr
            }
            records.append(record)
            
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
            print(f'Learning Rate: {current_lr:.8f}')
            print(f'Time per sample: {train_time:.6f} seconds')
            
            # Early stopping check
            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model = copy.deepcopy(self.model.state_dict())
                self.save_model(name="best_model.pt")
                print(f'New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping triggered after {patience} epochs without improvement.')
                    self.model.load_state_dict(best_model)
                    break
        
        # Record total training time
        total_train_time = time.time() - train_start_time
        print(f'Total training time: {total_train_time:.2f} seconds ({total_train_time/60:.2f} minutes)')
        
        # Create results DataFrame
        results_df = pd.DataFrame(records)
        
        # Save batch-level metrics
        batch_metrics = pd.DataFrame({
            'Batch Index': range(len(self.batch_train_losses)),
            'Batch Loss': self.batch_train_losses,
            'Batch Accuracy': self.batch_train_accuracies
        })
        batch_metrics.to_csv(os.path.join(self.save_path, 'batch_metrics.csv'), index=False)
        
        # Save experiment summary
        summary = {
            'total_epochs': epoch + 1,
            'early_stopped': epochs_no_improve == patience,
            'best_val_loss': float(best_val_loss),
            'best_val_accuracy': float(best_val_acc),
            'final_train_loss': float(train_loss),
            'final_train_accuracy': float(train_acc),
            'final_val_loss': float(val_loss),
            'final_val_accuracy': float(val_acc),
            'total_training_time': total_train_time,
            'average_time_per_epoch': total_train_time / (epoch + 1)
        }
        
        # Update metadata with training results
        self.experiment_metadata.update(summary)
        with open(metadata_path, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
        
        # Save results
        results_df.to_csv(os.path.join(self.save_path, 'training_results.csv'), index=False)
        
        # Plot results
        self.plot_training_results()
        
        return self.model, results_df
    
    def train_epoch(self):
        """Train for one epoch
        
        Returns:
            A tuple of (epoch_loss, epoch_accuracy, time_per_sample, batch_losses, batch_accuracies).
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        total_samples = 0
        total_time = 0.0
        
        # For tracking batch-level metrics
        batch_losses = []
        batch_accuracies = []
        
        for batch_idx, (inputs, labels, domains) in enumerate(self.train_loader):  # ACF data loader includes domain info
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Transfer to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            if domains is not None:
                domains = domains.to(self.device)
            
            # One-hot encoding for labels if needed
            if self.criterion.__class__.__name__ in ['BCELoss', 'BCEWithLogitsLoss']:
                labels_one_hot = F.one_hot(labels, self.num_classes).float()
            else:
                labels_one_hot = labels
            
            # Measure forward/backward pass time
            start_time = time.time()
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Call model
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
            batch_loss = loss.item()
            epoch_loss += batch_loss * batch_size
            
            # Calculate accuracy
            if outputs.shape[1] > 1:  # Multi-class
                predicted = torch.argmax(outputs, dim=1)
                correct = (predicted == labels).sum().item()
            else:  # Binary
                predicted = (outputs > 0.5).float()
                correct = (predicted == labels).sum().item()
            
            batch_accuracy = correct / batch_size
            epoch_accuracy += correct
            
            # Record batch-level metrics
            batch_losses.append(batch_loss)
            batch_accuracies.append(batch_accuracy)
            
            # Print batch progress periodically
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(self.train_loader)}: Loss {batch_loss:.4f}, Acc {batch_accuracy:.4f}')
        
        # Calculate averages
        epoch_loss /= total_samples
        epoch_accuracy /= total_samples
        time_per_sample = total_time / total_samples
        
        return epoch_loss, epoch_accuracy, time_per_sample, batch_losses, batch_accuracies
    
    def evaluate(self, data_loader):
        """Evaluate the model
        
        Args:
            data_loader: The data loader to use for evaluation
            
        Returns:
            A tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels, *extra in data_loader:
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
        axs[0, 0].plot(self.train_losses, label='Training')
        axs[0, 0].plot(self.val_losses, label='Validation')
        axs[0, 0].set_title('Loss Curves')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)
        
        # Plot training accuracy
        axs[0, 1].plot(self.train_accuracies, label='Training')
        axs[0, 1].plot(self.val_accuracies, label='Validation')
        axs[0, 1].set_title('Accuracy Curves')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Plot learning rate
        axs[1, 0].plot(self.learning_rates)
        axs[1, 0].set_title('Learning Rate')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Learning Rate')
        axs[1, 0].grid(True)
        
        # Plot batch training loss (most recent epoch)
        if len(self.batch_train_losses) > 0:
            batches_per_epoch = len(self.train_loader)
            recent_batches = self.batch_train_losses[-batches_per_epoch:]
            axs[1, 1].plot(recent_batches)
            axs[1, 1].set_title('Batch Losses (Last Epoch)')
            axs[1, 1].set_xlabel('Batch')
            axs[1, 1].set_ylabel('Loss')
            axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'training_results.png'))
        plt.close()
        
        # Also create a standalone figure with a more informative title
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title(f'Training and Validation Loss\n{self.experiment_metadata["model_name"]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, 'loss_curves.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.title(f'Training and Validation Accuracy\n{self.experiment_metadata["model_name"]}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, 'accuracy_curves.png'))
        plt.close()
        
        # Additionally plot batch losses across all epochs
        if len(self.batch_train_losses) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(self.batch_train_losses)
            plt.title('Batch Losses Across All Epochs')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(self.save_path, 'batch_losses.png'))
            plt.close()
        
        # Also plot confusion matrix
        self.plot_confusion_matrix()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix for validation data"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels, *_ in self.val_loader:
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
        
        # Create normalized confusion matrix
        if np.sum(cm) > 0:  # Avoid division by zero
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Normalized Confusion Matrix')
            plt.savefig(os.path.join(self.save_path, 'confusion_matrix_normalized.png'))
            plt.close()
        
        # Also save classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(self.save_path, 'classification_report.csv'))
        
        return cm
