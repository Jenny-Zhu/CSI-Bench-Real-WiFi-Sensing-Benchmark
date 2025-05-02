import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.optim.lr_scheduler import LambdaLR
from engine.base_trainer import BaseTrainer
from tqdm import tqdm

def warmup_schedule(epoch, warmup_epochs):
    """Warmup learning rate schedule."""
    if epoch < warmup_epochs:
        # Linear warmup
        return float(epoch) / float(max(1, warmup_epochs))
    else:
        # Cosine annealing
        return 0.5 * (1.0 + np.cos(np.pi * epoch / warmup_epochs))

class TaskTrainer(BaseTrainer):
    """Trainer for supervised learning tasks with CSI data."""
    
    def __init__(self, model, train_loader, val_loader=None, test_loader=None, criterion=None, optimizer=None, 
                 scheduler=None, device='cuda:0', save_path='./results', checkpoint_path=None, 
                 num_classes=None, label_mapper=None, config=None):
        """
        Initialize the task trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to use
            save_path: Path to save results
            checkpoint_path: Path to load checkpoint
            num_classes: Number of classes for the model
            label_mapper: LabelMapper for mapping between class indices and names
            config: Configuration object with training parameters
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.label_mapper = label_mapper
        self.config = config
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Load checkpoint if specified
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        # Log
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        # Training tracking
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
    
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
        
        # Set default configuration values if config is None
        if self.config is None:
            num_epochs = 30
            patience = 15
        else:
            # Number of epochs and patience from config
            num_epochs = getattr(self.config, 'num_epochs', 30)
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
                # Save the best model
                best_model_path = os.path.join(self.save_path, "best_model.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch,
                }, best_model_path)
                print(f"Best model saved to {best_model_path}")
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
            
            # Handle case where labels might be a tuple
            if isinstance(labels, tuple):
                labels = labels[0]
            
            # Create batch of labels
            batch_size = inputs.size(0)
            
            # Handle case where labels might be strings or scalars
            if isinstance(labels, str):
                try:
                    # Create a tensor of the same value repeated batch_size times
                    label_value = int(labels)
                    labels = torch.tensor([label_value] * batch_size).to(self.device)
                except:
                    labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            elif not hasattr(labels, 'shape') or len(labels.shape) == 0:
                # Handle scalar labels by repeating them
                label_value = int(labels)
                labels = torch.tensor([label_value] * batch_size).to(self.device)
            else:
                # If it's already a batch, just move to device
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
                # Handle case where labels might be a tuple
                if isinstance(labels, tuple):
                    labels = labels[0]
                
                # Create batch of labels
                batch_size = inputs.size(0)
                
                # Handle case where labels might be strings or scalars
                if isinstance(labels, str):
                    try:
                        # Create a tensor of the same value repeated batch_size times
                        label_value = int(labels)
                        labels = torch.tensor([label_value] * batch_size).to(self.device)
                    except:
                        labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
                elif not hasattr(labels, 'shape') or len(labels.shape) == 0:
                    # Handle scalar labels by repeating them
                    label_value = int(labels)
                    labels = torch.tensor([label_value] * batch_size).to(self.device)
                else:
                    # If it's already a batch, just move to device
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
        
        # Also plot confusion matrix
        self.plot_confusion_matrix()
    
    def plot_confusion_matrix(self, data_loader=None, epoch=None, mode='val'):
        """
        Plot the confusion matrix and save the figure.
        
        Args:
            data_loader: Dataloader to use for evaluation
            epoch: Current epoch
            mode: 'val' or 'test' mode
        """
        # Set evaluation mode
        self.model.eval()
        
        # Use validation loader if not specified
        if data_loader is None:
            if mode == 'val' and self.val_loader is not None:
                data_loader = self.val_loader
            elif mode == 'test' and self.test_loader is not None:
                data_loader = self.test_loader
            else:
                raise ValueError(f"No data loader available for mode {mode}")
        
        # Collect all predictions and labels
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Get data and labels
                if isinstance(batch, dict):
                    data = batch['data']
                    labels = batch['labels']
                else:
                    data, labels = batch
                
                # Handle different label formats
                if isinstance(labels, tuple):
                    # Use the first element as class label
                    labels = labels[0]
                
                # Move data to device
                data = data.to(self.device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)
                elif isinstance(labels, (list, np.ndarray)):
                    labels = torch.tensor(labels).to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                _, preds = torch.max(outputs, 1)
                
                # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Get class names if available
        class_names = None
        if self.label_mapper is not None:
            class_names = [self.label_mapper.get_name(i) for i in range(self.num_classes)]
        
        # Plot confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix ({mode})')
        
        # Save figure
        epoch_str = f'_epoch{epoch}' if epoch is not None else ''
        plt.savefig(os.path.join(self.save_path, f'confusion_matrix_{mode}{epoch_str}.png'))
        plt.close()
        
        # Generate and save classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Replace indices with class names if available
        if class_names is not None:
            # Create a mapping dictionary from indices to class names
            index_to_name = {}
            for i, name in enumerate(class_names):
                index_to_name[str(i)] = name
            
            # Replace indices with class names
            new_index = []
            for idx in report_df.index:
                if idx in index_to_name:
                    new_index.append(index_to_name[idx])
                else:
                    new_index.append(idx)
            
            report_df.index = new_index
        
        # Save report
        report_df.to_csv(os.path.join(self.save_path, f'classification_report_{mode}{epoch_str}.csv'))
        
        return report_df

    def calculate_metrics(self, data_loader, epoch=None):
        """
        Calculate overall performance metrics, including weighted F1 score.
        
        Args:
            data_loader: Data loader for evaluation
            epoch: Current epoch (optional)
            
        Returns:
            Tuple of (weighted_f1_score, per_class_f1_scores)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize lists to store predictions and ground truth
        all_preds = []
        all_targets = []
        
        # No gradient during evaluation
        with torch.no_grad():
            for batch in data_loader:
                # Get data and move to device
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    data, targets = batch
                else:
                    # Handle case where batch is a dictionary
                    data = batch['input']
                    targets = batch['target']
                
                # Move to device
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                
                # Append to lists
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Convert lists to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate weighted F1 score
        from sklearn.metrics import f1_score, classification_report
        weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # Calculate per-class F1 scores
        per_class_f1 = f1_score(all_targets, all_preds, average=None)
        
        # Get detailed classification report
        report = classification_report(all_targets, all_preds, output_dict=True)
        
        # Save the report to a CSV file if epoch is None (final evaluation)
        if epoch is None and hasattr(self, 'save_path'):
            import pandas as pd
            # Convert report to DataFrame
            report_df = pd.DataFrame(report).transpose()
            
            # Determine split name from data_loader (assuming it's in the dataloader's dataset attributes)
            split_name = getattr(data_loader.dataset, 'split', 'unknown')
            
            # Save to CSV
            report_path = os.path.join(self.save_path, f'classification_report_{split_name}.csv')
            report_df.to_csv(report_path)
            print(f"Classification report saved to {report_path}")
        
        return weighted_f1, per_class_f1
