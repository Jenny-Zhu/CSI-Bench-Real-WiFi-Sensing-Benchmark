import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import copy
from tqdm import tqdm
from engine.meta_learning.lstm_optimizer import LSTMOptimizer, CoordinateWiseLSTMOptimizer

def lstm_meta_train(model, meta_optimizer, train_loader, val_loader=None, lstm_optimizer=None,
                  meta_batch_size=4, n_way=2, k_shot=5, q_query=15, num_iterations=60000,
                  meta_validation_interval=1000, device='cuda', save_path='./results'):
    """
    LSTM Meta-Learner training process based on "Optimization as a Model for Few-Shot Learning"
    
    Args:
        model: Model to train
        meta_optimizer: Optimizer for the LSTM optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        lstm_optimizer: LSTM optimizer instance (will be created if None)
        meta_batch_size: Meta batch size (number of tasks per update)
        n_way: N-way classification
        k_shot: K-shot support set
        q_query: Query set size
        num_iterations: Number of meta-training iterations
        meta_validation_interval: Interval to perform meta-validation
        device: Device to use
        save_path: Path to save model
        
    Returns:
        Trained model, LSTM optimizer and training records
    """
    # Initialize training records
    meta_losses = []
    meta_accuracies = []
    val_losses = []
    val_accuracies = []
    iteration_times = []
    
    # Initialize LSTM optimizer if not provided
    if lstm_optimizer is None:
        # Use the default hidden size
        hidden_size = 20
        lstm_optimizer = CoordinateWiseLSTMOptimizer(hidden_size=hidden_size)
        lstm_optimizer.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Get iterator
    train_iter = iter(train_loader)
    
    best_val_acc = 0.0
    best_model = None
    best_lstm_optimizer = None
    
    print(f"Starting LSTM meta-training: {num_iterations} iterations, validation every {meta_validation_interval}")
    
    # Training loop
    for iteration in tqdm(range(num_iterations)):
        start_time = time.time()
        
        # Reset iterator if it's exhausted
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Extract support and query sets
        if isinstance(batch, dict):
            # If batch is in dictionary format
            support_x = batch['support'][0].to(device)  # [meta_batch_size, n_way*k_shot, ...]
            support_y = batch['support'][1].to(device)  # [meta_batch_size, n_way*k_shot]
            query_x = batch['query'][0].to(device)      # [meta_batch_size, n_way*q_query, ...]
            query_y = batch['query'][1].to(device)      # [meta_batch_size, n_way*q_query]
        else:
            # If batch is in tuple format
            support_x, support_y, query_x, query_y = batch
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)
        
        # Meta batch size may be smaller than what the loader provides
        actual_meta_batch_size = min(meta_batch_size, support_x.size(0))
        
        # Initialize meta loss
        meta_loss = 0
        meta_acc = 0
        
        # Loop through each task in the meta batch
        for i in range(actual_meta_batch_size):
            # Get support and query set for the current task
            task_support_x = support_x[i]
            task_support_y = support_y[i]
            task_query_x = query_x[i]
            task_query_y = query_y[i]
            
            # Clone model for this task
            task_model = copy.deepcopy(model)
            
            # Reset LSTM optimizer state
            lstm_optimizer.reset_state()
            
            # Inner loop optimization
            for _ in range(5):  # Typically 5 update steps for few-shot learning
                # Forward pass on support set
                support_logits = task_model(task_support_x)
                support_loss = criterion(support_logits, task_support_y)
                
                # Compute gradients
                support_loss.backward()
                
                # Get gradients and parameters
                gradients = {}
                parameters = {}
                
                for name, param in task_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.data.clone()
                        parameters[name] = param.data.clone()
                
                # Get parameter updates from LSTM optimizer
                updates = lstm_optimizer(parameters, gradients)
                
                # Apply updates and reset gradients
                for name, param in task_model.named_parameters():
                    if name in updates:
                        param.data = param.data + updates[name]
                    param.grad = None
            
            # Compute loss on query set with adapted model
            with torch.no_grad():
                query_logits = task_model(task_query_x)
                query_loss = criterion(query_logits, task_query_y)
                
                # Calculate accuracy
                pred = query_logits.argmax(dim=1)
                task_acc = (pred == task_query_y).float().mean().item()
            
            # Accumulate meta loss and accuracy
            meta_loss += query_loss
            meta_acc += task_acc
        
        # Average meta loss and accuracy
        meta_loss = meta_loss / actual_meta_batch_size
        meta_acc = meta_acc / actual_meta_batch_size
        
        # Update LSTM optimizer
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
        
        # Record results
        meta_losses.append(meta_loss.item())
        meta_accuracies.append(meta_acc)
        iteration_time = time.time() - start_time
        iteration_times.append(iteration_time)
        
        # Print progress
        if (iteration + 1) % 100 == 0:
            print(f'Iteration {iteration+1}: Meta Loss = {meta_loss.item():.4f}, '
                  f'Meta Accuracy = {meta_acc:.4f}, Time = {iteration_time:.2f}s')
        
        # Periodic meta validation
        if val_loader is not None and (iteration + 1) % meta_validation_interval == 0:
            val_loss, val_acc = lstm_meta_validate(model, lstm_optimizer, val_loader, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f'Validation: Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model.state_dict())
                best_lstm_optimizer = copy.deepcopy(lstm_optimizer.state_dict())
                
                # Save models
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'lstm_optimizer_state_dict': lstm_optimizer.state_dict(),
                    'meta_optimizer_state_dict': meta_optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss
                }, os.path.join(save_path, 'best_lstm_meta_model.pt'))
    
    # Plot training curves
    plot_lstm_training_curves(meta_losses, meta_accuracies, val_losses, val_accuracies, 
                             meta_validation_interval, save_path)
    
    # Load best model if available
    if best_model is not None:
        model.load_state_dict(best_model)
        lstm_optimizer.load_state_dict(best_lstm_optimizer)
    
    # Create and return training records
    records = {
        'meta_losses': meta_losses,
        'meta_accuracies': meta_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'iteration_times': iteration_times,
        'best_val_accuracy': best_val_acc
    }
    
    return model, lstm_optimizer, records

def lstm_meta_validate(model, lstm_optimizer, val_loader, device, num_tasks=10):
    """
    Perform LSTM meta-validation
    
    Args:
        model: Model to validate
        lstm_optimizer: LSTM optimizer
        val_loader: Validation data loader
        device: Device to use
        num_tasks: Number of tasks to validate on
        
    Returns:
        Average loss and accuracy across tasks
    """
    model.eval()
    lstm_optimizer.eval()
    
    val_losses = []
    val_accuracies = []
    criterion = nn.CrossEntropyLoss()
    
    # Create iterator
    val_iter = iter(val_loader)
    
    # Validate on num_tasks tasks
    for _ in range(num_tasks):
        # Get next batch
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)
        
        # Extract support and query sets
        if isinstance(batch, dict):
            support_x = batch['support'][0].to(device)
            support_y = batch['support'][1].to(device)
            query_x = batch['query'][0].to(device)
            query_y = batch['query'][1].to(device)
        else:
            support_x, support_y, query_x, query_y = batch
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)
        
        # Handle case where validation loader doesn't provide meta batches
        if len(support_x.shape) < 3:  # Not batched
            support_x = support_x.unsqueeze(0)
            support_y = support_y.unsqueeze(0)
            query_x = query_x.unsqueeze(0)
            query_y = query_y.unsqueeze(0)
        
        # Process each meta batch
        for i in range(support_x.size(0)):
            # Clone model for this task
            task_model = copy.deepcopy(model)
            
            # Reset LSTM optimizer state
            lstm_optimizer.reset_state()
            
            # Inner loop adaptation
            for _ in range(5):  # 5 adaptation steps
                # Forward pass on support set
                support_logits = task_model(support_x[i])
                support_loss = criterion(support_logits, support_y[i])
                
                # Compute gradients
                support_loss.backward()
                
                # Get gradients and parameters
                gradients = {}
                parameters = {}
                
                for name, param in task_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.data.clone()
                        parameters[name] = param.data.clone()
                
                # Get parameter updates from LSTM optimizer
                with torch.no_grad():
                    updates = lstm_optimizer(parameters, gradients)
                
                # Apply updates and reset gradients
                for name, param in task_model.named_parameters():
                    if name in updates:
                        param.data = param.data + updates[name]
                    param.grad = None
            
            # Evaluate on query set
            with torch.no_grad():
                query_logits = task_model(query_x[i])
                query_loss = criterion(query_logits, query_y[i])
                
                # Calculate accuracy
                pred = query_logits.argmax(dim=1)
                task_acc = (pred == query_y[i]).float().mean().item()
                
                val_losses.append(query_loss.item())
                val_accuracies.append(task_acc)
    
    # Calculate averages
    avg_loss = np.mean(val_losses)
    avg_acc = np.mean(val_accuracies)
    
    return avg_loss, avg_acc

def plot_lstm_training_curves(meta_losses, meta_accuracies, val_losses, val_accuracies, 
                             meta_validation_interval, save_path):
    """
    Plot LSTM meta-training curves
    
    Args:
        meta_losses: List of meta losses
        meta_accuracies: List of meta accuracies
        val_losses: List of validation losses
        val_accuracies: List of validation accuracies
        meta_validation_interval: Interval to perform meta-validation
        save_path: Path to save plot
    """
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Meta loss subplot
    axs[0, 0].plot(meta_losses)
    axs[0, 0].set_title('Meta Training Loss')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)
    
    # Meta accuracy subplot
    axs[0, 1].plot(meta_accuracies)
    axs[0, 1].set_title('Meta Training Accuracy')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].grid(True)
    
    # Validation iterations
    if len(val_losses) > 0:
        val_iterations = np.arange(meta_validation_interval, 
                                  meta_validation_interval * (len(val_losses) + 1), 
                                  meta_validation_interval)
        
        # Validation loss subplot
        axs[1, 0].plot(val_iterations, val_losses)
        axs[1, 0].set_title('Validation Loss')
        axs[1, 0].set_xlabel('Iteration')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].grid(True)
        
        # Validation accuracy subplot
        axs[1, 1].plot(val_iterations, val_accuracies)
        axs[1, 1].set_title('Validation Accuracy')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Accuracy')
        axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'lstm_meta_training_curves.png'))
    plt.close()
