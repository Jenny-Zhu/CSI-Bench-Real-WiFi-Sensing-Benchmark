import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import copy
from tqdm import tqdm

def maml_train(model, meta_optimizer, train_loader, val_loader=None, inner_lr=0.01, 
             meta_batch_size=4, n_way=2, k_shot=5, q_query=15, num_iterations=60000,
             meta_validation_interval=1000, device='cuda', save_path='./results'):
    """
    Implement MAML (Model-Agnostic Meta-Learning) training process
    
    Args:
        model: Model to train
        meta_optimizer: Meta optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        inner_lr: Inner loop learning rate
        meta_batch_size: Meta batch size (number of tasks per update)
        n_way: N-way classification
        k_shot: K-shot support set
        q_query: Query set size
        num_iterations: Number of meta-training iterations
        meta_validation_interval: Interval to perform meta-validation
        device: Device to use
        save_path: Path to save model
        
    Returns:
        Trained model and training records
    """
    # Initialize training records
    meta_losses = []
    meta_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    iteration_times = []
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create MAML training loop
    print(f"Starting MAML training: {num_iterations} iterations, validation every {meta_validation_interval}")
    
    # Get iterator
    train_iter = iter(train_loader)
    
    best_val_acc = 0.0
    best_model = None
    
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
        # if isinstance(batch, dict):
        #     # If batch is in dictionary format
        #     support_x = batch['support'][0].to(device)  # [meta_batch_size, n_way*k_shot, ...]
        #     support_y = batch['support'][1].to(device)  # [meta_batch_size, n_way*k_shot]
        #     query_x = batch['query'][0].to(device)      # [meta_batch_size, n_way*q_query, ...]
        #     query_y = batch['query'][1].to(device)      # [meta_batch_size, n_way*q_query]
        # else:
        #     # If batch is in tuple format
        #     support_x, support_y, query_x, query_y = batch
        #     support_x = support_x.to(device)
        #     support_y = support_y.to(device)
        #     query_x = query_x.to(device)
        #     query_y = query_y.to(device)
        
        # # Meta batch size may be smaller than what the loader provides
        # actual_meta_batch_size = min(meta_batch_size, support_x.size(0))
        
        # # Initialize meta loss
        # meta_loss = 0
        # meta_acc = 0
        
        # # Loop through each task in the meta batch
        # for i in range(actual_meta_batch_size):
        #     # Get support and query set for the current task
        #     task_support_x = support_x[i]
        #     task_support_y = support_y[i]
        #     task_query_x = query_x[i]
        #     task_query_y = query_y[i]
            
        #     # Clone model to save initial weights
        #     fast_weights = {name: param.clone() for name, param in model.named_parameters()}
            
        #     # Inner loop optimization - few gradient steps
        #     for _ in range(5):  # Typical MAML uses 1-5 steps
        #         # Forward pass
        #         support_logits = model(task_support_x, params=fast_weights)
        #         support_loss = criterion(support_logits, task_support_y)
                
        #         # Compute gradients
        #         grads = torch.autograd.grad(support_loss, fast_weights.values(),
        #                                   create_graph=True, allow_unused=True)
                
        #         # Update fast weights
        #         fast_weights = {name: param - inner_lr * (grad if grad is not None else 0)
        #                       for ((name, param), grad) in zip(fast_weights.items(), grads)}
            
        #     # Compute loss on query set with updated parameters
        #     query_logits = model(task_query_x, params=fast_weights)
        #     query_loss = criterion(query_logits, task_query_y)
            
        #     # Accumulate meta loss
        #     meta_loss += query_loss
            
        #     # Calculate accuracy
        #     with torch.no_grad():
        #         pred = query_logits.argmax(dim=1)
        #         task_acc = (pred == task_query_y).float().mean().item()
        #         meta_acc += task_acc
        # Inside the maml_train function, update the inner loop adaptation logic

        # For each task in the meta batch
        for task_idx in range(actual_meta_batch_size):
            # Get task data
            task_support_x = support_x[task_idx]
            task_support_y = support_y[task_idx]
            task_query_x = query_x[task_idx]
            task_query_y = query_y[task_idx]
            
            # Inner loop adaptation
            # Create a copy of model parameters for fast adaptation
            fast_weights = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad}
            
            # Inner loop adaptation
            for _ in range(5):  # 5 adaptation steps
                # Forward pass with current fast weights
                # Use model.forward_with_weights or implement a temporary forward pass
                # For each model type, this might need to be customized
                with torch.set_grad_enabled(True):
                    logits = model.forward_with_weights(task_support_x, fast_weights)
                    inner_loss = criterion(logits, task_support_y)
                    
                    # Compute gradients for the adaptation
                    grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=True)
                    
                    # Update fast weights (inner loop update)
                    fast_weights = {name: param - inner_lr * grad 
                                    for (name, param), grad in zip(fast_weights.items(), grads)}
            
            # Evaluate on query set using adapted model (Meta objective)
            query_logits = model.forward_with_weights(task_query_x, fast_weights)
            task_meta_loss = criterion(query_logits, task_query_y)
            
            # Accumulate meta loss
            meta_loss += task_meta_loss / actual_meta_batch_size
            
            # Compute accuracy
            with torch.no_grad():
                pred = query_logits.argmax(dim=1)
                correct = (pred == task_query_y).sum().item()
                total = task_query_y.size(0)
                task_acc = correct / total
                meta_acc += task_acc / actual_meta_batch_size
                # Average meta loss
                meta_loss = meta_loss / actual_meta_batch_size
                meta_acc = meta_acc / actual_meta_batch_size
        
        # Meta optimization step - update model parameters
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
            val_loss, val_acc = meta_validate(model, val_loader, inner_lr, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f'Validation: Loss = {val_loss:.4f}, Accuracy = {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model.state_dict())
                
                # Save model
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': meta_optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss
                }, os.path.join(save_path, 'best_maml_model.pt'))
    
    # Plot training curves
    plot_maml_training_curves(meta_losses, meta_accuracies, val_losses, val_accuracies, 
                             meta_validation_interval, save_path)
    
    # Load best model if available
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Create and return training records
    records = {
        'meta_losses': meta_losses,
        'meta_accuracies': meta_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'iteration_times': iteration_times,
        'best_val_accuracy': best_val_acc
    }
    
    return model, records

def meta_validate(model, val_loader, inner_lr, device, num_tasks=10):
    """
    Perform meta-validation
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        inner_lr: Inner loop learning rate
        device: Device to use
        num_tasks: Number of tasks to validate on
        
    Returns:
        Average loss and accuracy across tasks
    """
    model.eval()
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
            # Clone model weights for adaptation
            fast_weights = {name: param.clone() for name, param in model.named_parameters()}
            
            # Inner loop adaptation
            for _ in range(5):  # 5 adaptation steps
                support_logits = model(support_x[i], params=fast_weights)
                support_loss = criterion(support_logits, support_y[i])
                
                # Compute gradients
                grads = torch.autograd.grad(support_loss, fast_weights.values(),
                                          create_graph=True, allow_unused=True)
                
                # Update fast weights
                fast_weights = {name: param - inner_lr * (grad if grad is not None else 0)
                              for ((name, param), grad) in zip(fast_weights.items(), grads)}
            
            # Evaluate on query set
            with torch.no_grad():
                query_logits = model(query_x[i], params=fast_weights)
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

def plot_maml_training_curves(meta_losses, meta_accuracies, val_losses, val_accuracies, 
                             meta_validation_interval, save_path):
    """
    Plot MAML training curves
    
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
    
    # Plot training loss
    axs[0, 0].plot(meta_losses)
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)
    
    # Plot validation loss
    axs[0, 1].plot(val_losses)
    axs[0, 1].set_title('Validation Loss')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].grid(True)
    
    # Plot training accuracy
    axs[1, 0].plot(meta_accuracies)
    axs[1, 0].set_title('Training Accuracy')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].grid(True)
    
    # Plot validation accuracy
    axs[1, 1].plot(val_accuracies)
    axs[1, 1].set_title('Validation Accuracy')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'maml_training_curves.png'))
    plt.close()
