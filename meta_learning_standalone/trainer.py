"""
Standalone trainer for meta-learning.
This module provides training utilities without circular imports.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

# Define the MAMLTrainer class directly here
class MAMLTrainer:
    def __init__(self, model, device, inner_lr=0.01, meta_lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, task_batch):
        """Perform a single meta-training step"""
        # Implementation moved to train_meta_standalone.py
        pass
    
    def evaluate(self, data_loader, num_adaptation_steps=5):
        """Evaluate the model on the given data loader"""
        # Implementation moved to train_meta_standalone.py
        pass
