#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-Shot Adapter module for WiFi sensing models.
This module provides a class for adapting pre-trained models to new environments
with only a few examples.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

class FewShotAdapter:
    """
    A class to adapt pre-trained models to new environments with few-shot learning.
    
    This adapter applies gradient-based meta-learning to quickly adapt a model
    to new domains using only a few examples (shots).
    
    Args:
        model (nn.Module): The pre-trained model to adapt
        device (torch.device): Device to use for computation
        inner_lr (float): Learning rate for the adaptation process
        num_inner_steps (int): Number of gradient steps for adaptation
        k_shot (int): Number of examples per class for adaptation
    """
    
    def __init__(self, model, device, inner_lr=0.01, num_inner_steps=10, k_shot=5):
        self.model = model
        self.device = device
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.k_shot = k_shot
        self.criterion = nn.CrossEntropyLoss()
        
    def prepare_support_query_data(self, dataloader, k_shot=None):
        """
        Prepare support and query sets from a dataloader.
        
        The support set contains k examples per class for adaptation,
        while the query set contains the remaining examples for evaluation.
        
        Args:
            dataloader: DataLoader containing the full dataset
            k_shot (int, optional): Number of examples per class for the support set
        
        Returns:
            tuple: (support_inputs, support_labels, query_inputs, query_labels)
        """
        if k_shot is None:
            k_shot = self.k_shot
            
        # Extract all data from the dataloader
        all_inputs = []
        all_labels = []
        
        for inputs, labels in dataloader:
            all_inputs.append(inputs)
            all_labels.append(labels)
            
        # Concatenate all batches
        all_inputs = torch.cat(all_inputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Get unique classes
        unique_classes = torch.unique(all_labels).cpu().numpy()
        num_classes = len(unique_classes)
        
        # Prepare support and query sets
        support_indices = []
        query_indices = []
        
        for cls in unique_classes:
            # Get indices of examples belonging to this class
            cls_indices = torch.where(all_labels == cls)[0]
            
            # If there are fewer examples than k_shot, use all of them
            if len(cls_indices) <= k_shot:
                support_indices.append(cls_indices)
                # No query examples for this class
            else:
                # Randomly select k examples for support set
                perm = torch.randperm(len(cls_indices))
                support_indices.append(cls_indices[perm[:k_shot]])
                query_indices.append(cls_indices[perm[k_shot:]])
        
        # Concatenate indices for all classes
        support_indices = torch.cat(support_indices)
        
        # Only concatenate query indices if there are any
        if query_indices:
            query_indices = torch.cat(query_indices)
            
            # Extract support and query data
            support_inputs = all_inputs[support_indices].to(self.device)
            support_labels = all_labels[support_indices].to(self.device)
            query_inputs = all_inputs[query_indices].to(self.device)
            query_labels = all_labels[query_indices].to(self.device)
            
            return support_inputs, support_labels, query_inputs, query_labels
        else:
            # If no query examples, return only support data
            support_inputs = all_inputs[support_indices].to(self.device)
            support_labels = all_labels[support_indices].to(self.device)
            
            return support_inputs, support_labels, None, None
    
    def adapt(self, support_inputs, support_labels, clone_model=True):
        """
        Adapt the model to the support set with gradient-based meta-learning.
        
        Args:
            support_inputs (torch.Tensor): Input data for adaptation
            support_labels (torch.Tensor): Labels for adaptation
            clone_model (bool): Whether to clone the model before adaptation
            
        Returns:
            nn.Module: The adapted model
        """
        # Clone the model if requested
        if clone_model:
            adapted_model = self._clone_model(self.model)
        else:
            adapted_model = self.model
            
        # Set model to training mode
        adapted_model.train()
        
        # Create optimizer for adaptation
        optimizer = optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        
        # Adapt the model with gradient steps
        for step in range(self.num_inner_steps):
            # Forward pass
            outputs = adapted_model(support_inputs)
            loss = self.criterion(outputs, support_labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Set model back to evaluation mode
        adapted_model.eval()
        
        return adapted_model
    
    def evaluate(self, model, inputs, labels):
        """
        Evaluate a model on the given inputs and labels.
        
        Args:
            model (nn.Module): Model to evaluate
            inputs (torch.Tensor): Input data
            labels (torch.Tensor): Ground truth labels
            
        Returns:
            dict: Dictionary containing accuracy, f1_score, and predictions
        """
        # Ensure model is in evaluation mode
        model.eval()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
        # Convert to numpy for metrics calculation
        predictions_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Calculate metrics
        acc = accuracy_score(labels_np, predictions_np)
        f1 = f1_score(labels_np, predictions_np, average='weighted')
        
        return {
            'accuracy': acc,
            'f1_score': f1,
            'predictions': predictions_np,
            'labels': labels_np
        }
    
    def adapt_and_evaluate(self, dataloader, k_shot=None, save_path=None):
        """
        Adapt the model to a new environment and evaluate performance before and after adaptation.
        
        Args:
            dataloader: DataLoader containing the new environment data
            k_shot (int, optional): Number of examples per class for adaptation
            save_path (str, optional): Path to save results
            
        Returns:
            dict: Dictionary containing original and adapted performance, and improvement
        """
        if k_shot is None:
            k_shot = self.k_shot
            
        # Prepare support and query data
        support_inputs, support_labels, query_inputs, query_labels = self.prepare_support_query_data(dataloader, k_shot)
        
        # If no query data available, we cannot evaluate
        if query_inputs is None or query_labels is None:
            print("Warning: Not enough data for evaluation after creating support set")
            return None
            
        # Evaluate original model
        original_results = self.evaluate(self.model, query_inputs, query_labels)
        
        # Adapt model and evaluate
        adapted_model = self.adapt(support_inputs, support_labels)
        adapted_results = self.evaluate(adapted_model, query_inputs, query_labels)
        
        # Calculate improvement
        improvement = {
            'accuracy': adapted_results['accuracy'] - original_results['accuracy'],
            'f1_score': adapted_results['f1_score'] - original_results['f1_score']
        }
        
        # Save confusion matrices if save_path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            # Original model confusion matrix
            cm_orig = confusion_matrix(original_results['labels'], original_results['predictions'])
            plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_orig)
            disp.plot()
            plt.title(f"Original Model Confusion Matrix (Accuracy: {original_results['accuracy']:.4f})")
            plt.savefig(os.path.join(save_path, 'original_confusion_matrix.png'))
            plt.close()
            
            # Adapted model confusion matrix
            cm_adapted = confusion_matrix(adapted_results['labels'], adapted_results['predictions'])
            plt.figure(figsize=(10, 8))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_adapted)
            disp.plot()
            plt.title(f"Adapted Model Confusion Matrix (Accuracy: {adapted_results['accuracy']:.4f})")
            plt.savefig(os.path.join(save_path, 'adapted_confusion_matrix.png'))
            plt.close()
        
        # Return all results
        return {
            'original': original_results,
            'adapted': adapted_results,
            'improvement': improvement,
            'k_shot': k_shot
        }
    
    def evaluate_k_shots(self, dataloader, k_shots_list=[1, 3, 5, 10], save_path=None):
        """
        Evaluate performance with different numbers of shots (k).
        
        Args:
            dataloader: DataLoader containing the new environment data
            k_shots_list (list): List of k values to evaluate
            save_path (str, optional): Path to save results
            
        Returns:
            dict: Dictionary mapping k values to performance metrics
        """
        results = {}
        
        # Get original performance (0-shot)
        all_inputs = []
        all_labels = []
        
        for inputs, labels in dataloader:
            all_inputs.append(inputs)
            all_labels.append(labels)
            
        all_inputs = torch.cat(all_inputs, dim=0).to(self.device)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)
        
        orig_results = self.evaluate(self.model, all_inputs, all_labels)
        results['0-shot'] = orig_results
        
        # Evaluate for each k value
        for k in k_shots_list:
            print(f"Evaluating with {k}-shot learning...")
            k_result = self.adapt_and_evaluate(dataloader, k_shot=k)
            if k_result is not None:
                results[f'{k}-shot'] = k_result
        
        # Save plot if save_path is provided
        if save_path:
            self._plot_k_shot_results(results, save_path)
        
        return results
    
    def _clone_model(self, model):
        """Create a copy of the model with the same parameters."""
        clone = type(model)(**model.get_init_params())
        clone.load_state_dict(model.state_dict())
        clone.to(self.device)
        return clone
    
    def _plot_k_shot_results(self, results, save_path):
        """Plot accuracy and F1-score for different k values."""
        k_values = ['0-shot'] + [f'{k}-shot' for k in results.keys() if k != '0-shot' and isinstance(k, str)]
        
        accuracies = []
        f1_scores = []
        
        for k in k_values:
            if k == '0-shot':
                accuracies.append(results[k]['accuracy'])
                f1_scores.append(results[k]['f1_score'])
            else:
                accuracies.append(results[k]['adapted']['accuracy'])
                f1_scores.append(results[k]['adapted']['f1_score'])
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(k_values, accuracies, 'o-', label='Accuracy')
        plt.xlabel('Number of Shots')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Number of Shots')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(k_values, f1_scores, 'o-', label='F1-score')
        plt.xlabel('Number of Shots')
        plt.ylabel('F1-score')
        plt.title('F1-score vs. Number of Shots')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'k_shot_performance.png'))
        plt.close() 