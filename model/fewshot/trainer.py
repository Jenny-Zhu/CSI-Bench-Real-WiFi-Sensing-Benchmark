import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.metrics import confusion_matrix, classification_report, f1_score

class FewShotTrainer:
    """
    Trainer for few-shot learning adaptation of pre-trained models.
    """
    
    def __init__(
        self,
        base_model,
        support_loader=None,
        query_loader=None,
        adaptation_steps=10,
        adaptation_lr=0.01,
        device=None,
        save_path='./results',
        finetune_all=False
    ):
        """
        Initialize the few-shot trainer.
        
        Args:
            base_model: Pre-trained model to adapt
            support_loader: DataLoader for support set (few-shot examples)
            query_loader: DataLoader for query set (test examples)
            adaptation_steps: Number of steps for adaptation
            adaptation_lr: Learning rate for adaptation
            device: Device to use for training
            save_path: Path to save results
            finetune_all: Whether to fine-tune all parameters or just the classifier
        """
        self.base_model = base_model
        self.support_loader = support_loader
        self.query_loader = query_loader
        self.adaptation_steps = adaptation_steps
        self.adaptation_lr = adaptation_lr
        self.finetune_all = finetune_all
        self.save_path = save_path
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Move model to device
        self.base_model.to(self.device)
        
        # Create directories if needed
        os.makedirs(save_path, exist_ok=True)
        
    def collate_support_set(self, support_loader=None):
        """
        Collect all samples from a support set into single tensors.
        
        Args:
            support_loader: Support set data loader
            
        Returns:
            Tuple of (support_x, support_y)
        """
        if support_loader is None:
            support_loader = self.support_loader
            
        if support_loader is None:
            raise ValueError("No support loader provided")
            
        # Collect all data from support loader
        support_x = []
        support_y = []
        
        for inputs, labels in support_loader:
            # Handle different label formats
            if isinstance(labels, tuple):
                labels = labels[0]  # Use first element if it's a tuple
                
            support_x.append(inputs)
            support_y.append(labels)
            
        # Concatenate all batches
        support_x = torch.cat(support_x, dim=0)
        support_y = torch.cat(support_y, dim=0)
        
        return support_x, support_y
        
    def adapt_model(self, adapted_model=None, support_loader=None, criterion=None):
        """
        Adapt the model to a support set.
        
        Args:
            adapted_model: Model to adapt (defaults to base_model)
            support_loader: DataLoader for support set
            criterion: Loss function for adaptation
            
        Returns:
            Adapted model
        """
        if adapted_model is None:
            adapted_model = self.base_model
            
        if support_loader is None:
            support_loader = self.support_loader
            
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        # Collect all support set data
        support_x, support_y = self.collate_support_set(support_loader)
        
        # Determine which parameters to update during adaptation
        if self.finetune_all:
            # Fine-tune the entire model
            params_to_update = adapted_model.parameters()
        else:
            # Only fine-tune the classifier
            params_to_update = adapted_model.classifier.parameters()
            
        # Create optimizer for adaptation
        optimizer = torch.optim.Adam(params_to_update, lr=self.adaptation_lr)
        
        # Set model to training mode
        adapted_model.train()
        
        # Perform adaptation steps
        for step in range(self.adaptation_steps):
            # Forward pass
            logits = adapted_model(support_x.to(self.device))
            loss = criterion(logits, support_y.to(self.device))
            
            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Set model back to evaluation mode
        adapted_model.eval()
        
        return adapted_model
        
    def evaluate(self, model=None, data_loader=None):
        """
        Evaluate the model on a data loader.
        
        Args:
            model: Model to evaluate (defaults to base_model)
            data_loader: DataLoader to evaluate on (defaults to query_loader)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if model is None:
            model = self.base_model
            
        if data_loader is None:
            data_loader = self.query_loader
            
        if data_loader is None:
            raise ValueError("No data loader provided for evaluation")
            
        # Set model to evaluation mode
        model.eval()
        
        # Collect predictions and labels
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                # Handle different label formats
                if isinstance(labels, tuple):
                    labels = labels[0]  # Use first element if it's a tuple
                    
                # Forward pass
                outputs = model(inputs.to(self.device))
                _, preds = torch.max(outputs, 1)
                
                # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = np.mean(all_preds == all_labels)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Create detailed classification report
        report = classification_report(all_labels, all_preds, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'report': report
        }
        
    def compare_with_without_adaptation(self, save_results=True):
        """
        Compare model performance with and without few-shot adaptation.
        
        Args:
            save_results: Whether to save the results
            
        Returns:
            Dictionary with comparison results
        """
        # Evaluate base model without adaptation
        print("Evaluating model without adaptation...")
        base_results = self.evaluate(model=self.base_model)
        
        # Create a copy of the model for adaptation
        adapted_model = copy.deepcopy(self.base_model)
        
        # Adapt the model
        print(f"Adapting model with {self.adaptation_steps} steps...")
        adapted_model = self.adapt_model(adapted_model=adapted_model)
        
        # Evaluate adapted model
        print("Evaluating adapted model...")
        adapted_results = self.evaluate(model=adapted_model)
        
        # Compare results
        comparison = {
            'base_accuracy': base_results['accuracy'],
            'adapted_accuracy': adapted_results['accuracy'],
            'base_f1_score': base_results['f1_score'],
            'adapted_f1_score': adapted_results['f1_score'],
            'accuracy_improvement': adapted_results['accuracy'] - base_results['accuracy'],
            'f1_improvement': adapted_results['f1_score'] - base_results['f1_score']
        }
        
        print("\nComparison Results:")
        print(f"Without adaptation - Accuracy: {comparison['base_accuracy']:.4f}, F1-score: {comparison['base_f1_score']:.4f}")
        print(f"With adaptation    - Accuracy: {comparison['adapted_accuracy']:.4f}, F1-score: {comparison['adapted_f1_score']:.4f}")
        print(f"Improvement        - Accuracy: {comparison['accuracy_improvement']:.4f}, F1-score: {comparison['f1_improvement']:.4f}")
        
        # Save results if requested
        if save_results:
            # Save comparison summary
            summary_file = os.path.join(self.save_path, 'fewshot_comparison_summary.json')
            import json
            with open(summary_file, 'w') as f:
                json.dump(comparison, f, indent=4)
                
            # Plot confusion matrices
            self.plot_confusion_matrices(
                base_preds=base_results['predictions'],
                base_labels=base_results['labels'],
                adapted_preds=adapted_results['predictions'],
                adapted_labels=adapted_results['labels']
            )
            
        return {
            'base_results': base_results,
            'adapted_results': adapted_results,
            'comparison': comparison
        }
        
    def plot_confusion_matrices(self, base_preds, base_labels, adapted_preds, adapted_labels):
        """
        Plot confusion matrices for base and adapted models.
        
        Args:
            base_preds: Predictions from the base model
            base_labels: True labels for the base model
            adapted_preds: Predictions from the adapted model
            adapted_labels: True labels for the adapted model
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot confusion matrix for base model
        cm_base = confusion_matrix(base_labels, base_preds)
        sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title('Without Few-Shot Adaptation')
        
        # Plot confusion matrix for adapted model
        cm_adapted = confusion_matrix(adapted_labels, adapted_preds)
        sns.heatmap(cm_adapted, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title('With Few-Shot Adaptation')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.save_path, 'fewshot_confusion_matrices.png'))
        plt.close()
        
    def evaluate_support_set_sizes(self, k_shots_list=[1, 3, 5, 10], save_results=True):
        """
        Evaluate the effect of different support set sizes on adaptation.
        
        Args:
            k_shots_list: List of k-shot values to evaluate
            save_results: Whether to save the results
            
        Returns:
            Dictionary with results for different k-shot values
        """
        # Get all support data
        full_support_x, full_support_y = self.collate_support_set()
        
        # Get unique labels
        unique_labels = torch.unique(full_support_y).cpu().numpy()
        num_classes = len(unique_labels)
        
        # Results dictionary
        k_shot_results = {}
        
        # Evaluate base model without adaptation
        base_results = self.evaluate(model=self.base_model)
        
        # For each k-shot value
        for k in k_shots_list:
            print(f"\nEvaluating with {k}-shot adaptation...")
            
            # Create a balanced subset of the support set
            support_x_subset = []
            support_y_subset = []
            
            for label in unique_labels:
                # Find indices of samples with this label
                indices = torch.where(full_support_y == label)[0]
                
                # Take k samples (or all if less than k) 
                k_indices = indices[:min(k, len(indices))]
                
                # Add to subset
                support_x_subset.append(full_support_x[k_indices])
                support_y_subset.append(full_support_y[k_indices])
                
            # Concatenate across classes
            support_x_subset = torch.cat(support_x_subset, dim=0)
            support_y_subset = torch.cat(support_y_subset, dim=0)
            
            # Create a copy of the model for adaptation
            adapted_model = copy.deepcopy(self.base_model)
            
            # Adapt the model using the k-shot subset
            adapted_model.train()
            params_to_update = (
                adapted_model.parameters() if self.finetune_all 
                else adapted_model.classifier.parameters()
            )
            optimizer = torch.optim.Adam(params_to_update, lr=self.adaptation_lr)
            criterion = nn.CrossEntropyLoss()
            
            for step in range(self.adaptation_steps):
                # Forward pass
                logits = adapted_model(support_x_subset.to(self.device))
                loss = criterion(logits, support_y_subset.to(self.device))
                
                # Backward pass and update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Evaluate adapted model
            adapted_model.eval()
            adapted_results = self.evaluate(model=adapted_model)
            
            # Store results
            k_shot_results[f'{k}-shot'] = {
                'accuracy': adapted_results['accuracy'],
                'f1_score': adapted_results['f1_score'],
                'accuracy_improvement': adapted_results['accuracy'] - base_results['accuracy'],
                'f1_improvement': adapted_results['f1_score'] - base_results['f1_score']
            }
            
            print(f"{k}-shot - Accuracy: {adapted_results['accuracy']:.4f}, F1-score: {adapted_results['f1_score']:.4f}")
            print(f"Improvement - Accuracy: {k_shot_results[f'{k}-shot']['accuracy_improvement']:.4f}, F1-score: {k_shot_results[f'{k}-shot']['f1_improvement']:.4f}")
        
        # Add base results (0-shot)
        k_shot_results['0-shot'] = {
            'accuracy': base_results['accuracy'],
            'f1_score': base_results['f1_score'],
            'accuracy_improvement': 0.0,
            'f1_improvement': 0.0
        }
        
        # Save results if requested
        if save_results:
            # Save comparison summary
            summary_file = os.path.join(self.save_path, 'fewshot_kshot_results.json')
            import json
            with open(summary_file, 'w') as f:
                json.dump(k_shot_results, f, indent=4)
                
            # Plot k-shot performance
            self.plot_kshot_performance(k_shot_results, k_shots_list)
            
        return k_shot_results
        
    def plot_kshot_performance(self, k_shot_results, k_shots_list):
        """
        Plot performance metrics for different k-shot values.
        
        Args:
            k_shot_results: Results dictionary from evaluate_support_set_sizes
            k_shots_list: List of k-shot values that were evaluated
        """
        # Include 0-shot in k_shots_list
        x_values = [0] + k_shots_list
        
        # Extract metrics for each k-shot value
        accuracies = [k_shot_results['0-shot']['accuracy']]
        f1_scores = [k_shot_results['0-shot']['f1_score']]
        
        for k in k_shots_list:
            accuracies.append(k_shot_results[f'{k}-shot']['accuracy'])
            f1_scores.append(k_shot_results[f'{k}-shot']['f1_score'])
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot accuracy
        ax1.plot(x_values, accuracies, marker='o', linestyle='-', linewidth=2)
        ax1.set_xlabel('Number of shots (k)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs. Number of Shots')
        ax1.grid(True)
        
        # Plot F1-score
        ax2.plot(x_values, f1_scores, marker='o', linestyle='-', linewidth=2, color='orange')
        ax2.set_xlabel('Number of shots (k)')
        ax2.set_ylabel('F1-score')
        ax2.set_title('F1-score vs. Number of Shots')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.save_path, 'fewshot_kshot_performance.png'))
        plt.close() 