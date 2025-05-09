#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-Shot Learning Pipeline for WiFi Sensing Benchmark

This script implements a complete pipeline for few-shot learning adaptation:
1. Loads the best pre-trained model for a given task
2. Performs few-shot adaptation to new settings (cross-env, cross-user, cross-device)
3. Tunes the model using k-shot support data from the corresponding support splits
4. Evaluates the adapted model on the corresponding test splits
5. Saves the adapted model and performance metrics

Usage:
    python train_fewshot_pipeline.py --task MotionSourceRecognition 
                                     --model vit
                                     --config configs/local_default_config.json
"""

import os
import sys

# Add project root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import torch
import numpy as np
import pandas as pd
import time
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import hashlib

# Import model and training utilities
from model.fewshot import FewShotAdaptiveModel, FewShotTrainer
from load.supervised.benchmark_loader import load_benchmark_supervised

def find_best_model(results_dir, task, model_type):
    """
    Find the best pre-trained model for a given task and model type
    
    Args:
        results_dir: Directory containing results
        task: Task name (e.g., 'MotionSourceRecognition')
        model_type: Model type (e.g., 'vit', 'transformer')
        
    Returns:
        Path to the best model checkpoint, or None if not found
    """
    # Look in supervised training results directory
    supervised_dir = os.path.join(results_dir, task, model_type)
    print(f"—————————RESULTS:{results_dir}")
    if not os.path.exists(supervised_dir):
        print(f"No supervised training results found for {model_type} on {task}")
        return None
    
    # Look for model checkpoints
    best_acc = -1
    best_model_path = None
    experiment_dirs = [d for d in os.listdir(supervised_dir) 
                      if os.path.isdir(os.path.join(supervised_dir, d))]
    
    for exp_dir in experiment_dirs:
        checkpoint_path = os.path.join(supervised_dir, exp_dir, f"{model_type}_{task}_best.pth")
        # Check if checkpoint exists
        if os.path.exists(checkpoint_path):
            # Look for metrics file to find accuracy
            metrics_path = os.path.join(supervised_dir, exp_dir, "metrics.json")
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    # Get accuracy from metrics
                    val_acc = metrics.get('val_accuracy', -1)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_model_path = checkpoint_path
                except:
                    pass
            
    if best_model_path is None:
        print(f"No valid model checkpoint found for {model_type} on {task}")
    else:
        print(f"Found best model checkpoint with validation accuracy {best_acc:.4f}: {best_model_path}")
        
    return best_model_path

def load_support_data(data_dir, task, support_split, batch_size=32, data_key='CSI_amps'):
    """
    Load support data for few-shot adaptation
    
    Args:
        data_dir: Root directory of the dataset
        task: Task name (e.g., 'MotionSourceRecognition')
        support_split: Support split name (e.g., 'support_cross_env')
        batch_size: Batch size for data loaders
        data_key: Key for CSI data in h5 files
        
    Returns:
        DataLoader for support data
    """
    print(f"Loading {support_split} support data for {task}...")
    
    data = load_benchmark_supervised(
        dataset_root=data_dir,
        task_name=task,
        batch_size=batch_size,
        file_format="h5",
        data_key=data_key,
        num_workers=4,
        shuffle_train=True,
        train_split="train_id",
        val_split=support_split,
        test_splits=[]  # No test splits needed here
    )
    
    # Extract support data loader (val loader in this context)
    support_loader = data['loaders'].get('val')
    num_classes = data['num_classes']
    
    if support_loader is None:
        print(f"Error: Support data loader for {support_split} not found")
        return None, None
    
    return support_loader, num_classes

def load_test_data(data_dir, task, test_split, batch_size=32, data_key='CSI_amps'):
    """
    Load test data for evaluation
    
    Args:
        data_dir: Root directory of the dataset
        task: Task name (e.g., 'MotionSourceRecognition')
        test_split: Test split name (e.g., 'test_cross_env')
        batch_size: Batch size for data loaders
        data_key: Key for CSI data in h5 files
        
    Returns:
        DataLoader for test data
    """
    print(f"Loading {test_split} test data for {task}...")
    
    data = load_benchmark_supervised(
        dataset_root=data_dir,
        task_name=task,
        batch_size=batch_size,
        file_format="h5",
        data_key=data_key,
        num_workers=4,
        shuffle_train=False,
        train_split="train_id",
        val_split="val_id",
        test_splits=[test_split]
    )
    
    # Extract test data loader
    test_loader = data['loaders'].get(f'test_{test_split}')
    
    if test_loader is None:
        print(f"Error: Test data loader for {test_split} not found")
        return None
    
    return test_loader

def adapt_and_evaluate(
    model_path, 
    model_type, 
    task,
    support_loader, 
    test_loader, 
    num_classes,
    adaptation_lr, 
    adaptation_steps, 
    k_shots,
    finetune_all,
    win_len=500,
    feature_size=232,
    save_dir=None,
    adaptation_scenario='cross_env',
    device=None
):
    """
    Adapt a pre-trained model to support data and evaluate on test data
    
    Args:
        model_path: Path to pre-trained model checkpoint
        model_type: Model type (e.g., 'vit', 'transformer')
        task: Task name
        support_loader: DataLoader for support data
        test_loader: DataLoader for test data
        num_classes: Number of classes
        adaptation_lr: Learning rate for adaptation
        adaptation_steps: Number of adaptation steps
        k_shots: Number of examples per class for adaptation
        finetune_all: Whether to fine-tune all parameters
        win_len: Window length for input data
        feature_size: Feature size for input data
        save_dir: Directory to save results
        adaptation_scenario: Scenario name (e.g., 'cross_env', 'cross_user', 'cross_device')
        device: Device to use for computation
        
    Returns:
        Tuple of (adapted_model, results)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create few-shot adaptive model from pre-trained model
    print(f"Loading pre-trained model from {model_path}...")
    fewshot_model = FewShotAdaptiveModel.from_pretrained(
        model_path=model_path,
        model_type=model_type,
        num_classes=num_classes,
        adaptation_lr=adaptation_lr,
        adaptation_steps=adaptation_steps,
        finetune_all=finetune_all,
        device=device,
        win_len=win_len,
        feature_size=feature_size
    )
    
    # Create few-shot trainer
    trainer = FewShotTrainer(
        base_model=fewshot_model,
        support_loader=support_loader,
        query_loader=test_loader,
        adaptation_steps=adaptation_steps,
        adaptation_lr=adaptation_lr,
        device=device,
        save_path=save_dir if save_dir else './results',
        finetune_all=finetune_all
    )
    
    # Evaluate original model
    print("Evaluating original model on test set...")
    original_results = trainer.evaluate()
    
    # Extract k-shot support data
    print(f"Adapting model with {k_shots}-shot learning...")
    
    # Adapt model
    adapted_model = trainer.adapt_model()
    
    # Evaluate adapted model
    print("Evaluating adapted model on test set...")
    adapted_results = trainer.evaluate(model=adapted_model)
    
    # Calculate improvement
    improvement = {
        'accuracy': adapted_results['accuracy'] - original_results['accuracy'],
        'f1_score': adapted_results['f1_score'] - original_results['f1_score']
    }
    
    # Combine results
    results = {
        'original': original_results,
        'adapted': adapted_results,
        'improvement': improvement
    }
    
    # Print results
    print(f"\n=== Results for {adaptation_scenario} adaptation ===")
    print(f"Original accuracy: {original_results['accuracy']:.4f}")
    print(f"Adapted accuracy:  {adapted_results['accuracy']:.4f}")
    print(f"Improvement:      {improvement['accuracy']:+.4f}")
    print(f"Original F1 score: {original_results['f1_score']:.4f}")
    print(f"Adapted F1 score:  {adapted_results['f1_score']:.4f}")
    print(f"Improvement:      {improvement['f1_score']:+.4f}")
    
    return adapted_model, results

def save_results(
    adapted_model, 
    results, 
    config, 
    save_dir, 
    task, 
    model_type, 
    adaptation_scenario
):
    """
    Save adapted model and results
    
    Args:
        adapted_model: Adapted model
        results: Results dictionary
        config: Configuration dictionary
        save_dir: Directory to save results
        task: Task name
        model_type: Model type
        adaptation_scenario: Adaptation scenario name
    """
    # Create timestamp and unique ID for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = f"{model_type}_{task}_{config['adaptation_lr']}_{config['adaptation_steps']}_{config['k_shots']}"
    run_id = hashlib.md5(param_str.encode()).hexdigest()[:10]
    
    # Create save directory
    result_dir = os.path.join(save_dir, task, model_type, f"fewshot_{adaptation_scenario}_{run_id}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Save model checkpoint
    model_path = os.path.join(result_dir, f"{model_type}_{task}_fewshot_{adaptation_scenario}.pth")
    torch.save(adapted_model.state_dict(), model_path)
    print(f"Saved adapted model to {model_path}")
    
    # Save results
    results_path = os.path.join(result_dir, f"fewshot_{adaptation_scenario}_results.json")
    with open(results_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        serializable_results[key][k] = v.tolist()
                    elif isinstance(v, np.float32) or isinstance(v, np.float64):
                        serializable_results[key][k] = float(v)
                    elif isinstance(v, np.int32) or isinstance(v, np.int64):
                        serializable_results[key][k] = int(v)
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=4)
    print(f"Saved results to {results_path}")
    
    # Save configuration
    config_path = os.path.join(result_dir, f"{model_type}_{task}_fewshot_{adaptation_scenario}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to {config_path}")
    
    # Create confusion matrix plot
    original_cm = confusion_matrix(
        results['original']['labels'], 
        results['original']['predictions']
    )
    adapted_cm = confusion_matrix(
        results['adapted']['labels'], 
        results['adapted']['predictions']
    )
    
    # Normalize confusion matrices
    original_cm_norm = original_cm.astype('float') / original_cm.sum(axis=1)[:, np.newaxis]
    adapted_cm_norm = adapted_cm.astype('float') / adapted_cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    im1 = ax1.imshow(original_cm_norm, cmap='Blues')
    ax1.set_title(f"Original Model (Acc: {results['original']['accuracy']:.4f})")
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    
    im2 = ax2.imshow(adapted_cm_norm, cmap='Blues')
    ax2.set_title(f"Adapted Model (Acc: {results['adapted']['accuracy']:.4f})")
    ax2.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"confusion_matrix_{adaptation_scenario}.png"))
    
    return result_dir

def run_fewshot_pipeline(config):
    """
    Run the complete few-shot learning pipeline
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with results for each adaptation scenario
    """
    # Extract parameters from config
    task = config.get('task', 'MotionSourceRecognition')
    model_type = config.get('model', 'vit')
    data_dir = config.get('training_dir', 'wifi_benchmark_dataset')
    results_dir = config.get('output_dir', './results')
    k_shots = config.get('k_shots', 5)
    adaptation_lr = config.get('adaptation_lr', 0.01)
    adaptation_steps = config.get('adaptation_steps', 10)
    finetune_all = config.get('finetune_all', False)
    batch_size = config.get('batch_size', 32)
    win_len = config.get('win_len', 500)
    feature_size = config.get('feature_size', 232)
    data_key = 'CSI_amps'  # Default for WiFi sensing data
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find best pre-trained model
    model_path = find_best_model(results_dir, task, model_type)
    if model_path is None:
        print(f"Error: Could not find a pre-trained model for {model_type} on {task}")
        return None
    
    # Create directory for few-shot results
    fewshot_results_dir = os.path.join(results_dir, 'fewshot')
    os.makedirs(fewshot_results_dir, exist_ok=True)
    
    # Define adaptation scenarios
    adaptation_scenarios = [
        ('cross_env', 'support_cross_env', 'test_cross_env'),
        ('cross_user', 'support_cross_user', 'test_cross_user'),
        ('cross_device', 'support_cross_device', 'test_cross_device')
    ]
    
    # Store results for all scenarios
    all_results = {}
    
    # Process each adaptation scenario
    for scenario, support_split, test_split in adaptation_scenarios:
        print(f"\n=== Running few-shot adaptation for {scenario} scenario ===")
        
        # Load support data
        support_loader, num_classes = load_support_data(
            data_dir=data_dir,
            task=task,
            support_split=support_split,
            batch_size=batch_size,
            data_key=data_key
        )
        
        if support_loader is None:
            print(f"Skipping {scenario} scenario due to missing support data")
            continue
        
        # Load test data
        test_loader = load_test_data(
            data_dir=data_dir,
            task=task,
            test_split=test_split,
            batch_size=batch_size,
            data_key=data_key
        )
        
        if test_loader is None:
            print(f"Skipping {scenario} scenario due to missing test data")
            continue
        
        # Create scenario-specific config
        scenario_config = config.copy()
        scenario_config.update({
            'support_split': support_split,
            'test_split': test_split,
            'adaptation_scenario': scenario,
            'k_shots': k_shots,
            'adaptation_lr': adaptation_lr,
            'adaptation_steps': adaptation_steps,
            'finetune_all': finetune_all
        })
        
        # Adapt and evaluate
        adapted_model, results = adapt_and_evaluate(
            model_path=model_path,
            model_type=model_type,
            task=task,
            support_loader=support_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            adaptation_lr=adaptation_lr,
            adaptation_steps=adaptation_steps,
            k_shots=k_shots,
            finetune_all=finetune_all,
            win_len=win_len,
            feature_size=feature_size,
            save_dir=fewshot_results_dir,
            adaptation_scenario=scenario,
            device=device
        )
        
        # Save results
        result_dir = save_results(
            adapted_model=adapted_model,
            results=results,
            config=scenario_config,
            save_dir=fewshot_results_dir,
            task=task,
            model_type=model_type,
            adaptation_scenario=scenario
        )
        
        # Store results
        all_results[scenario] = {
            'results': results,
            'result_dir': result_dir
        }
    
    # Summarize results across all scenarios
    print("\n=== Few-Shot Adaptation Summary ===")
    print(f"Task: {task}, Model: {model_type}")
    print(f"{'Scenario':<15} {'Original Acc':<15} {'Adapted Acc':<15} {'Improvement':<15}")
    print('-' * 60)
    
    for scenario, data in all_results.items():
        results = data['results']
        orig_acc = results['original']['accuracy']
        adapted_acc = results['adapted']['accuracy']
        improvement = results['improvement']['accuracy']
        print(f"{scenario:<15} {orig_acc:.4f}{'':>9} {adapted_acc:.4f}{'':>9} {improvement:+.4f}{'':>9}")
    
    # Create summary file
    summary_path = os.path.join(fewshot_results_dir, f"{model_type}_{task}_fewshot_summary.json")
    
    summary = {
        'task': task,
        'model': model_type,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'scenarios': {}
    }
    
    for scenario, data in all_results.items():
        results = data['results']
        summary['scenarios'][scenario] = {
            'original_accuracy': float(results['original']['accuracy']),
            'adapted_accuracy': float(results['adapted']['accuracy']),
            'improvement': float(results['improvement']['accuracy']),
            'original_f1_score': float(results['original']['f1_score']),
            'adapted_f1_score': float(results['adapted']['f1_score']),
            'f1_improvement': float(results['improvement']['f1_score']),
            'result_dir': data['result_dir']
        }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nSummary saved to {summary_path}")
    
    return all_results

def load_config(config_path=None):
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        'task': 'MotionSourceRecognition',
        'model': 'vit',
        'training_dir': 'wifi_benchmark_dataset',
        'output_dir': './results',
        'k_shots': 5,
        'adaptation_lr': 0.01,
        'adaptation_steps': 10,
        'finetune_all': False,
        'batch_size': 32,
        'win_len': 500,
        'feature_size': 232
    }
    
    # If no config path provided, use defaults
    if config_path is None:
        return default_config
    
    # Load configuration from file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Extract few-shot specific configuration
        if 'fewshot' in config:
            fewshot_config = config['fewshot']
            config['k_shots'] = fewshot_config.get('k_shots', 5)
            config['adaptation_lr'] = fewshot_config.get('adaptation_lr', 0.01)
            config['adaptation_steps'] = fewshot_config.get('adaptation_steps', 10)
            config['finetune_all'] = fewshot_config.get('finetune_all', False)
        
        # Merge with defaults to ensure all required parameters are present
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading configuration from {config_path}: {e}")
        print("Using default configuration")
        return default_config

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Few-shot learning pipeline for WiFi sensing')
    
    # Basic parameters
    parser.add_argument('--task', type=str, default=None,
                        help='Task name (e.g., MotionSourceRecognition)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model type (e.g., vit, transformer)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    
    # Override parameters
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing the dataset')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--k_shots', type=int, default=None,
                        help='Number of examples per class for few-shot adaptation')
    parser.add_argument('--adaptation_lr', type=float, default=None,
                        help='Learning rate for few-shot adaptation')
    parser.add_argument('--adaptation_steps', type=int, default=None,
                        help='Number of adaptation steps for few-shot learning')
    parser.add_argument('--finetune_all', action='store_true',
                        help='Fine-tune all model parameters instead of just the classifier')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for data loaders')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with command-line arguments
    if args.task is not None:
        config['task'] = args.task
    if args.model is not None:
        config['model'] = args.model
    if args.data_dir is not None:
        config['training_dir'] = args.data_dir
    if args.results_dir is not None:
        config['output_dir'] = args.results_dir
    if args.k_shots is not None:
        config['k_shots'] = args.k_shots
    if args.adaptation_lr is not None:
        config['adaptation_lr'] = args.adaptation_lr
    if args.adaptation_steps is not None:
        config['adaptation_steps'] = args.adaptation_steps
    if args.finetune_all:
        config['finetune_all'] = True
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    # Print configuration
    print("Running few-shot learning pipeline with the following configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run pipeline
    run_fewshot_pipeline(config)

if __name__ == '__main__':
    main() 