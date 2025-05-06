#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd  # 确保导入pandas
from load.supervised.benchmark_loader import load_benchmark_supervised
from tqdm import tqdm
import json
import hashlib
import time

# Import model classes from models.py
from model.supervised.models import (
    MLPClassifier, 
    LSTMClassifier, 
    ResNet18Classifier, 
    TransformerClassifier, 
    ViTClassifier,
    PatchTST,
    TimesFormer1D
)

# Import TaskTrainer
from engine.supervised.task_trainer import TaskTrainer

# Model factory dictionary
MODEL_TYPES = {
    'mlp': MLPClassifier,
    'lstm': LSTMClassifier,
    'resnet18': ResNet18Classifier,
    'transformer': TransformerClassifier,
    'vit': ViTClassifier,
    'patchtst': PatchTST,
    'timesformer1d': TimesFormer1D
}

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Train a supervised model on WiFi benchmark dataset')
        parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                            help='Root directory of the dataset')
        parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                            help='Name of the task to train on')
        parser.add_argument('--model', type=str, default='vit', 
                            choices=['mlp', 'lstm', 'resnet18', 'transformer', 'vit', 'patchtst', 'timesformer1d'],
                            help='Type of model to train')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--data_key', type=str, default='CSI_amps',
                            help='Key for CSI data in h5 files')
        parser.add_argument('--save_dir', type=str, default='results',
                            help='Directory to save checkpoints')
        parser.add_argument('--output_dir', type=str, default=None,
                            help='Directory to save results (defaults to save_dir if not specified)')
        parser.add_argument('--weight_decay', type=float, default=1e-5, 
                            help='Weight decay for optimizer')
        parser.add_argument('--warmup_epochs', type=int, default=5,
                            help='Number of warmup epochs')
        parser.add_argument('--patience', type=int, default=15,
                            help='Patience for early stopping')
        # Additional model parameters
        parser.add_argument('--win_len', type=int, default=500, 
                            help='Window length for WiFi CSI data')
        parser.add_argument('--feature_size', type=int, default=98, 
                            help='Feature size for WiFi CSI data')
        parser.add_argument('--in_channels', type=int, default=1, 
                            help='Number of input channels for convolutional models')
        parser.add_argument('--emb_dim', type=int, default=128, 
                            help='Embedding dimension for transformer models')
        parser.add_argument('--d_model', type=int, default=256, 
                            help='Model dimension for Transformer model')
        parser.add_argument('--dropout', type=float, default=0.1, 
                            help='Dropout rate for regularization')
        # PatchTST specific parameters
        parser.add_argument('--patch_len', type=int, default=16,
                            help='Patch length for PatchTST model')
        parser.add_argument('--stride', type=int, default=8,
                            help='Stride for patches in PatchTST model')
        parser.add_argument('--pool', type=str, default='cls', choices=['cls', 'mean'],
                            help='Pooling method for PatchTST (cls or mean)')
        parser.add_argument('--head_dropout', type=float, default=0.2,
                            help='Dropout rate for classification head')
        # TimesFormer-1D specific parameters
        parser.add_argument('--patch_size', type=int, default=4,
                            help='Patch size for TimesFormer-1D model')
        parser.add_argument('--attn_dropout', type=float, default=0.1,
                            help='Dropout rate for attention layers')
        parser.add_argument('--mlp_ratio', type=float, default=4.0,
                            help='MLP ratio for transformer blocks')
        parser.add_argument('--depth', type=int, default=6,
                            help='Number of transformer layers')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads')
        
        args = parser.parse_args()
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = args.save_dir
    
    # Ensure directories exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if running in SageMaker
    is_sagemaker = os.path.exists('/opt/ml/model')
    
    # Generate a unique experiment ID based only on parameters hash
    # This way, same parameters will generate same experiment ID and overwrite previous results
    param_str = f"{args.learning_rate}_{args.batch_size}_{args.epochs}_{args.weight_decay}_{args.warmup_epochs}_{args.win_len}_{args.feature_size}"
    if hasattr(args, 'dropout') and args.dropout is not None:
        param_str += f"_{args.dropout}"
    if hasattr(args, 'emb_dim') and args.emb_dim is not None:
        param_str += f"_{args.emb_dim}"
    if hasattr(args, 'd_model') and args.d_model is not None:
        param_str += f"_{args.d_model}"
    if hasattr(args, 'in_channels') and args.in_channels is not None:
        param_str += f"_{args.in_channels}"
    
    experiment_id = f"params_{hashlib.md5(param_str.encode()).hexdigest()[:10]}"
    
    if is_sagemaker:
        print("Running in SageMaker environment")
        model_dir = '/opt/ml/model'
        # If running in SageMaker, ensure we save in the model directory with task/model/experiment structure
        # Directory structure: model_dir/task_name/model/experiment_id/
        results_dir = os.path.join(model_dir, args.task_name, args.model, experiment_id)
        os.makedirs(results_dir, exist_ok=True)
        # Also create the checkpoints directory under the task/model/experiment structure
        # Directory structure: save_dir/task_name/model/experiment_id/
        checkpoint_dir = os.path.join(args.save_dir, args.task_name, args.model, experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        print("Running in local environment")
        # Use specified directories for local environment with task/model/experiment structure
        # Directory structure: output_dir/task_name/model/experiment_id/
        results_dir = os.path.join(args.output_dir, args.task_name, args.model, experiment_id)
        os.makedirs(results_dir, exist_ok=True)
        # Create the checkpoints directory under the task/model/experiment structure
        # Directory structure: save_dir/task_name/model/experiment_id/
        checkpoint_dir = os.path.join(args.save_dir, args.task_name, args.model, experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create best_performance.json file at model level if it doesn't exist
        model_level_dir = os.path.join(args.output_dir, args.task_name, args.model)
        best_performance_path = os.path.join(model_level_dir, "best_performance.json")
        if not os.path.exists(best_performance_path):
            with open(best_performance_path, "w") as f:
                json.dump({
                    "best_test_accuracy": 0.0,
                    "best_test_f1_score": 0.0,
                    "best_experiment_id": None,
                    "best_experiment_params": {}
                }, f, indent=4)
    
    print(f"Experiment ID: {experiment_id}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Model checkpoints will be saved to: {checkpoint_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_dir} for task {args.task_name}...")
    data = load_benchmark_supervised(
        dataset_root=args.data_dir,
        task_name=args.task_name,
        batch_size=args.batch_size,
        file_format="h5",
        data_key=args.data_key,
        num_workers=4
    )
    
    # Extract data from the returned dictionary
    loaders = data['loaders']
    num_classes = data['num_classes']
    label_mapper = data['label_mapper']
    
    # Get training and validation loaders
    train_loader = loaders['train']
    val_loader = loaders.get('val')
    
    if val_loader is None:
        print("Warning: No validation data found. Using training data for validation.")
        val_loader = train_loader
    
    # Count unique labels in the dataset
    all_labels = []
    dataset = train_loader.dataset
    print(f"Detected {num_classes} classes in the dataset")
    
    # Get test loaders
    test_loaders = {k: v for k, v in loaders.items() if k.startswith('test')}
    
    # Prepare model
    print(f"Creating {args.model.upper()} model...")
    ModelClass = MODEL_TYPES[args.model]
    
    model_kwargs = {'num_classes': num_classes}
    
    # Add model-specific parameters
    if args.model in ['mlp', 'vit', 'patchtst', 'timesformer1d']:
        model_kwargs.update({'win_len': args.win_len, 'feature_size': args.feature_size})
    
    # ResNet18 specific parameters
    if args.model == 'resnet18':
        model_kwargs.update({'in_channels': args.in_channels})
    
    # LSTM specific parameters
    if args.model == 'lstm':
        model_kwargs.update({'feature_size': args.feature_size})
    
    # Transformer specific parameters
    if args.model == 'transformer':
        model_kwargs.update({
            'feature_size': args.feature_size,
            'd_model': args.d_model,
            'dropout': args.dropout
        })
    
    # ViT specific parameters
    if args.model == 'vit':
        model_kwargs.update({
            'emb_dim': args.emb_dim,
            'dropout': args.dropout
        })
    
    # PatchTST specific parameters
    if args.model == 'patchtst':
        model_kwargs.update({
            'patch_len': args.patch_len,
            'stride': args.stride,
            'emb_dim': args.emb_dim, 
            'pool': args.pool,
            'head_dropout': args.head_dropout,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'dropout': args.dropout
        })
    
    # TimesFormer-1D specific parameters
    if args.model == 'timesformer1d':
        model_kwargs.update({
            'patch_size': args.patch_size,
            'emb_dim': args.emb_dim,
            'depth': args.depth,
            'num_heads': args.num_heads,
            'attn_dropout': args.attn_dropout,
            'head_dropout': args.head_dropout,
            'mlp_ratio': args.mlp_ratio,
            'dropout': args.dropout
        })
    
    # Initialize model
    model = ModelClass(**model_kwargs)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Set up optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Create config dictionary for saving
    config = {
        'model': args.model,
        'task': args.task_name,
        'num_classes': num_classes,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'win_len': args.win_len,
        'feature_size': args.feature_size,
    }
    
    # Save configuration
    config_path = os.path.join(results_dir, f"{args.model}_{args.task_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create scheduler
    num_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def warmup_cosine_schedule(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    # Create TaskTrainer
    trainer = TaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=checkpoint_dir,
        num_classes=num_classes,
        config={
            'model': args.model,
            'task': args.task_name,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'warmup_epochs': args.warmup_epochs,
            'patience': args.patience,
        },
        label_mapper=label_mapper
    )
    
    # Train the model with early stopping
    best_epoch, history = trainer.train(
        epochs=args.epochs,
        patience=args.patience
    )
    
    # Evaluate on test dataset
    print("\nEvaluating on test splits:")
    all_results = {}
    for key, loader in test_loaders.items():
        print(f"Evaluating on {key} split:")
        metrics = trainer.evaluate(loader)
        all_results[key] = metrics
        print(f"{key} accuracy: {metrics['accuracy']:.4f}, F1-score: {metrics['f1_score']:.4f}")
        
        # Generate confusion matrix
        print(f"Generating confusion matrix for {key} split...")
        confusion_path = os.path.join(results_dir, f"{args.model}_{args.task_name}_{key}_confusion.png")
        trainer.plot_confusion_matrix(data_loader=loader, mode=key, save_path=confusion_path)
    
    # Save test results
    results_file = os.path.join(results_dir, f"{args.model}_{args.task_name}_results.json")
    
    # Some objects in the metrics might not be JSON serializable, so we need to convert them
    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Process all keys and values in all_results
    for key in all_results:
        all_results[key] = {k: convert_to_json_serializable(v) for k, v in all_results[key].items()}
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Save summary that includes training history, best epoch
    summary = {
        'best_epoch': best_epoch,
        'best_val_loss': float(history.iloc[best_epoch]['val_loss']),
        'best_val_accuracy': float(history.iloc[best_epoch]['val_accuracy']),
        'test_accuracy': all_results.get('test', {}).get('accuracy', 0.0),
        'test_f1_score': all_results.get('test', {}).get('f1_score', 0.0),
        'experiment_id': experiment_id,
        'experiment_completed': True
    }
    
    summary_file = os.path.join(results_dir, f"{args.model}_{args.task_name}_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save training history
    history_file = os.path.join(results_dir, f"{args.model}_{args.task_name}_train_history.csv")
    history.to_csv(history_file, index=False)
    
    print(f"\nTraining and evaluation completed.")
    print(f"Best model from epoch {best_epoch}, saved to {checkpoint_dir}")
    print(f"Results saved to {results_dir}")
    
    # Check if this is the best model for the task so far
    if not is_sagemaker:
        # Update best_performance.json if performance improved
        best_performance_path = os.path.join(model_level_dir, "best_performance.json")
        
        try:
            with open(best_performance_path, 'r') as f:
                best_performance = json.load(f)
            
            # Get current best test accuracy
            current_best = best_performance.get('best_test_accuracy', 0.0)
            
            # Compare current model performance with the best so far
            test_accuracy = all_results.get('test', {}).get('accuracy', 0.0)
            
            if test_accuracy > current_best:
                print(f"New best model! Test accuracy: {test_accuracy:.4f} (previous best: {current_best:.4f})")
                
                # Update best performance
                best_performance.update({
                    'best_test_accuracy': test_accuracy,
                    'best_test_f1_score': all_results.get('test', {}).get('f1_score', 0.0),
                    'best_experiment_id': experiment_id,
                    'best_experiment_params': config,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                
                # Save updated best performance
                with open(best_performance_path, 'w') as f:
                    json.dump(best_performance, f, indent=4)
            else:
                print(f"Not the best model. Current test accuracy: {test_accuracy:.4f} (best: {current_best:.4f})")
        except Exception as e:
            print(f"Warning: Failed to update best_performance.json: {e}")
    
    return summary['test_accuracy'], summary, model

if __name__ == '__main__':
    import math  # Import math here for the scheduler function
    main()