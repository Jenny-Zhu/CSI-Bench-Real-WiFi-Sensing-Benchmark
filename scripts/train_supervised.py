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
    ViTClassifier
)

# Import TaskTrainer
from engine.supervised.task_trainer import TaskTrainer

# Model factory dictionary
MODEL_TYPES = {
    'mlp': MLPClassifier,
    'lstm': LSTMClassifier,
    'resnet18': ResNet18Classifier,
    'transformer': TransformerClassifier,
    'vit': ViTClassifier
}

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Train a supervised model on WiFi benchmark dataset')
        parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                            help='Root directory of the dataset')
        parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                            help='Name of the task to train on')
        parser.add_argument('--model_name', type=str, default='vit', 
                            choices=['mlp', 'lstm', 'resnet18', 'transformer', 'vit'],
                            help='Type of model to train')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--data_key', type=str, default='CSI_amps',
                            help='Key for CSI data in h5 files')
        parser.add_argument('--save_dir', type=str, default='checkpoints',
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
                            help='Embedding dimension for ViT model')
        parser.add_argument('--d_model', type=int, default=256, 
                            help='Model dimension for Transformer model')
        parser.add_argument('--dropout', type=float, default=0.1, 
                            help='Dropout rate for regularization')
        
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
        # Directory structure: model_dir/task_name/model_name/experiment_id/
        results_dir = os.path.join(model_dir, args.task_name, args.model_name, experiment_id)
        os.makedirs(results_dir, exist_ok=True)
        # Also create the checkpoints directory under the task/model/experiment structure
        # Directory structure: save_dir/task_name/model_name/experiment_id/
        checkpoint_dir = os.path.join(args.save_dir, args.task_name, args.model_name, experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        print("Running in local environment")
        # Use specified directories for local environment with task/model/experiment structure
        # Directory structure: output_dir/task_name/model_name/experiment_id/
        results_dir = os.path.join(args.output_dir, args.task_name, args.model_name, experiment_id)
        os.makedirs(results_dir, exist_ok=True)
        # Create the checkpoints directory under the task/model/experiment structure
        # Directory structure: save_dir/task_name/model_name/experiment_id/
        checkpoint_dir = os.path.join(args.save_dir, args.task_name, args.model_name, experiment_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create best_performance.json file at model level if it doesn't exist
        model_level_dir = os.path.join(args.output_dir, args.task_name, args.model_name)
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
    
    # Create model based on selected type
    print(f"Creating {args.model_name.upper()} model...")
    ModelClass = MODEL_TYPES[args.model_name]
    
    # Common parameters for all models
    model_params = {
        'num_classes': num_classes
    }
    
    # Add additional parameters based on model type
    if args.model_name in ['mlp', 'vit']:
        model_params.update({
            'win_len': args.win_len,
            'feature_size': args.feature_size
        })
    
    if args.model_name == 'resnet18':
        model_params.update({
            'in_channels': args.in_channels
        })
    
    if args.model_name == 'lstm':
        model_params.update({
            'feature_size': args.feature_size,
            'dropout': args.dropout
        })
    
    if args.model_name == 'transformer':
        model_params.update({
            'feature_size': args.feature_size,
            'd_model': args.d_model,
            'dropout': args.dropout
        })
    
    if args.model_name == 'vit':
        model_params.update({
            'emb_dim': args.emb_dim,
            'dropout': args.dropout,
            'in_channels': args.in_channels
        })
    
    model = ModelClass(**model_params)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create configuration object for trainer
    config = type('Config', (), {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'num_classes': num_classes,
        'device': str(device),
        'save_dir': checkpoint_dir,  # Use the checkpoint_dir for model checkpoints
        'output_dir': results_dir,  # Use results_dir for output
        'results_subdir': 'supervised',
        'model_name': args.model_name,
        'task_name': args.task_name
    })
    
    # Save the configuration
    config_path = os.path.join(results_dir, f"{args.model_name}_{args.task_name}_config.json")
    with open(config_path, "w") as f:
        # Convert config to dict
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        json.dump(config_dict, f, indent=4)
    
    print(f"Configuration saved to {config_path}")
    
    # Create trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    trainer = TaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=checkpoint_dir,  # Use checkpoint_dir for model checkpoints
        num_classes=num_classes,
        label_mapper=label_mapper,
        config=config
    )
    
    # Train the model
    trained_model, training_results = trainer.train()
    
    # Evaluate on test set(s) if available
    print("\nEvaluating on test sets:")
    
    # 统一处理训练结果，无论它是DataFrame还是字典
    overall_metrics = {
        'model_name': args.model_name,
        'task_name': args.task_name
    }
    
    # 转换training_results为标准格式
    if isinstance(training_results, pd.DataFrame):
        # 从DataFrame中提取数据
        train_history = {
            'epochs': training_results['Epoch'].tolist() if 'Epoch' in training_results.columns else list(range(1, args.epochs + 1)),
            'train_loss_history': training_results['Train Loss'].tolist() if 'Train Loss' in training_results.columns else [],
            'val_loss_history': training_results['Val Loss'].tolist() if 'Val Loss' in training_results.columns else [],
            'train_accuracy_history': training_results['Train Accuracy'].tolist() if 'Train Accuracy' in training_results.columns else [],
            'val_accuracy_history': training_results['Val Accuracy'].tolist() if 'Val Accuracy' in training_results.columns else []
        }
        
        # 找到最佳验证准确率和对应的epoch
        if 'Val Accuracy' in training_results.columns:
            best_idx = training_results['Val Accuracy'].idxmax()
            best_epoch = int(training_results.loc[best_idx, 'Epoch'])
            best_val_accuracy = float(training_results.loc[best_idx, 'Val Accuracy'])
        else:
            best_epoch = args.epochs
            best_val_accuracy = 0.0
    else:
        # 如果已经是字典，直接提取需要的数据
        train_history = {
            'train_loss_history': training_results.get('train_loss_history', []),
            'val_loss_history': training_results.get('val_loss_history', []),
            'train_accuracy_history': training_results.get('train_accuracy_history', []),
            'val_accuracy_history': training_results.get('val_accuracy_history', [])
        }
        
        # 从字典中获取最佳验证准确率和对应的epoch
        best_epoch = training_results.get('best_epoch', args.epochs)
        best_val_accuracy = training_results.get('best_val_accuracy', 0.0)
    
    # 添加最佳结果到整体指标
    overall_metrics['best_epoch'] = best_epoch
    overall_metrics['best_val_accuracy'] = best_val_accuracy
    
    for key in loaders:
        if key.startswith('test'):
            print(f"\nEvaluating on {key} split:")
            test_loss, test_acc = trainer.evaluate(loaders[key])
            print(f"{key} loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")
            
            # Store overall accuracy for this test set
            overall_metrics[f'{key}_accuracy'] = float(test_acc)
            
            # Calculate overall F1 score (weighted average)
            f1_score, _ = trainer.calculate_metrics(data_loader=loaders[key])
            overall_metrics[f'{key}_f1_score'] = float(f1_score)
            print(f"{key} F1 score (weighted): {f1_score:.4f}")
            
            # Plot confusion matrix for this test set
            confusion_path = os.path.join(results_dir, f"{args.model_name}_{args.task_name}_{key}_confusion.png")
            trainer.plot_confusion_matrix(data_loader=loaders[key], mode=key, epoch=None)
    
    # Save training results summary with overall metrics
    results_file = os.path.join(results_dir, f"{args.model_name}_{args.task_name}_results.json")
    with open(results_file, 'w') as f:
        # 创建包含训练历史和整体指标的结果
        serializable_results = {}
        serializable_results.update(train_history)
        serializable_results.update(overall_metrics)
        
        # 确保所有numpy类型都被转换为Python原生类型
        def convert_to_json_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            else:
                return obj
        
        serializable_results = convert_to_json_serializable(serializable_results)
        json.dump(serializable_results, f, indent=4)
    
    # Save a separate summary file just for overall metrics
    summary_file = os.path.join(results_dir, f"{args.model_name}_{args.task_name}_summary.json")
    with open(summary_file, 'w') as f:
        # 同样确保summary也是可序列化的
        json.dump(convert_to_json_serializable(overall_metrics), f, indent=4)
    
    # Add experiment parameters to overall metrics for tracking
    experiment_params = {
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'win_len': args.win_len,
        'feature_size': args.feature_size,
        'dropout': getattr(args, 'dropout', 0.1),
        'emb_dim': getattr(args, 'emb_dim', None),
        'd_model': getattr(args, 'd_model', None),
        'in_channels': getattr(args, 'in_channels', None),
    }
    
    # In local environment, update the best performance record if this experiment has better results
    if not is_sagemaker:
        model_level_dir = os.path.join(args.output_dir, args.task_name, args.model_name)
        best_performance_path = os.path.join(model_level_dir, "best_performance.json")
        
        # Read current best performance
        with open(best_performance_path, 'r') as f:
            best_performance = json.load(f)
        
        # Get the primary test accuracy for comparison
        test_key = 'test_accuracy' if 'test_accuracy' in overall_metrics else 'test1_accuracy'
        current_accuracy = overall_metrics.get(test_key, 0.0)
        
        # Update if current experiment has better accuracy
        if current_accuracy > best_performance['best_test_accuracy']:
            best_performance['best_test_accuracy'] = current_accuracy
            best_performance['best_experiment_id'] = experiment_id
            best_performance['best_experiment_params'] = experiment_params
            
            # If available, also update F1 score
            f1_key = 'test_f1_score' if 'test_f1_score' in overall_metrics else 'test1_f1_score'
            if f1_key in overall_metrics:
                best_performance['best_test_f1_score'] = overall_metrics[f1_key]
            
            # Write updated best performance record
            with open(best_performance_path, 'w') as f:
                json.dump(best_performance, f, indent=4)
            
            print(f"New best performance achieved! Updated {best_performance_path}")
    
    print(f"\nTraining completed. Results saved to {results_file}")
    print(f"Overall performance metrics saved to {summary_file}")
    print(f"Model checkpoints saved to {checkpoint_dir}")
    print(f"Final model and results saved to {results_dir}")

if __name__ == "__main__":
    main()