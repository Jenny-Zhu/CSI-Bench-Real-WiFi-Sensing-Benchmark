import os
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import math  # Add this import at the top
import re


# Import training engines
from engine.supervised.task_trainer import TaskTrainer
from engine.supervised.task_trainer_acf import TaskTrainerACF

# Import data loaders
from load import (
    load_csi_supervised,
    load_acf_supervised,
    load_model_scratch
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train supervised learning models')
    
    # Data directories
    parser.add_argument('--training-dir', type=str, default=None,
                      help='Directory containing training data')
    parser.add_argument('--test-dirs', type=str, nargs='+', default=None,
                      help='List of directories containing test data. Can specify multiple paths')
    parser.add_argument('--output-dir', type=str, default='experiments',
                      help='Directory to save output results')
    parser.add_argument('--results-subdir', type=str, default='supervised',
                      help='Subdirectory under output directory to save results')
    
    # Data parameters
    parser.add_argument('--mode', type=str, choices=['csi', 'acf'], default='csi',
                      help='Data modality to use (csi or acf)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Ratio of data to use for training (from training dir)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                      help='Ratio of data to use for validation (from training dir)')
    parser.add_argument('--win-len', type=int, default=250,
                      help='Window length for CSI data')
    parser.add_argument('--feature-size', type=int, default=90,
                      help='Feature size for CSI data')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15, help='Patience for early stopping')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--in-channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--freeze-backbone', action='store_true', help='Whether to freeze the backbone network')
    
    # Data parameters
    parser.add_argument('--integrated-loader', action='store_true', help='Whether to use the integrated data loader')
    parser.add_argument('--task', type=str, default='ThreeClass', 
                      help='Task type for integrated loader (e.g. ThreeClass, HumanNonhuman)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Training device')
    parser.add_argument('--model-name', type=str, default='WiT', help='Model name')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_supervised_csi(args):
    """Train a model on CSI data"""
    print(f"Starting supervised learning training on CSI modality...")

    # 设置训练目录和验证目录
    train_dir = args.training_dir
    val_dir = args.training_dir
    
    # 检查是否是SageMaker环境
    in_sagemaker = os.path.exists('/opt/ml/input/data')
    
    # 在非SageMaker环境中检查目录是否存在
    if not in_sagemaker:
        if not os.path.exists(train_dir):
            raise ValueError(f"Training directory {train_dir} does not exist")
    else:
        # 在SageMaker环境中，确保目录存在
        if not os.path.exists(train_dir):
            raise ValueError(f"SageMaker training directory {train_dir} does not exist")

    # 使用更新后的load_csi_supervised加载数据
    train_loader, val_loader, test_loaders_dict = load_csi_supervised(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        task=args.task,
        test_dirs=args.test_dirs
    )
    
    # 检查测试集是否存在
    has_multiple_test_sets = len(test_loaders_dict) > 0

    # Load model
    model = load_model_scratch(
        model_name=args.model_name,
        task=args.task,
        win_len=args.win_len,
        feature_size=args.feature_size,
        in_channels=getattr(args, 'in_channels', 1)  # Default to 1 if not provided
    )
    print("Using randomly initialized model")
    
    # Setup save path - 在SageMaker环境中使用不同的保存路径
    if in_sagemaker:
        save_path = os.path.join('/opt/ml/model', args.results_subdir, 
                                f"{args.task}_{args.model_name}_csi")
    else:
        save_path = os.path.join(args.output_dir, args.results_subdir, 
                                f"{args.task}_{args.model_name}_csi")
    os.makedirs(save_path, exist_ok=True)
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create config object for trainer
    config = argparse.Namespace()
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.num_epochs = args.num_epochs
    config.patience = args.patience
    config.num_classes = args.num_classes
    config.output_dir = args.output_dir
    config.results_subdir = args.results_subdir
    config.model_name = f"{args.task}_{args.model_name}_csi"
    config.device = args.device
    
    # Create trainer and start training
    trainer = TaskTrainer(
        model=model,
        data_loader=(train_loader, val_loader),
        config=config,
        criterion=criterion
    )
    
    model, results_df = trainer.train()
    
    # Save training results
    results_df.to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
    
    # Plot training results
    plot_results(results_df, save_path)
    
    # Test on all test sets
    test_results_all = []
    
    # Create a summary DataFrame for all test results
    test_summary = pd.DataFrame(columns=['Test Set', 'Loss', 'Accuracy'])
    
    # 如果没有测试集，则直接返回
    if not has_multiple_test_sets:
        print("No test sets available. Skipping evaluation.")
        return model, results_df
    
    print("\nEvaluating model on test sets:")
    for test_name, test_loader in test_loaders_dict.items():
        print(f"\nEvaluating on test set: {test_name}")
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        # Create test results DataFrame
        test_results = pd.DataFrame({
            'Test Loss': [test_loss],
            'Test Accuracy': [test_acc]
        })
        
        # Add to summary
        test_summary = pd.concat([
            test_summary, 
            pd.DataFrame({'Test Set': [test_name], 'Loss': [test_loss], 'Accuracy': [test_acc]})
        ])
        
        # Save individual test results
        test_set_dir = os.path.join(save_path, f'test_{test_name}')
        os.makedirs(test_set_dir, exist_ok=True)
        test_results.to_csv(os.path.join(test_set_dir, 'test_results.csv'), index=False)
        
        print(f"Test set: {test_name}, Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        test_results_all.append((test_name, test_loss, test_acc))
    
    # Save summary of all test results
    test_summary.to_csv(os.path.join(save_path, 'test_summary.csv'), index=False)
    
    # Plot test results comparison
    if len(test_results_all) > 1:
        plot_test_comparison(test_results_all, save_path)
    
    print(f"CSI supervised training completed! Model and results saved to {save_path}")
    
    return model, results_df


def train_supervised_acf(args):
    """Train a model on ACF data"""
    print(f"Starting supervised learning training on ACF modality...")
    
    # 确保训练目录和验证目录存在
    train_dir = os.path.join(args.training_dir, 'train')
    val_dir = os.path.join(args.training_dir, 'validation')
    
    # 检查训练和验证目录是否存在
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory {train_dir} does not exist")
    if not os.path.exists(val_dir):
        raise ValueError(f"Validation directory {val_dir} does not exist")
    
    # 使用更新后的load_acf_supervised加载数据
    train_loader, val_loader, test_loaders_dict = load_acf_supervised(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        task=args.task,
        test_dirs=args.test_dirs
    )
    
    # 检查测试集是否存在
    has_multiple_test_sets = len(test_loaders_dict) > 0
    
    # Load model
    model = load_model_scratch(
        model_name=args.model_name,
        task=args.task,
        win_len=args.win_len,
        feature_size=args.feature_size,
        in_channels=getattr(args, 'in_channels', 1)  # Default to 1 if not provided
    )
    print("Using randomly initialized model")
    
    # Setup save path
    save_path = os.path.join(args.output_dir, args.results_subdir, 
                           f"{args.task}_{args.model_name}_acf")
    os.makedirs(save_path, exist_ok=True)
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create config object for trainer
    config = argparse.Namespace()
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.num_epochs = args.num_epochs
    config.patience = args.patience
    config.num_classes = args.num_classes
    config.output_dir = args.output_dir
    config.results_subdir = args.results_subdir
    config.model_name = f"{args.task}_{args.model_name}_acf"
    config.freeze_backbone = args.freeze_backbone
    config.device = args.device
    
    # Create trainer and start training
    trainer = TaskTrainerACF(
        model=model,
        data_loader=(train_loader, val_loader),
        config=config,
        criterion=criterion
    )
    
    model, results_df = trainer.train()
    
    # Save training results
    results_df.to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
    
    # Plot training results
    plot_results(results_df, save_path)
    
    # Test on all test sets
    test_results_all = []
    
    # Create a summary DataFrame for all test results
    test_summary = pd.DataFrame(columns=['Test Set', 'Loss', 'Accuracy'])
    
    # 如果没有测试集，则直接返回
    if not has_multiple_test_sets:
        print("No test sets available. Skipping evaluation.")
        return model, results_df
    
    print("\nEvaluating model on test sets:")
    for test_name, test_loader in test_loaders_dict.items():
        print(f"\nEvaluating on test set: {test_name}")
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        # Create test results DataFrame
        test_results = pd.DataFrame({
            'Test Loss': [test_loss],
            'Test Accuracy': [test_acc]
        })
        
        # Add to summary
        test_summary = pd.concat([
            test_summary, 
            pd.DataFrame({'Test Set': [test_name], 'Loss': [test_loss], 'Accuracy': [test_acc]})
        ])
        
        # Save individual test results
        test_set_dir = os.path.join(save_path, f'test_{test_name}')
        os.makedirs(test_set_dir, exist_ok=True)
        test_results.to_csv(os.path.join(test_set_dir, 'test_results.csv'), index=False)
        
        print(f"Test set: {test_name}, Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        
        test_results_all.append((test_name, test_loss, test_acc))
    
    # Save summary of all test results
    test_summary.to_csv(os.path.join(save_path, 'test_summary.csv'), index=False)
    
    # Plot test results comparison
    if len(test_results_all) > 1:
        plot_test_comparison(test_results_all, save_path)
    
    print(f"ACF supervised training completed! Model and results saved to {save_path}")
    
    return model, results_df


def plot_results(results, save_path):
    """Plot training and validation curves"""
    # Plot validation accuracy
    fig_val_acc = plt.figure(figsize=(7, 7))
    sn.lineplot(x=results['Epoch'], y=results['Val Accuracy'])
    plt.title('Validation Accuracy')
    plt.savefig(os.path.join(save_path, "val_acc.png"))
    plt.close()
    
    # Plot validation loss
    fig_val_loss = plt.figure(figsize=(7, 7))
    sn.lineplot(x=results['Epoch'], y=results['Val Loss'])
    plt.title('Validation Loss')
    plt.savefig(os.path.join(save_path, "val_loss.png"))
    plt.close()
    
    # Plot training accuracy
    fig_train_acc = plt.figure(figsize=(7, 7))
    sn.lineplot(x=results['Epoch'], y=results['Train Accuracy'])
    plt.title('Training Accuracy')
    plt.savefig(os.path.join(save_path, "train_acc.png"))
    plt.close()
    
    # Plot training loss
    fig_train_loss = plt.figure(figsize=(7, 7))
    sn.lineplot(x=results['Epoch'], y=results['Train Loss'])
    plt.title('Training Loss')
    plt.savefig(os.path.join(save_path, "train_loss.png"))
    plt.close()


def plot_test_comparison(test_results, save_path):
    """Plot comparison of test results across multiple test sets
    
    Args:
        test_results: List of tuples (test_name, test_loss, test_acc)
        save_path: Directory to save the plots
    """
    # Extract data
    test_names = [result[0] for result in test_results]
    test_loss = [result[1] for result in test_results]
    test_acc = [result[2] for result in test_results]
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(test_names, test_acc, color='skyblue')
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "test_accuracy_comparison.png"))
    plt.close()
    
    # Plot loss comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(test_names, test_loss, color='salmon')
    plt.title('Test Loss Comparison')
    plt.ylabel('Loss')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "test_loss_comparison.png"))
    plt.close()


def main(args=None):
    """Main function"""
    # If arguments not provided, parse command line arguments
    if args is None:
        args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Check if running in SageMaker environment
    in_sagemaker = os.path.exists('/opt/ml/input/data')
    
    # In SageMaker, use the right directories
    if in_sagemaker:
        print("Running in SageMaker environment")
        # SageMaker mounts input data to these directories
        if os.path.exists('/opt/ml/input/data/training'):
            args.training_dir = '/opt/ml/input/data/training'
            print(f"Using SageMaker training directory: {args.training_dir}")
        
        # Check for test directories - SageMaker may have them as separate channels
        test_dirs = []
        if os.path.exists('/opt/ml/input/data/test'):
            test_dirs.append('/opt/ml/input/data/test')
        
        # Check for additional test directories
        test_channel_pattern = re.compile(r'test\d+')
        for channel_dir in os.listdir('/opt/ml/input/data'):
            if test_channel_pattern.match(channel_dir):
                channel_path = os.path.join('/opt/ml/input/data', channel_dir)
                test_dirs.append(channel_path)
        
        if test_dirs:
            args.test_dirs = test_dirs
            print(f"Using SageMaker test directories: {args.test_dirs}")
    
    # For backwards compatibility, set training_dir to csi/acf_data_dir if specified
    if not args.training_dir:
        if args.mode.lower() == 'csi' and hasattr(args, 'csi_data_dir'):
            args.training_dir = args.csi_data_dir
        elif args.mode.lower() == 'acf' and hasattr(args, 'acf_data_dir'):
            args.training_dir = args.acf_data_dir
        else:
            raise ValueError("No training directory specified. Use --training-dir.")
    
    # Choose training function based on data modality
    if args.mode.lower() == 'csi':
        train_supervised_csi(args)
    elif args.mode.lower() == 'acf':
        train_supervised_acf(args)
    else:
        raise ValueError(f"Unknown modality: {args.mode}. Please choose 'csi' or 'acf'")


if __name__ == "__main__":
    main() 