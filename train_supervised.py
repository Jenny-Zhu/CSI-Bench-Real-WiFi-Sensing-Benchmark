import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the new module structure
from engine.supervised.task_trainer import TaskTrainer
from load import (
    load_data_supervised,
    load_model_pretrained,
    save_data_supervised,
    load_acf_supervised,
    load_acf_unseen_environ,
    load_csi_supervised_integrated,
    load_csi_unseen_integrated
)
from tools.visualization import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Supervised learning for WiFi signals')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--mode', type=str, default='csi', choices=['csi', 'acf'], help='Training mode: csi or acf')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--freeze-backbone', action='store_true', help='Whether to freeze backbone network')
    parser.add_argument('--pretrained', action='store_true', help='Whether to use pretrained model')
    
    # Data parameters
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    parser.add_argument('--unseen-test', action='store_true', help='Whether to test on unseen environments')
    parser.add_argument('--integrated-loader', action='store_true', help='Whether to use integrated data loader')
    parser.add_argument('--task', type=str, default='ThreeClass', help='Task type for integrated loader (e.g., ThreeClass, HumanNonhuman)')
    
    # Path configuration
    parser.add_argument('--csi-data-dir', type=str, default='/opt/ml/input/data/csi', help='CSI data directory')
    parser.add_argument('--acf-data-dir', type=str, default='/opt/ml/input/data/acf', help='ACF data directory')
    parser.add_argument('--pretrained-model', type=str, default=None, help='Pretrained model path')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--model-name', type=str, default='WiT', help='Model name')
    parser.add_argument('--results-subdir', type=str, default='supervised', help='Results subdirectory')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Training device')

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_supervised_csi(args):
    print(f"Starting CSI modality supervised training...")
    
    # Prepare data
    if args.integrated_loader:
        print(f"Using integrated data loader with task: {args.task}")
        data_loaders = load_csi_supervised_integrated(
            args.csi_data_dir,
            task=args.task,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    else:
        data_loaders = load_data_supervised(
            args.csi_data_dir, 
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    
    train_loader, val_loader, test_loader = data_loaders
    
    # Load model
    if args.pretrained and args.pretrained_model:
        model = load_model_pretrained(args.pretrained_model, num_classes=args.num_classes)
        print(f"Loaded pretrained model: {args.pretrained_model}")
    else:
        from load import load_model_scratch
        model = load_model_scratch(num_classes=args.num_classes)
        print("Using randomly initialized model")
    
    # Freeze backbone network (if needed)
    if args.freeze_backbone:
        print("Freezing backbone network...")
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
    
    # Set save path
    model_type = "pretrained" if args.pretrained else "scratch"
    freeze_status = "frozen" if args.freeze_backbone else "unfrozen"
    loader_type = "integrated" if args.integrated_loader else "standard"
    save_path = os.path.join(args.output_dir, args.results_subdir, 
                            f"{args.model_name}_csi_{model_type}_{freeze_status}_{loader_type}")
    os.makedirs(save_path, exist_ok=True)
    
    # Set training configuration
    config = argparse.Namespace()
    config.num_epochs = args.num_epochs
    config.patience = args.patience
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.warmup_epochs = args.warmup_epochs
    config.save_path = save_path
    config.num_classes = args.num_classes
    
    # Set loss function
    criterion = nn.CrossEntropyLoss()
    
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
    
    # Test model
    print("Evaluating model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Test results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(test_loader)
    
    print(f"CSI supervised training completed! Model and results saved to {save_path}")
    
    return model, results_df


def train_supervised_acf(args):
    print(f"Starting ACF modality supervised training...")
    
    # Prepare data
    if args.unseen_test:
        if args.integrated_loader:
            data_loaders = load_csi_unseen_integrated(
                args.acf_data_dir,
                task=args.task,
                batch_size=args.batch_size
            )
            print(f"Using integrated loader for unseen environments with task: {args.task}")
        else:
            data_loaders = load_acf_unseen_environ(
                args.acf_data_dir,
                batch_size=args.batch_size
            )
            print("Using test set with unseen environments...")
    else:
        data_loaders = load_acf_supervised(
            args.acf_data_dir,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    
    train_loader, val_loader, test_loader = data_loaders
    
    # Load model
    if args.pretrained and args.pretrained_model:
        model = load_model_pretrained(args.pretrained_model, num_classes=args.num_classes)
        print(f"Loaded pretrained model: {args.pretrained_model}")
    else:
        from load import load_model_scratch
        model = load_model_scratch(num_classes=args.num_classes)
        print("Using randomly initialized model")
    
    # Freeze backbone network (if needed)
    if args.freeze_backbone:
        print("Freezing backbone network...")
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
    
    # Set save path
    model_type = "pretrained" if args.pretrained else "scratch"
    freeze_status = "frozen" if args.freeze_backbone else "unfrozen"
    test_type = "unseen" if args.unseen_test else "seen"
    loader_type = "integrated" if args.integrated_loader else "standard"
    save_path = os.path.join(args.output_dir, args.results_subdir, 
                            f"{args.model_name}_acf_{model_type}_{freeze_status}_{test_type}_{loader_type}")
    os.makedirs(save_path, exist_ok=True)
    
    # Set training configuration
    config = argparse.Namespace()
    config.num_epochs = args.num_epochs
    config.patience = args.patience
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.warmup_epochs = args.warmup_epochs
    config.save_path = save_path
    config.num_classes = args.num_classes
    
    # Set loss function
    criterion = nn.CrossEntropyLoss()
    
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
    
    # Test model
    print("Evaluating model on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Test results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(test_loader)
    
    print(f"ACF supervised training completed! Model and results saved to {save_path}")
    
    return model, results_df


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Choose training function based on mode
    if args.mode.lower() == 'csi':
        train_supervised_csi(args)
    elif args.mode.lower() == 'acf':
        train_supervised_acf(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Please choose 'csi' or 'acf'")


if __name__ == "__main__":
    main() 