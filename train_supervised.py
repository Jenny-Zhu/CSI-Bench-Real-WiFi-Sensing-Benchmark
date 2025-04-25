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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train supervised learning model')
    
    # Data directories
    parser.add_argument('--csi-data-dir', type=str, default=None,
                      help='Directory containing CSI data')
    parser.add_argument('--acf-data-dir', type=str, default=None,
                      help='Directory containing ACF data')
    parser.add_argument('--train-data-dir', type=str, default=None,
                      help='Directory containing training data (used with unseen_test)')
    parser.add_argument('--output-dir', type=str, default='experiments',
                      help='Directory to save output results')
    parser.add_argument('--results-subdir', type=str, default='supervised',
                      help='Subdirectory under output_dir to save results')
    
    # Data parameters
    parser.add_argument('--mode', type=str, choices=['csi', 'acf'], default='csi',
                      help='Data modality to use (csi or acf)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Ratio of data to use for training')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                      help='Ratio of data to use for validation')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                      help='Ratio of data to use for testing')
    parser.add_argument('--win-len', type=int, default=250,
                      help='Window length for CSI data')
    parser.add_argument('--feature-size', type=int, default=90,
                      help='Feature size for CSI data')
    parser.add_argument('--sample-rate', type=int, default=100,
                      help='Sampling rate of the data')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--in-channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--freeze-backbone', action='store_true', help='Whether to freeze backbone network')
    parser.add_argument('--pretrained', action='store_true', help='Whether to use pretrained model')
    
    # Data parameters
    parser.add_argument('--unseen-test', action='store_true', help='Whether to test on unseen environments')
    parser.add_argument('--integrated-loader', action='store_true', help='Whether to use integrated data loader')
    parser.add_argument('--task', type=str, default='ThreeClass', help='Task type for integrated loader (e.g., ThreeClass, HumanNonhuman)')
    
    # Path configuration
    parser.add_argument('--pretrained-model', type=str, default=None, help='Pretrained model path')
    parser.add_argument('--model-name', type=str, default='WiT', help='Model name')
    
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
    """Train a model for CSI modality."""
    print(f"Starting CSI modality supervised training...")
    
    # # Prepare data
    # if args.unseen_test:
    #     # For unseen test environment
    #     if args.integrated_loader:
    #         # Test loader only
    #         test_loader = load_csi_unseen_integrated(
    #             args.csi_data_dir,
    #             task=args.task,
    #             batch_size=args.batch_size
    #         )
    #         print(f"Using integrated loader for unseen environments with task: {args.task}")
            
    #         # Need to create train and val loaders from a different directory
    #         # Assume train data is in the parent directory or provided separately
    #         if hasattr(args, 'train_data_dir') and args.train_data_dir:
    #             train_dir = args.train_data_dir
    #         else:
    #             # Try to use parent directory
    #             train_dir = os.path.dirname(args.csi_data_dir.rstrip('/\\'))
    #             if not train_dir or train_dir == args.csi_data_dir:
    #                 train_dir = args.csi_data_dir
            
    #         # Load training data
    #         print(f"Loading training data from: {train_dir}")
    #         train_loader, val_loader, _ = load_csi_supervised_integrated(
    #             train_dir,
    #             task=args.task,
    #             batch_size=args.batch_size,
    #             train_ratio=args.train_ratio,
    #             val_ratio=args.val_ratio,
    #             test_ratio=args.test_ratio
    #         )
    #     else:
    #         # Legacy mode
    #         train_loader, val_loader, test_loader = load_data_supervised(
    #             'OW_HM3', 
    #             args.batch_size, 
    #             args.win_len, 
    #             args.sample_rate
    #         )
    # else:
    # Normal training mode
    if args.integrated_loader:
        # Get all three loaders from the same function
        train_loader, val_loader, test_loader = load_csi_supervised_integrated(
            args.csi_data_dir,
            task=args.task,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        print(f"Using integrated loader with task: {args.task}")
    else:
        # Legacy mode
        train_loader, test_loader = load_data_supervised(
            'OW_HM3', 
            args.batch_size, 
            args.win_len, 
            args.sample_rate
        )
        # Create a validation loader by splitting the test loader
        # This is a legacy behavior and might not be optimal
        val_loader = test_loader  # Use test loader as validation in legacy mode

    # Load model
    if args.pretrained and args.pretrained_model:
        # Load pretrained model
        model = load_model_pretrained(
            checkpoint_path=args.pretrained_model, 
            num_classes=args.num_classes,
            win_len=args.win_len,
            feature_size=args.feature_size,
            in_channels=args.in_channels
        )
        print(f"Loaded pretrained model: {args.pretrained_model}")
    else:
        # Create new model from scratch
        from load import load_model_scratch
        model = load_model_scratch(
            num_classes=args.num_classes, 
            win_len=args.win_len, 
            feature_size=args.feature_size,
            in_channels=args.in_channels
        )
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
    config.output_dir = args.output_dir
    config.results_subdir = args.results_subdir
    config.model_name = args.model_name
    
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
    """Train a model for ACF modality."""
    print(f"Starting ACF modality supervised training...")
    
    # Prepare data
    if args.unseen_test:
        # For unseen test environment
        if args.integrated_loader:
            # Test loader only
            test_loader = load_csi_unseen_integrated(
                args.acf_data_dir,
                task=args.task,
                batch_size=args.batch_size
            )
            print(f"Using integrated loader for unseen environments with task: {args.task}")
            
            # Need to create train and val loaders from a different directory
            # Assume train data is in the parent directory or provided separately
            if hasattr(args, 'train_data_dir') and args.train_data_dir:
                train_dir = args.train_data_dir
            else:
                # Try to use parent directory
                train_dir = os.path.dirname(args.acf_data_dir.rstrip('/\\'))
                if not train_dir or train_dir == args.acf_data_dir:
                    train_dir = args.acf_data_dir
            
            # Load training data
            print(f"Loading training data from: {train_dir}")
            train_loader, val_loader, _ = load_acf_supervised(
                train_dir,
                task=args.task,
                batch_size=args.batch_size
            )
        else:
            # Legacy unseen test mode
            test_loader = load_acf_unseen_environ(
                args.acf_data_dir,
                task=args.task
            )
            print("Using test set with unseen environments...")
            
            # Need to create train and val loaders from a different directory
            if hasattr(args, 'train_data_dir') and args.train_data_dir:
                train_dir = args.train_data_dir
            else:
                # Try to use parent directory
                train_dir = os.path.dirname(args.acf_data_dir.rstrip('/\\'))
                if not train_dir or train_dir == args.acf_data_dir:
                    train_dir = args.acf_data_dir
            
            # Load training data
            print(f"Loading training data from: {train_dir}")
            train_loader, val_loader, _ = load_acf_supervised(
                train_dir,
                task=args.task,
                batch_size=args.batch_size
            )
    else:
        # Normal training mode
        train_loader, val_loader, test_loader = load_acf_supervised(
            args.acf_data_dir,
            task=args.task,
            batch_size=args.batch_size
        )
        print(f"Using ACF data loader with task: {args.task}")
    
    # Load model
    if args.pretrained and args.pretrained_model:
        # Load pretrained model
        model = load_model_pretrained(
            checkpoint_path=args.pretrained_model, 
            num_classes=args.num_classes,
            win_len=args.win_len,
            feature_size=args.feature_size,
            in_channels=args.in_channels
        )
        print(f"Loaded pretrained model: {args.pretrained_model}")
    else:
        # Create new model from scratch
        from load import load_model_scratch
        model = load_model_scratch(
            num_classes=args.num_classes, 
            win_len=args.win_len, 
            feature_size=args.feature_size,
            in_channels=args.in_channels
        )
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
    save_path = os.path.join(args.output_dir, args.results_subdir, 
                            f"{args.model_name}_acf_{model_type}_{freeze_status}_{test_type}")
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
    config.output_dir = args.output_dir
    config.results_subdir = args.results_subdir
    config.model_name = args.model_name
    
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
    
    # 根据任务自动设置正确的类别数
    task_to_classes = {
        'HumanNonhuman': 2,
        'FourClass': 4,
        'HumanID': 4,
        'HumanMotion': 3,
        'ThreeClass': 3,
        'DetectionandClassification': 5,
        'Detection': 2,
        'NTUHumanID': 15,
        'NTUHAR': 6,
        'Widar': 22
    }
    
    if args.task in task_to_classes and args.num_classes != task_to_classes[args.task]:
        print(f"Automatically updating num_classes from {args.num_classes} to {task_to_classes[args.task]} based on task '{args.task}'")
        args.num_classes = task_to_classes[args.task]
    
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