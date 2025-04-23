import argparse
import os
import torch
import numpy as np
import random
from tqdm import tqdm

# Import the new module structure
from engine.pretraining.ssl_trainer import SSLTrainer
from load import (
    load_acf_data_unsupervised,
    load_csi_data_unsupervised,
    load_model_unsupervised_joint_csi_var,
    load_model_unsupervised_joint
)
from tools.loss_function import NtXentLoss
from data.augmentation.csi_augmentation import DataAugmentation
from data.augmentation.acf_augmentation import DataAugmentACF


def parse_args():
    parser = argparse.ArgumentParser(description='Self-supervised pretraining for WiFi signals')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--mode', type=str, default='csi', choices=['csi', 'acf'], help='Pretraining mode: csi or acf')
    parser.add_argument('--depth', type=int, default=6, help='Model depth')
    parser.add_argument('--in-channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--emb-size', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--freq-out', type=int, default=10, help='Frequency output')
    
    # Data parameters
    parser.add_argument('--sample-rate', type=int, default=100, help='Sample rate')
    parser.add_argument('--time-seg', type=int, default=5, help='Time segment length')
    parser.add_argument('--win-len', type=int, default=250, help='Window length (ACF)')
    parser.add_argument('--feature-size', type=int, default=98, help='Feature size (ACF)')
    
    # Path configuration
    parser.add_argument('--csi-data-dir', type=str, default='/opt/ml/input/data/csi', help='CSI data directory')
    parser.add_argument('--acf-data-dir', type=str, default='/opt/ml/input/data/acf', help='ACF data directory')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--model-name', type=str, default='WiT', help='Model name')
    parser.add_argument('--results-subdir', type=str, default='ssl_pretrain', help='Results subdirectory')
    
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


def pretrain_csi(args):
    print(f"Starting CSI modality pretraining...")
    
    # Prepare data
    train_loader = load_csi_data_unsupervised(args.csi_data_dir, args.batch_size)
    
    # Load model
    model = load_model_unsupervised_joint_csi_var(
        emb_size=args.emb_size,
        depth=args.depth,
        freq_out=args.freq_out,
        in_channels=args.in_channels
    )
    
    # Set save path
    save_path = os.path.join(args.output_dir, args.results_subdir, f"{args.model_name}_csi")
    os.makedirs(save_path, exist_ok=True)
    
    # Set training configuration
    config = argparse.Namespace()
    config.num_epochs = args.num_epochs
    config.patience = args.patience
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.warmup_epochs = args.warmup_epochs
    config.save_path = save_path
    
    # Set loss function and data augmentation
    criterion = NtXentLoss()
    augmentor = DataAugmentation()
    
    # Create trainer and start training
    trainer = SSLTrainer(
        model=model,
        data_loader=train_loader,
        config=config,
        criterion=criterion,
        augmentor=augmentor
    )
    
    model, results_df = trainer.train()
    
    # Save training results
    results_df.to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
    print(f"CSI pretraining completed! Model and results saved to {save_path}")
    
    return model, results_df


def pretrain_acf(args):
    print(f"Starting ACF modality pretraining...")
    
    # Prepare data
    train_loader = load_acf_data_unsupervised(args.acf_data_dir, args.batch_size)
    
    # Load model
    model = load_model_unsupervised_joint(
        win_len=args.win_len,
        feature_size=args.feature_size,
        depth=args.depth,
        in_channels=args.in_channels
    )
    
    # Set save path
    save_path = os.path.join(args.output_dir, args.results_subdir, f"{args.model_name}_acf")
    os.makedirs(save_path, exist_ok=True)
    
    # Set training configuration
    config = argparse.Namespace()
    config.num_epochs = args.num_epochs
    config.patience = args.patience
    config.learning_rate = args.learning_rate
    config.weight_decay = args.weight_decay
    config.warmup_epochs = args.warmup_epochs
    config.save_path = save_path
    
    # Set loss function and data augmentation
    criterion = NtXentLoss()
    augmentor = DataAugmentACF(feature_size=args.feature_size)
    
    # Create trainer and start training
    trainer = SSLTrainer(
        model=model,
        data_loader=train_loader,
        config=config,
        criterion=criterion,
        augmentor=augmentor
    )
    
    model, results_df = trainer.train()
    
    # Save training results
    results_df.to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
    print(f"ACF pretraining completed! Model and results saved to {save_path}")
    
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
        pretrain_csi(args)
    elif args.mode.lower() == 'acf':
        pretrain_acf(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Please choose 'csi' or 'acf'")


if __name__ == "__main__":
    main() 