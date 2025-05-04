import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import logging
from pathlib import Path
import datetime

# Import from our standalone module instead of the problematic module
from meta_learning_data import load_csi_data_benchmark
from model.meta_learning.meta_model import BaseMetaModel
from engine.meta_learning.meta_trainer import MetaTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# A list of available model types
AVAILABLE_MODELS = ['mlp', 'lstm', 'resnet18', 'transformer', 'vit']

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='Train a meta-learning model on WiFi benchmark dataset')
    parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset/tasks/motion_source_recognition',
                        help='Root directory of the dataset')
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                        help='Key in h5 file for CSI data')
    parser.add_argument('--n_way', type=int, default=2,
                        help='Number of classes per task')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of support examples per class')
    parser.add_argument('--q_query', type=int, default=5,
                        help='Number of query examples per class')
    parser.add_argument('--meta_batch_size', type=int, default=4,
                        help='Number of tasks per batch')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Outer loop learning rate')
    parser.add_argument('--meta_method', type=str, default='maml',
                        choices=['maml', 'lstm'], help='Meta-learning method')
    parser.add_argument('--inner_steps', type=int, default=5,
                        help='Number of inner loop steps')
    parser.add_argument('--num_iterations', type=int, default=60000,
                        help='Number of iterations for meta-learning')
    parser.add_argument('--meta_validation_interval', type=int, default=1000,
                        help='Interval for meta-validation')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    # Model parameters
    parser.add_argument('--model_type', type=str, default='vit', 
                        choices=AVAILABLE_MODELS,
                        help='Type of model to train')
    parser.add_argument('--win_len', type=int, default=232, 
                        help='Window length for WiFi CSI data')
    parser.add_argument('--feature_size', type=int, default=500, 
                        help='Feature size for WiFi CSI data')
    parser.add_argument('--in_channels', type=int, default=1, 
                        help='Number of input channels')
    parser.add_argument('--emb_dim', type=int, default=128, 
                        help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    loaders = load_csi_data_benchmark(
        data_dir=args.data_dir,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        batch_size=args.meta_batch_size,
        file_ext='.h5',
        data_key=args.data_key,
        num_workers=4
    )
    
    print(f"Available splits: {list(loaders.keys())}")
    
    # Create model using BaseMetaModel from the codebase
    print(f"Creating {args.model_type.upper()} meta-learning model...")
    model = BaseMetaModel(
        model_type=args.model_type,
        win_len=args.win_len,
        feature_size=args.feature_size,
        in_channels=args.in_channels,
        emb_dim=args.emb_dim,
        num_classes=args.n_way,
        dropout=args.dropout,
        inner_lr=args.inner_lr
    )
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create meta optimizer
    meta_optimizer = optim.Adam(model.parameters(), lr=args.meta_lr)
    
    # Create configuration object
    config = type('Config', (), {
        'meta_method': args.meta_method,
        'inner_lr': args.inner_lr,
        'meta_lr': args.meta_lr,
        'n_way': args.n_way,
        'k_shot': args.k_shot,
        'q_query': args.q_query,
        'meta_batch_size': args.meta_batch_size,
        'num_iterations': args.num_iterations,
        'meta_validation_interval': args.meta_validation_interval,
        'device': device,
        'save_dir': args.save_dir,
        'model_type': args.model_type
    })
    
    # Save the configuration
    save_path = os.path.join(args.save_dir, f"{args.model_type}_{args.meta_method}_config.json")
    with open(save_path, "w") as f:
        # Convert config to dict
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        json.dump(config_dict, f, indent=4)
    
    # Create meta-trainer
    meta_trainer = MetaTrainer(
        model=model,
        data_loader=loaders,
        config=config,
        meta_optimizer=meta_optimizer
    )
    
    # Train the model
    trained_model, training_records = meta_trainer.train()
    
    # Test the model if test loader is available
    if 'test' in loaders:
        print("\nTesting on test set...")
        test_results = meta_trainer.meta_test(loaders['test'], num_tasks=50, num_adaptation_steps=args.inner_steps)
        print(f"Test Results - Accuracy: {test_results['accuracy']:.4f}±{test_results['std_accuracy']:.4f}")
        
        # Save test results
        save_test_path = os.path.join(args.save_dir, f"{args.model_type}_{args.meta_method}_test_results.json")
        with open(save_test_path, "w") as f:
            json.dump(test_results, f, indent=4)
    
    print(f"\nTraining completed. Results saved to {args.save_dir}")

if __name__ == "__main__":
    main()
