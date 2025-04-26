import os
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
from tqdm import tqdm

# Import meta training components
from engine.meta_learning.meta_trainer import MetaTrainer
from load.meta_learning.model_loader import load_meta_model
from load.meta_learning.data_loader import load_csi_data_benchmark

def parse_args():
    """Parse command line arguments for meta-learning"""
    parser = argparse.ArgumentParser(description='Train meta-learning models')
    
    # Data directories
    parser.add_argument('--training-dir', type=str, default=None,
                      help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='experiments',
                      help='Directory to save output results')
    parser.add_argument('--results-subdir', type=str, default='meta',
                      help='Subdirectory under output directory to save results')
    
    # Meta-learning parameters
    parser.add_argument('--meta-method', type=str, default='maml', choices=['maml', 'lstm'],
                      help='Meta-learning method (maml or lstm)')
    parser.add_argument('--n-way', type=int, default=3,
                      help='N-way classification for meta-learning tasks')
    parser.add_argument('--k-shot', type=int, default=5,
                      help='K-shot learning (samples per class in support set)')
    parser.add_argument('--q-query', type=int, default=5,
                      help='Number of query samples per class')
    parser.add_argument('--meta-batch-size', type=int, default=4,
                      help='Number of tasks per meta-batch')
    parser.add_argument('--inner-lr', type=float, default=0.01,
                      help='Inner loop learning rate')
    parser.add_argument('--meta-lr', type=float, default=0.001,
                      help='Meta optimizer learning rate')
    parser.add_argument('--num-iterations', type=int, default=10000,
                      help='Number of meta-training iterations')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='vit',
                      choices=['mlp', 'lstm', 'resnet18', 'transformer', 'vit'],
                      help='Model architecture')
    parser.add_argument('--emb-dim', type=int, default=128,
                      help='Embedding dimension')
    parser.add_argument('--win-len', type=int, default=250,
                      help='CSI window length')
    parser.add_argument('--feature-size', type=int, default=90,
                      help='CSI feature size')
    parser.add_argument('--in-channels', type=int, default=1,
                      help='Number of input channels')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to train on (cuda or cpu)')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    """Main function for meta-learning"""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Ensure output directory exists
    os.makedirs(os.path.join(args.output_dir, args.results_subdir), exist_ok=True)
    
    # Load data
    data_loader = load_csi_data_benchmark(
        data_dir=args.training_dir,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query
    )
    
    # Load model
    model = load_meta_model(
        model_type=args.model_type,
        win_len=args.win_len,
        feature_size=args.feature_size,
        n_way=args.n_way,
        device=device,
        in_channels=args.in_channels,
        emb_dim=args.emb_dim
    )
    
    # Set save path
    save_path = os.path.join(args.output_dir, args.results_subdir, 
                           f"{args.model_type}_{args.meta_method}_{args.n_way}way_{args.k_shot}shot")
    os.makedirs(save_path, exist_ok=True)
    
    # Meta-learning configuration
    config = argparse.Namespace()
    config.meta_method = args.meta_method
    config.inner_lr = args.inner_lr
    config.meta_lr = args.meta_lr
    config.n_way = args.n_way
    config.k_shot = args.k_shot
    config.q_query = args.q_query
    config.meta_batch_size = args.meta_batch_size
    config.num_iterations = args.num_iterations
    config.device = device
    config.save_path = save_path
    
    # Create meta-learning trainer
    trainer = MetaTrainer(
        model=model,
        data_loader={'train': data_loader['train'], 'val': data_loader.get('val')},
        config=config
    )
    
    # Start training
    print(f"Starting meta-learning training with {args.model_type} model...")
    model, results = trainer.train()
    print("Meta-learning training completed!")

if __name__ == "__main__":
    main()
