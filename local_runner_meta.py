import os
import torch
import numpy as np
import random
import argparse

from load.meta_learning.data_loader import load_csi_data_benchmark
from load.meta_learning.model_loader import load_meta_model
from engine.meta_learning.meta_trainer import MetaTrainer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_meta_learning_direct(config):
    """
    Run meta-learning pipeline directly using a config dict or Namespace.
    """
    # Convert config to Namespace if it's a dict
    if isinstance(config, dict):
        args = argparse.Namespace(**config)
    else:
        args = config

    # Set random seed
    set_seed(getattr(args, 'seed', 42))

    # Set device
    if not hasattr(args, 'device') or args.device is None or args.device == 'cuda' and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Prepare output directory
    save_path = os.path.join(
        args.output_dir, args.results_subdir,
        f"{args.model_type}_{args.meta_method}_{args.n_way}way_{args.k_shot}shot"
    )
    os.makedirs(save_path, exist_ok=True)

    # Load meta-learning data
    data_loader = load_csi_data_benchmark(
        data_dir=args.training_dir,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        batch_size=args.meta_batch_size
    )

    # Load meta-learning model
    model = load_meta_model(
        model_type=args.model_type,
        win_len=args.win_len,
        feature_size=args.feature_size,
        n_way=args.n_way,
        device=device,
        in_channels=getattr(args, 'in_channels', 1),
        emb_dim=getattr(args, 'emb_dim', 128)
    )

    # Set up meta-learning config
    meta_config = argparse.Namespace()
    meta_config.meta_method = args.meta_method
    meta_config.inner_lr = args.inner_lr
    meta_config.meta_lr = args.meta_lr
    meta_config.n_way = args.n_way
    meta_config.k_shot = args.k_shot
    meta_config.q_query = args.q_query
    meta_config.meta_batch_size = args.meta_batch_size
    meta_config.num_iterations = args.num_iterations
    meta_config.meta_validation_interval = getattr(args, 'meta_validation_interval', 1000)
    meta_config.device = device
    meta_config.save_path = save_path

    # Add these lines:
    meta_config.output_dir = args.output_dir
    meta_config.results_subdir = args.results_subdir
    meta_config.model_name = f"{args.model_type}_{args.meta_method}_{args.n_way}way_{args.k_shot}shot"

    # Create meta-learning trainer
    trainer = MetaTrainer(
        model=model,
        data_loader={'train': data_loader.get('train'), 'val': data_loader.get('val')},
        config=meta_config
    )

    # Start training
    print(f"Starting meta-learning training with {args.model_type} model...")
    model, results = trainer.train()
    print("Meta-learning training completed!")

    return model, results

# Example usage:
if __name__ == "__main__":
    # Example config (replace with your own or parse from file/args)
    config = {
        'training_dir': "C:\\Guozhen\\Code\\Github\\WiFiSSL\\dataset\\task\\HM3\\CSIMAT100\\train",
        'output_dir': "C:\\Guozhen\\Code\\Github\\temp",
        'results_subdir': 'meta',
        'model_type': 'lstm',
        'meta_method': 'maml',
        'n_way': 2,
        'k_shot': 5,
        'q_query': 5,
        'meta_batch_size': 4,
        'num_iterations': 10000,
        'win_len': 250,
        'feature_size': 90,
        'in_channels': 1,
        'emb_dim': 128,
        'inner_lr': 0.01,
        'meta_lr': 0.001,
        'seed': 42,
        'device': 'cuda'
    }
    run_meta_learning_direct(config)