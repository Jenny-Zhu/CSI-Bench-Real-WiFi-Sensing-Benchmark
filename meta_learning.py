#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry script for meta-learning pipeline
"""

import argparse
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import the new module structure
from engine.meta_learning.meta_trainer import MetaTrainer
from load import (
    load_csi_data_benchmark,
    load_csi_model_benchmark
)


def parse_args():
    parser = argparse.ArgumentParser(description='Meta-learning for WiFi signals')

    # Meta-learning parameters
    parser.add_argument('--meta-method', type=str, default='maml', choices=['maml', 'lstm'], help='Meta-learning method')
    parser.add_argument('--meta-batch-size', type=int, default=4, help='Meta batch size')
    parser.add_argument('--inner-lr', type=float, default=0.01, help='Inner loop learning rate')
    parser.add_argument('--meta-lr', type=float, default=0.001, help='Meta learning rate')
    parser.add_argument('--num-iterations', type=int, default=60000, help='Number of training iterations')
    parser.add_argument('--meta-validation-interval', type=int, default=1000, help='Meta validation interval')
    
    # Task parameters
    parser.add_argument('--n-way', type=int, default=2, help='N-way classification')
    parser.add_argument('--k-shot', type=int, default=5, help='K-shot support set')
    parser.add_argument('--q-query', type=int, default=15, help='Query set size')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='csi', choices=['csi'], help='Model type')
    parser.add_argument('--emb-size', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=6, help='Model depth')
    parser.add_argument('--in-channels', type=int, default=1, help='Number of input channels')
    
    # Data parameters
    parser.add_argument('--resize-height', type=int, default=64, help='Resize height')
    parser.add_argument('--resize-width', type=int, default=64, help='Resize width')
    
    # Path configuration
    parser.add_argument('--data-dir', type=str, default='/opt/ml/input/data/benchmark', help='Benchmark data directory')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--model-name', type=str, default='WiT_Meta', help='Model name')
    parser.add_argument('--results-subdir', type=str, default='meta_learning', help='Results subdirectory')
    
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


def run_meta_learning(args):
    print(f"Starting meta-learning training using {args.meta_method.upper()} algorithm...")
    
    # Prepare data
    data_loader = load_csi_data_benchmark(
        data_dir=args.data_dir,
        resize_height=args.resize_height,
        resize_width=args.resize_width,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query
    )
    
    # Load model
    model = load_csi_model_benchmark(
        emb_size=args.emb_size,
        depth=args.depth,
        in_channels=args.in_channels,
        n_way=args.n_way
    )
    
    # Set save path
    save_path = os.path.join(args.output_dir, args.results_subdir, 
                           f"{args.model_name}_{args.meta_method}_{args.n_way}way_{args.k_shot}shot")
    os.makedirs(save_path, exist_ok=True)
    
    # Set meta-learning configuration
    config = argparse.Namespace()
    config.meta_method = args.meta_method
    config.inner_lr = args.inner_lr
    config.meta_lr = args.meta_lr
    config.n_way = args.n_way
    config.k_shot = args.k_shot
    config.q_query = args.q_query
    config.meta_batch_size = args.meta_batch_size
    config.num_iterations = args.num_iterations
    config.meta_validation_interval = args.meta_validation_interval
    config.save_path = save_path
    
    # Create meta-learning trainer
    trainer = MetaTrainer(
        model=model,
        data_loader={'train': data_loader['train'], 'val': data_loader['val'] if 'val' in data_loader else None},
        config=config
    )
    
    # Start training
    model, results = trainer.train()
    
    # Save training results
    meta_losses = results.get('meta_losses', [])
    meta_accuracies = results.get('meta_accuracies', [])
    val_losses = results.get('val_losses', [])
    val_accuracies = results.get('val_accuracies', [])
    
    # Create results DataFrame and save
    import pandas as pd
    results_df = pd.DataFrame({
        'iteration': range(1, len(meta_losses) + 1),
        'meta_loss': meta_losses,
        'meta_accuracy': meta_accuracies
    })
    
    if val_losses and val_accuracies:
        # Add validation results (note: validation results have different iteration intervals)
        val_iterations = list(range(args.meta_validation_interval, 
                                   args.meta_validation_interval * (len(val_losses) + 1), 
                                   args.meta_validation_interval))
        val_df = pd.DataFrame({
            'iteration': val_iterations,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        })
        # Merge training and validation results
        results_df = pd.merge_asof(results_df, val_df, on='iteration')
    
    results_df.to_csv(os.path.join(save_path, 'training_history.csv'), index=False)
    
    # Test model
    print("Evaluating model on test set...")
    if 'test' in data_loader:
        test_results = trainer.evaluate(data_loader['test'])
        print(f"Test results - Accuracy: {test_results['accuracy']:.4f}±{test_results['std_accuracy']:.4f}, Loss: {test_results['loss']:.4f}")
        
        # Save test results
        with open(os.path.join(save_path, 'test_results.txt'), 'w') as f:
            f.write(f"Test accuracy: {test_results['accuracy']:.4f}±{test_results['std_accuracy']:.4f}\n")
            f.write(f"Test loss: {test_results['loss']:.4f}\n")
            f.write(f"Adaptation curve: {test_results['adaptation_curve']}\n")
    
    print(f"Meta-learning training completed! Model and results saved to {save_path}")
    
    return model, results


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Run meta-learning training
    run_meta_learning(args)


if __name__ == "__main__":
    main()
