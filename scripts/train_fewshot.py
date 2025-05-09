#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import pandas as pd
import json
import hashlib
import time
from tqdm import tqdm

# Load benchmark data
from load.supervised.benchmark_loader import load_benchmark_supervised

# Import few-shot model and trainer
from model.fewshot import FewShotAdaptiveModel, FewShotTrainer

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Few-shot adaptation of WiFi sensing models for new settings')
        
        # Dataset parameters
        parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                            help='Root directory of the dataset')
        parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                            help='Name of the task to train on')
        parser.add_argument('--data_key', type=str, default='CSI_amps',
                            help='Key for CSI data in h5 files')
        
        # Model parameters
        parser.add_argument('--model', type=str, default='vit', 
                            choices=['mlp', 'lstm', 'resnet18', 'transformer', 'vit', 'patchtst', 'timesformer1d'],
                            help='Type of model to load')
        parser.add_argument('--model_path', type=str, required=True,
                            help='Path to the pre-trained model checkpoint')
        
        # Few-shot adaptation parameters
        parser.add_argument('--adaptation_lr', type=float, default=0.01,
                            help='Learning rate for few-shot adaptation')
        parser.add_argument('--adaptation_steps', type=int, default=10,
                            help='Number of adaptation steps for few-shot learning')
        parser.add_argument('--finetune_all', action='store_true',
                            help='Fine-tune all model parameters instead of just the classifier')
        parser.add_argument('--k_shots', type=int, default=5,
                            help='Number of examples per class for few-shot adaptation')
        parser.add_argument('--eval_shots', action='store_true',
                            help='Evaluate different shot values (1, 3, 5, 10) and compare')
        
        # Other parameters
        parser.add_argument('--batch_size', type=int, default=32, 
                            help='Batch size for data loaders')
        parser.add_argument('--save_dir', type=str, default='results/fewshot',
                            help='Directory to save results')
        parser.add_argument('--support_split', type=str, default='val_id',
                            help='Split to use for few-shot support set (few examples from new environment)')
        parser.add_argument('--query_split', type=str, default='test_cross_env',
                            help='Split to use for query set (testing in new environment)')
        parser.add_argument('--no_compare', action='store_true',
                            help='Skip comparison with/without few-shot adaptation')
        
        # Model shape parameters (must match pre-trained model)
        parser.add_argument('--win_len', type=int, default=500, 
                            help='Window length for WiFi CSI data')
        parser.add_argument('--feature_size', type=int, default=98, 
                            help='Feature size for WiFi CSI data')
        parser.add_argument('--in_channels', type=int, default=1, 
                            help='Number of input channels')
        parser.add_argument('--emb_dim', type=int, default=128, 
                            help='Embedding dimension for models')
                            
        args = parser.parse_args()
    
    # Create unique experiment ID
    param_str = f"{args.model}_{args.task_name}_{args.adaptation_lr}_{args.adaptation_steps}_{args.k_shots}"
    experiment_id = f"fewshot_{hashlib.md5(param_str.encode()).hexdigest()[:10]}"
    
    # Create save directories
    save_dir = os.path.join(args.save_dir, args.task_name, args.model, experiment_id)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Experiment ID: {experiment_id}")
    print(f"Results will be saved to: {save_dir}")
    
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
        num_workers=4,
        shuffle_train=True,
        train_split="train_id",
        val_split=args.support_split,  # Use specified split for support set
        test_splits=[args.query_split],  # Use specified split for query set
        use_root_as_task_dir=False  # 本地训练模式下不直接使用根目录作为任务目录
    )
    
    # Extract data from the returned dictionary
    loaders = data['loaders']
    num_classes = data['num_classes']
    label_mapper = data['label_mapper']
    
    # Get support and query loaders
    support_loader = loaders.get('val')  # Using validation set as support set for few-shot learning
    query_loader = loaders.get(f'test_{args.query_split}')
    
    if support_loader is None:
        print(f"Error: Support set '{args.support_split}' not found")
        return
        
    if query_loader is None:
        print(f"Error: Query set '{args.query_split}' not found")
        return
    
    print(f"Support set: {args.support_split}")
    print(f"Query set: {args.query_split}")
    
    # Create few-shot adaptive model from pre-trained model
    print(f"Loading pre-trained model from {args.model_path}...")
    fewshot_model = FewShotAdaptiveModel.from_pretrained(
        model_path=args.model_path,
        model_type=args.model,
        num_classes=num_classes,
        adaptation_lr=args.adaptation_lr,
        adaptation_steps=args.adaptation_steps,
        finetune_all=args.finetune_all,
        device=device,
        win_len=args.win_len,
        feature_size=args.feature_size,
        in_channels=args.in_channels,
        emb_dim=args.emb_dim
    )
    
    # Create few-shot trainer
    trainer = FewShotTrainer(
        base_model=fewshot_model,
        support_loader=support_loader,
        query_loader=query_loader,
        adaptation_steps=args.adaptation_steps,
        adaptation_lr=args.adaptation_lr,
        device=device,
        save_path=save_dir,
        finetune_all=args.finetune_all
    )
    
    # Save configuration
    config = {
        'model': args.model,
        'task': args.task_name,
        'num_classes': num_classes,
        'adaptation_lr': args.adaptation_lr,
        'adaptation_steps': args.adaptation_steps,
        'finetune_all': args.finetune_all,
        'k_shots': args.k_shots,
        'support_split': args.support_split,
        'query_split': args.query_split,
        'batch_size': args.batch_size,
        'win_len': args.win_len,
        'feature_size': args.feature_size,
        'experiment_id': experiment_id
    }
    
    config_path = os.path.join(save_dir, f"{args.model}_{args.task_name}_fewshot_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Compare performance with and without few-shot adaptation
    if not args.no_compare:
        print("\nComparing performance with and without few-shot adaptation...")
        comparison_results = trainer.compare_with_without_adaptation(save_results=True)
        
        # Print a summary of the comparison
        comparison = comparison_results['comparison']
        print("\nSummary of comparison:")
        print(f"Without adaptation - Accuracy: {comparison['base_accuracy']:.4f}, F1-score: {comparison['base_f1_score']:.4f}")
        print(f"With adaptation    - Accuracy: {comparison['adapted_accuracy']:.4f}, F1-score: {comparison['adapted_f1_score']:.4f}")
        print(f"Improvement        - Accuracy: {comparison['accuracy_improvement']:.4f}, F1-score: {comparison['f1_improvement']:.4f}")
    
    # Evaluate different k-shot values if requested
    if args.eval_shots:
        print("\nEvaluating performance with different numbers of shots...")
        k_shot_results = trainer.evaluate_support_set_sizes(
            k_shots_list=[1, 3, 5, 10],
            save_results=True
        )
        
        # Print a summary of k-shot performance
        print("\nSummary of k-shot performance:")
        print(f"0-shot (no adaptation) - Accuracy: {k_shot_results['0-shot']['accuracy']:.4f}, F1-score: {k_shot_results['0-shot']['f1_score']:.4f}")
        for k in [1, 3, 5, 10]:
            result = k_shot_results[f'{k}-shot']
            print(f"{k}-shot adaptation    - Accuracy: {result['accuracy']:.4f}, F1-score: {result['f1_score']:.4f} (Improvement: {result['accuracy_improvement']:.4f})")
    
    print("\nFew-shot adaptation evaluation completed.")
    print(f"Results saved to {save_dir}")
    
    return save_dir

if __name__ == '__main__':
    main() 