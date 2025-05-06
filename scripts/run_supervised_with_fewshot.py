#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple script to run supervised learning with few-shot evaluation.
This script combines supervised learning with few-shot adaptation
to demonstrate how the model can adapt to new environments.

Usage:
    python run_supervised_with_fewshot.py --model vit --task MotionSourceRecognition
    
Additional flags:
    --evaluate_fewshot: Enable few-shot evaluation after supervised training
    --fewshot_support_split: Split to use for support set (default: val_id)
    --fewshot_query_split: Split to use for query set (default: test_cross_env)
    --epochs: Number of epochs for supervised training
    --batch_size: Batch size for training
    --output_dir: Directory to save results
"""

import os
import sys
import argparse
import subprocess

def main():
    """Run supervised training with few-shot evaluation"""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run supervised learning with few-shot evaluation')
    
    # Pipeline parameters
    parser.add_argument('--model', type=str, default='vit',
                        help='Model architecture to use (mlp, lstm, resnet18, transformer, vit)')
    parser.add_argument('--task', type=str, default='MotionSourceRecognition',
                        help='Task to run (MotionSourceRecognition, etc.)')
    parser.add_argument('--training_dir', type=str, 
                        default='C:\\Guozhen\\Code\\Github\\WiAL-Real-WiFi-Sensing-Benchmark\\wifi_benchmark_dataset',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
                        
    # Few-shot evaluation parameters
    parser.add_argument('--evaluate_fewshot', action='store_true',
                        help='Evaluate few-shot learning after supervised training')
    parser.add_argument('--fewshot_support_split', type=str, default='val_id',
                        help='Split to use for few-shot support set (few examples from new environment)')
    parser.add_argument('--fewshot_query_split', type=str, default='test_cross_env',
                        help='Split to use for query set (testing in new environment)')
    parser.add_argument('--fewshot_adaptation_lr', type=float, default=0.01,
                        help='Learning rate for few-shot adaptation')
    parser.add_argument('--fewshot_adaptation_steps', type=int, default=10,
                        help='Number of adaptation steps for few-shot learning')
    parser.add_argument('--fewshot_finetune_all', action='store_true',
                        help='Fine-tune all model parameters instead of just the classifier')
    parser.add_argument('--fewshot_eval_shots', action='store_true',
                        help='Evaluate different shot values (1, 3, 5, 10) and compare')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Build direct command to run train_supervised.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_supervised.py")
    
    cmd = [
        "python", train_script,
        f"--data_dir={args.training_dir}",
        f"--task_name={args.task}",
        f"--model={args.model}",
        f"--batch_size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--save_dir={args.output_dir}",
        f"--output_dir={args.output_dir}",
        "--test_splits=all"
    ]
    
    # Add few-shot evaluation parameters if enabled
    if args.evaluate_fewshot:
        cmd.append("--evaluate_fewshot")
        cmd.append(f"--fewshot_support_split={args.fewshot_support_split}")
        cmd.append(f"--fewshot_query_split={args.fewshot_query_split}")
        cmd.append(f"--fewshot_adaptation_lr={args.fewshot_adaptation_lr}")
        cmd.append(f"--fewshot_adaptation_steps={args.fewshot_adaptation_steps}")
        
        if args.fewshot_finetune_all:
            cmd.append("--fewshot_finetune_all")
        
        if args.fewshot_eval_shots:
            cmd.append("--fewshot_eval_shots")
    
    # Print and run the command
    print("Running supervised training with few-shot evaluation...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    subprocess.run(cmd)
    
    print("\nTraining and evaluation completed.")
    print("Results are saved in the specified output directory.")

if __name__ == '__main__':
    main() 