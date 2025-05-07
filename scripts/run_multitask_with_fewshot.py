#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to run multitask learning with few-shot evaluation.
This script combines multitask learning with few-shot adaptation
to demonstrate how the model can adapt to new environments.

Usage:
    python run_multitask_with_fewshot.py --model transformer --tasks "MotionSourceRecognition,HumanMotion"
    
Additional flags:
    --evaluate_fewshot: Enable few-shot evaluation after multitask training
    --fewshot_task: Task to use for few-shot adaptation (must be one of the multitask tasks)
    --fewshot_support_split: Split to use for support set (default: val_id)
    --fewshot_query_split: Split to use for query set (default: test_cross_env)
    --epochs: Number of epochs for multitask training
    --batch_size: Batch size for training
    --output_dir: Directory to save results
"""

import os
import sys
import argparse
import subprocess

def main():
    """Run multitask training with few-shot evaluation"""
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run multitask learning with few-shot evaluation')
    
    # Pipeline parameters
    parser.add_argument('--model', type=str, default='transformer',
                        help='Model architecture to use (transformer, patchtst, timesformer1d)')
    parser.add_argument('--tasks', type=str, default='MotionSourceRecognition,HumanMotion',
                        help='Comma-separated list of tasks to train on')
    parser.add_argument('--training_dir', type=str, 
                        default='wifi_benchmark_dataset',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank for parameter-efficient fine-tuning')
                        
    # Few-shot evaluation parameters
    parser.add_argument('--evaluate_fewshot', action='store_true',
                        help='Evaluate few-shot learning after multitask training')
    parser.add_argument('--fewshot_task', type=str, default=None,
                        help='Task to use for few-shot adaptation (must be one of the multitask tasks)')
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
    
    # Verify fewshot_task is included in the main tasks
    if args.evaluate_fewshot:
        task_list = args.tasks.split(',')
        if args.fewshot_task is None:
            # Default to first task if not specified
            args.fewshot_task = task_list[0]
        elif args.fewshot_task not in task_list:
            print(f"Error: fewshot_task '{args.fewshot_task}' must be one of the multitask tasks: {task_list}")
            return
    
    # Build direct command to run train_multitask_adapter.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_multitask_adapter.py")
    
    cmd = [
        "python", train_script,
        f"--data_dir={args.training_dir}",
        f"--tasks={args.tasks}",
        f"--model_type={args.model}",
        f"--batch_size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--lr={args.lr}",
        f"--save_dir={args.output_dir}/multitask",
        f"--lora_r={args.lora_r}",
        "--test_splits=all"
    ]
    
    # Run the multitask training
    print("Running multitask training...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    multitask_process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if the multitask training was successful
    if multitask_process.returncode != 0:
        print("Error during multitask training:")
        print(multitask_process.stderr)
        return
    
    # Extract model path from the output
    model_path = None
    for line in multitask_process.stdout.split('\n'):
        if "Saved multitask adapters" in line and ".pt" in line:
            # Extract the full path to the saved model
            start_idx = line.find("results/")
            if start_idx != -1:
                end_idx = line.find(".pt", start_idx) + 3
                model_path = line[start_idx:end_idx]
                break
    
    if model_path is None:
        print("Could not find model path in the output")
        return
    
    # Add few-shot evaluation if enabled
    if args.evaluate_fewshot:
        # Build command to run few-shot training
        fewshot_script = os.path.join(script_dir, "train_fewshot.py")
        
        fewshot_cmd = [
            "python", fewshot_script,
            f"--data_dir={args.training_dir}",
            f"--task_name={args.fewshot_task}",
            f"--model={args.model}",
            f"--model_path={model_path}",
            f"--batch_size={args.batch_size}",
            f"--adaptation_lr={args.fewshot_adaptation_lr}",
            f"--adaptation_steps={args.fewshot_adaptation_steps}",
            f"--save_dir={args.output_dir}/fewshot/multitask",
            f"--support_split={args.fewshot_support_split}",
            f"--query_split={args.fewshot_query_split}"
        ]
        
        if args.fewshot_finetune_all:
            fewshot_cmd.append("--finetune_all")
        
        if args.fewshot_eval_shots:
            fewshot_cmd.append("--eval_shots")
        
        # Print and run the command
        print("\nRunning few-shot adaptation on multitask model...")
        print(f"Command: {' '.join(fewshot_cmd)}")
        
        # Run the command
        subprocess.run(fewshot_cmd)
    
    print("\nTraining and evaluation completed.")
    print("Results are saved in the specified output directory.")

if __name__ == '__main__':
    main() 