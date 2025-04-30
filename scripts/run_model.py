#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Training Wrapper Script

This script provides a simple interface to train models by automatically selecting
the appropriate configuration file based on the model name and task.

Usage:
    python scripts/run_model.py --model [model_name] --task [task_name]
    
Example:
    python scripts/run_model.py --model lstm --task MotionSourceRecognition
"""

import os
import sys
import argparse
import json
import subprocess

# Directory paths
CONFIG_DIR = "configs"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# Available models
AVAILABLE_MODELS = ['mlp', 'lstm', 'resnet18', 'transformer', 'vit']

# Available tasks (can be expanded)
AVAILABLE_TASKS = ['MotionSourceRecognition', 'HumanMotion', 'DetectionandClassification', 'HumanID', 'NTUHAR']

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a WiFi sensing model with automatic config selection')
    parser.add_argument('--model', type=str, required=True, choices=AVAILABLE_MODELS,
                        help='Model architecture to train')
    parser.add_argument('--task', type=str, required=True,
                        help='Task to train the model on')
    parser.add_argument('--custom_config', type=str, default=None,
                        help='Optional path to a custom config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (overrides config)')
    
    args = parser.parse_args()
    
    # If task is not in the list of available tasks, warn the user
    if args.task not in AVAILABLE_TASKS:
        print(f"Warning: Task '{args.task}' is not in the list of known tasks: {AVAILABLE_TASKS}")
        print("Continuing anyway, but you may want to verify your task name.")
    
    # Determine the config file to use
    if args.custom_config:
        config_file = args.custom_config
        print(f"Using custom config file: {config_file}")
    else:
        # Look for a model-specific config in the configs directory
        model_config = os.path.join(CONFIG_DIR, f"{args.model}_config.json")
        if os.path.exists(model_config):
            config_file = model_config
            print(f"Using existing config file: {config_file}")
        else:
            # Create a new config file based on the model and task
            config_file = create_config_file(args.model, args.task)
            print(f"Created new config file: {config_file}")
    
    # Modify the config if any overrides are specified
    if any([args.epochs, args.batch_size, args.output_dir]):
        modify_config(config_file, args)
    
    # Run the local_runner with the selected config
    command = f"python scripts/local_runner.py --pipeline supervised --config_file {config_file}"
    print(f"Running command: {command}")
    
    # Execute the command
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return 1
    
    return 0

def create_config_file(model, task):
    """Create a new configuration file for the specified model and task."""
    # Base configuration template
    config = {
        "model_name": model,
        "task": task,
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "win_len": 232,
        "feature_size": 500,
        "results_subdir": f"{model}_{task.lower()}",
        "training_dir": "wifi_benchmark_dataset",
        "output_dir": f"./results/{model}_{task.lower()}"
    }
    
    # Add model-specific parameters
    if model in ['transformer', 'vit']:
        config["emb_dim"] = 128
        config["dropout"] = 0.1
        config["learning_rate"] = 0.0005  # Transformers often need lower learning rate
    
    if model in ['resnet18', 'vit']:
        config["in_channels"] = 1
    
    # Create the config file
    config_filename = os.path.join(CONFIG_DIR, f"{model}_{task.lower()}_config.json")
    
    # Make sure the configs directory exists
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_filename

def modify_config(config_file, args):
    """Modify configuration file with command line overrides."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Apply overrides
    if args.epochs:
        config['num_epochs'] = args.epochs
        print(f"Overriding number of epochs: {args.epochs}")
    
    if args.batch_size:
        config['batch_size'] = args.batch_size
        print(f"Overriding batch size: {args.batch_size}")
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
        print(f"Overriding output directory: {args.output_dir}")
    
    # Save modified config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    sys.exit(main()) 