#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - Local Environment

This script allows you to run any of the three pipelines:
1. Pretraining (Self-supervised learning)
2. Supervised learning
3. Meta-learning

Usage:
    python local_runner.py --pipeline pretraining
    python local_runner.py --pipeline supervised
    python local_runner.py --pipeline meta
    
Additional parameters:
    --data_dir: Directory containing the input data
    --output_dir: Directory to save output results
    --mode: Data modality to use (csi or acf)
    --config_file: JSON configuration file to override defaults
"""

import os
import sys
import subprocess
import torch
import time
import argparse
import json
from datetime import datetime

#==============================================================================
# CONFIGURATION SECTION - MODIFY PARAMETERS HERE
#==============================================================================

# Pipeline to run: 'pretraining', 'supervised', or 'meta'
PIPELINE = 'supervised'

# Data and Output Directories
DATA_DIR = r"H:\CSIMAT100"
OUTPUT_DIR = r"C:\Users\weiha\Desktop\bench_mark_output"

# Data Modality
MODE = 'csi'  # Options: 'csi', 'acf'

# Supervised Learning Options
PRETRAINED = False  # Use pretrained model for supervised learning
PRETRAINED_MODEL = None  # Path to pretrained model (when PRETRAINED is True)
FREEZE_BACKBONE = False  # Freeze backbone network for supervised learning
INTEGRATED_LOADER = True  # Use integrated data loader for supervised learning
TASK = 'FourClass'  # Task type for integrated loader (e.g., ThreeClass, HumanNonhuman)

# Model Parameters
WIN_LEN =500  # Window length for CSI data
FEATURE_SIZE = 232  # Feature size for CSI data

# Common Training Parameters
SEED = 42
BATCH_SIZE = 16
MODEL_NAME = 'WiT'

# Advanced Configuration
CONFIG_FILE = None  # Path to JSON configuration file to override defaults

#==============================================================================
# END OF CONFIGURATION SECTION
#==============================================================================

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Set device string for command line arguments
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_command(cmd, display_output=True):
    """Run a command and stream the output."""
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True
    )
    
    # Stream the output
    output = []
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        if display_output:
            print(line.rstrip())
        output.append(line.rstrip())
    
    process.wait()
    return process.returncode, '\n'.join(output)

# Configure pretraining pipeline
def get_pretrain_config(data_dir=None, output_dir=None, mode='csi'):
    """Get configuration for pretraining pipeline."""
    # Set default paths if not provided
    if data_dir is None:
        data_dir = DATA_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    return {
        # Data parameters
        'csi_data_dir': os.path.join(data_dir, 'csi'),
        'acf_data_dir': os.path.join(data_dir, 'acf'),
        'output_dir': output_dir,
        'results_subdir': 'ssl_pretrain',
        
        # Training parameters
        'batch_size': BATCH_SIZE,
        'learning_rate': 1e-5,
        'weight_decay': 0.001,
        'num_epochs': 100,
        'warmup_epochs': 5,
        'patience': 20,
        
        # Model parameters
        'mode': mode,  # Options: 'csi', 'acf'
        'depth': 6,
        'in_channels': 1,
        'emb_size': 128,
        'freq_out': 10,
        
        # Other parameters
        'seed': SEED,
        'device': DEVICE,
        'model_name': MODEL_NAME
    }

# Configure supervised learning pipeline
def get_supervised_config(data_dir=None, output_dir=None, mode='csi', pretrained=False, pretrained_model=None):
    """Get configuration for supervised learning pipeline."""
    # Set default paths if not provided
    if data_dir is None:
        data_dir = DATA_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    config = {
        # Data parameters
        'csi_data_dir': data_dir ,
        'acf_data_dir': data_dir,
        'output_dir': output_dir,
        'results_subdir': 'supervised',
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'max_samples': 5000,  # Limit maximum samples to avoid memory issues
        
        # Training parameters
        'batch_size': 16,  # Reduced batch size to avoid memory issues
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': 100,
        'warmup_epochs': 5,
        'patience': 15,
        
        # Model parameters
        'mode': mode,  # Options: 'csi', 'acf'
        'num_classes': 2,
        'freeze_backbone': False,
        'pretrained': pretrained,
        'sample_rate': 100,  # Add sample_rate parameter
        
        # Integrated loader options
        'integrated_loader': INTEGRATED_LOADER,
        'task': TASK,
        
        # Other parameters
        'seed': SEED,
        'device': DEVICE,
        'model_name': MODEL_NAME,
        'unseen_test': False,
        'win_len': WIN_LEN,
        'feature_size': FEATURE_SIZE
    }
    
    if pretrained and pretrained_model:
        config['pretrained_model'] = pretrained_model
    
    return config

# Configure meta-learning pipeline
def get_meta_config(data_dir=None, output_dir=None):
    """Get configuration for meta-learning pipeline."""
    # Set default paths if not provided
    if data_dir is None:
        data_dir = DATA_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    return {
        # Data parameters
        'data_dir': os.path.join(data_dir, 'benchmark'),
        'output_dir': output_dir,
        'results_subdir': 'meta_learning',
        'resize_height': 64,
        'resize_width': 64,
        
        # Meta-learning parameters
        'meta_method': 'maml',  # Options: 'maml', 'lstm'
        'meta_batch_size': 4,
        'inner_lr': 0.01,
        'meta_lr': 0.001,
        'num_iterations': 10000,
        'meta_validation_interval': 1000,
        
        # Task parameters
        'n_way': 2,
        'k_shot': 5,
        'q_query': 15,
        
        # Model parameters
        'model_type': 'csi',
        'emb_size': 128,
        'depth': 6,
        'in_channels': 1,
        
        # Other parameters
        'seed': SEED,
        'device': DEVICE,
        'model_name': f"{MODEL_NAME}_Meta"
    }

def run_pretraining(config):
    """Run pretraining pipeline."""
    print("\n========== Running Pretraining Pipeline ==========\n")
    
    # Build command
    cmd = "python pretrain.py "
    
    # Add all arguments
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd += f"--{key} "
        else:
            cmd += f"--{key} {value} "
    
    print(f"Running command: {cmd}")
    start_time = time.time()
    returncode, output = run_command(cmd)
    end_time = time.time()
    
    if returncode == 0:
        print(f"\nPretraining completed successfully in {(end_time - start_time)/60:.2f} minutes")
    else:
        print(f"\nPretraining failed with return code {returncode}")
    
    return returncode == 0

def run_supervised(config):
    """Run supervised learning pipeline."""
    print("\n========== Running Supervised Learning Pipeline ==========\n")
    
    # Build command
    cmd = "python train_supervised.py "
    
    # Add all arguments
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd += f"--{key.replace('_', '-')} "
        else:
            cmd += f"--{key.replace('_', '-')} {value} "
    
    print(f"Running command: {cmd}")
    start_time = time.time()
    returncode, output = run_command(cmd)
    end_time = time.time()
    
    if returncode == 0:
        print(f"\nSupervised learning completed successfully in {(end_time - start_time)/60:.2f} minutes")
    else:
        print(f"\nSupervised learning failed with return code {returncode}")
    
    return returncode == 0

def run_meta_learning(config):
    """Run meta-learning pipeline."""
    print("\n========== Running Meta-Learning Pipeline ==========\n")
    
    # Build command
    cmd = "python meta_learning.py "
    
    # Add all arguments
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd += f"--{key} "
        else:
            cmd += f"--{key} {value} "
    
    print(f"Running command: {cmd}")
    start_time = time.time()
    returncode, output = run_command(cmd)
    end_time = time.time()
    
    if returncode == 0:
        print(f"\nMeta-learning completed successfully in {(end_time - start_time)/60:.2f} minutes")
    else:
        print(f"\nMeta-learning failed with return code {returncode}")
    
    return returncode == 0

def main():
    """Parse arguments and run the specified pipeline."""
    parser = argparse.ArgumentParser(description='Run WiFi Sensing Pipelines')
    parser.add_argument('--pipeline', type=str, 
                        choices=['pretraining', 'supervised', 'meta'],
                        help='Which pipeline to run')
    
    # Data and output directories
    parser.add_argument('--data_dir', type=str,
                        help='Directory containing the input data')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save output results')
    
    # Optional configuration overrides
    parser.add_argument('--mode', type=str, choices=['csi', 'acf'],
                        help='Data modality to use (csi or acf)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model for supervised learning')
    parser.add_argument('--pretrained_model', type=str,
                        help='Path to pretrained model for supervised learning')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone network for supervised learning')
    parser.add_argument('--integrated_loader', action='store_true',
                        help='Use integrated data loader for supervised learning')
    parser.add_argument('--task', type=str,
                        help='Task type for integrated loader (e.g., ThreeClass, HumanNonhuman)')
    parser.add_argument('--config_file', type=str,
                        help='JSON configuration file to override defaults')
    parser.add_argument('--max_samples', type=int,
                        help='Maximum number of samples to load (to prevent memory issues)')
    
    args = parser.parse_args()
    
    # Use command line arguments if provided, otherwise use defaults from configuration section
    pipeline = args.pipeline if args.pipeline else PIPELINE
    data_dir = args.data_dir if args.data_dir else DATA_DIR
    output_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    mode = args.mode if args.mode else MODE
    pretrained = args.pretrained if args.pretrained else PRETRAINED
    pretrained_model = args.pretrained_model if args.pretrained_model else PRETRAINED_MODEL
    freeze_backbone = args.freeze_backbone if args.freeze_backbone else FREEZE_BACKBONE
    integrated_loader = args.integrated_loader if args.integrated_loader else INTEGRATED_LOADER
    task = args.task if args.task else TASK
    config_file = args.config_file if args.config_file else CONFIG_FILE
    max_samples = args.max_samples if args.max_samples else 5000
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the appropriate configuration based on the pipeline
    if pipeline == 'pretraining':
        config = get_pretrain_config(data_dir, output_dir, mode)
    elif pipeline == 'supervised':
        config = get_supervised_config(
            data_dir, 
            output_dir,
            mode,
            pretrained,
            pretrained_model
        )
        if freeze_backbone:
            config['freeze_backbone'] = True
        if integrated_loader:
            config['integrated_loader'] = True
            config['task'] = task
    elif pipeline == 'meta':
        config = get_meta_config(data_dir, output_dir)
    else:
        print(f"Error: Unknown pipeline '{pipeline}'. Please choose from 'pretraining', 'supervised', or 'meta'.")
        return
    
    # Load configuration from file if specified
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_overrides = json.load(f)
            config.update(config_overrides)
    
    # Run the specified pipeline
    if pipeline == 'pretraining':
        run_pretraining(config)
    elif pipeline == 'supervised':
        run_supervised(config)
    elif pipeline == 'meta':
        run_meta_learning(config)

if __name__ == "__main__":
    main()
