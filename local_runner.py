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
# For supervised learning with integrated loader, DATA_DIR should point directly 
# to the folder containing .mat files or to the parent folder containing these files
# Do not use subdirectories like 'csi' or 'acf' in the path unless your files are organized this way
DATA_DIR = "C:\\Guozhen\\Code\\Github\\WiFiSSL\\dataset\\task\\HM3\\CSIMAT100"
OUTPUT_DIR = "C:\\Guozhen\\Code\\Github\\temp"

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
BATCH_SIZE = 2
EPOCH_NUMBER = 1  # Number of training epochs
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

def run_command(cmd, display_output=True, timeout=1800):
    """
    Run command and display output in real-time with timeout handling.
    
    Args:
        cmd: Command to execute
        display_output: Whether to display command output
        timeout: Command execution timeout in seconds, default 30 minutes
        
    Returns:
        Tuple of (return_code, output_string)
    """
    try:
        # Start process
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            shell=True
        )
        
        # For storing output
        output = []
        start_time = time.time()
        
        # Main loop
        while process.poll() is None:
            # Check for timeout
            if timeout and time.time() - start_time > timeout:
                if display_output:
                    print(f"\nError: Command execution timed out ({timeout} seconds), terminating...")
                process.kill()
                return -1, '\n'.join(output + [f"Error: Command execution timed out ({timeout} seconds)"])
            
            # Read output line by line without blocking
            try:
                line = process.stdout.readline()
                if line:
                    line = line.rstrip()
                    if display_output:
                        print(line)
                    output.append(line)
                else:
                    # Small sleep to reduce CPU usage
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error reading output: {str(e)}")
                time.sleep(0.1)
        
        # Ensure all remaining output is read
        remaining_output, _ = process.communicate()
        if remaining_output:
            for line in remaining_output.splitlines():
                if display_output:
                    print(line)
                output.append(line)
                
        return process.returncode, '\n'.join(output)
        
    except KeyboardInterrupt:
        # User interruption
        if 'process' in locals() and process.poll() is None:
            print("\nUser interrupted, terminating process...")
            process.kill()
        return -2, "User interrupted execution"
        
    except Exception as e:
        # Other exceptions
        error_msg = f"Error executing command: {str(e)}"
        if display_output:
            print(f"\nError: {error_msg}")
        
        # Kill process if still running
        if 'process' in locals() and process.poll() is None:
            process.kill()
        
        return -1, error_msg

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
        'num_epochs': EPOCH_NUMBER,
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
        'csi_data_dir': data_dir,  # Direct use of data_dir without joining
        'acf_data_dir': data_dir,  # Direct use of data_dir without joining
        'output_dir': output_dir,
        'results_subdir': 'supervised',
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        
        # Training parameters
        'batch_size': BATCH_SIZE,  # Use global BATCH_SIZE instead of hardcoded value
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': EPOCH_NUMBER,
        'warmup_epochs': 5,
        'patience': 15,
        
        # Model parameters
        'mode': mode,  # Options: 'csi', 'acf'
        'num_classes': 2,
        'freeze_backbone': FREEZE_BACKBONE,  # Use global setting
        'pretrained': pretrained,
        
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
            # Add quotes for path-like arguments to handle spaces and special characters
            if key in ['csi_data_dir', 'acf_data_dir', 'output_dir', 'pretrained_model', 'results_subdir']:
                cmd += f"--{key} \"{value}\" "
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

def run_supervised_direct(config):
    """直接导入并运行train_supervised模块"""
    print("\n========== 运行监督学习管道 ==========\n")
    
    # 保存原始参数
    old_argv = sys.argv.copy()
    
    # 构建新参数列表
    sys.argv = ["train_supervised.py"]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                sys.argv.append(f"--{key.replace('_', '-')}")
        else:
            sys.argv.append(f"--{key.replace('_', '-')}")
            sys.argv.append(f"{value}")
    
    print(f"运行命令: {' '.join(sys.argv)}")
    
    # 直接导入并运行
    try:
        start_time = time.time()
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_supervised", "./train_supervised.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 假设train_supervised.py中有main函数
        if hasattr(module, 'main'):
            module.main()
        
        end_time = time.time()
        print(f"\n监督学习完成，耗时 {(end_time - start_time)/60:.2f} 分钟")
        success = True
    except Exception as e:
        print(f"\n监督学习失败: {str(e)}")
        success = False
    finally:
        # 恢复原始参数
        sys.argv = old_argv
    
    return success

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
            # Add quotes for path-like arguments to handle spaces and special characters
            if key in ['data_dir', 'output_dir', 'results_subdir']:
                cmd += f"--{key} \"{value}\" "
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
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    
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
    epochs = args.epochs if args.epochs else EPOCH_NUMBER
    
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
    
    # 更新epochs参数
    if args.epochs:
        config['num_epochs'] = epochs
        print(f"Using custom epoch number: {epochs}")
    
    # Load configuration from file if specified
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_overrides = json.load(f)
            config.update(config_overrides)
    
    # Run the specified pipeline
    if pipeline == 'pretraining':
        run_pretraining(config)
    elif pipeline == 'supervised':
        run_supervised_direct(config)
    elif pipeline == 'meta':
        run_meta_learning(config)

if __name__ == "__main__":
    main()
