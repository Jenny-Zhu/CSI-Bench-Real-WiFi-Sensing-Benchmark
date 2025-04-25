#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - Local Environment

This script allows you to run any of the two pipelines:
1. Supervised learning
2. Meta-learning

Usage:
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

# Pipeline to run: 'supervised', or 'meta'
PIPELINE = 'supervised'

# Data and Output Directories
# For supervised learning with integrated loader, DATA_DIR should point directly 
# to the folder containing .mat files or to the parent folder containing these files
# Do not use subdirectories like 'csi' or 'acf' in the path unless your files are organized this way
DATA_DIR = "C:\\Users\\weiha\\Desktop\\demo"
OUTPUT_DIR = "C:\\Users\\weiha\\Desktop\\bench_mark_output"

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
    
    # Override with pretrained model path if provided
    if pretrained and pretrained_model:
        config['pretrained_model'] = pretrained_model
    
    return config


def get_meta_config(data_dir=None, output_dir=None):
    """Get configuration for meta-learning pipeline."""
    # Set default paths if not provided
    if data_dir is None:
        data_dir = DATA_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    return {
        # Data parameters
        'csi_data_dir': data_dir,
        'output_dir': output_dir,
        'results_subdir': 'meta',
        
        # Training parameters
        'batch_size': BATCH_SIZE,
        'learning_rate': 1e-4,
        'meta_lr': 0.01,
        'weight_decay': 1e-5,
        'num_epochs': EPOCH_NUMBER,
        'patience': 15,
        
        # Meta-learning parameters
        'n_way': 3,  # Number of classes per task
        'k_shot': 5,  # Number of examples per class
        'q_query': 5,  # Number of query examples
        'meta_batch_size': 4,  # Number of tasks per meta-batch
        
        # Model parameters
        'mode': MODE,  # Only 'csi' mode is currently supported for meta-learning
        
        # Other parameters
        'seed': SEED,
        'device': DEVICE,
        'model_name': MODEL_NAME
    }


def run_supervised_direct(config):
    """Run supervised learning pipeline directly without subprocess."""
    print(f"\n===== Running supervised learning pipeline directly =====")
    print(f"Data directory: {config.get('csi_data_dir', 'N/A') if config.get('mode') == 'csi' else config.get('acf_data_dir', 'N/A')}")
    print(f"Output directory: {config.get('output_dir', 'N/A')}")
    print(f"Mode: {config.get('mode', 'N/A')}")
    print(f"Task: {config.get('task', 'N/A')}")
    print(f"Batch size: {config.get('batch_size', 'N/A')}")
    print(f"Epochs: {config.get('num_epochs', 'N/A')}")
    print(f"Device: {config.get('device', 'N/A')}")
    
    try:
        # Import train_supervised module
        from train_supervised import main as train_supervised_main
        
        # Create argparse Namespace from config dictionary
        args = argparse.Namespace(**config)
        
        # Run the pipeline
        train_supervised_main(args)
        print(f"Supervised training completed successfully.")
        return 0
        
    except Exception as e:
        print(f"Error running supervised pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def run_meta_learning(config):
    """Run meta-learning pipeline directly without subprocess."""
    print(f"\n===== Running meta-learning pipeline directly =====")
    print(f"Data directory: {config.get('csi_data_dir', 'N/A')}")
    print(f"Output directory: {config.get('output_dir', 'N/A')}")
    print(f"Mode: {config.get('mode', 'N/A')}")
    print(f"Batch size: {config.get('batch_size', 'N/A')}")
    print(f"Meta batch size: {config.get('meta_batch_size', 'N/A')}")
    print(f"Epochs: {config.get('num_epochs', 'N/A')}")
    print(f"Device: {config.get('device', 'N/A')}")
    
    try:
        # Import meta_learning module
        from meta_learning import main as meta_learning_main
        
        # Create argparse Namespace from config dictionary
        args = argparse.Namespace(**config)
        
        # Run the pipeline
        meta_learning_main(args)
        print(f"Meta-learning completed successfully.")
        return 0
        
    except Exception as e:
        print(f"Error running meta-learning pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description='WiFi Sensing Pipeline Runner')
    
    # Required arguments
    parser.add_argument('--pipeline', type=str, choices=['supervised', 'meta'], 
                       default=PIPELINE, help='Pipeline to run')
    
    # Data directories
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                       help='Directory containing the input data')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='Directory to save output results')
    
    # Mode configuration
    parser.add_argument('--mode', type=str, choices=['csi', 'acf'], default=MODE,
                       help='Data modality to use')
    
    # Supervised learning options
    parser.add_argument('--pretrained', action='store_true', default=PRETRAINED,
                       help='Use pretrained model for supervised learning')
    parser.add_argument('--pretrained_model', type=str, default=PRETRAINED_MODEL,
                       help='Path to pretrained model')
    parser.add_argument('--freeze_backbone', action='store_true', default=FREEZE_BACKBONE,
                       help='Freeze backbone network for supervised learning')
    parser.add_argument('--integrated_loader', action='store_true', default=INTEGRATED_LOADER,
                       help='Use integrated data loader for supervised learning')
    parser.add_argument('--task', type=str, default=TASK,
                       help='Task type for integrated loader (e.g., ThreeClass, HumanNonhuman)')
    
    # Meta-learning options
    parser.add_argument('--n_way', type=int, default=3,
                       help='Number of classes per task for meta-learning')
    parser.add_argument('--k_shot', type=int, default=5,
                       help='Number of examples per class for meta-learning')
    parser.add_argument('--q_query', type=int, default=5,
                       help='Number of query examples for meta-learning')
    parser.add_argument('--meta_batch_size', type=int, default=4,
                       help='Number of tasks per meta-batch for meta-learning')
    
    # Common parameters
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=EPOCH_NUMBER,
                       help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for training')
    parser.add_argument('--seed', type=int, default=SEED,
                       help='Random seed for reproducibility')
    parser.add_argument('--win_len', type=int, default=WIN_LEN,
                       help='Window length for CSI data')
    parser.add_argument('--feature_size', type=int, default=FEATURE_SIZE,
                       help='Feature size for CSI data')
    
    # Advanced configuration
    parser.add_argument('--config_file', type=str, default=CONFIG_FILE,
                       help='JSON configuration file to override defaults')
    parser.add_argument('--subprocess', action='store_true', default=False,
                       help='Run in subprocess mode')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration from file if provided
    config = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
            print(f"Loaded configuration from {args.config_file}")
    
    # Get configuration for the selected pipeline
    if args.pipeline == 'supervised':
        # Configure supervised learning
        config.update(get_supervised_config(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            pretrained=args.pretrained,
            pretrained_model=args.pretrained_model
        ))
        
        # Update with command line arguments
        config.update({
            'mode': args.mode,
            'freeze_backbone': args.freeze_backbone,
            'integrated_loader': args.integrated_loader,
            'task': args.task,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'seed': args.seed,
            'win_len': args.win_len,
            'feature_size': args.feature_size
        })
        
        # Run supervised learning
        return run_supervised_direct(config)
    
    elif args.pipeline == 'meta':
        # Configure meta-learning
        config.update(get_meta_config(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        ))
        
        # Update with command line arguments
        config.update({
            'mode': args.mode,
            'n_way': args.n_way,
            'k_shot': args.k_shot,
            'q_query': args.q_query,
            'meta_batch_size': args.meta_batch_size,
            'batch_size': args.batch_size,
            'num_epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'seed': args.seed
        })
        
        # Run meta-learning
        return run_meta_learning(config)
    
    else:
        print(f"Unknown pipeline: {args.pipeline}")
        return 1


if __name__ == "__main__":
    # Capture start time
    start_time = time.time()
    
    # Run main function
    exit_code = main()
    
    # Calculate and display execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Exit with appropriate code
    sys.exit(exit_code)
