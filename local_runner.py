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
    --training_dir: Directory containing the training data
    --test_dirs: List of directories containing test data
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
# For supervised learning with integrated loader, TRAINING_DIR should point directly 
# to the folder containing .mat files or to the parent folder containing these files
TRAINING_DIR = "/Users/leo/Downloads/demo"
# Test directories can be a list of paths to evaluate on multiple test sets
TEST_DIRS = ["/Users/leo/Downloads/demo/test"]  # Example: ["C:\\Users\\weiha\\Desktop\\test1", "C:\\Users\\weiha\\Desktop\\test2"]
OUTPUT_DIR = "/Users/leo/Downloads/output"

# Data Modality
MODE = 'csi'  # Options: 'csi', 'acf'

# Supervised Learning Options
FREEZE_BACKBONE = False  # Freeze backbone network for supervised learning
INTEGRATED_LOADER = True  # Use integrated data loader for supervised learning
TASK = 'FourClass'  # Task type for integrated loader (e.g., ThreeClass, HumanNonhuman)

# Model Parameters
WIN_LEN = 500  # Window length for CSI data
FEATURE_SIZE = 232  # Feature size for CSI data

# Common Training Parameters
SEED = 42
BATCH_SIZE = 8
EPOCH_NUMBER = 1  # Number of training epochs
MODEL_NAME = 'ViT'

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
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("CUDA not available. Using MPS (Apple Silicon GPU).")
else:
    device = torch.device("cpu")
    print("Neither CUDA nor MPS available. Using CPU.")

# Print PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Set device string for command line arguments
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

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
def get_supervised_config(training_dir=None, test_dirs=None, output_dir=None, mode='csi'):
    """Get configuration for supervised learning pipeline."""
    # Set default paths if not provided
    if training_dir is None:
        training_dir = TRAINING_DIR
    if test_dirs is None:
        test_dirs = TEST_DIRS
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Define number of classes based on task
    task_class_mapping = {
        'HumanNonhuman': 2, 
        'FourClass': 4, 
        'NTUHumanID': 15, 
        'NTUHAR': 6, 
        'HumanID': 4, 
        'Widar': 22,
        'HumanMotion': 3, 
        'ThreeClass': 3, 
        'DetectionandClassification': 5, 
        'Detection': 2
    }
    
    # Get number of classes based on task
    num_classes = task_class_mapping.get(TASK, 2)  # Default to 2 if task not found
    
    config = {
        # Data parameters
        'training_dir': training_dir,
        'test_dirs': test_dirs,
        'output_dir': output_dir,
        'results_subdir': 'supervised',
        'train_ratio': 0.8,
        'val_ratio': 0.2,
        
        # Training parameters
        'batch_size': BATCH_SIZE,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'num_epochs': EPOCH_NUMBER,
        'warmup_epochs': 5,
        'patience': 15,
        
        # Model parameters
        'mode': mode,
        'num_classes': num_classes,  # Now set based on task
        'freeze_backbone': FREEZE_BACKBONE,
        
        # Integrated loader options
        'integrated_loader': INTEGRATED_LOADER,
        'task': TASK,
        
        # Other parameters
        'seed': SEED,
        'device': DEVICE,
        'model_name': MODEL_NAME,
        'win_len': WIN_LEN,
        'feature_size': FEATURE_SIZE
    }
    
    return config


def get_meta_config(training_dir=None, output_dir=None):
    """Get configuration for meta-learning pipeline."""
    # Set default paths if not provided
    if training_dir is None:
        training_dir = TRAINING_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    return {
        # Data parameters
        'training_dir': training_dir,
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
        'model_name': MODEL_NAME,
        'win_len': WIN_LEN,
        'feature_size': FEATURE_SIZE
    }


def run_supervised_direct(config):
    """Run supervised learning pipeline directly using Python API"""
    print("Running supervised learning pipeline directly...")
    
    # Import here to avoid circular imports
    from train_supervised import main
    
    # Convert config dict to Namespace for compatibility
    args = argparse.Namespace(**config)
    
    # Call main function
    return main(args)


def run_meta_learning(config):
    """Run meta-learning pipeline directly using Python API"""
    print("Running meta-learning pipeline directly...")
    
    # Import here to avoid circular imports
    from meta_learning import main
    
    # Convert config dict to Namespace for compatibility
    args = argparse.Namespace(**config)
    
    # Call main function
    return main(args)


def main():
    """Main function to execute WiFi sensing pipelines"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run WiFi sensing pipeline')
    parser.add_argument('--pipeline', type=str, default=PIPELINE, 
                      choices=['supervised', 'meta'],
                      help='Pipeline to run')
    parser.add_argument('--training_dir', type=str, default=TRAINING_DIR,
                      help='Directory containing training data')
    parser.add_argument('--test_dirs', type=str, nargs='+', default=TEST_DIRS,
                      help='List of directories containing test data. Can specify multiple paths')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                      help='Directory to save output results')
    parser.add_argument('--mode', type=str, default=MODE,
                      choices=['csi', 'acf'],
                      help='Data modality to use')
    parser.add_argument('--config_file', type=str, default=CONFIG_FILE,
                      help='JSON configuration file to override defaults')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load optional config file if provided
    config = {}
    if args.config_file and os.path.exists(args.config_file):
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
    
    # Get pipeline config based on selected pipeline
    if args.pipeline == 'supervised':
        pipe_config = get_supervised_config(args.training_dir, args.test_dirs, args.output_dir, args.mode)
    elif args.pipeline == 'meta':
        pipe_config = get_meta_config(args.training_dir, args.output_dir)
        # Meta-learning currently only supports CSI mode
        if args.mode != 'csi':
            print("Warning: Meta-learning only supports CSI mode. Switching to CSI.")
            pipe_config['mode'] = 'csi'
    else:
        raise ValueError(f"Unknown pipeline: {args.pipeline}")
    
    # Override with values from config file
    for key, value in config.items():
        pipe_config[key] = value
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in pipe_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Save configuration for reproducibility
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(args.output_dir, f"config_{args.pipeline}_{timestamp}.json")
    with open(config_path, 'w') as f:
        json.dump(pipe_config, f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    # Run the selected pipeline
    start_time = time.time()
    try:
        if args.pipeline == 'supervised':
            run_supervised_direct(pipe_config)
        elif args.pipeline == 'meta':
            run_meta_learning(pipe_config)
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nPipeline execution completed in {elapsed_time:.2f} seconds.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
