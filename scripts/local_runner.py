#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - Local Environment

This script serves as the main entry point for WiFi sensing benchmark.
It incorporates functionality from train.py, run_model.py, and the original local_runner.py.

Configuration File Management:
1. The configs folder now only contains template configuration files
2. Generated configuration files are saved to the results folder using a unified directory structure: results/TASK/MODEL/EXPERIMENT_ID/
   - Supervised learning: results/TASK/MODEL/EXPERIMENT_ID/supervised_config.json
   - Multitask learning: results/TASK/MODEL/EXPERIMENT_ID/multitask_config.json
3. All runtime parameters should be loaded from the configuration file, command-line arguments are no longer used

Usage:
    python local_runner.py --config_file [config_path]
    
Additional parameters:
    --config_file: JSON configuration file to use for all settings
"""

import os
import sys
import subprocess
import torch
import time
import argparse
import json
from datetime import datetime
import importlib.util
import pandas as pd

# Fix encoding issues on Windows
import io
import locale

# Try to set UTF-8 mode for Windows
if hasattr(sys, 'setdefaultencoding'):
    sys.setdefaultencoding('utf-8')

# Set stdout encoding to UTF-8 if possible
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
elif hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(SCRIPT_DIR)
print(f"root_dir is {ROOT_DIR}")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "local_default_config.json")

# Ensure results directory exists
DEFAULT_RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

def validate_config(config, required_fields=None):
    """
    Validate if the configuration contains all necessary parameters
    
    Args:
        config: Configuration dictionary
        required_fields: List of required fields, if None use default required fields
        
    Returns:
        True if validation succeeds, False otherwise
    """
    if required_fields is None:
        # Define basic required fields
        required_fields = [
            "pipeline", "training_dir", "output_dir", 
            "task", "win_len", "feature_size", "batch_size", "epochs"
        ]
        
    missing_fields = []
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"Error: Configuration file is missing the following required parameters: {', '.join(missing_fields)}")
        return False
    
    # Validate if pipeline is valid - hardcoded valid options
    valid_pipelines = ["supervised", "multitask"]
    if config["pipeline"] not in valid_pipelines:
        print(f"Error: Invalid pipeline value: '{config['pipeline']}'")
        print(f"Available options: {valid_pipelines}")
        return False
    
    return True

# Load configuration from JSON file
def load_config(config_path=None):
    """Load configuration from JSON file"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate if the configuration file contains all necessary parameters
        if not validate_config(config):
            sys.exit(1)
            
        print(f"Loaded configuration from {config_path}")
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load config file: {e}")
        sys.exit(1)

# Load the configuration
CONFIG = load_config(DEFAULT_CONFIG_PATH)

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

def get_supervised_config(custom_config=None):
    """
    Get configuration for supervised learning pipeline.
    
    Args:
        custom_config: Custom configuration dictionary
        
    Returns:
        Configuration dictionary
    """
    # Custom configuration must be provided
    if custom_config is None:
        print("Error: Configuration parameters must be provided!")
        sys.exit(1)
    
    # Create configuration dictionary
    config = {
        # Data parameters
        'training_dir': custom_config['training_dir'],
        'test_dirs': custom_config.get('test_dirs', []),
        'output_dir': custom_config['output_dir'],
        'results_subdir': f"{custom_config['model']}_{custom_config['task'].lower()}",
        'train_ratio': 0.8,
        'val_ratio': 0.2,
        
        # Training parameters
        'batch_size': custom_config['batch_size'],
        'learning_rate': custom_config.get('learning_rate', 1e-4),
        'weight_decay': custom_config.get('weight_decay', 1e-5),
        'epochs': custom_config['epochs'],
        'warmup_epochs': custom_config.get('warmup_epochs', 5),
        'patience': custom_config.get('patience', 15),
        
        # Integrated loader options
        'integrated_loader': True,  # Always use integrated loader
        'task': custom_config['task'],
        
        # Other parameters
        'seed': custom_config.get('seed', 42),
        'device': DEVICE,
        'model': custom_config['model'],
        'win_len': custom_config['win_len'],
        'feature_size': custom_config['feature_size'],
        
        # Test split options
        'test_splits': custom_config.get('test_splits', 'all')
    }
    
    # If model_params exists, add it to config
    if 'model_params' in custom_config:
        config['model_params'] = custom_config['model_params']
    
    return config

def get_multitask_config(custom_config=None):
    """
    Get configuration for multitask learning pipeline
    
    Args:
        custom_config: Custom configuration dictionary
        
    Returns:
        Configuration dictionary
    """
    # Custom configuration must be provided
    if custom_config is None:
        print("Error: Configuration parameters must be provided!")
        sys.exit(1)
    
    # Ensure there is a task name
    if 'task' not in custom_config and 'tasks' in custom_config:
        # If there is no single task name but there is a tasks list, use the first task as the task name
        tasks = custom_config.get('tasks')
        if isinstance(tasks, str):
            # If it's a string, it might be a comma-separated list
            task_list = tasks.split(',')
            if task_list:
                custom_config['task'] = task_list[0]
        elif isinstance(tasks, list) and tasks:
            # If it's a list and not empty
            custom_config['task'] = tasks[0]
    
    # If there is still no task, set a default value
    if 'task' not in custom_config:
        custom_config['task'] = 'multitask'
    
    # Create configuration dictionary
    config = {
        # Data parameters
        'training_dir': custom_config['training_dir'],
        'output_dir': custom_config['output_dir'],
        'results_subdir': f"{custom_config['model']}_{custom_config['task'].lower()}",
        
        # Training parameters
        'batch_size': custom_config['batch_size'],
        'learning_rate': custom_config.get('learning_rate', 5e-4),
        'weight_decay': custom_config.get('weight_decay', 1e-5),
        'epochs': custom_config['epochs'],
        'win_len': custom_config['win_len'],
        'feature_size': custom_config['feature_size'],
        
        # Model parameters
        'model': custom_config['model'],
        'emb_dim': custom_config.get('emb_dim', 128),
        'dropout': custom_config.get('dropout', 0.1),
        
        # Task parameters - keep both task and tasks
        'task': custom_config.get('task'),
        'tasks': custom_config.get('tasks'),
    }
    
    # If transformer_config.json exists, try to load it
    transform_path = os.path.join(CONFIG_DIR, "transformer_config.json")
    if os.path.exists(transform_path):
        print(f"Using existing configuration file: {transform_path}")
        with open(transform_path, 'r') as f:
            transformer_config = json.load(f)
            for k, v in transformer_config.items():
                if k in config:
                    config[k] = v
    
    # Ensure tasks parameter exists and has the correct format
    if not config.get('tasks'):
        if config.get('task'):
            config['tasks'] = config['task']
        else:
            print("Error: Multitask configuration must specify either 'tasks' or 'task' parameter!")
            sys.exit(1)
    
    # If model_params exists, add it to config
    if 'model_params' in custom_config:
        config['model_params'] = custom_config['model_params']
    
    return config

def run_supervised_direct(config):
    """
    Run supervised learning pipeline directly.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Return code from the process
    """
    # Get necessary parameters
    task_name = config.get('task')
    model_name = config.get('model')
    training_dir = config.get('training_dir')
    base_output_dir = config.get('output_dir')
    
    # Build basic command
    cmd = f"{sys.executable} {os.path.join(SCRIPT_DIR, 'train_supervised.py')}"
    cmd += f" --data_dir=\"{training_dir}\""
    cmd += f" --task_name={task_name}"
    cmd += f" --model={model_name}"
    cmd += f" --batch_size={config.get('batch_size')}"
    cmd += f" --epochs={config.get('epochs')}"
    cmd += f" --win_len={config.get('win_len')}"
    cmd += f" --feature_size={config.get('feature_size')}"
    cmd += f" --save_dir=\"{base_output_dir}\""
    cmd += f" --output_dir=\"{base_output_dir}\""
    
    # 禁用分布式训练和CPU优化
    cmd += " --num_workers=0"
    cmd += " --use_root_data_path"  # 不带值的标志参数
    
    # 禁用pin_memory以解决MPS警告
    # MPS设备不支持pin_memory，所以我们需要显式禁用它
    cmd += " --no_pin_memory"
    
    # Add test split parameters (if they exist)
    if 'test_splits' in config:
        cmd += f" --test_splits=\"{config['test_splits']}\""
    
    # Add other model-specific parameters
    important_params = ['learning_rate', 'weight_decay', 'warmup_epochs', 'patience', 
                         'emb_dim', 'dropout', 'd_model']
    for param in important_params:
        if param in config:
            cmd += f" --{param}={config[param]}"
    
    # Add parameters from model_params
    if 'model_params' in config:
        for key, value in config['model_params'].items():
            cmd += f" --{key}={value}"
    
    # Run command and capture output
    print(f"Running supervised learning: {cmd}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # Track experiment_id
    experiment_id = None
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        # Parse experiment_id from output
        if "Experiment ID:" in line:
            experiment_id = line.split("Experiment ID:")[1].strip()
    
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Error running supervised learning: return code {return_code}")
    else:
        print("Supervised learning completed successfully.")
        
        # If experiment_id was successfully obtained, save configuration directly to experiment directory
        if experiment_id:
            exp_dir = os.path.join(base_output_dir, task_name, model_name, experiment_id)
            config_filename = os.path.join(exp_dir, "supervised_config.json")
            
            try:
                # Ensure directory exists
                os.makedirs(exp_dir, exist_ok=True)
                
                # Save configuration file
                with open(config_filename, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"Configuration saved to model directory: {config_filename}")
            except Exception as e:
                print(f"Error saving configuration file: {str(e)}")
    
    return return_code

def run_multitask_direct(config):
    """
    Run multitask learning pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Return code (0 for success, non-zero for failure)
    """
    print("Running multitask learning with the following configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Get task parameters
    tasks = config.get('tasks')
    if not tasks:
        print("Error: 'tasks' parameter is missing or empty. Please specify at least one task.")
        return 1
    
    # Ensure tasks has the correct format - should be a comma-separated string without spaces
    if isinstance(tasks, list):
        tasks = ','.join(tasks)
    
    # Get basic parameters
    task_name = config.get('task', 'multitask').lower()
    model_name = config.get('model')
    base_output_dir = config.get('output_dir')
    
    # Build command
    cmd = f"{sys.executable} {os.path.join(SCRIPT_DIR, 'train_multitask_adapter.py')}"
    cmd += f" --tasks=\"{tasks}\""
    cmd += f" --model={model_name}"
    cmd += f" --data_dir=\"{config.get('training_dir')}\""
    cmd += f" --epochs={config.get('epochs')}"
    cmd += f" --batch_size={config.get('batch_size')}"
    cmd += f" --win_len={config.get('win_len')}"
    cmd += f" --feature_size={config.get('feature_size')}"
    
    # 禁用分布式训练和CPU优化
    cmd += " --num_workers=0"
    cmd += " --use_root_data_path"  # 不带值的标志参数
    
    # 禁用pin_memory以解决MPS警告
    # MPS设备不支持pin_memory，所以我们需要显式禁用它
    cmd += " --no_pin_memory"
    
    # Handle optional parameters from model_params
    if 'model_params' in config:
        model_params = config['model_params']
        for key, value in model_params.items():
            cmd += f" --{key}={value}"
    else:
        # If model_params doesn't exist, handle individual parameters
        for param in ['lr', 'emb_dim', 'dropout', 'patience']:
            if param in config:
                cmd += f" --{param}={config[param]}"
    
    # Add test_splits (if they exist)
    if 'test_splits' in config:
        cmd += f" --test_splits=\"{config['test_splits']}\""
    
    # Run command and capture output
    print(f"Running command: {cmd}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # Track experiment_id
    experiment_id = None
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        # Parse experiment_id from output
        if "Experiment ID:" in line:
            experiment_id = line.split("Experiment ID:")[1].strip()
    
    return_code = process.wait()
    
    if return_code != 0:
        print(f"Error running multitask learning: return code {return_code}")
    else:
        print("Multitask learning completed successfully.")
        
        # If experiment_id was successfully obtained, save configuration directly to experiment directory
        if experiment_id:
            exp_dir = os.path.join(base_output_dir, task_name, model_name, experiment_id)
            config_filename = os.path.join(exp_dir, "multitask_config.json")
            
            try:
                # Ensure directory exists
                os.makedirs(exp_dir, exist_ok=True)
                
                # Save configuration file
                with open(config_filename, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"Configuration saved to model directory: {config_filename}")
            except Exception as e:
                print(f"Error saving configuration file: {str(e)}")
    
    return return_code

def main():
    """Main entry point function"""
    # Parse command line arguments - only accept config_file
    parser = argparse.ArgumentParser(description='Run WiFi Sensing Pipeline')
    
    # config_file is the only required parameter
    parser.add_argument('--config_file', type=str, default=DEFAULT_CONFIG_PATH,
                        help='JSON configuration file for all settings')
    
    args = parser.parse_args()
    
    # Load configuration from file
    config = load_config(args.config_file)
    
    # Extract pipeline type from configuration
    pipeline = config.get('pipeline')
    
    # Ensure pipeline value is valid
    valid_pipelines = ["supervised", "multitask"]
    if pipeline not in valid_pipelines:
        print(f"Error: Invalid pipeline value: '{pipeline}'")
        print(f"Available options: {valid_pipelines}")
        return 1
    
    # Set data directory environment variable
    if 'training_dir' in config:
        os.environ['WIFI_DATA_DIR'] = config['training_dir']
    
    # Get all available models
    available_models = config.get('available_models', [])
    
    # If no available models defined or empty list, use a default
    if not available_models:
        print("Warning: No available models specified in configuration. Using default model 'mlp'.")
        available_models = ['mlp']
    
    # Record results for all models
    results = {}
    
    # Run each model in a loop
    for model in available_models:
        print(f"\n{'='*60}")
        print(f"Starting training for model: {model}")
        print(f"{'='*60}\n")
        
        # Create a new config copy for each model
        model_config = config.copy()
        model_config['model'] = model
        
        # Get specific pipeline configuration
        if pipeline == 'multitask':
            pipeline_config = get_multitask_config(model_config)
        else:  # Default to supervised learning
            pipeline_config = get_supervised_config(model_config)
        
        # Run appropriate pipeline
        start_time = time.time()
        if pipeline == 'multitask':
            return_code = run_multitask_direct(pipeline_config)
        else:  # Default to supervised learning
            return_code = run_supervised_direct(pipeline_config)
        
        # Record results
        end_time = time.time()
        results[model] = {
            'status': 'SUCCESS' if return_code == 0 else 'FAILED',
            'return_code': return_code,
            'run_time': end_time - start_time
        }
        
        print(f"\nModel {model} training {'successful' if return_code == 0 else 'failed'}")
        print(f"Running time: {(end_time - start_time)/60:.2f} minutes")
    
    # Print summary of all model runs
    print(f"\n{'='*60}")
    print(f"All model training completed")
    print(f"{'='*60}")
    print(f"Results summary:")
    
    successful = 0
    failed = 0
    for model, result in results.items():
        status = result['status']
        run_time = result['run_time']
        print(f"  - {model}: {status}, Running time: {run_time/60:.2f} minutes")
        
        if status == 'SUCCESS':
            successful += 1
        else:
            failed += 1
    
    print(f"\nSuccessful: {successful}/{len(results)}, Failed: {failed}/{len(results)}")
    
    # Return non-zero status code if any model failed
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
