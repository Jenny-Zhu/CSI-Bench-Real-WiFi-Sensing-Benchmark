#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - Local Environment

This script serves as the main entry point for WiFi sensing benchmark.
It incorporates functionality from train.py, run_model.py, and the original local_runner.py.

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

# Default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(SCRIPT_DIR)
print(f"root_dir is {ROOT_DIR}")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "local_default_config.json")

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
            "pipeline", "training_dir", "output_dir", "mode", "model", 
            "task", "win_len", "feature_size", "batch_size", "epochs"
        ]
        
    missing_fields = []
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"Error: Configuration file is missing the following required parameters: {', '.join(missing_fields)}")
        return False
    
    # Validate if pipeline is valid
    if config["pipeline"] not in config.get("available_pipelines", ["supervised", "multitask"]):
        print(f"Error: Invalid pipeline value: '{config['pipeline']}'")
        print(f"Available options: {config.get('available_pipelines', ['supervised', 'multitask'])}")
        return False
    
    # Validate if model is valid
    if config["model"] not in config.get("available_models", []):
        print(f"Error: Invalid model value: '{config['model']}'")
        print(f"Available options: {config.get('available_models', [])}")
        return False
    
    # Validate if task is valid
    if config["task"] not in config.get("available_tasks", []):
        print(f"Error: Invalid task value: '{config['task']}'")
        print(f"Available options: {config.get('available_tasks', [])}")
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
            
        # Process few-shot config to ensure backward compatibility
        if 'fewshot' in config:
            fewshot_config = config['fewshot']
            # Set legacy parameters for compatibility
            config['enable_few_shot'] = fewshot_config.get('enabled', False) or config.get('evaluate_fewshot', False)
            config['k_shot'] = fewshot_config.get('k_shots', 5)
            config['inner_lr'] = fewshot_config.get('adaptation_lr', 0.01)
            config['num_inner_steps'] = fewshot_config.get('adaptation_steps', 10)
            config['fewshot_support_split'] = fewshot_config.get('support_split', 'val_id')
            config['fewshot_query_split'] = fewshot_config.get('query_split', 'test_cross_env')
            config['fewshot_finetune_all'] = fewshot_config.get('finetune_all', False)
            config['fewshot_eval_shots'] = fewshot_config.get('eval_shots', False)
            
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
        
    # Get task class mapping
    task_class_mapping = custom_config.get("task_class_mapping", {})
    
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
        
        # Model parameters
        'mode': custom_config['mode'],
        'num_classes': task_class_mapping.get(custom_config['task'], 2),  # Default to 2 if task not found
        'freeze_backbone': custom_config.get('freeze_backbone', False),
        
        # Integrated loader options
        'integrated_loader': custom_config.get('integrated_loader', True),
        'task': custom_config['task'],
        
        # Other parameters
        'seed': custom_config.get('seed', 42),
        'device': DEVICE,
        'model': custom_config['model'],
        'win_len': custom_config['win_len'],
        'feature_size': custom_config['feature_size'],
        
        # Test split options
        'test_splits': custom_config.get('test_splits', 'all'),

        # Few-shot learning parameters
        'evaluate_fewshot': custom_config.get('evaluate_fewshot', False),
        'fewshot_support_split': custom_config.get('fewshot_support_split', 'val_id'),
        'fewshot_query_split': custom_config.get('fewshot_query_split', 'test_cross_env'),
        'fewshot_adaptation_lr': custom_config.get('fewshot_adaptation_lr', 0.01),
        'fewshot_adaptation_steps': custom_config.get('fewshot_adaptation_steps', 10),
        'fewshot_finetune_all': custom_config.get('fewshot_finetune_all', False),
        'fewshot_eval_shots': custom_config.get('fewshot_eval_shots', False),
        
        # Legacy few-shot parameters (for backwards compatibility)
        'enable_few_shot': custom_config.get('enable_few_shot', False),
        'k_shot': custom_config.get('k_shot', 5),
        'inner_lr': custom_config.get('inner_lr', 0.01),
        'num_inner_steps': custom_config.get('num_inner_steps', 10)
    }
    
    # If model_params exists, add it to config
    if 'model_params' in custom_config:
        config['model_params'] = custom_config['model_params']
    
    return config

def get_multitask_config(custom_config=None):
    """
    Get configuration for multitask learning pipeline.
    
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
        'output_dir': custom_config['output_dir'],
        'results_subdir': f"{custom_config['model']}_{custom_config['task'].lower() if 'task' in custom_config else 'multitask'}",
        
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
        
        # Task parameters - default to a single task
        'task': custom_config.get('task'),
        'tasks': custom_config.get('tasks'),
    }
    
    # If transformer_config.json exists, try to load it
    transform_path = os.path.join(CONFIG_DIR, "transformer_config.json")
    if os.path.exists(transform_path):
        print(f"Using existing config file: {transform_path}")
        with open(transform_path, 'r') as f:
            transformer_config = json.load(f)
            for k, v in transformer_config.items():
                if k in config:
                    config[k] = v
    
    # Ensure tasks parameter exists and is properly formatted
    if not config.get('tasks'):
        if config.get('task'):
            config['tasks'] = config['task']
        else:
            print("Error: Multitask configuration must specify 'tasks' or 'task' parameter!")
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
    # Get task and model names from config
    task_name = config.get('task')
    model_name = config.get('model')
    
    # Get base directories
    base_output_dir = config.get('output_dir')
    training_dir = config.get('training_dir')
    
    # Update save_dir to include task/model structure
    model_output_dir = os.path.join(base_output_dir, task_name, model_name)
    
    # Ensure model directory exists
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Build command as a string for better control of quoting
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
    
    # Add test_splits if present in config
    if 'test_splits' in config:
        cmd += f" --test_splits=\"{config['test_splits']}\""
    
    # Handle few-shot learning parameters
    if config.get('enable_few_shot', False) or config.get('evaluate_fewshot', False) or config.get('fewshot_eval_shots', False):
        cmd += " --enable_few_shot"
        cmd += f" --k_shot={config.get('k_shot', 5)}"
        cmd += f" --inner_lr={config.get('inner_lr', 0.01)}"
        cmd += f" --num_inner_steps={config.get('num_inner_steps', 10)}"
        
        # Add support and query splits if present
        if 'fewshot_support_split' in config:
            cmd += f" --fewshot_support_split={config['fewshot_support_split']}"
        if 'fewshot_query_split' in config:
            cmd += f" --fewshot_query_split={config['fewshot_query_split']}"
        if config.get('fewshot_finetune_all', False):
            cmd += " --fewshot_finetune_all"
    
    # Add other model-specific parameters
    for param in ['learning_rate', 'weight_decay', 'warmup_epochs', 'patience', 
                  'emb_dim', 'dropout', 'd_model', 'freeze_backbone', 'mode']:
        if param in config:
            param_name = param.replace('_', '-')
            cmd += f" --{param_name}={config[param]}"
    
    # Add model_params parameters
    if 'model_params' in config:
        for key, value in config['model_params'].items():
            cmd += f" --{key}={value}"
    
    # Run the command
    print(f"Running supervised learning with: {cmd}")
    return_code = subprocess.call(cmd, shell=True)
    
    if return_code != 0:
        print(f"Error running supervised learning: return code {return_code}")
    else:
        print("Supervised learning completed successfully.")
    
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
    
    # Get the tasks parameter correctly
    tasks = config.get('tasks')
    if not tasks:
        print("Error: 'tasks' parameter is missing or empty. Please specify at least one task.")
        return 1
    
    # Ensure tasks is properly formatted - should be a comma-separated string without spaces
    if isinstance(tasks, list):
        tasks = ','.join(tasks)
    
    # Build command directly as a string for better control of quotes and escaping
    cmd = f"{sys.executable} {os.path.join(SCRIPT_DIR, 'train_multitask_adapter.py')}"
    cmd += f" --tasks=\"{tasks}\""
    cmd += f" --model={config.get('model')}"
    cmd += f" --data_dir=\"{config.get('training_dir')}\""
    cmd += f" --epochs={config.get('epochs')}"
    cmd += f" --batch_size={config.get('batch_size')}"
    cmd += f" --win_len={config.get('win_len')}"
    cmd += f" --feature_size={config.get('feature_size')}"
    
    # Add few-shot flag if specified
    if config.get('enable_few_shot', False) or config.get('evaluate_fewshot', False):
        cmd += " --enable_few_shot"
        cmd += f" --k_shot={config.get('k_shot', 5)}"
        cmd += f" --inner_lr={config.get('inner_lr', 0.01)}"
        cmd += f" --num_inner_steps={config.get('num_inner_steps', 10)}"
    
    # Handle optional parameters from model_params if present
    if 'model_params' in config:
        model_params = config['model_params']
        for key, value in model_params.items():
            cmd += f" --{key}={value}"
    else:
        # Handle individual params if model_params not present
        for param in ['lr', 'emb_dim', 'dropout', 'patience']:
            if param in config:
                cmd += f" --{param}={config[param]}"
    
    # Add test_splits if present
    if 'test_splits' in config:
        cmd += f" --test_splits=\"{config['test_splits']}\""
    
    # Run the command
    print(f"Running command: {cmd}")
    return subprocess.call(cmd, shell=True)

def save_config(config, pipeline='supervised'):
    """
    Save configuration to a file
    
    Args:
        config: Configuration dictionary
        pipeline: Pipeline type ('supervised' or 'multitask')
        
    Returns:
        Path to the saved config file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(CONFIG_DIR, pipeline), exist_ok=True)
    
    # Generate config filename
    if pipeline == 'multitask':
        tasks = config.get('tasks')
        model = config.get('model')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_filename = os.path.join(CONFIG_DIR, pipeline,f"multitask_{model}_{timestamp}.json")
    else:
        model = config.get('model')
        task_name = config.get('task').lower()
        config_filename = os.path.join(CONFIG_DIR, pipeline,f"supervised_{model}_{task_name}_config.json")
    
    # Save config
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_filename}")
    return config_filename

def main():
    """Main entry point for the WiFi sensing pipeline runner"""
    # Parse command line arguments - only accept config_file
    parser = argparse.ArgumentParser(description='Run WiFi sensing pipeline')
    
    # Configuration file is the only required parameter
    parser.add_argument('--config_file', type=str, default=DEFAULT_CONFIG_PATH,
                        help='JSON configuration file to use for all settings')
    
    args = parser.parse_args()
    
    # Load configuration from file
    config = load_config(args.config_file)
    
    # Extract pipeline from the configuration
    pipeline = config.get('pipeline')
    
    # Ensure pipeline value is valid
    available_pipelines = config.get('available_pipelines', ['supervised', 'multitask'])
    if pipeline not in available_pipelines:
        print(f"Error: Invalid pipeline value: '{pipeline}'")
        print(f"Available options: {available_pipelines}")
        return 1
    
    # Set data directory environment variable
    if 'training_dir' in config:
        os.environ['WIFI_DATA_DIR'] = config['training_dir']
    
    # Get pipeline-specific configuration
    if pipeline == 'multitask':
        config = get_multitask_config(config)
    else:  # Default to supervised
        config = get_supervised_config(config)
    
    # Save configuration for future reference
    config_file = save_config(config, pipeline)
    
    # Run appropriate pipeline
    if pipeline == 'multitask':
        return run_multitask_direct(config)
    else:  # Default to supervised
        return run_supervised_direct(config)

if __name__ == "__main__":
    sys.exit(main())
