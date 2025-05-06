#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - Local Environment

This script serves as the main entry point for WiFi sensing benchmark.
It incorporates functionality from train.py, run_model.py, and the original local_runner.py.

Usage:
    python local_runner.py --model [model_name] --task [task_name]
    python local_runner.py --pipeline supervised --config_file [config_path]
    python local_runner.py --pipeline meta
    
Additional parameters:
    --model: Model architecture to train (mlp, lstm, resnet18, transformer, vit)
    --task: Task to train the model on (MotionSourceRecognition, HumanMotion, etc.)
    --training_dir: Directory containing the training data
    --test_dirs: List of directories containing test data
    --output_dir: Directory to save output results
    --mode: Data modality to use (csi or acf)
    --config_file: JSON configuration file to override defaults
    --epochs: Number of epochs to train
    --batch_size: Batch size for training
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
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "local_default_config.json")

# Load default configuration from JSON file
def load_default_config():
    """Load the default configuration from the JSON config file"""
    try:
        with open(DEFAULT_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        print(f"Loaded default configuration from {DEFAULT_CONFIG_PATH}")
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load default config file: {e}")
        print("Using hardcoded default values instead.")
        # Fallback to hardcoded defaults
        return {
            "pipeline": "supervised",
            "training_dir": "wifi_benchmark_dataset",
            "test_dirs": [],
            "output_dir": "../results",
            "mode": "csi",
            "freeze_backbone": False,
            "integrated_loader": True,
            "task": "MotionSourceRecognition",
            "win_len": 250,
            "feature_size": 98,
            "seed": 42,
            "batch_size": 8,
            "epochs": 10,
            "model": "transformer",
            "task_class_mapping": {
                "HumanNonhuman": 2, 
                "MotionSourceRecognition": 4, 
                "NTUHumanID": 15, 
                "NTUHAR": 6, 
                "HumanID": 4, 
                "Widar": 22,
                "HumanMotion": 3, 
                "ThreeClass": 3, 
                "DetectionandClassification": 5, 
                "Detection": 2
            },
            "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit"],
            "available_tasks": [
                "MotionSourceRecognition", 
                "HumanMotion", 
                "DetectionandClassification", 
                "HumanID", 
                "NTUHAR",
                "HumanNonhuman",
                "NTUHumanID",
                "Widar",
                "ThreeClass",
                "Detection"
            ],
            "available_pipelines": ["supervised", "meta", "multitask"]
        }

# Load the default configuration
DEFAULT_CONFIG = load_default_config()

# Extract configuration values
AVAILABLE_MODELS = DEFAULT_CONFIG.get("available_models", ["mlp", "lstm", "resnet18", "transformer", "vit"])
AVAILABLE_TASKS = DEFAULT_CONFIG.get("available_tasks", [])
TASK_CLASS_MAPPING = DEFAULT_CONFIG.get("task_class_mapping", {})
AVAILABLE_PIPELINES = DEFAULT_CONFIG.get("available_pipelines", ["supervised", "meta", "multitask"])

# Default values from config
PIPELINE = DEFAULT_CONFIG.get("pipeline", "supervised")
TRAINING_DIR = DEFAULT_CONFIG.get("training_dir", "wifi_benchmark_dataset")
TEST_DIRS = DEFAULT_CONFIG.get("test_dirs", [])
OUTPUT_DIR = DEFAULT_CONFIG.get("output_dir", "./results")
MODE = DEFAULT_CONFIG.get("mode", "csi")
FREEZE_BACKBONE = DEFAULT_CONFIG.get("freeze_backbone", False)
INTEGRATED_LOADER = DEFAULT_CONFIG.get("integrated_loader", True)
TASK = DEFAULT_CONFIG.get("task", "MotionSourceRecognition")
WIN_LEN = DEFAULT_CONFIG.get("win_len", 250)
FEATURE_SIZE = DEFAULT_CONFIG.get("feature_size", 98)
SEED = DEFAULT_CONFIG.get("seed", 42)
BATCH_SIZE = DEFAULT_CONFIG.get("batch_size", 8)
EPOCH_NUMBER = DEFAULT_CONFIG.get("epochs", 10)
MODEL_NAME = DEFAULT_CONFIG.get("model", "transformer")

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

def get_supervised_config(args=None, custom_config=None):
    """
    Get configuration for supervised learning pipeline.
    
    Args:
        args: Command line arguments
        custom_config: Custom configuration dictionary
        
    Returns:
        Configuration dictionary
    """
    # Start with default configuration
    config = {
        # Data parameters
        'training_dir': TRAINING_DIR,
        'test_dirs': TEST_DIRS,
        'output_dir': OUTPUT_DIR,
        'results_subdir': f"{MODEL_NAME}_{TASK.lower()}",
        'train_ratio': 0.8,
        'val_ratio': 0.2,
        
        # Training parameters
        'batch_size': BATCH_SIZE,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': EPOCH_NUMBER,
        'warmup_epochs': 5,
        'patience': 15,
        
        # Model parameters
        'mode': MODE,
        'num_classes': TASK_CLASS_MAPPING.get(TASK, 2),  # Default to 2 if task not found
        'freeze_backbone': FREEZE_BACKBONE,
        
        # Integrated loader options
        'integrated_loader': INTEGRATED_LOADER,
        'task': TASK,
        
        # Other parameters
        'seed': SEED,
        'device': DEVICE,
        'model': MODEL_NAME,
        'win_len': WIN_LEN,
        'feature_size': FEATURE_SIZE
    }
    
    # Override with custom config if provided
    if custom_config:
        for key, value in custom_config.items():
            config[key] = value
    
    # Override with command line arguments if provided
    if args:
        if hasattr(args, 'model') and args.model:
            config['model'] = args.model
        
        if hasattr(args, 'task') and args.task:
            config['task'] = args.task
            # Update num_classes based on task
            config['num_classes'] = TASK_CLASS_MAPPING.get(args.task, 2)
            # Update results_subdir based on model and task
            config['results_subdir'] = f"{config['model']}_{args.task.lower()}"
        
        if hasattr(args, 'training_dir') and args.training_dir:
            config['training_dir'] = args.training_dir
        
        if hasattr(args, 'test_dirs') and args.test_dirs:
            config['test_dirs'] = args.test_dirs
        
        if hasattr(args, 'output_dir') and args.output_dir:
            config['output_dir'] = args.output_dir
        
        if hasattr(args, 'batch_size') and args.batch_size:
            config['batch_size'] = args.batch_size
        
        if hasattr(args, 'epochs') and args.epochs:
            config['epochs'] = args.epochs
        
        if hasattr(args, 'mode') and args.mode:
            config['mode'] = args.mode
    
    return config

def get_meta_config(args=None, custom_config=None):
    """Get configuration for meta-learning pipeline."""
    # This is a placeholder for meta-learning configuration
    # Will be implemented by other team members
    config = {
        'training_dir': TRAINING_DIR,
        'output_dir': OUTPUT_DIR,
        'results_subdir': 'meta',
        # Other meta-learning parameters will be added here
    }
    
    # Override with custom config if provided
    if custom_config:
        for key, value in custom_config.items():
            config[key] = value
    
    # Override with command line arguments if provided
    if args:
        if hasattr(args, 'training_dir') and args.training_dir:
            config['training_dir'] = args.training_dir
        
        if hasattr(args, 'output_dir') and args.output_dir:
            config['output_dir'] = args.output_dir
    
    return config

def get_multitask_config(args=None, custom_config=None):
    """
    Get configuration for multitask learning pipeline.
    
    Args:
        args: Command line arguments
        custom_config: Custom configuration dictionary
        
    Returns:
        Configuration dictionary
    """
    # Use project root for save paths
    project_root = ROOT_DIR
    default_save_dir = os.path.join(project_root, 'results', 'multitask')
    
    # Start with default configuration
    config = {
        # Data parameters
        'data_dir': TRAINING_DIR,
        'output_dir': OUTPUT_DIR,
        'results_subdir': 'multitask',
        
        # Training parameters
        'batch_size': BATCH_SIZE,
        'lr': 1e-4,
        'epochs': EPOCH_NUMBER,
        'patience': 10,
        
        # Model parameters
        'model': MODEL_NAME,
        'tasks': TASK,
        'win_len': WIN_LEN,
        'feature_size': FEATURE_SIZE,
        
        # LoRA parameters
        'lora_r': 8,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        
        # Save directory
        'save_dir': default_save_dir
    }
    
    # Override with custom config if provided
    if custom_config:
        for key, value in custom_config.items():
            config[key] = value
    
    # Override with command line arguments if provided
    if args:
        if hasattr(args, 'tasks') and args.tasks:
            config['tasks'] = args.tasks
            
        if hasattr(args, 'model') and args.model:
            config['model'] = args.model
            
        if hasattr(args, 'training_dir') and args.training_dir:
            config['data_dir'] = args.training_dir
            
        if hasattr(args, 'output_dir') and args.output_dir:
            config['output_dir'] = args.output_dir
            # Make sure save_dir is properly set under output_dir
            config['save_dir'] = os.path.join(args.output_dir, 'multitask')
            
        if hasattr(args, 'batch_size') and args.batch_size:
            config['batch_size'] = args.batch_size
            
        if hasattr(args, 'epochs') and args.epochs:
            config['epochs'] = args.epochs
    
    # Ensure save_dir is always set
    if 'save_dir' not in config or config['save_dir'] is None:
        config['save_dir'] = default_save_dir
    
    return config

def create_or_load_config(args):
    """
    Create a new config or load from a file.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configuration dictionary
    """
    # Check if a config file is specified
    if args.config_file and os.path.exists(args.config_file):
        print(f"Loading configuration from {args.config_file}")
        with open(args.config_file, 'r') as f:
            config = json.load(f)
            return config
    
    # If model is specified, look for model-specific config
    if args.model:
        model_config = os.path.join(CONFIG_DIR, f"{args.model}_config.json")
        if os.path.exists(model_config):
            print(f"Using existing config file: {model_config}")
            with open(model_config, 'r') as f:
                config = json.load(f)
                
            # Override config with command line arguments
            if args.task:
                config['task'] = args.task
                # Update results_subdir based on model and task
                config['results_subdir'] = f"{args.model}_{args.task.lower()}"
                # Update num_classes based on task
                config['num_classes'] = TASK_CLASS_MAPPING.get(args.task, 2)
                
            if args.epochs:
                config['epochs'] = args.epochs
                
            if args.batch_size:
                config['batch_size'] = args.batch_size
                
            if args.output_dir:
                config['output_dir'] = args.output_dir
                
            return config
        
        # If task is specified, look for model+task specific config
        if args.task:
            model_task_config = os.path.join(CONFIG_DIR, f"{args.model}_{args.task.lower()}_config.json")
            if os.path.exists(model_task_config):
                print(f"Using existing config file: {model_task_config}")
                with open(model_task_config, 'r') as f:
                    config = json.load(f)
                    
                # Override config with command line arguments
                if args.epochs:
                    config['epochs'] = args.epochs
                    
                if args.batch_size:
                    config['batch_size'] = args.batch_size
                    
                if args.output_dir:
                    config['output_dir'] = args.output_dir
                    
                return config
    
    # Otherwise, create a new configuration based on the pipeline
    if args.pipeline == 'meta':
        return get_meta_config(args)
    elif args.pipeline == 'multitask':
        return get_multitask_config(args)
    else:  # Default to supervised
        return get_supervised_config(args)

def run_supervised_direct(config):
    """
    Run supervised learning pipeline by directly calling the train_supervised.py script
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Return code from the process
    """
    # Get task and model names from config
    task_name = config.get('task', TASK)
    model_name = config.get('model', MODEL_NAME)
    
    # Get base directories
    base_output_dir = config.get('output_dir', OUTPUT_DIR)
    
    # Update save_dir to include task/model structure
    # Note: We no longer create the experiment_id folder here, it will be created by train_supervised.py
    # Format: base_output_dir/task/model/
    model_output_dir = os.path.join(base_output_dir, task_name, model_name)
    
    # Ensure model directory exists
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Build command with all necessary arguments
    cmd = [
        "python", os.path.join(SCRIPT_DIR, "train_supervised.py"),
        f"--data_dir={config.get('training_dir', TRAINING_DIR)}",
        f"--task_name={task_name}",
        f"--model={model_name}",
        f"--batch_size={config.get('batch_size', BATCH_SIZE)}",
        f"--epochs={config.get('epochs', EPOCH_NUMBER)}",
        f"--learning_rate={config.get('learning_rate', 1e-4)}",
        f"--weight_decay={config.get('weight_decay', 1e-5)}",
        f"--warmup_epochs={config.get('warmup_epochs', 5)}",
        f"--patience={config.get('patience', 15)}",
        f"--win_len={config.get('win_len', WIN_LEN)}",
        f"--feature_size={config.get('feature_size', FEATURE_SIZE)}",
        f"--save_dir={base_output_dir}",
        f"--output_dir={base_output_dir}"
    ]
    
    # Add advanced parameters if they exist in the config
    if 'in_channels' in config:
        cmd.append(f"--in_channels={config['in_channels']}")
    if 'emb_dim' in config:
        cmd.append(f"--emb_dim={config['emb_dim']}")
    if 'dropout' in config:
        cmd.append(f"--dropout={config['dropout']}")
    if 'd_model' in config:
        cmd.append(f"--d_model={config['d_model']}")
    
    # Run the command
    print(f"Running supervised learning with: {' '.join(cmd)}")
    return_code, output = run_command(' '.join(cmd))
    
    # Check the result
    if return_code != 0:
        print(f"Error running supervised learning: {output}")
    else:
        print("Supervised learning completed successfully.")
        
        # Extract experiment_id from output if available
        experiment_id = None
        for line in output.split('\n'):
            if 'Experiment ID:' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    experiment_id = parts[1].strip()
                    break
        
        if experiment_id:
            print(f"Results saved to {os.path.join(model_output_dir, experiment_id)}")
        else:
            print(f"Results saved to {model_output_dir}")
    
    return return_code

def run_meta_learning(config):
    """
    Run meta-learning pipeline - placeholder for future implementation
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Return code from the process
    """
    print("Meta-learning pipeline will be implemented by another team member.")
    print("Current configuration:", json.dumps(config, indent=2))
    return 0

def run_multitask_direct(config):
    """
    Run multitask learning pipeline by directly calling the train_multitask_adapter.py script
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Return code from the process
    """
    # Get tasks list and model type
    tasks = config.get('tasks', TASK)
    model = config.get('model', MODEL_NAME)
    
    # Get project root directory for resolving paths
    project_root = ROOT_DIR
    
    # Ensure save_dir is an absolute path
    save_dir = config.get('save_dir')
    if save_dir is None:
        # Use default path if save_dir is not specified
        save_dir = os.path.join(project_root, 'results', 'multitask')
        print(f"save_dir not specified in config, using default: {save_dir}")
    elif not os.path.isabs(save_dir):
        save_dir = os.path.join(project_root, save_dir)
    
    # Update the config with the corrected save_dir
    config['save_dir'] = save_dir
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the command with environment variable
    if sys.platform == 'win32':
        cmd = f"set PYTHONPATH={project_root} && python {os.path.join(SCRIPT_DIR, 'train_multitask_adapter.py')}"
    else:
        cmd = f"PYTHONPATH={project_root} python {os.path.join(SCRIPT_DIR, 'train_multitask_adapter.py')}"
    
    # Add command line arguments
    cmd += f" --tasks={tasks}"
    cmd += f" --model={model}"
    
    # Ensure data_dir is resolved correctly
    data_dir = config.get('data_dir', TRAINING_DIR)
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)
    cmd += f" --data_dir={data_dir}"
    
    cmd += f" --batch_size={config.get('batch_size', BATCH_SIZE)}"
    cmd += f" --epochs={config.get('epochs', EPOCH_NUMBER)}"
    cmd += f" --lr={config.get('lr', 1e-4)}"
    cmd += f" --save_dir={save_dir}"
    cmd += f" --patience={config.get('patience', 10)}"
    cmd += f" --win_len={config.get('win_len', WIN_LEN)}"
    cmd += f" --feature_size={config.get('feature_size', FEATURE_SIZE)}"
    
    # Add LoRA parameters if they exist in the config
    if 'lora_r' in config:
        cmd += f" --lora_r={config['lora_r']}"
    if 'lora_alpha' in config:
        cmd += f" --lora_alpha={config['lora_alpha']}"
    if 'lora_dropout' in config:
        cmd += f" --lora_dropout={config['lora_dropout']}"
    
    # Add any transformer-specific parameters
    if 'emb_dim' in config:
        cmd += f" --emb_dim={config['emb_dim']}"
    if 'dropout' in config:
        cmd += f" --dropout={config['dropout']}"
    
    # Run the command
    print(f"Running multitask learning with: {cmd}")
    return_code, output = run_command(cmd)
    
    # Check the result
    if return_code != 0:
        print(f"Error running multitask learning: {output}")
    else:
        print("Multitask learning completed successfully.")
        print(f"Results saved to {save_dir}")
        
        # Extract experiment ID from output if available
        experiment_id = None
        for line in output.split('\n'):
            if 'Experiment ID:' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    experiment_id = parts[1].strip()
                    break
        
        if experiment_id:
            # Print path to the results for each task
            tasks_list = tasks.split(',')
            for task in tasks_list:
                task_result_path = os.path.join(save_dir, task, model, experiment_id)
                print(f"Results for task {task} saved to: {task_result_path}")
    
    return return_code

def save_config(config, pipeline='supervised'):
    """
    Save configuration to a file
    
    Args:
        config: Configuration dictionary
        pipeline: Pipeline type ('supervised', 'meta', or 'multitask')
        
    Returns:
        Path to the saved config file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(CONFIG_DIR, pipeline), exist_ok=True)
    
    # Generate config filename
    if pipeline == 'meta':
        config_filename = os.path.join(CONFIG_DIR, pipeline, f"meta_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    elif pipeline == 'multitask':
        tasks = config.get('tasks', TASK)
        model = config.get('model', MODEL_NAME)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        config_filename = os.path.join(CONFIG_DIR, pipeline,f"multitask_{model}_{timestamp}.json")
    else:
        model = config.get('model', MODEL_NAME)
        task_name = config.get('task', TASK).lower()
        config_filename = os.path.join(CONFIG_DIR, pipeline,f"supervised_{model}_{task_name}_config.json")
    
    # Save config
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {config_filename}")
    return config_filename

def main():
    """Main entry point for the WiFi sensing pipeline runner"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run WiFi sensing pipeline')
    
    # Pipeline selection
    parser.add_argument('--pipeline', type=str, default=PIPELINE, choices=AVAILABLE_PIPELINES,
                        help='Pipeline to run (supervised, meta, or multitask)')
    
    # Model and task selection
    parser.add_argument('--model', type=str, choices=AVAILABLE_MODELS,
                        help='Model architecture to train')
    parser.add_argument('--task', type=str,
                        help='Task to train on')
    parser.add_argument('--tasks', type=str,
                        help='Comma-separated list of tasks for multitask learning')
    
    # Data directories
    parser.add_argument('--training_dir', type=str,
                        help='Directory containing training data')
    parser.add_argument('--test_dirs', type=str, nargs='+',
                        help='Directories containing test data')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training')
    parser.add_argument('--mode', type=str, choices=['csi', 'acf'],
                        help='Data modality (csi or acf)')
    
    # Configuration
    parser.add_argument('--config_file', type=str,
                        help='JSON configuration file to override defaults')
    
    args = parser.parse_args()
    
    # Validate arguments based on pipeline
    if args.pipeline == 'multitask' and not args.tasks:
        print("Error: --tasks argument is required for multitask pipeline")
        print("Example: --tasks \"MotionSourceRecognition,HumanMotion\"")
        return 1
    
    if args.pipeline == 'supervised' and not args.task and not args.model:
        print("Warning: Running supervised pipeline without specifying --task or --model")
        print("Default task: {}, Default model: {}".format(TASK, MODEL_NAME))
    
    # Load or create configuration
    config = create_or_load_config(args)
    
    # Save configuration for future reference
    config_file = save_config(config, args.pipeline)
    
    # Run appropriate pipeline
    if args.pipeline == 'meta':
        return run_meta_learning(config)
    elif args.pipeline == 'multitask':
        return run_multitask_direct(config)
    else:  # Default to supervised
        return run_supervised_direct(config)

if __name__ == "__main__":
    sys.exit(main())
