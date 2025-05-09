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
print(f"root_dir is {ROOT_DIR}")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "local_default_config.json")

# Load default configuration from JSON file
def load_default_config():
    """Load the default configuration from the JSON config file"""
    try:
        with open(DEFAULT_CONFIG_PATH, 'r') as f:
            config = json.load(f)
            
            # Process nested few-shot configs if present
            if 'fewshot' in config:
                fewshot_config = config['fewshot']
                # Apply few-shot settings at the root level for compatibility
                config['k_shot'] = fewshot_config.get('k_shots', 5)
                config['inner_lr'] = fewshot_config.get('adaptation_lr', 0.01)
                config['num_inner_steps'] = fewshot_config.get('adaptation_steps', 10)
                config['enable_few_shot'] = True if fewshot_config.get('eval_shots', False) else False
            
            # Process supervised few-shot configs if present
            if 'supervised_fewshot' in config:
                supervised_fewshot = config['supervised_fewshot']
                # Apply supervised few-shot settings at the root level
                config['evaluate_fewshot'] = supervised_fewshot.get('evaluate_fewshot', False)
                config['fewshot_support_split'] = supervised_fewshot.get('fewshot_support_split', 'val_id')
                config['fewshot_query_split'] = supervised_fewshot.get('fewshot_query_split', 'test_cross_env')
                config['fewshot_adaptation_lr'] = supervised_fewshot.get('fewshot_adaptation_lr', 0.01)
                config['fewshot_adaptation_steps'] = supervised_fewshot.get('fewshot_adaptation_steps', 10)
                config['fewshot_finetune_all'] = supervised_fewshot.get('fewshot_finetune_all', False)
                config['fewshot_eval_shots'] = supervised_fewshot.get('fewshot_eval_shots', False)
                
                # Set legacy parameters for compatibility
                if supervised_fewshot.get('evaluate_fewshot', False):
                    config['enable_few_shot'] = True
            
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
            "available_pipelines": ["supervised", "meta", "multitask", "fewshot"],
            "test_splits": "all",
            # Few-shot learning parameters
            "evaluate_fewshot": False,
            "fewshot_support_split": "val_id",
            "fewshot_query_split": "test_cross_env",
            "fewshot_adaptation_lr": 0.01,
            "fewshot_adaptation_steps": 10,
            "fewshot_finetune_all": False,
            "fewshot_eval_shots": False,
            "fewshot_task": None, # For multitask only - which task to adapt
            "enable_few_shot": False,
            "k_shot": 5,
            "inner_lr": 0.01,
            "num_inner_steps": 10,
        }

# Load the default configuration
DEFAULT_CONFIG = load_default_config()

# Extract configuration values
AVAILABLE_MODELS = DEFAULT_CONFIG.get("available_models", ["mlp", "lstm", "resnet18", "transformer", "vit"])
AVAILABLE_TASKS = DEFAULT_CONFIG.get("available_tasks", [])
TASK_CLASS_MAPPING = DEFAULT_CONFIG.get("task_class_mapping", {})
AVAILABLE_PIPELINES = DEFAULT_CONFIG.get("available_pipelines", ["supervised", "meta", "multitask", "fewshot"])

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
        'feature_size': FEATURE_SIZE,
        
        # Test split options
        'test_splits': DEFAULT_CONFIG.get('test_splits', 'all'),

        # Few-shot learning parameters
        'evaluate_fewshot': False,
        'fewshot_support_split': 'val_id',
        'fewshot_query_split': 'test_cross_env',
        'fewshot_adaptation_lr': 0.01,
        'fewshot_adaptation_steps': 10,
        'fewshot_finetune_all': False,
        'fewshot_eval_shots': False,
        
        # Legacy few-shot parameters (for backwards compatibility)
        'enable_few_shot': False,
        'k_shot': 5,
        'inner_lr': 0.01,
        'num_inner_steps': 10
    }
    
    # Override with custom config if provided
    if custom_config:
        for key, value in custom_config.items():
            config[key] = value
    
    # Override with command line arguments if provided
    if args:
        if hasattr(args, 'model') and args.model:
            config['model'] = args.model
        
        # Handle task parameters (unified approach)
        if hasattr(args, 'tasks') and args.tasks:
            # For supervised learning, use the first task if multiple are specified
            tasks = args.tasks.split(',') if ',' in args.tasks else [args.tasks]
            task = tasks[0]  # Get first task for supervised learning
            config['task'] = task
            # Update num_classes based on task
            config['num_classes'] = TASK_CLASS_MAPPING.get(task, 2)
            # Update results_subdir based on model and task
            config['results_subdir'] = f"{config['model']}_{task.lower()}"
        elif hasattr(args, 'task') and args.task:
            config['task'] = args.task
            # Update num_classes based on task
            config['num_classes'] = TASK_CLASS_MAPPING.get(args.task, 2)
            # Update results_subdir based on model and task
            config['results_subdir'] = f"{config['model']}_{args.task.lower()}"
        
        # Handle directory parameters
        if hasattr(args, 'training_dir') and args.training_dir:
            config['training_dir'] = args.training_dir
        
        if hasattr(args, 'test_dirs') and args.test_dirs:
            config['test_dirs'] = args.test_dirs
        
        if hasattr(args, 'output_dir') and args.output_dir:
            config['output_dir'] = args.output_dir
        
        # Handle training parameters
        if hasattr(args, 'batch_size') and args.batch_size:
            config['batch_size'] = args.batch_size
        
        if hasattr(args, 'epochs') and args.epochs:
            config['epochs'] = args.epochs
        
        if hasattr(args, 'mode') and args.mode:
            config['mode'] = args.mode
            
        if hasattr(args, 'test_splits') and args.test_splits:
            config['test_splits'] = args.test_splits
        
        # Handle model parameters
        if hasattr(args, 'feature_size') and args.feature_size:
            config['feature_size'] = args.feature_size
            
        if hasattr(args, 'win_len') and args.win_len:
            config['win_len'] = args.win_len
            
        if hasattr(args, 'emb_dim') and args.emb_dim:
            config['emb_dim'] = args.emb_dim
            
        if hasattr(args, 'dropout') and args.dropout:
            config['dropout'] = args.dropout
            
        # Add new few-shot learning parameters
        if hasattr(args, 'evaluate_fewshot'):
            config['evaluate_fewshot'] = args.evaluate_fewshot
            
        if hasattr(args, 'fewshot_support_split'):
            config['fewshot_support_split'] = args.fewshot_support_split
            
        if hasattr(args, 'fewshot_query_split'):
            config['fewshot_query_split'] = args.fewshot_query_split
            
        if hasattr(args, 'fewshot_adaptation_lr'):
            config['fewshot_adaptation_lr'] = args.fewshot_adaptation_lr
            
        if hasattr(args, 'fewshot_adaptation_steps'):
            config['fewshot_adaptation_steps'] = args.fewshot_adaptation_steps
            
        if hasattr(args, 'fewshot_finetune_all'):
            config['fewshot_finetune_all'] = args.fewshot_finetune_all
            
        if hasattr(args, 'fewshot_eval_shots'):
            config['fewshot_eval_shots'] = args.fewshot_eval_shots
            
        # Legacy few-shot parameters (for backwards compatibility)
        if hasattr(args, 'enable_few_shot'):
            config['enable_few_shot'] = args.enable_few_shot
            
        if hasattr(args, 'k_shot') and args.k_shot:
            config['k_shot'] = args.k_shot
            
        if hasattr(args, 'inner_lr') and args.inner_lr:
            config['inner_lr'] = args.inner_lr
            
        if hasattr(args, 'num_inner_steps') and args.num_inner_steps:
            config['num_inner_steps'] = args.num_inner_steps
    
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
    # Start with default configuration
    config = {
        # Data parameters
        'training_dir': TRAINING_DIR,
        'output_dir': OUTPUT_DIR,
        'results_subdir': f"{MODEL_NAME}_{TASK.lower()}",
        
        # Training parameters
        'batch_size': BATCH_SIZE,
        'learning_rate': 5e-4,
        'weight_decay': 1e-5,
        'epochs': EPOCH_NUMBER,
        'win_len': WIN_LEN,
        'feature_size': FEATURE_SIZE,
        
        # Model parameters
        'model': MODEL_NAME,
        'emb_dim': 128,
        'dropout': 0.1,
        
        # Task parameters - default to a single task
        'task': TASK,
        'tasks': None,  # This will be overridden by args.tasks if provided
    }
    
    # Override with transformer-specific configuration file if it exists
    transform_path = os.path.join(CONFIG_DIR, "transformer_config.json")
    if os.path.exists(transform_path):
        print(f"Using existing config file: {transform_path}")
        with open(transform_path, 'r') as f:
            transformer_config = json.load(f)
            for k, v in transformer_config.items():
                if k in config:
                    config[k] = v
    
    # Override with custom configuration if provided
    if custom_config:
        for k, v in custom_config.items():
            config[k] = v
    
    # Override with command line arguments if provided
    if args:
        # Handle tasks parameter
        if hasattr(args, 'tasks') and args.tasks:
            config['tasks'] = args.tasks
            # Also set task for backwards compatibility
            # Extract first task for legacy code that expects a single task
            first_task = args.tasks.split(',')[0] if ',' in args.tasks else args.tasks
            config['task'] = first_task
        elif hasattr(args, 'task') and args.task:
            config['task'] = args.task
            # Also set tasks for newer code that expects a list of tasks
            config['tasks'] = args.task
            print(f"Warning: Using deprecated 'task' parameter. Please use 'tasks' instead.")
            
        # Handle model parameter
        if hasattr(args, 'model') and args.model:
            config['model'] = args.model
            
        # Handle training directory
        if hasattr(args, 'training_dir') and args.training_dir:
            config['training_dir'] = args.training_dir
            
        # Handle output directory
        if hasattr(args, 'output_dir') and args.output_dir:
            config['output_dir'] = args.output_dir
            
        # Handle batch size
        if hasattr(args, 'batch_size') and args.batch_size:
            config['batch_size'] = args.batch_size
            
        # Handle epochs
        if hasattr(args, 'epochs') and args.epochs:
            config['epochs'] = args.epochs
        
        # Handle model parameters
        if hasattr(args, 'feature_size') and args.feature_size:
            config['feature_size'] = args.feature_size
            
        if hasattr(args, 'win_len') and args.win_len:
            config['win_len'] = args.win_len
            
        if hasattr(args, 'emb_dim') and args.emb_dim:
            config['emb_dim'] = args.emb_dim
            
        if hasattr(args, 'dropout') and args.dropout:
            config['dropout'] = args.dropout
            
        # Handle few-shot parameters
        if hasattr(args, 'evaluate_fewshot') and args.evaluate_fewshot:
            config['enable_few_shot'] = True
        if hasattr(args, 'k_shot') and args.k_shot:
            config['k_shot'] = args.k_shot
        if hasattr(args, 'inner_lr') and args.inner_lr:
            config['inner_lr'] = args.inner_lr
        if hasattr(args, 'num_inner_steps') and args.num_inner_steps:
            config['num_inner_steps'] = args.num_inner_steps
    
    # Ensure tasks parameter exists and is properly formatted
    if not config.get('tasks'):
        if config.get('task'):
            config['tasks'] = config['task']
        else:
            config['tasks'] = TASK
    
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
            print(f"DEBUG - Raw config loaded: few-shot settings:")
            print(f"  enable_few_shot (before processing): {config.get('enable_few_shot', 'Not set')}")
            print(f"  fewshot section: {config.get('fewshot', 'Not present')}")
            print(f"  supervised_fewshot section: {config.get('supervised_fewshot', 'Not present')}")
            
            # Process nested few-shot configs if present
            if 'fewshot' in config:
                fewshot_config = config['fewshot']
                # Apply few-shot settings at the root level for compatibility
                config['k_shot'] = fewshot_config.get('k_shots', 5)
                config['inner_lr'] = fewshot_config.get('adaptation_lr', 0.01)
                config['num_inner_steps'] = fewshot_config.get('adaptation_steps', 10)
                # Only set enable_few_shot to True if eval_shots is True
                # If enable_few_shot is explicitly set to False in the config, respect that
                if 'enable_few_shot' not in config and fewshot_config.get('eval_shots', False):
                    config['enable_few_shot'] = True
                    print(f"DEBUG - Setting enable_few_shot=True from fewshot.eval_shots")
                else:
                    print(f"DEBUG - Not setting enable_few_shot from fewshot.eval_shots because:")
                    print(f"  'enable_few_shot' in config: {'enable_few_shot' in config}")
                    print(f"  fewshot_config.get('eval_shots'): {fewshot_config.get('eval_shots')}")
            
            # Process supervised few-shot configs if present
            if 'supervised_fewshot' in config:
                supervised_fewshot = config['supervised_fewshot']
                # Apply supervised few-shot settings at the root level
                config['evaluate_fewshot'] = supervised_fewshot.get('evaluate_fewshot', False)
                config['fewshot_support_split'] = supervised_fewshot.get('fewshot_support_split', 'val_id')
                config['fewshot_query_split'] = supervised_fewshot.get('fewshot_query_split', 'test_cross_env')
                config['fewshot_adaptation_lr'] = supervised_fewshot.get('fewshot_adaptation_lr', 0.01)
                config['fewshot_adaptation_steps'] = supervised_fewshot.get('fewshot_adaptation_steps', 10)
                config['fewshot_finetune_all'] = supervised_fewshot.get('fewshot_finetune_all', False)
                config['fewshot_eval_shots'] = supervised_fewshot.get('fewshot_eval_shots', False)
                
                # Set legacy parameters for compatibility
                if supervised_fewshot.get('evaluate_fewshot', False) or supervised_fewshot.get('fewshot_eval_shots', False):
                    config['enable_few_shot'] = True
                    config['k_shot'] = supervised_fewshot.get('k_shots', 5)
                    config['inner_lr'] = supervised_fewshot.get('fewshot_adaptation_lr', 0.01)
                    config['num_inner_steps'] = supervised_fewshot.get('fewshot_adaptation_steps', 10)
                    print(f"DEBUG - Setting enable_few_shot=True from supervised_fewshot")
                else:
                    print(f"DEBUG - Not setting enable_few_shot from supervised_fewshot because:")
                    print(f"  supervised_fewshot.get('evaluate_fewshot'): {supervised_fewshot.get('evaluate_fewshot')}")
                    print(f"  supervised_fewshot.get('fewshot_eval_shots'): {supervised_fewshot.get('fewshot_eval_shots')}")
            
            print(f"DEBUG - Final few-shot settings:")
            print(f"  enable_few_shot: {config.get('enable_few_shot', 'Not set')}")
            print(f"  evaluate_fewshot: {config.get('evaluate_fewshot', 'Not set')}")
            print(f"  fewshot_eval_shots: {config.get('fewshot_eval_shots', 'Not set')}")
            print(f"  k_shot: {config.get('k_shot', 'Not set')}")
            
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
    training_dir = config.get('training_dir', TRAINING_DIR)
    
    # Update save_dir to include task/model structure
    # Note: We no longer create the experiment_id folder here, it will be created by train_supervised.py
    # Format: base_output_dir/task/model/
    model_output_dir = os.path.join(base_output_dir, task_name, model_name)
    
    # Ensure model directory exists
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Build command as a string for better control of quoting
    cmd = f"{sys.executable} {os.path.join(SCRIPT_DIR, 'train_supervised.py')}"
    cmd += f" --data_dir=\"{training_dir}\""
    cmd += f" --task_name={task_name}"
    cmd += f" --model={model_name}"
    cmd += f" --batch_size={config.get('batch_size', BATCH_SIZE)}"
    cmd += f" --epochs={config.get('epochs', EPOCH_NUMBER)}"
    cmd += f" --learning_rate={config.get('learning_rate', 1e-4)}"
    cmd += f" --weight_decay={config.get('weight_decay', 1e-5)}"
    cmd += f" --warmup_epochs={config.get('warmup_epochs', 5)}"
    cmd += f" --patience={config.get('patience', 15)}"
    cmd += f" --win_len={config.get('win_len', WIN_LEN)}"
    cmd += f" --feature_size={config.get('feature_size', FEATURE_SIZE)}"
    cmd += f" --save_dir=\"{base_output_dir}\""
    cmd += f" --output_dir=\"{base_output_dir}\""
    
    # Add test_splits if present in config
    if 'test_splits' in config:
        cmd += f" --test_splits=\"{config['test_splits']}\""
    else:
        # Make sure we pass 'all' as default for test_splits
        cmd += " --test_splits=all"
    
    # FIXED: Handle few-shot learning parameters 
    # Now we check enable_few_shot, evaluate_fewshot, and fewshot_eval_shots
    if config.get('enable_few_shot', False) or config.get('evaluate_fewshot', False) or config.get('fewshot_eval_shots', False):
        print(f"DEBUG - Adding few-shot flag to command line")
        cmd += " --enable_few_shot"
        cmd += f" --k_shot={config.get('k_shot', 5)}"
        cmd += f" --inner_lr={config.get('inner_lr', config.get('fewshot_adaptation_lr', 0.01))}"
        cmd += f" --num_inner_steps={config.get('num_inner_steps', config.get('fewshot_adaptation_steps', 10))}"
    
    # Add advanced parameters if they exist in the config
    if 'in_channels' in config:
        cmd += f" --in_channels={config['in_channels']}"
    if 'emb_dim' in config:
        cmd += f" --emb_dim={config['emb_dim']}"
    if 'dropout' in config:
        cmd += f" --dropout={config['dropout']}"
    if 'd_model' in config:
        cmd += f" --d_model={config['d_model']}"
    
    # Run the command
    print(f"Running supervised learning with: {cmd}")
    return_code = subprocess.call(cmd, shell=True)
    
    if return_code != 0:
        print(f"Error running supervised learning: return code {return_code}")
    else:
        print("Supervised learning completed successfully.")
    
    return return_code

def run_meta_learning(config):
    """Run meta-learning pipeline with the given configuration"""
    
    # Build command
    cmd_path = os.path.join(ROOT_DIR, "train_multi_model.py")
    
    # Extract config values
    tasks = config.get("tasks", [config.get("task", TASK)])
    model = config.get("model", MODEL_NAME)
    
    # Set up command
    cmd = [
        "python", cmd_path,
        f"--tasks={','.join(tasks)}",
        f"--model={model}",
        f"--mode={config.get('mode', MODE)}",
        f"--seed={config.get('seed', SEED)}",
        f"--batch_size={config.get('batch_size', BATCH_SIZE)}",
        f"--epochs={config.get('epochs', EPOCH_NUMBER)}",
        f"--patience={config.get('patience', 15)}",
        f"--pipeline=meta"
    ]
    
    # Add optional parameters
    if "output_dir" in config:
        cmd.append(f"--output_dir={config.get('output_dir')}")
    if "enable_few_shot" in config and config["enable_few_shot"]:
        cmd.append(f"--enable_few_shot")
        cmd.append(f"--k_shot={config.get('k_shot', 5)}")
        cmd.append(f"--inner_lr={config.get('inner_lr', 0.01)}")
        cmd.append(f"--num_inner_steps={config.get('num_inner_steps', 10)}")
        
    # Execute the command
    print(f"Running meta-learning command: {' '.join(cmd)}")
    returncode, output = run_command(' '.join(cmd))
    
    return returncode

def run_fewshot_pipeline(config):
    """Run few-shot learning pipeline with the given configuration"""
    
    # Build command
    cmd_path = os.path.join(ROOT_DIR, "scripts", "train_fewshot_pipeline.py")
    
    # Extract config values
    task = config.get("task", TASK)
    model = config.get("model", MODEL_NAME)
    
    # Create a temporary config file with the required settings
    import tempfile
    import json
    
    # Extract few-shot specific settings from config
    fewshot_config = {
        'task': task,
        'model': model,
        'training_dir': config.get('training_dir', TRAINING_DIR),
        'output_dir': config.get('output_dir', OUTPUT_DIR),
        'k_shots': config.get('k_shots', 5),
        'adaptation_lr': config.get('adaptation_lr', 0.01),
        'adaptation_steps': config.get('adaptation_steps', 10),
        'finetune_all': config.get('finetune_all', False),
        'batch_size': config.get('batch_size', BATCH_SIZE),
        'win_len': config.get('win_len', WIN_LEN),
        'feature_size': config.get('feature_size', FEATURE_SIZE)
    }
    
    # Create a temporary file for the config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(fewshot_config, f, indent=4)
        temp_config_path = f.name
    
    # Set up command
    cmd = [
        "python", cmd_path,
        f"--task={task}",
        f"--model={model}",
        f"--config={temp_config_path}"
    ]
    
    # Execute the command
    print(f"Running few-shot pipeline command: {' '.join(cmd)}")
    returncode, output = run_command(' '.join(cmd))
    
    # Clean up temporary config file
    try:
        os.unlink(temp_config_path)
    except:
        pass
    
    return returncode

def run_multitask_direct(config):
    """Run multitask learning pipeline directly"""
    
    # Import needed modules
    import torch
    
    # Build command
    cmd_path = os.path.join(ROOT_DIR, "train_multi_model.py")
    
    # Extract config values
    tasks = config.get("tasks", [config.get("task", TASK)])
    if isinstance(tasks, str):
        # Convert comma-separated string to list if needed
        tasks = tasks.split(',')
    model = config.get("model", MODEL_NAME)
    
    # Parse test_splits
    test_splits = config.get("test_splits", "all")
    
    # Handle empty test_dirs array gracefully
    test_dirs = config.get("test_dirs", [])
    test_dirs_arg = f"--test_dirs={','.join(test_dirs)}" if test_dirs else ""
    
    # Set up command
    cmd = [
        "python", cmd_path,
        f"--tasks={','.join(tasks)}",
        f"--model={model}",
        f"--mode={config.get('mode', MODE)}",
        f"--training_dir={config.get('training_dir', TRAINING_DIR)}",
        test_dirs_arg,
        f"--output_dir={config.get('output_dir', OUTPUT_DIR)}",
        f"--seed={config.get('seed', SEED)}",
        f"--batch_size={config.get('batch_size', BATCH_SIZE)}",
        f"--epochs={config.get('epochs', EPOCH_NUMBER)}",
        f"--patience={config.get('patience', 15)}",
        f"--win_len={config.get('win_len', WIN_LEN)}",
        f"--feature_size={config.get('feature_size', FEATURE_SIZE)}",
        f"--device={DEVICE}",
        f"--test_splits={test_splits}",
        f"--pipeline=multitask"
    ]
    
    # Add optional parameters
    if "enable_few_shot" in config and config["enable_few_shot"]:
        cmd.append(f"--enable_few_shot")
        cmd.append(f"--k_shot={config.get('k_shot', 5)}")
        cmd.append(f"--inner_lr={config.get('inner_lr', 0.01)}")
        cmd.append(f"--num_inner_steps={config.get('num_inner_steps', 10)}")
        
    # Remove empty arguments
    cmd = [arg for arg in cmd if not arg.endswith("=")]
    
    # Execute the command
    print(f"Running multitask command: {' '.join(cmd)}")
    returncode, output = run_command(' '.join(cmd))
    
    return returncode

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
    """Run the main function"""
    parser = argparse.ArgumentParser(description='WiFi Sensing Pipeline Runner - Local Environment')
    
    # Pipeline selection
    parser.add_argument('--pipeline', type=str, choices=AVAILABLE_PIPELINES, default=PIPELINE,
                        help=f'Learning pipeline to run (choices: {", ".join(AVAILABLE_PIPELINES)})')
    
    # Basic parameters
    parser.add_argument('--model', type=str, choices=AVAILABLE_MODELS, default=MODEL_NAME,
                        help=f'Model architecture (choices: {", ".join(AVAILABLE_MODELS)})')
    parser.add_argument('--task', type=str, choices=AVAILABLE_TASKS, default=TASK,
                        help=f'Task to train on (choices: {", ".join(AVAILABLE_TASKS)})')
    parser.add_argument('--tasks', type=str, default=None,
                        help='Comma-separated list of tasks for multitask learning')
    
    # File paths
    parser.add_argument('--training_dir', type=str, default=TRAINING_DIR,
                        help='Directory containing the training data')
    parser.add_argument('--test_dirs', type=str, default=None,
                        help='Comma-separated list of directories containing test data')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='Directory to save output results')
    parser.add_argument('--config_file', type=str, default=None,
                        help='JSON configuration file to override defaults')
    
    # Model parameters
    parser.add_argument('--mode', type=str, default=MODE,
                        help='Data modality to use (csi or acf)')
    parser.add_argument('--win_len', type=int, default=WIN_LEN,
                        help='Window length for input data')
    parser.add_argument('--feature_size', type=int, default=FEATURE_SIZE,
                        help='Feature size for input data')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=EPOCH_NUMBER,
                        help='Number of epochs to train')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--test_splits', type=str, default='all',
                        help='Comma-separated list of test splits to use (or "all")')
    
    # Few-shot learning parameters
    parser.add_argument('--enable_few_shot', action='store_true',
                        help='Enable few-shot learning adaptation for cross-domain scenarios')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of examples per class for few-shot adaptation')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Learning rate for few-shot adaptation')
    parser.add_argument('--num_inner_steps', type=int, default=10,
                        help='Number of gradient steps for few-shot adaptation')
    
    # Supervised few-shot parameters
    parser.add_argument('--evaluate_fewshot', action='store_true',
                        help='Evaluate few-shot learning after supervised training')
    parser.add_argument('--fewshot_support_split', type=str, default='val_id',
                        help='Split to use for few-shot support set')
    parser.add_argument('--fewshot_query_split', type=str, default='test_cross_env',
                        help='Split to use for query set')
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
    
    # If tasks specified as comma-separated list, parse it
    if args.tasks is not None:
        args.tasks = args.tasks.split(',')
        
    # If test_dirs specified as comma-separated list, parse it
    if args.test_dirs is not None:
        args.test_dirs = args.test_dirs.split(',')
        
    # Create or load configuration
    config = create_or_load_config(args)
    
    # Save configuration for reference
    save_config(config, pipeline=args.pipeline)
    
    # Run appropriate pipeline
    if args.pipeline == 'supervised':
        return run_supervised_direct(config)
    elif args.pipeline == 'meta':
        return run_meta_learning(config)
    elif args.pipeline == 'multitask':
        return run_multitask_direct(config)
    elif args.pipeline == 'fewshot':
        return run_fewshot_pipeline(config)
    else:
        print(f"Unsupported pipeline: {args.pipeline}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
