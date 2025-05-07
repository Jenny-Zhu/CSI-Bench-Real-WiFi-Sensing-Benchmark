#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - Local Environment

This script serves as the main entry point for WiFi sensing benchmark.
It incorporates functionality from train.py, run_model.py, and the original local_runner.py.

配置文件管理说明：
1. configs文件夹现在只包含模板配置文件
2. 运行生成的配置文件会保存到results文件夹中，使用统一的目录结构: results/TASK/MODEL/EXPERIMENT_ID/
   - 监督学习：results/TASK/MODEL/EXPERIMENT_ID/supervised_config.json
   - 多任务学习：results/TASK/MODEL/EXPERIMENT_ID/multitask_config.json
3. 所有运行时参数都应从配置文件加载，不再使用命令行参数

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

# 确保results目录存在
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
    获取多任务学习流水线的配置
    
    Args:
        custom_config: 自定义配置字典
        
    Returns:
        配置字典
    """
    # 必须提供自定义配置
    if custom_config is None:
        print("错误: 必须提供配置参数!")
        sys.exit(1)
    
    # 确保有任务名称
    if 'task' not in custom_config and 'tasks' in custom_config:
        # 如果没有单一任务名但有tasks列表，使用第一个任务作为task名
        tasks = custom_config.get('tasks')
        if isinstance(tasks, str):
            # 如果是字符串，可能是逗号分隔的列表
            task_list = tasks.split(',')
            if task_list:
                custom_config['task'] = task_list[0]
        elif isinstance(tasks, list) and tasks:
            # 如果是列表且不为空
            custom_config['task'] = tasks[0]
    
    # 如果仍然没有task，设置默认值
    if 'task' not in custom_config:
        custom_config['task'] = 'multitask'
    
    # 创建配置字典
    config = {
        # 数据参数
        'training_dir': custom_config['training_dir'],
        'output_dir': custom_config['output_dir'],
        'results_subdir': f"{custom_config['model']}_{custom_config['task'].lower()}",
        
        # 训练参数
        'batch_size': custom_config['batch_size'],
        'learning_rate': custom_config.get('learning_rate', 5e-4),
        'weight_decay': custom_config.get('weight_decay', 1e-5),
        'epochs': custom_config['epochs'],
        'win_len': custom_config['win_len'],
        'feature_size': custom_config['feature_size'],
        
        # 模型参数
        'model': custom_config['model'],
        'emb_dim': custom_config.get('emb_dim', 128),
        'dropout': custom_config.get('dropout', 0.1),
        
        # 任务参数 - 保留task和tasks
        'task': custom_config.get('task'),
        'tasks': custom_config.get('tasks'),
    }
    
    # 如果transformer_config.json存在，尝试加载它
    transform_path = os.path.join(CONFIG_DIR, "transformer_config.json")
    if os.path.exists(transform_path):
        print(f"使用现有配置文件: {transform_path}")
        with open(transform_path, 'r') as f:
            transformer_config = json.load(f)
            for k, v in transformer_config.items():
                if k in config:
                    config[k] = v
    
    # 确保tasks参数存在且格式正确
    if not config.get('tasks'):
        if config.get('task'):
            config['tasks'] = config['task']
        else:
            print("错误: 多任务配置必须指定 'tasks' 或 'task' 参数!")
            sys.exit(1)
    
    # 如果model_params存在，添加到config中
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
    experiment_id = config.get('experiment_id', f"params_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Get base directories
    base_output_dir = config.get('output_dir')
    training_dir = config.get('training_dir')
    
    # Update save_dir to include task/model/experiment structure
    model_output_dir = os.path.join(base_output_dir, task_name, model_name, experiment_id)
    
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
    cmd += f" --experiment_id=\"{experiment_id}\""
    
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
    print(f"运行监督学习: {cmd}")
    return_code = subprocess.call(cmd, shell=True)
    
    if return_code != 0:
        print(f"运行监督学习出错: 返回码 {return_code}")
    else:
        print("监督学习成功完成.")
    
    return return_code

def run_multitask_direct(config):
    """
    Run multitask learning pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Return code (0 for success, non-zero for failure)
    """
    print("运行多任务学习，配置如下:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Get the tasks parameter correctly
    tasks = config.get('tasks')
    if not tasks:
        print("错误: 'tasks'参数缺失或为空。请至少指定一个任务。")
        return 1
    
    # Ensure tasks is properly formatted - should be a comma-separated string without spaces
    if isinstance(tasks, list):
        tasks = ','.join(tasks)
    
    # 获取experiment_id
    experiment_id = config.get('experiment_id', f"params_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Build command directly as a string for better control of quotes and escaping
    cmd = f"{sys.executable} {os.path.join(SCRIPT_DIR, 'train_multitask_adapter.py')}"
    cmd += f" --tasks=\"{tasks}\""
    cmd += f" --model={config.get('model')}"
    cmd += f" --data_dir=\"{config.get('training_dir')}\""
    cmd += f" --epochs={config.get('epochs')}"
    cmd += f" --batch_size={config.get('batch_size')}"
    cmd += f" --win_len={config.get('win_len')}"
    cmd += f" --feature_size={config.get('feature_size')}"
    cmd += f" --experiment_id=\"{experiment_id}\""
    
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
    print(f"运行命令: {cmd}")
    return subprocess.call(cmd, shell=True)

def save_config(config, pipeline='supervised'):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        pipeline: 流水线类型 ('supervised' 或 'multitask')
        
    Returns:
        保存的配置文件路径
    """
    # 注意：配置文件现在保存在results目录而不是configs目录
    # 监督学习和多任务学习都使用相同的目录结构：results/TASK/MODEL/EXPERIMENT_ID/
    # 获取基本参数
    base_output_dir = config.get('output_dir', './results')
    model = config.get('model')
    
    # 使用传入的experiment_id或生成一个新的
    experiment_id = config.get('experiment_id')
    if not experiment_id:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f"params_{timestamp}"
        # 将新生成的experiment_id添加到配置中
        config['experiment_id'] = experiment_id
    
    # 获取task名称
    task_name = config.get('task', 'default_task').lower()
    
    # 创建目录结构 - 无论是监督学习还是多任务学习都使用相同结构
    task_dir = os.path.join(base_output_dir, task_name)
    model_dir = os.path.join(task_dir, model)
    experiment_dir = os.path.join(model_dir, experiment_id)
    
    # 确保目录存在
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 生成配置文件名
    if pipeline == 'multitask':
        config_filename = os.path.join(experiment_dir, f"multitask_config.json")
    else:
        config_filename = os.path.join(experiment_dir, f"supervised_config.json")
    
    # 保存配置
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"配置已保存到 {config_filename}")
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
    
    # Generate a unique experiment ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_id = f"params_{timestamp}"
    
    # Get pipeline-specific configuration
    if pipeline == 'multitask':
        config = get_multitask_config(config)
    else:  # Default to supervised
        config = get_supervised_config(config)
    
    # 添加experiment_id到配置中
    config['experiment_id'] = experiment_id
    
    # Save configuration for future reference
    config_file = save_config(config, pipeline)
    
    # Run appropriate pipeline
    if pipeline == 'multitask':
        return run_multitask_direct(config)
    else:  # Default to supervised
        return run_supervised_direct(config)

if __name__ == "__main__":
    sys.exit(main())
