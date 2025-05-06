#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - SageMaker Environment

This script allows you to run the supervised learning pipeline in the SageMaker environment.
It creates a SageMaker PyTorch Estimator for submitting training jobs.

Key features:
1. Batch execution of training tasks, with each task(task) using a single instance to run multiple models
2. Support for overriding default settings using JSON configuration files

Usage example:
```
import sagemaker_runner
runner = sagemaker_runner.SageMakerRunner()
runner.run_batch_by_task(tasks=['MotionSourceRecognition'], models=['vit', 'transformer'])
```
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3
import re
import numpy as np
import pandas as pd

# Default path settings
CODE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory containing the code
CONFIG_DIR = os.path.join(CODE_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "sagemaker_default_config.json")

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
            "s3_data_base": "s3://rnd-sagemaker/Data/Benchmark/",
            "s3_output_base": "s3://rnd-sagemaker/Benchmark_Log/",
            "mode": "csi",
            "task": "MotionSourceRecognition",
            "win_len": 250,
            "feature_size": 98,
            "seed": 42,
            "batch_size": 8,
            "epochs": 10,
            "model_name": "transformer", 
            "instance_type": "ml.g4dn.xlarge",
            "instance_count": 1,
            "framework_version": "1.12.1",
            "py_version": "py38",
            "base_job_name": "wifi-sensing-supervised",
            "batch_wait_time": 30,
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
                "Detection": 2,
                "demo": 2
            },
            "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit"],
            "available_tasks": ["MotionSourceRecognition", "HumanMotion", "DetectionandClassification", "HumanID", "NTUHAR", "HumanNonhuman", "NTUHumanID", "Widar", "ThreeClass", "Detection"],
            "available_pipelines": ["supervised", "meta", "multitask"],
            "enable_few_shot": False,
            "k_shot": 5,
            "inner_lr": 0.01,
            "num_inner_steps": 10
        }

# Load the default configuration
DEFAULT_CONFIG = load_default_config()

# Extract configuration values
S3_DATA_BASE = DEFAULT_CONFIG.get("s3_data_base", "s3://rnd-sagemaker/Data/Benchmark/")
S3_OUTPUT_BASE = DEFAULT_CONFIG.get("s3_output_base", "s3://rnd-sagemaker/Benchmark_Log/")
AVAILABLE_TASKS = DEFAULT_CONFIG.get("available_tasks", [
    "MotionSourceRecognition", 
])
AVAILABLE_MODELS = DEFAULT_CONFIG.get("available_models", ["mlp", "lstm", "resnet18", "transformer", "vit"])
TASK_CLASS_MAPPING = DEFAULT_CONFIG.get("task_class_mapping", {})
AVAILABLE_PIPELINES = DEFAULT_CONFIG.get("available_pipelines", ["supervised", "meta", "multitask"])

# SageMaker settings
INSTANCE_TYPE = DEFAULT_CONFIG.get("instance_type", "ml.g4dn.xlarge")
INSTANCE_COUNT = DEFAULT_CONFIG.get("instance_count", 1)
FRAMEWORK_VERSION = DEFAULT_CONFIG.get("framework_version", "1.12.1")
PY_VERSION = DEFAULT_CONFIG.get("py_version", "py38")
BASE_JOB_NAME = DEFAULT_CONFIG.get("base_job_name", "wifi-sensing-supervised")
EBS_VOLUME_SIZE = DEFAULT_CONFIG.get("ebs_volume_size", 30)  # Set default to 30GB

# Data modality
MODE = DEFAULT_CONFIG.get("mode", "csi")

# Default task
TASK = DEFAULT_CONFIG.get("task", "MotionSourceRecognition")

# Model parameters
WIN_LEN = DEFAULT_CONFIG.get("win_len", 250)
FEATURE_SIZE = DEFAULT_CONFIG.get("feature_size", 98)

# Common training parameters
SEED = DEFAULT_CONFIG.get("seed", 42)
BATCH_SIZE = DEFAULT_CONFIG.get("batch_size", 8)
EPOCH_NUMBER = DEFAULT_CONFIG.get("epochs", 10)
PATIENCE = DEFAULT_CONFIG.get("patience", 15)
MODEL_NAME = DEFAULT_CONFIG.get("model_name", "transformer")

# Batch settings
BATCH_WAIT_TIME = DEFAULT_CONFIG.get("batch_wait_time", 30)

class SageMakerRunner:
    """Class to handle SageMaker training job creation and execution"""
    
    def __init__(self, role=None):
        """Initialize SageMaker session and role"""
        self.session = sagemaker.Session()
        self.role = role or sagemaker.get_execution_role()
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M")  # Shorter format for job names
        
        # Verify default configuration
        self.default_config = DEFAULT_CONFIG
        
        # Verify the rnd-sagemaker bucket exists
        s3 = boto3.resource('s3')
        bucket_name = S3_DATA_BASE.split('/')[2]  # Extract bucket name from S3 path
        if bucket_name not in [bucket.name for bucket in s3.buckets.all()]:
            print(f"Error: The bucket '{bucket_name}' does not exist. Please create it first.")
            sys.exit(1)
        
        # 检查S3路径是否存在
        s3_client = boto3.client('s3')
        bucket = S3_DATA_BASE.split('/')[2]
        prefix = '/'.join(S3_DATA_BASE.split('/')[3:])
        if not prefix.endswith('/'):
            prefix += '/'
        
        try:
            # 尝试列出S3路径的内容
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
            
            if 'CommonPrefixes' in response:
                print(f"Contents of S3 path {S3_DATA_BASE}:")
                for obj in response['CommonPrefixes']:
                    folder = obj['Prefix'].split('/')[-2]
                    print(f"  - {folder}/")
                    
                # 检查是否存在tasks目录
                tasks_prefix = prefix + 'tasks/'
                tasks_resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=tasks_prefix, Delimiter='/')
                
                if 'CommonPrefixes' in tasks_resp:
                    print(f"Tasks available in {S3_DATA_BASE}tasks/:")
                    for obj in tasks_resp['CommonPrefixes']:
                        task_name = obj['Prefix'].split('/')[-2]
                        print(f"  - {task_name}/")
                else:
                    print(f"Warning: No tasks found in {S3_DATA_BASE}tasks/")
            else:
                print(f"Warning: S3 path {S3_DATA_BASE} appears to be empty")
        except Exception as e:
            print(f"Warning: Error checking S3 path {S3_DATA_BASE}: {e}")
        
        print(f"SageMaker Runner initialized:")
        print(f"  S3 Data Base: {S3_DATA_BASE}")
        print(f"  S3 Output Base: {S3_OUTPUT_BASE}")
        print(f"  Available Tasks: {', '.join(AVAILABLE_TASKS)}")
        print(f"  Available Models: {', '.join(AVAILABLE_MODELS)}")
        print(f"  Timestamp: {self.timestamp}")
    
    def convert_to_json_serializable(self, obj):
        """
        Recursively convert all NumPy types to Python native types for JSON serialization
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable object
        """
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    def run_batch_by_task(self, tasks=None, models=None, mode='csi', instance_type=None, wait_time=BATCH_WAIT_TIME, test_splits=None, enable_few_shot=False, k_shot=5, inner_lr=0.01, num_inner_steps=10):
        """
        Run batch processing by task, with each task executed on a single instance.
        
        Args:
            tasks (list): List of tasks to process (defaults to all available tasks)
            models (list): List of models to run for each task (defaults to all available models)
            mode (str): Data modality ('csi' or 'acf')
            instance_type (str): SageMaker instance type
            wait_time (int): Wait time between task submission in seconds
            test_splits (str): Test splits to use, comma-separated or 'all'
            enable_few_shot (bool): Whether to enable few-shot learning
            k_shot (int): Number of examples per class for few-shot learning
            inner_lr (float): Learning rate for few-shot adaptation
            num_inner_steps (int): Number of steps for few-shot adaptation
            
        Returns:
            list: List of job information dictionaries
        """
        print(f"Starting batch execution by task...")
        
        # Create batch timestamp for job naming
        batch_timestamp = self.timestamp
        
        # Set defaults if not provided
        tasks_to_run = tasks or AVAILABLE_TASKS
        models_to_run = models or AVAILABLE_MODELS
        
        # Convert to lists if strings are provided
        if isinstance(tasks_to_run, str):
            tasks_to_run = [tasks_to_run]
        if isinstance(models_to_run, str):
            models_to_run = [models_to_run]
        
        print(f"Tasks to run: {tasks_to_run}")
        print(f"Models to run: {models_to_run}")
        
        # Store all jobs
        all_jobs = []
        
        # Process each task
        for i, task_name in enumerate(tasks_to_run):
            print(f"\n----------------------------")
            print(f"Processing task {i+1}/{len(tasks_to_run)}: {task_name}")
            print(f"----------------------------")
            
            # Determine number of classes for this task
            num_classes = TASK_CLASS_MAPPING.get(task_name, 2)
            print(f"Task has {num_classes} classes")
            
            # Build hyperparameters dictionary
            hyperparameters = {
                # Data parameters
                "dataset_root": S3_DATA_BASE,  # Changed from data_dir to dataset_root
                "task_name": task_name,
                "mode": mode,
                "file_format": "h5",  # Add file format parameter
                "num_workers": 4,     # Add number of workers parameter
                
                # Model list - note this is a new parameter not supported by the standard script
                "models": ",".join(models_to_run),  # Comma-separated list of models
                
                # Training parameters
                "batch_size": BATCH_SIZE,
                "epochs": EPOCH_NUMBER,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "warmup_epochs": 5,
                "patience": PATIENCE,
                
                # Model parameters
                "win_len": WIN_LEN,
                "feature_size": FEATURE_SIZE,
                "seed": SEED,
                "save_dir": "/opt/ml/model",  # Use SageMaker model directory
                "output_dir": "/opt/ml/model",  # Set output_dir to model directory as well
                "data_key": 'CSI_amps',  # Add data_key parameter
                
                # Testing parameters
                "test_splits": test_splits or DEFAULT_CONFIG.get("test_splits", "all"),  # Default to all test splits
                
                # S3保存参数
                "save_to_s3": S3_OUTPUT_BASE,  # 在S3上保存结果
                
                # Few-shot learning parameters
                "enable_few_shot": enable_few_shot,
                "k_shot": k_shot,
                "inner_lr": inner_lr,
                "num_inner_steps": num_inner_steps
            }
            
            # 标志类型参数 - 传递空字符串而不是布尔值，这样在命令行中只会出现--flag而不是--flag True
            flag_parameters = [
                "debug",              # 启用详细日志记录
                "adaptive_path",      # 自动适应数据路径
                "try_all_paths",      # 尝试所有路径组合
                "direct_upload",      # 直接上传到S3
                "upload_final_model", # 上传最终模型
                "skip_train_for_debug", # 调试时跳过训练
            ]
            
            # Add enable_few_shot to flag parameters if it's enabled
            if enable_few_shot:
                flag_parameters.append("enable_few_shot")
            
            # 为所有标志类型参数设置空字符串
            for flag in flag_parameters:
                hyperparameters[flag] = ""
            
            # 从配置文件添加额外参数（如果有）
            # 仅添加在train_multi_model.py中定义的参数
            allowed_params = [
                "dataset_root", "task_name", "mode", "file_format", "num_workers",
                "models", "batch_size", "epochs", "learning_rate", "weight_decay",
                "warmup_epochs", "patience", "win_len", "feature_size", "seed",
                "save_dir", "output_dir", "data_key", "debug", "test_splits",
                "in_channels", "emb_dim", "d_model", "dropout", 
                "enable_few_shot", "k_shot", "inner_lr", "num_inner_steps"
            ]
            
            for key, value in DEFAULT_CONFIG.items():
                if key in allowed_params and key not in hyperparameters:
                    hyperparameters[key] = value
            
            # 打印完整的参数列表，用于调试命令行参数
            print("\nCommand line arguments that will be passed to the script:")
            cmd_args = []
            for key, value in hyperparameters.items():
                if isinstance(value, str) and value == "":
                    # 标志参数只添加--key，不添加值
                    cmd_args.append(f"--{key}")
                elif not (isinstance(value, bool) and not value):  # Skip False values
                    # 对于有值的参数，添加--key value
                    cmd_args.append(f"--{key} {value}")
            
            # 将参数分组显示，便于阅读
            print("\n标志参数:")
            flag_args = [arg for arg in cmd_args if "=" not in arg and " " not in arg]
            print("  " + "\n  ".join(flag_args))
            
            print("\n值参数:")
            value_args = [arg for arg in cmd_args if "=" in arg or " " in arg]
            print("  " + "\n  ".join(value_args))
            
            # 对将生成的命令行进行格式验证，确保没有明显问题
            print("\n完整命令行:")
            full_cmd = "train_multi_model.py " + " ".join(cmd_args)
            print(full_cmd)
            
            # 检查命令行长度，过长可能会导致问题
            if len(full_cmd) > 1000:
                print(f"\n警告: 命令行长度 ({len(full_cmd)}) 很长，可能导致问题")
            
            # 验证参数
            for param in ["dataset_root", "task_name", "models"]:
                if param not in hyperparameters or not hyperparameters[param]:
                    print(f"\n警告: 必需参数 {param} 缺失或为空!")
            
            print("\n")
            
            # Shorten task name for job naming (use first 8 chars or full name if shorter)
            short_task = task_name.lower()[:8]
            # Create job name with shortened format for timestamp and task name
            job_name = f"{BASE_JOB_NAME}-{short_task}-multi-{batch_timestamp}"
            
            # Ensure job name meets SageMaker's requirements (max 63 chars)
            if len(job_name) > 63:
                # If still too long, further shorten
                job_name = f"wifi-{short_task}-multi-{batch_timestamp[-8:]}"
            
            # Make sure job name is valid for SageMaker (only alphanumeric and hyphens)
            job_name = re.sub(r'[^a-zA-Z0-9-]', '-', job_name)
            
            # Output path
            s3_output_path = f"{S3_OUTPUT_BASE}{task_name}/"
            if not s3_output_path.endswith('/'):
                s3_output_path += '/'
            
            print(f"Creating training job for task '{task_name}' running models: {', '.join(models_to_run)}")
            print(f"Job name: {job_name}")
            print(f"Output path: {s3_output_path}")
            
            # Create PyTorch estimator
            instance_type_to_use = instance_type or INSTANCE_TYPE
            
            # Note: To reduce S3 storage of source code, you can:
            # 1. Use git_config instead of source_dir (requires code to be in a git repo)
            # 2. Use dependencies parameter for small scripts
            # 3. Set custom SAGEMAKER_SUBMIT_DIRECTORY environment variable
            estimator = PyTorch(
                entry_point="train_multi_model.py",  # Note: This uses a new training script
                #source_dir=".",  # Using source_dir will upload a copy to S3 as sourcedir.tar.gz
                source_dir=".",  # Still need this for local development
                # If your code is in a git repo, you can use this to avoid creating sourcedir.tar.gz:
                # git_config={
                #    'repo': 'https://github.com/your-username/your-repo.git',
                #    'branch': 'main'
                # },
                role=self.role,
                framework_version=FRAMEWORK_VERSION,
                py_version=PY_VERSION,
                instance_count=INSTANCE_COUNT,
                instance_type=instance_type_to_use,
                max_run=86400 * 3,  # 72 hours max runtime
                keep_alive_period_in_seconds=1800,  # 30 min keep alive after training
                output_path=s3_output_path,
                base_job_name=job_name,
                hyperparameters=hyperparameters,
                metric_definitions=[
                    {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
                    {'Name': 'train:accuracy', 'Regex': 'Train Accuracy: ([0-9\\.]+)'},
                    {'Name': 'validation:loss', 'Regex': 'Val Loss: ([0-9\\.]+)'},
                    {'Name': 'validation:accuracy', 'Regex': 'Val Accuracy: ([0-9\\.]+)'}
                ],
                volume_size=EBS_VOLUME_SIZE,  # Increase EBS volume size
                debugger_hook_config=False,  # Disable debugger
                disable_profiler=True,        # Disable profiler
                environment={
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',  # Ensure code is properly located
                    'PYTHONPATH': '/opt/ml/code',                  # Add code directory to Python path
                    'SAGEMAKER_MODEL_SERVER_WORKERS': '1',         # Limit number of model server workers
                    'SM_DISABLE_PROFILER': 'true',                 # Disable profiler via env var
                    'SM_DISABLE_DEBUGGER': 'true',                 # Disable debugger via env var
                    'LOG_LEVEL': 'INFO'                            # Set logging level
                }
            )
            
            # Prepare data inputs with more explicit configuration
            data_channels = {
                'training': TrainingInput(
                    s3_data=S3_DATA_BASE,
                    distribution='FullyReplicated',
                    content_type='application/x-directory',
                    s3_data_type='S3Prefix',
                    input_mode='File'
                )
            }
            
            # 添加调试信息
            print(f"S3 data path for training: {S3_DATA_BASE}")
            print(f"Training input configuration: {data_channels['training']}")
            
            # 在实际的SageMaker环境中，数据将被下载到以下路径:
            print(f"In SageMaker environment, data will be downloaded to: /opt/ml/input/data/training/")
            print(f"Expected task path in SageMaker: /opt/ml/input/data/training/tasks/{task_name}/")
            
            # Start training job
            print(f"Starting SageMaker training job...")
            estimator.fit(inputs=data_channels, job_name=job_name, wait=False)
            
            # Create job info
            job_info = {
                'job_name': job_name,
                'estimator': estimator,
                'inputs': data_channels,
                'config': {
                    'task_name': task_name,
                    'output_dir': s3_output_path,
                    'model_name': 'multi-model'  # Note this is no longer a single model name
                },
                'models': models_to_run,
                'batch_id': batch_timestamp,
                'task_group': task_name
            }
            
            # Add to job lists
            all_jobs.append(job_info)
            
            # Wait longer between tasks
            if wait_time > 0 and task_name != tasks_to_run[-1]:
                print(f"Waiting {wait_time} seconds before starting next task...")
                try:
                    time.sleep(wait_time)
                except KeyboardInterrupt:
                    print("\nBatch submission interrupted by user.")
                    break
        
        # Return batch information
        batch_info = {
            'batch_timestamp': batch_timestamp,
            'batch_mode': 'by-task',
            'tasks': tasks_to_run,
            'models': models_to_run,
            'instance_type': instance_type or INSTANCE_TYPE,
            'jobs': all_jobs
        }
        
        # Update batch summary to create initial status report
        self._update_batch_summary(all_jobs, batch_timestamp)
        
        print(f"\nBatch execution initiated!")
        print(f"Tasks: {len(tasks_to_run)}")
        print(f"Models: {len(models_to_run)}")
        print(f"Total jobs: {len(all_jobs)}")
        print(f"Batch ID: {batch_timestamp}")
        print(f"You can monitor the jobs in SageMaker console.")
        
        return batch_info
    
    def _update_batch_summary(self, jobs, batch_timestamp):
        """Update batch summary file with job details"""
        summary_dir = os.path.join(CODE_DIR, "batch_summaries")
        os.makedirs(summary_dir, exist_ok=True)
        
        # Create text and JSON summaries
        summary_text_file = os.path.join(summary_dir, f"batch_summary_{batch_timestamp}.txt")
        summary_json_file = os.path.join(summary_dir, f"batch_summary_{batch_timestamp}.json")
        
        # Create text summary
        with open(summary_text_file, "w") as f:
            f.write(f"Batch Training Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Jobs: {len(jobs)}\n")
            f.write(f"Batch Timestamp: {batch_timestamp}\n\n")
            
            for job_info in jobs:
                f.write(f"Job: {job_info['job_name']}\n")
                # 安全地获取输入数据路径
                try:
                    # 直接访问配置信息中的S3_DATA_BASE
                    input_path = S3_DATA_BASE
                    f.write(f"  Input: {input_path}\n")
                except Exception as e:
                    f.write(f"  Input: S3 path (could not get exact path, error: {str(e)})\n")
                    
                f.write(f"  Output: {job_info['config']['output_dir']}\n")
                f.write(f"  Task: {job_info['config']['task_name']}\n")
                
                # Add information about multiple models if available
                if 'models' in job_info:
                    f.write(f"  Models: {', '.join(job_info['models'])}\n")
                    
                f.write("\n")
        
        # Create JSON summary (easier to parse programmatically)
        summary_data = {
            "timestamp": batch_timestamp,
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_jobs": len(jobs),
            "jobs": {}
        }
        
        for i, job_info in enumerate(jobs):
            job_key = f"job_{i}_{job_info['config']['task_name']}"
            summary_data["jobs"][job_key] = {
                "job_name": job_info['job_name'],
                "task": job_info['config']['task_name'],
                "input": S3_DATA_BASE,  # 使用全局变量而不是对象属性
                "output": job_info['config']['output_dir']
            }
            
            # Add information about multiple models if available
            if 'models' in job_info:
                summary_data["jobs"][job_key]["models"] = job_info['models']
        
        # Ensure all data is JSON serializable
        summary_data = self.convert_to_json_serializable(summary_data)
        
        with open(summary_json_file, "w") as f:
            json.dump(summary_data, f, indent=2)

    def test_hyperparameters(self, task_name=TASK, models=None):
        """
        测试超参数解析，用于本地调试参数传递问题
        
        Args:
            task_name: 任务名称
            models: 要测试的模型列表
        
        Returns:
            None, 但会打印出解析结果
        """
        print(f"Testing hyperparameter parsing for task: {task_name}")
        
        # 使用与run_batch_by_task相同的逻辑构建超参数
        models_to_run = models or AVAILABLE_MODELS
        
        # 构建超参数
        hyperparameters = {
            # 数据参数
            "dataset_root": S3_DATA_BASE,
            "task_name": task_name,
            "mode": MODE,
            "file_format": "h5",
            "num_workers": 4,
            
            # 模型列表
            "models": ",".join(models_to_run),
            
            # 训练参数
            "batch_size": BATCH_SIZE,
            "epochs": EPOCH_NUMBER,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "warmup_epochs": 5,
            "patience": PATIENCE,
            
            # 模型参数
            "win_len": WIN_LEN,
            "feature_size": FEATURE_SIZE,
            "seed": SEED,
            "save_dir": "./model",
            "output_dir": "./output",
            "data_key": 'CSI_amps',
            
            # Testing parameters
            "test_splits": DEFAULT_CONFIG.get("test_splits", "all"),  # Default to all test splits
            
            # S3参数
            "save_to_s3": S3_OUTPUT_BASE,
        }
        
        # 添加标志参数
        flag_parameters = [
            "debug", "adaptive_path", "try_all_paths", 
            "direct_upload", "upload_final_model", "skip_train_for_debug"
        ]
        
        for flag in flag_parameters:
            hyperparameters[flag] = ""
        
        # 生成命令行并进行模拟解析
        cmd_args = []
        for key, value in hyperparameters.items():
            if isinstance(value, str) and value == "":
                cmd_args.append(f"--{key}")
            elif not (isinstance(value, bool) and not value):
                cmd_args.append(f"--{key}")
                cmd_args.append(str(value))
        
        print("\n生成的命令行参数:")
        print(" ".join(cmd_args))
        
        # 尝试使用Python的argparse模块解析参数
        try:
            import subprocess
            import sys
            
            # 创建测试脚本
            test_script = """
import sys
import argparse

def test_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str)
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--models', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--adaptive_path', action='store_true')
    parser.add_argument('--try_all_paths', action='store_true')
    parser.add_argument('--direct_upload', action='store_true')
    parser.add_argument('--upload_final_model', action='store_true')
    parser.add_argument('--skip_train_for_debug', action='store_true')
    
    args, unknown = parser.parse_known_args()
    print("\\n解析结果:")
    for arg, val in vars(args).items():
        print(f"  {arg}: {val}")
    if unknown:
        print("\\n未知参数:")
        print(f"  {unknown}")

if __name__ == '__main__':
    print("测试参数解析...")
    print(f"参数: {sys.argv[1:]}")
    test_parse()
"""
            # 保存测试脚本
            test_file = "test_args_parse.py"
            with open(test_file, "w") as f:
                f.write(test_script)
            
            # 运行测试脚本，传入我们的参数
            cmd = [sys.executable, test_file] + cmd_args
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            print("\n参数解析测试结果:")
            print(result.stdout)
            
            if result.stderr:
                print("\n错误输出:")
                print(result.stderr)
            
            # 清理
            try:
                os.remove(test_file)
            except:
                pass
                
        except Exception as e:
            print(f"测试参数解析时出错: {e}")

    def run_multitask(self, tasks=None, model_type="transformer", instance_type=None, test_splits=None, enable_few_shot=False, k_shot=5, inner_lr=0.01, num_inner_steps=10):
        """
        Run multitask learning pipeline.
        
        Args:
            tasks (str): Comma-separated list of tasks to train on
            model_type (str): Model architecture to use
            instance_type (str): SageMaker instance type
            test_splits (str): Test splits to use, comma-separated or 'all'
            enable_few_shot (bool): Whether to enable few-shot learning
            k_shot (int): Number of examples per class for few-shot learning
            inner_lr (float): Learning rate for few-shot adaptation
            num_inner_steps (int): Number of steps for few-shot adaptation
            
        Returns:
            dict: Job information
        """
        print(f"Running multitask learning for model: {model_type}, tasks: {tasks}")
        
        # Create batch timestamp for job naming
        timestamp = self.timestamp
        
        # Ensure tasks is a string
        if isinstance(tasks, list):
            tasks = ','.join(tasks)
        elif tasks is None:
            # Default to MotionSourceRecognition
            tasks = "MotionSourceRecognition"
        
        # Build hyperparameters dictionary
        hyperparameters = {
            # Data parameters
            "dataset_root": S3_DATA_BASE,  # Use S3 path
            "tasks": tasks,
            "mode": MODE,
            "file_format": "h5",  # Add file format parameter
            "num_workers": 4,     # Add number of workers parameter
            
            # Training parameters
            "batch_size": BATCH_SIZE,
            "epochs": EPOCH_NUMBER,
            "lr": 1e-4,          # Note: multitask uses lr, not learning_rate
            "patience": PATIENCE,
            
            # Model parameters
            "model": model_type,
            "win_len": WIN_LEN,
            "feature_size": FEATURE_SIZE,
            "seed": SEED,
            "save_dir": "/opt/ml/model",  # Use SageMaker model directory
            "output_dir": "/opt/ml/model",  # Set output_dir to model directory as well
            
            # LoRA parameters
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            
            # Testing parameters
            "test_splits": test_splits or DEFAULT_CONFIG.get("test_splits", "all"),  # Default to all test splits
            
            # S3保存参数
            "save_to_s3": S3_OUTPUT_BASE,  # 在S3上保存结果
            
            # Few-shot learning parameters
            "k_shot": k_shot,
            "inner_lr": inner_lr,
            "num_inner_steps": num_inner_steps
        }
        
        # 标志类型参数 - 传递空字符串而不是布尔值，这样在命令行中只会出现--flag而不是--flag True
        flag_parameters = [
            "debug",              # 启用详细日志记录
            "adaptive_path",      # 自动适应数据路径
            "direct_upload",      # 直接上传到S3
            "upload_final_model", # 上传最终模型
        ]
        
        # Add enable_few_shot to flag parameters if it's enabled
        if enable_few_shot:
            flag_parameters.append("enable_few_shot")
        
        # 为所有标志类型参数设置空字符串
        for flag in flag_parameters:
            hyperparameters[flag] = ""
        
        # 打印完整的参数列表，用于调试命令行参数
        print("\nCommand line arguments that will be passed to the script:")
        cmd_args = []
        for key, value in hyperparameters.items():
            if isinstance(value, str) and value == "":
                # 标志参数只添加--key，不添加值
                cmd_args.append(f"--{key}")
            elif not (isinstance(value, bool) and not value):  # Skip False values
                # 对于有值的参数，添加--key value
                cmd_args.append(f"--{key} {value}")
        
        # 将参数分组显示，便于阅读
        print("\n标志参数:")
        flag_args = [arg for arg in cmd_args if "=" not in arg and " " not in arg]
        print("  " + "\n  ".join(flag_args))
        
        print("\n值参数:")
        value_args = [arg for arg in cmd_args if "=" in arg or " " in arg]
        print("  " + "\n  ".join(value_args))
        
        # 对将生成的命令行进行格式验证，确保没有明显问题
        print("\n完整命令行:")
        full_cmd = "train_multitask_adapter.py " + " ".join(cmd_args)
        print(full_cmd)
        
        # 检查命令行长度，过长可能会导致问题
        if len(full_cmd) > 1000:
            print(f"\n警告: 命令行长度 ({len(full_cmd)}) 很长，可能导致问题")
        
        # 验证参数
        for param in ["dataset_root", "tasks", "model"]:
            if param not in hyperparameters or not hyperparameters[param]:
                print(f"\n警告: 必需参数 {param} 缺失或为空!")
        
        print("\n")
        
        # Shorten task name for job naming (use first 8 chars or full name if shorter)
        short_tasks = tasks.lower()[:8]
        # Create job name with shortened format for timestamp and task name
        job_name = f"{BASE_JOB_NAME}-{short_tasks}-{model_type}-{timestamp[-6:]}"
        
        # Ensure job name meets SageMaker requirements
        job_name = re.sub(r'[^a-zA-Z0-9-]', '-', job_name)
        if len(job_name) > 63:
            job_name = job_name[:60] + timestamp[-3:]
        
        # Output path - use a structure that matches the local runner
        s3_output_path = f"{S3_OUTPUT_BASE}multitask/"
        
        print(f"Creating multitask learning job: {job_name}")
        print(f"Output path: {s3_output_path}")
        
        # Create PyTorch estimator
        instance_type_to_use = instance_type or INSTANCE_TYPE
        
        estimator = PyTorch(
            entry_point="train_multitask_adapter.py",
            source_dir=".",
            role=self.role,
            framework_version=FRAMEWORK_VERSION,
            py_version=PY_VERSION,
            instance_count=INSTANCE_COUNT,
            instance_type=instance_type_to_use,
            max_run=86400,  # 24 hours max runtime
            output_path=s3_output_path,
            base_job_name=job_name,
            hyperparameters=hyperparameters,
            volume_size=EBS_VOLUME_SIZE,
            environment={
                'PYTHONPATH': '/opt/ml/code',  # Add code directory to Python path
                'LOG_LEVEL': 'INFO'            # Set logging level
            }
        )
        
        # Prepare data inputs
        data_channels = {
            'training': TrainingInput(
                s3_data=S3_DATA_BASE,
                distribution='FullyReplicated',
                content_type='application/x-directory',
                s3_data_type='S3Prefix',
                input_mode='File'
            )
        }
        
        # Start training job
        print(f"Starting SageMaker training job...")
        estimator.fit(inputs=data_channels, job_name=job_name, wait=False)
        
        # Create job info
        job_info = {
            'job_name': job_name,
            'estimator': estimator,
            'inputs': data_channels,
            'config': {
                'tasks': tasks,
                'model_type': model_type,
                'output_dir': s3_output_path,
                'save_dir': '/opt/ml/model/results/multitask'
            }
        }
        
        print(f"Multitask learning job submitted: {job_name}")
        print(f"You can monitor the job in SageMaker console.")
        
        return job_info
    
    def run_batch_multitask(self, task_groups=None, model_type="transformer", instance_type=None, wait_time=BATCH_WAIT_TIME, test_splits=None, enable_few_shot=False, k_shot=5, inner_lr=0.01, num_inner_steps=10):
        """
        Run batch of multitask learning jobs, each with a different group of tasks.
        
        Args:
            task_groups (list): List of task groups, where each group is a list of tasks
            model_type (str): Model architecture to use
            instance_type (str): SageMaker instance type
            wait_time (int): Wait time between job submission in seconds
            test_splits (str): Test splits to use, comma-separated or 'all'
            enable_few_shot (bool): Whether to enable few-shot learning
            k_shot (int): Number of examples per class for few-shot learning
            inner_lr (float): Learning rate for few-shot adaptation
            num_inner_steps (int): Number of steps for few-shot adaptation
            
        Returns:
            list: List of job information dictionaries
        """
        print(f"Starting batch execution of multitask learning jobs...")
        
        # Create batch timestamp
        batch_timestamp = self.timestamp
        
        # Default task groups if none provided
        if task_groups is None:
            # Create two default groups
            task_groups = [
                ['MotionSourceRecognition', 'HumanMotion'],
                ['MotionSourceRecognition', 'HumanID'],
                ['MotionSourceRecognition', 'HumanMotion', 'HumanID']
            ]
        
        # Convert strings to lists if needed
        processed_groups = []
        for group in task_groups:
            if isinstance(group, str):
                # If it's a comma-separated string
                processed_groups.append(group.split(','))
            else:
                # Keep as is if already a list
                processed_groups.append(group)
        
        print(f"Running {len(processed_groups)} task groups with model type: {model_type}")
        for i, group in enumerate(processed_groups):
            print(f"Group {i+1}: {', '.join(group)}")
        
        # Store all jobs
        all_jobs = []
        
        # Run each task group
        for i, task_group in enumerate(processed_groups):
            print(f"\n----------------------------")
            print(f"Processing task group {i+1}/{len(processed_groups)}: {', '.join(task_group)}")
            print(f"----------------------------")
            
            # Run multitask job
            job_info = self.run_multitask(
                tasks=task_group,
                model_type=model_type,
                instance_type=instance_type,
                test_splits=test_splits,
                enable_few_shot=enable_few_shot,
                k_shot=k_shot,
                inner_lr=inner_lr,
                num_inner_steps=num_inner_steps
            )
            
            # Add batch information
            job_info['batch_id'] = batch_timestamp
            all_jobs.append(job_info)
            
            # Wait between submissions
            if wait_time > 0 and i < len(processed_groups) - 1:
                print(f"Waiting {wait_time} seconds before starting next task group...")
                try:
                    time.sleep(wait_time)
                except KeyboardInterrupt:
                    print("\nBatch submission interrupted by user.")
                    break
        
        # Return batch information
        batch_info = {
            'batch_timestamp': batch_timestamp,
            'batch_mode': 'multitask-groups',
            'task_groups': processed_groups,
            'model_type': model_type,
            'instance_type': instance_type or INSTANCE_TYPE,
            'enable_few_shot': enable_few_shot,
            'k_shot': k_shot,
            'inner_lr': inner_lr,
            'num_inner_steps': num_inner_steps,
            'jobs': all_jobs
        }
        
        # Update batch summary
        self._update_batch_summary(all_jobs, batch_timestamp)
        
        print(f"\nBatch execution initiated!")
        print(f"Task Groups: {len(processed_groups)}")
        print(f"Model: {model_type}")
        print(f"Total jobs: {len(all_jobs)}")
        print(f"Batch ID: {batch_timestamp}")
        print(f"Few-shot learning: {'Enabled' if enable_few_shot else 'Disabled'}")
        if enable_few_shot:
            print(f"K-shot: {k_shot}, Inner LR: {inner_lr}, Inner Steps: {num_inner_steps}")
        print(f"You can monitor the jobs in SageMaker console.")
        
        return batch_info

def main():
    """Main entry point for SageMaker runner"""
    parser = argparse.ArgumentParser(description="Run SageMaker training jobs for WiFi sensing")
    
    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Supervised batch parser
    batch_parser = subparsers.add_parser('batch', help='Run batch of supervised learning jobs')
    batch_parser.add_argument('--tasks', type=str, nargs='+', help='List of tasks to run')
    batch_parser.add_argument('--models', type=str, nargs='+', help='List of models to run')
    batch_parser.add_argument('--mode', type=str, default='csi', help='Data modality (csi or acf)')
    batch_parser.add_argument('--instance_type', type=str, help='SageMaker instance type')
    batch_parser.add_argument('--wait_time', type=int, default=BATCH_WAIT_TIME, help='Wait time between jobs')
    batch_parser.add_argument('--test_splits', type=str, help='Test splits to use (comma-separated or "all")')
    
    # Few-shot learning parameters
    batch_parser.add_argument('--enable_few_shot', action='store_true', help='Enable few-shot learning adaptation')
    batch_parser.add_argument('--k_shot', type=int, default=5, help='Number of examples per class for few-shot adaptation')
    batch_parser.add_argument('--inner_lr', type=float, default=0.01, help='Learning rate for few-shot adaptation')
    batch_parser.add_argument('--num_inner_steps', type=int, default=10, help='Number of adaptation steps for few-shot learning')
    
    # Multitask parser
    multitask_parser = subparsers.add_parser('multitask', help='Run multitask learning job')
    multitask_parser.add_argument('--tasks', type=str, required=True, help='Comma-separated list of tasks')
    multitask_parser.add_argument('--model', type=str, default='transformer', help='Model architecture')
    multitask_parser.add_argument('--instance_type', type=str, help='SageMaker instance type')
    multitask_parser.add_argument('--test_splits', type=str, help='Test splits to use (comma-separated or "all")')
    
    # Few-shot learning parameters for multitask
    multitask_parser.add_argument('--enable_few_shot', action='store_true', help='Enable few-shot learning adaptation')
    multitask_parser.add_argument('--k_shot', type=int, default=5, help='Number of examples per class for few-shot adaptation')
    multitask_parser.add_argument('--inner_lr', type=float, default=0.01, help='Learning rate for few-shot adaptation')
    multitask_parser.add_argument('--num_inner_steps', type=int, default=10, help='Number of adaptation steps for few-shot learning')
    
    # Batch multitask parser
    batch_mt_parser = subparsers.add_parser('batch-multitask', help='Run batch of multitask learning jobs')
    batch_mt_parser.add_argument('--task_groups', type=str, nargs='+', help='List of task groups (format: "task1,task2 task3,task4")')
    batch_mt_parser.add_argument('--model', type=str, default='transformer', help='Model architecture')
    batch_mt_parser.add_argument('--instance_type', type=str, help='SageMaker instance type')
    batch_mt_parser.add_argument('--wait_time', type=int, default=BATCH_WAIT_TIME, help='Wait time between jobs')
    batch_mt_parser.add_argument('--test_splits', type=str, help='Test splits to use (comma-separated or "all")')
    
    # Few-shot learning parameters for batch multitask
    batch_mt_parser.add_argument('--enable_few_shot', action='store_true', help='Enable few-shot learning adaptation')
    batch_mt_parser.add_argument('--k_shot', type=int, default=5, help='Number of examples per class for few-shot adaptation')
    batch_mt_parser.add_argument('--inner_lr', type=float, default=0.01, help='Learning rate for few-shot adaptation')
    batch_mt_parser.add_argument('--num_inner_steps', type=int, default=10, help='Number of adaptation steps for few-shot learning')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create SageMaker runner
    runner = SageMakerRunner()
    
    if args.command == 'batch':
        # Run batch jobs
        runner.run_batch_by_task(
            tasks=args.tasks,
            models=args.models,
            mode=args.mode,
            instance_type=args.instance_type,
            wait_time=args.wait_time,
            test_splits=args.test_splits,
            enable_few_shot=args.enable_few_shot,
            k_shot=args.k_shot,
            inner_lr=args.inner_lr,
            num_inner_steps=args.num_inner_steps
        )
    elif args.command == 'multitask':
        # Run multitask job
        runner.run_multitask(
            tasks=args.tasks,
            model_type=args.model,
            instance_type=args.instance_type,
            test_splits=args.test_splits,
            enable_few_shot=args.enable_few_shot,
            k_shot=args.k_shot,
            inner_lr=args.inner_lr,
            num_inner_steps=args.num_inner_steps
        )
    elif args.command == 'batch-multitask':
        # Run batch multitask jobs
        runner.run_batch_multitask(
            task_groups=args.task_groups,
            model_type=args.model,
            instance_type=args.instance_type,
            wait_time=args.wait_time,
            test_splits=args.test_splits,
            enable_few_shot=args.enable_few_shot,
            k_shot=args.k_shot,
            inner_lr=args.inner_lr,
            num_inner_steps=args.num_inner_steps
        )
    else:
        # If no command provided, show help
        parser.print_help()

if __name__ == "__main__":
    main()
