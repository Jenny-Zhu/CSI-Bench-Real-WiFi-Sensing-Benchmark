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
            "framework_version": "1.13.1",
            "py_version": "py39",
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
            "available_pipelines": ["supervised", "meta", "multitask"]
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
FRAMEWORK_VERSION = DEFAULT_CONFIG.get("framework_version", "1.13.1")
PY_VERSION = DEFAULT_CONFIG.get("py_version", "py39")
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
    
    def run_batch_by_task(self, tasks=None, models=None, mode='csi', instance_type=None, wait_time=BATCH_WAIT_TIME):
        """
        Run batch training jobs by task (each task using a single instance to run all models)
        
        Args:
            tasks (list): List of tasks to run. If None, use all available tasks
            models (list): List of models to run. If None, use all available models
            mode (str): Data modality ('csi' or 'acf')
            instance_type (str): SageMaker instance type
            wait_time (int): Wait time between job submissions in seconds
            
        Returns:
            dict: Dictionary containing details of all launched jobs
        """
        print(f"Starting batch execution by task...")
        
        # Use provided tasks or available tasks
        if tasks is None or len(tasks) == 0:
            tasks_to_run = AVAILABLE_TASKS
        else:
            tasks_to_run = [t for t in tasks if t in AVAILABLE_TASKS]
            if len(tasks_to_run) < len(tasks):
                print(f"Warning: Some requested tasks are not in the available tasks list.")
        
        # Use provided models or available models
        if models is None or len(models) == 0:
            models_to_run = AVAILABLE_MODELS
        else:
            models_to_run = [m for m in models if m in AVAILABLE_MODELS]
            if len(models_to_run) < len(models):
                print(f"Warning: Some requested models are not in the available models list.")
        
        print(f"Tasks to run ({len(tasks_to_run)}): {', '.join(tasks_to_run)}")
        print(f"Models to run ({len(models_to_run)}): {', '.join(models_to_run)}")
        
        # Create a batch timestamp to group jobs
        batch_timestamp = self.timestamp  # Use the same timestamp for all jobs in batch
        
        # Store all jobs
        all_jobs = []
        task_job_groups = {}
        
        # For each task, launch a single training instance to run all models
        for task_name in tasks_to_run:
            print(f"\n----------------------------")
            print(f"Processing task: {task_name}")
            print(f"----------------------------")
            
            # Determine number of classes for this task
            num_classes = TASK_CLASS_MAPPING.get(task_name, 2)
            print(f"Task has {num_classes} classes")
            
            # Check if S3 path exists for this task - helps with debugging data access issues
            s3_client = boto3.client('s3')
            try:
                # Extract bucket and prefix from S3 data path
                s3_parts = S3_DATA_BASE.replace('s3://', '').split('/', 1)
                if len(s3_parts) == 2:
                    bucket = s3_parts[0]
                    prefix = s3_parts[1]
                    if not prefix.endswith('/'):
                        prefix += '/'
                    
                    # Check for specific path structures for this task
                    task_paths_to_check = [
                        f"{prefix}tasks/{task_name}/",
                        f"{prefix}{task_name}/",
                        f"{prefix}Benchmark/tasks/{task_name}/"
                    ]
                    
                    print(f"Checking S3 for data paths:")
                    for task_path in task_paths_to_check:
                        try:
                            response = s3_client.list_objects_v2(
                                Bucket=bucket,
                                Prefix=task_path,
                                MaxKeys=5
                            )
                            if 'Contents' in response:
                                print(f"  ✓ Found data at s3://{bucket}/{task_path}")
                                for item in response['Contents'][:5]:
                                    print(f"    - {item['Key']}")
                            else:
                                print(f"  ✗ No data found at s3://{bucket}/{task_path}")
                        except Exception as e:
                            print(f"  ! Error checking s3://{bucket}/{task_path}: {str(e)}")
            except Exception as e:
                print(f"Error checking S3 paths: {str(e)}")
            
            # Build hyperparameters dictionary
            hyperparameters = {
                # Data parameters
                "dataset_root": S3_DATA_BASE,  # Changed from data_dir to dataset_root
                "task_name": task_name,
                "mode": mode,
                "file_format": "h5",  # Add file format parameter
                "num_workers": 4,     # Add number of workers parameter
                "data_key": 'CSI_amps',  # Add data_key parameter
                
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
                
                # Adaptive path handling - always enable in SageMaker
                "adaptive_path": "",  # Flag parameter (empty string means True)
                "try_all_paths": "",  # Flag parameter (empty string means True)
                
                # S3 upload parameters
                "save_to_s3": S3_OUTPUT_BASE,  # S3 path to save results
                "direct_upload": "",  # Flag parameter for direct S3 upload
                
                # Debug parameters
                "debug": "",  # Enable detailed logging
            }
            
            # Additional model parameters that may be needed based on local_runner.py
            if 'in_channels' in DEFAULT_CONFIG:
                hyperparameters['in_channels'] = DEFAULT_CONFIG['in_channels']
            if 'emb_dim' in DEFAULT_CONFIG:
                hyperparameters['emb_dim'] = DEFAULT_CONFIG['emb_dim']
            if 'd_model' in DEFAULT_CONFIG:
                hyperparameters['d_model'] = DEFAULT_CONFIG['d_model']
            if 'dropout' in DEFAULT_CONFIG:
                hyperparameters['dropout'] = DEFAULT_CONFIG['dropout']
            
            # Flag parameters - pass empty string for flags that should be enabled
            flag_parameters = [
                "debug",              # Enable verbose logging
                "adaptive_path",      # Auto-adapt data path
                "try_all_paths",      # Try all path combinations
                "direct_upload",      # Direct upload to S3
                "upload_final_model", # Upload final model
            ]
            
            # Set all flag parameters with empty string (indicates flag is present)
            for flag in flag_parameters:
                hyperparameters[flag] = ""
            
            # Print command line arguments for debugging
            print("\nCommand line arguments that will be passed to the script:")
            cmd_args = []
            for key, value in hyperparameters.items():
                if isinstance(value, str) and value == "":
                    # For flag parameters, only add --key without value
                    cmd_args.append(f"--{key}")
                elif not (isinstance(value, bool) and not value):  # Skip False values
                    # For parameters with values, add --key value
                    cmd_args.append(f"--{key} {value}")
            
            # Group and display parameters for readability
            print("\nFlag parameters:")
            flag_args = [arg for arg in cmd_args if "=" not in arg and " " not in arg]
            print("  " + "\n  ".join(flag_args))
            
            print("\nValue parameters:")
            value_args = [arg for arg in cmd_args if "=" in arg or " " in arg]
            print("  " + "\n  ".join(value_args))
            
            # Validate command line format
            print("\nFull command line:")
            full_cmd = "train_multi_model.py " + " ".join(cmd_args)
            print(full_cmd)
            
            # Check command line length - too long can cause issues
            if len(full_cmd) > 1000:
                print(f"\nWarning: Command line length ({len(full_cmd)}) is long and may cause issues")
            
            # Validate required parameters
            for param in ["dataset_root", "task_name", "models"]:
                if param not in hyperparameters or not hyperparameters[param]:
                    print(f"\nWarning: Required parameter {param} is missing or empty!")
            
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
            
            # 更新批处理大小，减小以降低内存使用
            if 'batch_size' in hyperparameters and int(hyperparameters['batch_size']) > 8:
                print(f"注意: 已将batch_size从{hyperparameters['batch_size']}降低至8，以减少内存使用")
                hyperparameters['batch_size'] = 8
            
            # Create PyTorch estimator
            # 使用内存更大的实例
            instance_type_to_use = instance_type or "ml.g4dn.2xlarge"  # 默认使用更大的实例
            print(f"使用实例类型: {instance_type_to_use}")
            
            # Create estimator with optimized configuration
            estimator = PyTorch(
                entry_point="entry_script.py",  # Use our custom entry script to bypass Horovod/smdebug
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
                volume_size=200,  # 增加EBS卷大小至100GB，确保有足够的磁盘空间
                debugger_hook_config=False,  # Disable debugger
                disable_profiler=True,        # Disable profiler
                code_location=None,           # 禁止将代码上传至S3特定位置
                dependencies=None,            # 不上传dependencies
                source_dir=".",               # 直接使用当前目录，不创建TAR文件
                disable_upload=True,          # 尝试完全禁用源代码上传
                environment={
                    # Code directories
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',  # Ensure code is properly located
                    'PYTHONPATH': '/opt/ml/code',                  # Add code directory to Python path
                    
                    # 指定实际要运行的脚本
                    'SAGEMAKER_PROGRAM': 'train_multi_model.py',   # 指定入口脚本
                    'SAGEMAKER_USER_ENTRY_POINT': 'entry_script.py', # 确保使用我们的入口脚本
                    
                    # Disable SageMaker debugging tools
                    'SAGEMAKER_MODEL_SERVER_WORKERS': '1',         # Limit number of model server workers
                    'SM_DISABLE_PROFILER': 'true',                 # Disable profiler via env var
                    'SM_DISABLE_DEBUGGER': 'true',                 # Disable debugger via env var
                    'SMDEBUG_DISABLED': 'true',                    # Disable SMDebug tool
                    
                    # Disable Horovod integration
                    'HOROVOD_WITH_PYTORCH': '0',                   # Disable Horovod for PyTorch
                    'HOROVOD_WITHOUT_PYTORCH': '1',                # Explicitly disable Horovod-PyTorch integration
                    'USE_HOROVOD': 'false',                        # Disable general Horovod usage
                    
                    # 内存优化设置
                    'OMP_NUM_THREADS': '1',                        # 限制OpenMP线程数
                    'MKL_NUM_THREADS': '1',                        # 限制MKL线程数
                    'CUDA_LAUNCH_BLOCKING': '1',                   # 阻塞式CUDA操作，减少内存峰值
                    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128', # 限制CUDA内存分配
                    
                    # Miscellaneous settings
                    'LOG_LEVEL': 'INFO',                           # Set logging level
                    'TORCH_CUDNN_V8_API_ENABLED': '1',             # Enable CUDNN v8 API for PyTorch 1.13+
                    
                    # Disable unnecessary features
                    'DISABLE_SMDATAPARALLEL': '1',                 # Disable SM distributed training
                }
            )
            
            # Prepare data inputs with explicit configuration
            data_channels = {
                'training': TrainingInput(
                    s3_data=S3_DATA_BASE,
                    distribution='FullyReplicated',
                    content_type='application/x-directory',
                    s3_data_type='S3Prefix',
                    input_mode='File'
                )
            }
            
            # Add debug information
            print(f"S3 data path for training: {S3_DATA_BASE}")
            print(f"Training input configuration: {data_channels['training']}")
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
            task_job_groups[task_name] = job_info
            
            # Wait between tasks
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
            'jobs': all_jobs,
            'task_groups': task_job_groups
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

    def run_multitask(self, tasks=None, model_type="transformer", instance_type=None):
        """
        Run multitask learning pipeline on SageMaker
        
        Args:
            tasks (list or str): List of tasks or comma-separated string of tasks
            model_type (str): Base model type (transformer, vit, etc.)
            instance_type (str): SageMaker instance type
            
        Returns:
            dict: Dictionary containing job details
        """
        print(f"Starting multitask learning job...")
        
        # Convert tasks to comma-separated string if it's a list
        if isinstance(tasks, list):
            tasks_str = ",".join(tasks)
        else:
            tasks_str = tasks or TASK  # Default to single task if not specified
        
        # Get task list for job naming
        task_list = tasks_str.split(',')
        print(f"Running multitask learning with tasks: {task_list}")
        print(f"Using model type: {model_type}")
        
        # Create a timestamp for this job
        job_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create job name with shortened task names
        short_tasks = "_".join([t[:5].lower() for t in task_list[:3]])
        if len(task_list) > 3:
            short_tasks += f"_plus{len(task_list)-3}"
        
        job_name = f"wifi-multitask-{short_tasks}-{model_type}-{job_timestamp[-6:]}"
        
        # Ensure job name meets SageMaker requirements
        job_name = re.sub(r'[^a-zA-Z0-9-]', '-', job_name)
        if len(job_name) > 63:
            job_name = job_name[:60] + job_timestamp[-3:]
        
        # Output path - use a structure that matches the local runner
        s3_output_path = f"{S3_OUTPUT_BASE}multitask/"
        
        print(f"Creating multitask learning job: {job_name}")
        print(f"Output path: {s3_output_path}")
        
        # Build hyperparameters
        hyperparameters = {
            # Data parameters
            "data_dir": S3_DATA_BASE,
            "tasks": tasks_str,
            "model_type": model_type,
            
            # Training parameters
            "batch_size": BATCH_SIZE,
            "epochs": EPOCH_NUMBER,
            "lr": 1e-4,
            "patience": 10,
            
            # Model parameters
            "win_len": WIN_LEN,
            "feature_size": FEATURE_SIZE,
            
            # LoRA parameters
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            
            # Save directory (will be auto-set in SageMaker to use the correct structure)
            "save_dir": "/opt/ml/model/results/multitask"
        }
        
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
                'tasks': tasks_str,
                'model_type': model_type,
                'output_dir': s3_output_path,
                'save_dir': '/opt/ml/model/results/multitask'
            }
        }
        
        print(f"Multitask learning job submitted: {job_name}")
        print(f"You can monitor the job in SageMaker console.")
        
        return job_info
    
    def run_batch_multitask(self, task_groups=None, model_type="transformer", instance_type=None, wait_time=BATCH_WAIT_TIME):
        """
        Run batch multitask training jobs
        
        Args:
            task_groups (list): List of task groups, where each group is a list of tasks
            model_type (str): Base model type (transformer, vit, etc.)
            instance_type (str): SageMaker instance type
            wait_time (int): Wait time between job submissions in seconds
            
        Returns:
            list: List of job information dictionaries
        """
        print(f"Starting batch multitask execution...")
        
        # If no task groups provided, use a default
        if not task_groups:
            # Example: Group related tasks together
            task_groups = [
                ["MotionSourceRecognition", "HumanMotion"],
                ["HumanNonhuman", "HumanID"],
                ["NTUHAR", "NTUHumanID"]
            ]
        
        # Validate task groups
        valid_task_groups = []
        for group in task_groups:
            valid_tasks = [t for t in group if t in AVAILABLE_TASKS]
            if len(valid_tasks) >= 2:  # At least 2 tasks for multitask learning
                valid_task_groups.append(valid_tasks)
            else:
                print(f"Warning: Skipping task group {group} (not enough valid tasks)")
        
        if not valid_task_groups:
            print("Error: No valid task groups found")
            return []
        
        # Create batch timestamp
        batch_timestamp = self.timestamp
        
        # Store all jobs
        all_jobs = []
        
        # For each task group, launch a training job
        for i, task_group in enumerate(valid_task_groups):
            print(f"\n----------------------------")
            print(f"Processing task group {i+1}/{len(valid_task_groups)}: {task_group}")
            print(f"----------------------------")
            
            # Run multitask job for this group
            job_info = self.run_multitask(
                tasks=task_group,
                model_type=model_type,
                instance_type=instance_type
            )
            
            # Add to job list
            job_info['batch_id'] = batch_timestamp
            job_info['group_id'] = i
            all_jobs.append(job_info)
            
            # Wait between submissions
            if wait_time > 0 and i < len(valid_task_groups) - 1:
                print(f"Waiting {wait_time} seconds before next submission...")
                try:
                    time.sleep(wait_time)
                except KeyboardInterrupt:
                    print("\nBatch submission interrupted by user.")
                    break
        
        # Create batch summary
        self._update_batch_summary(all_jobs, f"multitask_{batch_timestamp}")
        
        print(f"\nBatch multitask execution initiated!")
        print(f"Total jobs: {len(all_jobs)}")
        print(f"Batch ID: multitask_{batch_timestamp}")
        
        return all_jobs

    def test_environment_only(self, instance_type=None, wait=False):
        """
        提交一个极简的作业来仅测试环境配置，无需完整数据下载。
        这个方法使用特殊的'test_environment'标志来指示我们只进行环境测试。
        
        Args:
            instance_type (str): SageMaker实例类型，默认使用小型实例
            wait (bool): 是否等待作业完成
            
        Returns:
            dict: 包含作业详细信息的字典
        """
        print(f"启动快速环境测试作业...")
        
        # 创建此作业的时间戳
        job_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # 创建作业名称
        job_name = f"wifi-env-test-{job_timestamp[-6:]}"
        job_name = re.sub(r'[^a-zA-Z0-9-]', '-', job_name)
        
        # 输出路径
        s3_output_path = f"{S3_OUTPUT_BASE}env_tests/"
        
        print(f"创建环境测试作业: {job_name}")
        print(f"输出路径: {s3_output_path}")
        
        # 构建超参数 - 只需要极少的参数
        hyperparameters = {
            "test_only": "",  # 标记这只是一个测试作业
        }
        
        # 使用测试脚本
        test_script = "test_sagemaker_env.py"
        
        # 创建PyTorch估计器 - 使用较大内存的实例
        instance_type_to_use = instance_type or "ml.g4dn.2xlarge"
        print(f"使用实例类型: {instance_type_to_use}")
        
        estimator = PyTorch(
            entry_point="entry_script.py",  # 使用自定义入口脚本，它会禁用Horovod并处理依赖
            role=self.role,
            framework_version=FRAMEWORK_VERSION,
            py_version=PY_VERSION,
            instance_count=1,
            instance_type=instance_type_to_use,
            max_run=900,  # 最多15分钟 - 环境测试不需要更长时间
            output_path=s3_output_path,
            base_job_name=job_name,
            hyperparameters=hyperparameters,
            volume_size=50,  # 增加卷大小到50GB
            debugger_hook_config=False,  # 禁用调试器
            disable_profiler=True,        # 禁用分析器
            source_dir=".",               # 直接使用当前目录，不创建TAR文件
            code_location=None,           # 禁止将代码上传至S3特定位置
            dependencies=None,            # 不上传dependencies
            environment={
                # 代码目录
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code',
                'PYTHONPATH': '/opt/ml/code',
                
                # 指定实际要运行的脚本
                'SAGEMAKER_PROGRAM': test_script,
                
                # 禁用SageMaker调试工具
                'SM_DISABLE_PROFILER': 'true',
                'SM_DISABLE_DEBUGGER': 'true',
                'SMDEBUG_DISABLED': 'true',
                
                # 禁用Horovod集成
                'HOROVOD_WITH_PYTORCH': '0',
                'HOROVOD_WITHOUT_PYTORCH': '1',
                'USE_HOROVOD': 'false',
                
                # 内存优化设置
                'OMP_NUM_THREADS': '1',                # 限制OpenMP线程数
                'MKL_NUM_THREADS': '1',                # 限制MKL线程数
                'CUDA_LAUNCH_BLOCKING': '1',           # 阻塞式CUDA操作，减少内存峰值
                
                # 其他设置
                'LOG_LEVEL': 'INFO',
                'TORCH_CUDNN_V8_API_ENABLED': '1',
                
                # 禁用不必要的功能
                'DISABLE_SMDATAPARALLEL': '1'
            }
        )
        
        # 创建一个空的或非常小的输入数据集
        # 使用与培训任务相同的格式，但内容极少
        try:
            # 准备一个空目录作为输入
            import tempfile
            tmp_dir = tempfile.mkdtemp()
            os.makedirs(os.path.join(tmp_dir, 'tasks'), exist_ok=True)
            
            # 在临时目录中创建一个空文件以确保数据通道有效
            with open(os.path.join(tmp_dir, 'tasks', 'test_file.txt'), 'w') as f:
                f.write("This is a minimal test file to make the data channel valid.\n")
            
            # 使用Session上传到S3
            s3_prefix = f"test-env-inputs/{job_timestamp}"
            s3_test_input = self.session.upload_data(tmp_dir, 
                                                bucket=S3_DATA_BASE.split('/')[2],
                                                key_prefix=s3_prefix)
            
            print(f"已创建最小测试输入数据: {s3_test_input}")
        except Exception as e:
            print(f"创建测试输入数据时出错: {e}")
            # 如果失败，回退到标准数据位置，但仍然会避免下载大部分内容
            s3_test_input = S3_DATA_BASE
        
        # 使用我们的测试输入准备数据通道
        data_channels = {
            'training': TrainingInput(
                s3_data=s3_test_input,
                distribution='FullyReplicated',
                content_type='application/x-directory',
                s3_data_type='S3Prefix',
                input_mode='File'
            )
        }
        
        # 启动作业
        print(f"启动SageMaker环境测试作业...")
        estimator.fit(inputs=data_channels, job_name=job_name, wait=wait)
        
        if wait:
            print("作业已完成。")
            # 打印CloudWatch Logs的URL
            region = boto3.session.Session().region_name
            log_url = f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#logStream:group=/aws/sagemaker/TrainingJobs;prefix={job_name}"
            print(f"CloudWatch Logs URL: {log_url}")
        else:
            print(f"作业已提交，继续在后台运行...")
            print(f"在SageMaker控制台监控作业: {job_name}")
        
        # 返回作业信息
        job_info = {
            'job_name': job_name,
            'estimator': estimator,
            'inputs': data_channels,
            'config': {
                'task_name': 'env_test',
                'output_dir': s3_output_path
            }
        }
        
        return job_info

def main():
    """Main function to execute from command line"""
    parser = argparse.ArgumentParser(description='Run WiFi sensing pipeline on SageMaker')
    parser.add_argument('--tasks', type=str, nargs='+',
                      help='List of tasks to run. Use space to separate multiple tasks')
    parser.add_argument('--models', type=str, nargs='+',
                      help='List of models to run. Use space to separate multiple models')
    parser.add_argument('--mode', type=str, default=MODE,
                      choices=['csi', 'acf'],
                      help='Data modality to use')
    parser.add_argument('--instance-type', dest='instance_type', type=str, default=INSTANCE_TYPE,
                      help='SageMaker instance type for training')
    parser.add_argument('--batch-wait', dest='batch_wait', type=int, default=BATCH_WAIT_TIME,
                      help='Wait time between batch job submissions in seconds')
    parser.add_argument('--volume-size', dest='volume_size', type=int, default=EBS_VOLUME_SIZE,
                      help='Size of the EBS volume in GB')
    parser.add_argument('--test-args', dest='test_args', action='store_true',
                      help='Test argument parsing without running a job')
    parser.add_argument('--test-env', dest='test_env', action='store_true',
                      help='Test environment configuration by submitting a minimal job')
    parser.add_argument('--wait', dest='wait', action='store_true',
                      help='Wait for the job to complete (when using --test-env)')
    parser.add_argument('--pipeline', type=str, default='supervised',
                      choices=AVAILABLE_PIPELINES,
                      help='Pipeline to run (supervised, meta, or multitask)')
    parser.add_argument('--task-groups', dest='task_groups', type=str, nargs='+',
                      help='Task groups for multitask learning, format: "task1,task2 task3,task4"')
    parser.add_argument('--model-type', dest='model_type', type=str, default='transformer',
                      help='Base model type for multitask learning')
    
    args = parser.parse_args()
    
    # Create SageMaker runner instance
    runner = SageMakerRunner()
    
    # 如果要仅测试环境配置
    if args.test_env:
        print("正在进行快速环境测试 - 将提交最小作业，只测试环境配置")
        runner.test_environment_only(
            instance_type=args.instance_type,
            wait=args.wait
        )
        return
    
    # If using test mode, only test argument parsing
    if args.test_args:
        print("Running in test mode - will only test argument parsing")
        for task in args.tasks or AVAILABLE_TASKS[:1]:
            runner.test_hyperparameters(task_name=task, models=args.models)
        return
    
    # Handle different pipelines
    if args.pipeline == 'multitask':
        # Parse task groups if provided
        task_groups = None
        if args.task_groups:
            task_groups = [group.split(',') for group in args.task_groups]
        
        # Run multitask batch
        runner.run_batch_multitask(
            task_groups=task_groups,
            model_type=args.model_type,
            instance_type=args.instance_type,
            wait_time=args.batch_wait
        )
    else:
        # Default to supervised pipeline with batch execution
        print(f"Running batch jobs with {len(args.tasks or AVAILABLE_TASKS)} tasks and {len(args.models or AVAILABLE_MODELS)} models")
        runner.run_batch_by_task(
            tasks=args.tasks,
            models=args.models,
            mode=args.mode,
            instance_type=args.instance_type,
            wait_time=args.batch_wait
        )

if __name__ == "__main__":
    main()
