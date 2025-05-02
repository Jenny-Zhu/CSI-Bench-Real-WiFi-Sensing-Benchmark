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
            "num_epochs": 10,
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
            "available_tasks": ["MotionSourceRecognition", "HumanMotion", "DetectionandClassification", "HumanID", "NTUHAR", "HumanNonhuman", "NTUHumanID", "Widar", "ThreeClass", "Detection"]
        }

# Load the default configuration
DEFAULT_CONFIG = load_default_config()

# Extract configuration values
S3_DATA_BASE = DEFAULT_CONFIG.get("s3_data_base", "s3://rnd-sagemaker/Data/Benchmark/")
S3_OUTPUT_BASE = DEFAULT_CONFIG.get("s3_output_base", "s3://rnd-sagemaker/Benchmark_Log/")
AVAILABLE_TASKS = DEFAULT_CONFIG.get("available_tasks", [
    "MotionSourceRecognition", "HumanMotion", "DetectionandClassification", 
    "HumanID", "NTUHAR", "HumanNonhuman", "NTUHumanID", "Widar", "ThreeClass", "Detection"
])
AVAILABLE_MODELS = DEFAULT_CONFIG.get("available_models", ["mlp", "lstm", "resnet18", "transformer", "vit"])
TASK_CLASS_MAPPING = DEFAULT_CONFIG.get("task_class_mapping", {})

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
EPOCH_NUMBER = DEFAULT_CONFIG.get("num_epochs", 10)
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
                "num_epochs": EPOCH_NUMBER,
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
                
                # 调试参数 - 启用详细日志记录
                "debug": "",  # 标志类型参数，不需要传递值
                
                # 添加数据迁移选项，针对SageMaker环境处理
                "adaptive_path": "",  # 标志参数，允许自动适应数据路径
                "try_all_paths": ""   # 尝试多种路径组合
            }
            
            # 从配置文件添加额外参数（如果有）
            # 仅添加在train_multi_model.py中定义的参数
            allowed_params = [
                "dataset_root", "task_name", "mode", "file_format", "num_workers",
                "models", "batch_size", "num_epochs", "learning_rate", "weight_decay",
                "warmup_epochs", "patience", "win_len", "feature_size", "seed",
                "save_dir", "output_dir", "data_key", "debug",
                "in_channels", "emb_dim", "d_model", "dropout"
            ]
            
            for key, value in DEFAULT_CONFIG.items():
                if key in allowed_params and key not in hyperparameters:
                    hyperparameters[key] = value
            
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
            
            estimator = PyTorch(
                entry_point="train_multi_model.py",  # Note: This uses a new training script
                source_dir=".",
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
                volume_size=EBS_VOLUME_SIZE  # Increase EBS volume size
            )
            
            # Prepare data inputs with more explicit configuration
            data_channels = {
                'training': TrainingInput(
                    s3_data_type='S3Prefix',
                    s3_data=S3_DATA_BASE,
                    distribution='FullyReplicated',
                    content_type='application/x-directory',
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
            task_job_groups[task_name] = job_info
            
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
                f.write(f"  Input: {job_info['inputs']['training'].s3_data}\n")
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
                "input": str(job_info['inputs']['training'].s3_data),
                "output": job_info['config']['output_dir']
            }
            
            # Add information about multiple models if available
            if 'models' in job_info:
                summary_data["jobs"][job_key]["models"] = job_info['models']
        
        # Ensure all data is JSON serializable
        summary_data = self.convert_to_json_serializable(summary_data)
        
        with open(summary_json_file, "w") as f:
            json.dump(summary_data, f, indent=2)

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
    
    args = parser.parse_args()
    
    # Create SageMaker runner instance
    runner = SageMakerRunner()
    
    # Determine tasks and models to use
    tasks = args.tasks or AVAILABLE_TASKS
    models = args.models or AVAILABLE_MODELS
    
    # Start batch execution
    print(f"Running batch jobs with {len(tasks)} tasks and {len(models)} models")
    runner.run_batch_by_task(
        tasks=tasks,
        models=models,
        mode=args.mode,
        instance_type=args.instance_type,
        wait_time=args.batch_wait
    )

if __name__ == "__main__":
    main()
