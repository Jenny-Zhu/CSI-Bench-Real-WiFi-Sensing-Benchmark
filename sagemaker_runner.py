#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - SageMaker Environment

This script allows you to run the supervised learning pipeline in the SageMaker environment.
It creates a SageMaker PyTorch Estimator for submitting training jobs.

Usage in a JupyterLab notebook:
```
import sagemaker_runner
runner = sagemaker_runner.SageMakerRunner()
runner.run_supervised()
```

For batch execution of multiple tasks and models:
```
import sagemaker_runner
runner = sagemaker_runner.SageMakerRunner()
runner.run_batch(tasks=['HumanNonhuman', 'FourClass'], models=['Transformer', 'CNN'])
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

# Default paths
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
            "pipeline": "supervised",
            "s3_data_base": "s3://rnd-sagemaker/Data/Benchmark/",
            "s3_output_base": "s3://rnd-sagemaker/Benchmark_Log/",
            "mode": "csi",
            "freeze_backbone": False,
            "integrated_loader": True,
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
            "batch_mode": "by-task",
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
            "task_test_dirs": {},
            "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit"],
            "available_tasks": ["MotionSourceRecognition", "HumanMotion", "DetectionandClassification", "HumanID", "NTUHAR", "HumanNonhuman", "NTUHumanID", "Widar", "ThreeClass", "Detection"]
        }

# Load the default configuration
DEFAULT_CONFIG = load_default_config()

# Extract configuration values
S3_DATA_BASE = DEFAULT_CONFIG.get("s3_data_base", "s3://rnd-sagemaker/Data/Benchmark/")
S3_OUTPUT_BASE = DEFAULT_CONFIG.get("s3_output_base", "s3://rnd-sagemaker/Benchmark_Log/")
AVAILABLE_TASKS = DEFAULT_CONFIG.get("available_tasks", [
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
])
AVAILABLE_MODELS = DEFAULT_CONFIG.get("available_models", ["mlp", "lstm", "resnet18", "transformer", "vit"])
TASK_TEST_DIRS = DEFAULT_CONFIG.get("task_test_dirs", {})
TASK_CLASS_MAPPING = DEFAULT_CONFIG.get("task_class_mapping", {})

# SageMaker Settings
INSTANCE_TYPE = DEFAULT_CONFIG.get("instance_type", "ml.g4dn.xlarge")
INSTANCE_COUNT = DEFAULT_CONFIG.get("instance_count", 1)
FRAMEWORK_VERSION = DEFAULT_CONFIG.get("framework_version", "1.12.1")
PY_VERSION = DEFAULT_CONFIG.get("py_version", "py38")
BASE_JOB_NAME = DEFAULT_CONFIG.get("base_job_name", "wifi-sensing-supervised")

# Data Modality
MODE = DEFAULT_CONFIG.get("mode", "csi")

# Supervised Learning Options
FREEZE_BACKBONE = DEFAULT_CONFIG.get("freeze_backbone", False)
INTEGRATED_LOADER = DEFAULT_CONFIG.get("integrated_loader", True)
TASK = DEFAULT_CONFIG.get("task", "MotionSourceRecognition")
DEFAULT_TASK = TASK

# Model Parameters
WIN_LEN = DEFAULT_CONFIG.get("win_len", 250)
FEATURE_SIZE = DEFAULT_CONFIG.get("feature_size", 98)

# Common Training Parameters
SEED = DEFAULT_CONFIG.get("seed", 42)
BATCH_SIZE = DEFAULT_CONFIG.get("batch_size", 8)
EPOCH_NUMBER = DEFAULT_CONFIG.get("num_epochs", 10)
PATIENCE = DEFAULT_CONFIG.get("patience", 15)
MODEL_NAME = DEFAULT_CONFIG.get("model_name", "transformer")

# Batch Settings
BATCH_WAIT_TIME = DEFAULT_CONFIG.get("batch_wait_time", 30)
BATCH_MODE = DEFAULT_CONFIG.get("batch_mode", "by-task")
CONFIG_FILE = None  # Path to JSON configuration file to override defaults

class SageMakerRunner:
    """Class to handle SageMaker training job creation and execution"""
    
    def __init__(self, role=None, batch_mode=BATCH_MODE):
        """Initialize SageMaker session and role"""
        self.session = sagemaker.Session()
        self.role = role or sagemaker.get_execution_role()
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.batch_mode = batch_mode
        
        # Verify the rnd-sagemaker bucket exists
        s3 = boto3.resource('s3')
        bucket_name = "rnd-sagemaker"
        if bucket_name not in [bucket.name for bucket in s3.buckets.all()]:
            print(f"Error: The bucket '{bucket_name}' does not exist. Please create it first.")
            sys.exit(1)
        
        print(f"SageMaker Runner initialized:")
        print(f"  S3 Data Base: {S3_DATA_BASE}")
        print(f"  S3 Output Base: {S3_OUTPUT_BASE}")
        print(f"  Available Tasks: {', '.join(AVAILABLE_TASKS)}")
        print(f"  Available Models: {', '.join(AVAILABLE_MODELS)}")
        print(f"  Default Batch Mode: {self.batch_mode}")
        print(f"  Timestamp: {self.timestamp}")
    
    def get_supervised_config(self, training_dir=None, test_dirs=None, output_dir=None, mode='csi', task=None, model_name=None):
        """Get configuration for supervised learning pipeline."""
        # Set default values
        current_task = task or DEFAULT_TASK
        current_model = model_name or MODEL_NAME
        
        # Set default paths if not provided, adjusting for the current task
        # Update paths to match new data structure: base_path/tasks/task_name/
        if training_dir is None:
            # Point to the base directory containing tasks folder
            # Ensure S3_DATA_BASE ends with a slash for consistency
            base_path = S3_DATA_BASE if S3_DATA_BASE.endswith('/') else f"{S3_DATA_BASE}/"
            training_dir = base_path
        
        # Test directories are not needed with the new data loading approach
        # as they are determined by the splits inside the task directory
        if test_dirs is None:
            test_dirs = []
            
        if output_dir is None:
            output_dir = f"{S3_OUTPUT_BASE}{current_task}/"
        
        # Get number of classes based on task
        num_classes = TASK_CLASS_MAPPING.get(current_task, 2)  # Default to 2 if task not found
        
        config = {
            # Data parameters - use underscores instead of dashes for parameter names
            # to match the new data loading convention
            'data_dir': training_dir,
            'task_name': current_task,
            'output_dir': output_dir,
            'results_subdir': f'supervised/{current_model}',  # Include model in results path
            
            # Training parameters
            'batch_size': BATCH_SIZE,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'num_epochs': EPOCH_NUMBER,
            'warmup_epochs': 5,
            'patience': PATIENCE,
            
            # Model parameters
            'mode': mode,
            'num_classes': num_classes,
            'freeze_backbone': FREEZE_BACKBONE,
            
            # Integrated loader options
            'integrated_loader': INTEGRATED_LOADER,
            
            # Other parameters
            'seed': SEED,
            'device': 'cuda',  # SageMaker instances will have GPU
            'model_name': current_model.lower(),  # 修改为model_name以与本地运行脚本一致
            'win_len': WIN_LEN,
            'feature_size': FEATURE_SIZE,
            'data_key': 'CSI_amps'  # Add data_key parameter
        }
        
        return config
    
    def run_supervised(self, training_dir=None, test_dirs=None, output_dir=None, mode='csi', 
                       config_file=None, instance_type=None, task=None, model_name=None):
        """Run supervised learning pipeline on SageMaker"""
        print("Preparing supervised learning pipeline...")
        
        # Use specified task or fall back to default
        current_task = task or DEFAULT_TASK
        current_model = model_name or MODEL_NAME
        
        # Get configuration with task-specific paths
        config = self.get_supervised_config(
            training_dir,
            test_dirs,
            output_dir, 
            mode or MODE,
            current_task,
            current_model
        )
        
        # Create a job name with timestamp and task
        timestamp = self.timestamp
        job_name = f"{BASE_JOB_NAME}-{current_task.lower()}-{current_model.lower()}-{timestamp}"
        job_name = re.sub(r'[^a-zA-Z0-9-]', '-', job_name)  # Replace invalid chars with hyphens
        
        # Create SageMaker Estimator
        instance_type_to_use = instance_type or INSTANCE_TYPE
        
        # Prepare data path (input)
        data_path = config.get('data_dir', S3_DATA_BASE)
        
        # Prepare path for output (results, models)
        # Organize as output_base/task/model/
        s3_output_path = config.get('output_dir', f"{S3_OUTPUT_BASE}{current_task}/{current_model}/")
        
        # Ensure all paths end with slash for consistency
        if not s3_output_path.endswith('/'):
            s3_output_path += '/'
        
        print(f"Using data path: {data_path}")
        print(f"Using output path: {s3_output_path}")
        
        # Debug: Print metrics definition
        print(f"Metrics will be captured and reported to CloudWatch.")
        
        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point="train_supervised.py",
            source_dir=".",  # Use current directory
            role=self.role,
            framework_version=FRAMEWORK_VERSION,
            py_version=PY_VERSION,
            instance_count=INSTANCE_COUNT,
            instance_type=instance_type_to_use,
            max_run=86400,  # 24 hours max runtime
            keep_alive_period_in_seconds=1800,  # 30 min keep alive after training
            output_path=s3_output_path,
            base_job_name=job_name,
            hyperparameters={
                # Data parameters
                "data_dir": data_path,
                "task_name": current_task,
                "model_type": current_model,
                
                # Training parameters
                "batch_size": config.get('batch_size', BATCH_SIZE),
                "num_epochs": config.get('num_epochs', EPOCH_NUMBER),
                "learning_rate": config.get('learning_rate', 1e-4),
                "weight_decay": config.get('weight_decay', 1e-5),
                "warmup_epochs": config.get('warmup_epochs', 5),
                "patience": config.get('patience', PATIENCE),
                
                # Model parameters
                "mode": config.get('mode', MODE),
                "win_len": config.get('win_len', WIN_LEN),
                "feature_size": config.get('feature_size', FEATURE_SIZE),
                "seed": config.get('seed', SEED),
                "save_dir": "/opt/ml/model",  # Use SageMaker model directory
                "output_dir": "/opt/ml/model"  # Set output_dir to model directory as well
            },
            metric_definitions=[
                {'Name': 'train:loss', 'Regex': 'Epoch \\d+/\\d+, Training Loss: ([0-9\\.]+)'},
                {'Name': 'train:accuracy', 'Regex': 'Epoch \\d+/\\d+, Training Accuracy: ([0-9\\.]+)'},
                {'Name': 'validation:loss', 'Regex': 'Validation Loss: ([0-9\\.]+)'},
                {'Name': 'validation:accuracy', 'Regex': 'Validation Accuracy: ([0-9\\.]+)'},
                {'Name': 'best_epoch', 'Regex': 'Best epoch: (\\d+), Best validation accuracy: ([0-9\\.]+)'}
            ]
        )
        
        # Prepare data inputs
        data_channels = {
            'training': data_path
        }
        
        # Start training job
        print("Starting SageMaker training job...")
        estimator.fit(inputs=data_channels, job_name=job_name, wait=False)
        
        print(f"SageMaker job {job_name} launched successfully!")
        print(f"You can monitor the job progress in SageMaker console.")
        print(f"The model artifacts will be saved to s3://{s3_output_path}{job_name}/output/")
        
        # Return job information
        return {
            'job_name': job_name,
            'estimator': estimator,
            'config': config,
            'timestamp': timestamp,
            'task': current_task,
            'model': current_model
        }
    
    def run_batch(self, tasks=None, models=None, mode='csi', instance_type=None, wait_time=BATCH_WAIT_TIME):
        """
        Run batch training jobs for multiple task-model combinations
        
        Args:
            tasks (list): List of tasks to run. If None, uses all available tasks.
            models (list): List of models to run. If None, uses all available models.
            mode (str): Data modality ('csi' or 'acf')
            instance_type (str): SageMaker instance type
            wait_time (int): Time to wait between job submissions in seconds
            
        Returns:
            dict: Dictionary containing details of all launched jobs
        """
        # Use default tasks and models if none provided
        if tasks is None:
            tasks = AVAILABLE_TASKS
        if models is None:
            models = AVAILABLE_MODELS
            
        # Initialize results dictionary
        jobs = {}
        
        # Get current timestamp for all jobs
        batch_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        print(f"=== Starting batch training: {len(tasks)} tasks × {len(models)} models = {len(tasks) * len(models)} jobs ===")
        print(f"Tasks: {', '.join(tasks)}")
        print(f"Models: {', '.join(models)}")
        
        # Common data directory for all tasks (with new data loading approach)
        data_dir = S3_DATA_BASE
        
        # Loop through all combinations
        for i, current_task in enumerate(tasks):
            for j, current_model in enumerate(models):
                job_index = i * len(models) + j + 1
                total_jobs = len(tasks) * len(models)
                
                print(f"\n[{job_index}/{total_jobs}] Starting job: Task={current_task}, Model={current_model}")
                
                # Create task-specific output path
                output_dir = f"{S3_OUTPUT_BASE}{current_task}/"
                
                try:
                    # Run supervised training with current task and model
                    job_details = self.run_supervised(
                        training_dir=data_dir,
                        test_dirs=None,  # No specific test dirs with new approach
                        output_dir=output_dir,
                        mode=mode,
                        instance_type=instance_type,
                        task=current_task,
                        model_name=current_model
                    )
                    
                    # Store job details
                    job_key = f"{current_task}_{current_model}"
                    jobs[job_key] = job_details
                    
                    # Write job details to a summary file
                    if job_index % 5 == 0 or job_index == total_jobs:  # Only update summary periodically
                        self._update_batch_summary(jobs, batch_timestamp)
                    
                    # Wait between job submissions to avoid throttling
                    if job_index < total_jobs:  # No need to wait after the last job
                        print(f"Waiting {wait_time}s before next job...")
                        time.sleep(wait_time)
                        
                except Exception as e:
                    print(f"Error with task={current_task}, model={current_model}: {str(e)}")
                    # Continue with next combination despite errors
        
        # Final summary update
        self._update_batch_summary(jobs, batch_timestamp)
        print(f"\n=== Batch complete: {len(jobs)}/{total_jobs} jobs launched ===")
        print(f"Summary saved to: batch_summaries/batch_summary_{batch_timestamp}.txt")
        return jobs
    
    def run_batch_by_task(self, tasks=None, models=None, mode='csi', instance_type=None, wait_time=BATCH_WAIT_TIME):
        """Run batch jobs organized by task"""
        print(f"Starting batch execution by task...")
        
        # Use provided tasks or available tasks
        if tasks is None or len(tasks) == 0:
            tasks_to_run = AVAILABLE_TASKS
        else:
            tasks_to_run = [t for t in tasks if t in AVAILABLE_TASKS]
            if len(tasks_to_run) < len(tasks):
                print(f"Warning: Some tasks requested are not in available tasks list.")
        
        # Use provided models or available models
        if models is None or len(models) == 0:
            models_to_run = AVAILABLE_MODELS
        else:
            models_to_run = [m for m in models if m in AVAILABLE_MODELS]
            if len(models_to_run) < len(models):
                print(f"Warning: Some models requested are not in available models list.")
        
        print(f"Tasks to run ({len(tasks_to_run)}): {', '.join(tasks_to_run)}")
        print(f"Models to run ({len(models_to_run)}): {', '.join(models_to_run)}")
        
        # Create a batch timestamp to group jobs
        batch_timestamp = self.timestamp  # Use the same timestamp for all jobs in batch
        
        # Store all jobs
        all_jobs = []
        task_job_groups = {}
        
        # For each task, launch training for all models
        for task_name in tasks_to_run:
            print(f"\n----------------------------")
            print(f"Processing task: {task_name}")
            print(f"----------------------------")
            
            # Determine number of classes for this task
            num_classes = TASK_CLASS_MAPPING.get(task_name, 2)
            print(f"Task has {num_classes} classes")
            
            # Store per-task jobs
            task_jobs = []
            
            # Launch training jobs for each model
            for model_name in models_to_run:
                print(f"\nLaunching job for task '{task_name}' with model '{model_name}'...")
                
                # S3 output path following task/model structure
                s3_output_path = f"{S3_OUTPUT_BASE}{task_name}/{model_name}/"
                
                if not s3_output_path.endswith('/'):
                    s3_output_path += '/'
                
                job_info = self.run_supervised(
                    task=task_name,
                    model_name=model_name,
                    output_dir=s3_output_path,
                    mode=mode,
                    instance_type=instance_type
                )
                
                # Add batch identifier
                job_info['batch_id'] = batch_timestamp
                job_info['task_group'] = task_name
                
                # Add to job lists
                task_jobs.append(job_info)
                all_jobs.append(job_info)
                
                # Wait between job submissions to space them out
                if wait_time > 0 and model_name != models_to_run[-1]:
                    print(f"Waiting {wait_time} seconds before submitting next job...")
                    try:
                        time.sleep(wait_time)
                    except KeyboardInterrupt:
                        print("\nBatch submission interrupted by user.")
                        break
            
            # Store task job group
            task_job_groups[task_name] = task_jobs
            
            # Wait longer between tasks
            if wait_time > 0 and task_name != tasks_to_run[-1]:
                print(f"Waiting {wait_time*2} seconds before starting next task...")
                try:
                    time.sleep(wait_time * 2)
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
        
        # Call update batch summary to create initial status report
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
        
        # Create both text and JSON summaries
        summary_text_file = os.path.join(summary_dir, f"batch_summary_{batch_timestamp}.txt")
        summary_json_file = os.path.join(summary_dir, f"batch_summary_{batch_timestamp}.json")
        
        # Create text summary
        with open(summary_text_file, "w") as f:
            f.write(f"Batch Training Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total Jobs: {len(jobs)}\n")
            f.write(f"Batch Timestamp: {batch_timestamp}\n\n")
            
            for job_key, job_details in jobs.items():
                f.write(f"Job: {job_key}\n")
                f.write(f"  Job Name: {job_details['job_name']}\n")
                f.write(f"  Input: {job_details['inputs']['training']}\n")
                f.write(f"  Output: {job_details['config']['output_dir']}\n")
                f.write(f"  Task: {job_details['config']['task_name']}\n")
                f.write(f"  Model: {job_details['config']['model_name']}\n")
                
                # Add information about multiple models if available
                if 'models' in job_details:
                    f.write(f"  Models: {', '.join(job_details['models'])}\n")
                    
                f.write("\n")
        
        # Create JSON summary (easier to parse programmatically)
        summary_data = {
            "timestamp": batch_timestamp,
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_jobs": len(jobs),
            "jobs": {}
        }
        
        for job_key, job_details in jobs.items():
            summary_data["jobs"][job_key] = {
                "job_name": job_details['job_name'],
                "task": job_details['config']['task_name'],
                "model": job_details['config']['model_name'],
                "input": job_details['inputs']['training'],
                "output": job_details['config']['output_dir']
            }
            
            # Add information about multiple models if available
            if 'models' in job_details:
                summary_data["jobs"][job_key]["models"] = job_details['models']
        
        with open(summary_json_file, "w") as f:
            json.dump(summary_data, f, indent=2)
    
    def run_batch_auto(self, tasks=None, models=None, mode='csi', instance_type=None, 
                     wait_time=BATCH_WAIT_TIME, batch_mode=None):
        """
        根据指定的批处理模式自动选择批处理方法
        
        Args:
            tasks (list): 要运行的任务列表。如果为None，则使用所有可用任务。
            models (list): 要运行的模型列表。如果为None，则使用所有可用模型。
            mode (str): 数据模态 ('csi' 或 'acf')
            instance_type (str): SageMaker实例类型
            wait_time (int): 作业提交之间等待的时间（秒）
            batch_mode (str): 批处理模式 ('by-task' 或 'individual')。如果为None，则使用初始化时设置的模式。
            
        Returns:
            dict: 包含所有启动作业详情的字典
        """
        # 如果未提供批处理模式，则使用实例的默认模式
        if batch_mode is None:
            batch_mode = self.batch_mode
            
        print(f"自动选择批处理模式: {batch_mode}")
        
        if batch_mode == 'by-task':
            return self.run_batch_by_task(tasks, models, mode, instance_type, wait_time)
        else:
            return self.run_batch(tasks, models, mode, instance_type, wait_time)

    def create_or_load_config(self, args):
        """
        Create a new config or load from a file.
        
        Args:
            args: Command line arguments
            
        Returns:
            Configuration dictionary
        """
        # Check if a config file is specified
        if hasattr(args, 'config_file') and args.config_file and os.path.exists(args.config_file):
            print(f"Loading configuration from {args.config_file}")
            with open(args.config_file, 'r') as f:
                config = json.load(f)
                return config
        
        # If model is specified, look for model-specific config
        if hasattr(args, 'model') and args.model:
            model_config = os.path.join(CONFIG_DIR, f"{args.model}_config.json")
            if os.path.exists(model_config):
                print(f"Using existing config file: {model_config}")
                with open(model_config, 'r') as f:
                    config = json.load(f)
                    
                # Override config with command line arguments
                if hasattr(args, 'task') and args.task:
                    config['task_name'] = args.task
                    # Update results_subdir based on model and task
                    config['results_subdir'] = f"supervised/{args.model}"
                    # Update num_classes based on task
                    config['num_classes'] = TASK_CLASS_MAPPING.get(args.task, 2)
                    
                if hasattr(args, 'num_epochs') and args.num_epochs:
                    config['num_epochs'] = args.num_epochs
                    
                if hasattr(args, 'batch_size') and args.batch_size:
                    config['batch_size'] = args.batch_size
                    
                if hasattr(args, 'output_dir') and args.output_dir:
                    config['output_dir'] = args.output_dir
                    
                return config
            
            # If task is specified, look for model+task specific config
            if hasattr(args, 'task') and args.task:
                model_task_config = os.path.join(CONFIG_DIR, f"{args.model}_{args.task.lower()}_config.json")
                if os.path.exists(model_task_config):
                    print(f"Using existing config file: {model_task_config}")
                    with open(model_task_config, 'r') as f:
                        config = json.load(f)
                        
                    # Override config with command line arguments
                    if hasattr(args, 'num_epochs') and args.num_epochs:
                        config['num_epochs'] = args.num_epochs
                        
                    if hasattr(args, 'batch_size') and args.batch_size:
                        config['batch_size'] = args.batch_size
                        
                    if hasattr(args, 'output_dir') and args.output_dir:
                        config['output_dir'] = args.output_dir
                        
                    return config
        
        # Otherwise, use the model and task arguments to get configuration
        return self.get_supervised_config(
            training_dir=args.data_dir if hasattr(args, 'data_dir') and args.data_dir else None,
            test_dirs=None,  # SageMaker不需要显式指定测试目录
            output_dir=args.output_dir if hasattr(args, 'output_dir') and args.output_dir else None,
            mode=args.mode if hasattr(args, 'mode') and args.mode else MODE,
            task=args.task if hasattr(args, 'task') and args.task else DEFAULT_TASK,
            model_name=args.model if hasattr(args, 'model') and args.model else MODEL_NAME
        )

def main():
    """Main function to execute from command line"""
    parser = argparse.ArgumentParser(description='Run WiFi sensing pipeline on SageMaker')
    parser.add_argument('--task', type=str, default=DEFAULT_TASK, choices=AVAILABLE_TASKS,
                      help=f'Task to run (default: {DEFAULT_TASK}). Available tasks: {", ".join(AVAILABLE_TASKS)}')
    parser.add_argument('--tasks', type=str, nargs='+',
                      help='List of tasks to run. Use space to separate multiple tasks')
    parser.add_argument('--models', type=str, nargs='+',
                      help='List of models to run. Use space to separate multiple models')
    parser.add_argument('--data-dir', dest='data_dir', type=str, default=None,
                      help='S3 URI containing data (default: S3_DATA_BASE)')
    parser.add_argument('--output-dir', dest='output_dir', type=str, default=None,
                      help='S3 URI to save output results (default: task-specific directory)')
    parser.add_argument('--mode', type=str, default=MODE,
                      choices=['csi', 'acf'],
                      help='Data modality to use')
    parser.add_argument('--config-file', dest='config_file', type=str, default=CONFIG_FILE,
                      help='JSON configuration file to override defaults')
    parser.add_argument('--instance-type', dest='instance_type', type=str, default=INSTANCE_TYPE,
                      help='SageMaker instance type for training')
    parser.add_argument('--model', type=str, default=MODEL_NAME, choices=AVAILABLE_MODELS,
                      help=f'Model to use (default: {MODEL_NAME})')
    parser.add_argument('--epochs', dest='num_epochs', type=int, default=None,
                      help='Number of epochs to train (default: from config)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=None,
                      help='Batch size for training (default: from config)')
    parser.add_argument('--batch-mode', dest='batch_mode', type=str, default=BATCH_MODE,
                      choices=['by-task', 'individual'],
                      help='Batch mode to use when running multiple models/tasks')
    parser.add_argument('--batch-wait', dest='batch_wait', type=int, default=BATCH_WAIT_TIME,
                      help='Wait time between batch job submissions in seconds')
    
    args = parser.parse_args()
    
    # 创建SageMaker运行器实例
    runner = SageMakerRunner(batch_mode=args.batch_mode)
    
    # 确定要使用的任务和模型
    tasks = args.tasks or [args.task] if args.task else AVAILABLE_TASKS
    models = args.models or [args.model] if args.model else AVAILABLE_MODELS
    
    # 判断是单任务单模型，还是多任务/多模型
    if len(tasks) == 1 and len(models) == 1:
        # 单任务单模型，直接运行单个作业
        print(f"Running single job: Task={tasks[0]}, Model={models[0]}")
        
        # 加载配置
        config = runner.create_or_load_config(args)
        
        # 运行单个任务
        runner.run_supervised(
            training_dir=args.data_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            config_file=args.config_file,
            instance_type=args.instance_type,
            task=tasks[0],
            model_name=models[0]
        )
    else:
        # 多任务或多模型，使用批处理
        print(f"Running batch jobs with {len(tasks)} tasks and {len(models)} models")
        runner.run_batch_auto(
            tasks=tasks,
            models=models,
            mode=args.mode,
            instance_type=args.instance_type,
            wait_time=args.batch_wait,
            batch_mode=args.batch_mode
        )

if __name__ == "__main__":
    main()
