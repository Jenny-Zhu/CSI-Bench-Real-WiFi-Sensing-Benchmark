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

#==============================================================================
# CONFIGURATION SECTION - MODIFY PARAMETERS HERE
#==============================================================================

# S3 Base Paths
S3_DATA_BASE = "s3://rnd-sagemaker/Data/Benchmark/"
S3_OUTPUT_BASE = "s3://rnd-sagemaker/Benchmark_Log/"

# Available tasks
TASKS = ['demo']

# Available models
MODELS = ['ViT']

# Map tasks to their test directories
TASK_TEST_DIRS = {
    'demo': ["test/"]
}

# SageMaker Settings
INSTANCE_TYPE = "ml.g4dn.xlarge"  # GPU instance for training
INSTANCE_COUNT = 1
FRAMEWORK_VERSION = "1.12.1"  # Match with requirements.txt
PY_VERSION = "py38"
BASE_JOB_NAME = "wifi-sensing-supervised"

# Data Modality
MODE = 'csi'  # Options: 'csi', 'acf'

# Supervised Learning Options
FREEZE_BACKBONE = False  # Freeze backbone network for supervised learning
INTEGRATED_LOADER = True  # Use integrated data loader for supervised learning
DEFAULT_TASK = TASKS[0]  # Default task for supervised learning

# Model Parameters
WIN_LEN = 250  # Window length for CSI data
FEATURE_SIZE = 98  # Feature size for CSI data

# Common Training Parameters
SEED = 42
BATCH_SIZE = 8
EPOCH_NUMBER = 1  # Number of training epochs
PATIENCE = 15  # Early stopping patience
MODEL_NAME = 'ViT'

# Advanced Configuration
CODE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory containing the code
CONFIG_FILE = None  # Path to JSON configuration file to override defaults

# Batch Settings
BATCH_WAIT_TIME = 30  # Time in seconds to wait between job submissions
BATCH_MODE = 'by-task'  # Default batch mode: 'by-task' or 'individual'

#==============================================================================
# END OF CONFIGURATION SECTION
#==============================================================================

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
        print(f"  Available Tasks: {', '.join(TASKS)}")
        print(f"  Available Models: {', '.join(MODELS)}")
        print(f"  Default Batch Mode: {self.batch_mode}")
        print(f"  Timestamp: {self.timestamp}")
    
    def get_supervised_config(self, training_dir=None, test_dirs=None, output_dir=None, mode='csi', task=None, model_name=None):
        """Get configuration for supervised learning pipeline."""
        # Set default values
        current_task = task or DEFAULT_TASK
        current_model = model_name or MODEL_NAME
        
        # Set default paths if not provided, adjusting for the current task
        if training_dir is None:
            training_dir = f"{S3_DATA_BASE}{current_task}/"
        if test_dirs is None:
            test_dirs = [f"{S3_DATA_BASE}{current_task}/{test_dir}" for test_dir in TASK_TEST_DIRS.get(current_task, ["test/"])]
        if output_dir is None:
            output_dir = f"{S3_OUTPUT_BASE}{current_task}/"
        
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
            'Detection': 2,
            'demo': 2  # 默认为2类，根据需要进行调整
        }
        
        # Get number of classes based on task
        num_classes = task_class_mapping.get(current_task, 2)  # Default to 2 if task not found
        
        config = {
            # Data parameters
            'training-dir': training_dir,
            'test-dirs': test_dirs,
            'output-dir': output_dir,
            'results-subdir': f'supervised/{current_model}',  # Include model in results path
            'train-ratio': 0.8,
            
            # Training parameters
            'batch-size': BATCH_SIZE,
            'learning-rate': 1e-4,
            'weight-decay': 1e-5,
            'num-epochs': EPOCH_NUMBER,
            'warmup-epochs': 5,
            'patience': PATIENCE,
            
            # Model parameters
            'mode': mode,
            'num-classes': num_classes,
            'freeze-backbone': FREEZE_BACKBONE,
            
            # Integrated loader options
            'integrated-loader': INTEGRATED_LOADER,
            'task': current_task,
            
            # Other parameters
            'seed': SEED,
            'device': 'cuda',  # SageMaker instances will have GPU
            'model-name': current_model,
            'win-len': WIN_LEN,
            'feature-size': FEATURE_SIZE
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
        
        # Override with values from config file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    for key, value in file_config.items():
                        config[key] = value
                print(f"Loaded configuration from {config_file}")
            except Exception as e:
                print(f"Error loading config file: {str(e)}")
        
        # Convert config to hyperparameters dict for SageMaker
        hyperparameters = {}
        for key, value in config.items():
            # Skip warmup-epochs which isn't supported
            if key == 'warmup-epochs':
                continue
                
            # Skip input channels as they'll be handled by SageMaker's input mechanism
            if key in ['training-dir', 'test-dirs']:
                continue
                
            # Handle boolean flags properly (don't include value)
            elif key == 'freeze-backbone':
                if value:
                    hyperparameters[key] = ''  # Include flag without value to set True
                # Skip if False - absence of flag means False
            elif key == 'integrated-loader':
                if value:
                    hyperparameters[key] = ''  # Include flag without value to set True
                # Skip if False - absence of flag means False
            # Handle other lists
            elif isinstance(value, list):
                hyperparameters[key] = ' '.join(str(item) for item in value)
            else:
                hyperparameters[key] = value
        
        # Create PyTorch estimator
        instance_type = instance_type or INSTANCE_TYPE
        job_name = f"{BASE_JOB_NAME}-{current_task}-{current_model}-{self.timestamp}"
        
        # Ensure requirements.txt is included
        dependencies = ["requirements.txt"]
        
        # Print information about requirements.txt
        req_path = os.path.join(CODE_DIR, "requirements.txt")
        if not os.path.exists(req_path):
            print(f"Warning: requirements.txt not found at {req_path}")
        
        estimator = PyTorch(
            entry_point="train_supervised.py",
            source_dir=CODE_DIR,
            dependencies=dependencies,
            role=self.role,
            instance_type=instance_type,
            instance_count=INSTANCE_COUNT,
            framework_version=FRAMEWORK_VERSION,
            py_version=PY_VERSION,
            hyperparameters=hyperparameters,
            output_path=config['output-dir'],
            base_job_name=job_name,
            disable_profiler=True,
            debugger_hook_config=False,
            environment={
                "HOROVOD_WITH_PYTORCH": "0", 
                "SAGEMAKER_PROGRAM": "train_supervised.py"
            }
        )
        
        # Setup input channels for SageMaker
        inputs = {
            "training": config['training-dir']
        }
        
        # Add test directories as separate channels if provided
        if config['test-dirs'] and len(config['test-dirs']) > 0:
            for i, test_dir in enumerate(config['test-dirs']):
                channel_name = f"test{i+1}" if i > 0 else "test"
                inputs[channel_name] = test_dir
        
        # Print configuration
        print(f"Job: {job_name}")
        print(f"  Task: {current_task}")
        print(f"  Model: {current_model}")
        print(f"  Instance: {instance_type}")
        print(f"  Training path: {inputs['training']}")
        
        # Launch training job
        print("Launching SageMaker job...")
        estimator.fit(inputs, wait=False)
        
        print(f"Job '{job_name}' launched successfully.")
        
        # Return job details
        return {
            'job_name': job_name,
            'estimator': estimator,
            'config': config,
            'inputs': inputs
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
            tasks = TASKS
        if models is None:
            models = MODELS
            
        # Initialize results dictionary
        jobs = {}
        
        # Get current timestamp for all jobs
        batch_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        print(f"=== Starting batch training: {len(tasks)} tasks × {len(models)} models = {len(tasks) * len(models)} jobs ===")
        print(f"Tasks: {', '.join(tasks)}")
        print(f"Models: {', '.join(models)}")
        
        # Loop through all combinations
        for i, current_task in enumerate(tasks):
            for j, current_model in enumerate(models):
                job_index = i * len(models) + j + 1
                total_jobs = len(tasks) * len(models)
                
                print(f"\n[{job_index}/{total_jobs}] Starting job: Task={current_task}, Model={current_model}")
                
                # Create task-specific paths
                training_dir = f"{S3_DATA_BASE}{current_task}/"
                test_dirs = [f"{S3_DATA_BASE}{current_task}/{test_dir}" for test_dir in TASK_TEST_DIRS.get(current_task, ["test/"])]
                output_dir = f"{S3_OUTPUT_BASE}{current_task}/"
                
                try:
                    # Run supervised training with current task and model
                    job_details = self.run_supervised(
                        training_dir=training_dir,
                        test_dirs=test_dirs,
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
        """
        Run batch training jobs grouped by task - one job per task that trains all models
        
        这种方式每个任务只创建一个训练作业，在一个作业中训练多个模型，
        减少了作业数量，使管理更加简单
        
        Args:
            tasks (list): List of tasks to run. If None, uses all available tasks.
            models (list): List of models to run. If None, uses all available models.
            mode (str): Data modality ('csi' or 'acf')
            instance_type (str): SageMaker instance type
            wait_time (int): Time to wait between job submissions in seconds
            
        Returns:
            dict: Dictionary containing details of all launched jobs
        """
        # 使用默认任务和模型（如果未提供）
        if tasks is None:
            tasks = TASKS
        if models is None:
            models = MODELS
            
        # 初始化结果字典
        jobs = {}
        
        # 所有作业使用相同的时间戳
        batch_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        print(f"=== Starting batch training by task: {len(tasks)} tasks (training {len(models)} models per task) ===")
        print(f"Tasks: {', '.join(tasks)}")
        print(f"Models: {', '.join(models)}")
        
        # 循环遍历每个任务
        for i, current_task in enumerate(tasks):
            job_index = i + 1
            
            print(f"\n[{job_index}/{len(tasks)}] Starting job: Task={current_task} (all models)")
            
            # 创建特定于任务的路径
            training_dir = f"{S3_DATA_BASE}{current_task}/"
            test_dirs = [f"{S3_DATA_BASE}{current_task}/{test_dir}" for test_dir in TASK_TEST_DIRS.get(current_task, ["test/"])]
            output_dir = f"{S3_OUTPUT_BASE}{current_task}/"
            
            try:
                # 使用自定义脚本运行所有模型训练
                # 这里我们创建一个额外的参数来指定要训练的所有模型
                config = self.get_supervised_config(
                    training_dir=training_dir,
                    test_dirs=test_dirs,
                    output_dir=output_dir,
                    mode=mode,
                    task=current_task,
                    # 使用第一个模型作为默认，其他模型将在训练脚本中处理
                    model_name=models[0] if models else MODEL_NAME
                )
                
                # 添加一个特殊参数，指定要训练的所有模型
                config['all-models'] = models
                
                # 转换配置为超参数字典
                hyperparameters = {}
                for key, value in config.items():
                    # 跳过warmup-epochs（不支持）
                    if key == 'warmup-epochs':
                        continue
                        
                    # 跳过输入通道（将由SageMaker的输入机制处理）
                    if key in ['training-dir', 'test-dirs']:
                        continue
                        
                    # 正确处理布尔标志（不包含值）
                    elif key == 'freeze-backbone':
                        if value:
                            hyperparameters[key] = ''  # 不带值包含标志表示True
                        # 如果为False则跳过 - 标志不存在表示False
                    elif key == 'integrated-loader':
                        if value:
                            hyperparameters[key] = ''  # 不带值包含标志表示True
                        # 如果为False则跳过 - 标志不存在表示False
                    # 处理所有模型列表
                    elif key == 'all-models':
                        hyperparameters[key] = ' '.join(value)
                    # 处理其他列表
                    elif isinstance(value, list):
                        hyperparameters[key] = ' '.join(str(item) for item in value)
                    else:
                        hyperparameters[key] = value
                
                # 创建PyTorch估计器
                instance_type = instance_type or INSTANCE_TYPE
                job_name = f"{BASE_JOB_NAME}-{current_task}-all-models-{self.timestamp}"
                
                # 确保包含requirements.txt
                dependencies = ["requirements.txt"]
                
                # 创建估计器
                estimator = PyTorch(
                    entry_point="train_multi_model.py",  # 需要一个新的训练脚本处理多个模型
                    source_dir=CODE_DIR,
                    dependencies=dependencies,
                    role=self.role,
                    instance_type=instance_type,
                    instance_count=INSTANCE_COUNT,
                    framework_version=FRAMEWORK_VERSION,
                    py_version=PY_VERSION,
                    hyperparameters=hyperparameters,
                    output_path=config['output-dir'],
                    base_job_name=job_name,
                    disable_profiler=True,
                    debugger_hook_config=False,
                    environment={
                        "HOROVOD_WITH_PYTORCH": "0", 
                        "SAGEMAKER_PROGRAM": "train_multi_model.py"
                    }
                )
                
                # 设置SageMaker的输入通道
                inputs = {
                    "training": config['training-dir']
                }
                
                # 如果提供了测试目录，则将其添加为单独的通道
                if config['test-dirs'] and len(config['test-dirs']) > 0:
                    for i, test_dir in enumerate(config['test-dirs']):
                        channel_name = f"test{i+1}" if i > 0 else "test"
                        inputs[channel_name] = test_dir
                
                # 打印配置
                print(f"Job: {job_name}")
                print(f"  Task: {current_task}")
                print(f"  Models: {', '.join(models)}")
                print(f"  Instance: {instance_type}")
                print(f"  Training path: {inputs['training']}")
                
                # 启动训练作业
                print("Launching SageMaker job...")
                estimator.fit(inputs, wait=False)
                
                print(f"Job '{job_name}' launched successfully.")
                
                # 存储作业详情
                job_key = f"{current_task}_all_models"
                jobs[job_key] = {
                    'job_name': job_name,
                    'estimator': estimator,
                    'config': config,
                    'inputs': inputs,
                    'models': models
                }
                
                # 写入作业详情到摘要文件
                self._update_batch_summary(jobs, batch_timestamp)
                
                # 在作业提交之间等待以避免限流
                if job_index < len(tasks):  # 最后一个作业后不需要等待
                    print(f"Waiting {wait_time}s before next job...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                print(f"Error with task={current_task}: {str(e)}")
                # 尽管有错误，仍继续下一个组合
        
        print(f"\n=== Batch complete: {len(jobs)}/{len(tasks)} jobs launched ===")
        print(f"Summary saved to: batch_summaries/batch_summary_{batch_timestamp}.txt")
        return jobs
    
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
                f.write(f"  Output: {job_details['config']['output-dir']}\n")
                f.write(f"  Task: {job_details['config']['task']}\n")
                f.write(f"  Model: {job_details['config']['model-name']}\n")
                
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
                "task": job_details['config']['task'],
                "model": job_details['config']['model-name'],
                "input": job_details['inputs']['training'],
                "output": job_details['config']['output-dir']
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

def main():
    """Main function to execute from command line"""
    parser = argparse.ArgumentParser(description='Run WiFi sensing pipeline on SageMaker')
    parser.add_argument('--task', type=str, default=DEFAULT_TASK, choices=TASKS,
                      help=f'Task to run (default: {DEFAULT_TASK}). Available tasks: {", ".join(TASKS)}')
    parser.add_argument('--training-dir', type=str, default=None,
                      help='S3 URI containing training data (default: task-specific directory)')
    parser.add_argument('--test-dirs', type=str, nargs='+', default=None,
                      help='List of S3 URIs containing test data. Can specify multiple paths')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='S3 URI to save output results (default: task-specific directory)')
    parser.add_argument('--mode', type=str, default=MODE,
                      choices=['csi', 'acf'],
                      help='Data modality to use')
    parser.add_argument('--config-file', type=str, default=CONFIG_FILE,
                      help='JSON configuration file to override defaults')
    parser.add_argument('--instance-type', type=str, default=INSTANCE_TYPE,
                      help='SageMaker instance type for training')
    parser.add_argument('--model', type=str, default=MODEL_NAME, choices=MODELS,
                      help=f'Model to use (default: {MODEL_NAME})')
    parser.add_argument('--batch', action='store_true',
                      help='Run batch training for multiple tasks and models')
    parser.add_argument('--batch-by-task', action='store_true',
                      help='Run batch training grouped by task (one job per task, all models)')
    parser.add_argument('--batch-mode', type=str, choices=['by-task', 'individual'], default=BATCH_MODE,
                      help=f'Batch mode to use when --batch is specified (default: {BATCH_MODE})')
    parser.add_argument('--batch-tasks', type=str, nargs='+',
                      help='List of tasks to run in batch mode. Use space to separate multiple tasks')
    parser.add_argument('--batch-models', type=str, nargs='+',
                      help='List of models to run in batch mode. Use space to separate multiple models')
    parser.add_argument('--wait-time', type=int, default=BATCH_WAIT_TIME,
                      help='Time to wait between job submissions in seconds')
    
    args = parser.parse_args()
    
    # 创建runner实例，传递batch_mode参数
    runner = SageMakerRunner(batch_mode=args.batch_mode)
    
    # 处理批量训练逻辑
    if args.batch or args.batch_by_task:
        tasks = args.batch_tasks or TASKS
        models = args.batch_models or MODELS
        
        # 确定批处理模式
        # 如果显式指定了batch-by-task，则使用按任务分组模式
        # 如果只指定了batch，则使用配置的默认模式
        batch_mode = 'by-task' if args.batch_by_task else args.batch_mode
        
        print(f"Using batch mode: {batch_mode}")
        runner.run_batch_auto(
            tasks=tasks,
            models=models,
            mode=args.mode,
            instance_type=args.instance_type,
            wait_time=args.wait_time,
            batch_mode=batch_mode
        )
    else:
        # 运行单个训练作业
        runner.run_supervised(
            args.training_dir, 
            args.test_dirs, 
            args.output_dir, 
            args.mode, 
            args.config_file,
            args.instance_type,
            args.task,
            args.model
        )

if __name__ == "__main__":
    main()
