#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - SageMaker Environment

This script allows you to run the supervised learning pipeline in the SageMaker environment.
It creates a SageMaker PyTorch Estimator for submitting training jobs.

Key features:
1. Batch execution of training tasks, with each task using a single instance to run multiple models
2. Support for configuration from JSON files

Usage example:
```
python sagemaker_runner.py --config configs/my_custom_config.json
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

def load_config(config_path=None):
    """Load configuration from JSON file"""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Process few-shot config to ensure backward compatibility
        if 'fewshot' in config:
            fewshot_config = config['fewshot']
            # Set legacy parameters for compatibility
            config['enable_few_shot'] = fewshot_config.get('enabled', False)
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

# Load the default configuration
DEFAULT_CONFIG = load_config()

class SageMakerRunner:
    """Class to handle SageMaker training job creation and execution"""
    
    def __init__(self, config):
        """Initialize SageMaker session and role"""
        self.session = sagemaker.Session()
        self.role = sagemaker.get_execution_role()
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M")  # Shorter format for job names
        
        # Load configuration
        self.config = config
        
        # Extract common parameters from config
        self.s3_data_base = self.config.get("s3_data_base")
        self.s3_output_base = self.config.get("s3_output_base")
        
        # Verify the S3 bucket exists
        self._verify_s3_bucket()
        
        print(f"SageMaker Runner initialized:")
        print(f"  S3 Data Base: {self.s3_data_base}")
        print(f"  S3 Output Base: {self.s3_output_base}")
        print(f"  Timestamp: {self.timestamp}")
    
    def _verify_s3_bucket(self):
        """Verify that the S3 bucket exists and list available data"""
        try:
            s3 = boto3.resource('s3')
            bucket_name = self.s3_data_base.split('/')[2]  # Extract bucket name from S3 path
            
            # Check if bucket exists
            if bucket_name not in [bucket.name for bucket in s3.buckets.all()]:
                print(f"Error: The bucket '{bucket_name}' does not exist. Please create it first.")
                sys.exit(1)
        
            # Check contents of S3 path
            s3_client = boto3.client('s3')
            bucket = self.s3_data_base.split('/')[2]
            prefix = '/'.join(self.s3_data_base.split('/')[3:])
            if not prefix.endswith('/'):
                prefix += '/'
        
            # Try to list S3 path contents
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
            
            if 'CommonPrefixes' in response:
                print(f"Contents of S3 path {self.s3_data_base}:")
                for obj in response['CommonPrefixes']:
                    folder = obj['Prefix'].split('/')[-2]
                    print(f"  - {folder}/")
                    
                # Check if tasks directory exists
                tasks_prefix = prefix + 'tasks/'
                tasks_resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=tasks_prefix, Delimiter='/')
                
                if 'CommonPrefixes' in tasks_resp:
                    print(f"Tasks available in {self.s3_data_base}tasks/:")
                    for obj in tasks_resp['CommonPrefixes']:
                        task_name = obj['Prefix'].split('/')[-2]
                        print(f"  - {task_name}/")
                else:
                    print(f"Warning: No tasks found in {self.s3_data_base}tasks/")
            else:
                print(f"Warning: S3 path {self.s3_data_base} appears to be empty")
        except Exception as e:
            print(f"Warning: Error checking S3 path {self.s3_data_base}: {e}")
    
    def run_batch_by_task(self, tasks=None, models=None, override_config=None):
        """
        Run batch processing by task, with each task executed on a single instance.
        
        Args:
            tasks (list): List of tasks to process (defaults to all available tasks)
            models (list): List of models to run for each task (defaults to all available models)
            override_config (dict): Configuration override parameters
            
        Returns:
            list: List of job information dictionaries
        """
        print(f"Starting batch execution by task...")
        
        # Create batch timestamp for job naming
        batch_timestamp = self.timestamp
        
        # Set defaults if not provided
        tasks_to_run = tasks or self.config.get('available_tasks')
        models_to_run = models or self.config.get('available_models')
        
        # Convert to lists if strings are provided
        if isinstance(tasks_to_run, str):
            tasks_to_run = [tasks_to_run]
        if isinstance(models_to_run, str):
            models_to_run = [models_to_run]
        
        # Prepare configuration
        base_config = dict(self.config)
        if override_config:
            base_config.update(override_config)
        
        # List to track job information
        jobs = []
        
        # Iterate through tasks
        for task in tasks_to_run:
            print(f"\nProcessing task: {task}")
            
            # Create task-specific configuration
            task_config = dict(base_config)
            task_config['task'] = task
            
            # Create an estimator that will run all models for this task
            estimator = self._create_estimator(
                task_config,
                base_job_name=f"{task_config.get('base_job_name', 'wifi-sensing')}-{task}",
                models=models_to_run
            )
            
            # Start training job
            job_name = f"{task}-{batch_timestamp}"
            estimator.fit(
                inputs=self._prepare_inputs(task_config),
                job_name=job_name,
                wait=False
            )
            
            # Track job info
            jobs.append({
                'job_name': job_name,
                'task': task,
                'models': models_to_run,
                'status': 'InProgress',
                'estimator': estimator
            })
            
            print(f"Submitted job: {job_name}")
            
            # Wait between jobs if specified
            if len(tasks_to_run) > 1 and task != tasks_to_run[-1]:
                wait_time = task_config.get('batch_wait_time', 30)
                print(f"Waiting {wait_time} seconds before submitting next job...")
                time.sleep(wait_time)
        
        return jobs
    
    def _create_estimator(self, config, base_job_name, models):
        """
        Create a SageMaker PyTorch Estimator from the given configuration
        """
        # Prepare hyperparameters
        hyperparameters = {
            'models': ','.join(models),
            'task_name': config.get('task'),
            'win_len': config.get('win_len', 500),
            'feature_size': config.get('feature_size', 232),
            'batch_size': config.get('batch_size', 16),
            'epochs': config.get('epochs', 100),
            'test_splits': config.get('test_splits', 'all'),
            'seed': config.get('seed', 42)
        }
        
        # Add few-shot parameters if enabled
        if config.get('enable_few_shot', False) or config.get('fewshot_eval_shots', False):
            hyperparameters['enable_few_shot'] = "True"
            hyperparameters['k_shot'] = config.get('k_shot', 5)
            hyperparameters['inner_lr'] = config.get('inner_lr', 0.01)
            hyperparameters['num_inner_steps'] = config.get('num_inner_steps', 10)
            hyperparameters['fewshot_support_split'] = config.get('fewshot_support_split', 'val_id')
            hyperparameters['fewshot_query_split'] = config.get('fewshot_query_split', 'test_cross_env')
            hyperparameters['fewshot_finetune_all'] = "True" if config.get('fewshot_finetune_all', False) else "False"
            hyperparameters['fewshot_eval_shots'] = "True" if config.get('fewshot_eval_shots', False) else "False"
        
        # Create estimator
        estimator = PyTorch(
            entry_point='entry_script.py',
            source_dir=CODE_DIR,
            role=self.role,
            framework_version=config.get('framework_version', '1.12.1'),
            py_version=config.get('py_version', 'py38'),
            instance_count=config.get('instance_count', 1),
            instance_type=config.get('instance_type', 'ml.g4dn.2xlarge'),
            base_job_name=base_job_name,
            hyperparameters=hyperparameters,
            max_run=172800  # 48 hours max run time
        )
        
        return estimator
    
    def _prepare_inputs(self, config):
        """
        Prepare input data channels for training
        确保只使用特定任务的数据路径
        """
        task = config.get('task')
        s3_data_base = config.get('s3_data_base')
        
        # 构建特定任务的数据路径
        # 确保路径末尾有斜杠
        if not s3_data_base.endswith('/'):
            s3_data_base += '/'
        
        # 构建特定任务的数据路径
        task_data_path = f"{s3_data_base}tasks/{task}/"
        
        print(f"使用任务特定的数据路径: {task_data_path}")
        
        # 定义输入通道
        input_data = {
            'training': TrainingInput(
                s3_data=task_data_path,
                content_type='application/x-recordio',
                s3_data_type='S3Prefix'
            )
        }
        
        return input_data
    
    def run_multitask(self, tasks=None, model_type="transformer", override_config=None):
        """
        Run a multitask learning job
        """
        print(f"Starting multitask learning job...")
        
        # Create task-specific configuration
        multi_config = dict(self.config)
        if override_config:
            multi_config.update(override_config)
        
        # Set model and tasks
        multi_config['model'] = model_type
        
        # Make sure tasks is provided and properly formatted
        if tasks:
            # Convert to list if string is provided
            if isinstance(tasks, str):
                tasks = tasks.split(',')
            multi_config['tasks'] = ','.join(tasks)
        elif 'tasks' not in multi_config:
            # Default to first two tasks if not specified
            available_tasks = multi_config.get('available_tasks', [])
            if len(available_tasks) >= 2:
                multi_config['tasks'] = ','.join(available_tasks[:2])
            else:
                print("Error: At least two tasks are required for multitask learning")
                return None
        
        # Create an estimator for multitask learning
        estimator = self._create_multitask_estimator(multi_config)
        
        # Start training job
        job_name = f"multitask-{self.timestamp}"
        estimator.fit(
            inputs=self._prepare_multitask_inputs(multi_config),
            job_name=job_name,
            wait=False
        )
        
        # Return job info
        job_info = {
            'job_name': job_name,
            'tasks': multi_config['tasks'],
            'model': model_type,
            'status': 'InProgress',
            'estimator': estimator
        }
        
        print(f"Submitted multitask job: {job_name}")
        return job_info
    
    def _create_multitask_estimator(self, config):
        """
        Create a SageMaker PyTorch Estimator for multitask learning
        """
        # Prepare hyperparameters
        hyperparameters = {
            'pipeline': 'multitask',
            'tasks': config.get('tasks'),
            'model': config.get('model', 'transformer'),
            'win_len': config.get('win_len', 500),
            'feature_size': config.get('feature_size', 232),
            'batch_size': config.get('batch_size', 16),
            'epochs': config.get('epochs', 100),
            'test_splits': config.get('test_splits', 'all'),
            'seed': config.get('seed', 42)
        }
        
        # Add few-shot parameters if enabled
        if config.get('enable_few_shot', False) or config.get('fewshot_eval_shots', False):
            hyperparameters['enable_few_shot'] = "True"
            hyperparameters['k_shot'] = config.get('k_shot', 5)
            hyperparameters['inner_lr'] = config.get('inner_lr', 0.01)
            hyperparameters['num_inner_steps'] = config.get('num_inner_steps', 10)
        
        # Add model-specific parameters if present
        if 'model_params' in config:
            for key, value in config['model_params'].items():
                # 将参数名称中的破折号替换为下划线
                fixed_key = key.replace('-', '_')
                hyperparameters[fixed_key] = value
        else:
            # Add common parameters
            for param in ['lr', 'emb_dim', 'dropout', 'patience']:
                if param in config:
                    hyperparameters[param] = config[param]
        
        # Create estimator
        estimator = PyTorch(
            entry_point='entry_script.py',
            source_dir=CODE_DIR,
            role=self.role,
            framework_version=config.get('framework_version', '1.12.1'),
            py_version=config.get('py_version', 'py38'),
            instance_count=config.get('instance_count', 1),
            instance_type=config.get('instance_type', 'ml.g4dn.2xlarge'),
            base_job_name='wifi-sensing-multitask',
            hyperparameters=hyperparameters,
            max_run=172800  # 48 hours max run time
        )
        
        return estimator
    
    def _prepare_multitask_inputs(self, config):
        """
        Prepare input data channels for multitask training
        确保只下载需要的任务数据
        """
        s3_data_base = config.get('s3_data_base')
        
        # 确保路径末尾有斜杠
        if not s3_data_base.endswith('/'):
            s3_data_base += '/'
        
        # 获取任务列表
        tasks_str = config.get('tasks', '')
        tasks_list = [t.strip() for t in tasks_str.split(',') if t.strip()]
        
        if not tasks_list:
            print("警告: 未指定任务，将使用整个数据目录")
            data_path = s3_data_base
        else:
            # 为多任务学习准备输入数据
            # 只使用指定任务的数据路径
            data_paths = []
            for task in tasks_list:
                task_path = f"{s3_data_base}tasks/{task}/"
                data_paths.append(task_path)
            
            # 如果只有一个任务，直接使用该任务的路径
            if len(data_paths) == 1:
                data_path = data_paths[0]
                print(f"多任务学习使用单一任务数据路径: {data_path}")
            else:
                # 如果有多个任务，则需要通过 manifest 文件或其他方式组合
                # 但当前 SageMaker 实现中，我们只能指定一个路径，所以使用父目录
                data_path = f"{s3_data_base}tasks/"
                print(f"多任务学习使用多个任务({len(tasks_list)}个)，数据路径: {data_path}")
                print(f"任务列表: {', '.join(tasks_list)}")
        
        # 定义输入通道
        input_data = {
            'training': TrainingInput(
                s3_data=data_path,
                content_type='application/x-recordio',
                s3_data_type='S3Prefix'
            )
        }
        
        return input_data

    def run_test_env(self):
        """
        运行一个快速的环境测试作业，用于验证依赖项和环境配置
        不下载完整数据集，只进行最小化的环境验证
        
        Returns:
            dict: 包含作业信息的字典
        """
        print(f"正在启动环境测试作业...")
        
        # 创建一个特化的测试配置
        test_config = dict(self.config)
        test_config['epochs'] = 1  # 设置为1轮
        test_config['batch_size'] = 2  # 使用小批量
        
        # 创建一个特殊的环境测试脚本参数
        hyperparameters = {
            'test_env': 'True',  # 告诉入口脚本这是环境测试
            'batch_size': 2,
            'epochs': 1,
            'seed': 42,
            'win_len': 10,  # 使用很小的窗口大小
            'feature_size': 10  # 使用很小的特征大小
        }
        
        # 创建测试环境的估算器
        estimator = PyTorch(
            entry_point='entry_script.py',
            source_dir=CODE_DIR,
            role=self.role,
            framework_version=test_config.get('framework_version', '1.12.1'),
            py_version=test_config.get('py_version', 'py38'),
            instance_count=1,
            instance_type=test_config.get('instance_type', 'ml.g4dn.2xlarge'),
            base_job_name='wifi-sensing-env-test',
            hyperparameters=hyperparameters,
            max_run=3600  # 最多运行1小时
        )
        
        # 准备一个最小的输入数据配置
        # SageMaker要求至少有一个输入数据通道，即使是在测试环境中
        minimal_inputs = {
            'training': TrainingInput(
                s3_data=f"{self.s3_data_base}/",
                content_type='application/x-recordio',
                s3_data_type='S3Prefix',
                input_mode='File'
            )
        }
        
        print(f"使用最小数据配置用于环境测试，路径: {self.s3_data_base}/")
        
        # 启动训练作业
        job_name = f"env-test-{self.timestamp}"
        estimator.fit(
            inputs=minimal_inputs,  # 提供最小的输入数据配置
            job_name=job_name,
            wait=False
        )
        
        # 返回作业信息
        job_info = {
            'job_name': job_name,
            'type': 'environment_test',
            'status': 'InProgress',
            'estimator': estimator
        }
        
        print(f"提交环境测试作业: {job_name}")
        print(f"此作业不会下载完整数据集，仅用于验证环境设置和依赖项")
        return job_info

def run_from_config(config_path=None):
    """
    运行SageMaker训练任务，基于配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
    """
    # 加载配置
    config = load_config(config_path)
    
    # 创建SageMaker运行器
    runner = SageMakerRunner(config)
    
    # 从配置中获取任务和模型
    tasks = config.get('task')
    models = config.get('model')
    pipeline = config.get('pipeline', 'supervised')
    
    # 根据配置的pipeline类型运行相应的任务
    if pipeline == 'multitask':
        # 对于多任务学习，需要确保有tasks参数
        if 'tasks' in config:
            tasks = config['tasks']
        else:
            # 如果未指定tasks，但指定了task，则使用task
            if tasks:
                tasks = [tasks]
            else:
                # 使用默认任务列表中的前两个任务
                tasks = config.get('available_tasks', [])[:2]
        
        # 运行多任务学习
        result = runner.run_multitask(tasks=tasks, model_type=models)
    else:
        # 对于监督学习，可以运行单个任务或多个任务
        if tasks and not isinstance(tasks, list):
            tasks = [tasks]
        
        # 如果指定了模型，确保是列表格式
        if models and not isinstance(models, list):
            if ',' in models:
                models = models.split(',')
            else:
                models = [models]
        
        # 运行批处理任务
        result = runner.run_batch_by_task(tasks=tasks, models=models)
    
    return result

def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description='Run SageMaker WiFi Sensing pipeline')
    
    # 只保留配置文件参数并添加测试环境选项
    parser.add_argument('--config', type=str, default=None,
                        help='JSON configuration file path')
    parser.add_argument('--test-env', action='store_true',
                        help='只测试环境配置和依赖项，而不进行完整训练')
    
    args = parser.parse_args()
    
    if args.test_env:
        # 运行环境测试作业
        config = load_config(args.config)
        runner = SageMakerRunner(config)
        runner.run_test_env()
    else:
        # 运行从配置文件加载的正常作业
        run_from_config(args.config)

if __name__ == "__main__":
    main()
