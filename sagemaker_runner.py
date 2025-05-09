#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - SageMaker Environment

This script enables running supervised learning and multi-task learning pipelines in a SageMaker environment.
It creates a SageMaker PyTorch Estimator to submit training jobs.

Main features:
1. Batch execution of training jobs, with each job running multiple models on a single instance
2. Support for multi-task learning and few-shot learning
3. Support for configuring parameters from JSON files

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
            
        # Process few-shot configuration to ensure backward compatibility
        if 'fewshot' in config:
            fewshot_config = config['fewshot']
            # Set compatibility parameters
            config['enable_few_shot'] = fewshot_config.get('enabled', False)
            config['k_shot'] = fewshot_config.get('k_shots', 5)
            config['inner_lr'] = fewshot_config.get('adaptation_lr', 0.01)
            config['num_inner_steps'] = fewshot_config.get('adaptation_steps', 10)
            config['fewshot_support_split'] = fewshot_config.get('support_split', 'val_id')
            config['fewshot_query_split'] = fewshot_config.get('query_split', 'test_cross_env')
            config['fewshot_finetune_all'] = fewshot_config.get('finetune_all', False)
            config['fewshot_eval_shots'] = fewshot_config.get('eval_shots', False)
            
        print(f"Configuration loaded from {config_path}")
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Unable to load configuration file: {e}")
        sys.exit(1)

# Load default configuration
DEFAULT_CONFIG = load_config()

class SageMakerRunner:
    """Class that handles SageMaker training job creation and execution"""
    
    def __init__(self, config):
        """Initialize SageMaker session and role"""
        self.session = sagemaker.Session()
        self.role = sagemaker.get_execution_role()
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M")  # Shorter format for job names
        
        # Load configuration
        self.config = config
        
        # Extract common parameters from configuration
        self.s3_data_base = self.config.get("s3_data_base")
        self.s3_output_base = self.config.get("s3_output_base")
        
        # Verify S3 bucket exists
        self._verify_s3_bucket()
        
        print(f"SageMaker Runner initialized:")
        print(f"  S3 data base path: {self.s3_data_base}")
        print(f"  S3 output base path: {self.s3_output_base}")
        print(f"  Timestamp: {self.timestamp}")
    
    def _verify_s3_bucket(self):
        """Verify S3 bucket exists and list available data"""
        try:
            s3 = boto3.resource('s3')
            bucket_name = self.s3_data_base.split('/')[2]  # Extract bucket name from S3 path
            
            # Check if the bucket exists
            if bucket_name not in [bucket.name for bucket in s3.buckets.all()]:
                print(f"Error: Bucket '{bucket_name}' does not exist. Please create it first.")
                sys.exit(1)
        
            # Check contents of the S3 path
            s3_client = boto3.client('s3')
            bucket = self.s3_data_base.split('/')[2]
            prefix = '/'.join(self.s3_data_base.split('/')[3:])
            if not prefix.endswith('/'):
                prefix += '/'
        
            # Try to list contents of the S3 path
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
                    print(f"Available tasks in {self.s3_data_base}tasks/:")
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
        Run batch by task, executing each task on a single instance.
        
        Parameters:
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
            
            # Launch training job
            job_name = f"{task}-{batch_timestamp}"
            estimator.fit(
                inputs=self._prepare_inputs(task_config),
                job_name=job_name,
                wait=False
            )
            
            # Track job information
            jobs.append({
                'job_name': job_name,
                'task': task,
                'models': models_to_run,
                'status': 'InProgress',
                'estimator': estimator
            })
            
            print(f"Job submitted: {job_name}")
            
            # If waiting time is specified, wait between jobs
            if len(tasks_to_run) > 1 and task != tasks_to_run[-1]:
                wait_time = task_config.get('batch_wait_time', 30)
                print(f"Waiting {wait_time} seconds before submitting next job...")
                time.sleep(wait_time)
        
        return jobs
    
    def _create_estimator(self, config, base_job_name, models):
        """
        Create SageMaker PyTorch Estimator from given configuration
        """
        # Prepare hyperparameters
        hyperparameters = {
            'models': ','.join(models),
            'task_name': config.get('task_name', config.get('task')),  # Support backward compatibility for task and task_name
            'win_len': config.get('win_len', 500),  # Match default value from train_supervised.py
            'feature_size': config.get('feature_size', 232),
            'batch_size': config.get('batch_size', 32),  # Match default value from train_multi_model.py
            'epochs': config.get('epochs', 100),
            'test_splits': config.get('test_splits', 'all'),
            'seed': config.get('seed', 42),
            'learning_rate': config.get('learning_rate', 0.001),  # Add default learning rate
            'weight_decay': config.get('weight_decay', 1e-5),  # Add default weight decay
            'warmup_epochs': config.get('warmup_epochs', 5),  # Add default warmup period
            'patience': config.get('patience', 15)  # Add default patience value
        }
        
        # Add few-shot parameters (if enabled)
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
            max_run=config.get('max_run', 24 * 3600),  # Default 24-hour maximum run time
            keep_alive_period_in_seconds=config.get('keep_alive_period', 1200)  # Default keep instance active 20 minutes
        )
        
        return estimator
    
    def _prepare_inputs(self, config):
        """
        Prepare training data input channels
        Use only specific task data paths
        """
        task = config.get('task', config.get('task_name'))
        
        # Ensure path ends with a slash
        s3_data_base = self.s3_data_base
        if not s3_data_base.endswith('/'):
            s3_data_base += '/'
        
        # Build specific task data path
        task_data_path = f"{s3_data_base}tasks/{task}/"
        
        print(f"Using specific task data path: {task_data_path}")
        
        # Define input channels
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
        Run multi-task learning job
        
        Parameters:
            tasks (list): List of tasks to process
            model_type (str): Model type to use
            override_config (dict): Configuration override parameters
            
        Returns:
            dict: Dictionary containing job information
        """
        print(f"Starting multi-task learning job...")
        
        # Create multi-task specific configuration
        multi_config = dict(self.config)
        if override_config:
            multi_config.update(override_config)
        
        # Set model and tasks
        multi_config['model'] = model_type
        
        # Ensure tasks are provided and formatted correctly
        if tasks:
            # If provided as a string, convert to list
            if isinstance(tasks, str):
                tasks = tasks.split(',')
            multi_config['tasks'] = ','.join(tasks)
        elif 'tasks' not in multi_config:
            # If not specified, use first two tasks by default
            available_tasks = multi_config.get('available_tasks', [])
            if len(available_tasks) >= 2:
                multi_config['tasks'] = ','.join(available_tasks[:2])
            else:
                print("Error: Multi-task learning requires at least two tasks")
                return None
        
        # Create estimator for multi-task learning
        estimator = self._create_multitask_estimator(multi_config)
        
        # Launch training job
        job_name = f"multitask-{self.timestamp}"
        estimator.fit(
            inputs=self._prepare_multitask_inputs(multi_config),
            job_name=job_name,
            wait=False
        )
        
        # Return job information
        job_info = {
            'job_name': job_name,
            'tasks': multi_config['tasks'],
            'model': model_type,
            'status': 'InProgress',
            'estimator': estimator
        }
        
        print(f"Multi-task job submitted: {job_name}")
        return job_info
    
    def _create_multitask_estimator(self, config):
        """
        Create SageMaker PyTorch Estimator for multi-task learning
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
        
        # Add few-shot parameters (if enabled)
        if config.get('enable_few_shot', False) or config.get('fewshot_eval_shots', False):
            hyperparameters['enable_few_shot'] = "True"
            hyperparameters['k_shot'] = config.get('k_shot', 5)
            hyperparameters['inner_lr'] = config.get('inner_lr', 0.01)
            hyperparameters['num_inner_steps'] = config.get('num_inner_steps', 10)
        
        # Add model-specific parameters (if exists)
        if 'model_params' in config:
            for key, value in config['model_params'].items():
                # Replace hyphen with underscore in parameter name
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
            max_run=config.get('max_run', 24 * 3600),  # Default 24-hour maximum run time
            keep_alive_period_in_seconds=config.get('keep_alive_period', 1200)  # Default keep instance active 20 minutes
        )
        
        return estimator
    
    def _prepare_multitask_inputs(self, config):
        """
        Prepare input data channels for multi-task training
        Ensure only download data required for tasks
        """
        # Ensure path ends with a slash
        s3_data_base = self.s3_data_base
        if not s3_data_base.endswith('/'):
            s3_data_base += '/'
        
        # Get task list
        tasks_str = config.get('tasks', '')
        tasks_list = [t.strip() for t in tasks_str.split(',') if t.strip()]
        
        if not tasks_list:
            print("Warning: No tasks specified, using entire data directory")
            data_path = s3_data_base
        else:
            # Prepare input data for multi-task learning
            # Use only specific task data paths
            data_paths = []
            for task in tasks_list:
                task_path = f"{s3_data_base}tasks/{task}/"
                data_paths.append(task_path)
            
            # If only one task, use its path directly
            if len(data_paths) == 1:
                data_path = data_paths[0]
                print(f"Multi-task learning using single task data path: {data_path}")
            else:
                # If multiple tasks, we need to combine them through a manifest file or other means
                # But in current SageMaker implementation, we can only specify one path, so use parent directory
                data_path = f"{s3_data_base}tasks/"
                print(f"Multi-task learning using multiple tasks ({len(tasks_list)} tasks), data path: {data_path}")
                print(f"Task list: {', '.join(tasks_list)}")
        
        # Define input channels
        input_data = {
            'training': TrainingInput(
                s3_data=data_path,
                content_type='application/x-recordio',
                s3_data_type='S3Prefix'
            )
        }
        
        return input_data

def run_from_config(config_path=None):
    """
    Run SageMaker training job based on configuration file
    
    Parameters:
        config_path: Path to configuration file, if None use default configuration
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create SageMaker runner
    runner = SageMakerRunner(config)
    
    # Get tasks and models from configuration
    tasks = config.get('task')
    models = config.get('model')
    pipeline = config.get('pipeline', 'supervised')
    
    # Run corresponding task based on pipeline type in configuration
    if pipeline == 'multitask':
        # For multi-task learning, ensure tasks parameter is provided
        if 'tasks' in config:
            tasks = config['tasks']
        else:
            # If tasks not specified but task is provided, use task
            if tasks:
                tasks = [tasks]
            else:
                # Use default list of first two tasks
                tasks = config.get('available_tasks', [])[:2]
        
        # Run multi-task learning
        result = runner.run_multitask(tasks=tasks, model_type=models)
    else:
        # For supervised learning, can run single task or multiple tasks
        if tasks and not isinstance(tasks, list):
            tasks = [tasks]
        
        # If model is specified, ensure it's in list format
        if models and not isinstance(models, list):
            if ',' in models:
                models = models.split(',')
            else:
                models = [models]
        
        # Run batch processing tasks
        result = runner.run_batch_by_task(tasks=tasks, models=models)
    
    return result

def main():
    """Main entry function"""
    parser = argparse.ArgumentParser(description='Run WiFi sensing pipeline in SageMaker')
    
    # Configuration file parameter
    parser.add_argument('--config', type=str, default=None,
                        help='JSON configuration file path')
    
    args = parser.parse_args()
    
    # Run normal job from configuration file
    run_from_config(args.config)

if __name__ == "__main__":
    main()
