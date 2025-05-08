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
            'task_name': config.get('task_name', config.get('task')),  # Support both task and task_name for backward compatibility
            'win_len': config.get('win_len', 500),
            'feature_size': config.get('feature_size', 232),
            'batch_size': config.get('batch_size', 32),  # Match default in train_multi_model.py
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
        Ensure to only use data paths specific to the task
        """
        task = config.get('task')
        s3_data_base = config.get('s3_data_base')
        
        # Build specific task data path
        # Ensure path ends with a slash
        if not s3_data_base.endswith('/'):
            s3_data_base += '/'
        
        # Build specific task data path
        task_data_path = f"{s3_data_base}tasks/{task}/"
        
        print(f"Using task-specific data path: {task_data_path}")
        
        # Define input channel
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
                # Replace dashes with underscores in parameter names
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
        Ensure to only download data needed for the tasks
        """
        s3_data_base = config.get('s3_data_base')
        
        # Ensure path ends with a slash
        if not s3_data_base.endswith('/'):
            s3_data_base += '/'
        
        # Get the task list
        tasks_str = config.get('tasks', '')
        tasks_list = [t.strip() for t in tasks_str.split(',') if t.strip()]
        
        if not tasks_list:
            print("Warning: No tasks specified, using entire data directory")
            data_path = s3_data_base
        else:
            # Prepare input data for multitask learning
            # Only use data paths for specified tasks
            data_paths = []
            for task in tasks_list:
                task_path = f"{s3_data_base}tasks/{task}/"
                data_paths.append(task_path)
            
            # If there is only one task, use that task's path directly
            if len(data_paths) == 1:
                data_path = data_paths[0]
                print(f"Multitask learning using single task data path: {data_path}")
            else:
                # If there are multiple tasks, we would need to combine via manifest file or other means
                # But in the current SageMaker implementation, we can only specify one path, so use the parent directory
                data_path = f"{s3_data_base}tasks/"
                print(f"Multitask learning using multiple tasks ({len(tasks_list)} tasks), data path: {data_path}")
                print(f"Task list: {', '.join(tasks_list)}")
        
        # Define input channel
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
        Run a quick environment test job to verify dependencies and environment configuration
        Use fake data for environment testing, no need to download the full dataset
        
        Returns:
            dict: Dictionary containing job information
        """
        print(f"Starting environment test job...")
        
        # Create a specialized test configuration
        test_config = dict(self.config)
        test_config['epochs'] = 1  # Set to 1 round
        test_config['batch_size'] = 2  # Use small batch
        
        # Create special environment test script parameters
        hyperparameters = {
            'test_env': 'True',  # Tell entry script this is environment test
            'batch_size': 2,
            'epochs': 1,
            'seed': 42,
            'win_len': 10,  # Use very small window size
            'feature_size': 10  # Use very small feature size
        }
        
        # Create and upload fake data to S3
        fake_data_s3_path = self._create_fake_test_data()
        
        # Create test environment estimator
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
            max_run=3600  # Maximum run time 1 hour
        )
        
        # Prepare using fake data input configuration
        minimal_inputs = {
            'training': TrainingInput(
                s3_data=fake_data_s3_path,
                content_type='application/octet-stream',
                s3_data_type='S3Prefix'
            )
        }
        
        print(f"Using fake data for environment test, path: {fake_data_s3_path}")
        
        # Start training job
        job_name = f"env-test-{self.timestamp}"
        estimator.fit(
            inputs=minimal_inputs,
            job_name=job_name,
            wait=False
        )
        
        # Return job info
        job_info = {
            'job_name': job_name,
            'type': 'environment_test',
            'status': 'InProgress',
            'estimator': estimator,
            'fake_data_path': fake_data_s3_path
        }
        
        print(f"Submitted environment test job: {job_name}")
        print(f"This job uses fake data, only for verifying environment setup and dependencies")
        return job_info

    def _create_fake_test_data(self):
        """
        Create fake data and upload to S3
        
        Returns:
            str: Fake data S3 path
        """
        import tempfile
        import numpy as np
        import os
        import time
        
        print("Creating fake data for environment test...")
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple numpy array as fake data
            fake_x = np.random.rand(100, 10, 10).astype(np.float32)  # 100 samples, window length 10, feature number 10
            fake_y = np.random.randint(0, 4, size=(100,)).astype(np.int32)  # 100 labels, values in 0-3
            
            # Save to temporary file
            data_file = os.path.join(temp_dir, 'fake_data.npz')
            np.savez(data_file, x=fake_x, y=fake_y)
            
            # Ensure uniqueness, use timestamp to create S3 path
            timestamp = int(time.time())
            s3_bucket = self.s3_data_base.split('/')[2]
            s3_prefix = f"test_env_data/{timestamp}"
            
            s3_fake_data_path = f"s3://{s3_bucket}/{s3_prefix}/fake_data.npz"
            
            # Upload to S3
            s3 = boto3.client('s3')
            print(f"Uploading fake data to {s3_fake_data_path}...")
            
            try:
                s3.upload_file(
                    Filename=data_file,
                    Bucket=s3_bucket,
                    Key=f"{s3_prefix}/fake_data.npz"
                )
                print("Fake data upload succeeded")
            except Exception as e:
                print(f"Error uploading fake data: {e}")
                # If upload fails, create a simpler file and try again
                simple_file = os.path.join(temp_dir, 'test.txt')
                with open(simple_file, 'w') as f:
                    f.write("This is a test file for SageMaker environment testing.")
                
                try:
                    s3.upload_file(
                        Filename=simple_file,
                        Bucket=s3_bucket,
                        Key=f"{s3_prefix}/test.txt"
                    )
                    print("Simple test file upload succeeded")
                    s3_fake_data_path = f"s3://{s3_bucket}/{s3_prefix}/"
                except Exception as e2:
                    print(f"Failed to upload simple test file: {e2}")
                    # Last option: use known existing path
                    s3_fake_data_path = f"s3://{s3_bucket}/test_env_data/"
                    print(f"Using possibly existing default path: {s3_fake_data_path}")
        
        return s3_fake_data_path

def run_from_config(config_path=None):
    """
    Run SageMaker training task based on configuration file
    
    Args:
        config_path: Configuration file path, if None use default configuration
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create SageMaker runner
    runner = SageMakerRunner(config)
    
    # Get tasks and models from configuration
    tasks = config.get('task')
    models = config.get('model')
    pipeline = config.get('pipeline', 'supervised')
    
    # Run corresponding task based on configuration's pipeline type
    if pipeline == 'multitask':
        # For multitask learning, ensure tasks parameter is provided
        if 'tasks' in config:
            tasks = config['tasks']
        else:
            # If tasks are not specified but task is provided, use task
            if tasks:
                tasks = [tasks]
            else:
                # Use first two tasks from default task list
                tasks = config.get('available_tasks', [])[:2]
        
        # Run multitask learning
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
        
        # Run batch processing task
        result = runner.run_batch_by_task(tasks=tasks, models=models)
    
    return result

def main():
    """Main entry function"""
    parser = argparse.ArgumentParser(description='Run SageMaker WiFi Sensing pipeline')
    
    # Keep configuration file parameter and add test environment option
    parser.add_argument('--config', type=str, default=None,
                        help='JSON configuration file path')
    parser.add_argument('--test-env', action='store_true',
                        help='Only test environment configuration and dependencies, without full training')
    
    args = parser.parse_args()
    
    if args.test_env:
        # Run environment test job
        config = load_config(args.config)
        runner = SageMakerRunner(config)
        runner.run_test_env()
    else:
        # Run normal job from configuration file
        run_from_config(args.config)

if __name__ == "__main__":
    main()
