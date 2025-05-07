#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - SageMaker Environment

This script allows you to run the supervised learning pipeline in the SageMaker environment.
It creates a SageMaker PyTorch Estimator for submitting training jobs.

Key features:
1. Batch execution of training tasks, with each task using a single instance to run multiple models
2. Support for overriding default settings using JSON configuration files

Usage example:
```
import sagemaker_runner
runner = sagemaker_runner.SageMakerRunner(config_file="configs/sagemaker_custom_config.json")
runner.run_batch_by_task()
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
    
    def __init__(self, config_file=None, role=None):
        """Initialize SageMaker session and role"""
        self.session = sagemaker.Session()
        self.role = role or sagemaker.get_execution_role()
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M")  # Shorter format for job names
        
        # Load configuration
        self.config = load_config(config_file)
        
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
        
        # Update batch summary file
        self._update_batch_summary(jobs, batch_timestamp)
        
        return jobs
    
    def _create_estimator(self, config, base_job_name, models):
        """
        Create a SageMaker PyTorch Estimator from the given configuration
        
        Args:
            config (dict): Configuration dictionary
            base_job_name (str): Base job name
            models (list): List of models to run
            
        Returns:
            PyTorch: SageMaker PyTorch Estimator
        """
        # Prepare hyperparameters
        hyperparameters = {
            'models': ','.join(models),
            'task': config.get('task'),
            'mode': config.get('mode', 'csi'),
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
            entry_point='sagemaker_entry_point.py',
            source_dir=CODE_DIR,
            role=self.role,
            framework_version=config.get('framework_version', '1.12.1'),
            py_version=config.get('py_version', 'py38'),
            instance_count=config.get('instance_count', 1),
            instance_type=config.get('instance_type', 'ml.g4dn.xlarge'),
            base_job_name=base_job_name,
            hyperparameters=hyperparameters,
            volume_size=config.get('ebs_volume_size', 30),
            max_run=172800  # 48 hours max run time
        )
        
        return estimator
    
    def _prepare_inputs(self, config):
        """
        Prepare input data channels for training
        
        Args:
            config (dict): Configuration dictionary
        
        Returns:
            dict: Dictionary of input channels
        """
        task = config.get('task')
        s3_data_base = config.get('s3_data_base')
        
        # Build S3 path to the specific task data
        task_data_path = f"{s3_data_base}tasks/{task}/"
        
        # Define input channels
        input_data = {
            'training': TrainingInput(
                s3_data=task_data_path,
                content_type='application/x-recordio',
                s3_data_type='S3Prefix'
            )
        }
        
        return input_data
    
    def _update_batch_summary(self, jobs, batch_timestamp):
        """
        Update batch summary file in S3 with job information
        
        Args:
            jobs (list): List of job information dictionaries
            batch_timestamp (str): Timestamp for the batch
        """
        # Convert jobs to JSON serializable format
        jobs_info = []
        for job in jobs:
            job_info = {
                'job_name': job['job_name'],
                'task': job['task'],
                'models': job['models'],
                'status': job['status']
            }
            jobs_info.append(job_info)
        
        # Create JSON string
        json_data = json.dumps({
            'timestamp': batch_timestamp,
            'jobs': jobs_info
        }, indent=2)
        
        # Upload to S3
        try:
            s3_output_base = self.s3_output_base
            s3_client = boto3.client('s3')
            bucket = s3_output_base.split('/')[2]
            prefix = '/'.join(s3_output_base.split('/')[3:])
            
            # Create a path for the summary
            if not prefix.endswith('/'):
                prefix += '/'
            key = f"{prefix}batch_summary_{batch_timestamp}.json"
            
            # Upload the file
            s3_client.put_object(
                Body=json_data,
                Bucket=bucket,
                Key=key
            )
            
            print(f"Batch summary uploaded to s3://{bucket}/{key}")
        except Exception as e:
            print(f"Warning: Could not upload batch summary: {e}")

    def run_multitask(self, tasks=None, model_type="transformer", override_config=None):
        """
        Run a multitask learning job
        
        Args:
            tasks (list): List of tasks for multitask learning
            model_type (str): Model architecture to use
            override_config (dict): Configuration override parameters
            
        Returns:
            dict: Job information
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
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            PyTorch: SageMaker PyTorch Estimator
        """
        # Prepare hyperparameters
        hyperparameters = {
            'pipeline': 'multitask',
            'tasks': config.get('tasks'),
            'model': config.get('model', 'transformer'),
            'mode': config.get('mode', 'csi'),
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
                hyperparameters[key] = value
        else:
            # Add common parameters
            for param in ['lr', 'emb_dim', 'dropout', 'patience']:
                if param in config:
                    hyperparameters[param] = config[param]
        
        # Create estimator
        estimator = PyTorch(
            entry_point='sagemaker_entry_point.py',
            source_dir=CODE_DIR,
            role=self.role,
            framework_version=config.get('framework_version', '1.12.1'),
            py_version=config.get('py_version', 'py38'),
            instance_count=config.get('instance_count', 1),
            instance_type=config.get('instance_type', 'ml.g4dn.xlarge'),
            base_job_name='wifi-sensing-multitask',
            hyperparameters=hyperparameters,
            volume_size=config.get('ebs_volume_size', 30),
            max_run=172800  # 48 hours max run time
        )
        
        return estimator
    
    def _prepare_multitask_inputs(self, config):
        """
        Prepare input data channels for multitask training
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            dict: Dictionary of input channels
        """
        s3_data_base = config.get('s3_data_base')
        
        # For multitask, we use the main data directory
        # Each task should be a subfolder
        data_path = s3_data_base
        
        # Define input channels
        input_data = {
            'training': TrainingInput(
                s3_data=data_path,
                content_type='application/x-recordio',
                s3_data_type='S3Prefix'
            )
        }
        
        return input_data

def main():
    """Main entry point when script is executed directly"""
    parser = argparse.ArgumentParser(description='Run SageMaker WiFi Sensing pipeline')
    
    # Basic parameters
    parser.add_argument('--config_file', type=str, default=None,
                        help='JSON configuration file to override defaults')
    parser.add_argument('--tasks', type=str, default=None,
                        help='Comma-separated list of tasks to run')
    parser.add_argument('--models', type=str, default=None, 
                        help='Comma-separated list of models to run for each task')
    parser.add_argument('--pipeline', type=str, default='supervised',
                        choices=['supervised', 'multitask'],
                        help='Type of pipeline to run')
    
    args = parser.parse_args()
    
    # Create SageMaker runner
    runner = SageMakerRunner(config_file=args.config_file)
    
    # Process tasks and models
    tasks = args.tasks.split(',') if args.tasks else None
    models = args.models.split(',') if args.models else None
    
    # Run appropriate pipeline
    if args.pipeline == 'multitask':
        runner.run_multitask(tasks=tasks)
    else:  # Default to supervised
        runner.run_batch_by_task(tasks=tasks, models=models)

if __name__ == "__main__":
    main()
