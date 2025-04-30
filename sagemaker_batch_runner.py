#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Batch Pipeline Runner - SageMaker Environment

This script enhances the original SageMaker runner to support batch training
across multiple tasks and models. It automatically creates a hierarchical
structure of results organized by task and model:

Task1/
  -- Model1/
  -- Model2/
Task2/
  -- Model1/
  -- Model2/

Usage in a JupyterLab notebook:
```
import sagemaker_batch_runner
runner = sagemaker_batch_runner.SageMakerBatchRunner()
runner.run_batch_training()
```
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from itertools import product

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3
import re

#==============================================================================
# CONFIGURATION SECTION - MODIFY PARAMETERS HERE
#==============================================================================

# Training Data S3 Location
# For supervised learning, this should point to a folder containing train and validation subdirectories
TRAINING_DIR = "s3://rnd-sagemaker/Data/Benchmark/demo/"
# Test directories can be a list of paths to evaluate on multiple test sets
TEST_DIRS = ["s3://rnd-sagemaker/Data/Benchmark/demo/test/"] 
OUTPUT_DIR = "s3://rnd-sagemaker/Benchmark_Log/demo/"

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

# Batch Training Settings
# Define tasks to run
TASKS = [
    'FourClass',  # Demo task: 4-class classification
    # Add more tasks as needed
]

# Define models to run for each task
MODELS = [
    'Transformer',  # Demo model: Our ViT implementation
    # Add more models as needed:
    # 'ResNet18',
    # 'LeNet',
    # 'LSTM',
    # 'GRUNet',
    # 'MLP',
    # 'CNNAttention'
]

# Model Parameters
WIN_LEN = 250  # Window length for CSI data
FEATURE_SIZE = 98  # Feature size for CSI data

# Common Training Parameters
SEED = 42
BATCH_SIZE = 8
EPOCH_NUMBER = 1  # Number of training epochs
PATIENCE = 15  # Early stopping patience

# Job management
MAX_PARALLEL_JOBS = 2  # Maximum number of SageMaker jobs to run in parallel
JOB_CHECK_INTERVAL = 60  # Seconds to wait between job status checks

# Advanced Configuration
CODE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory containing the code
CONFIG_FILE = None  # Path to JSON configuration file to override defaults

#==============================================================================
# END OF CONFIGURATION SECTION
#==============================================================================

class SageMakerBatchRunner:
    """Class to handle batch SageMaker training job creation and execution"""
    
    def __init__(self, role=None):
        """Initialize SageMaker session and role"""
        self.session = sagemaker.Session()
        self.role = role or sagemaker.get_execution_role()
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.active_jobs = {}  # Track running jobs
        
        # Verify the rnd-sagemaker bucket exists
        s3 = boto3.resource('s3')
        bucket_name = "rnd-sagemaker"
        if bucket_name not in [bucket.name for bucket in s3.buckets.all()]:
            print(f"Error: The bucket '{bucket_name}' does not exist. Please create it first.")
            sys.exit(1)
        
        print(f"Using S3 paths:")
        print(f"  Training: {TRAINING_DIR}")
        print(f"  Test: {TEST_DIRS}")
        print(f"  Output: {OUTPUT_DIR}")
    
    def get_supervised_config(self, task, model_name, training_dir=None, test_dirs=None, output_dir=None, mode='csi'):
        """Get configuration for supervised learning pipeline with task and model specific settings"""
        # Set default paths if not provided
        if training_dir is None:
            training_dir = TRAINING_DIR
        if test_dirs is None:
            test_dirs = TEST_DIRS
        if output_dir is None:
            # Create hierarchical output path for task/model
            output_dir = os.path.join(OUTPUT_DIR, task, model_name)
        
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
            'Detection': 2
        }
        
        # Get number of classes based on task
        num_classes = task_class_mapping.get(task, 2)  # Default to 2 if task not found
        
        # Adjust model-specific parameters if needed
        win_len = WIN_LEN
        feature_size = FEATURE_SIZE
        
        # Model-specific adjustments (if needed)
        if model_name == 'ResNet18':
            # Any specific settings for ResNet18
            pass
        elif model_name == 'LSTM':
            # Any specific settings for LSTM
            pass
        
        config = {
            # Data parameters
            'training-dir': training_dir,
            'test-dirs': test_dirs,
            'output-dir': output_dir,
            'results-subdir': f'supervised/{task}/{model_name}',
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
            'task': task,
            
            # Other parameters
            'seed': SEED,
            'device': 'cuda',  # SageMaker instances will have GPU
            'model-name': model_name,
            'win-len': win_len,
            'feature-size': feature_size
        }
        
        return config
    
    def run_supervised(self, task, model_name, training_dir=None, test_dirs=None, output_dir=None, 
                      mode='csi', config_file=None, instance_type=None, wait=False):
        """Run supervised learning pipeline on SageMaker for specific task and model"""
        print(f"Preparing supervised learning pipeline for task '{task}' with model '{model_name}'...")
        
        # Create task/model specific output dir
        if output_dir is None:
            model_output_dir = os.path.join(OUTPUT_DIR, task, model_name)
        else:
            model_output_dir = os.path.join(output_dir, task, model_name)
        
        # Get configuration
        config = self.get_supervised_config(
            task=task,
            model_name=model_name,
            training_dir=training_dir or TRAINING_DIR,
            test_dirs=test_dirs or TEST_DIRS,
            output_dir=model_output_dir,
            mode=mode or MODE
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
        job_name = f"{BASE_JOB_NAME}-{task}-{model_name}-{self.timestamp}"
        
        # Ensure requirements.txt is included
        dependencies = ["requirements.txt"]
        
        # Print information about requirements.txt
        req_path = os.path.join(CODE_DIR, "requirements.txt")
        if os.path.exists(req_path):
            print(f"Using requirements.txt from {req_path}")
            try:
                with open(req_path, 'r') as f:
                    requirements = f.read().strip()
                    print(f"Requirements file contains:\n{requirements}")
            except Exception as e:
                print(f"Error reading requirements.txt: {e}")
        else:
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
        print(f"\nSageMaker Job Configuration for {task}/{model_name}:")
        print(f"  Job Name: {job_name}")
        print(f"  Instance Type: {instance_type}")
        print(f"  Input Channels:")
        for channel, path in inputs.items():
            print(f"    {channel}: {path}")
        print(f"  Output Path: {config['output-dir']}")
        print(f"  Task: {task}")
        print(f"  Model: {model_name}")
        print(f"  Number of Classes: {config['num-classes']}")
        print(f"  Number of Epochs: {config['num-epochs']}")
        print(f"  Batch Size: {config['batch-size']}")
        
        # Launch training job
        print(f"\nLaunching SageMaker training job for {task}/{model_name}...")
        estimator.fit(inputs, wait=wait)
        
        print(f"\nSageMaker training job '{job_name}' launched.")
        print(f"Check the AWS SageMaker console for job status and logs.")
        
        # Return job details
        job_info = {
            'job_name': job_name,
            'estimator': estimator,
            'config': config,
            'inputs': inputs,
            'task': task,
            'model': model_name
        }
        
        # Track active job
        self.active_jobs[job_name] = job_info
        
        return job_info

    def check_job_status(self, job_name):
        """Check status of a SageMaker training job"""
        client = boto3.client('sagemaker')
        response = client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        return status
    
    def get_active_job_count(self):
        """Get count of currently active (non-completed) jobs"""
        active_count = 0
        jobs_to_remove = []
        
        for job_name, job_info in self.active_jobs.items():
            try:
                status = self.check_job_status(job_name)
                if status in ['Completed', 'Failed', 'Stopped']:
                    jobs_to_remove.append(job_name)
                else:
                    active_count += 1
            except Exception as e:
                print(f"Error checking job {job_name}: {str(e)}")
                jobs_to_remove.append(job_name)
        
        # Remove completed jobs from tracking
        for job_name in jobs_to_remove:
            del self.active_jobs[job_name]
            
        return active_count
    
    def run_batch_training(self, tasks=None, models=None, training_dir=None, test_dirs=None, 
                          output_dir=None, mode='csi', config_file=None, instance_type=None):
        """Run batch training for multiple tasks and models"""
        if tasks is None:
            tasks = TASKS
        if models is None:
            models = MODELS
            
        print(f"Starting batch training for tasks: {tasks}")
        print(f"Models to train: {models}")
        
        # Count total jobs
        total_jobs = len(tasks) * len(models)
        completed_jobs = 0
        
        # Timestamp for this batch run
        batch_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create summary log for this batch run
        summary_log = []
        
        # Process each task/model combination
        for task, model_name in product(tasks, models):
            # Wait if we have too many active jobs
            while self.get_active_job_count() >= MAX_PARALLEL_JOBS:
                print(f"Waiting for job slots... (active: {self.get_active_job_count()}, max: {MAX_PARALLEL_JOBS})")
                time.sleep(JOB_CHECK_INTERVAL)
            
            # Run job for this task/model combination
            try:
                job_info = self.run_supervised(
                    task=task,
                    model_name=model_name,
                    training_dir=training_dir,
                    test_dirs=test_dirs,
                    output_dir=output_dir,
                    mode=mode,
                    config_file=config_file,
                    instance_type=instance_type,
                    wait=False
                )
                
                # Add to summary log
                summary_log.append({
                    'task': task,
                    'model': model_name,
                    'job_name': job_info['job_name'],
                    'output_dir': job_info['config']['output-dir'],
                    'status': 'Launched',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                completed_jobs += 1
                print(f"Progress: {completed_jobs}/{total_jobs} jobs launched")
                
                # Small delay to avoid API rate limits
                time.sleep(2)
                
            except Exception as e:
                print(f"Error launching job for {task}/{model_name}: {str(e)}")
                # Add error to summary log
                summary_log.append({
                    'task': task,
                    'model': model_name,
                    'job_name': 'Failed',
                    'output_dir': '',
                    'status': f'Error: {str(e)}',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Save summary log to local file
        log_file = f"batch_training_summary_{batch_timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump(summary_log, f, indent=2)
        
        print(f"\nBatch training complete! {completed_jobs}/{total_jobs} jobs launched.")
        print(f"Summary log saved to: {log_file}")
        
        # Return summary log
        return summary_log

def main():
    """Main function to execute from command line"""
    parser = argparse.ArgumentParser(description='Run batch WiFi sensing training on SageMaker')
    parser.add_argument('--training-dir', type=str, default=TRAINING_DIR,
                      help='S3 URI containing training data')
    parser.add_argument('--test-dirs', type=str, nargs='+', default=TEST_DIRS,
                      help='List of S3 URIs containing test data. Can specify multiple paths')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                      help='S3 URI to save output results')
    parser.add_argument('--mode', type=str, default=MODE,
                      choices=['csi', 'acf'],
                      help='Data modality to use')
    parser.add_argument('--config-file', type=str, default=CONFIG_FILE,
                      help='JSON configuration file to override defaults')
    parser.add_argument('--instance-type', type=str, default=INSTANCE_TYPE,
                      help='SageMaker instance type for training')
    parser.add_argument('--tasks', type=str, nargs='+', default=TASKS,
                      help='List of tasks to run')
    parser.add_argument('--models', type=str, nargs='+', default=MODELS,
                      help='List of models to run for each task')
    parser.add_argument('--max-parallel', type=int, default=MAX_PARALLEL_JOBS,
                      help='Maximum number of parallel jobs')
    
    args = parser.parse_args()
    
    # Update global variable for max parallel jobs
    global MAX_PARALLEL_JOBS
    MAX_PARALLEL_JOBS = args.max_parallel
    
    runner = SageMakerBatchRunner()
    runner.run_batch_training(
        tasks=args.tasks,
        models=args.models,
        training_dir=args.training_dir, 
        test_dirs=args.test_dirs, 
        output_dir=args.output_dir, 
        mode=args.mode, 
        config_file=args.config_file,
        instance_type=args.instance_type
    )

if __name__ == "__main__":
    main() 