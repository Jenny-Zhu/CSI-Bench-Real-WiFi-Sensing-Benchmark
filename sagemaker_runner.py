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
"""

import os
import sys
import json
import argparse
from datetime import datetime

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
TASK = 'FourClass'  # Task type for integrated loader (e.g., ThreeClass, HumanNonhuman)

# Model Parameters
WIN_LEN = 250  # Window length for CSI data
FEATURE_SIZE = 98  # Feature size for CSI data

# Common Training Parameters
SEED = 42
BATCH_SIZE = 8
EPOCH_NUMBER = 1  # Number of training epochs
PATIENCE = 15  # Early stopping patience
MODEL_NAME = 'Transformer'

# Advanced Configuration
CODE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory containing the code
CONFIG_FILE = None  # Path to JSON configuration file to override defaults

#==============================================================================
# END OF CONFIGURATION SECTION
#==============================================================================

class SageMakerRunner:
    """Class to handle SageMaker training job creation and execution"""
    
    def __init__(self, role=None):
        """Initialize SageMaker session and role"""
        self.session = sagemaker.Session()
        self.role = role or sagemaker.get_execution_role()
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
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
    
    def get_supervised_config(self, training_dir=None, test_dirs=None, output_dir=None, mode='csi'):
        """Get configuration for supervised learning pipeline."""
        # Set default paths if not provided
        if training_dir is None:
            training_dir = TRAINING_DIR
        if test_dirs is None:
            test_dirs = TEST_DIRS
        if output_dir is None:
            output_dir = OUTPUT_DIR
        
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
        num_classes = task_class_mapping.get(TASK, 2)  # Default to 2 if task not found
        
        config = {
            # Data parameters
            'training-dir': training_dir,
            'test-dirs': test_dirs,
            'output-dir': output_dir,
            'results-subdir': 'supervised',
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
            'task': TASK,
            
            # Other parameters
            'seed': SEED,
            'device': 'cuda',  # SageMaker instances will have GPU
            'model-name': MODEL_NAME,
            'win-len': WIN_LEN,
            'feature-size': FEATURE_SIZE
        }
        
        return config
    
    def run_supervised(self, training_dir=None, test_dirs=None, output_dir=None, mode='csi', 
                       config_file=None, instance_type=None):
        """Run supervised learning pipeline on SageMaker"""
        print("Preparing supervised learning pipeline for SageMaker...")
        
        # Get configuration
        config = self.get_supervised_config(
            training_dir or TRAINING_DIR,
            test_dirs or TEST_DIRS,
            output_dir or OUTPUT_DIR, 
            mode or MODE
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
        job_name = f"{BASE_JOB_NAME}-{self.timestamp}"
        
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
        print("\nSageMaker Job Configuration:")
        print(f"  Job Name: {job_name}")
        print(f"  Instance Type: {instance_type}")
        print(f"  Input Channels:")
        for channel, path in inputs.items():
            print(f"    {channel}: {path}")
        print(f"  Output Path: {config['output-dir']}")
        print(f"  Model: {config['model-name']}")
        print(f"  Task: {config['task']}")
        print(f"  Number of Classes: {config['num-classes']}")
        print(f"  Number of Epochs: {config['num-epochs']}")
        print(f"  Batch Size: {config['batch-size']}")
        
        # Launch training job
        print("\nLaunching SageMaker training job...")
        estimator.fit(inputs, wait=False)
        
        print(f"\nSageMaker training job '{job_name}' launched.")
        print(f"Check the AWS SageMaker console for job status and logs.")
        
        # Return job details
        return {
            'job_name': job_name,
            'estimator': estimator,
            'config': config,
            'inputs': inputs
        }
    
def main():
    """Main function to execute from command line"""
    parser = argparse.ArgumentParser(description='Run WiFi sensing pipeline on SageMaker')
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
    
    args = parser.parse_args()
    
    runner = SageMakerRunner()
    runner.run_supervised(
        args.training_dir, 
        args.test_dirs, 
        args.output_dir, 
        args.mode, 
        args.config_file,
        args.instance_type
    )

if __name__ == "__main__":
    main()
