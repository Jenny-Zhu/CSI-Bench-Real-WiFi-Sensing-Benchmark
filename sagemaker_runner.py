#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - AWS SageMaker Environment

This script allows you to run any of the three pipelines on AWS SageMaker:
1. Pretraining (Self-supervised learning)
2. Supervised learning
3. Meta-learning

Usage:
    python sagemaker_runner.py --pipeline pretraining
    python sagemaker_runner.py --pipeline supervised
    python sagemaker_runner.py --pipeline meta
    python sagemaker_runner.py --pipeline all
    python sagemaker_runner.py --pipeline pretraining --create-job  # Creates a SageMaker Processing Job
"""

import os
import sys
import subprocess
import torch
import time
import argparse
import json
import boto3
from datetime import datetime

try:
    import sagemaker
    from sagemaker import get_execution_role
    from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
    SAGEMAKER_AVAILABLE = True
except ImportError:
    print("SageMaker SDK not available. Some features will be disabled.")
    SAGEMAKER_AVAILABLE = False

# Check SageMaker environment
try:
    if SAGEMAKER_AVAILABLE:
        role = get_execution_role()
        session = sagemaker.Session()
        region = session.boto_region_name
        account_id = session.account_id()
        bucket = session.default_bucket()
        
        print(f"SageMaker Role: {role}")
        print(f"Region: {region}")
        print(f"Account ID: {account_id}")
        print(f"Default S3 Bucket: {bucket}")
    else:
        role = None
        session = None
        region = "us-east-1"
        bucket = None
except Exception as e:
    print(f"Error setting up SageMaker session: {e}")
    print("Using local execution mode")
    role = None
    session = None
    region = "us-east-1"
    bucket = None

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\nUsing CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = torch.device("cpu")
    print("\nCUDA is not available. Using CPU.")

# Print PyTorch version
print(f"\nPyTorch version: {torch.__version__}")

# SageMaker paths are different from local paths
BASE_DIR = '/opt/ml'
DATA_DIR = os.path.join(BASE_DIR, 'input/data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'model')

# S3 paths for data and output
if bucket:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    S3_PREFIX = f"wifi-sensing/{timestamp}"
    S3_DATA_PATH = f"s3://{bucket}/{S3_PREFIX}/data"
    S3_OUTPUT_PATH = f"s3://{bucket}/{S3_PREFIX}/output"
    print(f"S3 Data Path: {S3_DATA_PATH}")
    print(f"S3 Output Path: {S3_OUTPUT_PATH}")
else:
    S3_DATA_PATH = None
    S3_OUTPUT_PATH = None
    print("S3 paths not available (local mode)")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Common parameters
SEED = 42
BATCH_SIZE = 16
MODEL_NAME = 'WiT'

# Set device string for command line arguments
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_command(cmd, display_output=True):
    """Run a command and stream the output."""
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=True
    )
    
    # Stream the output
    output = []
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        if display_output:
            print(line.rstrip())
        output.append(line.rstrip())
    
    process.wait()
    return process.returncode, '\n'.join(output)

def sync_data_from_s3(s3_path, local_path):
    """Sync data from S3 to local directory."""
    if not s3_path or not bucket:
        print("S3 path not available. Skipping S3 sync.")
        return
    
    os.makedirs(local_path, exist_ok=True)
    cmd = f"aws s3 sync {s3_path} {local_path}"
    print(f"Syncing data from {s3_path} to {local_path}...")
    returncode, output = run_command(cmd)
    if returncode == 0:
        print("Data sync completed successfully.")
    else:
        print(f"Data sync failed with return code {returncode}.")
    return returncode == 0

def sync_output_to_s3(local_path, s3_path):
    """Sync output data from local directory to S3."""
    if not s3_path or not bucket:
        print("S3 path not available. Skipping S3 sync.")
        return
    
    cmd = f"aws s3 sync {local_path} {s3_path}"
    print(f"Syncing output from {local_path} to {s3_path}...")
    returncode, output = run_command(cmd)
    if returncode == 0:
        print("Output sync completed successfully.")
    else:
        print(f"Output sync failed with return code {returncode}.")
    return returncode == 0

# Configure pretraining pipeline
PRETRAIN_CONFIG = {
    # Data parameters
    'csi_data_dir': os.path.join(DATA_DIR, 'csi'),
    'acf_data_dir': os.path.join(DATA_DIR, 'acf'),
    'output_dir': OUTPUT_DIR,
    'results_subdir': 'ssl_pretrain',
    
    # Training parameters
    'batch_size': BATCH_SIZE,
    'learning_rate': 1e-5,
    'weight_decay': 0.001,
    'num_epochs': 100,
    'warmup_epochs': 5,
    'patience': 20,
    
    # Model parameters
    'mode': 'csi',  # Options: 'csi', 'acf'
    'depth': 6,
    'in_channels': 1,
    'emb_size': 128,
    'freq_out': 10,
    
    # Other parameters
    'seed': SEED,
    'device': DEVICE,
    'model_name': MODEL_NAME
}

# Configure supervised learning pipeline
SUPERVISED_CONFIG = {
    # Data parameters
    'csi_data_dir': os.path.join(DATA_DIR, 'csi'),
    'acf_data_dir': os.path.join(DATA_DIR, 'acf'),
    'output_dir': OUTPUT_DIR,
    'results_subdir': 'supervised',
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    
    # Training parameters
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_epochs': 100,
    'warmup_epochs': 5,
    'patience': 15,
    
    # Model parameters
    'mode': 'csi',  # Options: 'csi', 'acf'
    'num_classes': 2,
    'freeze_backbone': False,  # Set to True to freeze the backbone
    'pretrained': False,  # Set to True to use a pretrained model
    
    # Other parameters
    'seed': SEED,
    'device': DEVICE,
    'model_name': MODEL_NAME,
    'unseen_test': False  # Set to True to test on unseen environments (ACF mode only)
}

# Configure meta-learning pipeline
META_CONFIG = {
    # Data parameters
    'data_dir': os.path.join(DATA_DIR, 'benchmark'),
    'output_dir': OUTPUT_DIR,
    'results_subdir': 'meta_learning',
    'resize_height': 64,
    'resize_width': 64,
    
    # Meta-learning parameters
    'meta_method': 'maml',  # Options: 'maml', 'lstm'
    'meta_batch_size': 4,
    'inner_lr': 0.01,
    'meta_lr': 0.001,
    'num_iterations': 10000,  # Reduced for demonstration
    'meta_validation_interval': 1000,
    
    # Task parameters
    'n_way': 2,
    'k_shot': 5,
    'q_query': 15,
    
    # Model parameters
    'model_type': 'csi',
    'emb_size': 128,
    'depth': 6,
    'in_channels': 1,
    
    # Other parameters
    'seed': SEED,
    'device': DEVICE,
    'model_name': f"{MODEL_NAME}_Meta"
}

def run_pretraining(config):
    """Run pretraining pipeline."""
    print("\n========== Running Pretraining Pipeline ==========\n")
    
    # Build command
    cmd = "python pretrain.py "
    
    # Add all arguments
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd += f"--{key} "
        else:
            cmd += f"--{key} {value} "
    
    print(f"Running command: {cmd}")
    start_time = time.time()
    returncode, output = run_command(cmd)
    end_time = time.time()
    
    if returncode == 0:
        print(f"\nPretraining completed successfully in {(end_time - start_time)/60:.2f} minutes")
        # Sync results to S3
        pretrain_output_dir = os.path.join(OUTPUT_DIR, config['results_subdir'])
        if S3_OUTPUT_PATH:
            s3_output_path = f"{S3_OUTPUT_PATH}/{config['results_subdir']}"
            sync_output_to_s3(pretrain_output_dir, s3_output_path)
    else:
        print(f"\nPretraining failed with return code {returncode}")
    
    return returncode == 0

def run_supervised(config):
    """Run supervised learning pipeline."""
    print("\n========== Running Supervised Learning Pipeline ==========\n")
    
    # Build command
    cmd = "python train_supervised.py "
    
    # Add all arguments
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd += f"--{key} "
        else:
            cmd += f"--{key} {value} "
    
    print(f"Running command: {cmd}")
    start_time = time.time()
    returncode, output = run_command(cmd)
    end_time = time.time()
    
    if returncode == 0:
        print(f"\nSupervised learning completed successfully in {(end_time - start_time)/60:.2f} minutes")
        # Sync results to S3
        supervised_output_dir = os.path.join(OUTPUT_DIR, config['results_subdir'])
        if S3_OUTPUT_PATH:
            s3_output_path = f"{S3_OUTPUT_PATH}/{config['results_subdir']}"
            sync_output_to_s3(supervised_output_dir, s3_output_path)
    else:
        print(f"\nSupervised learning failed with return code {returncode}")
    
    return returncode == 0

def run_meta_learning(config):
    """Run meta-learning pipeline."""
    print("\n========== Running Meta-Learning Pipeline ==========\n")
    
    # Build command
    cmd = "python meta_learning.py "
    
    # Add all arguments
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd += f"--{key} "
        else:
            cmd += f"--{key} {value} "
    
    print(f"Running command: {cmd}")
    start_time = time.time()
    returncode, output = run_command(cmd)
    end_time = time.time()
    
    if returncode == 0:
        print(f"\nMeta-learning completed successfully in {(end_time - start_time)/60:.2f} minutes")
        # Sync results to S3
        meta_output_dir = os.path.join(OUTPUT_DIR, config['results_subdir'])
        if S3_OUTPUT_PATH:
            s3_output_path = f"{S3_OUTPUT_PATH}/{config['results_subdir']}"
            sync_output_to_s3(meta_output_dir, s3_output_path)
    else:
        print(f"\nMeta-learning failed with return code {returncode}")
    
    return returncode == 0

def run_full_pipeline():
    """Run all three pipelines in sequence."""
    print("\n========== Running Full Pipeline ==========\n")
    print("Starting full pipeline execution...\n")
    
    # 1. Run pretraining
    print("=== STEP 1: PRETRAINING ===")
    pretraining_success = run_pretraining(PRETRAIN_CONFIG)
    if not pretraining_success:
        print("Pretraining failed. Stopping pipeline.")
        return False
    
    # Update supervised config to use pretrained model
    SUPERVISED_CONFIG['pretrained'] = True
    SUPERVISED_CONFIG['pretrained_model'] = os.path.join(
        OUTPUT_DIR, 
        'ssl_pretrain', 
        f"{MODEL_NAME}_{PRETRAIN_CONFIG['mode']}/best_model.pt"
    )
    
    # 2. Run supervised learning
    print("\n=== STEP 2: SUPERVISED LEARNING ===")
    supervised_success = run_supervised(SUPERVISED_CONFIG)
    if not supervised_success:
        print("Supervised learning failed. Stopping pipeline.")
        return False
    
    # 3. Run meta-learning
    print("\n=== STEP 3: META-LEARNING ===")
    meta_success = run_meta_learning(META_CONFIG)
    if not meta_success:
        print("Meta-learning failed.")
        return False
    
    # 4. Sync all results to S3
    if S3_OUTPUT_PATH:
        print("\n=== STEP 4: SYNCING ALL RESULTS TO S3 ===")
        sync_output_to_s3(OUTPUT_DIR, S3_OUTPUT_PATH)
    
    print("\nFull pipeline completed successfully!")
    return True

def create_processing_job(job_name, script_name, config, instance_type="ml.g4dn.xlarge", instance_count=1):
    """Create a SageMaker Processing job to run a pipeline."""
    if not SAGEMAKER_AVAILABLE:
        print("SageMaker SDK not available. Cannot create processing job.")
        return None
    
    # Create a processor
    processor = ScriptProcessor(
        role=role,
        image_uri=f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker",
        instance_count=instance_count,
        instance_type=instance_type,
        base_job_name=job_name,
        sagemaker_session=session
    )
    
    # Convert config to command line arguments
    cmd_args = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd_args.append(f"--{key}")
        else:
            cmd_args.append(f"--{key}")
            cmd_args.append(str(value))
    
    # Create processing job
    processor.run(
        code=script_name,
        arguments=cmd_args,
        inputs=[
            ProcessingInput(
                source=S3_DATA_PATH,
                destination="/opt/ml/input/data"
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/model",
                destination=S3_OUTPUT_PATH
            )
        ],
        wait=False
    )
    
    print(f"Created SageMaker Processing job: {job_name}")
    return processor

def download_results_from_s3(s3_path, local_path):
    """Download results from S3."""
    if not s3_path or not bucket:
        print("S3 path not available. Skipping download.")
        return
    
    os.makedirs(local_path, exist_ok=True)
    cmd = f"aws s3 sync {s3_path} {local_path}"
    print(f"Downloading results from {s3_path} to {local_path}...")
    returncode, output = run_command(cmd)
    if returncode == 0:
        print("Results download completed successfully.")
    else:
        print(f"Results download failed with return code {returncode}.")
    return returncode == 0

def main():
    """Parse arguments and run the specified pipeline."""
    parser = argparse.ArgumentParser(description='Run WiFi Sensing Pipelines on SageMaker')
    parser.add_argument('--pipeline', type=str, required=True, 
                        choices=['pretraining', 'supervised', 'meta', 'all'],
                        help='Which pipeline to run')
    
    # Optional configuration overrides
    parser.add_argument('--mode', type=str, choices=['csi', 'acf'],
                        help='Data modality to use (csi or acf)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model for supervised learning')
    parser.add_argument('--pretrained-model', type=str,
                        help='Path to pretrained model for supervised learning')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone network for supervised learning')
    parser.add_argument('--config-file', type=str,
                        help='JSON configuration file to override defaults')
    
    # SageMaker specific options
    parser.add_argument('--create-job', action='store_true',
                        help='Create a SageMaker Processing job instead of running locally')
    parser.add_argument('--instance-type', type=str, default='ml.g4dn.xlarge',
                        help='SageMaker instance type for processing job')
    parser.add_argument('--download-results', action='store_true',
                        help='Download results from S3 after job completes')
    parser.add_argument('--local-results-dir', type=str, default='./results',
                        help='Local directory to download results to')
    parser.add_argument('--s3-data-path', type=str,
                        help='S3 path to dataset (overwrites default)')
    parser.add_argument('--s3-output-path', type=str,
                        help='S3 path for outputs (overwrites default)')
    
    args = parser.parse_args()
    
    # Override S3 paths if specified
    global S3_DATA_PATH, S3_OUTPUT_PATH
    if args.s3_data_path:
        S3_DATA_PATH = args.s3_data_path
    if args.s3_output_path:
        S3_OUTPUT_PATH = args.s3_output_path
    
    # Override configs with command line arguments
    if args.mode:
        PRETRAIN_CONFIG['mode'] = args.mode
        SUPERVISED_CONFIG['mode'] = args.mode
    
    if args.pretrained:
        SUPERVISED_CONFIG['pretrained'] = True
    
    if args.pretrained_model:
        SUPERVISED_CONFIG['pretrained_model'] = args.pretrained_model
        SUPERVISED_CONFIG['pretrained'] = True
    
    if args.freeze_backbone:
        SUPERVISED_CONFIG['freeze_backbone'] = True
    
    # Load configuration from file if specified
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_overrides = json.load(f)
            
            if 'pretraining' in config_overrides:
                PRETRAIN_CONFIG.update(config_overrides['pretraining'])
            
            if 'supervised' in config_overrides:
                SUPERVISED_CONFIG.update(config_overrides['supervised'])
            
            if 'meta' in config_overrides:
                META_CONFIG.update(config_overrides['meta'])
    
    # Either create a SageMaker job or run locally
    if args.create_job:
        if not SAGEMAKER_AVAILABLE:
            print("SageMaker SDK not available. Cannot create processing job.")
            return
        
        if args.pipeline == 'pretraining':
            create_processing_job(
                job_name="wifi-pretraining",
                script_name="pretrain.py",
                config=PRETRAIN_CONFIG,
                instance_type=args.instance_type
            )
        elif args.pipeline == 'supervised':
            create_processing_job(
                job_name="wifi-supervised",
                script_name="train_supervised.py",
                config=SUPERVISED_CONFIG,
                instance_type=args.instance_type
            )
        elif args.pipeline == 'meta':
            create_processing_job(
                job_name="wifi-meta-learning",
                script_name="meta_learning.py",
                config=META_CONFIG,
                instance_type=args.instance_type
            )
        elif args.pipeline == 'all':
            print("Creating three separate jobs for each pipeline...")
            create_processing_job(
                job_name="wifi-pretraining",
                script_name="pretrain.py",
                config=PRETRAIN_CONFIG,
                instance_type=args.instance_type
            )
            create_processing_job(
                job_name="wifi-supervised",
                script_name="train_supervised.py",
                config=SUPERVISED_CONFIG,
                instance_type=args.instance_type
            )
            create_processing_job(
                job_name="wifi-meta-learning",
                script_name="meta_learning.py",
                config=META_CONFIG,
                instance_type=args.instance_type
            )
        
        # Download results if requested
        if args.download_results and S3_OUTPUT_PATH:
            print("Waiting for jobs to complete before downloading results...")
            time.sleep(10)  # Allow time for job to start
            download_results_from_s3(S3_OUTPUT_PATH, args.local_results_dir)
    else:
        # Run the specified pipeline locally
        if args.pipeline == 'pretraining':
            run_pretraining(PRETRAIN_CONFIG)
        elif args.pipeline == 'supervised':
            run_supervised(SUPERVISED_CONFIG)
        elif args.pipeline == 'meta':
            run_meta_learning(META_CONFIG)
        elif args.pipeline == 'all':
            run_full_pipeline()

if __name__ == "__main__":
    main()
