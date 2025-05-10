#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Training Script - Train multiple model architectures in one training job

This script can be run in a SageMaker environment to train and evaluate multiple model architectures
on the same task.

SageMaker Training Job Parameters Guide:
-----------------------------------------
当通过SageMaker启动训练任务时，可以使用以下参数禁用调试器和源代码打包：
1. 主要参数：
   - disable_profiler=True  # 禁用SageMaker性能分析器
   - debugger_hook_config=False  # 禁用SageMaker调试钩子
   - source_dir=None  # 不使用源代码目录，而是直接上传脚本

2. 环境变量设置 (在 environment 参数中):
   - SMDEBUG_DISABLED: 'true'
   - SM_DISABLE_DEBUGGER: 'true'
   - SAGEMAKER_DISABLE_PROFILER: 'true'
   - SMPROFILER_DISABLED: 'true'
   - SAGEMAKER_DISABLE_SOURCEDIR: 'true'

3. 使用禁用调试器的SageMaker示例代码：
```python
import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()

estimator = PyTorch(
    entry_point='train_multi_model.py',
    role=role,
    framework_version='2.0.0',
    py_version='py310',
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    disable_profiler=True,  # 禁用性能分析器
    debugger_hook_config=False,  # 禁用调试钩子
    source_dir=None,  # 不打包源代码目录
    environment={
        'SMDEBUG_DISABLED': 'true',
        'SM_DISABLE_DEBUGGER': 'true',
        'SAGEMAKER_DISABLE_PROFILER': 'true',
        'SMPROFILER_DISABLED': 'true',
        'SAGEMAKER_DISABLE_SOURCEDIR': 'true',
    },
    hyperparameters={
        'task_name': 'TestTask',
        'epochs': 10,
    }
)

estimator.fit()
```
"""

# Import os module to ensure it's available for use in subsequent code
import os
import sys

# Disable SMDebug and Horovod to avoid PyTorch version conflicts
try:
    sys.modules['smdebug'] = None
    sys.modules['smddp'] = None
    sys.modules['smprofiler'] = None
    
    # Also disable Horovod
    sys.modules['horovod'] = None
    sys.modules['horovod.torch'] = None
    
    # 设置所有已知的环境变量来禁用调试器
    os.environ['SMDEBUG_DISABLED'] = 'true'
    os.environ['SM_DISABLE_DEBUGGER'] = 'true'
    os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER'] = 'true'
    os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER_OUTPUT'] = 'true'
    os.environ['SMPROFILER_DISABLED'] = 'true'
    os.environ['SM_SMDEBUG_DISABLED'] = 'true'
    os.environ['SM_SMDDP_DISABLE_PROFILING'] = 'true'
    os.environ['SAGEMAKER_DISABLE_PROFILER'] = 'true'
    
    # 禁用源代码包装和其他功能
    os.environ['SAGEMAKER_DISABLE_SOURCEDIR'] = 'true'
    os.environ['SAGEMAKER_CONTAINERS_IGNORE_SRC_REQUIREMENTS'] = 'true'
    os.environ['SAGEMAKER_DISABLE_BUILT_IN_PROFILER'] = 'true'
    os.environ['SAGEMAKER_DISABLE_DEFAULT_RULES'] = 'true'
    
    # 针对性禁用文件生成
    os.environ['SAGEMAKER_TRAINING_JOB_END_DISABLED'] = 'true'
    
    print("已完全禁用 SageMaker Debugger 和相关功能")
except Exception as e:
    print(f"警告：禁用模块时出错: {e}")

import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import math
import pandas as pd
import shutil

# Detect if running in SageMaker environment
is_sagemaker = 'SM_MODEL_DIR' in os.environ

# If running in SageMaker, import S3 tools
if is_sagemaker:
    import boto3
    s3_client = boto3.client('s3')
else:
    s3_client = None

# Print original command line arguments for diagnostic purposes
print("Original command line arguments:", sys.argv)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary models and data loaders
try:
    from load.supervised.benchmark_loader import load_benchmark_supervised
    # Import model classes
    from model.supervised.models import (
        MLPClassifier, 
        LSTMClassifier, 
        ResNet18Classifier, 
        TransformerClassifier, 
        ViTClassifier
    )
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

# Model factory dictionary
MODEL_TYPES = {
    'mlp': MLPClassifier,
    'lstm': LSTMClassifier,
    'resnet18': ResNet18Classifier,
    'transformer': TransformerClassifier,
    'vit': ViTClassifier
}

# Task trainer class (extracted from scripts/train_supervised.py)
from engine.supervised.task_trainer import TaskTrainer

def upload_to_s3(local_path, s3_path):
    """
    Upload a local file or directory to S3
    
    Args:
        local_path: Path to the local file or directory
        s3_path: S3 path, format: 's3://bucket-name/path/to/destination'
    
    Returns:
        bool: Whether the upload was successful
    """
    if not s3_client:
        logger.warning("S3 client not initialized, skipping upload")
        return False
    
    if not s3_path.startswith('s3://'):
        logger.error(f"Invalid S3 path: {s3_path}")
        return False
    
    try:
        # Parse S3 path
        s3_parts = s3_path.replace('s3://', '').split('/', 1)
        if len(s3_parts) != 2:
            logger.error(f"Invalid S3 path format: {s3_path}")
            return False
        
        bucket_name = s3_parts[0]
        s3_key_prefix = s3_parts[1]
        if not s3_key_prefix.endswith('/'):
            s3_key_prefix += '/'
            
        # Check if S3 bucket exists and is accessible
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Successfully connected to S3 bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"Cannot access S3 bucket {bucket_name}: {e}")
            return False
        
        logger.info(f"Uploading {local_path} to S3 bucket {bucket_name}/{s3_key_prefix}")
        
        # Check if it's a file or directory
        if os.path.isfile(local_path):
            # Upload single file
            file_key = os.path.join(s3_key_prefix, os.path.basename(local_path))
            logger.info(f"Uploading single file: {local_path} -> s3://{bucket_name}/{file_key}")
            
            # Try to upload, with retries on failure
            max_retries = 5  # Increased from 3 to 5
            for retry in range(max_retries):
                try:
                    # Add metadata for tracking
                    s3_client.upload_file(
                        local_path, 
                        bucket_name, 
                        file_key,
                        ExtraArgs={
                            'Metadata': {
                                'source': 'train_multi_model',
                                'timestamp': str(int(time.time()))
                            }
                        }
                    )
                    
                    # Verify upload by checking if file exists in S3
                    try:
                        s3_client.head_object(Bucket=bucket_name, Key=file_key)
                        logger.info(f"Successfully uploaded and verified file: s3://{bucket_name}/{file_key}")
                        break
                    except Exception as ve:
                        if retry < max_retries - 1:
                            logger.warning(f"Upload verification failed, retrying ({retry+1}/{max_retries}): {ve}")
                        else:
                            raise ve
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"File upload failed, retrying ({retry+1}/{max_retries}): {e}")
                        time.sleep(2 * (retry + 1))  # Exponential backoff
                    else:
                        logger.error(f"File upload failed after maximum retries: {e}")
                        return False
        else:
            # Upload entire directory
            total_files = sum([len(files) for _, _, files in os.walk(local_path)])
            logger.info(f"Preparing to upload directory with {total_files} files")
            
            # Track successful and failed uploads
            successful_uploads = []
            failed_uploads = []
            
            # Walk directory and upload all files
            for root, _, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    
                    # Calculate relative path to maintain directory structure
                    relative_path = os.path.relpath(local_file_path, local_path)
                    s3_key = os.path.join(s3_key_prefix, relative_path)
                    
                    # Upload file with retry logic
                    max_retries = 5
                    upload_successful = False
                    
                    for retry in range(max_retries):
                        try:
                            # Add metadata for tracking
                            s3_client.upload_file(
                                local_file_path, 
                                bucket_name, 
                                s3_key,
                                ExtraArgs={
                                    'Metadata': {
                                        'source': 'train_multi_model',
                                        'timestamp': str(int(time.time()))
                                    }
                                }
                            )
                            
                            # Verify upload
                            try:
                                s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                                successful_uploads.append(relative_path)
                                upload_successful = True
                                
                                # Report progress at reasonable intervals
                                if len(successful_uploads) % 10 == 0 or len(successful_uploads) == 1:
                                    logger.info(f"Upload progress: {len(successful_uploads)}/{total_files} files ({len(successful_uploads)/total_files*100:.1f}%)")
                                
                                break
                            except Exception as ve:
                                if retry < max_retries - 1:
                                    logger.warning(f"Upload verification failed for {s3_key}, retrying ({retry+1}/{max_retries}): {ve}")
                                else:
                                    raise ve
                        except Exception as e:
                            if retry < max_retries - 1:
                                logger.warning(f"File upload failed for {local_file_path}, retrying ({retry+1}/{max_retries}): {e}")
                                time.sleep(2 * (retry + 1))  # Exponential backoff
                            else:
                                logger.error(f"File upload failed for {local_file_path} after maximum retries: {e}")
                                failed_uploads.append(relative_path)
                    
                    if not upload_successful:
                        failed_uploads.append(relative_path)
            
            # Generate and upload manifest of successful uploads
            try:
                if successful_uploads:
                    manifest_content = "\n".join(successful_uploads)
                    manifest_path = os.path.join(os.path.dirname(local_path), "successful_uploads.txt")
                    with open(manifest_path, 'w') as f:
                        f.write(manifest_content)
                    
                    manifest_key = os.path.join(s3_key_prefix, "successful_uploads.txt")
                    s3_client.upload_file(manifest_path, bucket_name, manifest_key)
                    logger.info(f"Uploaded manifest of successful uploads to s3://{bucket_name}/{manifest_key}")
            except Exception as e:
                logger.warning(f"Failed to upload success manifest: {e}")
            
            # Generate and upload list of failed uploads if any
            try:
                if failed_uploads:
                    failures_content = "\n".join(failed_uploads)
                    failures_path = os.path.join(os.path.dirname(local_path), "failed_uploads.txt")
                    with open(failures_path, 'w') as f:
                        f.write(failures_content)
                    
                    failures_key = os.path.join(s3_key_prefix, "failed_uploads.txt")
                    s3_client.upload_file(failures_path, bucket_name, failures_key)
                    logger.info(f"Uploaded list of failed uploads to s3://{bucket_name}/{failures_key}")
            except Exception as e:
                logger.warning(f"Failed to upload failures list: {e}")
            
            # Report upload statistics
            logger.info(f"Upload summary: {len(successful_uploads)} successful, {len(failed_uploads)} failed out of {total_files} files")
            
            # Consider the upload failed if more than half of the files failed
            if len(failed_uploads) > total_files / 2:
                logger.error(f"Too many files failed to upload ({len(failed_uploads)}/{total_files}), upload considered unsuccessful")
                return False
            elif failed_uploads:
                logger.warning(f"Some files failed to upload, but most succeeded ({len(successful_uploads)}/{total_files})")
            else:
                logger.info(f"All {total_files} files uploaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error during S3 upload: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def cleanup_sagemaker_storage():
    """
    Clean up unnecessary files in SageMaker environment to reduce storage usage
    """
    if not is_sagemaker:
        # Only run in SageMaker environment
        return
    
    logger.info("Cleaning up unnecessary files to reduce storage usage...")
    
    try:
        # Delete unnecessary temporary files and logs
        dirs_to_clean = [
            "/tmp",                        # Temporary directory
            "/opt/ml/output/profiler",     # Profiler output
            "/opt/ml/output/tensors",      # Debugger tensors
            "/opt/ml/output/debug-output", # Debug output
            "/opt/ml/code",                # 源代码目录
            "/opt/ml/code/.sourcedir.tar.gz", # 源代码打包
            "/opt/ml/model",               # 模型目录（不需要）
        ]
        
        # Only keep the smallest log files
        log_files = [
            "/opt/ml/output/data/logs/algo-1-stdout.log",
            "/opt/ml/output/data/logs/algo-1-stderr.log"
        ]
        
        # Clean up temporary directories (but don't delete all)
        for cleanup_dir in dirs_to_clean:
            if os.path.exists(cleanup_dir):
                logger.info(f"Cleaning directory: {cleanup_dir}")
                # Read-only access to directory content, don't delete recursively
                try:
                    for item in os.listdir(cleanup_dir):
                        item_path = os.path.join(cleanup_dir, item)
                        if os.path.isdir(item_path) and not item.startswith('.'):
                            try:
                                shutil.rmtree(item_path)
                            except Exception as e:
                                logger.warning(f"Could not remove directory {item_path}: {e}")
                        elif os.path.isfile(item_path) and not item.startswith('.'):
                            try:
                                os.remove(item_path)
                            except Exception as e:
                                logger.warning(f"Could not remove file {item_path}: {e}")
                except Exception as e:
                    logger.warning(f"Error cleaning directory {cleanup_dir}: {e}")
        
        # Clean up log files (keep last 10KB)
        for log_file in log_files:
            if os.path.exists(log_file) and os.path.getsize(log_file) > 10240:
                try:
                    with open(log_file, 'rb') as f:
                        # Jump to 10KB before end of file
                        f.seek(-10240, 2)  # 2 means from end of file
                        last_10kb = f.read()
                    
                    # Rewrite log file, keep only last 10KB
                    with open(log_file, 'wb') as f:
                        f.write(b"[...previous logs truncated...]\n")
                        f.write(last_10kb)
                    
                    logger.info(f"Truncated log file: {log_file}")
                except Exception as e:
                    logger.warning(f"Could not truncate log file {log_file}: {e}")
        
        # Clean up sourcedir cache
        sourcedir_cache = "/opt/ml/code/.sourcedir.tar.gz"
        if os.path.exists(sourcedir_cache):
            try:
                os.remove(sourcedir_cache)
                logger.info("Removed sourcedir cache")
            except Exception as e:
                logger.warning(f"Could not remove sourcedir cache: {e}")
        
        # Try to trigger memory cleanup
        import gc
        gc.collect()
        
        # 尝试删除 training_job_end.ts 文件
        debug_output_file = "/opt/ml/output/debug-output/training_job_end.ts"
        if os.path.exists(debug_output_file):
            try:
                os.remove(debug_output_file)
                logger.info("删除了 training_job_end.ts 文件")
            except Exception as e:
                logger.warning(f"无法删除 training_job_end.ts 文件: {e}")
        
        logger.info("Storage cleanup completed!")
    except Exception as e:
        logger.error(f"Error during storage cleanup: {e}")

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="WiFi Sensing Multi-Model Training")
    
    # Task parameters
    parser.add_argument('--task_name', type=str, required=True, help='Task name to train')
    
    # Data related parameters
    parser.add_argument('--data_root', type=str, default='/opt/ml/input/data/training', help='Data root directory')
    parser.add_argument('--tasks_dir', type=str, default='tasks', help='Tasks directory')
    parser.add_argument('--data_key', type=str, default='data', help='Data key')
    parser.add_argument('--file_format', type=str, default='h5', choices=['h5', 'npz', 'pt'], help='Data file format')
    parser.add_argument('--use_root_data_path', action='store_true', default=True, help='Use root directory as data path')
    parser.add_argument('--adaptive_path', action='store_true', default=True, help='Adaptively search path')
    parser.add_argument('--try_all_paths', action='store_true', default=True, help='Try all possible data paths')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./saved_models', help='Output directory')
    parser.add_argument('--save_to_s3', type=str, default=None, help='S3 path for saving results (s3://bucket/path)')
    
    # Model parameters
    parser.add_argument('--models', type=str, default='mlp,lstm,resnet18,transformer', help='Models to train, comma separated')
    parser.add_argument('--win_len', type=int, default=500, help='Window length')
    parser.add_argument('--feature_size', type=int, default=232, help='Feature size')
    parser.add_argument('--in_channels', type=int, default=1, help='Input channels')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    
    # Transformer/ViT specific parameters
    parser.add_argument('--d_model', type=int, default=64, help='Transformer model dimension')
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Test parameters
    parser.add_argument('--test_splits', type=str, default='test_id,test_ood,test_cross_env', help='Test splits, comma separated')
    
    # Visualization parameters
    parser.add_argument('--save_plots', action='store_true', help='Save visualizations and plots')
    parser.add_argument('--save_confusion_matrix', action='store_true', help='Save confusion matrices')
    parser.add_argument('--save_learning_curves', action='store_true', help='Save learning curves')
    parser.add_argument('--save_predictions', action='store_true', help='Save model predictions')
    parser.add_argument('--save_model', action='store_true', help='Save trained model weights')
    parser.add_argument('--plot_dpi', type=int, default=150, help='DPI for saved plots')
    parser.add_argument('--plot_format', type=str, default='png', choices=['png', 'jpg', 'pdf', 'svg'], help='File format for plots')
    
    # S3 upload parameters
    parser.add_argument('--verify_uploads', action='store_true', help='Verify S3 uploads')
    parser.add_argument('--max_retries', type=int, default=5, help='Maximum retry attempts for S3 uploads')
    
    # Experiment parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Parse known arguments
    args, unknown = parser.parse_known_args()
    
    # Convert models list to list
    args.all_models = [m.strip() for m in args.models.split(',')]
    
    # Convert test splits to list
    if args.test_splits == 'all':
        args.test_splits = ['test_id', 'test_ood', 'test_cross_env']
    else:
        args.test_splits = [ts.strip() for ts in args.test_splits.split(',')]
    
    # Process legacy parameters from SM_HP environment variables
    for k, v in os.environ.items():
        # Check SM_HP_* format environment variables
        if k.startswith('SM_HP_'):
            # Convert parameter name
            param_name = k[6:].lower().replace('-', '_')
            
            # Handle learning rate alias (learning_rate vs lr)
            if param_name == 'learning_rate':
                param_name = 'lr'
            
            # Check if parameter exists
            if hasattr(args, param_name):
                # Convert value based on parameter type
                if isinstance(getattr(args, param_name), bool):
                    if v.lower() in ('true', 'yes', '1'):
                        setattr(args, param_name, True)
                    elif v.lower() in ('false', 'no', '0'):
                        setattr(args, param_name, False)
                elif isinstance(getattr(args, param_name), int):
                    try:
                        setattr(args, param_name, int(v))
                    except ValueError:
                        pass
                elif isinstance(getattr(args, param_name), float):
                    try:
                        setattr(args, param_name, float(v))
                    except ValueError:
                        pass
                else:
                    setattr(args, param_name, v)
                    
                # Special case - models list
                if param_name == 'models':
                    args.all_models = [m.strip() for m in v.split(',')]
                    
                # Special case - test splits
                if param_name == 'test_splits' and v != 'all':
                    args.test_splits = [ts.strip() for ts in v.split(',')]
        
        # Check S3 output path
        if k == 'SAGEMAKER_S3_OUTPUT' and args.save_to_s3 is None:
            args.save_to_s3 = v
            print(f"Setting S3 output path from environment variable: {v}")
    
    # Set correct output directory in SageMaker environment
    if is_sagemaker:
        args.output_dir = '/opt/ml/output/data'
        logger.info(f"Running in SageMaker environment, setting output_dir to {args.output_dir}")
        
        # In SageMaker, enable all visualization options by default
        args.save_plots = True
        args.save_confusion_matrix = True
        args.save_learning_curves = True
        args.save_model = True
        args.save_predictions = True
        args.verify_uploads = True
    
    # Create output directories if needed
    if is_sagemaker:
        os.makedirs(os.path.join(args.output_dir, args.task_name), exist_ok=True)
    
    # Print parameters
    print("\n===== Parameters =====")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("=====================\n")
    
    return args

def set_seed(seed):
    """Set all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(model_name, data, args, device):
    """Train a model of the specified type"""
    logger.info(f"===== Starting training for {model_name.upper()} model =====")
    
    # Unpack data
    loaders = data['loaders']
    num_classes = data['num_classes']
    label_mapper = data['label_mapper']
    
    # Get train and validation sets
    train_loader = loaders['train']
    val_loader = loaders.get('val')
    
    if val_loader is None:
        logger.warning("No validation data found. Using training data for validation.")
        val_loader = train_loader
    
    # Create model
    logger.info(f"Creating {model_name.upper()} model...")
    ModelClass = MODEL_TYPES[model_name.lower()]
    
    # Common model parameters
    model_kwargs = {
        'num_classes': num_classes,
        'in_channels': args.in_channels,
        'win_len': args.win_len,
        'feature_size': args.feature_size,
        'emb_dim': args.emb_dim,
        'dropout': args.dropout
    }
    
    # Model-specific parameters
    if model_name.lower() == 'transformer':
        model_kwargs['d_model'] = args.d_model
    
    # Create model instance
    model = ModelClass(**model_kwargs)
    model = model.to(device)
    
    logger.info(f"Model created: {model_name}")
    
    # Create experiment ID from timestamp and model name
    import hashlib
    timestamp = int(time.time())
    experiment_id = f"params_{hashlib.md5(f'{model_name}_{args.task_name}_{timestamp}'.encode()).hexdigest()[:8]}"
    
    # Create directory structure that matches local pipeline
    # /output_dir/task_name/model_name/experiment_id/
    results_dir = os.path.join(args.output_dir, args.task_name, model_name, experiment_id)
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Create config
    config = {
        'model': model_name,
        'task': args.task_name,
        'num_classes': num_classes,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'win_len': args.win_len,
        'feature_size': args.feature_size,
        'experiment_id': experiment_id
    }
    
    # Save configuration
    config_path = os.path.join(results_dir, f"{model_name}_{args.task_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Create optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create scheduler
    num_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def warmup_cosine_schedule(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
    
    # Get test loaders
    test_loaders = {k: v for k, v in loaders.items() if k.startswith('test')}
    if not test_loaders:
        logger.warning("No test splits found in the dataset. Check split names and dataset structure.")
    else:
        logger.info(f"Loaded {len(test_loaders)} test splits: {list(test_loaders.keys())}")
    
    # Create TaskTrainer
    trainer = TaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=results_dir,  # Use results_dir for plots, metrics, etc.
        num_classes=num_classes,
        config=config,
        label_mapper=label_mapper
    )
    
    # Train the model with early stopping
    trained_model, training_results = trainer.train()
    
    # Track best epoch
    best_epoch = training_results.get('best_epoch', args.epochs)
    
    # Save training history
    if 'training_dataframe' in training_results:
        history = training_results['training_dataframe']
        history_file = os.path.join(results_dir, f"{model_name}_{args.task_name}_train_history.csv")
        history.to_csv(history_file, index=False)
        logger.info(f"Saved training history to {history_file}")
    
    # Store overall metrics
    overall_metrics = {}
    
    # Run evaluation on each test set
    for test_name, test_loader in test_loaders.items():
        logger.info(f"Evaluating on {test_name}...")
        test_loss, test_accuracy = trainer.evaluate(test_loader)
        
        # Calculate metrics
        try:
            test_f1, classification_report = trainer.calculate_metrics(test_loader)
            
            # Save classification report
            report_file = os.path.join(results_dir, f"classification_report_{test_name}.csv")
            classification_report.to_csv(report_file)
            logger.info(f"Saved classification report for {test_name} to {report_file}")
            
            # Generate and save confusion matrix
            confusion_matrix = trainer.plot_confusion_matrix(test_loader, save_path=os.path.join(results_dir, f"{model_name}_{args.task_name}_{test_name}_confusion.png"))
            
        except Exception as e:
            logger.error(f"Error calculating additional metrics for {test_name}: {e}")
            test_f1 = 0.0
            
        # Store metrics
        overall_metrics[test_name] = {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'f1_score': test_f1
        }
        
        logger.info(f"{test_name} Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    
    # Generate and save learning curves
    try:
        if 'train_loss_history' in training_results and 'val_loss_history' in training_results:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, len(training_results['train_loss_history'])+1), training_results['train_loss_history'], label='Train Loss')
            plt.plot(range(1, len(training_results['val_loss_history'])+1), training_results['val_loss_history'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{model_name} on {args.task_name} - Loss Curves')
            plt.legend()
            plt.grid(True)
            learning_curve_file = os.path.join(results_dir, f"{model_name}_{args.task_name}_learning_curves.png")
            plt.savefig(learning_curve_file)
            plt.close()
            logger.info(f"Saved learning curves to {learning_curve_file}")
            
            # Accuracy curves
            if 'train_accuracy_history' in training_results and 'val_accuracy_history' in training_results:
                plt.figure(figsize=(10, 5))
                plt.plot(range(1, len(training_results['train_accuracy_history'])+1), training_results['train_accuracy_history'], label='Train Accuracy')
                plt.plot(range(1, len(training_results['val_accuracy_history'])+1), training_results['val_accuracy_history'], label='Val Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title(f'{model_name} on {args.task_name} - Accuracy Curves')
                plt.legend()
                plt.grid(True)
                accuracy_curve_file = os.path.join(results_dir, f"{model_name}_{args.task_name}_accuracy_curves.png")
                plt.savefig(accuracy_curve_file)
                plt.close()
                logger.info(f"Saved accuracy curves to {accuracy_curve_file}")
    except Exception as e:
        logger.error(f"Error generating learning curves: {e}")
    
    # Save test results
    results_file = os.path.join(results_dir, f"{model_name}_{args.task_name}_results.json")
    
    # Make sure results are JSON serializable
    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        else:
            return obj
    
    # Process all metrics
    serializable_metrics = {}
    for key, value in overall_metrics.items():
        serializable_metrics[key] = {k: convert_to_json_serializable(v) for k, v in value.items()}
    
    with open(results_file, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
    
    # Save summary
    try:
        summary = {
            'best_epoch': best_epoch,
            'experiment_id': experiment_id,
            'experiment_completed': True
        }
        
        # Add val metrics if available
        if 'val_accuracy_history' in training_results and len(training_results['val_accuracy_history']) > 0:
            best_idx = best_epoch - 1 if best_epoch > 0 else 0
            if best_idx < len(training_results['val_accuracy_history']):
                summary['best_val_accuracy'] = training_results['val_accuracy_history'][best_idx]
                
            if 'val_loss_history' in training_results and best_idx < len(training_results['val_loss_history']):
                summary['best_val_loss'] = training_results['val_loss_history'][best_idx]
        
        # Add test results to summary
        for split_name, metrics in serializable_metrics.items():
            summary[f'{split_name}_accuracy'] = metrics['accuracy']
            summary[f'{split_name}_f1_score'] = metrics['f1_score']
        
        summary_file = os.path.join(results_dir, f"{model_name}_{args.task_name}_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Update best performance tracking
        model_dir = os.path.dirname(results_dir)
        best_performance_file = os.path.join(model_dir, "best_performance.json")
        
        # Check if there's an existing best performance file
        best_performance = {}
        if os.path.exists(best_performance_file):
            try:
                with open(best_performance_file, 'r') as f:
                    best_performance = json.load(f)
            except:
                pass
        
        # Calculate average test accuracy to determine if this is the best run
        avg_test_accuracy = sum([metrics['accuracy'] for metrics in serializable_metrics.values()]) / len(serializable_metrics)
        
        # Update best performance if this run is better
        if not best_performance or avg_test_accuracy > best_performance.get('avg_test_accuracy', 0):
            best_performance = {
                'experiment_id': experiment_id,
                'avg_test_accuracy': avg_test_accuracy,
                'best_epoch': best_epoch,
                'timestamp': time.time(),
                'test_metrics': serializable_metrics,
                'experiment_path': results_dir
            }
            
            with open(best_performance_file, 'w') as f:
                json.dump(best_performance, f, indent=4)
                
            logger.info(f"Updated best performance record for {model_name} on {args.task_name}")
            
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
    
    # Save overall test accuracy
    if len(serializable_metrics) > 0:
        all_test_accuracies = [metrics.get('accuracy', 0) for metrics in serializable_metrics.values()]
        overall_metrics['test_accuracy'] = sum(all_test_accuracies) / len(all_test_accuracies)
    
    logger.info(f"Training and evaluation completed. Results saved to {results_dir}")
    
    return trained_model, overall_metrics

def main():
    """
    Main function - Train multiple models on a specified task
    """
    try:
        # Disable SageMaker debugger and profiler
        os.environ['SMDEBUG_DISABLED'] = 'true'
        os.environ['SM_DISABLE_DEBUGGER'] = 'true'
        os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER'] = 'true'
        os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER_OUTPUT'] = 'true'
        os.environ['SMPROFILER_DISABLED'] = 'true'
        os.environ['SAGEMAKER_DISABLE_SOURCEDIR'] = 'true'  # Disable sourcedir packaging
        os.environ['SAGEMAKER_CONTAINERS_IGNORE_SRC_REQUIREMENTS'] = 'true'
        os.environ['SAGEMAKER_DISABLE_BUILT_IN_PROFILER'] = 'true'
        os.environ['SAGEMAKER_DISABLE_DEFAULT_RULES'] = 'true'
        os.environ['SAGEMAKER_TRAINING_JOB_END_DISABLED'] = 'true'
        
        # Log environment variables for debugging
        if is_sagemaker:
            logger.info("Running in SageMaker environment")
            logger.info("Environment variables:")
            for key in sorted([k for k in os.environ.keys() if k.startswith(('SM_', 'SAGEMAKER_'))]):
                logger.info(f"  {key}: {os.environ.get(key)}")

        # Get command line arguments
        args = get_args()
        
        # Log parsed arguments
        logger.info("Parsed arguments:")
        for arg_name, arg_value in sorted(vars(args).items()):
            logger.info(f"  {arg_name}: {arg_value}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Set random seed
        set_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
        
        # Log training start
        logger.info(f"Starting multi-model training, task: {args.task_name}")
        logger.info(f"Models to train: {args.all_models}")
        
        # Check for S3 output path in environment variables if not specified
        if args.save_to_s3 is None and is_sagemaker:
            sm_output_s3 = os.environ.get('SAGEMAKER_S3_OUTPUT')
            if sm_output_s3:
                logger.info(f"Found S3 output path in environment variables: {sm_output_s3}")
                args.save_to_s3 = sm_output_s3
        
        if args.save_to_s3:
            logger.info(f"Results will be uploaded to S3: {args.save_to_s3}")
        
        # Set data root directory - directly use SM_CHANNEL_TRAINING in SageMaker
        dataset_root = '/opt/ml/input/data/training' if is_sagemaker else args.data_root
        logger.info(f"Using data root directory: {dataset_root}")
        
        # Load data
        logger.info(f"Loading data from {dataset_root}, task name: {args.task_name}")
        data = load_benchmark_supervised(
            dataset_root=dataset_root,
            task_name=args.task_name,
            batch_size=args.batch_size,
            data_key=args.data_key,
            file_format=args.file_format,
            num_workers=args.num_workers,
            use_root_as_task_dir=args.use_root_data_path
        )
        
        # Check if data loaded successfully
        if not data or 'loaders' not in data:
            logger.error(f"Failed to load data for task {args.task_name}")
            sys.exit(1)
        
        logger.info(f"Data loaded successfully. Number of classes: {data['num_classes']}")
        logger.info(f"Available data loaders: {list(data['loaders'].keys())}")
        
        # Track model results
        successful_models = []
        failed_models = []
        
        # Store all results
        all_results = {}
        
        # Train each model
        for model_name in args.all_models:
            try:
                logger.info(f"\n{'='*40}\nTraining model: {model_name}\n{'='*40}")
                
                # Verify model compatibility
                try:
                    ModelClass = MODEL_TYPES[model_name.lower()]
                    logger.info(f"Model class {model_name} loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model class {model_name}: {e}")
                    failed_models.append((model_name, f"Model class error: {str(e)}"))
                    continue
                
                # Train model
                model, metrics = train_model(model_name, data, args, device)
                
                # Check if training succeeded
                if model is None or (isinstance(metrics, dict) and 'error' in metrics):
                    error_msg = metrics.get('error', 'Unknown error') if isinstance(metrics, dict) else 'Unknown error'
                    logger.error(f"Model {model_name} training failed: {error_msg}")
                    failed_models.append((model_name, error_msg))
                else:
                    all_results[model_name] = metrics
                    successful_models.append(model_name)
                    logger.info(f"Completed training for {model_name}")
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                failed_models.append((model_name, str(e)))
        
        # Print training summary
        logger.info("\n" + "="*60)
        logger.info("Training Summary")
        logger.info("="*60)
        logger.info(f"Task: {args.task_name}")
        logger.info(f"Successfully trained models ({len(successful_models)}): {', '.join(successful_models)}")
        logger.info(f"Failed models ({len(failed_models)}): {', '.join([m[0] for m in failed_models])}")
        
        if failed_models:
            logger.info("\nFailure details:")
            for model_name, error in failed_models:
                logger.info(f"  - {model_name}: {error}")
        
        # Save overall results summary
        results_path = os.path.join(args.output_dir, args.task_name, "multi_model_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        logger.info(f"All training completed. Results saved to {results_path}")
        logger.info("Results summary:")
        for model_name, metrics in all_results.items():
            logger.info(f"  - {model_name}: Test accuracy = {metrics.get('test_accuracy', 0.0):.4f}")
        
        # Upload results to S3 if running in SageMaker
        if is_sagemaker and args.save_to_s3:
            logger.info(f"Uploading results to S3: {args.save_to_s3}")
            
            # Ensure we use the complete output directory path
            output_dir = args.output_dir  # /opt/ml/output/data
            logger.info(f"Directory structure to upload:")
            
            # List contents of output directory
            if os.path.exists(output_dir):
                total_files = 0
                file_sizes = 0
                
                # Log directory structure for debugging
                for root, dirs, files in os.walk(output_dir):
                    rel_path = os.path.relpath(root, output_dir)
                    if rel_path == '.':
                        logger.info(f"Root directory: {output_dir}")
                    else:
                        logger.info(f"  Directory: {rel_path}/")
                    
                    # Skip logging too many files
                    if len(files) > 0:
                        logger.info(f"    Files ({len(files)} files):")
                        
                    for name in files[:10]:  # Only show first 10 files in each directory
                        total_files += 1
                        file_path = os.path.join(root, name)
                        file_sizes += os.path.getsize(file_path)
                        rel_path = os.path.relpath(file_path, output_dir)
                        logger.info(f"      - {rel_path} ({os.path.getsize(file_path)/1024:.1f} KB)")
                    
                    if len(files) > 10:
                        logger.info(f"      ... and {len(files) - 10} more files")
                
                logger.info(f"Total: {total_files} files, {file_sizes/1024/1024:.2f} MB")
                
                # Construct S3 destination path
                s3_output_path = args.save_to_s3.rstrip('/')
                logger.info(f"Uploading entire output directory: {output_dir} -> {s3_output_path}")
                
                # Upload directory - retry on failure
                try:
                    # First, create a manifest file listing all generated files
                    manifest_file = os.path.join(output_dir, "upload_manifest.txt")
                    with open(manifest_file, 'w') as f:
                        for root, _, files in os.walk(output_dir):
                            for file in files:
                                if file != "upload_manifest.txt":  # Skip the manifest itself
                                    rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                                    f.write(f"{rel_path}\n")
                
                    logger.info(f"Created upload manifest at {manifest_file}")
                    
                    # Upload each task directory separately to maintain structure
                    task_dir = os.path.join(output_dir, args.task_name)
                    if os.path.exists(task_dir):
                        task_s3_path = f"{s3_output_path}/{args.task_name}"
                        logger.info(f"Uploading task directory: {task_dir} -> {task_s3_path}")
                        
                        # Upload with verification
                        upload_success = upload_to_s3(task_dir, task_s3_path)
                        
                        if not upload_success:
                            logger.warning("First upload attempt failed. Retrying once...")
                            time.sleep(5)  # Brief delay before retry
                            upload_success = upload_to_s3(task_dir, task_s3_path)
                        
                        if upload_success:
                            logger.info(f"Successfully uploaded task results to {task_s3_path}")
                        else:
                            logger.error(f"Failed to upload task results to {task_s3_path} after retry")
                    else:
                        logger.error(f"Task directory {task_dir} does not exist. Cannot upload task results.")
                    
                    # Also upload the summary file
                    summary_file = os.path.join(output_dir, args.task_name, "multi_model_results.json")
                    if os.path.exists(summary_file):
                        summary_s3_key = f"{args.task_name}/multi_model_results.json"
                        summary_s3_path = f"{s3_output_path}/{summary_s3_key}"
                        
                        # Extract bucket and key
                        bucket_name = s3_output_path.replace('s3://', '').split('/')[0]
                        s3_key = f"{'/'.join(s3_output_path.replace('s3://', '').split('/')[1:])}/{summary_s3_key}"
                        
                        try:
                            s3_client.upload_file(summary_file, bucket_name, s3_key)
                            logger.info(f"Uploaded summary file to {summary_s3_path}")
                        except Exception as e:
                            logger.error(f"Failed to upload summary file: {e}")
                    
                    # Upload manifest file
                    try:
                        manifest_s3_key = "upload_manifest.txt"
                        bucket_name = s3_output_path.replace('s3://', '').split('/')[0]
                        s3_key = f"{'/'.join(s3_output_path.replace('s3://', '').split('/')[1:])}/{manifest_s3_key}"
                        
                        s3_client.upload_file(manifest_file, bucket_name, s3_key)
                        logger.info(f"Uploaded manifest file to {s3_output_path}/{manifest_s3_key}")
                    except Exception as e:
                        logger.error(f"Failed to upload manifest file: {e}")
                    
                except Exception as e:
                    logger.error(f"Error during upload process: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.error(f"Output directory {output_dir} does not exist. Cannot upload results.")
        
        # Clean up SageMaker storage
        if is_sagemaker:
            cleanup_sagemaker_storage()
        
        logger.info("Multi-model training completed successfully!")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 