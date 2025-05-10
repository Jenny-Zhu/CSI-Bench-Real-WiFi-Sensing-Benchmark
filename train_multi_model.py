#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Training Script - Train multiple model architectures in one training job

This script can be run in a SageMaker environment to train and evaluate multiple model architectures
on the same task.
"""

# Import os module to ensure it's available for use in subsequent code
import os
import sys

# Disable SMDebug and Horovod to avoid PyTorch version conflicts
try:
    sys.modules['smdebug'] = None
    os.environ['SMDEBUG_DISABLED'] = 'true'
    os.environ['SM_DISABLE_DEBUGGER'] = 'true'
    
    # Also disable Horovod
    sys.modules['horovod'] = None
    sys.modules['horovod.torch'] = None
    
    print("Disabled SMDebug and Horovod to avoid conflicts")
except Exception as e:
    print(f"Warning when disabling modules: {e}")

import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import logging

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
            
        # 检查S3存储桶是否存在并且可访问
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"成功连接到S3存储桶: {bucket_name}")
        except Exception as e:
            logger.error(f"无法访问S3存储桶 {bucket_name}: {e}")
            return False
        
        logger.info(f"正在上传 {local_path} 到 S3 存储桶 {bucket_name}/{s3_key_prefix}")
        
        # 检查是文件还是目录
        if os.path.isfile(local_path):
            # 上传单个文件
            file_key = os.path.join(s3_key_prefix, os.path.basename(local_path))
            logger.info(f"上传单个文件: {local_path} -> s3://{bucket_name}/{file_key}")
            
            file_size = os.path.getsize(local_path)
            logger.info(f"文件大小: {file_size / (1024*1024):.2f} MB")
            
            # 尝试上传，并在失败时重试
            max_retries = 3
            for retry in range(max_retries):
                try:
                    s3_client.upload_file(local_path, bucket_name, file_key)
                    logger.info(f"成功上传文件到 s3://{bucket_name}/{file_key}")
                    
                    # 验证文件是否成功上传
                    try:
                        result = s3_client.head_object(Bucket=bucket_name, Key=file_key)
                        s3_file_size = result['ContentLength']
                        if s3_file_size == file_size:
                            logger.info(f"验证成功: 文件大小匹配 ({s3_file_size} bytes)")
                        else:
                            logger.warning(f"验证警告: S3中的文件大小 ({s3_file_size} bytes) 与本地不一致 ({file_size} bytes)")
                    except Exception as e:
                        logger.warning(f"无法验证上传的文件: {e}")
                    
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"文件上传失败，重试中 ({retry+1}/{max_retries}): {e}")
                        time.sleep(1)  # 短暂延迟后重试
                    else:
                        logger.error(f"文件上传失败，已达到最大重试次数: {e}")
                        return False
        else:
            # 上传整个目录
            total_files = sum([len(files) for _, _, files in os.walk(local_path)])
            logger.info(f"准备上传目录，共 {total_files} 个文件")
            
            # 创建进度条字符
            progress_chars = "■□"
            progress_length = 30
            
            # 遍历目录上传所有文件
            uploaded_files = 0
            failed_files = 0
            
            for root, _, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    
                    # 计算相对路径
                    relative_path = os.path.relpath(local_file_path, local_path)
                    s3_key = os.path.join(s3_key_prefix, relative_path)
                    
                    # 跳过临时文件
                    if file.startswith('.') or file.endswith('.tmp') or file.endswith('.bak'):
                        logger.info(f"跳过临时文件: {local_file_path}")
                        continue
                        
                    # 上传文件，带有重试逻辑
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            s3_client.upload_file(local_file_path, bucket_name, s3_key)
                            uploaded_files += 1
                            
                            # 每上传5个文件或最后一个文件时，输出进度信息
                            if uploaded_files % 5 == 0 or uploaded_files + failed_files == total_files:
                                progress = (uploaded_files + failed_files) / total_files
                                progress_bar = progress_chars[0] * int(progress_length * progress) + progress_chars[1] * (progress_length - int(progress_length * progress))
                                logger.info(f"上传进度: [{progress_bar}] {uploaded_files}/{total_files} 文件 ({progress*100:.1f}%)")
                            break
                        except Exception as e:
                            if retry < max_retries - 1:
                                logger.warning(f"文件上传失败 {local_file_path}，重试中 ({retry+1}/{max_retries}): {e}")
                                time.sleep(1)  # 短暂延迟后重试
                            else:
                                logger.error(f"文件上传失败 {local_file_path}，已达到最大重试次数: {e}")
                                failed_files += 1
            
            # 报告最终状态
            if failed_files > 0:
                logger.warning(f"目录上传完成，但有 {failed_files}/{total_files} 个文件上传失败")
                if failed_files > total_files / 2:  # 如果超过一半文件失败，返回失败
                    return False
            else:
                logger.info(f"成功上传目录内容到 s3://{bucket_name}/{s3_key_prefix}，共 {uploaded_files} 个文件")
                
                # 验证上传结果
                try:
                    # 检查是否至少有一个对象存在
                    response = s3_client.list_objects_v2(
                        Bucket=bucket_name,
                        Prefix=s3_key_prefix,
                        MaxKeys=1
                    )
                    if 'Contents' in response and len(response['Contents']) > 0:
                        logger.info(f"验证成功: S3目标路径 {s3_key_prefix} 中存在对象")
                    else:
                        logger.warning(f"验证警告: S3目标路径 {s3_key_prefix} 似乎为空")
                except Exception as e:
                    logger.warning(f"无法验证S3上传结果: {e}")
        
        return True
    except Exception as e:
        logger.error(f"S3上传过程中发生错误: {e}")
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
        # Completely disable profiler and debugger
        # Set environment variables to disable them
        os.environ['SMDEBUG_DISABLED'] = 'true'
        os.environ['SM_DISABLE_DEBUGGER'] = 'true'
        os.environ['SM_DISABLE_PROFILER'] = 'true'
        
        # Delete unnecessary temporary files and logs
        dirs_to_clean = [
            "/tmp",                        # Temporary directory
            "/opt/ml/output/profiler",     # Profiler output
            "/opt/ml/output/tensors",      # Debugger tensors
        ]
        
        # Only keep the smallest log files
        log_files = [
            "/opt/ml/output/data/logs/algo-1-stdout.log",
            "/opt/ml/output/data/logs/algo-1-stderr.log"
        ]
        
        # Clean up temporary directories (but don't delete all)
        import shutil
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
                        f.write(last_10kb)
                        
                    logger.info(f"Truncated log file {log_file} to last 10KB")
                except Exception as e:
                    logger.warning(f"Could not truncate log file {log_file}: {e}")
                    
        # Ensure all model files and outputs are included in the model directory
        model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
        
        # Make sure task results are in model directory
        if args.output_dir and args.task_name and os.path.exists(os.path.join(args.output_dir, args.task_name)):
            task_dir = os.path.join(args.output_dir, args.task_name)
            model_task_dir = os.path.join(model_dir, args.task_name)
            
            # Create task directory in model_dir if it doesn't exist
            os.makedirs(model_task_dir, exist_ok=True)
            
            # Copy all task results to model directory
            logger.info(f"Copying task results from {task_dir} to {model_task_dir}")
            try:
                # Use robocopy-like approach
                for root, dirs, files in os.walk(task_dir):
                    # Create corresponding directories in model_dir
                    rel_dir = os.path.relpath(root, task_dir)
                    target_dir = os.path.join(model_task_dir, rel_dir) if rel_dir != '.' else model_task_dir
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # Copy files
                    for file in files:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(target_dir, file)
                        try:
                            shutil.copy2(src_file, dst_file)
                        except Exception as e:
                            logger.warning(f"Could not copy file {src_file} to {dst_file}: {e}")
                
                logger.info(f"Successfully copied all task results to model directory")
            except Exception as e:
                logger.error(f"Error copying task results to model directory: {e}")
        
        # Also copy the multi_model_results.json file to the model directory root
        results_json = os.path.join(args.output_dir, args.task_name, "multi_model_results.json")
        if os.path.exists(results_json):
            try:
                shutil.copy2(results_json, os.path.join(model_dir, "multi_model_results.json"))
                logger.info(f"Copied multi_model_results.json to model directory root")
            except Exception as e:
                logger.error(f"Error copying multi_model_results.json: {e}")
        
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train multiple models on WiFi benchmark dataset')
    
    # Required arguments
    parser.add_argument('--models', type=str, default='vit', 
                        help='Comma-separated list of models to train. E.g. "mlp,lstm,resnet18"')
    parser.add_argument('--dataset_root', type=str, default='wifi_benchmark_dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                        help='Name of the task to train on')
    
    # Also accept task-name with dash (SageMaker hyperparameters usually use dashes)
    parser.add_argument('--task-name', type=str, dest='task_name', default=None,
                        help='Name of the task to train on (dash version for SageMaker compatibility)')
    
    # Data parameters
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                        help='Key for CSI data in h5 files')
    parser.add_argument('--file_format', type=str, default='h5',
                        help='Format of the data files (h5, tfrecord, etc.)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    
    # Model parameters
    parser.add_argument('--win_len', type=int, default=250, 
                        help='Window length for WiFi CSI data')
    parser.add_argument('--feature_size', type=int, default=98, 
                        help='Feature size for WiFi CSI data')
    parser.add_argument('--in_channels', type=int, default=1, 
                        help='Number of input channels')
    parser.add_argument('--emb_dim', type=int, default=128, 
                        help='Embedding dimension for ViT model')
    parser.add_argument('--d_model', type=int, default=256, 
                        help='Model dimension for Transformer model')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test_splits', type=str, default='all', 
                        help='Test splits to evaluate on, comma-separated (e.g., "test_id,test_cross_env") or "all"')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='/opt/ml/model',
                        help='Directory to save checkpoints and models')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (defaults to save_dir if not specified)')
    
    # Add backward compatibility for data_dir parameter
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory of the dataset (deprecated, use dataset_root instead)')
    
    # Debug parameters - All boolean flag parameters
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Enable detailed debug output')
    parser.add_argument('--adaptive_path', action='store_true', default=False,
                        help='Automatically adapt to data path structure')
    parser.add_argument('--try_all_paths', action='store_true', default=False,
                        help='Try all possible path combinations')
    parser.add_argument('--direct_upload', action='store_true', default=True,
                        help='Directly upload results to S3 without using SageMaker auto-packaging')
    parser.add_argument('--upload_final_model', action='store_true', default=False,
                        help='Upload final model to S3')
    parser.add_argument('--skip_train_for_debug', action='store_true', default=False,
                     help='Only for debugging, skip actual training process')
    
    # 添加use_root_data_path参数，直接使用整个/opt/ml/input/data/training目录作为有效数据
    parser.add_argument('--use_root_data_path', action='store_true', default=True,
                        help='直接使用整个数据根目录作为任务目录')
    parser.add_argument('--use-root-data-path', action='store_true', dest='use_root_data_path', default=True,
                        help='直接使用整个数据根目录作为任务目录（短横线格式）')
    
    # Add S3 related parameters
    parser.add_argument('--save_to_s3', type=str, default=None,
                      help='S3 path for saving results, format: s3://bucket-name/path/')

    # Modify parsing logic, handle SageMaker passed non-standard parameters
    # First get original parameters, do preliminary processing
    args_to_parse = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # Fix double underscore prefix first - this is the most critical issue
        if arg.startswith('__'):
            arg = '--' + arg[2:]
            logger.warning(f"CRITICAL FIX: Replaced double underscore prefix with double dash: {sys.argv[i]} -> {arg}")
        
        # Fix parameter name format - Convert dash-separated parameters to underscore-separated format
        # But keep the -- prefix intact!
        if arg.startswith('--'):
            # Extract the parameter name without the prefix
            param_name = arg[2:]
            # Replace dash with underscore in the parameter name only
            fixed_param_name = param_name.replace('-', '_')
            # Re-add the proper prefix
            fixed_arg = f"--{fixed_param_name}"
            
            if fixed_arg != arg:
                logger.info(f"Fixed parameter format: {arg} -> {fixed_arg}")
                arg = fixed_arg
        
        # Ensure arg starts with -- (not __ or other prefix)
        if not arg.startswith('--') and arg[0] == '-':
            arg = f"--{arg[1:]}"
            logger.info(f"Restored proper argument prefix: {arg}")
        
        # Handle flag parameters followed by True or False
        if arg.startswith('--') and i + 1 < len(sys.argv):
            next_arg = sys.argv[i+1]
            if next_arg.lower() == 'true':
                # For --flag True case, only keep --flag
                args_to_parse.append(arg)
                i += 2
                continue
            elif next_arg.lower() == 'false':
                # For --flag False case, skip that parameter
                i += 2
                continue
            
        # Normal parameter addition
        args_to_parse.append(arg)
        i += 1
    
    try:
        # Log the final arguments for debugging
        logger.info(f"Actual arguments to parse: {args_to_parse}")
        
        # Final sanity check to ensure no __ prefixes
        for i, arg in enumerate(args_to_parse):
            if arg.startswith('__'):
                args_to_parse[i] = '--' + arg[2:]
                logger.warning(f"CRITICAL FIX: Found double underscore prefix after first pass: {arg} -> {args_to_parse[i]}")
        
        # Use preprocessed parameters for parsing
        args = parser.parse_args(args_to_parse)
        
        # Print actual parsed parameters (for debugging)
        logger.info(f"Parsed arguments: {args_to_parse}")
        
        # Check if we need to get task_name from environment variables
        # This is a critical parameter that might be missing from command line
        if args.task_name in (None, 'MotionSourceRecognition'):
            # Try to get from SM_HP_TASK_NAME or SM_HP_TASK-NAME environment variable
            env_task_name = os.environ.get('SM_HP_TASK_NAME') or os.environ.get('SM_HP_TASK-NAME')
            if env_task_name:
                args.task_name = env_task_name
                logger.info(f"Got task_name from environment variable: {args.task_name}")
        
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        logger.error(f"Original arguments: {sys.argv}")
        logger.error(f"Processed arguments: {args_to_parse}")
        
        # Try using original parameters parsing, ignore errors
        try:
            args = parser.parse_args()
        except:
            # Last fallback: Use default parameters
            logger.warning("Using default arguments due to parsing failure")
            args = parser.parse_args([])
    
    # If debug mode is enabled, set log level to DEBUG
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - verbose logging activated")
        # Print all parameters
        logger.debug("All command line arguments:")
        for arg, value in sorted(vars(args).items()):
            logger.debug(f"  {arg}: {value}")
    
    # For backward compatibility: if data_dir is provided but dataset_root is not, use data_dir
    if args.data_dir is not None and args.dataset_root == 'wifi_benchmark_dataset':
        logger.warning("Using data_dir instead of dataset_root (data_dir is deprecated)")
        args.dataset_root = args.data_dir
    
    # Parse all models to train
    if ',' in args.models:
        args.all_models = args.models.split(',')
    else:
        args.all_models = [args.models]
    
    # Validate model validity
    for model_name in args.all_models:
        if model_name.lower() not in MODEL_TYPES:
            logger.error(f"Unsupported model: {model_name}. Valid models: {list(MODEL_TYPES.keys())}")
            sys.exit(1)
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.save_dir
        
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
    
    # Model-specific parameters - Add specific parameters based on model type
    if model_name.lower() == 'transformer':
        model_kwargs['d_model'] = args.d_model
    
    # Create model instance
    model = ModelClass(**model_kwargs)
    model = model.to(device)
    
    logger.info(f"Model created: {model_name}")
    
    # Set base output directory - These directories will be added with experiment_id
    if is_sagemaker:
        # For SageMaker, use /opt/ml/model as the base path for saving models
        model_base_dir = '/opt/ml/model'
    else:
        model_base_dir = args.save_dir
    
    # Create model type directory, without experiment_id
    model_save_dir = os.path.join(model_base_dir, args.task_name, model_name)
    model_output_dir = os.path.join(args.output_dir, args.task_name, model_name)
    
    # Ensure directories exist
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Create config object
    config = argparse.Namespace(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        num_classes=num_classes,
        device=str(device),
        save_dir=model_save_dir,  # This is the base directory without experiment_id
        output_dir=model_output_dir,  # This is the base directory without experiment_id
        results_subdir='supervised',
        model_name=model_name,
        task_name=args.task_name,
        # Later, when adding experiment_id, config_dict will update these paths
    )
    
    # Don't save config yet, wait until after experiment_id is created

    # Create experiment ID from timestamp and model name
    import hashlib
    timestamp = int(time.time())
    experiment_id = f"params_{hashlib.md5(f'{model_name}_{args.task_name}_{timestamp}'.encode()).hexdigest()[:8]}"
    
    # Update output directory, ensure it includes experiment_id
    model_output_dir = os.path.join(args.output_dir, args.task_name, model_name, experiment_id)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Now save config to experiment directory
    config_path = os.path.join(model_output_dir, "supervised_config.json")
    with open(config_path, "w") as f:
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        # Update paths to reflect new location
        config_dict['output_dir'] = model_output_dir
        config_dict['save_dir'] = os.path.join(args.save_dir, args.task_name, model_name, experiment_id)
        config_dict['experiment_id'] = experiment_id
        json.dump(config_dict, f, indent=4)
    
    logger.info(f"Config saved to model directory: {config_path}")
    
    # Create trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    trainer = TaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=model_save_dir,
        num_classes=num_classes,
        label_mapper=label_mapper,
        config=config
    )
    
    # Train model
    trained_model, training_results = trainer.train()
    
    # Evaluate on test sets
    logger.info("Evaluating on test sets:")
    all_test_loaders = {k: v for k, v in loaders.items() if k.startswith('test')}
    
    # Filter test loaders based on test_splits parameter
    if args.test_splits.lower() != 'all':
        # Split the comma-separated string and create a list of test split names
        requested_splits = [split.strip() for split in args.test_splits.split(',')]
        test_loaders = {}
        for split_name in requested_splits:
            # Ensure the split name starts with 'test'
            if not split_name.startswith('test'):
                split_name = f"test_{split_name}"
            
            # Add the loader if it exists
            if split_name in all_test_loaders:
                test_loaders[split_name] = all_test_loaders[split_name]
            else:
                logger.warning(f"Requested test split '{split_name}' not found in available splits: {list(all_test_loaders.keys())}")
        
        if not test_loaders:
            logger.warning(f"None of the requested test splits were found. Using all available test splits instead.")
            test_loaders = all_test_loaders
    else:
        # Use all available test loaders
        test_loaders = all_test_loaders
    
    logger.info(f"Using test splits: {list(test_loaders.keys())}")
    
    overall_metrics = {
        'model_name': model_name,
        'task_name': args.task_name
    }
    
    # Convert training_results to standard format if needed
    import pandas as pd
    if isinstance(training_results, pd.DataFrame):
        train_history = {
            'epochs': training_results['Epoch'].tolist() if 'Epoch' in training_results.columns else list(range(1, args.epochs + 1)),
            'train_loss_history': training_results['Train Loss'].tolist() if 'Train Loss' in training_results.columns else [],
            'val_loss_history': training_results['Val Loss'].tolist() if 'Val Loss' in training_results.columns else [],
            'train_accuracy_history': training_results['Train Accuracy'].tolist() if 'Train Accuracy' in training_results.columns else [],
            'val_accuracy_history': training_results['Val Accuracy'].tolist() if 'Val Accuracy' in training_results.columns else []
        }
    else:
        # Assuming it's already a dictionary with history information
        train_history = training_results
    
    # Add best model info
    if hasattr(trainer, 'best_val_accuracy'):
        train_history['best_val_accuracy'] = trainer.best_val_accuracy
        # Add safety check, prevent attribute from not existing
        if hasattr(trainer, 'best_epoch'):
            train_history['best_epoch'] = trainer.best_epoch
        else:
            # If best_epoch doesn't exist, use current epoch or set to -1
            train_history['best_epoch'] = args.epochs  # Default use last epoch
            logger.warning(f"TaskTrainer missing best_epoch attribute, using last epoch ({args.epochs}) as best")
    
    # Run evaluation on each test set
    for test_name, test_loader in test_loaders.items():
        logger.info(f"Evaluating on {test_name}...")
        test_loss, test_accuracy, test_f1, test_cm = trainer.evaluate(trained_model, test_loader)
        logger.info(f"{test_name} Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
        
        overall_metrics[f"{test_name}_loss"] = test_loss
        overall_metrics[f"{test_name}_accuracy"] = test_accuracy
        overall_metrics[f"{test_name}_f1_score"] = test_f1
    
    # Overall test accuracy (use only 'test' if available, otherwise average all test sets)
    if 'test_accuracy' in overall_metrics:
        overall_metrics['test_accuracy'] = overall_metrics['test_accuracy'] 
    elif len(test_loaders) > 0:
        test_accuracies = [v for k, v in overall_metrics.items() if k.endswith('_accuracy') and k.startswith('test')]
        overall_metrics['test_accuracy'] = sum(test_accuracies) / len(test_accuracies)
    
    # Save model summary to experiment_id directory
    summary_file = os.path.join(model_output_dir, f"model_summary.json")
    
    # Merge results and add experiment_id
    summary_results = {**train_history, **overall_metrics, "experiment_id": experiment_id}
    
    # Ensure all data is JSON serializable
    def convert_to_json_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    summary_results = convert_to_json_serializable(summary_results)
    
    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=4)
    
    logger.info(f"Model summary saved to {summary_file}")
    
    # Update or create best_performance.json
    best_performance_file = os.path.join(os.path.dirname(os.path.dirname(model_output_dir)), model_name, "best_performance.json")
    os.makedirs(os.path.dirname(best_performance_file), exist_ok=True)
    
    # experiment_id already created, no need to create again
    # Use previously defined timestamp and experiment_id
    
    # Check if best_performance.json exists and compare results
    best_performance = {
        "best_experiment_id": experiment_id,
        "best_test_accuracy": overall_metrics.get('test_accuracy', 0.0),
        "best_test_f1_score": overall_metrics.get('test_f1_score', 0.0),
        "best_experiment_params": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout
        },
        "timestamp": timestamp
    }
    
    if os.path.exists(best_performance_file):
        try:
            with open(best_performance_file, 'r') as f:
                existing_best = json.load(f)
            
            # Only update if current model is better
            if overall_metrics.get('test_accuracy', 0.0) > existing_best.get('best_test_accuracy', 0.0):
                logger.info(f"New best model! Accuracy: {overall_metrics.get('test_accuracy', 0.0):.4f} > {existing_best.get('best_test_accuracy', 0.0):.4f}")
                with open(best_performance_file, 'w') as f:
                    json.dump(best_performance, f, indent=4)
            else:
                logger.info(f"Current model not better than existing best. Accuracy: {overall_metrics.get('test_accuracy', 0.0):.4f} <= {existing_best.get('best_test_accuracy', 0.0):.4f}")
        except Exception as e:
            logger.warning(f"Error reading existing best_performance.json: {e}")
            with open(best_performance_file, 'w') as f:
                json.dump(best_performance, f, indent=4)
    else:
        # Create new best_performance.json
        with open(best_performance_file, 'w') as f:
            json.dump(best_performance, f, indent=4)
        logger.info(f"Created new best_performance.json at {best_performance_file}")
    
    return trained_model, overall_metrics

def main():
    """
    简化的主函数 - 假设在SageMaker环境中运行，数据在根目录
    
    数据加载流程:
    1. 使用/opt/ml/input/data/training作为数据根目录
    2. 直接使用根目录作为任务目录，不进行额外的路径检查
    3. 加载数据并训练指定的模型
    
    返回:
        None，但会将模型文件写入指定的输出目录
    """
    try:
        # 记录环境变量，用于调试
        logger.info("在SageMaker环境中运行")
        logger.info("环境变量:")
        for key in sorted([k for k in os.environ.keys() if k.startswith(('SM_', 'SAGEMAKER_'))]):
            logger.info(f"  {key}: {os.environ.get(key)}")

        # 获取参数
        args = get_args()
        
        # 打印解析参数
        logger.info("解析的参数:")
        for arg_name, arg_value in sorted(vars(args).items()):
            logger.info(f"  {arg_name}: {arg_value}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 设置随机种子
        set_seed(args.seed)
        logger.info(f"随机种子设置为 {args.seed}")
        
        # 日志记录起始信息
        logger.info(f"开始多模型训练，任务: {args.task_name}")
        logger.info(f"待训练模型: {args.all_models}")
        
        # 直接使用 /opt/ml/input/data/training 作为数据根目录
        dataset_root = '/opt/ml/input/data/training'
        logger.info(f"使用数据根目录: {dataset_root}")
        
        # 强制使用根目录作为任务目录
        args.use_root_data_path = True
        
        # 加载数据
        logger.info(f"从 {dataset_root} 加载数据，任务名称: {args.task_name}")
        data = load_benchmark_supervised(
            dataset_root=dataset_root,
            task_name=args.task_name,
            batch_size=args.batch_size,
            data_key=args.data_key,
            file_format=args.file_format,
            num_workers=args.num_workers,
            use_root_as_task_dir=True  # 始终使用根目录作为任务目录
        )
        
        # 检查数据加载是否成功
        if not data or 'loaders' not in data:
            logger.error(f"加载任务 {args.task_name} 的数据失败")
            sys.exit(1)
        
        logger.info(f"数据加载成功。类别数量: {data['num_classes']}")
        logger.info(f"可用数据加载器: {list(data['loaders'].keys())}")
        
        # 记录模型运行结果
        successful_models = []
        failed_models = []
        
        # 训练每个模型
        all_results = {}
        for model_name in args.all_models:
            try:
                logger.info(f"\n{'='*40}\n训练模型: {model_name}\n{'='*40}")
                
                # 尝试加载模型类来验证兼容性
                try:
                    ModelClass = MODEL_TYPES[model_name.lower()]
                    logger.info(f"模型类 {model_name} 加载成功")
                except Exception as e:
                    logger.error(f"加载模型类 {model_name} 时出错: {e}")
                    failed_models.append((model_name, f"模型类错误: {str(e)}"))
                    continue
                
                # 训练模型
                model, metrics = train_model(model_name, data, args, device)
                
                # 检查训练是否成功
                if model is None or (isinstance(metrics, dict) and 'error' in metrics):
                    error_msg = metrics.get('error', '未知错误') if isinstance(metrics, dict) else '未知错误'
                    logger.error(f"模型 {model_name} 训练失败: {error_msg}")
                    failed_models.append((model_name, error_msg))
                else:
                    all_results[model_name] = metrics
                    successful_models.append(model_name)
                    logger.info(f"完成 {model_name} 的训练")
            except Exception as e:
                logger.error(f"训练 {model_name} 时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
                failed_models.append((model_name, str(e)))
        
        # 打印训练结果摘要
        logger.info("\n" + "="*60)
        logger.info("训练摘要")
        logger.info("="*60)
        logger.info(f"任务: {args.task_name}")
        logger.info(f"成功训练的模型 ({len(successful_models)}): {', '.join(successful_models)}")
        logger.info(f"失败的模型 ({len(failed_models)}): {', '.join([m[0] for m in failed_models])}")
        
        if failed_models:
            logger.info("\n失败详情:")
            for model_name, error in failed_models:
                logger.info(f"  - {model_name}: {error}")
        
        # 保存整体结果摘要
        results_path = os.path.join(args.output_dir, args.task_name, "multi_model_results.json")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        logger.info(f"所有训练完成。结果保存到 {results_path}")
        logger.info("结果摘要:")
        for model_name, metrics in all_results.items():
            logger.info(f"  - {model_name}: 测试准确率 = {metrics.get('test_accuracy', 0.0):.4f}")
        
        # 识别最佳模型
        if all_results:
            best_model = max(all_results.items(), key=lambda x: x[1].get('test_accuracy', 0.0))
            logger.info(f"\n最佳模型: {best_model[0]}, 测试准确率 {best_model[1].get('test_accuracy', 0.0):.4f}")
        
        # 检查S3保存参数，如果没有设置，尝试从环境变量获取
        if args.save_to_s3 is None:
            logger.info("没有设置save_to_s3参数，尝试从环境变量获取S3输出路径")
            # 从SageMaker环境变量获取S3输出路径
            sm_output_dir = os.environ.get('SM_OUTPUT_DATA_DIR')
            sm_model_dir = os.environ.get('SM_MODEL_DIR')
            
            # 如果有SM_OUTPUT_DATA_DIR，构建S3输出路径
            if sm_output_dir and sm_output_dir.startswith('/opt/ml/output/data'):
                # 尝试从SageMaker任务设置中推导S3路径
                sm_output_s3 = os.environ.get('SAGEMAKER_S3_OUTPUT')
                if sm_output_s3:
                    logger.info(f"找到SageMaker S3输出路径: {sm_output_s3}")
                    args.save_to_s3 = sm_output_s3
                else:
                    logger.warning("无法从环境变量获取S3输出路径")
        
        # 直接上传最终结果到S3（如果启用）
        if args.direct_upload:
            if args.save_to_s3:
                logger.info(f"直接上传结果到S3: {args.save_to_s3}")
                
                # 输出结果目录结构以便调试
                task_dir = os.path.join(args.output_dir, args.task_name)
                logger.info(f"实例上的结果目录结构:")
                if os.path.exists(task_dir):
                    # 输出目录结构
                    def print_dir_structure(path, prefix=""):
                        logger.info(f"{prefix}├── {os.path.basename(path)}/")
                        for item in sorted(os.listdir(path)):
                            item_path = os.path.join(path, item)
                            if os.path.isdir(item_path):
                                print_dir_structure(item_path, prefix + "│   ")
                            else:
                                logger.info(f"{prefix}│   ├── {item}")
                    
                    # 输出顶层目录结构
                    logger.info(f"任务目录 ({task_dir}) 的内容:")
                    for item in sorted(os.listdir(task_dir)):
                        item_path = os.path.join(task_dir, item)
                        if os.path.isdir(item_path):
                            logger.info(f"├── {item}/ (目录)")
                            # 显示模型子目录中的实验ID
                            for exp_id in sorted(os.listdir(item_path)):
                                logger.info(f"│   ├── {exp_id}/ (实验ID)")
                        else:
                            logger.info(f"├── {item} (文件)")
                
                    # 上传整个任务目录结构，保持原有目录结构
                    s3_task_path = f"{args.save_to_s3.rstrip('/')}/{args.task_name}"
                    logger.info(f"准备上传目录: {task_dir} -> {s3_task_path}")
                    
                    # 列出要上传的文件
                    all_files = []
                    for root, _, files in os.walk(task_dir):
                        for file in files:
                            all_files.append(os.path.join(root, file))
                    logger.info(f"找到 {len(all_files)} 个文件需要上传")
                    
                    # 执行上传
                    upload_success = upload_to_s3(task_dir, s3_task_path)
                    if upload_success:
                        logger.info(f"结果成功上传到 {s3_task_path}")
                        
                        # 创建一个标记文件，表明已通过S3直接上传了结果文件，用于跳过model.tar.gz打包
                        try:
                            from datetime import datetime
                            info_path = os.path.join('/opt/ml/model', 'direct_upload_info.txt')
                            with open(info_path, 'w') as f:
                                f.write(f"Results directly uploaded to S3: {s3_task_path}\n")
                                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                                f.write(f"Task: {args.task_name}\n")
                                f.write(f"Models: {', '.join(successful_models)}\n")
                            logger.info(f"Created upload info file: {info_path}")
                        except Exception as e:
                            logger.warning(f"Could not create info file: {e}")
                    else:
                        logger.error(f"上传到 {s3_task_path} 失败!")
                
                # 如果实例上的目录结构与预期不同，输出更多调试信息
                else:
                    logger.error(f"任务输出目录不存在: {task_dir}")
                    
                    # 尝试查找在其他可能位置的结果目录
                    possible_dirs = [
                        os.path.join('/opt/ml/model', args.task_name),
                        os.path.join('/opt/ml/output/data', args.task_name),
                        '/opt/ml/model',
                        '/opt/ml/output/data'
                    ]
                    
                    for possible_dir in possible_dirs:
                        if os.path.exists(possible_dir):
                            logger.info(f"找到可能的结果目录: {possible_dir}")
                            logger.info(f"目录内容:")
                            for item in sorted(os.listdir(possible_dir)):
                                logger.info(f"  - {item}")
                            
                            # 如果找到合适的目录，尝试上传
                            if args.task_name in possible_dir:
                                alt_s3_path = f"{args.save_to_s3.rstrip('/')}/{os.path.basename(possible_dir)}"
                                logger.info(f"尝试上传替代目录: {possible_dir} -> {alt_s3_path}")
                                alt_success = upload_to_s3(possible_dir, alt_s3_path)
                                if alt_success:
                                    logger.info(f"成功上传替代目录到 {alt_s3_path}")
            else:
                logger.warning("未设置save_to_s3参数，跳过上传到S3")
        else:
            logger.info("未启用direct_upload，依赖SageMaker自动上传结果")
        
        # 清理SageMaker存储以减少空间使用
        cleanup_sagemaker_storage()
        
        logger.info("多模型训练成功完成!")
    except Exception as e:
        logger.error(f"main函数中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 