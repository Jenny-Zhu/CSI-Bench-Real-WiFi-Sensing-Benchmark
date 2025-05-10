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
    os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER'] = 'true'
    os.environ['SMDATAPARALLEL_DISABLE_DEBUGGER_OUTPUT'] = 'true'
    
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
import math

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
            
            # 尝试上传，并在失败时重试
            max_retries = 3
            for retry in range(max_retries):
                try:
                    s3_client.upload_file(local_path, bucket_name, file_key)
                    logger.info(f"成功上传文件到 s3://{bucket_name}/{file_key}")
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
            
            # 遍历目录上传所有文件
            uploaded_files = 0
            failed_files = 0
            
            for root, _, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    
                    # 计算相对路径
                    relative_path = os.path.relpath(local_file_path, local_path)
                    s3_key = os.path.join(s3_key_prefix, relative_path)
                    
                    # 上传文件，带有重试逻辑
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            s3_client.upload_file(local_file_path, bucket_name, s3_key)
                            uploaded_files += 1
                            # 每上传10个文件或最后一个文件时，输出进度信息
                            if uploaded_files % 10 == 0 or uploaded_files == total_files:
                                logger.info(f"上传进度: {uploaded_files}/{total_files} 文件 ({uploaded_files/total_files*100:.1f}%)")
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
        # Delete unnecessary temporary files and logs
        dirs_to_clean = [
            "/tmp",                        # Temporary directory
            "/opt/ml/output/profiler",     # Profiler output
            "/opt/ml/output/tensors",      # Debugger tensors
            "/opt/ml/output/debug-output", # Debug output
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
    
    # Store overall metrics
    overall_metrics = {}
    
    # Run evaluation on each test set
    for test_name, test_loader in test_loaders.items():
        logger.info(f"Evaluating on {test_name}...")
        test_loss, test_accuracy = trainer.evaluate(test_loader)
        
        # Calculate metrics
        try:
            test_f1, _ = trainer.calculate_metrics(test_loader)
        except:
            test_f1 = 0.0
            
        # Store metrics
        overall_metrics[test_name] = {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'f1_score': test_f1
        }
        
        logger.info(f"{test_name} Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    
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
            
            # Print directory structure for debugging
            task_dir = os.path.join(args.output_dir, args.task_name)
            logger.info(f"Directory structure to upload:")
            
            # List contents of task directory
            if os.path.exists(task_dir):
                total_files = 0
                file_sizes = 0
                
                for root, dirs, files in os.walk(task_dir):
                    for name in files:
                        total_files += 1
                        file_path = os.path.join(root, name)
                        file_sizes += os.path.getsize(file_path)
                        # Only log key files to avoid excessive logging
                        if name.endswith(('.json', '.csv', '.png')) and total_files < 50:
                            rel_path = os.path.relpath(file_path, task_dir)
                            logger.info(f"  - {rel_path} ({os.path.getsize(file_path)/1024:.1f} KB)")
                
                logger.info(f"Total: {total_files} files, {file_sizes/1024/1024:.2f} MB")
                
                # Construct S3 target path - use the exact S3 path from environment
                s3_task_path = args.save_to_s3.rstrip('/')
                logger.info(f"Uploading directory: {task_dir} -> {s3_task_path}")
                
                # Upload directory - retry once if failed
                try:
                    upload_success = upload_to_s3(task_dir, s3_task_path)
                    if not upload_success:
                        logger.warning("First upload attempt failed. Retrying once...")
                        time.sleep(5)  # Short delay before retry
                        upload_success = upload_to_s3(task_dir, s3_task_path)
                    
                    if upload_success:
                        logger.info(f"Successfully uploaded results to {s3_task_path}")
                    else:
                        logger.error(f"Failed to upload results to {s3_task_path} after retry")
                except Exception as e:
                    logger.error(f"Error during upload: {e}")
            else:
                logger.error(f"Task directory {task_dir} does not exist. Cannot upload results.")
        
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