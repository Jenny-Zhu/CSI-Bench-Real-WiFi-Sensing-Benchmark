#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Training Script - Train multiple model architectures in one training job

This script can be run in a SageMaker environment to train and evaluate multiple model architectures
on the same task.
"""

# 禁用SMDebug和Horovod以避免PyTorch版本冲突
import os
os.environ['SM_DISABLE_PROFILER'] = 'true'
os.environ['SM_DISABLE_DEBUGGER'] = 'true'
os.environ['SMDEBUG_DISABLED'] = 'true'
os.environ['HOROVOD_WITH_PYTORCH'] = '0'
os.environ['HOROVOD_WITHOUT_PYTORCH'] = '1'
os.environ['USE_HOROVOD'] = 'false'

import sys
import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import logging

# 检测是否在SageMaker环境中运行
is_sagemaker = os.path.exists('/opt/ml/model')

# 如果在SageMaker中运行，导入S3工具
if is_sagemaker:
    try:
        import boto3
        s3_client = boto3.client('s3')
    except ImportError:
        print("Warning: boto3 not installed, S3 upload disabled")
        s3_client = None
else:
    s3_client = None

# 打印原始命令行参数，帮助诊断
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
    将本地文件或目录上传到S3
    
    Args:
        local_path: 本地文件或目录路径
        s3_path: S3路径，格式为's3://bucket-name/path/to/destination'
    
    Returns:
        bool: 上传是否成功
    """
    if not s3_client:
        logger.warning("S3 client not initialized, skipping upload")
        return False
    
    if not s3_path.startswith('s3://'):
        logger.error(f"Invalid S3 path: {s3_path}")
        return False
    
    try:
        # 解析S3路径
        s3_parts = s3_path.replace('s3://', '').split('/', 1)
        if len(s3_parts) != 2:
            logger.error(f"Invalid S3 path format: {s3_path}")
            return False
        
        bucket_name = s3_parts[0]
        s3_key_prefix = s3_parts[1]
        
        logger.info(f"Uploading {local_path} to S3 bucket {bucket_name}/{s3_key_prefix}")
        
        # 检查是文件还是目录
        if os.path.isfile(local_path):
            # 上传单个文件
            file_key = os.path.join(s3_key_prefix, os.path.basename(local_path))
            s3_client.upload_file(local_path, bucket_name, file_key)
            logger.info(f"Uploaded file to s3://{bucket_name}/{file_key}")
        else:
            # 上传整个目录
            for root, _, files in os.walk(local_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                    
                    # 计算相对路径
                    relative_path = os.path.relpath(local_file_path, local_path)
                    s3_key = os.path.join(s3_key_prefix, relative_path)
                    
                    # 上传文件
                    s3_client.upload_file(local_file_path, bucket_name, s3_key)
            
            logger.info(f"Uploaded directory contents to s3://{bucket_name}/{s3_key_prefix}")
        
        return True
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        return False

def cleanup_sagemaker_storage():
    """
    清理SageMaker环境中的不必要文件，减少存储使用
    """
    if not is_sagemaker:
        # 只在SageMaker环境中运行
        return
    
    logger.info("Cleaning up unnecessary files to reduce storage usage...")
    
    try:
        # 删除不必要的临时文件和日志
        dirs_to_clean = [
            "/tmp",                        # 临时目录
            "/opt/ml/output/profiler",     # 分析器输出
            "/opt/ml/output/tensors",      # 调试器张量
        ]
        
        # 只保留最小的日志文件
        log_files = [
            "/opt/ml/output/data/logs/algo-1-stdout.log",
            "/opt/ml/output/data/logs/algo-1-stderr.log"
        ]
        
        # 清理临时目录（但不全部删除）
        import shutil
        for cleanup_dir in dirs_to_clean:
            if os.path.exists(cleanup_dir):
                logger.info(f"Cleaning directory: {cleanup_dir}")
                # 仅限读取目录内容，不递归删除
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
        
        # 清理日志文件（保留最后10KB）
        for log_file in log_files:
            if os.path.exists(log_file) and os.path.getsize(log_file) > 10240:
                try:
                    with open(log_file, 'rb') as f:
                        # 跳转到文件末尾前10KB的位置
                        f.seek(-10240, 2)  # 2表示从文件末尾计算
                        last_10kb = f.read()
                    
                    # 重写日志文件，只保留最后10KB
                    with open(log_file, 'wb') as f:
                        f.write(b"[...previous logs truncated...]\n")
                        f.write(last_10kb)
                    
                    logger.info(f"Truncated log file: {log_file}")
                except Exception as e:
                    logger.warning(f"Could not truncate log file {log_file}: {e}")
        
        # 清理sourcedir缓存
        sourcedir_cache = "/opt/ml/code/.sourcedir.tar.gz"
        if os.path.exists(sourcedir_cache):
            try:
                os.remove(sourcedir_cache)
                logger.info("Removed sourcedir cache")
            except Exception as e:
                logger.warning(f"Could not remove sourcedir cache: {e}")
        
        # 尝试触发内存清理
        import gc
        gc.collect()
        
        logger.info("Storage cleanup completed!")
    except Exception as e:
        logger.error(f"Error during storage cleanup: {e}")

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
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='/opt/ml/model',
                        help='Directory to save checkpoints and models')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (defaults to save_dir if not specified)')
    
    # Add backward compatibility for data_dir parameter
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Root directory of the dataset (deprecated, use dataset_root instead)')
    
    # 调试参数 - 所有布尔标志参数
    parser.add_argument('--debug', action='store_true', default=False,
                        help='启用详细调试输出')
    parser.add_argument('--adaptive_path', action='store_true', default=False,
                        help='自动适应数据路径结构')
    parser.add_argument('--try_all_paths', action='store_true', default=False,
                        help='尝试所有可能的路径组合')
    parser.add_argument('--direct_upload', action='store_true', default=False,
                        help='直接上传结果到S3，不使用SageMaker自动打包')
    parser.add_argument('--upload_final_model', action='store_true', default=False,
                        help='上传最终模型到S3')
    parser.add_argument('--skip_train_for_debug', action='store_true', default=False,
                     help='仅用于调试，跳过实际训练过程')
    
    # 添加S3相关参数
    parser.add_argument('--save_to_s3', type=str, default=None,
                      help='S3路径用于保存结果, 格式: s3://bucket-name/path/')
    
    # 修改解析逻辑，处理SageMaker传递的非标准参数
    # 首先获取原始参数，进行预处理
    args_to_parse = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        # 处理标志参数后面跟着True或False的情况
        if arg.startswith('--') and i + 1 < len(sys.argv):
            next_arg = sys.argv[i+1]
            if next_arg.lower() == 'true':
                # 对于--flag True的情况，只保留--flag
                args_to_parse.append(arg)
                i += 2
                continue
            elif next_arg.lower() == 'false':
                # 对于--flag False的情况，跳过该参数
                i += 2
                continue
            
        # 正常添加参数
        args_to_parse.append(arg)
        i += 1
    
    try:
        # 使用预处理后的参数进行解析
        args = parser.parse_args(args_to_parse)
        
        # 打印实际解析的参数（调试用）
        logger.info(f"Parsed arguments: {args_to_parse}")
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        logger.error(f"Original arguments: {sys.argv}")
        logger.error(f"Processed arguments: {args_to_parse}")
        
        # 尝试使用原始参数解析，忽略错误
        try:
            args = parser.parse_args()
        except:
            # 最后的回退：使用默认参数
            logger.warning("Using default arguments due to parsing failure")
            args = parser.parse_args([])
    
    # 如果启用调试模式，设置日志级别为DEBUG
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled - verbose logging activated")
        # 打印所有参数
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
    model_params = {
        'num_classes': num_classes
    }
    
    # Add extra parameters based on model type
    if model_name.lower() in ['mlp', 'vit']:
        model_params.update({
            'win_len': args.win_len,
            'feature_size': args.feature_size
        })
    
    if model_name.lower() == 'resnet18':
        model_params.update({
            'in_channels': args.in_channels
        })
    
    if model_name.lower() == 'lstm':
        model_params.update({
            'feature_size': args.feature_size,
            'dropout': args.dropout
        })
    
    if model_name.lower() == 'transformer':
        model_params.update({
            'feature_size': args.feature_size,
            'd_model': args.d_model,
            'dropout': args.dropout
        })
    
    if model_name.lower() == 'vit':
        model_params.update({
            'emb_dim': args.emb_dim,
            'dropout': args.dropout,
            'in_channels': args.in_channels
        })
    
    # Create model instance and move to device
    model = ModelClass(**model_params).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 添加测试以检查输入张量形状
    try:
        # 获取一个样本批次来测试模型
        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
            inputs, labels = sample_batch[0], sample_batch[1]
        else:
            logger.error(f"Unexpected dataloader batch format: {type(sample_batch)}")
            inputs = sample_batch
        
        logger.info(f"Input tensor shape: {inputs.shape}")
        if model_name.lower() == 'lstm' or model_name.lower() == 'transformer':
            # 这些模型对输入形状比较敏感
            logger.info(f"Expected feature_size: {args.feature_size}")
            logger.info(f"Actual feature dimension size: {inputs.shape[3] if len(inputs.shape) > 3 else 'N/A'}")
        
        # 尝试进行一次前向传播来验证形状兼容性
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
            logger.info(f"Forward pass successful! Output shape: {outputs.shape}")
    except Exception as e:
        logger.error(f"Error during model input testing: {e}")
        logger.warning("继续执行，但模型可能在训练阶段出现问题")
    
    # Create specific save and output directories for each model
    # Format: output_path/task/model/
    model_save_dir = os.path.join(args.save_dir, args.task_name, model_name)
    model_output_dir = os.path.join(args.output_dir, args.task_name, model_name)
    
    # Ensure directories exist
    os.makedirs(model_save_dir, exist_ok=True)
    if model_output_dir != model_save_dir:
        os.makedirs(model_output_dir, exist_ok=True)
    
    # Create config object
    config = argparse.Namespace(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        num_classes=num_classes,
        device=str(device),
        save_dir=model_save_dir,
        output_dir=model_output_dir,
        results_subdir='supervised',
        model_name=model_name,
        task_name=args.task_name
    )
    
    # Save configuration
    config_path = os.path.join(model_output_dir, f"{model_name}_{args.task_name}_config.json")
    with open(config_path, "w") as f:
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        json.dump(config_dict, f, indent=4)
    
    logger.info(f"Configuration saved to {config_path}")
    
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
    test_loaders = {k: v for k, v in loaders.items() if k.startswith('test')}
    
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
        # 添加安全检查，防止属性不存在
        if hasattr(trainer, 'best_epoch'):
            train_history['best_epoch'] = trainer.best_epoch
        else:
            # 如果best_epoch不存在，使用当前epoch或设置为-1
            train_history['best_epoch'] = args.epochs  # 默认使用最后一个epoch
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
    
    # Save model summary
    summary_file = os.path.join(model_output_dir, f"{model_name}_{args.task_name}_summary.json")
    
    # Combine results
    summary_results = {**train_history, **overall_metrics}
    
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
    best_performance_file = os.path.join(os.path.dirname(model_output_dir), model_name, "best_performance.json")
    os.makedirs(os.path.dirname(best_performance_file), exist_ok=True)
    
    # Create experiment ID from timestamp and model name
    import hashlib
    timestamp = int(time.time())
    experiment_id = f"params_{hashlib.md5(f'{model_name}_{args.task_name}_{timestamp}'.encode()).hexdigest()[:8]}"
    
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
    """Main function to run multi-model training"""
    try:
        # 记录环境变量，帮助调试SageMaker环境中的问题
        if is_sagemaker:
            logger.info("Running in SageMaker environment")
            logger.info("Environment variables:")
            # 记录关键的环境变量
            for key in sorted([k for k in os.environ.keys() if k.startswith(('SM_', 'SAGEMAKER_'))]):
                logger.info(f"  {key}: {os.environ.get(key)}")
            
            # 尝试从环境变量解析超参数
            if os.environ.get('SM_HPS') is not None:
                try:
                    import json
                    hps = json.loads(os.environ.get('SM_HPS', '{}'))
                    logger.info(f"Hyperparameters from environment: {hps}")
                except Exception as e:
                    logger.warning(f"Failed to parse SM_HPS: {e}")
        
        # 获取参数
        args = get_args()
        
        # 打印解析后的参数
        logger.info("Parsed arguments:")
        for arg_name, arg_value in sorted(vars(args).items()):
            logger.info(f"  {arg_name}: {arg_value}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Set random seed
        set_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
        
        # Log starting info
        logger.info(f"Starting multi-model training for task: {args.task_name}")
        logger.info(f"Models to train: {args.all_models}")
        
        # Load data once for all models
        logger.info(f"Loading data from {args.dataset_root}")
        
        # Special handling for S3 paths in SageMaker environment
        if is_sagemaker:
            # In SageMaker, use /opt/ml/input/data/training as the dataset root
            original_path = args.dataset_root
            dataset_root = '/opt/ml/input/data/training'
            logger.info(f"SageMaker environment detected. Using local path: {dataset_root}")
            logger.info(f"Original S3 path: {original_path}")
            
            # 在SageMaker环境中，检查可能的目录结构
            # 1. 首先检查/opt/ml/input/data/training/tasks/TaskName结构
            tasks_task_path = os.path.join(dataset_root, 'tasks', args.task_name)
            # 2. 检查/opt/ml/input/data/training/TaskName结构
            direct_task_path = os.path.join(dataset_root, args.task_name)
            # 3. 检查/opt/ml/input/data/training本身是否就是任务目录
            root_as_task_path = dataset_root
            
            if os.path.exists(tasks_task_path):
                logger.info(f"Found task at path: {tasks_task_path}")
                # 这是预期的结构：/opt/ml/input/data/training/tasks/TaskName
                task_dir = tasks_task_path
            elif os.path.exists(direct_task_path):
                logger.info(f"Found task directly at: {direct_task_path}")
                # 替代结构：/opt/ml/input/data/training/TaskName
                task_dir = direct_task_path
            elif os.path.exists(os.path.join(root_as_task_path, 'train')):
                # 检查是否训练目录直接在根目录下
                logger.info(f"Root directory contains train subfolder, might be the task directory itself")
                task_dir = root_as_task_path
            else:
                # 记录目录内容以便调试
                logger.info(f"Contents of {dataset_root}: {os.listdir(dataset_root)}")
                if os.path.exists(os.path.join(dataset_root, 'tasks')):
                    logger.info(f"Contents of {os.path.join(dataset_root, 'tasks')}: {os.listdir(os.path.join(dataset_root, 'tasks'))}")
                task_dir = None  # 未找到任务目录
        else:
            dataset_root = args.dataset_root
            task_dir = None  # 初始化为None，稍后决定
        
        logger.info(f"Actual dataset root path: {dataset_root}")
        
        # 如果在SageMaker环境中已经找到任务目录，无需再次搜索
        if not is_sagemaker or task_dir is None:
            # 在非SageMaker环境中，或者在SageMaker环境中还未找到任务目录时
            if os.path.exists(dataset_root):
                logger.info(f"Dataset root exists: {dataset_root}")
                # 检查任务目录
                direct_task_path = os.path.join(dataset_root, args.task_name)
                if os.path.exists(direct_task_path):
                    logger.info(f"Task directory found at {direct_task_path}")
                    task_dir = direct_task_path
                else:
                    # 尝试tasks/task_name
                    tasks_dir = os.path.join(dataset_root, 'tasks')
                    if os.path.exists(tasks_dir):
                        tasks_task_path = os.path.join(tasks_dir, args.task_name)
                        if os.path.exists(tasks_task_path):
                            logger.info(f"Task directory found at {tasks_task_path}")
                            task_dir = tasks_task_path
                        else:
                            logger.warning(f"Task directory not found at {tasks_task_path}")
                            task_dir = None
                    else:
                        logger.warning(f"Neither {direct_task_path} nor {os.path.join(tasks_dir, args.task_name)} exists")
                        task_dir = None
            else:
                logger.warning(f"Dataset root path {dataset_root} does not exist")
                task_dir = None
        
        # 如果找到了任务目录，检查其内容
        if task_dir and os.path.exists(task_dir):
            logger.info(f"Final task directory selected: {task_dir}")
            logger.info(f"Content of task directory {task_dir}: {os.listdir(task_dir)}")
            
            # 进一步检查文件夹结构
            if os.path.exists(os.path.join(task_dir, 'metadata')):
                logger.info(f"Metadata directory found: {os.path.join(task_dir, 'metadata')}")
            if os.path.exists(os.path.join(task_dir, 'splits')):
                logger.info(f"Splits directory found: {os.path.join(task_dir, 'splits')}")
                logger.info(f"Contents of splits: {os.listdir(os.path.join(task_dir, 'splits'))}")
            if os.path.exists(os.path.join(task_dir, 'train')):
                logger.info(f"Train directory found: {os.path.join(task_dir, 'train')}")
        
        # 自适应路径处理逻辑
        if args.adaptive_path:
            logger.info("Adaptive path mode enabled - will search for alternative data paths")
            # 在SageMaker环境中尝试不同的常见路径结构
            possible_paths = []
            
            # 考虑常见的路径变体
            possible_paths.append(dataset_root)  # 直接使用下载路径
            possible_paths.append(os.path.join(dataset_root, 'tasks'))  # tasks子目录
            possible_paths.append(os.path.join(dataset_root, 'Benchmark'))  # Benchmark子目录
            possible_paths.append(os.path.join(dataset_root, 'Data', 'Benchmark'))  # Data/Benchmark路径
            
            # 检查每个可能的路径
            for path in possible_paths:
                if os.path.exists(path):
                    logger.info(f"Found valid path: {path}")
                    # 检查是否包含任务目录
                    task_path = os.path.join(path, args.task_name)
                    tasks_path = os.path.join(path, 'tasks', args.task_name)
                    
                    if os.path.exists(task_path):
                        logger.info(f"Found task directory at {task_path}")
                        dataset_root = path
                        break
                    elif os.path.exists(tasks_path):
                        logger.info(f"Found task directory at {tasks_path}")
                        dataset_root = path
                        break
        
        # 尝试所有路径组合（更彻底的搜索）
        if args.try_all_paths and is_sagemaker:
            logger.info("Try all paths mode enabled - will try multiple dataset_root options")
            # 创建备份的原始路径
            original_dataset_root = dataset_root
            
            # 定义可能的数据集根目录
            dataset_roots_to_try = [
                dataset_root,  # 原始路径
                os.path.dirname(dataset_root),  # 上级目录
                '/opt/ml/input/data',  # SageMaker数据根目录
                '/opt/ml/input',  # 更上一级
            ]
            
            # 将原始S3路径的各种变体添加到尝试列表
            if original_path.startswith('s3://'):
                s3_parts = original_path.replace('s3://', '').split('/')
                if len(s3_parts) > 1:
                    # 尝试几种可能的映射方式
                    dataset_roots_to_try.append(os.path.join(dataset_root, s3_parts[1]))  # bucket下的第一级目录
                    if len(s3_parts) > 2:
                        dataset_roots_to_try.append(os.path.join(dataset_root, s3_parts[2]))  # bucket下的第二级目录
                        dataset_roots_to_try.append(os.path.join(dataset_root, '/'.join(s3_parts[1:3])))  # 前两级目录组合
            
            # 记录所有尝试的路径
            for root in dataset_roots_to_try:
                if os.path.exists(root):
                    logger.info(f"Testing dataset_root: {root}")
                    try:
                        task_found = False
                        # 检查是否直接包含任务目录
                        if os.path.exists(os.path.join(root, args.task_name)):
                            logger.info(f"  - Found task directory directly: {os.path.join(root, args.task_name)}")
                            task_found = True
                        
                        # 检查是否包含tasks/任务目录
                        if os.path.exists(os.path.join(root, 'tasks', args.task_name)):
                            logger.info(f"  - Found task in tasks/ subdirectory: {os.path.join(root, 'tasks', args.task_name)}")
                            task_found = True
                        
                        # 尝试列出该目录下的内容
                        if not task_found:
                            logger.info(f"  - Directory contents: {os.listdir(root)}")
                            # 检查是否有tasks目录
                            if 'tasks' in os.listdir(root):
                                logger.info(f"    - tasks/ subdirectory contents: {os.listdir(os.path.join(root, 'tasks'))}")
                    except Exception as e:
                        logger.info(f"  - Error exploring path: {e}")
                else:
                    logger.info(f"Path does not exist: {root}")
            
        try:
            data = load_benchmark_supervised(
                dataset_root=dataset_root,
                task_name=args.task_name,
                batch_size=args.batch_size,
                data_key=args.data_key,
                file_format=args.file_format,
                num_workers=args.num_workers
            )
            
            # Check if data loaded successfully
            if not data or 'loaders' not in data:
                logger.error(f"Failed to load data for task {args.task_name}")
                sys.exit(1)
            
            logger.info(f"Data loaded successfully. Number of classes: {data['num_classes']}")
            
            # 增加更详细的数据集信息
            logger.info(f"Available loaders: {list(data['loaders'].keys())}")
            
            # 检查数据集大小
            if 'datasets' in data:
                for split_name, dataset in data['datasets'].items():
                    logger.info(f"Dataset '{split_name}' size: {len(dataset)}")
            
            # 检查标签映射
            if 'label_mapper' in data:
                label_mapper = data['label_mapper']
                logger.info(f"Label mapping: {label_mapper.label_to_idx}")
            
            # 添加数据形状验证
            if 'train' in data['loaders']:
                try:
                    sample_batch = next(iter(data['loaders']['train']))
                    if isinstance(sample_batch, (list, tuple)) and len(sample_batch) >= 2:
                        x, y = sample_batch[0], sample_batch[1]
                        logger.info(f"Sample batch shapes - X: {x.shape}, y: {y.shape}")
                        
                        # 根据数据形状自动确定哪些模型可能不兼容
                        incompatible_models = []
                        feature_size_actual = x.shape[3] if len(x.shape) > 3 else None
                        
                        # LSTM期望[batch, seq_len, feature_size]，实际为[batch, 1, win_len, feature_size]
                        if len(x.shape) == 4 and feature_size_actual != args.feature_size:
                            logger.warning(f"LSTM and Transformer might have compatibility issues! "
                                         f"Expected feature_size={args.feature_size}, but got {feature_size_actual}")
                            if feature_size_actual > 2 * args.feature_size:  # 大幅不匹配
                                logger.warning("feature_size mismatch is significant, models may fail")
                        
                        # 告知用户可能存在的问题
                        if incompatible_models:
                            logger.warning(f"Models {incompatible_models} might not be compatible with the input data shape.")
                            logger.warning("They will still be run, but might fail during training.")
                        
                        # 记录其他有用的信息
                        logger.info(f"X data type: {x.dtype}, device: {x.device}")
                        logger.info(f"X stats - min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")
                        logger.info(f"Y labels: {y.tolist()}")
                    else:
                        logger.info(f"Sample batch type: {type(sample_batch)}")
                        logger.info(f"Sample batch content: {sample_batch}")
                except Exception as e:
                    logger.warning(f"Could not get sample batch info: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
        
        # 记录模型运行结果以便生成摘要
        successful_models = []
        failed_models = []
        
        # Train each model
        all_results = {}
        for model_name in args.all_models:
            try:
                logger.info(f"\n{'='*40}\nTraining model: {model_name}\n{'='*40}")
                
                # 尝试载入模型类来验证兼容性
                try:
                    ModelClass = MODEL_TYPES[model_name.lower()]
                    logger.info(f"Model class {model_name} loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading model class for {model_name}: {e}")
                    failed_models.append((model_name, f"Model class error: {str(e)}"))
                    continue
                
                # 训练模型
                model, metrics = train_model(model_name, data, args, device)
                
                # 检查是否成功训练
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
        
        # 打印训练结果摘要
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
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
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        logger.info(f"All training completed. Results saved to {results_path}")
        logger.info("Summary of results:")
        for model_name, metrics in all_results.items():
            logger.info(f"  - {model_name}: Test Accuracy = {metrics.get('test_accuracy', 0.0):.4f}")
        
        # Identify the best model
        if all_results:
            best_model = max(all_results.items(), key=lambda x: x[1].get('test_accuracy', 0.0))
            logger.info(f"\nBest model: {best_model[0]} with test accuracy {best_model[1].get('test_accuracy', 0.0):.4f}")
        
        # 将最终结果直接上传到S3（如果启用）
        if args.direct_upload and args.save_to_s3 and is_sagemaker:
            logger.info(f"Directly uploading results to S3: {args.save_to_s3}")
            s3_task_path = f"{args.save_to_s3.rstrip('/')}/{args.task_name}"
            
            # 上传整个任务目录
            task_dir = os.path.join(args.output_dir, args.task_name)
            if os.path.exists(task_dir):
                upload_to_s3(task_dir, s3_task_path)
                logger.info(f"Results uploaded to {s3_task_path}")
        
        # 清理SageMaker存储以减少空间使用
        cleanup_sagemaker_storage()
        
        logger.info("Multi-model training completed successfully!")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 