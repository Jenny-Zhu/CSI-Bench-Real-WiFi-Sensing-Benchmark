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
    上传本地文件或目录到S3，保持原始目录结构
    
    Args:
        local_path: 本地文件或目录路径
        s3_path: S3路径，格式：'s3://bucket-name/path/to/destination'
    
    Returns:
        bool: 上传是否成功
    """
    if not s3_client:
        logger.warning("S3客户端未初始化，跳过上传")
        return False
    
    if not s3_path.startswith('s3://'):
        logger.error(f"无效的S3路径: {s3_path}")
        return False
    
    try:
        # 解析S3路径
        s3_parts = s3_path.replace('s3://', '').split('/', 1)
        if len(s3_parts) != 2:
            logger.error(f"无效的S3路径格式: {s3_path}")
            return False
        
        bucket_name = s3_parts[0]
        s3_key_prefix = s3_parts[1]
        
        # 确保前缀以斜杠结尾
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
                    # 添加元数据标记，表明这是直接上传而非model.tar.gz包装
                    s3_client.upload_file(
                        local_path, 
                        bucket_name, 
                        file_key,
                        ExtraArgs={
                            'Metadata': {
                                'upload-method': 'direct',
                                'content-type': 'application/octet-stream'
                            }
                        }
                    )
                    logger.info(f"成功上传文件到 s3://{bucket_name}/{file_key}")
                    break
                except Exception as e:
                    if retry < max_retries - 1:
                        logger.warning(f"文件上传失败，重试中 ({retry+1}/{max_retries}): {e}")
                        time.sleep(2)  # 失败后延迟更长时间再重试
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
            
            for root, dirs, files in os.walk(local_path):
                # 确保空目录也被创建
                if not files and not dirs:
                    # 创建一个空目录标记文件
                    relative_path = os.path.relpath(root, local_path)
                    s3_key = os.path.join(s3_key_prefix, relative_path, '.directory_marker')
                    try:
                        s3_client.put_object(
                            Bucket=bucket_name,
                            Key=s3_key,
                            Body=b'',
                            Metadata={'directory': 'true'}
                        )
                        logger.debug(f"创建空目录标记: s3://{bucket_name}/{s3_key}")
                    except Exception as e:
                        logger.warning(f"创建空目录标记失败: {e}")
                
                for file in files:
                    local_file_path = os.path.join(root, file)
                    
                    # 计算相对路径
                    relative_path = os.path.relpath(local_file_path, local_path)
                    s3_key = os.path.join(s3_key_prefix, relative_path)
                    
                    # 对Windows路径进行处理
                    s3_key = s3_key.replace('\\', '/')
                    
                    # 上传文件，带有重试逻辑
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            # 添加元数据标记，表明这是直接上传而非model.tar.gz包装
                            s3_client.upload_file(
                                local_file_path, 
                                bucket_name, 
                                s3_key,
                                ExtraArgs={
                                    'Metadata': {
                                        'upload-method': 'direct',
                                        'original-path': relative_path
                                    }
                                }
                            )
                            uploaded_files += 1
                            # 每上传10个文件或最后一个文件时，输出进度信息
                            if uploaded_files % 10 == 0 or uploaded_files == total_files:
                                logger.info(f"上传进度: {uploaded_files}/{total_files} 文件 ({uploaded_files/total_files*100:.1f}%)")
                            break
                        except Exception as e:
                            if retry < max_retries - 1:
                                logger.warning(f"文件上传失败 {local_file_path}，重试中 ({retry+1}/{max_retries}): {e}")
                                time.sleep(2)  # 失败后延迟更长时间再重试
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
        
        # 上传成功完成标记文件
        try:
            completion_marker = os.path.join(s3_key_prefix, '.upload_complete')
            s3_client.put_object(
                Bucket=bucket_name,
                Key=completion_marker,
                Body=f"Upload completed at {time.strftime('%Y-%m-%d %H:%M:%S')}".encode('utf-8'),
                Metadata={'upload-completion': 'true'}
            )
            logger.info(f"已创建上传完成标记: s3://{bucket_name}/{completion_marker}")
        except Exception as e:
            logger.warning(f"创建上传完成标记失败: {e}")
        
        return True
    except Exception as e:
        logger.error(f"S3上传过程中发生错误: {e}")
        return False

def cleanup_sagemaker_storage():
    """
    清理SageMaker环境中不必要的文件以减少存储使用
    同时确保不会生成model.tar.gz文件
    """
    if not is_sagemaker:
        # 仅在SageMaker环境中运行
        return
    
    logger.info("清理不必要的文件以减少存储使用...")
    
    try:
        # 删除不必要的临时文件和日志
        dirs_to_clean = [
            "/tmp",                        # 临时目录
            "/opt/ml/output/profiler",     # 分析器输出
            "/opt/ml/output/tensors",      # 调试器张量
            "/opt/ml/output/data/logs",    # 保留但清理大型日志文件
        ]
        
        # 检查是否存在model.tar.gz构建相关文件
        model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        if os.path.exists(model_dir):
            # 创建一个标记文件，表明我们使用直接上传而不是model.tar.gz
            direct_upload_marker = os.path.join(model_dir, '.direct_upload_marker')
            try:
                with open(direct_upload_marker, 'w') as f:
                    f.write(f"Direct upload enabled. Do not create model.tar.gz. {time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"已创建直接上传标记文件: {direct_upload_marker}")
            except Exception as e:
                logger.warning(f"创建直接上传标记文件失败: {e}")
        
        # 清理临时目录（但不要全部删除）
        import shutil
        for cleanup_dir in dirs_to_clean:
            if os.path.exists(cleanup_dir):
                logger.info(f"清理目录: {cleanup_dir}")
                # 只读取目录内容，不要递归删除
                try:
                    for item in os.listdir(cleanup_dir):
                        item_path = os.path.join(cleanup_dir, item)
                        if os.path.isdir(item_path) and not item.startswith('.'):
                            try:
                                shutil.rmtree(item_path)
                            except Exception as e:
                                logger.warning(f"无法删除目录 {item_path}: {e}")
                        elif os.path.isfile(item_path) and not item.startswith('.'):
                            try:
                                # 如果是日志文件并且超过1MB，则保留最后的10KB
                                if item_path.endswith('.log') and os.path.getsize(item_path) > 1024 * 1024:
                                    try:
                                        with open(item_path, 'rb') as f:
                                            # 跳到文件末尾前10KB
                                            f.seek(-10240, 2)  # 2表示从文件末尾
                                            last_10kb = f.read()
                                        
                                        # 重写日志文件，只保留最后10KB
                                        with open(item_path, 'wb') as f:
                                            f.write(b"[...Log truncated...]\n\n")
                                            f.write(last_10kb)
                                        logger.info(f"已截断大型日志文件: {item_path}")
                                    except Exception:
                                        # 如果截断失败，尝试删除
                                        os.remove(item_path)
                                else:
                                    # 其他文件直接删除
                                    os.remove(item_path)
                            except Exception as e:
                                logger.warning(f"无法删除文件 {item_path}: {e}")
                except Exception as e:
                    logger.warning(f"清理目录 {cleanup_dir} 时出错: {e}")
        
        # 禁用model.tar.gz的生成 - 在导出路径创建一个标记文件
        if 'SM_MODEL_DIR' in os.environ:
            model_dir = os.environ['SM_MODEL_DIR']
            no_tar_marker = os.path.join(model_dir, '.no_tar_gz')
            try:
                with open(no_tar_marker, 'w') as f:
                    f.write("direct_s3_upload=True\n")
                    f.write(f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                logger.info(f"已创建禁用tar.gz标记文件: {no_tar_marker}")
                
                # 创建README文件解释数据已上传到S3
                readme_path = os.path.join(model_dir, 'README.txt')
                with open(readme_path, 'w') as f:
                    f.write("Model and data files have been directly uploaded to S3.\n")
                    f.write("This directory contains only marker files to prevent model.tar.gz creation.\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                logger.info(f"已创建README文件: {readme_path}")
            except Exception as e:
                logger.warning(f"创建标记文件失败: {e}")
        
        # 清理日志文件（保留最后10KB）
        log_files = [
            "/opt/ml/output/data/logs/algo-1-stdout.log",
            "/opt/ml/output/data/logs/algo-1-stderr.log"
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file) and os.path.getsize(log_file) > 10240:
                try:
                    with open(log_file, 'rb') as f:
                        # 跳到文件末尾前10KB
                        f.seek(-10240, 2)  # 2表示从文件末尾
                        last_10kb = f.read()
                    
                    # 重写日志文件，只保留最后10KB
                    with open(log_file, 'wb') as f:
                        f.write(b"[...Log truncated...]\n\n")
                        f.write(last_10kb)
                    logger.info(f"已截断日志文件 {log_file}")
                except Exception as e:
                    logger.warning(f"截断日志文件 {log_file} 时出错: {e}")
        
        # 运行垃圾回收
        gc.collect()
        logger.info("已完成存储清理")
    except Exception as e:
        logger.error(f"清理存储时出错: {e}")

def get_args():
    """分析命令行参数"""
    parser = argparse.ArgumentParser(description="WiFi感知多模型训练")
    
    # 任务参数
    parser.add_argument('--task_name', type=str, required=True, help='要训练的任务名称')
    
    # 数据相关参数
    parser.add_argument('--data_root', type=str, default='/opt/ml/input/data/training', help='数据根目录')
    parser.add_argument('--tasks_dir', type=str, default='tasks', help='任务目录')
    parser.add_argument('--data_key', type=str, default='data', help='数据键')
    parser.add_argument('--file_format', type=str, default='h5', choices=['h5', 'npz', 'pt'], help='数据文件格式')
    parser.add_argument('--use_root_data_path', action='store_true', default=True, help='是否使用根目录作为数据路径')
    parser.add_argument('--adaptive_path', action='store_true', default=True, help='自适应搜索路径')
    parser.add_argument('--try_all_paths', action='store_true', default=True, help='尝试所有可能的数据路径')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='/opt/ml/output/data', help='输出目录')
    parser.add_argument('--save_to_s3', type=str, default=None, help='S3路径，用于保存结果 (s3://bucket/path)')
    
    # 模型参数
    parser.add_argument('--models', type=str, default='mlp,lstm,resnet18,transformer', help='要训练的模型，逗号分隔')
    parser.add_argument('--win_len', type=int, default=500, help='窗口长度')
    parser.add_argument('--feature_size', type=int, default=232, help='特征大小')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='预热轮数')
    parser.add_argument('--patience', type=int, default=15, help='早停的耐心值')
    parser.add_argument('--gpu', type=int, default=0, help='GPU索引')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数量')
    
    # 测试参数
    parser.add_argument('--test_splits', type=str, default='test_id,test_ood,test_cross_env', help='测试分割，逗号分隔')
    
    # 实验参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 少样本学习参数
    parser.add_argument('--enable_few_shot', action='store_true', default=False, help='启用少样本学习')
    parser.add_argument('--k_shot', type=int, default=5, help='每类样本数量')
    parser.add_argument('--inner_lr', type=float, default=0.01, help='内部学习率')
    parser.add_argument('--num_inner_steps', type=int, default=10, help='内部适应步骤')
    parser.add_argument('--fewshot_support_split', type=str, default='val_id', help='少样本支持集分割')
    parser.add_argument('--fewshot_query_split', type=str, default='test_cross_env', help='少样本查询集分割')
    
    # 解析SageMaker超参数
    args, unknown = parser.parse_known_args()
    
    # 将模型列表转换为列表
    args.all_models = [m.strip() for m in args.models.split(',')]
    
    # 将测试分割转换为列表
    if args.test_splits == 'all':
        args.test_splits = ['test_id', 'test_ood', 'test_cross_env']
    else:
        args.test_splits = [ts.strip() for ts in args.test_splits.split(',')]
    
    # 根据超参数环境变量覆盖参数
    for arg in unknown:
        if arg.startswith(('--sm-hp-', '--SM_HP_')):
            # Argument format: --sm-hp-name or --SM_HP_NAME
            parts = arg.split('-', 2) if arg.startswith('--sm-hp-') else arg.split('_', 2)
            if len(parts) < 3:
                continue
                
            # 获取参数名和值
            param_name = parts[-1].replace('-', '_').lower()
            
            # 查找下一个参数是否为值
            idx = sys.argv.index(arg) if arg in sys.argv else -1
            if idx >= 0 and idx < len(sys.argv) - 1 and not sys.argv[idx + 1].startswith('--'):
                param_value = sys.argv[idx + 1]
                
                # 设置参数
                if hasattr(args, param_name):
                    # 根据参数类型转换值
                    if isinstance(getattr(args, param_name), bool):
                        if param_value.lower() in ('true', 'yes', '1'):
                            setattr(args, param_name, True)
                        elif param_value.lower() in ('false', 'no', '0'):
                            setattr(args, param_name, False)
                    elif isinstance(getattr(args, param_name), int):
                        setattr(args, param_name, int(param_value))
                    elif isinstance(getattr(args, param_name), float):
                        setattr(args, param_name, float(param_value))
                    else:
                        setattr(args, param_name, param_value)
    
    # 处理环境变量中的参数
    for k, v in os.environ.items():
        # 检查SM_HP_格式的环境变量
        if k.startswith('SM_HP_'):
            # 转换参数名
            param_name = k[6:].lower().replace('-', '_')
            
            # 检查参数是否存在
            if hasattr(args, param_name):
                # 根据参数类型转换值
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
                    
                # 处理特殊情况 - 模型列表
                if param_name == 'models':
                    args.all_models = [m.strip() for m in v.split(',')]
                    
                # 处理特殊情况 - 测试分割
                if param_name == 'test_splits' and v != 'all':
                    args.test_splits = [ts.strip() for ts in v.split(',')]
        
        # 检查S3输出路径
        if k == 'SAGEMAKER_S3_OUTPUT' and args.save_to_s3 is None:
            args.save_to_s3 = v
            print(f"从环境变量设置S3输出路径: {v}")
    
    # 如果在SageMaker中运行，确保目录存在
    if is_sagemaker:
        os.makedirs(os.path.join(args.output_dir, args.task_name), exist_ok=True)
    
    # 输出参数
    print("\n===== 参数 =====")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print("================\n")
    
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
        
        # 检查S3保存参数，如果没有设置，尝试从环境变量获取
        if args.save_to_s3 is None:
            logger.info("未设置save_to_s3参数，尝试从环境变量获取S3输出路径")
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
        
        # 直接上传最终结果到S3（默认行为）
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
                        for exp_id in sorted(os.listdir(item_path))[:5]:  # 仅显示前5个条目
                            logger.info(f"│   ├── {exp_id}/ (实验ID)")
                        if len(os.listdir(item_path)) > 5:
                            logger.info(f"│   ├── ... 以及更多 ({len(os.listdir(item_path)) - 5} 个条目)")
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
                
                # 创建在SM_MODEL_DIR中的标记，表明我们使用直接上传
                if 'SM_MODEL_DIR' in os.environ:
                    model_dir = os.environ['SM_MODEL_DIR']
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # 创建标记文件和README
                    marker_path = os.path.join(model_dir, '.direct_s3_upload')
                    with open(marker_path, 'w') as f:
                        f.write(f"Results directly uploaded to S3 at: {s3_task_path}\n")
                        f.write(f"Upload timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Task: {args.task_name}\n")
                        f.write(f"Total files: {len(all_files)}\n")
                    
                    # 将结果摘要也复制到model_dir，以便在model.tar.gz中至少有一些信息
                    if os.path.exists(results_path):
                        model_results_path = os.path.join(model_dir, 'multi_model_results.json')
                        import shutil
                        shutil.copy2(results_path, model_results_path)
                        logger.info(f"结果摘要已复制到model目录: {model_results_path}")
                
                # 执行上传
                upload_success = upload_to_s3(task_dir, s3_task_path)
                if upload_success:
                    logger.info(f"结果成功上传到 {s3_task_path}")
                    
                    # 将上传成功标记写入model_dir
                    if 'SM_MODEL_DIR' in os.environ:
                        success_marker = os.path.join(os.environ['SM_MODEL_DIR'], '.upload_success')
                        with open(success_marker, 'w') as f:
                            f.write(f"Upload successful to: {s3_task_path}\n")
                            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                else:
                    logger.error(f"上传到 {s3_task_path} 失败!")
                    
                    # 在model_dir中记录上传失败
                    if 'SM_MODEL_DIR' in os.environ:
                        failure_marker = os.path.join(os.environ['SM_MODEL_DIR'], '.upload_failure')
                        with open(failure_marker, 'w') as f:
                            f.write(f"Upload failed to: {s3_task_path}\n")
                            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("Check logs for details on the failure.\n")
            
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
                        for item in sorted(os.listdir(possible_dir))[:10]:  # 只显示前10个文件
                            logger.info(f"  - {item}")
                        if len(os.listdir(possible_dir)) > 10:
                            logger.info(f"  - ... 以及更多 ({len(os.listdir(possible_dir)) - 10} 个文件)")
                        
                        # 如果找到合适的目录，尝试上传
                        if args.task_name in possible_dir or possible_dir.endswith('/model'):
                            alt_s3_path = f"{args.save_to_s3.rstrip('/')}/{os.path.basename(possible_dir)}"
                            logger.info(f"尝试上传替代目录: {possible_dir} -> {alt_s3_path}")
                            alt_success = upload_to_s3(possible_dir, alt_s3_path)
                            if alt_success:
                                logger.info(f"成功上传替代目录到 {alt_s3_path}")
                                
                                # 记录上传成功
                                if 'SM_MODEL_DIR' in os.environ:
                                    alt_success_marker = os.path.join(os.environ['SM_MODEL_DIR'], '.alt_upload_success')
                                    with open(alt_success_marker, 'w') as f:
                                        f.write(f"Alternative upload successful to: {alt_s3_path}\n")
                                        f.write(f"Source directory: {possible_dir}\n")
                                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            logger.warning("未设置save_to_s3参数，无法直接上传到S3。将依赖SageMaker默认机制上传模型")
        
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