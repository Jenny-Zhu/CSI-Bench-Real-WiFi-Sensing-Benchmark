#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型训练脚本 - SageMaker版本

此脚本可在单个训练作业中训练多个模型，用于在SageMaker中执行批量训练。
它基于原始的train_supervised.py，但增加了对多个模型的支持。

每个模型的训练结果将保存在各自的子目录中。
"""

import os
import sys
import json
import time
import argparse
import logging
import traceback
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入项目模块
from models import get_model
from load.supervised import load_csi_supervised, load_acf_supervised
from utils.metrics import compute_metrics
from utils.misc import set_seed, get_learning_rate, create_dir_if_not_exists
from utils.early_stopping import EarlyStopping

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Multi-model WiFi Sensing Training')
    
    # 数据参数
    parser.add_argument('--training-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--results-subdir', type=str, default='supervised')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    
    # 模型参数
    parser.add_argument('--mode', type=str, default='csi', choices=['csi', 'acf'])
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--model-name', type=str, default='Transformer')
    parser.add_argument('--all-models', type=str, nargs='+', default=None, 
                      help='Space-separated list of models to train')
    
    # 集成加载器选项
    parser.add_argument('--integrated-loader', action='store_true')
    parser.add_argument('--task', type=str, default=None)
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--win-len', type=int, default=250)
    parser.add_argument('--feature-size', type=int, default=98)
    
    # SageMaker 参数
    parser.add_argument('--test-dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # 解析参数
    args, _ = parser.parse_known_args()
    return args

def load_data(args):
    """根据模式加载数据集"""
    logger.info(f"Loading {args.mode.upper()} data from {args.training_dir}")
    
    # 获取所有测试目录
    test_dirs = []
    for i in range(10):  # 检查至多10个测试通道
        channel_name = f'SM_CHANNEL_TEST{i+1}' if i > 0 else 'SM_CHANNEL_TEST'
        if channel_name in os.environ:
            test_dirs.append(os.environ[channel_name])
    
    # 如果没有环境变量中的测试目录，使用args.test_dir
    if not test_dirs and args.test_dir:
        test_dirs = [args.test_dir]
    
    logger.info(f"Test directories: {test_dirs}")
    
    # 根据模式选择合适的加载函数
    if args.mode == 'csi':
        train_loader, val_loader, test_loaders = load_csi_supervised(
            args.training_dir, 
            train_batch_size=args.batch_size,
            task=args.task,
            test_dirs=test_dirs
        )
    elif args.mode == 'acf':
        train_loader, val_loader, test_loaders = load_acf_supervised(
            args.training_dir, 
            train_batch_size=args.batch_size,
            task=args.task,
            test_dirs=test_dirs
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    
    return train_loader, val_loader, test_loaders

def train_model(model, model_name, device, train_loader, val_loader, test_loaders, args, model_specific_output_dir):
    """训练特定模型"""
    # 创建模型特定的输出目录
    os.makedirs(model_specific_output_dir, exist_ok=True)
    
    # 模型移动到设备
    model = model.to(device)
    
    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(model_specific_output_dir, 'best_model.pth'))
    
    # 训练指标记录
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # 训练循环
    logger.info(f"Starting training for model: {model_name}")
    logger.info(f"Training on device: {device}, with {len(train_loader.dataset)} training samples")
    logger.info(f"Validation set size: {len(val_loader.dataset)} samples")
    
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)
            
            _, predicted = outputs.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                          f" ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        
        # 计算平均训练损失
        train_loss /= len(train_loader.dataset)
        train_acc = 100. * train_correct / train_total
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item() * data.size(0)
                
                _, predicted = outputs.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # 计算平均验证损失和准确率
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 打印进度
        logger.info(f'Epoch: {epoch}/{args.num_epochs-1}')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(model_specific_output_dir, 'best_model.pth')))
    
    # 训练总时间
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}min)")
    
    # 在测试集上评估
    logger.info("Evaluating on test sets")
    test_results = {}
    
    model.eval()
    for i, test_loader in enumerate(test_loaders):
        test_name = f"test_{i+1}"
        test_set_size = len(test_loader.dataset)
        logger.info(f"Evaluating on {test_name}, {test_set_size} samples")
        
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                
                _, predicted = outputs.max(1)
                
                all_targets.extend(target.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        # 计算指标
        metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
        test_results[test_name] = metrics
        
        logger.info(f"Test set {test_name} results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
    
    # 保存训练历史和测试结果
    results = {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_results': test_results,
        'training_time': total_time,
        'params': vars(args)
    }
    
    # 保存结果
    with open(os.path.join(model_specific_output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 保存模型配置
    model_config = {
        'model_name': model_name,
        'mode': args.mode,
        'num_classes': args.num_classes,
        'feature_size': args.feature_size,
        'win_len': args.win_len
    }
    
    with open(os.path.join(model_specific_output_dir, 'model_config.json'), 'w') as f:
        json.dump(model_config, f, indent=4)
    
    logger.info(f"Results saved to {model_specific_output_dir}")
    return results

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确定要训练的模型列表
    if args.all_models:
        # 使用命令行提供的所有模型
        models_to_train = args.all_models
    else:
        # 只使用默认模型
        models_to_train = [args.model_name]
    
    logger.info(f"Will train the following models: {', '.join(models_to_train)}")
    
    # 加载数据集（只需加载一次，所有模型共用）
    train_loader, val_loader, test_loaders = load_data(args)
    
    # 创建汇总结果
    all_results = {}
    
    # 设置输出根目录
    output_root = args.output_dir
    
    # 确保目录结构正确
    create_dir_if_not_exists(output_root)
    
    # 循环训练每个模型
    for model_name in models_to_train:
        try:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Training model: {model_name}")
            logger.info(f"{'=' * 50}")
            
            # 为每个模型创建特定的输出目录
            model_specific_output_dir = os.path.join(output_root, args.results_subdir, model_name)
            create_dir_if_not_exists(model_specific_output_dir)
            
            # 初始化模型
            model = get_model(
                model_name=model_name,
                num_classes=args.num_classes,
                in_channels=args.feature_size,
                seq_len=args.win_len,
                mode=args.mode
            )
            
            # 计算模型参数量
            model_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model {model_name} has {model_params:,} parameters")
            
            # 冻结主干网络（如果需要）
            if args.freeze_backbone:
                for name, param in model.named_parameters():
                    if "classifier" not in name:
                        param.requires_grad = False
                logger.info("Backbone network frozen")
            
            # 训练当前模型
            results = train_model(
                model, 
                model_name, 
                args.device, 
                train_loader, 
                val_loader, 
                test_loaders, 
                args,
                model_specific_output_dir
            )
            
            all_results[model_name] = {
                'val_acc': max(results['val_accuracies']),
                'test_results': results['test_results']
            }
            
            logger.info(f"Model {model_name} training completed.")
            
        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            # 继续训练下一个模型
    
    # 保存汇总结果
    summary_path = os.path.join(output_root, args.results_subdir, 'all_models_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logger.info(f"All models training completed. Summary saved to {summary_path}")

if __name__ == "__main__":
    main() 