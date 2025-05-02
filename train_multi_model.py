#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型训练脚本 - 在一个训练作业中训练多个模型架构

这个脚本可以在SageMaker环境中运行，用于训练和评估多个模型架构，处理相同的任务。
"""

import os
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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模型和数据加载器
try:
    from load.supervised.benchmark_loader import load_benchmark_supervised
    # 导入模型类
    from model.supervised.models import (
        MLPClassifier, 
        LSTMClassifier, 
        ResNet18Classifier, 
        TransformerClassifier, 
        ViTClassifier
    )
except ImportError as e:
    logger.error(f"导入失败: {e}")
    sys.exit(1)

# 模型工厂字典
MODEL_TYPES = {
    'mlp': MLPClassifier,
    'lstm': LSTMClassifier,
    'resnet18': ResNet18Classifier,
    'transformer': TransformerClassifier,
    'vit': ViTClassifier
}

# 任务训练器类（从scripts/train_supervised.py中提取）
from engine.supervised.task_trainer import TaskTrainer

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train multiple models on WiFi benchmark dataset')
    
    # 必要参数
    parser.add_argument('--all_models', type=str, default='vit', 
                        help='Space-separated list of models to train. E.g. "mlp lstm resnet18"')
    parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                        help='Name of the task to train on')
    
    # 数据参数
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                        help='Key for CSI data in h5 files')
    
    # 模型参数
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
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default='/opt/ml/model',
                        help='Directory to save checkpoints and models')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (defaults to save_dir if not specified)')
    
    args = parser.parse_args()
    
    # 解析要训练的所有模型
    if ' ' in args.all_models:
        args.all_models = args.all_models.split()
    else:
        args.all_models = [args.all_models]
    
    # 验证模型有效性
    for model_name in args.all_models:
        if model_name.lower() not in MODEL_TYPES:
            logger.error(f"不支持的模型: {model_name}. 有效的模型: {list(MODEL_TYPES.keys())}")
            sys.exit(1)
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = args.save_dir
        
    return args

def set_seed(seed):
    """设置所有随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_model(model_name, data, args, device):
    """训练指定类型的模型"""
    logger.info(f"===== 开始训练 {model_name.upper()} 模型 =====")
    
    # 解包数据
    loaders = data['loaders']
    num_classes = data['num_classes']
    label_mapper = data['label_mapper']
    
    # 获取训练集和验证集
    train_loader = loaders['train']
    val_loader = loaders.get('val')
    
    if val_loader is None:
        logger.warning("未找到验证数据。将使用训练数据作为验证。")
        val_loader = train_loader
    
    # 创建模型
    logger.info(f"创建 {model_name.upper()} 模型...")
    ModelClass = MODEL_TYPES[model_name.lower()]
    
    # 通用模型参数
    model_params = {
        'num_classes': num_classes
    }
    
    # 根据模型类型添加额外参数
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
    
    # 创建模型实例并移动到设备
    model = ModelClass(**model_params).to(device)
    logger.info(f"模型已创建，包含 {sum(p.numel() for p in model.parameters())} 个参数")
    
    # 为每个模型创建特定的保存和输出目录
    # 格式：output_path/task/model/
    model_save_dir = os.path.join(args.save_dir, args.task_name, model_name)
    model_output_dir = os.path.join(args.output_dir, args.task_name, model_name)
    
    # 确保目录存在
    os.makedirs(model_save_dir, exist_ok=True)
    if model_output_dir != model_save_dir:
        os.makedirs(model_output_dir, exist_ok=True)
    
    # 创建配置对象
    config = argparse.Namespace(
        num_epochs=args.num_epochs,
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
    
    # 保存配置
    config_path = os.path.join(model_output_dir, f"{model_name}_{args.task_name}_config.json")
    with open(config_path, "w") as f:
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        json.dump(config_dict, f, indent=4)
    
    logger.info(f"配置已保存到 {config_path}")
    
    # 创建训练器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
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
    
    # 训练模型
    trained_model, training_results = trainer.train()
    
    # 评估测试集
    results = {'model_name': model_name, 'task': args.task_name}
    results.update({
        'best_epoch': training_results['best_epoch'],
        'best_val_accuracy': float(training_results['best_val_accuracy'])
    })
    
    logger.info("\n评估测试集:")
    for key in loaders:
        if key.startswith('test'):
            logger.info(f"\n评估 {key} 集:")
            test_loss, test_acc = trainer.evaluate(loaders[key])
            logger.info(f"{key} 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")
            
            # 为该测试集绘制混淆矩阵
            confusion_path = os.path.join(model_output_dir, f"{model_name}_{args.task_name}_{key}_confusion.png")
            trainer.plot_confusion_matrix(data_loader=loaders[key], mode=key, epoch=None)
            
            # 添加测试结果
            results[f'{key}_loss'] = float(test_loss)
            results[f'{key}_accuracy'] = float(test_acc)
    
    # 保存训练结果摘要
    results_file = os.path.join(model_output_dir, f"{model_name}_{args.task_name}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"模型训练完成，结果保存至 {results_file}")
    
    return results

def main():
    """主函数：处理参数并启动多模型训练"""
    # 解析命令行参数
    args = get_args()
    
    # 检测是否在SageMaker环境中运行
    is_sagemaker = os.path.exists('/opt/ml/model')
    if is_sagemaker:
        logger.info("在SageMaker环境中运行")
    else:
        logger.info("在本地环境中运行")
    
    # 设置输出目录格式为 output_path/task/ 
    task_output_dir = os.path.join(args.output_dir, args.task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    
    # 设置保存目录（用于模型检查点）
    task_save_dir = os.path.join(args.save_dir, args.task_name)
    os.makedirs(task_save_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据（只需加载一次，所有模型共享相同的数据）
    logger.info(f"从 {args.data_dir} 加载任务 {args.task_name} 的数据...")
    data = load_benchmark_supervised(
        dataset_root=args.data_dir,
        task_name=args.task_name,
        batch_size=args.batch_size,
        file_format="h5",
        data_key=args.data_key,
        num_workers=4
    )
    
    logger.info(f"数据集包含 {data['num_classes']} 个类别")
    
    # 对每个模型进行训练
    all_results = []
    start_time = time.time()
    
    for i, model_name in enumerate(args.all_models):
        model_name = model_name.lower()
        logger.info(f"模型 {i+1}/{len(args.all_models)}: {model_name.upper()}")
        model_results = train_model(model_name, data, args, device)
        all_results.append(model_results)
    
    total_time = time.time() - start_time
    logger.info(f"全部 {len(args.all_models)} 个模型训练完成，总耗时: {total_time:.2f} 秒")
    
    # 保存所有模型的汇总结果
    summary_results = {
        'task': args.task_name,
        'models': args.all_models,
        'total_time': total_time,
        'results': all_results
    }
    
    # 将汇总结果保存在任务目录下
    summary_file = os.path.join(task_output_dir, f"all_models_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_results, f, indent=4)
    
    logger.info(f"汇总结果已保存至 {summary_file}")
    
    # 如果在SageMaker中运行，打印额外信息
    if is_sagemaker:
        logger.info("训练作业完成后，SageMaker将自动上传模型和结果到S3")

if __name__ == "__main__":
    main() 