#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数调优脚本

此脚本提供三种超参数调优方法：
1. 网格搜索：系统地搜索所有参数组合
2. 随机搜索：在给定范围内随机采样参数组合
3. 贝叶斯优化：使用Optuna库实现的贝叶斯优化

用法:
    python scripts/hyperparameter_tuning.py 
    --task_name MotionSourceRecognition 
    --model_name vit 
    --search_method optuna 
    --num_trials 20
"""

import os
import sys
import json
import argparse
import hashlib
import numpy as np
import subprocess
import itertools
import pandas as pd
from datetime import datetime
import random
from tqdm import tqdm
import optuna
import copy
import re

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_to_json_serializable(obj):
    """
    递归地将所有NumPy类型转换为Python原生类型，以便JSON序列化
    
    Args:
        obj: 需要转换的对象
        
    Returns:
        转换后的JSON可序列化对象
    """
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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="超参数调优系统")
    
    # 任务和模型参数
    parser.add_argument('--task_name', type=str, required=True,
                        help='要训练的任务名称')
    parser.add_argument('--model_name', type=str, required=True,
                        help='要训练的模型名称')
    
    # 调优方法选择
    parser.add_argument('--search_method', type=str, default='grid',
                        choices=['grid', 'random', 'optuna'],
                        help='超参数搜索方法: grid(网格搜索), random(随机搜索), optuna(贝叶斯优化)')
    
    # 搜索范围控制参数
    parser.add_argument('--num_trials', type=int, default=20,
                        help='随机搜索或贝叶斯优化的试验次数')
    
    # 数据和存储参数
    parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='存储结果的目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='存储模型检查点的目录')
    
    # 固定参数
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                        help='h5文件中CSI数据的键')
    
    # 网格搜索参数范围定义
    parser.add_argument('--learning_rates', type=str, default='0.001,0.0005,0.0001',
                        help='学习率集合，用逗号分隔')
    parser.add_argument('--batch_sizes', type=str, default='16,32,64',
                        help='批量大小集合，用逗号分隔')
    parser.add_argument('--weight_decays', type=str, default='1e-5,1e-4,1e-3',
                        help='权重衰减集合，用逗号分隔')
    parser.add_argument('--dropout_rates', type=str, default='0.1,0.3,0.5',
                        help='Dropout率集合，用逗号分隔')
    
    # 随机搜索和贝叶斯优化的参数范围
    parser.add_argument('--lr_min', type=float, default=0.0001,
                        help='学习率最小值')
    parser.add_argument('--lr_max', type=float, default=0.01,
                        help='学习率最大值')
    parser.add_argument('--batch_size_min', type=int, default=16,
                        help='批量大小最小值')
    parser.add_argument('--batch_size_max', type=int, default=128,
                        help='批量大小最大值')
    parser.add_argument('--weight_decay_min', type=float, default=1e-6,
                        help='权重衰减最小值')
    parser.add_argument('--weight_decay_max', type=float, default=1e-3,
                        help='权重衰减最大值')
    parser.add_argument('--dropout_min', type=float, default=0.0,
                        help='Dropout率最小值')
    parser.add_argument('--dropout_max', type=float, default=0.5,
                        help='Dropout率最大值')
    
    # 其他超参数，可以根据需要添加
    parser.add_argument('--win_len', type=int, default=250,
                        help='CSI数据窗口长度')
    parser.add_argument('--feature_size', type=int, default=98,
                        help='CSI数据特征大小')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='输入通道数')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='ViT模型的嵌入维度')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer模型的模型维度')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='预热轮数')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')
    
    return parser.parse_args()

def generate_grid_search_params(args):
    """生成网格搜索的参数组合"""
    # 解析参数字符串为列表
    learning_rates = [float(lr) for lr in args.learning_rates.split(',')]
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
    weight_decays = [float(wd) for wd in args.weight_decays.split(',')]
    dropout_rates = [float(dr) for dr in args.dropout_rates.split(',')]
    
    # 创建所有参数组合
    param_grid = []
    for lr, bs, wd, dr in itertools.product(learning_rates, batch_sizes, weight_decays, dropout_rates):
        params = {
            'learning_rate': lr,
            'batch_size': bs,
            'weight_decay': wd,
            'dropout': dr,
            'win_len': args.win_len,
            'feature_size': args.feature_size,
            'in_channels': args.in_channels,
            'emb_dim': args.emb_dim,
            'd_model': args.d_model,
            'epochs': args.epochs,
            'warmup_epochs': args.warmup_epochs,
            'patience': args.patience
        }
        param_grid.append(params)
    
    return param_grid

def generate_random_search_params(args):
    """生成随机搜索的参数组合"""
    param_grid = []
    
    for _ in range(args.num_trials):
        # 对连续参数使用对数均匀分布
        lr = np.exp(np.random.uniform(np.log(args.lr_min), np.log(args.lr_max)))
        wd = np.exp(np.random.uniform(np.log(args.weight_decay_min), np.log(args.weight_decay_max)))
        
        # 对离散参数使用随机选择
        bs = np.random.randint(args.batch_size_min, args.batch_size_max + 1)
        # 确保批量大小是8的倍数（对GPU通常有利）
        bs = max(8, bs - (bs % 8))
        
        # 对dropout使用均匀分布
        dr = np.random.uniform(args.dropout_min, args.dropout_max)
        
        params = {
            'learning_rate': float(lr),
            'batch_size': int(bs),
            'weight_decay': float(wd),
            'dropout': float(dr),
            'win_len': args.win_len,
            'feature_size': args.feature_size,
            'in_channels': args.in_channels,
            'emb_dim': args.emb_dim,
            'd_model': args.d_model,
            'epochs': args.epochs,
            'warmup_epochs': args.warmup_epochs,
            'patience': args.patience
        }
        param_grid.append(params)
    
    return param_grid

def train_with_params(params, args):
    """使用给定参数训练模型并返回性能指标"""
    cmd = [
        "python", "scripts/train_supervised.py",
        "--data_dir", args.data_dir,
        "--task_name", args.task_name,
        "--model_name", args.model_name,
        "--data_key", args.data_key,
        "--save_dir", args.save_dir,
        "--output_dir", args.output_dir,
        "--learning_rate", str(params['learning_rate']),
        "--batch_size", str(params['batch_size']),
        "--epochs", str(params['epochs']),
        "--weight_decay", str(params['weight_decay']),
        "--dropout", str(params['dropout']),
        "--win_len", str(params['win_len']),
        "--feature_size", str(params['feature_size']),
        "--in_channels", str(params['in_channels']),
        "--emb_dim", str(params['emb_dim']),
        "--d_model", str(params['d_model']),
        "--warmup_epochs", str(params['warmup_epochs']),
        "--patience", str(params['patience'])
    ]
    
    print(f"执行训练命令: {' '.join(cmd)}")
    
    try:
        # 执行命令并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 解析输出以提取结果
        output = result.stdout
        
        # 尝试找到实验ID
        experiment_id = None
        for line in output.split('\n'):
            if "Experiment ID:" in line:
                parts = line.split("Experiment ID:")
                if len(parts) > 1:
                    experiment_id = parts[1].strip()
        
        # 尝试从结果JSON文件中提取准确率
        results_dir = os.path.join(args.output_dir, args.task_name, args.model_name, experiment_id if experiment_id else "")
        summary_file = os.path.join(results_dir, f"{args.model_name}_{args.task_name}_summary.json")
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                
            # 寻找测试集准确率
            test_acc = None
            for key, value in summary.items():
                if "test" in key.lower() and "accuracy" in key.lower():
                    test_acc = value
                    break
            
            return {
                'params': params,
                'accuracy': test_acc,
                'experiment_id': experiment_id,
                'success': True
            }
        else:
            print(f"警告: 未找到结果文件 {summary_file}")
            return {
                'params': params,
                'accuracy': 0.0,
                'experiment_id': experiment_id,
                'success': False
            }
    
    except subprocess.CalledProcessError as e:
        print(f"训练失败，错误: {e}")
        print(f"错误信息: {e.stderr}")
        return {
            'params': params,
            'accuracy': 0.0,
            'experiment_id': None,
            'success': False
        }

def optuna_objective(trial, args):
    """Optuna优化目标函数"""
    # 使用Optuna建议的参数
    params = {
        'learning_rate': trial.suggest_float('learning_rate', args.lr_min, args.lr_max, log=True),
        'batch_size': 2 ** trial.suggest_int('batch_size_exp', int(np.log2(args.batch_size_min)), int(np.log2(args.batch_size_max))),
        'weight_decay': trial.suggest_float('weight_decay', args.weight_decay_min, args.weight_decay_max, log=True),
        'dropout': trial.suggest_float('dropout', args.dropout_min, args.dropout_max),
        'win_len': args.win_len,
        'feature_size': args.feature_size,
        'in_channels': args.in_channels,
        'emb_dim': args.emb_dim,
        'd_model': args.d_model,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience
    }
    
    # 训练并获取结果
    result = train_with_params(params, args)
    
    # 如果训练失败，返回一个很低的分数
    if not result['success'] or result['accuracy'] is None:
        return 0.0
    
    return result['accuracy']

def run_optuna_optimization(args):
    """运行Optuna贝叶斯优化"""
    # 创建Optuna学习任务
    study = optuna.create_study(direction='maximize')
    
    # 使用闭包传递args参数
    objective_with_args = lambda trial: optuna_objective(trial, args)
    
    # 运行优化
    study.optimize(objective_with_args, n_trials=args.num_trials)
    
    # 获取最佳参数
    best_params = study.best_params
    
    # 转换成我们使用的格式
    params = {
        'learning_rate': best_params['learning_rate'],
        'batch_size': 2 ** best_params['batch_size_exp'],
        'weight_decay': best_params['weight_decay'],
        'dropout': best_params['dropout'],
        'win_len': args.win_len,
        'feature_size': args.feature_size,
        'in_channels': args.in_channels,
        'emb_dim': args.emb_dim,
        'd_model': args.d_model,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience
    }
    
    # 创建结果对象
    return [{
        'params': params,
        'accuracy': study.best_value,
        'experiment_id': None  # Optuna不跟踪实验ID
    }]

def generate_summary(trials_results, model, task, output_dir, search_method):
    """生成超参数调优结果摘要"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 确保结果目录存在
    summary_dir = os.path.join(output_dir, task, model, "hyperparameter_tuning")
    os.makedirs(summary_dir, exist_ok=True)
    
    # 创建摘要数据
    summary = {
        "timestamp": timestamp,
        "model": model,
        "task": task,
        "search_method": search_method,
        "trials": trials_results,
        "best_trial": None
    }
    
    # 找出最佳试验
    if trials_results:
        # 按测试准确率排序
        sorted_trials = sorted(trials_results, key=lambda x: x.get("test_accuracy", 0.0), reverse=True)
        summary["best_trial"] = sorted_trials[0]
    
    # 确保所有数据都是JSON可序列化的
    summary = convert_to_json_serializable(summary)
    
    # 保存为JSON文件
    summary_file = os.path.join(summary_dir, f"hparam_summary_{search_method}_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    
    # 将试验结果转换为DataFrame并保存为CSV
    if trials_results:
        trials_df = pd.DataFrame(trials_results)
        csv_file = os.path.join(summary_dir, f"hparam_trials_{search_method}_{timestamp}.csv")
        trials_df.to_csv(csv_file, index=False)
        print(f"保存了{len(trials_results)}个试验结果到 {csv_file}")
    
    print(f"超参数调优结果已保存到 {summary_file}")
    return summary

def main():
    """主函数"""
    args = parse_args()
    
    print(f"开始对任务 {args.task_name} 使用模型 {args.model_name} 进行超参数调优")
    print(f"调优方法: {args.search_method}")
    
    # 根据选择的搜索方法生成参数组合
    if args.search_method == 'grid':
        print("使用网格搜索...")
        param_grid = generate_grid_search_params(args)
        print(f"生成了 {len(param_grid)} 个参数组合")
        
        # 执行网格搜索
        trials_results = []
        for i, params in enumerate(tqdm(param_grid, desc="网格搜索进度")):
            print(f"\n执行试验 {i+1}/{len(param_grid)}")
            result = train_with_params(params, args)
            trials_results.append(result)
            
    elif args.search_method == 'random':
        print("使用随机搜索...")
        param_grid = generate_random_search_params(args)
        print(f"生成了 {args.num_trials} 个随机参数组合")
        
        # 执行随机搜索
        trials_results = []
        for i, params in enumerate(tqdm(param_grid, desc="随机搜索进度")):
            print(f"\n执行试验 {i+1}/{args.num_trials}")
            result = train_with_params(params, args)
            trials_results.append(result)
            
    elif args.search_method == 'optuna':
        print("使用Optuna贝叶斯优化...")
        print(f"将执行 {args.num_trials} 次试验")
        
        # 执行Optuna贝叶斯优化
        trials_results = run_optuna_optimization(args)
    
    # 生成摘要
    summary = generate_summary(trials_results, args.model_name, args.task_name, args.output_dir, args.search_method)
    
    # 打印结果
    print("\n超参数调优完成!")
    print(f"最佳准确率: {summary['best_accuracy']}")
    print("最佳参数:")
    for key, value in summary['best_params'].items():
        print(f"  {key}: {value}")
    print(f"\n详细结果已保存到 {args.output_dir}/{args.task_name}/{args.model_name}/hyperparameter_tuning/")

if __name__ == "__main__":
    main() 