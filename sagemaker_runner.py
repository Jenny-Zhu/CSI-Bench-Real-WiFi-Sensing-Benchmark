#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Sensing Pipeline Runner - SageMaker Environment

This script allows you to run the supervised learning pipeline in the SageMaker environment.
It creates a SageMaker PyTorch Estimator for submitting training jobs.

主要功能:
1. 批量执行训练任务，每个任务(task)使用单一实例运行多个模型
2. 支持使用 JSON 配置文件覆盖默认设置

用法示例:
```
import sagemaker_runner
runner = sagemaker_runner.SageMakerRunner()
runner.run_batch_by_task(tasks=['MotionSourceRecognition'], models=['vit', 'transformer'])
```
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import boto3
import re
import numpy as np
import pandas as pd

# 默认路径设置
CODE_DIR = os.path.dirname(os.path.abspath(__file__))  # 包含代码的目录
CONFIG_DIR = os.path.join(CODE_DIR, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_DIR, "sagemaker_default_config.json")

# 从JSON文件加载默认配置
def load_default_config():
    """从JSON配置文件加载默认配置"""
    try:
        with open(DEFAULT_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        print(f"已从 {DEFAULT_CONFIG_PATH} 加载默认配置")
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"警告: 无法加载默认配置文件: {e}")
        print("使用硬编码的默认值代替.")
        # 回退到硬编码的默认值
        return {
            "s3_data_base": "s3://rnd-sagemaker/Data/Benchmark/",
            "s3_output_base": "s3://rnd-sagemaker/Benchmark_Log/",
            "mode": "csi",
            "task": "MotionSourceRecognition",
            "win_len": 250,
            "feature_size": 98,
            "seed": 42,
            "batch_size": 8,
            "num_epochs": 10,
            "model_name": "transformer", 
            "instance_type": "ml.g4dn.xlarge",
            "instance_count": 1,
            "framework_version": "1.12.1",
            "py_version": "py38",
            "base_job_name": "wifi-sensing-supervised",
            "batch_wait_time": 30,
            "task_class_mapping": {
                "HumanNonhuman": 2, 
                "MotionSourceRecognition": 4, 
                "NTUHumanID": 15, 
                "NTUHAR": 6, 
                "HumanID": 4, 
                "Widar": 22,
                "HumanMotion": 3, 
                "ThreeClass": 3, 
                "DetectionandClassification": 5, 
                "Detection": 2,
                "demo": 2
            },
            "available_models": ["mlp", "lstm", "resnet18", "transformer", "vit"],
            "available_tasks": ["MotionSourceRecognition", "HumanMotion", "DetectionandClassification", "HumanID", "NTUHAR", "HumanNonhuman", "NTUHumanID", "Widar", "ThreeClass", "Detection"]
        }

# 加载默认配置
DEFAULT_CONFIG = load_default_config()

# 提取配置值
S3_DATA_BASE = DEFAULT_CONFIG.get("s3_data_base", "s3://rnd-sagemaker/Data/Benchmark/")
S3_OUTPUT_BASE = DEFAULT_CONFIG.get("s3_output_base", "s3://rnd-sagemaker/Benchmark_Log/")
AVAILABLE_TASKS = DEFAULT_CONFIG.get("available_tasks", [
    "MotionSourceRecognition", "HumanMotion", "DetectionandClassification", 
    "HumanID", "NTUHAR", "HumanNonhuman", "NTUHumanID", "Widar", "ThreeClass", "Detection"
])
AVAILABLE_MODELS = DEFAULT_CONFIG.get("available_models", ["mlp", "lstm", "resnet18", "transformer", "vit"])
TASK_CLASS_MAPPING = DEFAULT_CONFIG.get("task_class_mapping", {})

# SageMaker设置
INSTANCE_TYPE = DEFAULT_CONFIG.get("instance_type", "ml.g4dn.xlarge")
INSTANCE_COUNT = DEFAULT_CONFIG.get("instance_count", 1)
FRAMEWORK_VERSION = DEFAULT_CONFIG.get("framework_version", "1.12.1")
PY_VERSION = DEFAULT_CONFIG.get("py_version", "py38")
BASE_JOB_NAME = DEFAULT_CONFIG.get("base_job_name", "wifi-sensing-supervised")

# 数据模态
MODE = DEFAULT_CONFIG.get("mode", "csi")

# 默认任务
TASK = DEFAULT_CONFIG.get("task", "MotionSourceRecognition")

# 模型参数
WIN_LEN = DEFAULT_CONFIG.get("win_len", 250)
FEATURE_SIZE = DEFAULT_CONFIG.get("feature_size", 98)

# 通用训练参数
SEED = DEFAULT_CONFIG.get("seed", 42)
BATCH_SIZE = DEFAULT_CONFIG.get("batch_size", 8)
EPOCH_NUMBER = DEFAULT_CONFIG.get("num_epochs", 10)
PATIENCE = DEFAULT_CONFIG.get("patience", 15)
MODEL_NAME = DEFAULT_CONFIG.get("model_name", "transformer")

# 批量设置
BATCH_WAIT_TIME = DEFAULT_CONFIG.get("batch_wait_time", 30)

class SageMakerRunner:
    """处理SageMaker训练作业创建和执行的类"""
    
    def __init__(self, role=None):
        """初始化SageMaker会话和角色"""
        self.session = sagemaker.Session()
        self.role = role or sagemaker.get_execution_role()
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        
        # 验证默认配置项
        self.default_config = DEFAULT_CONFIG
        
        # 验证rnd-sagemaker存储桶是否存在
        s3 = boto3.resource('s3')
        bucket_name = "rnd-sagemaker"
        if bucket_name not in [bucket.name for bucket in s3.buckets.all()]:
            print(f"错误: 存储桶 '{bucket_name}' 不存在. 请先创建.")
            sys.exit(1)
        
        print(f"SageMaker Runner 已初始化:")
        print(f"  S3 数据基础路径: {S3_DATA_BASE}")
        print(f"  S3 输出基础路径: {S3_OUTPUT_BASE}")
        print(f"  可用任务: {', '.join(AVAILABLE_TASKS)}")
        print(f"  可用模型: {', '.join(AVAILABLE_MODELS)}")
        print(f"  时间戳: {self.timestamp}")
    
    def convert_to_json_serializable(self, obj):
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
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    def run_batch_by_task(self, tasks=None, models=None, mode='csi', instance_type=None, wait_time=BATCH_WAIT_TIME):
        """
        按任务批量运行训练作业（每个任务使用一个实例运行所有模型）
        
        Args:
            tasks (list): 要运行的任务列表。如果为None，则使用所有可用任务
            models (list): 要运行的模型列表。如果为None，则使用所有可用模型
            mode (str): 数据模态 ('csi' 或 'acf')
            instance_type (str): SageMaker实例类型
            wait_time (int): 作业提交之间等待的时间（秒）
            
        Returns:
            dict: 包含所有启动作业详情的字典
        """
        print(f"按任务批量执行开始...")
        
        # 使用提供的任务或可用任务
        if tasks is None or len(tasks) == 0:
            tasks_to_run = AVAILABLE_TASKS
        else:
            tasks_to_run = [t for t in tasks if t in AVAILABLE_TASKS]
            if len(tasks_to_run) < len(tasks):
                print(f"警告: 部分请求的任务不在可用任务列表中.")
        
        # 使用提供的模型或可用模型
        if models is None or len(models) == 0:
            models_to_run = AVAILABLE_MODELS
        else:
            models_to_run = [m for m in models if m in AVAILABLE_MODELS]
            if len(models_to_run) < len(models):
                print(f"警告: 部分请求的模型不在可用模型列表中.")
        
        print(f"要运行的任务 ({len(tasks_to_run)}): {', '.join(tasks_to_run)}")
        print(f"要运行的模型 ({len(models_to_run)}): {', '.join(models_to_run)}")
        
        # 创建批处理时间戳以分组作业
        batch_timestamp = self.timestamp  # 对批处理中的所有作业使用相同的时间戳
        
        # 存储所有作业
        all_jobs = []
        task_job_groups = {}
        
        # 为每个任务启动单个训练实例，运行所有模型
        for task_name in tasks_to_run:
            print(f"\n----------------------------")
            print(f"处理任务: {task_name}")
            print(f"----------------------------")
            
            # 确定此任务的类数
            num_classes = TASK_CLASS_MAPPING.get(task_name, 2)
            print(f"任务有{num_classes}个类")
            
            # 构建超参数字典
            hyperparameters = {
                # 数据参数
                "data_dir": S3_DATA_BASE,
                "task_name": task_name,
                "mode": mode,
                
                # 模型列表 - 注意这里是一个新参数，不是标准脚本支持的
                "models": ",".join(models_to_run),  # 以逗号分隔的模型列表
                
                # 训练参数
                "batch_size": BATCH_SIZE,
                "num_epochs": EPOCH_NUMBER,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "warmup_epochs": 5,
                "patience": PATIENCE,
                
                # 模型参数
                "win_len": WIN_LEN,
                "feature_size": FEATURE_SIZE,
                "seed": SEED,
                "save_dir": "/opt/ml/model",  # 使用SageMaker模型目录
                "output_dir": "/opt/ml/model",  # 将output_dir也设置为模型目录
                "data_key": 'CSI_amps'  # 添加data_key参数
            }
            
            # 创建作业名称
            job_name = f"{BASE_JOB_NAME}-{task_name.lower()}-multi-models-{batch_timestamp}"
            job_name = re.sub(r'[^a-zA-Z0-9-]', '-', job_name)  # 将无效字符替换为连字符
            
            # 输出路径
            s3_output_path = f"{S3_OUTPUT_BASE}{task_name}/"
            if not s3_output_path.endswith('/'):
                s3_output_path += '/'
            
            print(f"创建任务 '{task_name}' 的训练作业，运行模型: {', '.join(models_to_run)}")
            print(f"输出路径: {s3_output_path}")
            
            # 创建PyTorch估计器
            instance_type_to_use = instance_type or INSTANCE_TYPE
            
            estimator = PyTorch(
                entry_point="train_multi_model.py",  # 注意: 这里使用了一个新的训练脚本
                source_dir=".",
                role=self.role,
                framework_version=FRAMEWORK_VERSION,
                py_version=PY_VERSION,
                instance_count=INSTANCE_COUNT,
                instance_type=instance_type_to_use,
                max_run=86400 * 3,  # 72小时最大运行时间
                keep_alive_period_in_seconds=1800,  # 训练后保持30分钟活动
                output_path=s3_output_path,
                base_job_name=job_name,
                hyperparameters=hyperparameters,
                metric_definitions=[
                    {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
                    {'Name': 'train:accuracy', 'Regex': 'Train Accuracy: ([0-9\\.]+)'},
                    {'Name': 'validation:loss', 'Regex': 'Val Loss: ([0-9\\.]+)'},
                    {'Name': 'validation:accuracy', 'Regex': 'Val Accuracy: ([0-9\\.]+)'}
                ]
            )
            
            # 准备数据输入
            data_channels = {
                'training': S3_DATA_BASE
            }
            
            # 启动训练作业
            print(f"启动SageMaker训练作业...")
            estimator.fit(inputs=data_channels, job_name=job_name, wait=False)
            
            # 创建作业信息
            job_info = {
                'job_name': job_name,
                'estimator': estimator,
                'inputs': data_channels,
                'config': {
                    'task_name': task_name,
                    'output_dir': s3_output_path,
                    'model_name': 'multi-model'  # 注意这里不再是单个模型名
                },
                'models': models_to_run,
                'batch_id': batch_timestamp,
                'task_group': task_name
            }
            
            # 添加到作业列表
            all_jobs.append(job_info)
            task_job_groups[task_name] = job_info
            
            # 在任务之间等待更长时间
            if wait_time > 0 and task_name != tasks_to_run[-1]:
                print(f"等待 {wait_time} 秒后开始下一个任务...")
                try:
                    time.sleep(wait_time)
                except KeyboardInterrupt:
                    print("\n批处理提交被用户中断.")
                    break
        
        # 返回批处理信息
        batch_info = {
            'batch_timestamp': batch_timestamp,
            'batch_mode': 'by-task',
            'tasks': tasks_to_run,
            'models': models_to_run,
            'instance_type': instance_type or INSTANCE_TYPE,
            'jobs': all_jobs,
            'task_groups': task_job_groups
        }
        
        # 更新批处理摘要以创建初始状态报告
        self._update_batch_summary(all_jobs, batch_timestamp)
        
        print(f"\n批处理执行已启动!")
        print(f"任务数: {len(tasks_to_run)}")
        print(f"模型数: {len(models_to_run)}")
        print(f"总作业数: {len(all_jobs)}")
        print(f"批处理ID: {batch_timestamp}")
        print(f"你可以在SageMaker控制台监控作业.")
        
        return batch_info
    
    def _update_batch_summary(self, jobs, batch_timestamp):
        """更新批处理摘要文件与作业详情"""
        summary_dir = os.path.join(CODE_DIR, "batch_summaries")
        os.makedirs(summary_dir, exist_ok=True)
        
        # 创建文本和JSON摘要
        summary_text_file = os.path.join(summary_dir, f"batch_summary_{batch_timestamp}.txt")
        summary_json_file = os.path.join(summary_dir, f"batch_summary_{batch_timestamp}.json")
        
        # 创建文本摘要
        with open(summary_text_file, "w") as f:
            f.write(f"批处理训练摘要 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"总作业数: {len(jobs)}\n")
            f.write(f"批处理时间戳: {batch_timestamp}\n\n")
            
            for job_info in jobs:
                f.write(f"作业: {job_info['job_name']}\n")
                f.write(f"  输入: {job_info['inputs']['training']}\n")
                f.write(f"  输出: {job_info['config']['output_dir']}\n")
                f.write(f"  任务: {job_info['config']['task_name']}\n")
                
                # 添加有关多个模型的信息（如果可用）
                if 'models' in job_info:
                    f.write(f"  模型: {', '.join(job_info['models'])}\n")
                    
                f.write("\n")
        
        # 创建JSON摘要（更易于程序解析）
        summary_data = {
            "timestamp": batch_timestamp,
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_jobs": len(jobs),
            "jobs": {}
        }
        
        for i, job_info in enumerate(jobs):
            job_key = f"job_{i}_{job_info['config']['task_name']}"
            summary_data["jobs"][job_key] = {
                "job_name": job_info['job_name'],
                "task": job_info['config']['task_name'],
                "input": job_info['inputs']['training'],
                "output": job_info['config']['output_dir']
            }
            
            # 添加有关多个模型的信息（如果可用）
            if 'models' in job_info:
                summary_data["jobs"][job_key]["models"] = job_info['models']
        
        # 确保所有数据都是JSON可序列化的
        summary_data = self.convert_to_json_serializable(summary_data)
        
        with open(summary_json_file, "w") as f:
            json.dump(summary_data, f, indent=2)

def main():
    """从命令行执行的主函数"""
    parser = argparse.ArgumentParser(description='在SageMaker上运行WiFi感知管道')
    parser.add_argument('--tasks', type=str, nargs='+',
                      help='要运行的任务列表。使用空格分隔多个任务')
    parser.add_argument('--models', type=str, nargs='+',
                      help='要运行的模型列表。使用空格分隔多个模型')
    parser.add_argument('--mode', type=str, default=MODE,
                      choices=['csi', 'acf'],
                      help='要使用的数据模态')
    parser.add_argument('--instance-type', dest='instance_type', type=str, default=INSTANCE_TYPE,
                      help='用于训练的SageMaker实例类型')
    parser.add_argument('--batch-wait', dest='batch_wait', type=int, default=BATCH_WAIT_TIME,
                      help='批处理作业提交之间的等待时间（秒）')
    
    args = parser.parse_args()
    
    # 创建SageMaker运行器实例
    runner = SageMakerRunner()
    
    # 确定要使用的任务和模型
    tasks = args.tasks or AVAILABLE_TASKS
    models = args.models or AVAILABLE_MODELS
    
    # 启动批处理
    print(f"运行批处理作业，包含 {len(tasks)} 个任务和 {len(models)} 个模型")
    runner.run_batch_by_task(
        tasks=tasks,
        models=models,
        mode=args.mode,
        instance_type=args.instance_type,
        wait_time=args.batch_wait
    )

if __name__ == "__main__":
    main()
