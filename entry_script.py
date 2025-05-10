#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SageMaker Entry Script - WiFi Sensing Benchmark

简化版入口脚本，直接运行训练代码，无需额外标记和设置
"""

import os
import sys
import subprocess
import gc
import logging

print("\n==========================================")
print("Starting custom entry script entry_script.py")
print("==========================================\n")

# 简化环境变量设置
os.environ['FORCE_DIRECT_S3_UPLOAD'] = 'true'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 设置matplotlib后端为Agg，避免显示问题
try:
    import matplotlib
    matplotlib.use('Agg')
    print("Set matplotlib backend to Agg")
except Exception as e:
    print(f"Warning: Failed to set matplotlib backend: {e}")

# Improve memory usage
print("Configuring memory optimization settings...")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 启用垃圾回收
gc.enable()
print("Active garbage collection enabled")

# 设置路径
print("Setting up paths...")
sys.path.insert(0, os.getcwd())

# 检查要执行的脚本
script_to_run = os.environ.get('SAGEMAKER_PROGRAM', 'train_multi_model.py')
print(f"Script to execute: {script_to_run}")

# 检查脚本是否存在
if not os.path.exists(script_to_run):
    print(f"Error: Script {script_to_run} not found")
    print(f"Python files in current directory: {[f for f in os.listdir('.') if f.endswith('.py')]}")
    sys.exit(1)

# 打印环境信息
print("\n===== Environment Information =====")
import platform
print(f"Python version: {platform.python_version()}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {', '.join(os.listdir('.'))[:200]}...")

# 尝试导入PyTorch来验证它加载正确
try:
    print("Attempting to import PyTorch...")
    import torch
    # 配置PyTorch内存分配器优化
    if torch.cuda.is_available():
        # 限制GPU内存增长
        torch.cuda.set_per_process_memory_fraction(0.7)  # 使用70%的可用GPU内存
        # 主动清空CUDA缓存
        torch.cuda.empty_cache()
        
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU total memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
except Exception as e:
    print(f"Warning when importing PyTorch: {e}")

# 构建命令来运行实际的训练脚本
cmd = [sys.executable, script_to_run]

# 从环境变量收集超参数
for key, value in os.environ.items():
    if key.startswith('SM_HP_'):
        # 转换 SM_HP_X 到 x
        param_name = key[6:].lower()
        
        # 添加超参数到命令行
        # 重要：确保我们使用正确的双破折号前缀
        param_prefix = "--"
        
        # 处理特殊的布尔值并归一化它们
        if value.lower() in ('true', 'yes', '1', 't', 'y'):
            cmd.append(f"{param_prefix}{param_name}")
        elif value.lower() in ('false', 'no', '0', 'f', 'n'):
            # 对于False布尔值，不添加参数
            pass
        else:
            cmd.append(f"{param_prefix}{param_name}")
            
            # 删除可能导致解析问题的不必要引号
            if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
                
            cmd.append(value)

# 其他重要参数
for param, env_var in [
    ('save_to_s3', 'SAGEMAKER_S3_OUTPUT'),
    ('data_root', 'SM_CHANNEL_TRAINING')
]:
    if env_var in os.environ and not any(arg == f"--{param}" for arg in cmd):
        cmd.append(f"--{param}")
        cmd.append(os.environ[env_var])

print(f"Final command: {' '.join(cmd)}")

# 执行带参数的训练脚本
try:
    # 执行脚本并等待它完成
    exit_code = subprocess.call(cmd)
    sys.exit(exit_code)
except Exception as e:
    print(f"Error executing training script: {e}")
    sys.exit(1) 