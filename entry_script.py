#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SageMaker Entry Script - Disable Horovod and SMDebug

This script disables Horovod and SMDebug integration before importing 
PyTorch to avoid version conflicts, then runs the actual training script.
"""

import os
import sys
import importlib.util
import subprocess

# 设置typing_extensions支持，用于解决typing.Literal导入问题
print("设置typing_extensions支持...")
# 确保typing_extensions已可用，并预先导入
try:
    import typing_extensions
    from typing_extensions import Literal
    # 在sys.modules中将Literal注入到typing模块
    if 'typing' in sys.modules:
        sys.modules['typing'].Literal = Literal
    print("成功导入typing_extensions并配置Literal支持")
except ImportError:
    print("警告: 未找到typing_extensions，某些功能可能不可用")

# 手动安装peft库，绕过版本依赖检查
print("正在安装peft库（绕过依赖检查）...")
try:
    # 安装特定版本的peft，使用--no-dependencies参数
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft==0.3.0", "--no-dependencies"])
    print("peft库安装成功")
except Exception as e:
    print(f"安装peft库时出错: {e}")
    # 这不是致命错误，尝试继续执行

# Disable Horovod integration to avoid version conflict
print("正在禁用Horovod集成以避免版本冲突...")
sys.modules['horovod'] = None
sys.modules['horovod.torch'] = None
sys.modules['horovod.tensorflow'] = None

# Disable SMDebug to avoid Horovod dependency
print("正在禁用SageMaker Debugger (smdebug)...")
sys.modules['smdebug'] = None
os.environ['SMDEBUG_DISABLED'] = 'true'
os.environ['SM_DISABLE_DEBUGGER'] = 'true'

# Set paths
print("设置路径...")
sys.path.insert(0, os.getcwd())

# Now run the actual training script
print("准备运行训练脚本...")
from subprocess import check_call

# 识别要执行的脚本
# 检查SAGEMAKER_PROGRAM环境变量
script_to_run = os.environ.get('SAGEMAKER_PROGRAM', 'train_multi_model.py')
print(f"将要执行的脚本: {script_to_run}")

# Get all command line arguments
args = sys.argv[1:]

# Print environment information
print("\n===== 环境信息 =====")
import platform
print(f"Python版本: {platform.python_version()}")
print(f"当前目录: {os.getcwd()}")
print(f"目录中的文件: {', '.join(os.listdir('.'))[:200]}...")
print(f"命令: python3 {script_to_run} {' '.join(args)}")
print("==================================\n")

# Try importing torch to verify it loads correctly without Horovod conflicts
try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print("PyTorch成功导入，没有Horovod冲突")
except Exception as e:
    print(f"导入PyTorch时出错: {e}")
    sys.exit(1)  # Exit if we can't import PyTorch

# 尝试导入peft库确认是否可以正确加载
try:
    import peft
    print(f"PEFT库版本: {peft.__version__}")
    print("PEFT库成功导入")
except Exception as e:
    print(f"导入PEFT库时出错: {e}")
    print("这可能会影响某些功能，但我们将继续执行")

# 检查脚本是否存在
if not os.path.exists(script_to_run):
    print(f"错误: 找不到脚本 {script_to_run}")
    print(f"当前目录中的Python文件: {[f for f in os.listdir('.') if f.endswith('.py')]}")
    sys.exit(1)

print(f"\n开始运行脚本 {script_to_run}...\n")

# Run the actual training script with the same arguments
try:
    ret = check_call([sys.executable, script_to_run] + args)
    sys.exit(ret)
except Exception as e:
    print(f"运行训练脚本时出错: {e}")
    sys.exit(1) 