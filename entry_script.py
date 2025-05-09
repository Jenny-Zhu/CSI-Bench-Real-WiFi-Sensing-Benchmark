#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SageMaker Entry Script - Disable Horovod and SMDebug

这个脚本在导入PyTorch前禁用Horovod和SMDebug集成，以避免版本冲突，然后运行实际的训练脚本。
主要功能：
1. 禁用Horovod和SMDebug
2. 安装peft及其依赖项
3. 直接传递所有参数给训练脚本
"""

import os
import sys
import subprocess
import gc
import logging

print("\n==========================================")
print("启动自定义入口脚本entry_script.py")
print("==========================================\n")

# 设置内存优化选项
print("配置内存优化设置...")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 减少TensorFlow日志（如果使用）

# 启用垃圾回收
gc.enable()
print("已启用主动垃圾回收")

# 立即禁用horovod和smdebug以防止冲突
print("禁用Horovod集成...")
sys.modules['horovod'] = None
sys.modules['horovod.torch'] = None
sys.modules['horovod.tensorflow'] = None
sys.modules['horovod.common'] = None
sys.modules['horovod.torch.elastic'] = None

print("禁用SMDebug...")
sys.modules['smdebug'] = None
os.environ['SMDEBUG_DISABLED'] = 'true'
os.environ['SM_DISABLE_DEBUGGER'] = 'true'

# 设置typing_extensions支持以解决typing.Literal导入问题
print("设置typing_extensions支持...")
try:
    import typing_extensions
    from typing_extensions import Literal
    # 将Literal注入到sys.modules中的typing模块
    if 'typing' in sys.modules:
        sys.modules['typing'].Literal = Literal
    print("成功导入typing_extensions并配置Literal支持")
except ImportError:
    print("警告：找不到typing_extensions，某些功能可能不可用")

# 手动安装peft库及其依赖项
print("安装peft库及其依赖项...")
try:
    # 先安装transformers
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.30.0", "--no-dependencies"])
    print("transformers库安装成功")
    
    # 安装accelerate
    subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate", "--no-dependencies"])
    print("accelerate库安装成功")
    
    # 然后使用--no-dependencies参数安装特定版本的peft
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft==0.3.0", "--no-dependencies"])
    print("peft库安装成功")
except Exception as e:
    print(f"安装peft库时出错：{e}")
    # 这不是致命错误，尝试继续执行

# 释放一些内存
gc.collect()

# 设置路径
print("设置路径...")
sys.path.insert(0, os.getcwd())

# 显示可用内存信息
try:
    import psutil
    process = psutil.Process(os.getpid())
    print(f"当前进程内存使用量：{process.memory_info().rss / (1024 * 1024):.2f} MB")
    
    virtual_memory = psutil.virtual_memory()
    print(f"系统内存状态：")
    print(f"  总内存：{virtual_memory.total / (1024**3):.2f} GB")
    print(f"  可用内存：{virtual_memory.available / (1024**3):.2f} GB")
    print(f"  内存使用率：{virtual_memory.percent}%")
except ImportError:
    print("无法导入psutil，跳过内存信息显示")
except Exception as e:
    print(f"获取内存信息时出错：{e}")

# 现在运行实际的训练脚本
print("准备运行训练脚本...")

# 检查要执行的脚本
# 检查SAGEMAKER_PROGRAM环境变量
script_to_run = os.environ.get('SAGEMAKER_PROGRAM', 'train_multi_model.py')
print(f"要执行的脚本：{script_to_run}")

# 检查脚本是否存在
if not os.path.exists(script_to_run):
    print(f"错误：找不到脚本 {script_to_run}")
    print(f"当前目录中的Python文件：{[f for f in os.listdir('.') if f.endswith('.py')]}")
    sys.exit(1)

# 打印环境信息
print("\n===== 环境信息 =====")
import platform
print(f"Python版本：{platform.python_version()}")
print(f"当前目录：{os.getcwd()}")
print(f"目录中的文件：{', '.join(os.listdir('.'))[:200]}...")

# 打印环境变量
print("\n===== 环境变量 =====")
sm_vars = [k for k in os.environ.keys() if k.startswith('SM_') or k.startswith('SAGEMAKER_')]
for var in sm_vars:
    print(f"{var}: {os.environ.get(var)}")
print("==================================\n")

# 尝试导入torch以验证其是否在没有Horovod冲突的情况下正确加载
try:
    print("尝试导入PyTorch（设置内存限制）...")
    import torch
    # 配置PyTorch内存分配器优化
    if torch.cuda.is_available():
        # 限制GPU内存增长
        torch.cuda.set_per_process_memory_fraction(0.7)  # 使用70%的可用GPU内存
        # 主动清理CUDA缓存
        torch.cuda.empty_cache()
        
    print(f"PyTorch版本：{torch.__version__}")
    print(f"CUDA可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本：{torch.version.cuda}")
        print(f"GPU设备：{torch.cuda.get_device_name(0)}")
        print(f"GPU总内存：{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"当前分配的GPU内存：{torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"当前GPU内存缓存：{torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    print("成功导入PyTorch，没有Horovod冲突")
    
    # 检查是否为环境测试模式
    test_env = os.environ.get('SM_HP_TEST_ENV') == 'True'
    if test_env:
        print("\n==========================================")
        print("运行环境测试模式，跳过数据下载和完整训练...")
        print("==========================================\n")
        
        # 创建简单的仿真测试
        import time
        import importlib
        
        print("验证PyTorch GPU可用性...")
        if torch.cuda.is_available():
            print(f"GPU可用：{torch.cuda.get_device_name(0)}")
            # 执行简单的GPU计算以验证功能
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            start = time.time()
            for _ in range(10):
                z = x @ y
            torch.cuda.synchronize()
            end = time.time()
            print(f"GPU矩阵乘法测试时间：{end-start:.4f}秒")
        else:
            print("警告：GPU不可用")
        
        print("\n验证常见库导入...\n")
        import_tests = [
            "numpy", "pandas", "matplotlib", "scipy", "sklearn", 
            "torch", "einops", "h5py", "torchvision", "typing_extensions"
        ]
        
        success = 0
        failed = 0
        
        for module in import_tests:
            try:
                importlib.import_module(module)
                version = "unknown"
                try:
                    mod = importlib.import_module(module)
                    if hasattr(mod, "__version__"):
                        version = mod.__version__
                    elif hasattr(mod, "VERSION"):
                        version = mod.VERSION
                    elif hasattr(mod, "version"):
                        version = mod.version
                except:
                    pass
                
                print(f"✓ 成功导入 {module} (版本: {version})")
                success += 1
            except ImportError as e:
                print(f"✗ 无法导入 {module}: {e}")
                failed += 1
        
        print(f"\n导入测试结果：{success}个成功，{failed}个失败")
        
        # 检查CUDA版本和PyTorch兼容性
        print("\n环境兼容性检查：")
        if torch.cuda.is_available():
            print(f"✓ CUDA可用：{torch.version.cuda}")
            print(f"✓ PyTorch使用CUDA：{torch.version.cuda}")
            print(f"✓ GPU设备：{torch.cuda.get_device_name(0)}")
            print(f"✓ GPU内存：{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("✗ CUDA不可用，将使用CPU")
        
        # 检查依赖项
        try:
            import peft
            print(f"✓ PEFT库可用，版本：{peft.__version__}")
        except ImportError:
            print("✗ PEFT库不可用")
        
        # 检查数据目录
        data_dir = os.environ.get('SM_CHANNEL_TRAINING', None)
        if data_dir and os.path.exists(data_dir):
            print(f"✓ 数据目录存在：{data_dir}")
            print(f"  文件数量：{len(os.listdir(data_dir))}")
        else:
            print("✗ 数据目录不存在或为空")
        
        print("\n环境测试完成，所有依赖项验证已完成")
        print("==========================================\n")
        sys.exit(0)  # 成功退出
        
except Exception as e:
    print(f"导入PyTorch时出错：{e}")
    sys.exit(1)  # 如果无法导入PyTorch，则退出

# 释放一些内存
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 尝试导入peft库以确认它能够正确加载
try:
    import peft
    print(f"PEFT库版本：{peft.__version__}")
    print("PEFT库成功导入")
except Exception as e:
    print(f"导入PEFT库时出错：{e}")
    print("这可能会影响某些功能，但我们将继续执行")

print(f"\n直接执行脚本: {script_to_run}...")
print(f"参数: {sys.argv[1:]}")

# 直接执行训练脚本，保留原始参数
from subprocess import check_call
try:
    # 直接运行脚本并传递参数
    ret = check_call([sys.executable, script_to_run] + sys.argv[1:])
    sys.exit(ret)
except Exception as e:
    print(f"运行脚本出错：{e}")
    sys.exit(1) 