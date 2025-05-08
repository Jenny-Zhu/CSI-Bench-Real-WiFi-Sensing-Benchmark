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
import gc

print("\n==========================================")
print("启动自定义入口脚本entry_script.py")
print("==========================================\n")

# 设置内存优化选项
print("配置内存优化设置...")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 减少TensorFlow日志，如果使用的话

# 启用垃圾回收
gc.enable()
print("已启用主动垃圾回收")

# 立即禁用horovod和smdebug以防止冲突
print("立即禁用Horovod集成...")
sys.modules['horovod'] = None
sys.modules['horovod.torch'] = None
sys.modules['horovod.tensorflow'] = None
sys.modules['horovod.common'] = None
sys.modules['horovod.torch.elastic'] = None

print("立即禁用SMDebug...")
sys.modules['smdebug'] = None
os.environ['SMDEBUG_DISABLED'] = 'true'
os.environ['SM_DISABLE_DEBUGGER'] = 'true'

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

# 释放一些内存
gc.collect()

# Set paths
print("设置路径...")
sys.path.insert(0, os.getcwd())

# 显示可用内存信息
try:
    import psutil
    process = psutil.Process(os.getpid())
    print(f"当前进程内存使用: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    
    virtual_memory = psutil.virtual_memory()
    print(f"系统内存情况:")
    print(f"  总内存: {virtual_memory.total / (1024**3):.2f} GB")
    print(f"  可用内存: {virtual_memory.available / (1024**3):.2f} GB")
    print(f"  内存使用率: {virtual_memory.percent}%")
except ImportError:
    print("无法导入psutil，跳过内存信息显示")
except Exception as e:
    print(f"获取内存信息时出错: {e}")

# Now run the actual training script
print("准备运行训练脚本...")
from subprocess import check_call

# 识别要执行的脚本
# 检查SAGEMAKER_PROGRAM环境变量
script_to_run = os.environ.get('SAGEMAKER_PROGRAM', 'train_multi_model.py')
print(f"将要执行的脚本: {script_to_run}")

# 获取并优化命令行参数
args = sys.argv[1:]
print(f"原始命令行参数: {args}")

# 修复参数名称格式 - 将带破折号的参数转换为带下划线的格式
formatted_args = []
i = 0
while i < len(args):
    arg = args[i]
    # 修复参数名称格式 (例如 --learning-rate 变成 --learning_rate)
    if arg.startswith('--'):
        fixed_arg = arg.replace('-', '_')
        if fixed_arg != arg:
            print(f"修复参数格式: {arg} -> {fixed_arg}")
            formatted_args.append(fixed_arg)
        else:
            formatted_args.append(arg)
    else:
        formatted_args.append(arg)
    i += 1

args = formatted_args
print(f"格式修复后的参数: {args}")

# 继续进行其他参数优化 - 减小batch_size以降低内存使用
modified_args = []
i = 0
while i < len(args):
    if args[i] == '--batch_size' and i+1 < len(args):
        try:
            batch_size = int(args[i+1])
            if batch_size > 4:
                print(f"警告: 检测到较大的batch_size ({batch_size})，已自动减小至4以降低内存使用")
                modified_args.extend(['--batch_size', '4'])
            else:
                modified_args.extend([args[i], args[i+1]])
        except ValueError:
            modified_args.extend([args[i], args[i+1]])
        i += 2
    else:
        modified_args.append(args[i])
        i += 1

if args != modified_args:
    print("注意: 已调整命令行参数以优化内存使用")
    args = modified_args

# Print environment information
print("\n===== 环境信息 =====")
import platform
print(f"Python版本: {platform.python_version()}")
print(f"当前目录: {os.getcwd()}")
print(f"目录中的文件: {', '.join(os.listdir('.'))[:200]}...")
print(f"命令: python3 {script_to_run} {' '.join(args)}")

# 打印环境变量信息
print("\n===== 环境变量 =====")
sm_vars = [k for k in os.environ.keys() if k.startswith('SM_') or k.startswith('SAGEMAKER_')]
for var in sm_vars:
    print(f"{var}: {os.environ.get(var)}")
print("==================================\n")

# Try importing torch to verify it loads correctly without Horovod conflicts
try:
    print("尝试导入PyTorch (设置内存限制)...")
    import torch
    # 设置PyTorch内存分配器的优化
    if torch.cuda.is_available():
        # 限制GPU内存增长
        torch.cuda.set_per_process_memory_fraction(0.7)  # 使用70%的可用GPU内存
        # 主动清理CUDA缓存
        torch.cuda.empty_cache()
        
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存总量: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"当前分配的GPU内存: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"当前GPU内存缓存: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    print("PyTorch成功导入，没有Horovod冲突")
except Exception as e:
    print(f"导入PyTorch时出错: {e}")
    sys.exit(1)  # Exit if we can't import PyTorch

# 释放一些内存
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

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

# 创建优化版本的包装脚本
print(f"\n创建优化版本包装脚本...")

# 创建一个简单的包装脚本，它将先导入我们的修改后再执行实际代码
wrapper_content = f"""#!/usr/bin/env python3
# 自动生成的包装脚本，用于避免Horovod依赖冲突并优化内存使用
import sys
import os
import gc

# 开启垃圾回收
gc.enable()

# 设置内存优化环境变量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 禁用Horovod相关模块
sys.modules['horovod'] = None
sys.modules['horovod.torch'] = None 
sys.modules['horovod.tensorflow'] = None
sys.modules['horovod.common'] = None
sys.modules['horovod.torch.elastic'] = None

# 禁用SMDebug
sys.modules['smdebug'] = None
os.environ['SMDEBUG_DISABLED'] = 'true'
os.environ['SM_DISABLE_DEBUGGER'] = 'true'

# 预处理命令行参数 - 修复参数格式（破折号改为下划线）
args = sys.argv[1:]
print(f"原始命令行参数: {args}")

formatted_args = []
i = 0
while i < len(args):
    arg = args[i]
    # 修复参数名称格式 (例如 --learning-rate 变成 --learning_rate)
    if arg.startswith('--'):
        fixed_arg = arg.replace('-', '_')
        if fixed_arg != arg:
            print(f"修复参数格式: {arg} -> {fixed_arg}")
            formatted_args.append(fixed_arg)
        else:
            formatted_args.append(arg)
    else:
        formatted_args.append(arg)
    i += 1

sys.argv[1:] = formatted_args
print(f"格式修复后的参数: {formatted_args}")

# 导入torch并配置内存优化
try:
    import torch
    if torch.cuda.is_available():
        # 限制内存使用
        torch.cuda.set_per_process_memory_fraction(0.7)
        torch.cuda.empty_cache()
        
        # 使用确定性算法，可能会降低性能但提高稳定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except Exception as e:
    print(f"配置PyTorch时出错: {e}")

# 主动执行垃圾回收
gc.collect()

# 然后导入并执行原始脚本
import {script_to_run.replace('.py', '')}

# 调用原始脚本的main函数（如果存在）
if hasattr({script_to_run.replace('.py', '')}, 'main'):
    {script_to_run.replace('.py', '')}.main()
"""

# 写入临时包装脚本
wrapper_script = "wrapper_script.py"
with open(wrapper_script, "w") as f:
    f.write(wrapper_content)
print(f"创建了优化版本包装脚本: {wrapper_script}")

print(f"\n开始运行优化后的脚本 {script_to_run}...\n")

# Run the wrapper script with the same arguments
try:
    print("使用内存优化包装脚本...")
    ret = check_call([sys.executable, wrapper_script] + args)
    sys.exit(ret)
except Exception as e:
    print(f"运行脚本时出错: {e}")
    
    # 作为备选方案，尝试直接运行原始脚本
    print("\n尝试直接运行原始脚本作为备选方案...")
    try:
        ret = check_call([sys.executable, script_to_run] + args)
        sys.exit(ret)
    except Exception as e2:
        print(f"运行原始脚本也失败: {e2}")
        sys.exit(1) 