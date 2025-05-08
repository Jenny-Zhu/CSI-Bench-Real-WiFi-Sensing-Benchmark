#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速SageMaker环境测试脚本

此脚本仅用于测试SageMaker环境配置和依赖项兼容性，不进行数据下载或训练。
可以通过提交一个使用此脚本的小型作业来快速验证环境。
"""

import os
import sys
import importlib
import json
import time
import argparse
import platform

def print_header(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f" {title} ")
    print("=" * 60)

def check_package(package_name):
    """检查包是否安装成功并打印版本信息"""
    try:
        if package_name == "torch":
            import torch
            version = torch.__version__
            print(f"✓ {package_name}: {version}")
            print(f"  - CUDA可用: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  - CUDA版本: {torch.version.cuda}")
                print(f"  - GPU设备: {torch.cuda.get_device_name(0)}")
                print(f"  - GPU数量: {torch.cuda.device_count()}")
                print(f"  - 当前设备索引: {torch.cuda.current_device()}")
            return True, version
        elif package_name == "torchvision":
            import torchvision
            version = torchvision.__version__
            print(f"✓ {package_name}: {version}")
            return True, version
        else:
            # 通用包检查
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", "未知")
            print(f"✓ {package_name}: {version}")
            return True, version
    except ImportError:
        print(f"✗ {package_name}: 未安装")
        return False, None
    except Exception as e:
        print(f"✗ {package_name}: 错误 - {str(e)}")
        return False, str(e)

def check_horovod_compatibility():
    """检查Horovod与PyTorch的兼容性"""
    print_header("Horovod兼容性检查")

    # 首先尝试导入PyTorch
    try:
        import torch
        torch_version = torch.__version__
        print(f"PyTorch版本: {torch_version}")
    except Exception as e:
        print(f"无法导入PyTorch: {e}")
        return False, str(e)
    
    # 检查sys.modules中是否已有horovod（可能已被我们的入口脚本禁用）
    if 'horovod' in sys.modules and sys.modules['horovod'] is None:
        print("Horovod已被禁用 (通过sys.modules['horovod'] = None)")
        return True, "已禁用"
    
    # 检查环境变量中是否禁用了Horovod
    horovod_env_vars = [
        ('HOROVOD_WITH_PYTORCH', os.environ.get('HOROVOD_WITH_PYTORCH')),
        ('HOROVOD_WITHOUT_PYTORCH', os.environ.get('HOROVOD_WITHOUT_PYTORCH')),
        ('USE_HOROVOD', os.environ.get('USE_HOROVOD'))
    ]
    
    print("Horovod环境变量:")
    for name, value in horovod_env_vars:
        print(f"  - {name}: {value}")
    
    # 尝试导入horovod并检查兼容性
    try:
        # 仅用于测试，正常情况不应该导入
        importlib.util.find_spec("horovod")
        print("Horovod包存在，但尚未导入")
        
        # 尝试导入horovod.torch（可能会失败）
        try:
            importlib.util.find_spec("horovod.torch")
            print("警告: horovod.torch模块存在，可能会与PyTorch版本冲突")
            return False, "可能与PyTorch冲突"
        except ImportError:
            print("horovod.torch模块不存在，不会有冲突")
            return True, "无冲突"
        except Exception as e:
            print(f"检查horovod.torch时出错: {e}")
            return False, str(e)
    except ImportError:
        print("Horovod包不存在，不会有冲突")
        return True, "未安装"
    except Exception as e:
        print(f"检查Horovod时出错: {e}")
        return False, str(e)

def check_smdebug_status():
    """检查SMDebug状态和设置"""
    print_header("SageMaker Debugger (SMDebug) 状态")
    
    # 检查环境变量
    smdebug_env_vars = [
        ('SMDEBUG_DISABLED', os.environ.get('SMDEBUG_DISABLED')),
        ('SM_DISABLE_DEBUGGER', os.environ.get('SM_DISABLE_DEBUGGER')),
    ]
    
    print("SMDebug环境变量:")
    for name, value in smdebug_env_vars:
        print(f"  - {name}: {value}")
    
    # 检查sys.modules中是否已禁用
    if 'smdebug' in sys.modules and sys.modules['smdebug'] is None:
        print("SMDebug已被禁用 (通过sys.modules['smdebug'] = None)")
        return True, "已禁用"
    
    # 检查是否可以导入smdebug（应该被禁用或正常导入）
    try:
        importlib.util.find_spec("smdebug")
        print("警告: SMDebug包存在且可导入，可能会尝试载入Horovod")
        return False, "可能会导致Horovod冲突"
    except ImportError:
        print("SMDebug包不存在或已被屏蔽，不会导入Horovod")
        return True, "无冲突"
    except Exception as e:
        print(f"检查SMDebug时出错: {e}")
        return False, str(e)

def check_data_access(minimal_check=True):
    """检查数据访问，如果minimal_check为True，则不尝试列出大型目录内容"""
    print_header("数据访问检查")
    
    # 检查SageMaker环境变量
    training_dir = os.environ.get('SM_CHANNEL_TRAINING')
    model_dir = os.environ.get('SM_MODEL_DIR')
    
    if not training_dir:
        print("警告: 未找到SM_CHANNEL_TRAINING环境变量")
    else:
        print(f"训练数据目录: {training_dir}")
        if os.path.exists(training_dir):
            print(f"  - 目录存在")
            if not minimal_check:  # 如果需要详细检查
                try:
                    contents = os.listdir(training_dir)
                    print(f"  - 包含 {len(contents)} 个项目")
                    if contents and len(contents) < 10:  # 仅当数量少时列出
                        print(f"  - 内容: {', '.join(contents)}")
                except Exception as e:
                    print(f"  - 无法列出目录内容: {e}")
        else:
            print(f"  - 目录不存在!")
    
    if not model_dir:
        print("警告: 未找到SM_MODEL_DIR环境变量")
    else:
        print(f"模型输出目录: {model_dir}")
        if os.path.exists(model_dir):
            print(f"  - 目录存在")
            try:
                # 尝试写入测试文件
                test_file = os.path.join(model_dir, "env_test_result.txt")
                with open(test_file, "w") as f:
                    f.write(f"环境测试完成于 {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  - 成功写入测试文件: {test_file}")
            except Exception as e:
                print(f"  - 写入测试失败: {e}")
        else:
            print(f"  - 目录不存在!")

def check_arg_format_conversion():
    """测试参数格式转换逻辑"""
    print_header("参数格式转换测试")
    
    # 创建测试参数列表，包括带有破折号和下划线的参数
    test_args = [
        "--learning-rate", "0.001",
        "--batch-size", "32",
        "--win_len", "250",
        "--feature_size", "98"
    ]
    
    print(f"测试参数: {test_args}")
    
    # 模拟参数转换逻辑
    formatted_args = []
    i = 0
    while i < len(test_args):
        arg = test_args[i]
        if arg.startswith('--'):
            fixed_arg = arg.replace('-', '_')
            if fixed_arg != arg:
                print(f"  转换: {arg} -> {fixed_arg}")
                formatted_args.append(fixed_arg)
            else:
                formatted_args.append(arg)
        else:
            formatted_args.append(arg)
        i += 1
    
    print(f"转换后的参数: {formatted_args}")
    
    # 验证结果
    expected_args = [
        "--learning_rate", "0.001",
        "--batch_size", "32",
        "--win_len", "250",
        "--feature_size", "98"
    ]
    
    if formatted_args == expected_args:
        print("✓ 参数转换测试通过!")
        return True
    else:
        print("✗ 参数转换测试失败!")
        print(f"  预期结果: {expected_args}")
        print(f"  实际结果: {formatted_args}")
        return False

def check_environment():
    """执行全面的环境检查"""
    print_header("SageMaker环境快速测试")
    
    # 系统信息
    print(f"Python版本: {platform.python_version()}")
    print(f"Python路径: {sys.executable}")
    print(f"平台信息: {platform.platform()}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查SageMaker环境变量
    sm_vars = [var for var in os.environ if var.startswith("SM_")]
    if sm_vars:
        print(f"检测到 {len(sm_vars)} 个SageMaker环境变量")
        print(f"  - SM_CURRENT_HOST: {os.environ.get('SM_CURRENT_HOST', '未设置')}")
        print(f"  - SM_NUM_GPUS: {os.environ.get('SM_NUM_GPUS', '未设置')}")
        print(f"  - SM_NUM_CPUS: {os.environ.get('SM_NUM_CPUS', '未设置')}")
    else:
        print("未检测到SageMaker环境变量，可能不在SageMaker环境中运行")
    
    # 检查核心依赖包
    print_header("依赖包检查")
    packages = [
        "torch", "torchvision", "numpy", "pandas", 
        "scipy", "matplotlib", "sklearn", "einops", "h5py"
    ]
    optional_packages = ["peft", "transformers"]
    
    package_status = {}
    
    print("核心依赖包:")
    for pkg in packages:
        success, version = check_package(pkg)
        package_status[pkg] = {"installed": success, "version": version}
    
    print("\n可选依赖包:")
    for pkg in optional_packages:
        success, version = check_package(pkg)
        package_status[pkg] = {"installed": success, "version": version}
    
    # 检查Horovod兼容性
    horovod_success, horovod_status = check_horovod_compatibility()
    
    # 检查SMDebug状态
    smdebug_success, smdebug_status = check_smdebug_status()
    
    # 检查数据访问 (快速检查，不列出大型目录)
    check_data_access(minimal_check=True)
    
    # 测试参数格式转换
    arg_format_success = check_arg_format_conversion()
    
    # 检查当前目录下的文件
    print_header("工作目录文件")
    try:
        files = os.listdir('.')
        pyfiles = [f for f in files if f.endswith('.py')]
        print(f"Python文件: {', '.join(pyfiles)}")
        print(f"总文件数: {len(files)}")
    except Exception as e:
        print(f"无法列出文件: {e}")
    
    # 检查requirements.txt
    if os.path.exists('requirements.txt'):
        print("\nrequirements.txt内容:")
        try:
            with open('requirements.txt', 'r') as f:
                content = f.read()
            print(content[:500] + ('...' if len(content) > 500 else ''))
        except Exception as e:
            print(f"无法读取requirements.txt: {e}")
    
    # 总结报告
    print_header("环境检查总结")
    
    torch_status = package_status.get("torch", {})
    if torch_status.get("installed"):
        print(f"✓ PyTorch已正确安装: {torch_status.get('version')}")
    else:
        print(f"✗ PyTorch安装出现问题")
    
    if horovod_success:
        print(f"✓ Horovod状态正常: {horovod_status}")
    else:
        print(f"✗ Horovod可能会导致问题: {horovod_status}")
    
    if smdebug_success:
        print(f"✓ SMDebug状态正常: {smdebug_status}")
    else:
        print(f"✗ SMDebug可能会导致问题: {smdebug_status}")
    
    if arg_format_success:
        print(f"✓ 参数格式转换正常")
    else:
        print(f"✗ 参数格式转换存在问题")
        
    # 保存报告
    try:
        model_dir = os.environ.get('SM_MODEL_DIR', '.')
        report_file = os.path.join(model_dir, 'environment_report.json')
        report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_status,
            "horovod_status": {
                "success": horovod_success,
                "status": horovod_status
            },
            "smdebug_status": {
                "success": smdebug_success,
                "status": smdebug_status
            },
            "arg_format_conversion": {
                "success": arg_format_success
            },
            "in_sagemaker": len(sm_vars) > 0
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ 环境报告已保存到: {report_file}")
    except Exception as e:
        print(f"✗ 保存报告失败: {e}")
    
    # 对所有测试结果进行汇总评估
    all_tests_passed = (
        torch_status.get("installed") and 
        horovod_success and 
        smdebug_success and
        arg_format_success
    )
    
    print("\n总结: " + ("✓ 环境配置正常，可以运行训练作业" if all_tests_passed else "✗ 环境配置存在问题，需要修复"))

def main():
    parser = argparse.ArgumentParser(description='SageMaker环境快速测试工具')
    parser.add_argument('--check-data', action='store_true',
                       help='详细检查数据目录（可能很慢）')
    args, unknown = parser.parse_known_args()
    
    # 执行环境检查
    start_time = time.time()
    check_environment()
    duration = time.time() - start_time
    
    print(f"\n\n环境检查完成，耗时: {duration:.2f}秒")
    
    # 可选: 详细检查数据目录（如果需要）
    if args.check_data:
        print("\n开始详细检查数据目录...")
        check_data_access(minimal_check=False)

if __name__ == "__main__":
    main() 