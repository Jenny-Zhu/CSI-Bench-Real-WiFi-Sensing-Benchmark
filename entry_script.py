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
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import argparse
from tqdm import tqdm

print("\n==========================================")
print("Starting custom entry script entry_script.py")
print("==========================================\n")

# Set memory optimization options
print("Configuring memory optimization settings...")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logs if used

# Enable garbage collection
gc.enable()
print("Active garbage collection enabled")

# Immediately disable horovod and smdebug to prevent conflicts
print("Disabling Horovod integration...")
sys.modules['horovod'] = None
sys.modules['horovod.torch'] = None
sys.modules['horovod.tensorflow'] = None
sys.modules['horovod.common'] = None
sys.modules['horovod.torch.elastic'] = None

print("Disabling SMDebug...")
sys.modules['smdebug'] = None
os.environ['SMDEBUG_DISABLED'] = 'true'
os.environ['SM_DISABLE_DEBUGGER'] = 'true'

# Set up typing_extensions support to resolve typing.Literal import issues
print("Setting up typing_extensions support...")
# Ensure typing_extensions is available and pre-import
try:
    import typing_extensions
    from typing_extensions import Literal
    # Inject Literal into the typing module in sys.modules
    if 'typing' in sys.modules:
        sys.modules['typing'].Literal = Literal
    print("Successfully imported typing_extensions and configured Literal support")
except ImportError:
    print("Warning: typing_extensions not found, some features may not be available")

# Manually install peft library and its dependencies
print("Installing peft library and its dependencies...")
try:
    # Install transformers first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.30.0", "--no-dependencies"])
    print("transformers library installation successful")
    
    # Install accelerate
    subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate", "--no-dependencies"])
    print("accelerate library installation successful")
    
    # Then install specific version of peft with --no-dependencies parameter
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft==0.3.0", "--no-dependencies"])
    print("peft library installation successful")
except Exception as e:
    print(f"Error installing peft library: {e}")
    # This is not a fatal error, trying to continue execution

# Release some memory
gc.collect()

# Set paths
print("Setting paths...")
sys.path.insert(0, os.getcwd())

# Display available memory information
try:
    import psutil
    process = psutil.Process(os.getpid())
    print(f"Current process memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    
    virtual_memory = psutil.virtual_memory()
    print(f"System memory status:")
    print(f"  Total memory: {virtual_memory.total / (1024**3):.2f} GB")
    print(f"  Available memory: {virtual_memory.available / (1024**3):.2f} GB")
    print(f"  Memory usage: {virtual_memory.percent}%")
except ImportError:
    print("Unable to import psutil, skipping memory information display")
except Exception as e:
    print(f"Error getting memory information: {e}")

# Now run the actual training script
print("Preparing to run training script...")
from subprocess import check_call

# Identify script to execute
# Check SAGEMAKER_PROGRAM environment variable
script_to_run = os.environ.get('SAGEMAKER_PROGRAM', 'train_multi_model.py')
print(f"Script to execute: {script_to_run}")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_arg(arg):
    """Format command line argument to standard format.
    Converts __param to --param and --param-name to --param_name
    """
    if arg.startswith('__'):
        return '--' + arg[2:]
    elif arg.startswith('--'):
        return arg.replace('-', '_')
    return arg

def validate_args(args):
    """Validate required arguments are present and check parameter compatibilities"""
    # 核心必需参数
    required_params = ['task_name', 'models', 'win_len', 'feature_size']
    missing_params = []
    
    # 检查必需参数
    for param in required_params:
        param_found = False
        for i, arg in enumerate(args):
            if arg.startswith('--'):
                # 去除前缀并比较
                arg_name = arg[2:].replace('-', '_')
                if arg_name == param or arg_name == param.replace('_', '-'):
                    param_found = True
                    break
        if not param_found:
            missing_params.append(param)
    
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
    
    # 检查参数类型兼容性
    # 在这里可以添加更多验证，比如检查批处理大小是否合理等
    for i, arg in enumerate(args):
        if arg == '--batch_size' and i+1 < len(args):
            try:
                batch_size = int(args[i+1])
                if batch_size > 32:
                    logging.warning(f"Large batch size detected: {batch_size}. Consider reducing to avoid memory issues.")
            except ValueError:
                pass
            
    # 确保参数名格式统一，替换task为task_name
    for i, arg in enumerate(args):
        if arg == '--task' and i+1 < len(args):
            logging.warning("Using deprecated parameter name 'task'. Please use 'task_name' instead.")
            args[i] = '--task_name'

# Process command line arguments
original_args = sys.argv[1:]
logging.info(f"Original command line arguments: {original_args}")

# Format arguments - handle both single and paired arguments
formatted_args = []
i = 0
while i < len(original_args):
    arg = original_args[i]
    if arg.startswith('__') or arg.startswith('--'):
        # Format the argument name
        formatted_arg = format_arg(arg)
        formatted_args.append(formatted_arg)
        # If next argument exists and doesn't start with __ or --, it's a value
        if i + 1 < len(original_args) and not (original_args[i + 1].startswith('__') or original_args[i + 1].startswith('--')):
            formatted_args.append(original_args[i + 1])
            i += 2
        else:
            i += 1
    else:
        formatted_args.append(arg)
        i += 1

logging.info(f"Parameters after format fixing: {formatted_args}")

# Validate arguments
try:
    validate_args(formatted_args)
except ValueError as e:
    logging.error(f"Argument validation failed: {e}")
    sys.exit(1)

# Update sys.argv with formatted arguments
sys.argv[1:] = formatted_args

# Print environment information
print("\n===== Environment Information =====")
import platform
print(f"Python version: {platform.python_version()}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {', '.join(os.listdir('.'))[:200]}...")
print(f"Command: python3 {script_to_run} {' '.join(args)}")

# Print environment variables
print("\n===== Environment Variables =====")
sm_vars = [k for k in os.environ.keys() if k.startswith('SM_') or k.startswith('SAGEMAKER_')]
for var in sm_vars:
    print(f"{var}: {os.environ.get(var)}")
print("==================================\n")

# Try importing torch to verify it loads correctly without Horovod conflicts
try:
    print("Attempting to import PyTorch (setting memory limits)...")
    import torch
    # Configure PyTorch memory allocator optimizations
    if torch.cuda.is_available():
        # Limit GPU memory growth
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of available GPU memory
        # Actively clean CUDA cache
        torch.cuda.empty_cache()
        
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"Current GPU memory cache: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    print("PyTorch successfully imported without Horovod conflicts")

    # Add environment test functionality
    # Check if in environment test mode
    test_env = os.environ.get('SM_HP_TEST_ENV') == 'True'
    if test_env:
        print("\n==========================================")
        print("Running environment test mode, skipping data download and full training...")
        print("==========================================\n")
        
        # Create a simple simulation test
        import time
        import importlib
        
        print("Verifying PyTorch GPU availability...")
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            # Perform simple GPU calculations to verify functionality
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            start = time.time()
            for _ in range(10):
                z = x @ y
            torch.cuda.synchronize()
            end = time.time()
            print(f"GPU matrix multiplication test time: {end-start:.4f} seconds")
        else:
            print("Warning: GPU not available")
        
        print("\nVerifying common library imports...\n")
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
                
                print(f"✓ Successfully imported {module} (version: {version})")
                success += 1
            except ImportError as e:
                print(f"✗ Could not import {module}: {e}")
                failed += 1
        
        print(f"\nImport test results: {success} successful, {failed} failed")
        
        # Check CUDA version and PyTorch compatibility
        print("\nEnvironment compatibility check:")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ PyTorch using CUDA: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("✗ CUDA not available, will use CPU")
        
        # Check dependencies
        try:
            import peft
            print(f"✓ PEFT library available, version: {peft.__version__}")
        except ImportError:
            print("✗ PEFT library not available")
        
        # Check data directory
        data_dir = os.environ.get('SM_CHANNEL_TRAINING', None)
        if data_dir and os.path.exists(data_dir):
            print(f"✓ Data directory exists: {data_dir}")
            print(f"  Number of files: {len(os.listdir(data_dir))}")
        else:
            print("✗ Data directory does not exist or is empty")
        
        print("\nEnvironment test complete, all dependency verifications completed")
        print("==========================================\n")
        sys.exit(0)  # Successful exit
except Exception as e:
    print(f"Error importing PyTorch: {e}")
    sys.exit(1)  # Exit if we can't import PyTorch

# Release some memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Try importing peft library to confirm it can be loaded correctly
try:
    import peft
    print(f"PEFT library version: {peft.__version__}")
    print("PEFT library successfully imported")
except Exception as e:
    print(f"Error importing PEFT library: {e}")
    print("This may affect some functionality, but we will continue execution")

# Check if script exists
if not os.path.exists(script_to_run):
    print(f"Error: Script {script_to_run} not found")
    print(f"Python files in current directory: {[f for f in os.listdir('.') if f.endswith('.py')]}")
    sys.exit(1)

# Create optimized version of wrapper script
print(f"\nCreating optimized wrapper script...")

# Modify wrapper script template using triple quotes for better readability
wrapper_content = """#!/usr/bin/env python3
# Automatically generated wrapper script to avoid Horovod dependency conflicts and optimize memory usage
import sys
import os
import gc
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable garbage collection
gc.enable()

# Set memory optimization environment variables
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Disable Horovod related modules
sys.modules['horovod'] = None
sys.modules['horovod.torch'] = None 
sys.modules['horovod.tensorflow'] = None
sys.modules['horovod.common'] = None
sys.modules['horovod.torch.elastic'] = None

# Disable SMDebug
sys.modules['smdebug'] = None
os.environ['SMDEBUG_DISABLED'] = 'true'
os.environ['SM_DISABLE_DEBUGGER'] = 'true'

def format_arg(arg):
    \"\"\"Format command line argument to standard format.
    Converts __param to --param and --param-name to --param_name
    \"\"\"
    if arg.startswith('__'):
        return '--' + arg[2:]
    elif arg.startswith('--'):
        return arg.replace('-', '_')
    return arg

def validate_args(args):
    \"\"\"Validate required arguments are present and check parameter compatibilities\"\"\"
    # 核心必需参数
    required_params = ['task_name', 'models', 'win_len', 'feature_size']
    missing_params = []
    
    # 检查必需参数
    for param in required_params:
        param_found = False
        for i, arg in enumerate(args):
            if arg.startswith('--'):
                # 去除前缀并比较
                arg_name = arg[2:].replace('-', '_')
                if arg_name == param or arg_name == param.replace('_', '-'):
                    param_found = True
                    break
        if not param_found:
            missing_params.append(param)
    
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
    
    # 检查参数类型兼容性
    # 在这里可以添加更多验证，比如检查批处理大小是否合理等
    for i, arg in enumerate(args):
        if arg == '--batch_size' and i+1 < len(args):
            try:
                batch_size = int(args[i+1])
                if batch_size > 32:
                    logger.warning(f"Large batch size detected: {batch_size}. Consider reducing to avoid memory issues.")
            except ValueError:
                pass
            
    # 确保参数名格式统一，替换task为task_name
    for i, arg in enumerate(args):
        if arg == '--task' and i+1 < len(args):
            logger.warning("Using deprecated parameter name 'task'. Please use 'task_name' instead.")
            args[i] = '--task_name'

# Check for SageMaker environment variables and add them to args
logger.info("Checking for SageMaker environment variables...")
sagemaker_params = {}

# Map SageMaker hyperparameters to CLI arguments
for key, value in os.environ.items():
    if key.startswith('SM_HP_'):
        # Convert key: SM_HP_WIN_LEN -> win_len
        param_name = key[6:].lower()  # Remove SM_HP_ prefix
        sagemaker_params[param_name] = value
        logger.info(f"Found SageMaker hyperparameter: {param_name} = {value}")

# If we're in SageMaker environment, add necessary paths automatically
if 'SM_MODEL_DIR' in os.environ:
    logger.info("Running in SageMaker environment")
    sagemaker_params['save_dir'] = os.environ['SM_MODEL_DIR']
    sagemaker_params['output_dir'] = os.environ['SM_OUTPUT_DATA_DIR']
    
    # Get training data directory from SM_CHANNEL_TRAINING
    if 'SM_CHANNEL_TRAINING' in os.environ:
        sagemaker_params['data_dir'] = os.environ['SM_CHANNEL_TRAINING']
        sagemaker_params['dataset_root'] = os.environ['SM_CHANNEL_TRAINING']
        logger.info(f"Setting data directory to: {sagemaker_params['data_dir']}")
    
    # Get number of GPUs
    if 'SM_NUM_GPUS' in os.environ:
        num_gpus = int(os.environ['SM_NUM_GPUS'])
        logger.info(f"Available GPUs: {num_gpus}")

# Special handling for models - convert comma-separated string to list if needed
if 'models' in sagemaker_params and ',' in sagemaker_params['models']:
    logger.info(f"Converted comma-separated models to list: {sagemaker_params['models']}")
    # Note: we don't actually convert to list here, as argparse will handle that
    
# Special handling for task_name vs task (handle backward compatibility)
if 'task' in sagemaker_params and 'task_name' not in sagemaker_params:
    logger.warning("Converting 'task' parameter to 'task_name' for consistency")
    sagemaker_params['task_name'] = sagemaker_params['task']

# Print information about directories and data paths
logger.info("===== Data Path Information =====")
# Print the current directory
logger.info(f"Current directory: {os.getcwd()}")
# Check for training data directory
data_dir = os.environ.get('SM_CHANNEL_TRAINING', None)
if data_dir:
    logger.info(f"Training data directory: {data_dir}")
    if os.path.exists(data_dir):
        logger.info(f"Training data directory exists")
        # List files in the training data directory
        try:
            files = os.listdir(data_dir)
            if files:
                logger.info(f"Files in training data directory: {files[:10]}")
                if len(files) > 10:
                    logger.info(f"... and {len(files) - 10} more files")
            else:
                logger.info("Training data directory is empty")
        except Exception as e:
            logger.error(f"Error listing training data directory: {e}")
    else:
        logger.warning(f"Training data directory does not exist")
        
# Check for model directory
model_dir = os.environ.get('SM_MODEL_DIR', None)
if model_dir:
    logger.info(f"Model directory: {model_dir}")
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created model directory")
        except Exception as e:
            logger.error(f"Error creating model directory: {e}")
logger.info("==================================")

# Preprocess command line arguments
args = sys.argv[1:]
logger.info(f"Original command line arguments: {args}")

# Format arguments - handle both single and paired arguments
formatted_args = []
i = 0
while i < len(args):
    arg = args[i]
    if arg.startswith('__') or arg.startswith('--'):
        # Format the argument name
        formatted_arg = format_arg(arg)
        formatted_args.append(formatted_arg)
        # If next argument exists and doesn't start with __ or --, it's a value
        if i + 1 < len(args) and not (args[i + 1].startswith('__') or args[i + 1].startswith('--')):
            formatted_args.append(args[i + 1])
            i += 2
        else:
            i += 1
    else:
        formatted_args.append(arg)
        i += 1

logger.info(f"Parameters after format fixing: {formatted_args}")

# Validate arguments
try:
    validate_args(formatted_args)
except ValueError as e:
    logger.error(f"Argument validation failed: {e}")
    sys.exit(1)

# Update sys.argv with formatted arguments
sys.argv[1:] = formatted_args

# Environment test mode check
if os.environ.get('SM_HP_TEST_ENV') == 'True':
    logger.info("Wrapper script detected environment test mode...")

# Import torch and configure memory optimizations
try:
    import torch
    if torch.cuda.is_available():
        # Limit memory usage
        torch.cuda.set_per_process_memory_fraction(0.7)
        torch.cuda.empty_cache()
        
        # Use deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except Exception as e:
    logger.error(f"Error configuring PyTorch: {e}")

# Actively perform garbage collection
gc.collect()

# Import necessary modules
import os
import sys
import json
import time
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Then import and execute the original script
import {script_name}

# Call the main function of the original script (if it exists)
if hasattr({script_name}, 'main'):
    {script_name}.main()
"""

# Replace {script_name} with the actual script name
script_name = script_to_run.replace('.py', '')
wrapper_content = wrapper_content.replace('{script_name}', script_name)

# Write temporary wrapper script
wrapper_script = "wrapper_script.py"
with open(wrapper_script, "w") as f:
    f.write(wrapper_content)
print(f"Created optimized wrapper script: {wrapper_script}")

print(f"\nStarting execution of optimized script {script_to_run}...\n")

# Run the wrapper script with the same arguments
try:
    print("Using memory optimized wrapper script...")
    ret = check_call([sys.executable, wrapper_script] + formatted_args)
    sys.exit(ret)
except Exception as e:
    print(f"Error running script: {e}")
    
    # As a fallback, try running the original script directly
    print("\nAttempting to run the original script as a fallback...")
    try:
        print(f"Original command line arguments: {sys.argv}")
        ret = check_call([sys.executable, script_to_run] + formatted_args)
        sys.exit(ret)
    except Exception as e2:
        print(f"Running original script also failed: {e2}")
        sys.exit(1) 

# Check for SageMaker environment variables and add them to args
print("Checking for SageMaker environment variables...")
sagemaker_params = {}

# Map SageMaker hyperparameters to CLI arguments
for key, value in os.environ.items():
    if key.startswith('SM_HP_'):
        # Convert key: SM_HP_WIN_LEN -> win_len
        param_name = key[6:].lower()  # Remove SM_HP_ prefix
        sagemaker_params[param_name] = value
        print(f"Found SageMaker hyperparameter: {param_name} = {value}")

# If we're in SageMaker environment, add necessary paths automatically
if 'SM_MODEL_DIR' in os.environ:
    print("Running in SageMaker environment")
    sagemaker_params['save_dir'] = os.environ['SM_MODEL_DIR']
    sagemaker_params['output_dir'] = os.environ['SM_OUTPUT_DATA_DIR']
    
    # Get training data directory from SM_CHANNEL_TRAINING
    if 'SM_CHANNEL_TRAINING' in os.environ:
        sagemaker_params['data_dir'] = os.environ['SM_CHANNEL_TRAINING']
        sagemaker_params['dataset_root'] = os.environ['SM_CHANNEL_TRAINING']
        print(f"Setting data directory to: {sagemaker_params['data_dir']}")
    
    # Get number of GPUs
    if 'SM_NUM_GPUS' in os.environ:
        num_gpus = int(os.environ['SM_NUM_GPUS'])
        print(f"Available GPUs: {num_gpus}")

# Special handling for models - convert comma-separated string to list if needed
if 'models' in sagemaker_params and ',' in sagemaker_params['models']:
    print(f"Converted comma-separated models to list: {sagemaker_params['models']}")
    # Note: we don't actually convert to list here, as argparse will handle that
    
# Special handling for task_name vs task (handle backward compatibility)
if 'task' in sagemaker_params and 'task_name' not in sagemaker_params:
    logging.warning("Converting 'task' parameter to 'task_name' for consistency")
    sagemaker_params['task_name'] = sagemaker_params['task']

# Print information about directories and data paths
print(f"\n===== Data Path Information =====")
# Print the current directory
print(f"Current directory: {os.getcwd()}")
# Check for training data directory
data_dir = os.environ.get('SM_CHANNEL_TRAINING', None)
if data_dir:
    print(f"Training data directory: {data_dir}")
    if os.path.exists(data_dir):
        print(f"Training data directory exists")
        # List files in the training data directory
        try:
            files = os.listdir(data_dir)
            if files:
                print(f"Files in training data directory: {files[:10]}")
                if len(files) > 10:
                    print(f"... and {len(files) - 10} more files")
            else:
                print("Training data directory is empty")
        except Exception as e:
            print(f"Error listing training data directory: {e}")
    else:
        print(f"Training data directory does not exist")
        
# Check for model directory
model_dir = os.environ.get('SM_MODEL_DIR', None)
if model_dir:
    print(f"Model directory: {model_dir}")
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir, exist_ok=True)
            print(f"Created model directory")
        except Exception as e:
            print(f"Error creating model directory: {e}")
print("==================================\n") 