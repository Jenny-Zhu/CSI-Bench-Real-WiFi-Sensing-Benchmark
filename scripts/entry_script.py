#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SageMaker Entry Script - Disable Horovod and SMDebug

This script disables Horovod and SMDebug integration before importing PyTorch to avoid version conflicts, then runs the actual training script.
Main functions:
1. Disable Horovod and SMDebug
2. Install peft and its dependencies
3. Directly pass all parameters to the training script
"""

import os
import sys
import subprocess
import gc
import logging

print("\n==========================================")
print("Starting custom entry script entry_script.py")
print("==========================================\n")

# Set memory optimization options
print("Configuring memory optimization settings...")
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TensorFlow logs (if used)

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

# Set up typing_extensions support to resolve issues with typing.Literal imports
print("Setting up typing_extensions support...")
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
    print("transformers library installed successfully")
    
    # Install accelerate
    subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate", "--no-dependencies"])
    print("accelerate library installed successfully")
    
    # Then install peft with --no-dependencies parameter
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft==0.3.0", "--no-dependencies"])
    print("peft library installed successfully")
except Exception as e:
    print(f"Error installing peft library: {e}")
    # This is not a fatal error, try to continue execution

# Free up some memory
gc.collect()

# Set up paths
print("Setting up paths...")
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
    print(f"  Memory usage percentage: {virtual_memory.percent}%")
except ImportError:
    print("Could not import psutil, skipping memory information display")
except Exception as e:
    print(f"Error getting memory information: {e}")

# Now run the actual training script
print("Preparing to run training script...")

# Check script to execute
# Check SAGEMAKER_PROGRAM environment variable
script_dir = os.path.dirname(os.path.abspath(__file__))
default_script = os.path.join(script_dir, 'train_multi_model.py')
script_to_run = os.environ.get('SAGEMAKER_PROGRAM', default_script)

# Set environment variable for compatibility with SageMaker
if 'SAGEMAKER_PROGRAM' not in os.environ:
    os.environ['SAGEMAKER_PROGRAM'] = default_script

print(f"Script to execute: {script_to_run}")

# Check if script exists
if not os.path.exists(script_to_run):
    print(f"Error: Script {script_to_run} not found")
    print(f"Python files in current directory: {[f for f in os.listdir('.') if f.endswith('.py')]}")
    sys.exit(1)

# Print environment information
print("\n===== Environment Information =====")
import platform
print(f"Python version: {platform.python_version()}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {', '.join(os.listdir('.'))[:200]}...")

# Print environment variables
print("\n===== Environment Variables =====")
sm_vars = [k for k in os.environ.keys() if k.startswith('SM_') or k.startswith('SAGEMAKER_')]
for var in sm_vars:
    print(f"{var}: {os.environ.get(var)}")
print("==================================\n")

# Add critical parameter check logic
critical_params = ['task_name', 'models', 'win_len', 'feature_size', 'batch_size']
print("\n===== Critical Parameter Check =====")
for param in critical_params:
    # Check both dash and underscore format
    dash_param = param.replace('_', '-')
    underscore_param = param.replace('-', '_')
    
    dash_env_var = f"SM_HP_{dash_param.upper()}"
    underscore_env_var = f"SM_HP_{underscore_param.upper()}"
    
    # Check if parameter exists in environment variables
    if dash_env_var in os.environ:
        print(f"✓ {param} found as {dash_env_var}: {os.environ[dash_env_var]}")
        # Special check for batch-size/batch_size
        if param == 'batch_size':
            print(f"   IMPORTANT: batch_size value from {dash_env_var}: {os.environ[dash_env_var]}")
    elif underscore_env_var in os.environ:
        print(f"✓ {param} found as {underscore_env_var}: {os.environ[underscore_env_var]}")
        # Special check for batch-size/batch_size
        if param == 'batch_size':
            print(f"   IMPORTANT: batch_size value from {underscore_env_var}: {os.environ[underscore_env_var]}")
    else:
        print(f"✗ {param} NOT FOUND in environment variables")
        # Special emphasis on missing batch_size
        if param == 'batch_size':
            print(f"   WARNING: batch_size parameter not found! Will use default value (32)!")
print("==================================\n")

# Try importing torch to verify it loads correctly without Horovod conflicts
try:
    print("Attempting to import PyTorch (with memory limits)...")
    import torch
    # Configure PyTorch memory allocator optimizations
    if torch.cuda.is_available():
        # Limit GPU memory growth
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of available GPU memory
        # Actively clear CUDA cache
        torch.cuda.empty_cache()
        
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU total memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"Current allocated GPU memory: {torch.cuda.memory_allocated(0) / (1024**3):.2f} GB")
        print(f"Current GPU memory cache: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB")
    print("Successfully imported PyTorch without Horovod conflicts")
    
    # Initialize distributed training environment (if in SageMaker environment)
    if is_sagemaker:
        print("\n===== Checking Distributed Training Environment =====")
        # Get distributed training related environment variables
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        world_rank = int(os.environ.get('RANK', '0'))
        
        print(f"Distributed training environment information:")
        print(f"  LOCAL_RANK: {local_rank}")
        print(f"  WORLD_SIZE: {world_size}")
        print(f"  RANK: {rank}")
        
        # Initialize distributed backend if there are multiple GPUs or nodes
        if world_size > 1:
            try:
                print("Initializing distributed backend...")
                # Initialize distributed environment
                torch.distributed.init_process_group(backend='nccl')
                # Set current device
                torch.cuda.set_device(local_rank)
                print(f"Distributed environment initialization successful, process {rank}/{world_size}, local rank {local_rank}")
                
                # Verify that distributed environment is correctly set up
                if torch.distributed.is_initialized():
                    # Confirm current process group size
                    dist_world_size = torch.distributed.get_world_size()
                    dist_rank = torch.distributed.get_rank()
                    print(f"Verifying distributed environment: Process group size={dist_world_size}, Current process rank={dist_rank}")
                else:
                    print("Warning: Distributed environment not properly initialized")
                
                # Synchronize all nodes
                torch.distributed.barrier()
                print("All nodes synchronized")
            except Exception as e:
                print(f"Error initializing distributed environment: {e}")
        else:
            print("Single GPU training environment, no need to initialize distributed backend")
        
        print("===== Distributed Training Environment Setup Complete =====\n")
    
    # Check if in test environment mode
    test_env = os.environ.get('SM_HP_TEST_ENV') == 'True'
    if test_env:
        print("\n==========================================")
        print("Running in test environment mode, skipping data download and full training...")
        print("==========================================\n")
        
        # Create simple simulation tests
        import time
        import importlib
        
        print("Verifying PyTorch GPU availability...")
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            # Perform simple GPU computation to verify functionality
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
                print(f"✗ Failed to import {module}: {e}")
                failed += 1

        print(f"\nImport test results: {success} successful, {failed} failed")
        
        # Exit successfully after environment test
        print("\nEnvironment test completed successfully, exiting")
        sys.exit(0)

except Exception as e:
    print(f"Warning when importing PyTorch: {e}")

# Get command-line arguments and prepare to run the training script
print("\nPassing arguments to training script...")

# Get the original command-line arguments
print(f"Original command-line arguments: {sys.argv}")

# Build command to run the actual training script
cmd = [sys.executable, script_to_run]

# Collect hyperparameters from environment variables
hyperparameters = {}
orig_hyperparameters = {}
for key, value in os.environ.items():
    if key.startswith('SM_HP_'):
        # Convert SM_HP_X to x
        param_name = key[6:].lower()
        orig_hyperparameters[param_name] = value
        hyperparameters[param_name] = value
        
        # Add hyperparameters to command line
        if param_name != 'test_env':  # Skip the test_env parameter
            # Convert dash-style parameters to underscore-style for compatibility with train_multi_model.py
            fixed_param_name = param_name.replace("-", "_")
            
            # Special handling for task_name parameter - ensure it's passed correctly
            if param_name == 'task_name' or fixed_param_name == 'task_name':
                fixed_param_name = 'task_name'  # Force the correct parameter name
            
            # Log parameter conversion for debugging
            if fixed_param_name != param_name:
                print(f"Converting parameter format: {param_name} -> {fixed_param_name}")
            
            # Important: Ensure we use proper double dash prefix
            param_prefix = "--"  # Always use double dash
            
            # Handle special boolean values and normalize them
            if value.lower() in ('true', 'yes', '1', 't', 'y'):
                cmd.append(f"{param_prefix}{fixed_param_name}")
            elif value.lower() in ('false', 'no', '0', 'f', 'n'):
                # For False boolean values, don't add the parameter
                pass
            else:
                cmd.append(f"{param_prefix}{fixed_param_name}")
                
                # Remove unnecessary quotes that might cause parsing issues
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                    
                cmd.append(value)

# Check for critical missing parameters
critical_params = {
    'win_len': '500',            # Default from logs
    'feature_size': '232',       # Default from logs
    'batch_size': '32',          # Default from logs
    'learning_rate': '0.001',    # Common default
    'weight_decay': '1e-5',      # Common default
    'warmup_epochs': '5',        # Common default
    'test_splits': 'all'         # Common default
}

# Special handling for dashes vs underscores
print("\n===== Command Validation =====")
missing_params = []
for param, default in critical_params.items():
    # Check if parameter is already in command line
    dash_param = param.replace('_', '-')
    param_in_cmd = False
    for arg in cmd:
        if arg == f"--{param}" or arg == f"--{dash_param}":
            param_in_cmd = True
            break
    
    # If not in command, try to add from environment
    if not param_in_cmd:
        env_key_underscore = f"SM_HP_{param.upper()}"
        env_key_dash = f"SM_HP_{dash_param.upper()}"
        
        if env_key_underscore in os.environ:
            cmd.append(f"--{param}")
            cmd.append(os.environ[env_key_underscore])
            print(f"Added missing parameter from env: --{param}={os.environ[env_key_underscore]}")
        elif env_key_dash in os.environ:
            cmd.append(f"--{param}")
            cmd.append(os.environ[env_key_dash])
            print(f"Added missing parameter from env: --{param}={os.environ[env_key_dash]}")
        else:
            # Parameter not in command line or environment, use default
            cmd.append(f"--{param}")
            cmd.append(default)
            print(f"Added missing parameter with default: --{param}={default}")
            missing_params.append(param)

# Fix any invalid prefixes and ensure all parameters use proper -- prefix
print("\n----- Parameter Format Check -----")
for i, arg in enumerate(cmd):
    # Check for invalid prefixes
    if arg.startswith('__'):  
        cmd[i] = '--' + arg[2:]  # Replace __ with --
        print(f"⚠️ CRITICAL FIX: Fixed invalid double underscore prefix: {arg} -> {cmd[i]}")
    # Check for single dash parameters that should be double dash
    elif arg.startswith('-') and not arg.startswith('--') and not arg[1:].isdigit():
        if len(arg) > 1:  # Ensure it's not just a negative number
            cmd[i] = '--' + arg[1:]  # Replace - with --
            print(f"⚠️ Fixed single dash prefix: {arg} -> {cmd[i]}")
    # Check for parameters with dash instead of underscore
    elif arg.startswith('--') and '-' in arg[2:]:
        old_arg = arg
        cmd[i] = '--' + arg[2:].replace('-', '_')
        print(f"ℹ️ Standardized parameter format: {old_arg} -> {cmd[i]}")

if missing_params:
    print(f"\n⚠️ WARNING: Had to use defaults for: {', '.join(missing_params)}")
print("=============================\n")

print(f"Original hyperparameters: {orig_hyperparameters}")
print(f"Parsed hyperparameters: {hyperparameters}")
print(f"Final command: {' '.join(cmd)}")

# Extra verification step to ensure no arguments have __ prefix
verified_cmd = []
for i, arg in enumerate(cmd):
    if arg.startswith('__'):
        fixed_arg = '--' + arg[2:]
        print(f"🚨 CRITICAL FIX: Replaced double underscore prefix: {arg} -> {fixed_arg}")
        verified_cmd.append(fixed_arg)
    else:
        verified_cmd.append(arg)

# Use the verified command
cmd = verified_cmd

# Final check to verify the command has proper prefixes
prefix_issues = False
for arg in cmd:
    if arg.startswith('__'):
        print(f"🚨 ERROR: Command still contains invalid prefix: {arg}")
        prefix_issues = True

if prefix_issues:
    print("🚨 WARNING: Command contains invalid prefixes after fixes. This might cause parameter parsing errors.")
else:
    print("✓ Command verification passed: All parameters have correct prefixes")

# Execute the training script with the parameters
try:
    # Execute the script and wait for it to complete
    exit_code = subprocess.call(cmd)
    sys.exit(exit_code)
except Exception as e:
    print(f"Error executing training script: {e}")
    sys.exit(1) 