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

# Disable Horovod integration to avoid version conflict
print("Disabling Horovod integration...")
sys.modules['horovod'] = None
sys.modules['horovod.torch'] = None
sys.modules['horovod.tensorflow'] = None

# Disable SMDebug to avoid Horovod dependency
print("Disabling SageMaker Debugger (smdebug)...")
sys.modules['smdebug'] = None
os.environ['SMDEBUG_DISABLED'] = 'true'

# Set paths
print("Setting up paths...")
sys.path.insert(0, os.getcwd())

# Now run the actual training script
print("Preparing to run training script...")
from subprocess import check_call

# Get all command line arguments
args = sys.argv[1:]

# Print environment information
print("\n===== Environment Information =====")
import platform
print(f"Python version: {platform.python_version()}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")
print(f"Command: python3 train_multi_model.py {' '.join(args)}")
print("==================================\n")

# Try importing torch to verify it loads correctly without Horovod conflicts
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print("PyTorch imported successfully without Horovod conflict")
except Exception as e:
    print(f"Error importing PyTorch: {e}")
    sys.exit(1)  # Exit if we can't import PyTorch

print("\nStarting training script...\n")

# Run the actual training script with the same arguments
try:
    ret = check_call([sys.executable, "train_multi_model.py"] + args)
    sys.exit(ret)
except Exception as e:
    print(f"Error running training script: {e}")
    sys.exit(1) 