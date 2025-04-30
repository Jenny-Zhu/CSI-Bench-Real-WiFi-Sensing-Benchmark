#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Shortcut Script

This script provides a simple entry point for training models.
It just forwards all arguments to scripts/run_model.py.

Usage:
    python train.py --model [model_name] --task [task_name]
"""

import sys
import subprocess

if __name__ == "__main__":
    # Forward all arguments to run_model.py
    args = " ".join(sys.argv[1:])
    command = f"python scripts/run_model.py {args}"
    
    print(f"Running: {command}")
    subprocess.run(command, shell=True) 