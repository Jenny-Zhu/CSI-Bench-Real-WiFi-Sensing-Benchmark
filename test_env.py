#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple SageMaker Environment Test Script

This script prints information about the SageMaker environment,
including PyTorch version and CUDA compatibility.
"""

import sys
import os
import subprocess
import json

def print_header(title):
    """Print a section header"""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def check_package_version(package_name):
    """Check and print package version"""
    try:
        if package_name == "torch":
            import torch
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else "N/A"
            print(f"{package_name}: {version} (CUDA: {cuda_available}, version: {cuda_version})")
            if cuda_available:
                device_count = torch.cuda.device_count()
                print(f"CUDA device count: {device_count}")
                for i in range(device_count):
                    print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            return True
        elif package_name == "torchvision":
            import torchvision
            print(f"{package_name}: {torchvision.__version__}")
            return True
        elif package_name == "transformers":
            import transformers
            print(f"{package_name}: {transformers.__version__}")
            return True
        elif package_name == "peft":
            import peft
            print(f"{package_name}: {peft.__version__}")
            return True
        else:
            # Generic package checking
            module = __import__(package_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{package_name}: {version}")
            return True
    except ImportError:
        print(f"{package_name}: Not installed")
        return False
    except Exception as e:
        print(f"{package_name}: Error - {str(e)}")
        return False

def main():
    """Main function to test the environment"""
    print_header("SageMaker Environment Test Script")
    
    # Print Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check for SageMaker environment
    print("\nChecking SageMaker environment variables:")
    sm_vars = [var for var in os.environ if var.startswith("SM_")]
    if sm_vars:
        print("SageMaker environment detected")
        print(f"Model directory: {os.environ.get('SM_MODEL_DIR', 'N/A')}")
        print(f"Training data directory: {os.environ.get('SM_CHANNEL_TRAINING', 'N/A')}")
    else:
        print("Not running in SageMaker environment")
    
    # Check packages
    print_header("Package Versions")
    packages = [
        "torch", "torchvision", "numpy", "pandas", 
        "matplotlib", "scipy", "sklearn", "peft", "transformers"
    ]
    
    for package in packages:
        check_package_version(package)
    
    # Check Horovod
    print("\nChecking Horovod:")
    try:
        import horovod.torch
        print("Horovod is installed and compatible with PyTorch")
    except ImportError:
        print("Horovod is not installed")
    except Exception as e:
        print(f"Horovod compatibility issue: {str(e)}")
    
    # Check filesystem
    print_header("Filesystem Check")
    for dir_path in ["/opt/ml/input/data/training", "/opt/ml/model"]:
        if os.path.exists(dir_path):
            print(f"{dir_path}: Exists")
            try:
                items = os.listdir(dir_path)
                print(f"  Contains {len(items)} items")
                if items:
                    print(f"  First few items: {', '.join(items[:5])}")
            except Exception as e:
                print(f"  Error listing directory: {str(e)}")
        else:
            print(f"{dir_path}: Does not exist")
    
    # Requirements.txt content
    print_header("Requirements.txt Content")
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        try:
            with open(req_file, "r") as f:
                content = f.read()
            print(content)
        except Exception as e:
            print(f"Error reading {req_file}: {str(e)}")
    else:
        print(f"{req_file} not found")
    
    # Environment summary
    print_header("Environment Summary")
    print(f"OS: {os.uname() if hasattr(os, 'uname') else sys.platform}")
    
    # Save report to file
    report = {
        "python_version": sys.version,
        "os": str(os.uname() if hasattr(os, 'uname') else sys.platform),
        "packages": {p: getattr(__import__(p), "__version__", "unknown") 
                    for p in packages if check_package_version(p)},
        "sagemaker_env": sm_vars is not None and len(sm_vars) > 0
    }
    
    try:
        with open(os.path.join(os.environ.get("SM_MODEL_DIR", "."), "env_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("Environment report saved to model directory")
    except Exception as e:
        print(f"Error saving report: {str(e)}")
    
    print_header("Test Complete")
    print("If PyTorch and all required packages are correctly installed,")
    print("you can proceed with your training job.")

if __name__ == "__main__":
    main() 