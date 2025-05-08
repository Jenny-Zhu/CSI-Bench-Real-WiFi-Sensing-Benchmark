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

# Manually install peft library, bypassing version dependency checks
print("Installing peft library and its dependencies...")
try:
    # Install accelerate first
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

# Get and optimize command line arguments
args = sys.argv[1:]
print(f"Original command line arguments: {args}")

# Fix parameter name format - Convert hyphenated parameters to underscore format
formatted_args = []
i = 0
while i < len(args):
    arg = args[i]
    # Fix parameter name format (e.g., --learning-rate becomes --learning_rate)
    if arg.startswith('--'):
        fixed_arg = arg.replace('-', '_')
        if fixed_arg != arg:
            print(f"Fixing parameter format: {arg} -> {fixed_arg}")
            formatted_args.append(fixed_arg)
        else:
            formatted_args.append(arg)
    else:
        formatted_args.append(arg)
    i += 1

args = formatted_args
print(f"Parameters after format fixing: {args}")

# Continue with other parameter optimizations - Reduce batch_size to lower memory usage
modified_args = []
i = 0
while i < len(args):
    if args[i] == '--batch_size' and i+1 < len(args):
        try:
            batch_size = int(args[i+1])
            if batch_size > 4:
                print(f"Warning: Detected large batch_size ({batch_size}), automatically reduced to 4 to lower memory usage")
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
    print("Note: Command line arguments adjusted to optimize memory usage")
    args = modified_args

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

# Modify wrapper script template, add test environment support
wrapper_content = f"""#!/usr/bin/env python3
# Automatically generated wrapper script to avoid Horovod dependency conflicts and optimize memory usage
import sys
import os
import gc
import subprocess

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

# Check if PEFT and accelerate are available, install if needed
try:
    import peft
    print(f"PEFT library available: {{peft.__version__}}")
except ImportError:
    print("Installing PEFT and its dependencies...")
    try:
        # Install accelerate first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate", "--no-dependencies"])
        # Then install PEFT
        subprocess.check_call([sys.executable, "-m", "pip", "install", "peft==0.3.0", "--no-dependencies"])
        print("PEFT installation successful")
    except Exception as e:
        print(f"Error installing PEFT: {{e}}")

# Preprocess command line arguments - Fix parameter format (hyphens to underscores)
args = sys.argv[1:]
print(f"Original command line arguments: {{args}}")

formatted_args = []
i = 0
while i < len(args):
    arg = args[i]
    # Fix parameter name format (e.g., --learning-rate becomes --learning_rate)
    if arg.startswith('--'):
        fixed_arg = arg.replace('-', '_')
        if fixed_arg != arg:
            print(f"Fixing parameter format: {{arg}} -> {{fixed_arg}}")
            formatted_args.append(fixed_arg)
        else:
            formatted_args.append(arg)
    else:
        formatted_args.append(arg)
    i += 1

sys.argv[1:] = formatted_args
print(f"Parameters after format fixing: {{formatted_args}}")

# Environment test mode check
if os.environ.get('SM_HP_TEST_ENV') == 'True':
    print("Wrapper script detected environment test mode...")

# Import torch and configure memory optimizations
try:
    import torch
    if torch.cuda.is_available():
        # Limit memory usage
        torch.cuda.set_per_process_memory_fraction(0.7)
        torch.cuda.empty_cache()
        
        # Use deterministic algorithms, may reduce performance but increase stability
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
except Exception as e:
    print(f"Error configuring PyTorch: {{e}}")

# Actively perform garbage collection
gc.collect()

# Then import and execute the original script
import {script_to_run.replace('.py', '')}

# Call the main function of the original script (if it exists)
if hasattr({script_to_run.replace('.py', '')}, 'main'):
    {script_to_run.replace('.py', '')}.main()
"""

# Write temporary wrapper script
wrapper_script = "wrapper_script.py"
with open(wrapper_script, "w") as f:
    f.write(wrapper_content)
print(f"Created optimized wrapper script: {wrapper_script}")

print(f"\nStarting execution of optimized script {script_to_run}...\n")

# Run the wrapper script with the same arguments
try:
    print("Using memory optimized wrapper script...")
    ret = check_call([sys.executable, wrapper_script] + args)
    sys.exit(ret)
except Exception as e:
    print(f"Error running script: {e}")
    
    # As a fallback, try running the original script directly
    print("\nAttempting to run the original script as a fallback...")
    try:
        ret = check_call([sys.executable, script_to_run] + args)
        sys.exit(ret)
    except Exception as e2:
        print(f"Running original script also failed: {e2}")
        sys.exit(1) 