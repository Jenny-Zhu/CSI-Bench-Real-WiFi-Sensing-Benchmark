import os
import json
import shutil

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def copy_configs_to_expected_paths():
    """Copy config files to the expected locations"""
    # Ensure config directories exist
    ensure_dir('configs')
    ensure_dir('configs/meta')
    ensure_dir('data/meta')
    ensure_dir('results/meta_testing')
    
    # Check if configs exist and create if not
    config_models = ['mlp', 'lstm', 'resnet18', 'transformer', 'vit']
    for model_type in config_models:
        config_path = f'configs/meta/{model_type}_config.json'
        
        if not os.path.exists(config_path):
            # Create default config
            config = {
                "model_type": model_type,
                "win_len": 232,
                "feature_size": 500,
                "in_channels": 1,
                "emb_dim": 128,
                "dropout": 0.1,
                "inner_lr": 0.01,
                "meta_lr": 0.001,
                "n_way": 3,
                "k_shot": 5,
                "q_query": 5,
                "batch_size": 1,  # Use 1 to avoid tensor size mismatch errors
                "num_iterations": 100
            }
            
            # Add model-specific parameters
            if model_type == 'lstm':
                config["hidden_size"] = 128
                config["num_layers"] = 2
            elif model_type == 'transformer':
                config["nhead"] = 8
                config["num_layers"] = 4
            elif model_type == 'vit':
                config["depth"] = 6
                config["num_heads"] = 4
                config["mlp_ratio"] = 4.0
            
            # Save config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            
            print(f"Created config file: {config_path}")
        else:
            print(f"Config file exists: {config_path}")

def create_test_wrappers():
    """Create wrapper scripts for testing with proper structure"""
    
    # Create a wrapper for test_meta_adaptation.py
    with open('run_adaptation_test.py', 'w', encoding='utf-8') as f:
        f.write('''"""
Wrapper script for test_meta_adaptation.py to ensure proper structure
"""
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run meta-adaptation test with proper structure')
    parser.add_argument('--model_type', type=str, default='mlp', 
                        choices=['mlp', 'lstm', 'resnet18', 'transformer', 'vit'],
                        help='Type of model to test')
    parser.add_argument('--test_type', type=str, default='test',
                        choices=['test', 'cross_env', 'cross_user', 'cross_device',
                                'adapt_1shot', 'adapt_5shot', 'cross_env_adapt_1shot',
                                'cross_env_adapt_5shot', 'cross_user_adapt_1shot',
                                'cross_user_adapt_5shot', 'cross_device_adapt_1shot',
                                'cross_device_adapt_5shot'],
                        help='Type of test to run')
    parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--checkpoint', type=str, 
                        help='Path to model checkpoint (if not provided, uses latest from results)')
    args = parser.parse_args()
    
    # Determine checkpoint path if not provided
    if not args.checkpoint:
        model_dir = f'results/meta/{args.model_type}_5shot'
        if os.path.exists(model_dir):
            # Find the best model checkpoint
            best_checkpoint = os.path.join(model_dir, 'best_model.pth')
            if os.path.exists(best_checkpoint):
                args.checkpoint = best_checkpoint
                print(f"Using checkpoint: {args.checkpoint}")
            else:
                print(f"No checkpoint found in {model_dir}")
                return
        else:
            print(f"Model directory {model_dir} not found.")
            return
    
    # Build command to run test_meta_adaptation.py
    cmd = f"python test_meta_adaptation.py --checkpoint {args.checkpoint} --test_type {args.test_type} --data_dir {args.data_dir} --task_name MotionSourceRecognition"
    
    print(f"Running command: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    main()
''')
    
    print("Created run_adaptation_test.py wrapper script")

def main():
    # Copy configs to expected paths
    copy_configs_to_expected_paths()
    
    # Create test wrappers
    create_test_wrappers()
    
    print("\nFile structure fixed to match script expectations.")
    print("Now you can use the following scripts without path issues:")
    print("1. train_meta_standalone.py - for training")
    print("2. run_adaptation_test.py - for testing")

if __name__ == "__main__":
    main() 