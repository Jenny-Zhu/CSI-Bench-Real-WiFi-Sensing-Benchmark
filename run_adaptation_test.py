"""
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
