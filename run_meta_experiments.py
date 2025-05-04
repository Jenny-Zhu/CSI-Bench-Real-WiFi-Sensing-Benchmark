import os
import argparse
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("meta_experiments.log"),
        logging.StreamHandler()
    ]
)

# Model types to evaluate
MODEL_TYPES = ['mlp', 'lstm', 'resnet18', 'transformer', 'vit']

# Test types to run
TEST_TYPES = [
    'test',
    'cross_env',
    'cross_user',
    'cross_device',
    'adapt_1shot',
    'adapt_5shot',
    'cross_env_adapt_1shot',
    'cross_env_adapt_5shot',
    'cross_user_adapt_1shot',
    'cross_user_adapt_5shot',
    'cross_device_adapt_1shot',
    'cross_device_adapt_5shot'
]

def run_command(cmd, description):
    """Run a command and log the output"""
    logging.info(f"Running: {description}")
    logging.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output
        for line in process.stdout:
            logging.info(line.strip())
        
        # Wait for the process to complete
        process.wait()
        elapsed_time = time.time() - start_time
        
        if process.returncode == 0:
            logging.info(f"Success: {description} completed in {elapsed_time:.2f} seconds")
            return True
        else:
            logging.error(f"Failed: {description} failed with code {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"Error: {description} - {str(e)}")
        return False

def train_model(model_type, args):
    """Train a model of the specified type"""
    cmd = [
        "python", "train_meta_all_models.py",
        "--model_type", model_type,
        "--data_dir", args.data_dir,
        "--task_name", args.task_name,
        "--n_way", str(args.n_way),
        "--k_shot", str(args.k_shot),
        "--q_query", str(args.q_query),
        "--batch_size", str(args.batch_size),
        "--num_iterations", str(args.num_iterations),
        "--inner_lr", str(args.inner_lr),
        "--meta_lr", str(args.meta_lr),
        "--eval_interval", str(args.eval_interval),
        "--save_dir", os.path.join(args.save_dir, "training"),
        "--seed", str(args.seed),
        "--device", args.device
    ]
    
    return run_command(cmd, f"Training {model_type.upper()} model")

def test_model(model_type, test_type, args):
    """Test a model of the specified type on a specific test type"""
    # Determine the path to the best model checkpoint
    checkpoint_path = os.path.join(
        args.save_dir, 
        "training", 
        f"{model_type}_{args.task_name}", 
        "best_model.pth"
    )
    
    # Check if the checkpoint exists
    if not os.path.exists(checkpoint_path):
        logging.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    cmd = [
        "python", "test_meta_models.py",
        "--checkpoint", checkpoint_path,
        "--data_dir", args.data_dir,
        "--task_name", args.task_name,
        "--test_type", test_type,
        "--n_way", str(args.n_way),
        "--batch_size", str(args.batch_size),
        "--inner_lr", str(args.inner_lr),
        "--adaptation_steps", str(args.adaptation_steps),
        "--output_dir", os.path.join(args.save_dir, "testing"),
        "--device", args.device
    ]
    
    return run_command(cmd, f"Testing {model_type.upper()} model on {test_type}")

def compare_models(args):
    """Compare all models across test types"""
    cmd = [
        "python", "compare_meta_models.py",
        "--results_dir", os.path.join(args.save_dir, "testing"),
        "--model_types"
    ] + MODEL_TYPES + [
        "--test_types"
    ] + TEST_TYPES + [
        "--output_dir", os.path.join(args.save_dir, "comparison")
    ]
    
    return run_command(cmd, "Comparing all models")

def main():
    parser = argparse.ArgumentParser(description='Run meta-learning experiments')
    parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                        help='Name of the task')
    parser.add_argument('--n_way', type=int, default=3,
                        help='Number of classes per task')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of support examples per class')
    parser.add_argument('--q_query', type=int, default=5,
                        help='Number of query examples per class')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of tasks per batch')
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='Number of training iterations')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Meta learning rate')
    parser.add_argument('--eval_interval', type=int, default=50,
                        help='Interval for evaluation during training')
    parser.add_argument('--adaptation_steps', type=int, default=5,
                        help='Number of adaptation steps during testing')
    parser.add_argument('--save_dir', type=str, default='results/meta_experiments',
                        help='Root directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--models', type=str, nargs='+', default=MODEL_TYPES,
                        help='Model types to train and evaluate')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only run testing')
    parser.add_argument('--skip_testing', action='store_true',
                        help='Skip testing and only run training')
    args = parser.parse_args()
    
    # Create save directories
    Path(os.path.join(args.save_dir, "training")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.save_dir, "testing")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.save_dir, "comparison")).mkdir(parents=True, exist_ok=True)
    
    # Start experiments
    logging.info("Starting meta-learning experiments")
    logging.info(f"Models to evaluate: {', '.join(args.models)}")
    
    # Run training for each model type
    if not args.skip_training:
        for model_type in args.models:
            if model_type not in MODEL_TYPES:
                logging.warning(f"Unknown model type: {model_type}, skipping")
                continue
            
            success = train_model(model_type, args)
            if not success:
                logging.error(f"Training failed for {model_type}, skipping testing")
                continue
    
    # Run testing for each model type and test type
    if not args.skip_testing:
        for model_type in args.models:
            if model_type not in MODEL_TYPES:
                continue
                
            for test_type in TEST_TYPES:
                success = test_model(model_type, test_type, args)
                if not success:
                    logging.warning(f"Testing failed for {model_type} on {test_type}")
    
    # Compare all models
    compare_models(args)
    
    logging.info("Experiments completed")

if __name__ == "__main__":
    main() 