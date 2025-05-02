import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from load.supervised.benchmark_loader import load_benchmark_supervised
from tqdm import tqdm
import json

# Import model classes from models.py
from model.supervised.models import (
    MLPClassifier, 
    LSTMClassifier, 
    ResNet18Classifier, 
    TransformerClassifier, 
    ViTClassifier
)

# Import TaskTrainer
from engine.supervised.task_trainer import TaskTrainer

# Model factory dictionary
MODEL_TYPES = {
    'mlp': MLPClassifier,
    'lstm': LSTMClassifier,
    'resnet18': ResNet18Classifier,
    'transformer': TransformerClassifier,
    'vit': ViTClassifier
}

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description='Train a supervised model on WiFi benchmark dataset')
        parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset',
                            help='Root directory of the dataset')
        parser.add_argument('--task_name', type=str, default='MotionSourceRecognition',
                            help='Name of the task to train on')
        parser.add_argument('--model_type', type=str, default='vit', 
                            choices=['mlp', 'lstm', 'resnet18', 'transformer', 'vit'],
                            help='Type of model to train')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs to train')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--data_key', type=str, default='CSI_amps',
                            help='Key for CSI data in h5 files')
        parser.add_argument('--save_dir', type=str, default='checkpoints',
                            help='Directory to save checkpoints')
        parser.add_argument('--output_dir', type=str, default=None,
                            help='Directory to save results (defaults to save_dir if not specified)')
        parser.add_argument('--weight_decay', type=float, default=1e-5, 
                            help='Weight decay for optimizer')
        parser.add_argument('--warmup_epochs', type=int, default=5,
                            help='Number of warmup epochs')
        parser.add_argument('--patience', type=int, default=15,
                            help='Patience for early stopping')
        # Additional model parameters
        parser.add_argument('--win_len', type=int, default=500, 
                            help='Window length for WiFi CSI data')
        parser.add_argument('--feature_size', type=int, default=232, 
                            help='Feature size for WiFi CSI data')
        parser.add_argument('--in_channels', type=int, default=1, 
                            help='Number of input channels')
        parser.add_argument('--emb_dim', type=int, default=128, 
                            help='Embedding dimension for ViT model')
        parser.add_argument('--d_model', type=int, default=256, 
                            help='Model dimension for Transformer model')
        parser.add_argument('--dropout', type=float, default=0.1, 
                            help='Dropout rate')
        args = parser.parse_args()
    
    # Set output_dir to save_dir if not specified
    if args.output_dir is None:
        args.output_dir = args.save_dir
    
    # Create save directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create output directory if it's different from save_dir
    if args.output_dir != args.save_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Detect environment and set appropriate paths
    is_sagemaker = os.path.exists('/opt/ml/model')
    
    if is_sagemaker:
        print("Running in SageMaker environment")
        model_dir = '/opt/ml/model'
        # If running in SageMaker, ensure we save in the model directory with task/model structure
        results_dir = os.path.join(model_dir, args.task_name, args.model_type)
        os.makedirs(results_dir, exist_ok=True)
        # Also create the checkpoints directory under the task/model structure
        checkpoint_dir = os.path.join(args.save_dir, args.task_name, args.model_type)
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        print("Running in local environment")
        # Use specified directories for local environment with task/model structure
        results_dir = os.path.join(args.output_dir, args.task_name, args.model_type)
        os.makedirs(results_dir, exist_ok=True)
        # Create the checkpoints directory under the task/model structure
        checkpoint_dir = os.path.join(args.save_dir, args.task_name, args.model_type)
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_dir} for task {args.task_name}...")
    data = load_benchmark_supervised(
        dataset_root=args.data_dir,
        task_name=args.task_name,
        batch_size=args.batch_size,
        file_format="h5",
        data_key=args.data_key,
        num_workers=4
    )
    
    # Extract data from the returned dictionary
    loaders = data['loaders']
    num_classes = data['num_classes']
    label_mapper = data['label_mapper']
    
    # Get training and validation loaders
    train_loader = loaders['train']
    val_loader = loaders.get('val')
    
    if val_loader is None:
        print("Warning: No validation data found. Using training data for validation.")
        val_loader = train_loader
    
    # Count unique labels in the dataset
    all_labels = []
    dataset = train_loader.dataset
    print(f"Detected {num_classes} classes in the dataset")
    
    # Create model based on selected type
    print(f"Creating {args.model_type.upper()} model...")
    ModelClass = MODEL_TYPES[args.model_type]
    
    # Common parameters for all models
    model_params = {
        'num_classes': num_classes
    }
    
    # Add additional parameters based on model type
    if args.model_type in ['mlp', 'vit']:
        model_params.update({
            'win_len': args.win_len,
            'feature_size': args.feature_size
        })
    
    if args.model_type == 'resnet18':
        model_params.update({
            'in_channels': args.in_channels
        })
    
    if args.model_type == 'lstm':
        model_params.update({
            'feature_size': args.feature_size,
            'dropout': args.dropout
        })
    
    if args.model_type == 'transformer':
        model_params.update({
            'feature_size': args.feature_size,
            'd_model': args.d_model,
            'dropout': args.dropout
        })
    
    if args.model_type == 'vit':
        model_params.update({
            'emb_dim': args.emb_dim,
            'dropout': args.dropout,
            'in_channels': args.in_channels
        })
    
    model = ModelClass(**model_params)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create configuration object for trainer
    config = type('Config', (), {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'patience': args.patience,
        'num_classes': num_classes,
        'device': str(device),
        'save_dir': checkpoint_dir,  # Use the checkpoint_dir for model checkpoints
        'output_dir': results_dir,  # Use results_dir for output
        'results_subdir': 'supervised',
        'model_name': args.model_type,
        'task_name': args.task_name
    })
    
    # Save the configuration
    config_path = os.path.join(results_dir, f"{args.model_type}_{args.task_name}_config.json")
    with open(config_path, "w") as f:
        # Convert config to dict
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        json.dump(config_dict, f, indent=4)
    
    print(f"Configuration saved to {config_path}")
    
    # Create trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    trainer = TaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_path=checkpoint_dir,  # Use checkpoint_dir for model checkpoints
        num_classes=num_classes,
        label_mapper=label_mapper,
        config=config
    )
    
    # Train the model
    trained_model, training_results = trainer.train()
    
    # Evaluate on test set(s) if available
    print("\nEvaluating on test sets:")
    for key in loaders:
        if key.startswith('test'):
            print(f"\nEvaluating on {key} split:")
            test_loss, test_acc = trainer.evaluate(loaders[key])
            print(f"{key} loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")
            
            # Plot confusion matrix for this test set
            confusion_path = os.path.join(results_dir, f"{args.model_type}_{args.task_name}_{key}_confusion.png")
            trainer.plot_confusion_matrix(data_loader=loaders[key], mode=key, epoch=None)
    
    # Save training results summary
    results_file = os.path.join(results_dir, f"{args.model_type}_{args.task_name}_results.json")
    with open(results_file, 'w') as f:
        # Include only serializable data
        serializable_results = {
            'best_epoch': training_results['best_epoch'],
            'best_val_accuracy': float(training_results['best_val_accuracy']),  # Convert tensor to float if necessary
            'train_loss_history': [float(x) for x in training_results['train_loss_history']],
            'val_loss_history': [float(x) for x in training_results['val_loss_history']],
            'train_accuracy_history': [float(x) for x in training_results['train_accuracy_history']],
            'val_accuracy_history': [float(x) for x in training_results['val_accuracy_history']]
        }
        json.dump(serializable_results, f, indent=4)
    
    print(f"\nTraining completed. Results saved to {results_file}")
    print(f"Model checkpoints saved to {checkpoint_dir}")
    print(f"Final model and results saved to {results_dir}")

if __name__ == "__main__":
    main()