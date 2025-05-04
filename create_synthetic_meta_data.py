import os
import argparse
import numpy as np
import random
import h5py
import json
from pathlib import Path
import shutil

def create_class_directories(output_dir, classes):
    """Create class directories for meta-learning"""
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_env_dir = os.path.join(output_dir, 'test_cross_env')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_env_dir, exist_ok=True)
    
    # Create class directories in each split
    for split_dir in [train_dir, val_dir, test_env_dir]:
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
    
    return train_dir, val_dir, test_env_dir

def generate_synthetic_csi(win_len=250, feature_size=98):
    """Generate synthetic CSI data"""
    # Create random CSI data
    return np.random.randn(win_len, feature_size).astype(np.float32)

def create_h5_file(filepath, data, data_key='CSI_amps'):
    """Create an h5 file with synthetic data"""
    with h5py.File(filepath, 'w') as f:
        f.create_dataset(data_key, data=data)

def create_synthetic_dataset(output_dir, classes, samples_per_class, win_len=250, feature_size=98, data_key='CSI_amps'):
    """Create a synthetic dataset for meta-learning"""
    # Create directory structure
    train_dir, val_dir, test_env_dir = create_class_directories(output_dir, classes)
    
    # Track file counts
    file_counts = {
        'train': {cls: 0 for cls in classes},
        'val': {cls: 0 for cls in classes},
        'test_cross_env': {cls: 0 for cls in classes}
    }
    
    # Generate different environments by shifting the distribution slightly
    environments = {
        'train': 0.0,  # base environment
        'val': 0.5,    # small shift
        'test_cross_env': 2.0  # larger shift for cross-environment testing
    }
    
    # Generate data for each split
    splits = {
        'train': {'dir': train_dir, 'ratio': 0.7},
        'val': {'dir': val_dir, 'ratio': 0.15},
        'test_cross_env': {'dir': test_env_dir, 'ratio': 0.15}
    }
    
    # Generate data
    all_files = []
    
    for split_name, split_info in splits.items():
        split_dir = split_info['dir']
        num_samples = int(samples_per_class * split_info['ratio'])
        env_shift = environments[split_name]
        
        print(f"Generating {num_samples} samples per class for {split_name} split...")
        
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(split_dir, class_name)
            
            for i in range(num_samples):
                # Generate synthetic CSI data with environment shift
                data = generate_synthetic_csi(win_len, feature_size) + env_shift * class_idx
                
                # Create filename
                file_id = f"{class_name}_{split_name}_{i:03d}"
                filepath = os.path.join(class_dir, f"{file_id}.h5")
                
                # Save file
                create_h5_file(filepath, data, data_key)
                
                # Add to tracking
                file_counts[split_name][class_name] += 1
                all_files.append({
                    'split': split_name,
                    'class': class_name,
                    'file_id': file_id,
                    'path': filepath
                })
    
    # Create split files
    for split_name in splits.keys():
        split_files = [f['file_id'] for f in all_files if f['split'] == split_name]
        with open(os.path.join(output_dir, f"{split_name}.json"), 'w') as f:
            json.dump(split_files, f)
    
    # Print summary
    print("\nSynthetic dataset created:")
    for split_name, counts in file_counts.items():
        print(f"  {split_name.capitalize()} split:")
        for cls, count in counts.items():
            print(f"    Class {cls}: {count} files")
    
    return all_files

def verify_synthetic_dataset(output_dir, data_key='CSI_amps'):
    """Verify the synthetic dataset"""
    print("\nVerifying synthetic dataset...")
    
    for split in ['train', 'val', 'test_cross_env']:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        # Get class directories
        class_dirs = [d for d in os.listdir(split_dir) 
                      if os.path.isdir(os.path.join(split_dir, d))]
        
        print(f"  {split.capitalize()} split - {len(class_dirs)} classes: {', '.join(class_dirs)}")
        
        # Check each class
        for cls in class_dirs:
            cls_dir = os.path.join(split_dir, cls)
            files = [f for f in os.listdir(cls_dir) if f.endswith('.h5')]
            
            if not files:
                print(f"    Class {cls}: No files found")
                continue
            
            # Sample a random file
            sample_file = random.choice(files)
            file_path = os.path.join(cls_dir, sample_file)
            
            try:
                with h5py.File(file_path, 'r') as f:
                    if data_key in f:
                        data = f[data_key][()]
                        print(f"    Class {cls}: {len(files)} files, sample shape: {data.shape}")
                    else:
                        print(f"    Class {cls}: {len(files)} files, but data key '{data_key}' not found")
            except Exception as e:
                print(f"    Class {cls}: Error reading file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Create synthetic dataset for meta-learning')
    parser.add_argument('--output_dir', type=str, default='synthetic_meta_data',
                       help='Output directory for synthetic dataset')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--samples_per_class', type=int, default=100,
                       help='Total samples per class across all splits')
    parser.add_argument('--win_len', type=int, default=250,
                       help='Window length for synthetic CSI data')
    parser.add_argument('--feature_size', type=int, default=98,
                       help='Feature size for synthetic CSI data')
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                       help='Key in h5 file for CSI data')
    parser.add_argument('--clear', action='store_true',
                       help='Clear output directory if it exists')
    args = parser.parse_args()
    
    # Create or clear output directory
    if os.path.exists(args.output_dir) and args.clear:
        print(f"Clearing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate class names
    classes = [f"Class{i}" for i in range(args.num_classes)]
    
    print(f"Creating synthetic dataset with {args.num_classes} classes: {classes}")
    print(f"Window length: {args.win_len}, Feature size: {args.feature_size}")
    print(f"Total samples per class: {args.samples_per_class}")
    
    # Create synthetic dataset
    create_synthetic_dataset(
        output_dir=args.output_dir,
        classes=classes,
        samples_per_class=args.samples_per_class,
        win_len=args.win_len,
        feature_size=args.feature_size,
        data_key=args.data_key
    )
    
    # Verify the dataset
    verify_synthetic_dataset(args.output_dir, args.data_key)
    
    print(f"\nSynthetic dataset created in {args.output_dir}")
    print("You can now use train_mlp_meta.py to train with this data")

if __name__ == "__main__":
    main() 