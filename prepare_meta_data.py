import os
import json
import argparse
import shutil
import random
import h5py
import numpy as np
import glob
from pathlib import Path

def load_splits(data_dir):
    """Load train/val/test splits"""
    splits = {}
    splits_dir = os.path.join(data_dir, 'splits')
    
    # Load train, val, and cross-environment test splits
    for split_name in ['train_id.json', 'val_id.json', 'test_cross_env.json']:
        split_path = os.path.join(splits_dir, split_name)
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                split_data = json.load(f)
                # Store with simplified name (remove .json suffix)
                splits[split_name.split('.')[0]] = split_data
    
    return splits

def load_metadata(data_dir):
    """Load dataset metadata"""
    metadata_path = os.path.join(data_dir, 'metadata', 'subset_metadata.csv')
    if os.path.exists(metadata_path):
        import pandas as pd
        return pd.read_csv(metadata_path)
    return None

def get_class_labels(metadata):
    """Extract class labels from metadata"""
    if 'label' in metadata.columns:
        return metadata['label'].unique().tolist()
    return None

def find_h5_files(data_dir):
    """Find all h5 files in the data directory recursively"""
    h5_files = []
    
    # Search recursively for .h5 files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    
    # If we didn't find any, try looking for specific patterns
    if not h5_files:
        patterns = [
            os.path.join(data_dir, '**', '*.h5'),
            os.path.join(data_dir, '*.h5'),
            os.path.join(data_dir, 'data', '**', '*.h5'),
            os.path.join(data_dir, 'data', '*.h5')
        ]
        
        for pattern in patterns:
            h5_files = glob.glob(pattern, recursive=True)
            if h5_files:
                print(f"Found {len(h5_files)} files using pattern: {pattern}")
                break
    
    return h5_files

def create_meta_data_structure(data_dir, output_dir, splits, metadata, data_key='CSI_amps'):
    """Create meta-learning dataset structure"""
    # Find h5 files
    print("\nSearching for h5 files...")
    h5_files = find_h5_files(data_dir)
    
    if not h5_files:
        print("No h5 files found! Please check your data directory structure.")
        return
    
    print(f"Found {len(h5_files)} h5 files")
    
    # Create a mapping from filename to full path
    file_path_map = {}
    for filepath in h5_files:
        filename = os.path.basename(filepath)
        # Remove .h5 extension if present
        if filename.endswith('.h5'):
            filename = filename[:-3]
        file_path_map[filename] = filepath
    
    # Print some example mappings
    print("Sample filename to path mappings:")
    sample_count = min(5, len(file_path_map))
    for i, (fname, fpath) in enumerate(list(file_path_map.items())[:sample_count]):
        print(f"  {fname} -> {fpath}")
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_cross_env_dir = os.path.join(output_dir, 'test_cross_env')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_cross_env_dir, exist_ok=True)
    
    # Map file IDs to metadata
    file_to_metadata = {}
    if metadata is not None:
        # Check which column might contain the file names
        file_cols = [col for col in metadata.columns if 'file' in col.lower()]
        if file_cols:
            file_col = file_cols[0]
            print(f"Using '{file_col}' column for file names")
            
            for _, row in metadata.iterrows():
                if file_col in row and 'label' in row:
                    file_id = row[file_col]
                    # Strip .h5 extension if present
                    if isinstance(file_id, str) and file_id.endswith('.h5'):
                        file_id = file_id[:-3]
                    file_to_metadata[file_id] = {
                        'label': row['label']
                    }
    
    # Process each split
    missing_files = []
    processed_files = 0
    
    for split_name, file_list in splits.items():
        # Skip if empty
        if not file_list:
            continue
        
        # Determine output directory
        if split_name == 'train_id':
            out_dir = train_dir
        elif split_name == 'val_id':
            out_dir = val_dir
        elif split_name == 'test_cross_env':
            out_dir = test_cross_env_dir
        else:
            continue
        
        # Count files per class
        class_counts = {}
        
        # Process each file
        for file_name in file_list:
            # Get class label from metadata
            if file_name in file_to_metadata:
                class_label = file_to_metadata[file_name]['label']
            else:
                # Try to extract class from filename (first part before underscore)
                class_label = file_name.split('_')[0]
            
            # Create class directory
            class_dir = os.path.join(out_dir, str(class_label))
            os.makedirs(class_dir, exist_ok=True)
            
            # Update counts
            if class_label not in class_counts:
                class_counts[class_label] = 0
            
            # Source file path - try different options
            source_path = None
            
            # Check if in our map
            if file_name in file_path_map:
                source_path = file_path_map[file_name]
            else:
                # Try some common patterns
                potential_paths = [
                    os.path.join(data_dir, 'data', f"{file_name}.h5"),
                    os.path.join(data_dir, f"{file_name}.h5"),
                    os.path.join(data_dir, 'data', file_name)
                ]
                
                for path in potential_paths:
                    if os.path.exists(path):
                        source_path = path
                        break
            
            # Only process if source exists
            if source_path and os.path.exists(source_path):
                # Create a symlink or copy the file
                dest_path = os.path.join(class_dir, f"{file_name}.h5")
                
                try:
                    # Try creating a symlink first (more efficient)
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.symlink(os.path.abspath(source_path), dest_path)
                except (OSError, AttributeError):
                    # Fall back to copying if symlinks aren't supported
                    shutil.copy(source_path, dest_path)
                
                processed_files += 1
                class_counts[class_label] += 1
            else:
                missing_files.append(file_name)
        
        # Print summary for this split
        print(f"Split {split_name} - Files per class:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")
    
    print(f"\nProcessed {processed_files} files total")
    if missing_files:
        print(f"Could not find {len(missing_files)} files")
        if len(missing_files) < 10:
            print(f"Missing files: {missing_files}")
        else:
            print(f"First 10 missing files: {missing_files[:10]}")

def verify_meta_data(output_dir, data_key='CSI_amps'):
    """Verify the meta-learning dataset structure"""
    for split in ['train', 'val', 'test_cross_env']:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        # Get class directories
        class_dirs = [d for d in os.listdir(split_dir) 
                      if os.path.isdir(os.path.join(split_dir, d))]
        
        if not class_dirs:
            print(f"Warning: No class directories found in {split_dir}")
            continue
        
        print(f"\nVerifying {split} split:")
        print(f"  Found {len(class_dirs)} classes: {', '.join(class_dirs)}")
        
        # Check each class
        for cls in class_dirs:
            cls_dir = os.path.join(split_dir, cls)
            files = [f for f in os.listdir(cls_dir) if f.endswith('.h5')]
            print(f"  Class {cls}: {len(files)} files")
            
            # Sample and verify a random file
            if files:
                sample_file = random.choice(files)
                file_path = os.path.join(cls_dir, sample_file)
                
                try:
                    with h5py.File(file_path, 'r') as f:
                        if data_key in f:
                            data = f[data_key][()]
                            print(f"    Sample file: {sample_file}, Shape: {data.shape}")
                        else:
                            print(f"    Warning: Data key '{data_key}' not found in {sample_file}")
                            print(f"    Available keys: {list(f.keys())}")
                except Exception as e:
                    print(f"    Error reading {sample_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Prepare meta-learning dataset structure')
    parser.add_argument('--data_dir', type=str, default='wifi_benchmark_dataset/tasks/MotionSourceRecognition',
                       help='Root directory of the dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for meta-learning dataset (default: data_dir/meta_data)')
    parser.add_argument('--data_key', type=str, default='CSI_amps',
                       help='Key in h5 file for CSI data')
    parser.add_argument('--verify', action='store_true',
                       help='Verify the meta-learning dataset structure')
    args = parser.parse_args()
    
    # Set output directory if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'meta_data')
    
    # Only verify if requested
    if args.verify:
        print(f"Verifying meta-learning dataset in {args.output_dir}")
        verify_meta_data(args.output_dir, args.data_key)
        return
    
    # Load splits and metadata
    splits = load_splits(args.data_dir)
    metadata = load_metadata(args.data_dir)
    
    # Check if splits were loaded
    if not splits:
        print(f"Error: No splits found in {args.data_dir}/splits")
        return
    
    print(f"Loaded splits: {', '.join(splits.keys())}")
    if metadata is not None:
        print(f"Loaded metadata with {len(metadata)} entries")
        class_labels = get_class_labels(metadata)
        if class_labels:
            print(f"Found {len(class_labels)} classes: {class_labels}")
    
    # Create meta-learning dataset structure
    create_meta_data_structure(args.data_dir, args.output_dir, splits, metadata, args.data_key)
    
    # Verify the created structure
    print("\nVerifying created meta-learning dataset structure:")
    verify_meta_data(args.output_dir, args.data_key)
    
    print(f"\nMeta-learning dataset structure created in {args.output_dir}")
    print("You can now use train_mlp_meta.py to train with this data")

if __name__ == "__main__":
    main() 