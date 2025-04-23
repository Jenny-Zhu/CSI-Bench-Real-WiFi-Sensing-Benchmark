import os
import json
import glob
import h5py
import numpy as np

def create_metadata(data_dir, patterns=['*.h5', '*.npy'], output_file=None):
    """
    Create metadata for all data files in the specified directory.
    
    Args:
        data_dir (str): Directory containing data files
        patterns (list): List of glob patterns to match files
        output_file (str, optional): Path to output JSON file
        
    Returns:
        dict: Metadata dictionary
    """
    metadata = {
        'directory': data_dir,
        'files': []
    }
    
    # Find all matching files
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(data_dir, pattern)))
    
    # Process each file
    for file_path in all_files:
        file_metadata = _extract_file_metadata(file_path)
        if file_metadata:
            metadata['files'].append(file_metadata)
    
    # Save metadata if output file is specified
    if output_file:
        save_metadata(metadata, output_file)
    
    return metadata

def _extract_file_metadata(file_path):
    """
    Extract metadata from a single file.
    
    Args:
        file_path (str): Path to data file
        
    Returns:
        dict: File metadata or None if extraction fails
    """
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1]
    
    try:
        if file_ext == '.h5':
            return _extract_h5_metadata(file_path, file_name)
        elif file_ext == '.npy':
            return _extract_npy_metadata(file_path, file_name)
        else:
            print(f"Unsupported file extension: {file_ext}")
            return None
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        return None

def _extract_h5_metadata(file_path, file_name):
    """Extract metadata from H5 file."""
    with h5py.File(file_path, 'r') as f:
        datasets = list(f.keys())
        
        # Determine main dataset
        main_dataset = datasets[0] if datasets else None
        
        # Get shape and type if main dataset exists
        if main_dataset:
            shape = f[main_dataset].shape
            dtype = str(f[main_dataset].dtype)
            
            return {
                'file_name': file_name,
                'file_path': file_path,
                'file_type': 'h5',
                'datasets': datasets,
                'main_dataset': main_dataset,
                'shape': shape,
                'dtype': dtype,
                'size_bytes': os.path.getsize(file_path)
            }
    
    return None

def _extract_npy_metadata(file_path, file_name):
    """Extract metadata from NPY file."""
    data = np.load(file_path, mmap_mode='r')  # Memory-mapped for large files
    
    return {
        'file_name': file_name,
        'file_path': file_path,
        'file_type': 'npy',
        'shape': data.shape,
        'dtype': str(data.dtype),
        'size_bytes': os.path.getsize(file_path)
    }

def save_metadata(metadata, output_file):
    """
    Save metadata to JSON file.
    
    Args:
        metadata (dict): Metadata dictionary
        output_file (str): Path to output JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Convert NumPy types to native Python types for JSON serialization
    def json_serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return obj
    
    # Save as JSON
    with open(output_file, 'w') as f:
        json.dump(metadata, f, default=json_serialize, indent=2)
    
    print(f"Metadata saved to {output_file}")
