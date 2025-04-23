import numpy as np
import h5py
import os
import glob

def convert_npy_to_h5(npy_file, h5_file=None, dataset_name='csi', compression='gzip'):
    """
    Convert NPY file to H5 format.
    
    Args:
        npy_file (str): Path to NPY file
        h5_file (str, optional): Path to output H5 file. If None, uses same name as NPY with .h5 extension
        dataset_name (str): Name of the dataset inside the H5 file
        compression (str): Compression method for H5 file
        
    Returns:
        str: Path to the created H5 file
    """
    # Load NPY data
    data = np.load(npy_file)
    
    # Determine output file name if not provided
    if h5_file is None:
        h5_file = os.path.splitext(npy_file)[0] + '.h5'
    
    # Create H5 file and save data
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset(dataset_name, data=data, compression=compression)
    
    print(f"Converted {npy_file} to {h5_file}")
    return h5_file

def convert_h5_to_npy(h5_file, npy_file=None, dataset_name='csi'):
    """
    Convert H5 file to NPY format.
    
    Args:
        h5_file (str): Path to H5 file
        npy_file (str, optional): Path to output NPY file. If None, uses same name as H5 with .npy extension
        dataset_name (str): Name of the dataset inside the H5 file
        
    Returns:
        str: Path to the created NPY file
    """
    # Determine output file name if not provided
    if npy_file is None:
        npy_file = os.path.splitext(h5_file)[0] + '.npy'
    
    # Load H5 data and save as NPY
    with h5py.File(h5_file, 'r') as f:
        if dataset_name not in f:
            available = list(f.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found in {h5_file}. Available: {available}")
        
        data = f[dataset_name][:]
        np.save(npy_file, data)
    
    print(f"Converted {h5_file} to {npy_file}")
    return npy_file

def batch_convert_npy_to_h5(directory, pattern='*.npy', dataset_name='csi', compression='gzip'):
    """
    Convert multiple NPY files to H5 format.
    
    Args:
        directory (str): Directory containing NPY files
        pattern (str): Glob pattern to match NPY files
        dataset_name (str): Name of the dataset inside the H5 files
        compression (str): Compression method for H5 files
        
    Returns:
        list: Paths to created H5 files
    """
    # Find all matching NPY files
    npy_files = glob.glob(os.path.join(directory, pattern))
    
    h5_files = []
    for npy_file in npy_files:
        h5_file = convert_npy_to_h5(npy_file, dataset_name=dataset_name, compression=compression)
        h5_files.append(h5_file)
    
    return h5_files

def batch_convert_h5_to_npy(directory, pattern='*.h5', dataset_name='csi'):
    """
    Convert multiple H5 files to NPY format.
    
    Args:
        directory (str): Directory containing H5 files
        pattern (str): Glob pattern to match H5 files
        dataset_name (str): Name of the dataset inside the H5 files
        
    Returns:
        list: Paths to created NPY files
    """
    # Find all matching H5 files
    h5_files = glob.glob(os.path.join(directory, pattern))
    
    npy_files = []
    for h5_file in h5_files:
        npy_file = convert_h5_to_npy(h5_file, dataset_name=dataset_name)
        npy_files.append(npy_file)
    
    return npy_files
