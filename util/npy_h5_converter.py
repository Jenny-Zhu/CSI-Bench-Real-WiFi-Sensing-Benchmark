import numpy as np
import h5py
import os
import glob

def convert_npy_to_hdf5(source_folder, target_folder):
    # Ensure the target directory exists
    os.makedirs(target_folder, exist_ok=True)

    # List all .npy files in the source directory
    npy_files = glob.glob(os.path.join(source_folder, '*.npy'))

    for npy_file in npy_files:
        # Load data with allow_pickle=True to handle object arrays
        data = np.load(npy_file, allow_pickle=True)

        # Assuming data consists of arrays; convert to uniform dtype
        if data.dtype == object:
            # Attempt to convert all subarrays to a uniform dtype and combine them into a single array
            try:
                # Assume subarrays are uniformly shaped and can be directly converted to a specific dtype
                uniform_data = np.array([np.array(item, dtype=np.float32) for item in data])
            except Exception as e:
                print(f"Error processing {npy_file}: {e}")
                continue  # Skip to the next file if conversion fails


        # Define the new file path with .h5 extension
        base_name = os.path.basename(npy_file)
        new_file_path = os.path.join(target_folder, base_name.replace('.npy', '.h5'))

        # Save to HDF5
        with h5py.File(new_file_path, 'w') as hf:
            hf.create_dataset('data', data=uniform_data)
        print(f"Converted {npy_file} to {new_file_path}")

# Example usage
data_dir_list_npy = ["E:\\Dataset\\Dataset_OW\\HM3_ESP32\\SSL\\intermediate_test\\ESP32_new"]
sample_rate = 100
win_len = 500
for data_dir in data_dir_list_npy:
    source_folder = data_dir
    target_folder = data_dir
    convert_npy_to_hdf5(source_folder, target_folder)
