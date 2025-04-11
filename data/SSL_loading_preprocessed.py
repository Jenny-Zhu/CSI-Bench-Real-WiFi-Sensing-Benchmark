import numpy as np
import os
import glob
from torch.utils.data import Dataset
import h5py

#
# class SSLCSIDatasetNPY(Dataset):
#     def __init__(self, data_dir_list, win_len, sample_rate):
#         self.samples = []
#         for data_dir in data_dir_list:
#             # Load .npy files and populate the samples list
#             npy_files = glob.glob(os.path.join(data_dir, f'*sr{sample_rate}_wl{win_len}*.npy'))
#             for npy_file in npy_files:
#                 file_size = os.path.getsize(npy_file)
#                 if file_size == 0:
#                     print(f"Skipping empty file: {npy_file}")
#                     continue
#                 try:
#                     data = np.load(npy_file, allow_pickle=True)
#                     for sample in data:
#                         self.samples.append(sample)  # Append each sample to the samples list
#                 except Exception as e:
#                     print(f"Failed to load {npy_file} (Size: {file_size} bytes): {e}")
#
#             print(f"Loaded {len(self.samples)} samples from {len(npy_files)} files from {data_dir}.")
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         # Customize this method based on how you want to access your data
#         return self.samples[idx]


class SSLCSIDatasetNPY(Dataset):
    def __init__(self, data_dir_list, win_len, sample_rate):
        self.sample_info = []  # List to hold the file path and index of samples within the file
        for data_dir in data_dir_list:
            npy_files = glob.glob(os.path.join(data_dir, f'*sr{sample_rate}_wl{win_len}*.npy'))
            for file in npy_files:
                data_length = len(np.load(file, mmap_mode='r'))  # Load with memory mapping to get the length without loading data into RAM
                for index in range(data_length):
                    self.sample_info.append((file, index))

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        file_path, index = self.sample_info[idx]
        data = np.load(file_path, mmap_mode='r')[index]  # Load specific sample
        return data


class SSLCSIDatasetHDF5(Dataset):
    def __init__(self, data_dir_list, win_len, sample_rate):
        self.file_paths = []
        for data_dir in data_dir_list:
            self.file_paths = glob.glob(os.path.join(data_dir, f'*sr{sample_rate}_wl{win_len}*.h5'))

    def __len__(self):
        # Length can be dynamically calculated if not stored
        total_length = 0
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as f:
                total_length += f['data'].shape[0]
        return total_length

    def __getitem__(self, idx):
        for file_path in self.file_paths:
            # if os.path.exists(file_path):
            #     print("File exists.")
            # else:
            #     print("File does not exist. Check the path.")
            # try:
            #     with h5py.File(file_path, 'r') as file:
            #         print("Successfully opened the file.")
            #         # Optionally, list the datasets in the file
            #         print("Datasets in the file:", list(file.keys()))
            # except OSError as e:
            #     print("Failed to read the file:", e)
            with h5py.File(file_path, 'r') as f:
                data = f['data']
                if idx < len(data):
                    return data[idx]
                idx -= len(data)
        raise IndexError("Index out of range")