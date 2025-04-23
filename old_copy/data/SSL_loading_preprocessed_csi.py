import os
import torch
import numpy as np
import h5py
import scipy.io
import mat73
from torch.utils.data import Dataset


# class SSLCSIDatasetMAT(Dataset):
#     def __init__(self, data_dir):
#         """
#         data_dir: Directory with .mat files, each containing 'CSI_amps'
#                   of shape (N, T, F). T=500 is fixed, but F may differ.
#         """
#         self.file_paths = [
#             os.path.join(data_dir, fname)
#             for fname in os.listdir(data_dir)
#             if fname.endswith('seg_5.mat')
#         ]
#
#         self.sample_index = []  # List of (file_idx, sample_idx)
#         self.feature_sizes = []  # Parallel list storing F for each sample
#
#         # Pre-scan each file to find how many samples (N) and feature dim (F)
#         for f_idx, file_path in enumerate(self.file_paths):
#             # Try opening with h5py
#             try:
#                 with h5py.File(file_path, 'r') as f:
#                     csi_amps_dset = f['CSI_amps']
#                     # h5py shape is (F, T, N) for MATLAB (N, T, F) saved data
#                     shape = csi_amps_dset.shape
#                     num_samples = shape[2]  # N is third dim
#                     feature_dim = shape[0]   # F is first dim
#             except Exception:
#                 # Fallback to scipy or mat73
#                 try:
#                     csi_amps = scipy.io.loadmat(file_path)['CSI_amps']
#                     # scipy shape is (N, T, F)
#                     num_samples = csi_amps.shape[0]
#                     feature_dim = csi_amps.shape[2]
#                 except NotImplementedError:
#                     csi_amps = mat73.loadmat(file_path)['CSI_amps']
#                     num_samples = csi_amps.shape[0]
#                     feature_dim = csi_amps.shape[2]
#
#             # Record (f_idx, s_idx) and feature size for each sample
#             for s_idx in range(num_samples):
#                 self.sample_index.append((f_idx, s_idx))
#                 self.feature_sizes.append(feature_dim)
#
#     def __len__(self):
#         return len(self.sample_index)
#
#     def __getitem__(self, idx):
#         """
#         Return a single sample of shape (T, F).
#         """
#         file_idx, sample_idx = self.sample_index[idx]
#         file_path = self.file_paths[file_idx]
#
#         # Load that single sample
#         try:
#             with h5py.File(file_path, 'r') as f:
#                 # h5py data shape (F, T, N), get sample_idx from third dim
#                 csi_amps = f['CSI_amps'][:, :, sample_idx]  # shape (F, T)
#                 csi_amps = csi_amps.T  # transpose to (T, F)
#         except Exception:
#             try:
#                 # scipy/mat73 data shape (N, T, F), direct index
#                 csi_amps = scipy.io.loadmat(file_path)['CSI_amps'][sample_idx]
#             except NotImplementedError:
#                 csi_amps = mat73.loadmat(file_path)['CSI_amps'][sample_idx]
#
#         sample_tensor = torch.from_numpy(np.array(csi_amps)).float()
#         # sample_tensor: shape (T, F)
#         # print(f"Sample index={idx}, shape={sample_tensor.shape}")
#         return sample_tensor
#
# class SSLCSIDatasetMAT(Dataset):
#     def __init__(self, data_dir):
#         """
#         data_dir: Directory with .mat files
#         Maintains both preloaded data and feature sizes
#         """
#         self.file_paths = [
#             os.path.join(data_dir, fname)
#             for fname in os.listdir(data_dir)
#             if fname.endswith('seg_5.mat')
#         ]
#
#         self.data = []  # Stores preloaded samples as tensors
#         self.feature_sizes = []  # Maintains original F dimension for each sample
#
#         # Load all data while tracking feature dimensions
#         for file_path in self.file_paths:
#             try:  # Try h5py first
#                 with h5py.File(file_path, 'r') as f:
#                     csi_amps = f['CSI_amps'][()]  # (F, T, N)
#                     feature_dim = csi_amps.shape[0]
#                     num_samples = csi_amps.shape[2]
#
#                     # Convert entire array to tensor once
#                     csi_tensor = torch.from_numpy(csi_amps).float()
#
#                     # Process samples
#                     for s_idx in range(num_samples):
#                         sample = csi_tensor[:, :, s_idx].permute(1, 0)  # (T, F)
#                         self.data.append(sample)
#                         self.feature_sizes.append(feature_dim)
#
#             except Exception:  # Fallback to scipy/mat73
#                 try:
#                     csi_amps = scipy.io.loadmat(file_path)['CSI_amps']  # (N, T, F)
#                 except NotImplementedError:
#                     csi_amps = mat73.loadmat(file_path)['CSI_amps']
#
#                 feature_dim = csi_amps.shape[2]
#                 num_samples = csi_amps.shape[0]
#                 csi_tensor = torch.from_numpy(csi_amps).float()
#
#                 for s_idx in range(num_samples):
#                     self.data.append(csi_tensor[s_idx])
#                     self.feature_sizes.append(feature_dim)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]

class SSLCSIDatasetMAT(Dataset):
    def __init__(self, data_dir):
        self.file_paths = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.endswith('seg_5.mat')
        ]

        self.sample_info = []  # (file_path, sample_idx)
        self.feature_sizes = []  # Feature dimension for each sample
        self.cache = {}  # In-memory cache for small files

        # First pass: index all samples
        for file_path in self.file_paths:
            try:  # Try HDF5 format first
                with h5py.File(file_path, 'r') as f:
                    dset = f['CSI_amps']
                    F, T, N = dset.shape
                    self._add_samples(file_path, N, F)

            except:  # Fallback to MATLAB formats
                try:
                    # Try loading with mat73 first for v7.3 files
                    data = mat73.loadmat(file_path)
                    csi_amps = data['CSI_amps']
                except:
                    # Fallback to scipy for older MATLAB formats
                    data = scipy.io.loadmat(file_path)
                    csi_amps = data['CSI_amps']

                N, T, F = csi_amps.shape
                # Cache small files (<1GB) in memory
                if csi_amps.nbytes < 1e9:
                    self.cache[file_path] = torch.from_numpy(csi_amps).float()
                self._add_samples(file_path, N, F)

    def _add_samples(self, file_path, num_samples, feature_dim):
        """Helper to add samples to tracking structures"""
        for s_idx in range(num_samples):
            self.sample_info.append((file_path, s_idx))
            self.feature_sizes.append(feature_dim)

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        file_path, s_idx = self.sample_info[idx]

        # Check in-memory cache first
        if file_path in self.cache:
            return self.cache[file_path][s_idx]

        try:  # Try HDF5 loading
            with h5py.File(file_path, 'r') as f:
                dset = f['CSI_amps']
                sample = dset[:, :, s_idx].T  # Transpose to (T, F)
                return torch.from_numpy(sample).float()
        except:  # Fallback to MATLAB loading
            try:
                data = mat73.loadmat(file_path)
            except:
                data = scipy.io.loadmat(file_path)
            csi_amps = data['CSI_amps']
            return torch.from_numpy(csi_amps[s_idx]).float()

    def __del__(self):
        """Cleanup any open resources"""
        if hasattr(self, "cache"):
            self.cache.clear()