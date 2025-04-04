import torch
from torch.utils.data import Dataset
import random
import numpy as np
from collections import defaultdict
import os
import scipy.io as sio
import h5py

# Base CSI Dataset for loading and preprocessing .mat files
class CSIDataset(Dataset):
    def __init__(self, folder_path, resize_height=64):
        """
        :param folder_path: Directory containing .mat files
        :param resize_height: The target height (e.g., number of subcarriers) to resize/pad CSI data to
        """
        self.folder_path = folder_path
        self.resize_height = resize_height
        self.data, self.labels = self.load_matlab_CSI()

    def find_mat_files(self):
        """Recursively find all .mat files in the folder."""
        mat_files = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
        return mat_files

    def load_matlab_CSI(self):
        """Load and preprocess all .mat files in the folder."""
        mat_files = self.find_mat_files()

        data_list = []
        labels = []

        for file_path in mat_files:
            try:
                mat_data = sio.loadmat(file_path, squeeze_me=False)
            except NotImplementedError:
                mat_data = {}
                with h5py.File(file_path, 'r') as f:
                    for key in f.keys():
                        mat_data[key] = np.array(f[key])

            csi_content = mat_data['csi_trace']['csi'][0][0]  # shape: (subcarriers, antennas, samples)
            label = mat_data['label'][0]                      # assuming label is stored like [1] or [0]

            # Compute amplitude and reshape
            csi_amplitude = abs(csi_content)
            csi_reshaped = csi_amplitude.reshape(-1, csi_amplitude.shape[-1])  # shape: (subcarriers * antennas, time)

            # Resize/pad CSI to fixed height
            current_subca = csi_reshaped.shape[0]
            if current_subca > self.resize_height:
                csi_resized = csi_reshaped[:self.resize_height, :]
            else:
                pad_rows = self.resize_height - current_subca
                csi_resized = np.pad(csi_reshaped, ((0, pad_rows), (0, 0)), mode='constant')

            data_list.append(csi_resized)
            labels.append(label)

        return np.array(data_list), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :return: tuple of (sample, label)
            - sample: np.array of shape (H, W) = (resize_height, time)
            - label: int class label
        """
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

# Meta-learning Task Sampler for few-shot learning (MAML)
class CSITaskDataset:
    def __init__(self, folder_path, k_shot=5, q_query=15, resize_height=64):
        """
        :param folder_path: Path to folder with .mat files
        :param k_shot: Number of support samples per class
        :param q_query: Number of query samples per class
        :param resize_height: Height of CSI input for standardization
        """
        self.dataset = CSIDataset(folder_path, resize_height)
        self.k_shot = k_shot
        self.q_query = q_query

        # Group dataset indices by class label
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.dataset.labels):
            self.class_to_indices[int(label)].append(idx)

        self.classes = list(self.class_to_indices.keys())
        assert len(self.classes) >= 2, "MAML requires at least 2 classes."

    def sample_task(self, num_classes=2):
        """
        Sample a few-shot task with support and query sets.
        :return: x_s, y_s, x_q, y_q tensors
        """
        selected_classes = random.sample(self.classes, num_classes)
        support_x, support_y, query_x, query_y = [], [], [], []

        for class_idx, class_label in enumerate(selected_classes):
            indices = self.class_to_indices[class_label]
            if len(indices) < self.k_shot + self.q_query:
                continue  # Not enough data for this class

            chosen = random.sample(indices, self.k_shot + self.q_query)
            support_ids = chosen[:self.k_shot]
            query_ids = chosen[self.k_shot:]

            for sid in support_ids:
                x, y = self.dataset[sid]
                x = torch.tensor(x).unsqueeze(0).float()  # shape: (1, H, W)
                support_x.append(x)
                support_y.append(class_idx)

            for qid in query_ids:
                x, y = self.dataset[qid]
                x = torch.tensor(x).unsqueeze(0).float()
                query_x.append(x)
                query_y.append(class_idx)

        return (
            torch.cat(support_x), torch.tensor(support_y),
            torch.cat(query_x), torch.tensor(query_y)
        )
