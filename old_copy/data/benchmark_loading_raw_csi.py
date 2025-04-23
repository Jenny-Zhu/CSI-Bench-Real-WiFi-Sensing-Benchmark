import torch
from torch.utils.data import Dataset
import numpy as np
import random
from collections import defaultdict
import os
import scipy.io as sio
import h5py
import random

class BKCSIDatasetMAT(Dataset):
    def __init__(self, folder_path, resize_height=64, resize_width=100, label_keywords=None):
        """
        Dataset class for loading CSI .mat files and generating samples.
        :param folder_path: Directory containing .mat files OR a list of directories
        """
        if isinstance(folder_path, str):
            self.folder_paths = [folder_path]  # make it a list
        elif isinstance(folder_path, list):
            self.folder_paths = folder_path
        else:
            raise TypeError("folder_path should be a string or a list of strings.")

        # Save resize settings
        self.resize_height = resize_height
        self.resize_width = resize_width

        # Save label mapping (like {'good':0, 'bad':1})
        self.label_keywords = label_keywords

        # Load all CSI data and their labels
        self.data, self.labels = self.load_matlab_CSI()
    
    
    def normalize_csi(self, csi):
        """Normalizes CSI (subcarriers, antennas, samples)"""
        chnnorm = np.sqrt(np.sum(np.abs(csi) ** 2, axis=2, keepdims=True))
        csi_normalized = csi / (chnnorm + np.finfo(float).eps)

        # Replace NaNs and Infs with 0
        csi_normalized[np.isnan(csi_normalized)] = 0
        csi_normalized[np.isinf(csi_normalized)] = 0

        return csi_normalized

    def find_mat_files(self):
        # Find all .mat files under the folder(s)
        mat_files = []
        for folder in self.folder_paths:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('.mat'):
                        mat_files.append(os.path.join(root, file))
        return mat_files

    def load_matlab_CSI(self):
        # Load CSI data from .mat files and extract amplitudes
        mat_files = self.find_mat_files()
        data_list, labels = [], []

        for file_path in mat_files:
            # Find which label this file belongs to
            label = None
            for keyword, value in self.label_keywords.items():
                if keyword.lower() in file_path.lower():
                    label = value
                    break
            if label is None:
                continue  # Skip files that don't match any keyword

            # Load .mat file contents
            try:
                mat_data = sio.loadmat(file_path, squeeze_me=False)
            except NotImplementedError:
                # If .mat is v7.3, fall back to h5py
                mat_data = {}
                with h5py.File(file_path, 'r') as f:
                    for key in f.keys():
                        mat_data[key] = np.array(f[key])

            # Extract CSI content
            try:
                csi_content = mat_data['csi_trace']['csi'][0][0]
            except Exception:
                continue # Skip broken or missing files

            # Take amplitude and reshape into 2D array (subcarriers × time or antennas × time)
            csi_content = self.normalize_csi(csi_content)
            csi_amplitude = abs(csi_content)
            csi_reshaped = csi_amplitude.reshape(-1, csi_amplitude.shape[-1])

            # Resize or pad subcarrier dimension (height)
            current_subca = csi_reshaped.shape[0]
            if current_subca > self.resize_height:
                csi_resized = csi_reshaped[:self.resize_height, :]
            else:
                pad_rows = self.resize_height - current_subca
                csi_resized = np.pad(csi_reshaped, ((0, pad_rows), (0, 0)), mode='constant')

            # Resize or pad time dimension (width)
            current_width = csi_reshaped.shape[1]
            if current_width > self.resize_width:
                csi_resized = csi_resized[:, :self.resize_width]
            else:
                pad_cols = self.resize_width - current_width
                csi_resized = np.pad(csi_resized, ((0, 0), (0, pad_cols)), mode='constant')

            data_list.append(csi_resized)
            labels.append(label)

        return np.array(data_list), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class CSITaskDataset:
    def __init__(self, folder_path, k_shot=5, q_query=15, resize_height=64, resize_width=100, label_keywords=None):
        """
        Task sampler for MAML meta-learning from CSI data.
        :param folder_path: Path to .mat data
        :param k_shot: Number of support samples per class
        :param q_query: Number of query samples per class
        :param resize_height: Height for CSI subcarriers
        :param label_keywords: Dictionary like {"good": 0, "bad": 1}
        """
        # Load raw CSI dataset
        self.dataset = BKCSIDatasetMAT(folder_path, resize_height, resize_width, label_keywords)
        self.k_shot = k_shot # Support samples per class
        self.q_query = q_query # Query samples per class

        # Group samples by label
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.dataset.labels):
            self.class_to_indices[int(label)].append(idx)

        # List of available classes
        self.classes = list(self.class_to_indices.keys())

        # Ensure at least two classes exist
        assert len(self.classes) >= 2, "MAML requires at least 2 classes."

    def sample_task(self, num_classes=2, max_retry=5):
        """
        Sample one few-shot learning task.
        Randomly pick classes, and select k support + q query examples for each.
        """
        for attempt in range(max_retry):
            selected_classes = random.sample(self.classes, num_classes)
            support_x, support_y, query_x, query_y = [], [], [], []

            for class_idx, class_label in enumerate(selected_classes):
                indices = self.class_to_indices[class_label]

                # If not enough examples, skip this class
                if len(indices) < self.k_shot + self.q_query:
                    continue

                # Randomly sample examples for support and query sets
                chosen = random.sample(indices, self.k_shot + self.q_query)
                support_ids = chosen[:self.k_shot]
                query_ids = chosen[self.k_shot:]

                # Collect support samples
                for sid in support_ids:
                    x, y = self.dataset[sid]
                    x = torch.tensor(x).unsqueeze(0).float()
                    if torch.all(x == 0):
                        print(f"[Warning] All-zero CSI sample skipped during sampling.")
                        continue 
                    support_x.append(x)
                    support_y.append(class_idx)

                # Collect query samples
                for qid in query_ids:
                    x, y = self.dataset[qid]
                    x = torch.tensor(x).unsqueeze(0).float()
                    if torch.all(x == 0):
                        continue 
                    query_x.append(x)
                    query_y.append(class_idx)

            # Check if valid task
            if len(support_x) > 0 and len(query_x) > 0:
                if len(torch.unique(torch.tensor(query_y))) >= num_classes:
                    return (
                        torch.stack(support_x), torch.tensor(support_y),
                        torch.stack(query_x), torch.tensor(query_y)
                    )

        # 🔥 After max_retry attempts, raise error
        raise RuntimeError(f"Failed to sample a valid task after {max_retry} attempts. Please check your dataset!")

class MultiSourceTaskDataset:
    def __init__(self, datasets):
        """
        Wrapper to sample few-shot tasks across multiple CSITaskDataset sources.
        :param datasets: list of CSITaskDataset instances
        """
        self.datasets = datasets

    def sample_task(self):
        """
        Randomly sample a task from one of the datasets.
        """
        dataset = random.choice(self.datasets)
        return dataset.sample_task()
