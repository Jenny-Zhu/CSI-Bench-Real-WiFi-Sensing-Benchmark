import torch
from torch.utils.data import Dataset
import numpy as np
import random
from collections import defaultdict
import os
import scipy.io as sio
import h5py
import random

class CSIDataset(Dataset):
    def __init__(self, folder_path, resize_height=64, resize_width = 100, label_keywords=None):
        """
        Dataset class for loading CSI .mat files and generating samples.
        :param folder_path: Directory containing .mat files
        :param resize_height: Resize subcarriers to this fixed height
        :param label_keywords: Dictionary mapping keyword in file path to label, e.g., {"good": 0, "bad": 1}
        """
        self.folder_path = folder_path
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.label_keywords = label_keywords
        self.data, self.labels = self.load_matlab_CSI()

    def find_mat_files(self):
        mat_files = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
        return mat_files

    def load_matlab_CSI(self):
        mat_files = self.find_mat_files()
        data_list, labels = [], []

        for file_path in mat_files:
            label = None
            for keyword, value in self.label_keywords.items():
                if keyword.lower() in file_path.lower():
                    label = value
                    break
            if label is None:
                continue  # Skip files that don't match any keyword

            try:
                mat_data = sio.loadmat(file_path, squeeze_me=False)
            except NotImplementedError:
                mat_data = {}
                with h5py.File(file_path, 'r') as f:
                    for key in f.keys():
                        mat_data[key] = np.array(f[key])

            try:
                csi_content = mat_data['csi_trace']['csi'][0][0]
            except Exception:
                continue

            csi_amplitude = abs(csi_content)
            csi_reshaped = csi_amplitude.reshape(-1, csi_amplitude.shape[-1])

            # Pad or crop
            current_subca = csi_reshaped.shape[0]
            if current_subca > self.resize_height:
                csi_resized = csi_reshaped[:self.resize_height, :]
            else:
                pad_rows = self.resize_height - current_subca
                csi_resized = np.pad(csi_reshaped, ((0, pad_rows), (0, 0)), mode='constant')

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
    def __init__(self, folder_path, k_shot=5, q_query=15, resize_height=64, resize_width = 100, label_keywords=None):
        """
        Task sampler for MAML meta-learning from CSI data.
        :param folder_path: Path to .mat data
        :param k_shot: Number of support samples per class
        :param q_query: Number of query samples per class
        :param resize_height: Height for CSI subcarriers
        :param label_keywords: Dictionary like {"good": 0, "bad": 1}
        """
        self.dataset = CSIDataset(folder_path, resize_height, resize_width, label_keywords)
        self.k_shot = k_shot
        self.q_query = q_query

        # Group samples by label
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.dataset.labels):
            self.class_to_indices[int(label)].append(idx)

        self.classes = list(self.class_to_indices.keys())
        assert len(self.classes) >= 2, "MAML requires at least 2 classes."

    def run_task(self, num_classes=2):
        """
        Sample one few-shot task: support and query sets from selected classes.
        :return: x_s, y_s, x_q, y_q (support and query tensors)
        """
        selected_classes = random.sample(self.classes, num_classes)
        support_x, support_y, query_x, query_y = [], [], [], []

        for class_idx, class_label in enumerate(selected_classes):
            indices = self.class_to_indices[class_label]
            if len(indices) < self.k_shot + self.q_query:
                continue

            chosen = random.sample(indices, self.k_shot + self.q_query)
            support_ids = chosen[:self.k_shot]
            query_ids = chosen[self.k_shot:]

            for sid in support_ids:
                x, y = self.dataset[sid]
                x = torch.tensor(x).unsqueeze(0).float()  # (1, H, W)
                support_x.append(x)
                support_y.append(class_idx)

            for qid in query_ids:
                x, y = self.dataset[qid]
                x = torch.tensor(x).unsqueeze(0).float()
                query_x.append(x)
                query_y.append(class_idx)

        return (
            torch.stack(support_x), torch.tensor(support_y),
            torch.stack(query_x), torch.tensor(query_y)
        )


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
