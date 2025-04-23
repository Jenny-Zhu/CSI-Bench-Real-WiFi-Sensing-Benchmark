import torch
from torch.utils.data import Dataset
import numpy as np
import random
from collections import defaultdict
import os
import scipy.io as sio
import h5py
from data.datasets.base_dataset import BaseDataset
from data.preprocessing.csi_preprocessing import normalize_csi

class BKCSIDatasetMAT(BaseDataset):
    """Dataset for loading CSI data from .mat files for meta-learning."""
    
    def __init__(self, data_dir, resize_height=64, resize_width=100, label_keywords=None, transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: The directory containing the data.
            resize_height: The height to resize the CSI data to.
            resize_width: The width to resize the CSI data to.
            label_keywords: A dictionary mapping keywords to labels.
            transform: The transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.label_keywords = label_keywords
        
        self.data, self.labels = self.load_matlab_CSI()
    
    def find_mat_files(self):
        """Find all .mat files in the data directory.
        
        Returns:
            A list of paths to .mat files.
        """
        mat_files = []
        for folder in self.data_dir:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('.mat'):
                        mat_files.append(os.path.join(root, file))
        return mat_files
    
    def load_matlab_CSI(self):
        """Load CSI data from .mat files.
        
        Returns:
            A tuple of (data, labels).
        """
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
                continue  # Skip broken or missing files
            
            # Take amplitude and reshape into 2D array (subcarriers × time or antennas × time)
            csi_content = normalize_csi(csi_content)
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

class CSITaskDataset:
    """Dataset for sampling tasks for meta-learning."""
    
    def __init__(self, folder_path, k_shot=5, q_query=15, resize_height=64, resize_width=100, label_keywords=None):
        """Initialize the dataset.
        
        Args:
            folder_path: The directory containing the data.
            k_shot: The number of support samples per class.
            q_query: The number of query samples per class.
            resize_height: The height to resize the CSI data to.
            resize_width: The width to resize the CSI data to.
            label_keywords: A dictionary mapping keywords to labels.
        """
        # Load raw CSI dataset
        self.dataset = BKCSIDatasetMAT(folder_path, resize_height, resize_width, label_keywords)
        self.k_shot = k_shot
        self.q_query = q_query
        
        # Group samples by label
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.dataset.labels):
            self.class_to_indices[int(label)].append(idx)
        
        # List of available classes
        self.classes = list(self.class_to_indices.keys())
        
        # Ensure at least two classes exist
        assert len(self.classes) >= 2, "MAML requires at least 2 classes."
    
    def sample_task(self, num_classes=2, max_retry=5):
        """Sample a task for meta-learning.
        
        Args:
            num_classes: The number of classes to sample.
            max_retry: The maximum number of retries.
            
        Returns:
            A tuple of (support_x, support_y, query_x, query_y).
            
        Raises:
            RuntimeError: If a valid task cannot be sampled after max_retry attempts.
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
        
        # After max_retry attempts, raise error
        raise RuntimeError(f"Failed to sample a valid task after {max_retry} attempts. Please check your dataset!")

class MultiSourceTaskDataset:
    """Dataset for sampling tasks from multiple sources."""
    
    def __init__(self, datasets):
        """Initialize the dataset.
        
        Args:
            datasets: A list of CSITaskDataset instances.
        """
        self.datasets = datasets
    
    def sample_task(self):
        """Sample a task from one of the datasets.
        
        Returns:
            A tuple of (support_x, support_y, query_x, query_y).
        """
        dataset = random.choice(self.datasets)
        return dataset.sample_task()
