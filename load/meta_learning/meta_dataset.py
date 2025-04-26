import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MetaTaskDataset(Dataset):
    """
    Dataset for meta-learning: samples N-way K-shot tasks from a directory structure.
    Assumes data_dir/class_x/*.npy or *.pt or *.csv, etc.
    """
    def __init__(self, data_dir, n_way=3, k_shot=5, q_query=5, transform=None, file_ext='.npy'):
        super().__init__()
        self.data_dir = data_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.transform = transform
        self.file_ext = file_ext

        # Build class-to-files mapping
        self.class_to_files = {}
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(file_ext)]
                if len(files) >= k_shot + q_query:
                    self.class_to_files[class_name] = files
        self.classes = list(self.class_to_files.keys())

    def __len__(self):
        # Number of possible tasks is huge; just return a large number
        return 100000

    def __getitem__(self, idx):
        # Sample N classes
        selected_classes = random.sample(self.classes, self.n_way)
        support_x, support_y, query_x, query_y = [], [], [], []

        for label, class_name in enumerate(selected_classes):
            files = random.sample(self.class_to_files[class_name], self.k_shot + self.q_query)
            support_files = files[:self.k_shot]
            query_files = files[self.k_shot:]

            for f in support_files:
                x = self._load_file(f)
                if self.transform:
                    x = self.transform(x)
                support_x.append(x)
                support_y.append(label)
            for f in query_files:
                x = self._load_file(f)
                if self.transform:
                    x = self.transform(x)
                query_x.append(x)
                query_y.append(label)

        # Convert to tensors
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y, dtype=torch.long)

        return {
            'support': (support_x, support_y),
            'query': (query_x, query_y)
        }

    def _load_file(self, filepath):
        if filepath.endswith('.npy'):
            return torch.from_numpy(np.load(filepath)).float()
        elif filepath.endswith('.pt'):
            return torch.load(filepath).float()
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
