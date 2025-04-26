import os
import torch
import torch.utils.data
from data.datasets.csi.meta_learning import CSITaskDataset, MultiSourceTaskDataset

from torch.utils.data import DataLoader
from .meta_dataset import MetaTaskDataset

def load_csi_data_benchmark(
    data_dir,
    n_way=3,
    k_shot=5,
    q_query=5,
    batch_size=4,
    file_ext='.npy',
    num_workers=4,
    **kwargs
):
    """
    Returns a dict of DataLoaders for meta-learning (train/val/test if available).
    """
    loaders = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            dataset = MetaTaskDataset(
                split_dir,
                n_way=n_way,
                k_shot=k_shot,
                q_query=q_query,
                file_ext=file_ext
            )
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
    return loaders
