import torch
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict
import random

class FeatureBucketBatchSampler(Sampler):
    """
    Batch sampler that groups samples with similar feature dimensions 
    to minimize padding and improve efficiency.
    
    Args:
        data_source: Dataset to sample from
        batch_size: Size of mini-batch
        drop_last: If True, drop the last incomplete batch
        sort_key: Function that returns a sort key for each sample
        bucket_size_multiplier: Bucket size = batch_size * bucket_size_multiplier
    """
    def __init__(self, data_source, batch_size, drop_last=False, 
                 sort_key=lambda x: x, bucket_size_multiplier=100):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sort_key = sort_key
        self.bucket_size_multiplier = bucket_size_multiplier
        
        # Calculate lengths once for all samples
        self.lengths = [self.sort_key(data_source[i]) for i in range(len(data_source))]
        
    def __iter__(self):
        # Create buckets and assign samples
        indices = list(range(len(self.data_source)))
        
        # Shuffle indices
        random.shuffle(indices)
        
        # Define bucket size
        bucket_size = min(self.batch_size * self.bucket_size_multiplier, len(self.data_source))
        
        # Group indices into buckets and sort by length within buckets
        for i in range(0, len(indices), bucket_size):
            bucket = indices[i:i + bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx])
            
            # Create batches from the sorted bucket
            for j in range(0, len(bucket), self.batch_size):
                batch = bucket[j:j + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

class SimilarLengthBatchSampler(Sampler):
    """
    Batch sampler that groups samples with similar lengths to minimize padding.
    
    This version uses clustering to group samples with similar lengths.
    
    Args:
        data_source: Dataset to sample from
        batch_size: Size of mini-batch
        drop_last: If True, drop the last incomplete batch
        get_length: Function that returns length for each sample
        num_buckets: Number of buckets to create
    """
    def __init__(self, data_source, batch_size, drop_last=False, 
                 get_length=lambda x: len(x), num_buckets=10):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.get_length = get_length
        self.num_buckets = num_buckets
        
        # Get lengths of all samples
        self.lengths = np.array([self.get_length(data_source[i]) for i in range(len(data_source))])
        
        # Create buckets
        self.buckets = self._create_buckets()
        
    def _create_buckets(self):
        # Find min and max lengths
        min_len, max_len = self.lengths.min(), self.lengths.max()
        
        # Create bucket boundaries
        boundaries = np.linspace(min_len, max_len, self.num_buckets + 1)
        
        # Assign samples to buckets
        buckets = defaultdict(list)
        for idx, length in enumerate(self.lengths):
            # Find the bucket this sample belongs to
            bucket_idx = np.digitize(length, boundaries) - 1
            bucket_idx = min(bucket_idx, self.num_buckets - 1)  # Handle edge case
            buckets[bucket_idx].append(idx)
            
        return buckets
    
    def __iter__(self):
        # Shuffle samples within each bucket
        for bucket_idx in self.buckets:
            random.shuffle(self.buckets[bucket_idx])
        
        # Create a list of buckets and shuffle their order
        bucket_indices = list(self.buckets.keys())
        random.shuffle(bucket_indices)
        
        # Yield batches
        for bucket_idx in bucket_indices:
            bucket = self.buckets[bucket_idx]
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
    
    def __len__(self):
        if self.drop_last:
            return sum(len(bucket) // self.batch_size for bucket in self.buckets.values())
        else:
            return sum((len(bucket) + self.batch_size - 1) // self.batch_size 
                       for bucket in self.buckets.values())
