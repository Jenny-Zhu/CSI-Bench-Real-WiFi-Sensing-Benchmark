import random
from torch.utils.data import Sampler


class FeatureBucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        dataset: an instance of SSLCSIDatasetMAT
        batch_size: desired batch size
        shuffle: whether to shuffle samples
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 1) Build the dictionary: feature_dim -> list of sample indices
        self.feature_buckets = {}
        for idx, fdim in enumerate(dataset.feature_sizes):
            if fdim not in self.feature_buckets:
                self.feature_buckets[fdim] = []
            self.feature_buckets[fdim].append(idx)

        # 2) For each feature dimension, shuffle the list of indices if needed
        if self.shuffle:
            for fdim in self.feature_buckets:
                random.shuffle(self.feature_buckets[fdim])

        # 3) Build a list of (feature_dim) keys in random order for each epoch
        self.bucket_keys = list(self.feature_buckets.keys())
        if self.shuffle:
            random.shuffle(self.bucket_keys)

        # We'll build the batch index lists in self._batches
        self._batches = self._create_batches()

    def _create_batches(self):
        """
        Create a list of batches, where each batch is a list of dataset indices.
        We'll go bucket by bucket.
        """
        batches = []
        for fdim in self.bucket_keys:
            indices_in_bucket = self.feature_buckets[fdim]
            # chunk into batch_size
            for i in range(0, len(indices_in_bucket), self.batch_size):
                batch_indices = indices_in_bucket[i: i + self.batch_size]
                batches.append(batch_indices)
        return batches

    def __iter__(self):
        """
        Each epoch, we want a fresh randomization (if self.shuffle=True).
        We can either rebuild _batches or rely on a fixed shuffle per epoch.
        For simplicity, let's re-shuffle each epoch in this function.
        """
        if self.shuffle:
            # re-shuffle the bucket keys
            random.shuffle(self.bucket_keys)
            # also re-shuffle the indices within each bucket
            for fdim in self.bucket_keys:
                random.shuffle(self.feature_buckets[fdim])
            # re-build the batches
            self._batches = []
            for fdim in self.bucket_keys:
                indices_in_bucket = self.feature_buckets[fdim]
                for i in range(0, len(indices_in_bucket), self.batch_size):
                    batch_indices = indices_in_bucket[i: i + self.batch_size]
                    self._batches.append(batch_indices)

        # Finally yield each batch
        for batch_indices in self._batches:
            yield batch_indices

    def __len__(self):
        """
        Total number of batches across all buckets.
        """
        return len(self._batches)