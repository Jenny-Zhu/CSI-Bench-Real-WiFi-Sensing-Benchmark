import torch
from torch.utils.data import Dataset
import random

class DummyCSITaskDataset:
    def __init__(self, input_len=100, k_shot=5, q_query=15):
        self.input_len = input_len
        self.k_shot = k_shot
        self.q_query = q_query

    def generate_sample(self, label):
        signal = torch.randn(self.input_len) + (5 if label == 1 else 0)
        return signal.unsqueeze(0), label

    def sample_task(self):
        support_set, query_set = [], []
        for label in [0, 1]:
            support_set += [self.generate_sample(label) for _ in range(self.k_shot)]
            query_set  += [self.generate_sample(label) for _ in range(self.q_query)]

        x_s, y_s = zip(*support_set)
        x_q, y_q = zip(*query_set)
        return torch.stack(x_s), torch.tensor(y_s), torch.stack(x_q), torch.tensor(y_q)
