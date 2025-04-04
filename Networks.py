import torch
import torch.nn as nn
import torch.nn.functional as F

class CSINet(nn.Module):
    def __init__(self, input_len=100):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.fc = nn.Linear(32 * input_len, 2)  # binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
