import torch
import torch.nn as nn
import torch.nn.functional as F

class CSI2DCNN(nn.Module):
    def __init__(self, input_size=(64, 1000), num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)  # H/2, W/2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)  # H/4, W/4

        # Compute flattened size after conv + pool
        h, w = input_size[0] // 4, input_size[1] // 4
        self.fc1 = nn.Linear(32 * h * w, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):  # x: (B, 1, H, W)
        x = F.relu(self.pool1(self.conv1(x)))  # → (B, 16, H/2, W/2)
        x = F.relu(self.pool2(self.conv2(x)))  # → (B, 32, H/4, W/4)
        x = x.view(x.size(0), -1)              # Flatten
        x = F.relu(self.fc1(x))                # FC layer
        x = self.fc2(x)                        # Output logits
        return x
