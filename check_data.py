import os
import torch
from load.supervised.benchmark_loader import load_benchmark_supervised

# Load data from the dataset
data = load_benchmark_supervised(
    dataset_root='wifi_benchmark_dataset/tasks',
    task_name='MotionSourceRecognition',
    batch_size=1,
    file_format="h5",
    data_key="CSI_amps",
    num_workers=0
)

# Get a single data sample
train_loader = data['loaders']['train']
for inputs, labels in train_loader:
    print(f"Input shape: {inputs.shape}")
    print(f"Label: {labels}")
    break

# Print the model parameters needed
feature_size = inputs.shape[-1]
win_len = inputs.shape[-2]
print(f"\nRecommended parameters for LSTM model:")
print(f"--feature_size {feature_size}")
print(f"--win_len {win_len}") 