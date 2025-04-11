import torch
import os
from torch.utils.data import Dataset
import numpy as np
import mat73
import matplotlib.pyplot as plt
from .SSL_dataloading import rescale_csi

file_path = "E:\Dataset\Dataset_OW\DatasetHP\HM3_HP\DanHome\Data\Fan\loc1\Fan\csi_20220305002915_all.mat"  # List your .mat files here
data = mat73.loadmat(file_path)
csi = data['csi_trace']['csi']

csi_data = data['csi_trace']['csi'][0][0]
_,_,resampled_data = rescale_csi(csi_data, set_sampling_rate=1500)
# Example: Plot the original and resampled signal of a specific tx, rx, subcarrier
plt.figure()
plt.plot(csi_data[0, 0, 0], label='Original')
plt.plot(resampled_data[0, 0, 0], label='Resampled', linestyle='--')
plt.legend()
plt.title('Comparison of Original and Resampled Data')
plt.xlabel('Time Index')
plt.ylabel('Signal Amplitude')
plt.show()

print(csi.shape[3])
# Calculate amplitude and phase
amplitude = np.abs(csi)
phase = np.angle(csi)

# Stack the amplitude and phase into two channels
csi_transformed = np.stack((amplitude, phase), axis=0)  # Adjust axis if needed based on your data dimensions 

print(csi_transformed.shape)
print(csi_transformed[...,1:3])
# csi_val = csi[0,0,0]
# print(csi_val)

