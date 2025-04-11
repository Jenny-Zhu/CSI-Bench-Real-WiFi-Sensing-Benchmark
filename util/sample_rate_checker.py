import numpy as np
from scipy.signal import resample
def rescale_csi(data,set_sampling_rate):
    csi_data = data['csi_trace']['csi'][0]
    mactimer = data['csi_trace']['mactimer'][0, 0].flatten()
    # Calculate actual sampling rate
    time_diff = np.diff(mactimer) / 1e6  # Convert microseconds to seconds
    actual_sampling_rate = 1 / np.mean(time_diff)

    # Check if the actual sampling rate calculation is reasonable
    if actual_sampling_rate < 1:
        actual_sampling_rate = 1 / np.mean(time_diff)

    # Calculate the sampling rate ratio and factor
    sampling_rate_ratio = set_sampling_rate / actual_sampling_rate

    # and it has dimensions [tx, rx, subcarriers, time_samples]
    tx, rx, subcarriers, time_samples = csi_data.shape

    # Calculate the new number of time samples
    new_time_samples = int(time_samples * sampling_rate_ratio)

    # Initialize an array to hold the resampled data
    resampled_data = np.zeros((tx, rx, subcarriers, new_time_samples))

    # Resample each combination of tx, rx, subcarriers
    for i in range(tx):
        for j in range(rx):
            for k in range(subcarriers):
                resampled_data[i, j, k] = resample(csi_data[i, j, k], new_time_samples)

    return actual_sampling_rate, sampling_rate_ratio,resampled_data


###Test and visualize

import matplotlib.pyplot as plt

csi_trace = load_mat_file(os.path.join(folder_name, file_path))
csi_data = data['csi_trace']['csi'][0]
_,_,resampled_data = rescale_csi(data)
# Example: Plot the original and resampled signal of a specific tx, rx, subcarrier
plt.figure()
plt.plot(csi_data[0, 0, 0], label='Original')
plt.plot(resampled_data[0, 0, 0], label='Resampled', linestyle='--')
plt.legend()
plt.title('Comparison of Original and Resampled Data')
plt.xlabel('Time Index')
plt.ylabel('Signal Amplitude')
plt.show()