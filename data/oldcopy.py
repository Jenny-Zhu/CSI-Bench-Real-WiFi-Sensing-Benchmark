import scipy
import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import mat73
from scipy.signal import resample


def load_mat_file(file_path):
    """
  Load the .mat file and return the csi_trace data.
  """
    try:
        data = scipy.io.loadmat(file_path, simplify_cells=True)
        if 'csi_trace' in data:
            # Extract the CSI data from the nested structure
            csi_trace = data['csi_trace']['csi']
            return csi_trace, data
        else:
            raise ValueError("csi_trace not found in the .mat file")
    except NotImplementedError:
        data = mat73.loadmat(file_path)
        if 'csi_trace' in data:
            # Extract the CSI data from the nested structure
            csi_trace = data['csi_trace']['csi']
            return csi_trace, data
        else:
            raise ValueError("csi_trace not found in the .mat file")



def rescale_csi(data,csi_trace,set_sampling_rate):
    csi_data = csi_trace
    timestamp = data['csi_trace']['mactimer']
    # Calculate actual sampling rate
    time_diff = np.diff(timestamp) / 1e6  # Convert microseconds to seconds
    actual_sampling_rate = 1 / np.mean(time_diff)
    # Check if the actual sampling rate calculation is reasonable
    if actual_sampling_rate < 1:
        actual_sampling_rate = 1 / np.mean(time_diff)
    sampling_rate_ratio = set_sampling_rate / actual_sampling_rate
    # and it has dimensions [tx, rx, sub_carriers, time_samples]
    tx, rx, sub_carriers, time_samples = csi_data.shape
    # Calculate the new number of time samples
    new_time_samples = int(time_samples * sampling_rate_ratio)
    resampled_data = np.zeros((tx, rx, sub_carriers, new_time_samples), dtype='complex')
    # Resample each combination of tx, rx, sub_carriers
    for i in range(tx):
        for j in range(rx):
            for k in range(sub_carriers):
                resampled_data[i, j, k] = resample(csi_data[i, j, k], new_time_samples)

    return actual_sampling_rate, sampling_rate_ratio, resampled_data


def transform_csi_to_real(segment):
    """
  Transform CSI from complex value to amplitude and phase and save in two channels
  """
    # Calculate amplitude and phase
    amplitude = np.abs(segment)
    phase = np.angle(segment)
    # Stack the amplitude and phase into two channels
    csi_transformed = np.stack((amplitude, phase), axis=0)  # Adjust axis if needed based on your data dimensions
    return csi_transformed


def segment_csi_trace(csi_trace, sample_length):
    """
    Segment the csi data into chunks based on the sample_length.
    """
    num_segments = csi_trace.shape[-1] // sample_length
    sample_list = []
    for i in range(num_segments):
        start_index = i * sample_length
        end_index = start_index + sample_length
        segment = csi_trace[..., start_index:end_index]
        sample_list.append(segment)
    return sample_list


class SSLCSIDataset(Dataset):
    def __init__(self, data_dir, device, win_len, sample_rate):
        """data_dir = root_dir +'/SSL/'"""
        self.samples = []
        folder_name = os.path.join(data_dir, device)
        # file_list = []
        # file_list = os.listdir(folder_name)
        # file_list = [i for i in file_list if i.startswith('motion_csi') or i.startswith('csi') or i.startswith('fall_csi')]
        # print(file_list)
        file_list = glob.glob(folder_name+"\\2024\\*\\*csi*.mat")
        num=0
        for file_path in file_list:
            if num/100==0:
                print(f"loading # {num} files")
                print("loading file: "+file_path)
                num = num + 1
            else:
                num = num+1
            csi_trace,data = load_mat_file(os.path.join(folder_name, file_path))
            _,_,csi_trace_resampled = rescale_csi(data, csi_trace, sample_rate)
            csi_transformed = transform_csi_to_real(csi_trace_resampled)
            sample_length = win_len
            sample_list = segment_csi_trace(csi_transformed, sample_length)
            self.samples.extend(sample_list)
        # # self.samples = torch.stack(self.samples, dim=0)
        # self.samples = torch.unsqueeze(torch.cat(self.samples, dim=0),dim=-3)

    def __len__(self):
        # print("samples length is: ", len(self.samples))
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # Convert the sample to a PyTorch tensor
        sample_tensor = torch.from_numpy(sample).float()
        return sample_tensor



###############TEST############################################
# import numpy as np
# import mat73
# import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import ttk
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#
# file_path = "E:\Dataset\Dataset_OW\DatasetHP\HM3_HP\DanHome\Data\Fan\loc1\Fan\csi_20220305002915_all.mat"
# # file_path = "E:\Dataset\Dataset_OW\DatasetHP\ServerHP\\2023\\1B31WA000005\motion_csi_20230217160351_part1.mat"
#
# csi_trace,data = load_mat_file(file_path)
#
# _,_,resampled_data = rescale_csi(data,csi_trace, set_sampling_rate=1500)
#
#
#
#
# def plot_data():
#     try:
#         # Sample indices for demonstration
#         tx_index, rx_index, subcarrier_index = 0, 0, 0
#
#         # Prepare x data based on the length of y data
#         y_original = np.abs(csi_trace[tx_index, rx_index, subcarrier_index])  # Assuming csi_data is available
#         y_resampled = np.abs(
#             resampled_data[tx_index, rx_index, subcarrier_index])  # Assuming resampled_data is available
#         x_original = np.arange(len(y_original))
#         x_resampled = np.arange(len(y_resampled))
#
#         # Creating two separate figures for original and resampled data
#         fig1, ax1 = plt.subplots()
#         ax1.plot(x_original, y_original, label='Original Magnitude')
#         ax1.set_title('Original Data Magnitude')
#         ax1.set_xlabel('Time Samples')
#         ax1.set_ylabel('Magnitude')
#         ax1.legend()
#
#         fig2, ax2 = plt.subplots()
#         ax2.plot(x_resampled, y_resampled, label='Resampled Magnitude')
#         ax2.set_title('Resampled Data Magnitude')
#         ax2.set_xlabel('Time Samples')
#         ax2.set_ylabel('Magnitude')
#         ax2.legend()
#
#         # Displaying figures on Tkinter canvas
#         canvas1 = FigureCanvasTkAgg(fig1, master=root)
#         canvas1_widget = canvas1.get_tk_widget()
#         canvas1_widget.grid(row=0, column=0)
#         canvas1.draw()
#
#         canvas2 = FigureCanvasTkAgg(fig2, master=root)
#         canvas2_widget = canvas2.get_tk_widget()
#         canvas2_widget.grid(row=0, column=1)
#         canvas2.draw()
#
#     except Exception as e:
#         print("Failed to plot due to:", e)
#
#
# # Tkinter window setup
# root = tk.Tk()
# root.title("Data Visualization")
# plot_button = ttk.Button(root, text="Plot Data", command=plot_data)
# plot_button.grid(row=1, column=0, columnspan=2)
#
# root.mainloop()
