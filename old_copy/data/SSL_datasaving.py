import scipy
import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import mat73
import h5py

from scipy.signal import resample
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError


def load_mat_file(file_path):
    try:
        data = scipy.io.loadmat(file_path, simplify_cells=True)
    except NotImplementedError:
        data = mat73.loadmat(file_path)
    if 'csi_trace' in data:
        csi_trace = data['csi_trace']['csi']
        return csi_trace, data
    else:
        raise ValueError("csi_trace not found in the .mat file")


def select_subcarriers(csi_data, num_subcarriers=56):
    """
    Randomly selects a fixed number of subcarriers across all tx-rx pairs from the CSI data.

    Parameters:
    - csi_data: numpy array of shape [tx, rx, subcarriers, time_samples]
    - num_subcarriers: int, number of subcarriers to select

    Returns:
    - numpy array of shape [1, 1, num_subcarriers, time_samples] with the selected subcarriers.
    """
    total_subcarriers = np.prod(csi_data.shape[:-1])

    if total_subcarriers < num_subcarriers:
        raise ValueError("The total number of subcarriers across all tx-rx pairs is less than the requested number to select.")

    # Flatten the tx, rx, and subcarrier dimensions to treat as a single long list of subcarriers
    flattened_csi_data = csi_data.reshape(-1, csi_data.shape[-1])
    # Generate indices for the subcarriers to select
    selected_indices = np.random.choice(total_subcarriers, num_subcarriers, replace=False)
    # Select the subcarriers
    selected_csi_data = flattened_csi_data[selected_indices]
    # Reshape the data to match desired output shape: (1, 1, num_subcarriers, time_samples)
    selected_csi_data = selected_csi_data.reshape(1, 1, num_subcarriers, csi_data.shape[-1])

    return selected_csi_data


# def rescale_csi(data, csi_trace, set_sampling_rate):
#     csi_data = csi_trace
#     timestamp = data['csi_trace']['mactimer']
#
#     if len(timestamp) < 100:
#         return None, "Not enough timestamps to calculate sampling rate."
#
#     time_diff = np.diff(timestamp) / 1e6  # Convert microseconds to seconds
#
#     # if np.any(time_diff <= 0):
#     #     return None, "Timestamps must be strictly increasing."
#
#     actual_sampling_rate = 1 / np.mean(time_diff)
#     if actual_sampling_rate <= 0 or not np.isfinite(actual_sampling_rate):
#         return None, "Invalid calculated sampling rate."
#
#     sampling_rate_ratio = set_sampling_rate / actual_sampling_rate
#     if sampling_rate_ratio <= 0 or not np.isfinite(sampling_rate_ratio):
#         return None, "Invalid sampling rate ratio."
#
#     tx, rx, sub_carriers, time_samples = csi_data.shape
#     new_time_samples = int(time_samples * sampling_rate_ratio)
#     if new_time_samples <= 0:
#         return None, "Calculated number of new time samples is not positive."
#
#     resampled_data = np.zeros((tx, rx, sub_carriers, new_time_samples), dtype='complex')
#     for i in range(tx):
#         for j in range(rx):
#             for k in range(sub_carriers):
#                 resampled_data[i, j, k] = resample(csi_data[i, j, k], new_time_samples)
#
#     return resampled_data, None
def rescale_csi(data, csi_trace, set_sampling_rate):
    csi_data = csi_trace
    timestamp = data['csi_trace']['mactimer']

    if len(timestamp) < 2:
        return None, "Not enough timestamps to calculate sampling rate."

    time_diff = np.diff(timestamp) / 1e6  # Convert microseconds to seconds

    if np.any(time_diff <= 0):
        return None, "Timestamps must be strictly increasing."

    actual_sampling_rate = 1 / np.mean(time_diff)
    if actual_sampling_rate <= 0 or not np.isfinite(actual_sampling_rate):
        return None, "Invalid calculated sampling rate."

    # sampling_rate_ratio = set_sampling_rate / actual_sampling_rate
    # if sampling_rate_ratio <= 0 or not np.isfinite(sampling_rate_ratio):
    #     return None, "Invalid sampling rate ratio."
    #
    # tx, rx, sub_carriers, time_samples = csi_data.shape
    # new_time_samples = int(time_samples * sampling_rate_ratio)
    # if new_time_samples <= 0:
    #     return None, "Calculated number of new time samples is not positive."
    #
    # # Initialize the array with complex64 to reduce memory usage
    # resampled_data = np.zeros((tx, rx, sub_carriers, new_time_samples), dtype='complex64')
    # # Process chunks of data to limit memory usage
    # chunk_size = 10000  # Adjust the chunk size based on available memory and requirements
    #
    # for i in range(tx):
    #     for j in range(rx):
    #         for k in range(sub_carriers):
    #             csi_chunk = csi_data[i, j, k, :]
    #             resampled_chunk = np.array([], dtype='complex64')
    #             for start in range(0, len(csi_chunk), chunk_size):
    #                 end = min(start + chunk_size, len(csi_chunk))
    #                 temp_resampled = resample(csi_chunk[start:end], int((end - start) * sampling_rate_ratio))
    #                 resampled_chunk = np.concatenate((resampled_chunk, temp_resampled))
    #             resampled_data[i, j, k] = resampled_chunk
    #
    # return resampled_data, None

    down_factor = int(actual_sampling_rate // set_sampling_rate)
    if down_factor > 1:
        resampled_data = data[..., ::down_factor]
    else:
        # No downsampling needed if the factor is less than or equal to 1
        resampled_data = data
    # sampling_rate_ratio = set_sampling_rate / actual_sampling_rate
    # # and it has dimensions [tx, rx, sub_carriers, time_samples]
    # tx, rx, sub_carriers, time_samples = csi_data.shape
    # # Calculate the new number of time samples
    # new_time_samples = int(time_samples * sampling_rate_ratio)
    # resampled_data = np.zeros((tx, rx, sub_carriers, new_time_samples), dtype='complex')
    # # Resample each combination of tx, rx, sub_carriers
    # for i in range(tx):
    #     for j in range(rx):
    #         for k in range(sub_carriers):
    #             resampled_data[i, j, k] = resample(csi_data[i, j, k], new_time_samples)
    return actual_sampling_rate,down_factor,resampled_data
    # return actual_sampling_rate, sampling_rate_ratio, resampled_data


def transform_csi_to_real(segment):
    """
  Transform CSI from complex value to amplitude and phase and save in two channels
  """
    # Calculate amplitude and phase
    amplitude = np.abs(segment)
    phase = np.angle(segment)
    # Stack the amplitude and phase into two channels
    # Adjust axis if needed based on your data dimensions
    return np.stack((amplitude, phase), axis=0)


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


# def process_file(file_path, sample_rate, win_len):
#     try:
#         csi_trace, data = load_mat_file(file_path)
#         selected_csi = select_subcarriers(csi_trace)
#         csi_trace_resampled, error_message = rescale_csi(data, selected_csi, sample_rate)
#         if error_message:
#             return None, error_message  # Pass error message back to caller
#
#         csi_transformed = transform_csi_to_real(csi_trace_resampled)
#         sample_list = segment_csi_trace(csi_transformed, win_len)
#         return sample_list, None
#     except Exception as e:
#         return None, str(e)
#
def process_file(file_path, sample_rate, win_len):
    try:
        csi_trace, data = load_mat_file(file_path)
        if csi_trace is None:
            return None, f"Failed to load CSI trace from {file_path}"

        selected_csi = select_subcarriers(csi_trace)
        if selected_csi is None:
            return None, "Failed to select subcarriers"

        csi_trace_resampled, error_message = rescale_csi(data, selected_csi, sample_rate)
        if csi_trace_resampled is None:
            return None, error_message or f"Failed to rescale CSI data in {file_path}"

        csi_transformed = transform_csi_to_real(csi_trace_resampled)
        if csi_transformed is None:
            return None, f"Failed to transform CSI data in {file_path}"

        sample_list = segment_csi_trace(csi_transformed, win_len)
        if sample_list is None or not sample_list:
            return None, f"No segments created from CSI data in {file_path}"

        return sample_list, None
    except Exception as e:
        return None, str(e)


# def save_with_hdf5(data, file_path):
#     with h5py.File(file_path, 'w') as f:
#         # Create dataset with the exact shape of the input data
#         dset = f.create_dataset("samples", data=data)
#         print("Data saved!")
class SSLCSIDatasetSaving(Dataset):
    def __init__(self, data_dir, device, win_len, sample_rate):
        self.samples = []
        folder_name = os.path.join(data_dir, device)
        file_list = glob.glob(folder_name + "\\*\\*\\*\\*\\*\\*csi*.mat")
        save_path = os.path.join(folder_name, 'SSL', 'intermediate_test', 'ESP32_new')
        os.makedirs(save_path, exist_ok=True)

        future_to_path = {}  # Dictionary to map futures to file paths
        with ProcessPoolExecutor() as executor:
            for file_path in file_list:
                future = executor.submit(process_file, file_path, sample_rate, win_len)
                future_to_path[future] = file_path  # Store the mapping

            count = 0
            for future in as_completed(future_to_path):
                file_path = future_to_path[future]  # Get the file path associated with the future
                try:
                    result, error_message = future.result(timeout=120)
                    if error_message:
                        print(f"Error processing file {file_path}: {error_message}")
                        continue  # Skip this file and continue with the next

                    self.samples.extend(result)
                    count += 1
                    if count % 10 == 0 and len(self.samples)>0: # or len(self.samples) > 200
                        print(f"{count} files processed, current sample size: {len(self.samples)}")
                        if count > 0:
                            try:
                                print("Attempting to save...")
                                np.save(os.path.join(save_path, f'samples_sr{sample_rate}_wl{win_len}_num{count}.npy'),
                                        np.array(self.samples, dtype=object))
                                print("Save successful")
                            except Exception as e:
                                print(f'Fail to save file: {e}')
                        self.samples = []
                except TimeoutError:
                    print(f"Timeout occurred for file {file_path}")
                except Exception as e:
                    print(f"General error processing file {file_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.from_numpy(self.samples[index]).float()





# ##############TEST############################################
# import numpy as np
# import mat73
# import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import ttk
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#
# file_path = "E:\Dataset\Dataset_OW\HM3_IoT\HM3.0 11_6_2023 Test House IoT\Data\Fan\\test1\loc1\Data-SP4_20231107_142556_oneTimeSaving___-Echo1_2G.mat"
# # file_path = "E:\Dataset\Dataset_OW\DatasetHP\ServerHP\\2023\\1B31WA000005\motion_csi_20230217160351_part1.mat"
#
# csi_trace,data = load_mat_file(file_path)
# selected_csi = select_subcarriers(csi_trace)
# _,_,resampled_data = rescale_csi(data,selected_csi, set_sampling_rate=1500)
#
#
#
#
# def plot_data():
#     try:
#         # Sample indices for demonstration
#         tx_index, rx_index, subcarrier_index = 0, 0, 0
#         # subcarrier_index = 0
#
#         # Prepare x data based on the length of y data
#         y_original = np.abs(selected_csi[tx_index, rx_index, subcarrier_index])  # Assuming csi_data is available
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
