import scipy
import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import mat73
import h5py

from scipy.signal import resample
from concurrent.futures import ProcessPoolExecutor, as_completed
from .Util import load_files_by_type


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
        raise ValueError(
            "The total number of subcarriers across all tx-rx pairs is less than the requested number to select.")

    # Flatten the tx, rx, and subcarrier dimensions to treat as a single long list of subcarriers
    flattened_csi_data = csi_data.reshape(-1, csi_data.shape[-1])
    # Generate indices for the subcarriers to select
    selected_indices = np.random.choice(total_subcarriers, num_subcarriers, replace=False)
    # Select the subcarriers
    selected_csi_data = flattened_csi_data[selected_indices]
    # Reshape the data to match desired output shape: (1, 1, num_subcarriers, time_samples)
    selected_csi_data = selected_csi_data.reshape(1, 1, num_subcarriers, csi_data.shape[-1])

    return selected_csi_data


def rescale_csi(data, csi_trace, set_sampling_rate):
    csi_data = csi_trace
    timestamp = data['csi_trace']['mactimer']
    # Calculate actual sampling rate
    time_diff = np.diff(timestamp) / 1e6  # Convert microseconds to seconds
    actual_sampling_rate = 1 / np.mean(time_diff)
    # Check if the actual sampling rate calculation is reasonable
    if actual_sampling_rate < 1:
        actual_sampling_rate = 1 / np.mean(time_diff)
    down_factor = int(actual_sampling_rate // set_sampling_rate)
    if down_factor > 1:
        resampled_data = csi_data[..., ::down_factor]
    else:
        # No downsampling needed if the factor is less than or equal to 1
        resampled_data = csi_data
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


def segment_csi_trace_w_motion(file_path, csi_trace, down_factor):
    """
    Segment the csi data into chunks based on the sample_length.
    """
    file_name = os.path.basename(file_path)
    file_name = file_name[:-4]
    motion_label_path = os.path.join(os.path.dirname(file_path),'parsed_data_1500_bw_20','link_5',f'*{file_name}*.mat')
    motion_label_file = glob.glob(motion_label_path)
    if len(motion_label_file) != 1:
        raise ValueError(
            "Fail to search the label file")
    else:
        motion_file = scipy.io.loadmat(motion_label_file[0], simplify_cells=True)
        motion_label = motion_file['IDData']['motion_decision_vector']
        if sum(motion_label) > 0:
            # mapping_factor = int(csi_trace.shape[-1]//len(motion_label))
            motion_indices = np.round(np.where(motion_label == 1)[0]*30/down_factor)
            sample_list = []
            last_index = -1     # Last index processed to ensure no overlap
            for index in motion_indices:
                # Calculate start and end of the segment
                start = int(index - 250)
                end = int(index + 250)
                # Check if we are within bounds and there's no overlap
                if start >= 0 and end <= csi_trace.shape[-1] and start > last_index:
                    segment = csi_trace[..., start:end]
                    sample_list.append(segment)
                    last_index = end
            return sample_list
        else:
            raise ValueError('No motion detected')



def process_file(file_path, sample_rate, win_len):
    try:
        csi_trace, data = load_mat_file(file_path)
        selected_csi = select_subcarriers(csi_trace)
        _, down_factor, csi_trace_resampled = rescale_csi(data, selected_csi, sample_rate)
        csi_transformed = transform_csi_to_real(csi_trace_resampled)
        sample_list = segment_csi_trace_w_motion(file_path, csi_transformed, down_factor)
        return sample_list, None  # On success
    except Exception as e:
        return None, str(e)  # On failure




class CSIDatasetOW_HM3(Dataset):
    def __init__(self, win_len, sample_rate,if_test=0):
        task = 'OW_HM3'
        self.samples = []
        self.labels = []
        if if_test:
            set_type = 'test'
        else:
            set_type = 'train'
        save_path = os.path.join('dataset', 'metadata', f'HM3_sr{sample_rate}_wl{win_len}_all')
        os.makedirs(save_path, exist_ok=True)
        for ind,label in enumerate(['Human','Pet','IRobot','Fan']):
            self.samples_subject = []
            print(f"Processing {label} data for {set_type}")
            files = load_files_by_type(task,label,set_type)
            future_to_path = {}
            with ProcessPoolExecutor() as executor:
                for file_path in files:
                    future = executor.submit(process_file, file_path, sample_rate, win_len)
                    future_to_path[future] = file_path  # Store the mapping
                count = 0
                for future in as_completed(future_to_path):
                    file_path = future_to_path[future]  # Get the file path associated with the future
                    try:
                        result, error_message = future.result(timeout=120)  # Unpack the result and the error message
                        if error_message:
                            print(f"Error processing file {file_path}: {error_message}")
                            continue  # Skip this file and continue with the next

                        for data_array in result:
                            self.samples_subject.append(data_array)  # Add all samples in data_array to samples list
                            self.samples.append(data_array)  # Add all samples in data_array to samples list
                            self.labels.extend(
                                [ind])  # Extend labels list with repeated labels for each sample
                        count += 1
                        if count % 20 == 0:
                            print(f"{count} files processed.")
                    except TimeoutError:
                        print(f"Timeout occurred for file {file_path}")
                    except Exception as e:
                        print(f"General error processing file {file_path}: {e}")
            try:
                print("Attempting to save...")
                with h5py.File(os.path.join(save_path, f'samples_{label}_{set_type}_num{count}.h5'),
                               'w') as hf:
                    hf.create_dataset('data', data=np.array(self.samples_subject))
                # np.save(os.path.join(save_path, f'samples_{label}_{set_type}_num{count}.npy'),
                #         np.array(self.samples_subject, dtype=object))
                print("Save successful")
            except Exception as e:
                print(f'Fail to save file: {e}')

    def __len__(self):
        assert len(self.samples) == len(self.labels), "Mismatch between number of samples and labels"
        return len(self.samples)

    def __getitem__(self, index):
        if index >= len(self.samples):
            raise IndexError("Index out of range")
        return torch.from_numpy(self.samples[index]).float(), self.labels[index]
