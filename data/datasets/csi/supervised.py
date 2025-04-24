import os
import torch
import numpy as np
import h5py
import scipy.io as sio
import mat73
from torch.utils.data import random_split
from data.datasets.base_dataset import BaseDataset
from data.preprocessing.csi_preprocessing import normalize_csi

class CSIDatasetOW_HM3(BaseDataset):
    """Dataset for supervised learning with CSI data."""
    
    def __init__(self, data_dir, win_len=250, sample_rate=100, if_test=0, transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing CSI data.
            win_len: Window length for segmentation.
            sample_rate: Sampling rate of the data.
            if_test: Whether to use test data (0 for train, 1 for test).
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        self.win_len = win_len
        self.sample_rate = sample_rate
        self.if_test = if_test
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load CSI data from files."""
        for dir_path in self.data_dir:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.h5') or file.endswith('.mat'):
                        file_path = os.path.join(root, file)
                        self.process_file(file_path)
    
    def process_file(self, file_path):
        """Process a single CSI file.
        
        Args:
            file_path: Path to the CSI file.
        """
        try:
            if file_path.endswith('.mat'):
                data, label = self.load_mat_file(file_path)
            elif file_path.endswith('.h5'):
                data, label = self.load_h5_file(file_path)
            else:
                return
            
            # Skip if no valid data or label
            if data is None or label is None:
                return
            
            # Add to dataset
            self.data.append(data)
            self.labels.append(label)
        
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    def load_mat_file(self, file_path):
        """Load data from a .mat file.
        
        Args:
            file_path: Path to the .mat file.
            
        Returns:
            A tuple of (data, label).
        """
        try:
            mat_data = sio.loadmat(file_path)
            # Extract data and label based on your file structure
            # This is a placeholder
            data = mat_data.get('csi_data', None)
            label = mat_data.get('label', None)
            
            if data is not None:
                data = normalize_csi(data)
                data = torch.from_numpy(data).float()
            
            if label is not None:
                label = torch.from_numpy(label).long()
            
            return data, label
            
        except Exception as e:
            print(f"Error loading MAT file {file_path}: {e}")
            return None, None
    
    def load_h5_file(self, file_path):
        """Load data from an .h5 file.
        
        Args:
            file_path: Path to the .h5 file.
            
        Returns:
            A tuple of (data, label).
        """
        try:
            with h5py.File(file_path, 'r') as f:
                # Extract data and label based on your file structure
                # This is a placeholder
                data = np.array(f.get('csi_data', None))
                label = np.array(f.get('label', None))
                
                if data is not None:
                    data = normalize_csi(data)
                    data = torch.from_numpy(data).float()
                
                if label is not None:
                    label = torch.from_numpy(label).long()
                
                return data, label
                
        except Exception as e:
            print(f"Error loading HDF5 file {file_path}: {e}")
            return None, None

class CSIDatasetOW_HM3_H5(BaseDataset):
    """Dataset for supervised learning with CSI data from HDF5 files."""
    
    def __init__(self, data_dir, win_len=250, sample_rate=100, if_test=0, transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing CSI data.
            win_len: Window length for segmentation.
            sample_rate: Sampling rate of the data.
            if_test: Whether to use test data (0 for train, 1 for test).
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        self.win_len = win_len
        self.sample_rate = sample_rate
        self.if_test = if_test
        
        # 添加数据跟踪信息
        self.files_processed = 0
        self.files_skipped = 0
        self.samples_loaded = 0
        
        # Load data
        self.data = []
        self.labels = []
        self.load_data()
        
        print(f"CSIDatasetOW_HM3_H5 initialization completed: Processed {self.files_processed} files, skipped {self.files_skipped} files, loaded {self.samples_loaded} samples")
    
    def load_data(self):
        """Load CSI data from HDF5 files."""
        for dir_path in self.data_dir:
            if not os.path.exists(dir_path):
                print(f"Warning: Directory {dir_path} does not exist")
                continue
                
            h5_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.h5')]
            
            if not h5_files:
                print(f"Warning: No .h5 files found in directory {dir_path}")
                continue
                
            print(f"Found {len(h5_files)} .h5 files in directory {dir_path}")
            
            for file_path in h5_files:
                try:
                    print(f"Processing file: {file_path}")
                    with h5py.File(file_path, 'r') as f:
                        # Determine if this is a test or train file based on naming convention
                        is_test_file = 'test' in os.path.basename(file_path).lower()
                        
                        # Only process files matching the requested split
                        if (self.if_test == 1 and is_test_file) or (self.if_test == 0 and not is_test_file):
                            # Check keys in file
                            print(f"File {os.path.basename(file_path)} contains keys: {list(f.keys())}")
                            
                            # Extract data and labels - customize based on actual structure
                            csi_data = None
                            labels = None
                            
                            # Try different key names to accommodate different data formats
                            for key in ['csi_data', 'csi', 'data']:
                                if key in f:
                                    csi_data = np.array(f[key])
                                    print(f"Loaded CSI data from key '{key}', shape: {csi_data.shape}")
                                    break
                                    
                            for key in ['labels', 'label', 'y']:
                                if key in f:
                                    labels = np.array(f[key])
                                    print(f"Loaded labels from key '{key}', shape: {labels.shape}")
                                    break
                            
                            if csi_data is None:
                                print(f"Warning: No CSI data found in file {file_path}")
                                self.files_skipped += 1
                                continue
                                
                            if labels is None:
                                print(f"Warning: No labels found in file {file_path}")
                                self.files_skipped += 1
                                continue
                            
                            # Process and add each example
                            samples_in_file = 0
                            for i in range(len(labels)):
                                try:
                                    sample = csi_data[i]
                                    label = labels[i]
                                    
                                    # Check validity and display warnings
                                    if np.isnan(sample).any():
                                        print(f"Warning: Sample {i} contains NaN values, will attempt to clean")
                                        sample = np.nan_to_num(sample)
                                        
                                    # Normalize and convert to tensor
                                    try:
                                        sample = normalize_csi(sample)
                                        sample_tensor = torch.from_numpy(sample).float()
                                        label_tensor = torch.tensor(label).long()
                                        
                                        self.data.append(sample_tensor)
                                        self.labels.append(label_tensor)
                                        samples_in_file += 1
                                        self.samples_loaded += 1
                                    except Exception as e:
                                        print(f"Error processing sample {i}: {e}")
                                except Exception as e:
                                    print(f"Error accessing sample {i}: {e}")
                            
                            print(f"Loaded {samples_in_file} samples from file {os.path.basename(file_path)}")
                            self.files_processed += 1
                        else:
                            print(f"Skipping file {os.path.basename(file_path)}: Does not match current dataset split")
                            self.files_skipped += 1
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    self.files_skipped += 1
        
        if not self.data:
            print(f"Warning: No data loaded! Check data directory and file format.")
        else:
            print(f"Successfully loaded dataset: {len(self.data)} samples")
            
    def __len__(self):
        """Get the length of the dataset."""
        return len(self.labels)
    
    def __getitem__(self, index):
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample.
            
        Returns:
            A tuple of (sample, label).
        """
        # 添加索引边界检查
        if index >= len(self.labels) or index >= len(self.data):
            raise IndexError(f"Index {index} out of bounds for dataset with {len(self.labels)} labels and {len(self.data)} samples")
            
        sample = self.data[index]
        label = self.labels[index]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

class CSIDatasetMAT(BaseDataset):
    """Dataset for supervised learning with CSI data from MAT files.
    Integrated from legacy implementation, works for both CSI and ACF data."""
    
    def __init__(self, data_dir, task='ThreeClass', transform=None):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing CSI/ACF data.
            task: Task type ('ThreeClass', 'HumanNonhuman', etc.).
            transform: Transform to apply to the data.
        """
        super().__init__(data_dir, transform)
        
        self.task = task
        self.samples = []
        self.labels = []
        
        # Task to class mapping
        self.class_mappings = {
            'HumanNonhuman': {'human': 1, 'nonhuman': 0},
            'FourClass': {'empty': 0, 'human': 1, 'animal': 2, 'object': 3},
            'HumanID': {'person1': 0, 'person2': 1, 'person3': 2, 'person4': 3},
            'HumanMotion': {'static': 0, 'walking': 1, 'running': 2},
            'ThreeClass': {'empty': 0, 'human': 1, 'nonhuman': 2},
            'DetectionandClassification': {'empty': 0, 'human': 1, 'animal': 2, 'object': 3, 'multiple': 4},
            'Detection': {'empty': 0, 'nonempty': 1},
            'NTUHumanID': {f'person{i}': i for i in range(15)},
            'NTUHAR': {'walking': 0, 'sitting': 1, 'standing': 2, 'jumping': 3, 'falling': 4, 'lying': 5},
            'Widar': {f'activity{i}': i for i in range(22)}
        }
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load data from MAT files."""
        # Collect all MAT files from the directories
        all_mat_files = []
        for dir_path in self.data_dir:
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.mat'):
                        all_mat_files.append(os.path.join(root, file))
        
        # Process each MAT file
        for file_path in all_mat_files:
            try:
                print(f"Loading file: {file_path}")
                samples = mat73.loadmat(file_path)['X']
                samples_tensor = torch.from_numpy(samples).float()
                
                # Handle different tensor shapes
                if samples_tensor.shape[0] == 250:  # Single sample case
                    samples_tensor = samples_tensor.unsqueeze(0)
                
                # Get label for this file
                label = self.generate_label(file_path)
                
                # If valid label is found, add samples and labels
                if label is not None:
                    self.samples.append(samples_tensor)
                    # Add same label for all samples in this file
                    for i in range(samples_tensor.shape[0]):
                        self.labels.append(label)
                else:
                    print(f"Skipping file {file_path} - no valid label determined")
                
            except Exception as e:
                print(f"Error loading MAT file {file_path}: {e}")
        
        # Combine all sample tensors and add channel dimension if needed
        if len(self.samples) > 0:
            try:
                # 合并前检查样本总数和标签总数是否匹配
                total_samples = sum(s.shape[0] for s in self.samples)
                if total_samples != len(self.labels):
                    print(f"WARNING: Mismatch between total samples ({total_samples}) and labels ({len(self.labels)})")
                    # 根据情况调整
                    if total_samples > len(self.labels):
                        print("Truncating samples to match label count")
                        # 截取样本到标签数量
                        temp_samples = []
                        sample_count = 0
                        for s in self.samples:
                            sample_batch_size = s.shape[0]
                            if sample_count + sample_batch_size <= len(self.labels):
                                temp_samples.append(s)
                                sample_count += sample_batch_size
                            else:
                                # 只取部分样本
                                remaining = len(self.labels) - sample_count
                                if remaining > 0:
                                    temp_samples.append(s[:remaining])
                                    sample_count += remaining
                                break
                        self.samples = temp_samples
                    else:
                        print("Truncating labels to match sample count")
                        self.labels = self.labels[:total_samples]
                
                self.samples = torch.unsqueeze(torch.cat(self.samples, dim=0), dim=-3)
                
                # 最后安全检查
                if self.samples.shape[0] != len(self.labels):
                    print(f"CRITICAL: After processing, sample count ({self.samples.shape[0]}) still doesn't match label count ({len(self.labels)})")
                    min_size = min(self.samples.shape[0], len(self.labels))
                    self.samples = self.samples[:min_size]
                    self.labels = self.labels[:min_size]
                    print(f"Final adjustment: truncated to {min_size} samples and labels")
                
                print(f"Dataset loaded: {len(self.labels)} samples with shape {self.samples.shape}")
            except Exception as e:
                print(f"Error combining samples: {e}")
                import traceback
                traceback.print_exc()
                # 失败时采用简单回退方案
                if len(self.samples) > 0:
                    first_batch = self.samples[0]
                    self.samples = torch.unsqueeze(first_batch, dim=-3)
                    self.labels = self.labels[:first_batch.shape[0]]
                    print(f"Fallback to using only first batch: {self.samples.shape[0]} samples")
        else:
            print("No valid samples found!")
    
    def generate_label(self, file_path):
        """Generate label based on file name and task.
        
        Args:
            file_path: Path to the data file.
            
        Returns:
            Label as an integer.
        """
        file_name = os.path.basename(file_path).lower()
        
        # Get mapping for the current task
        mapping = self.class_mappings.get(self.task, {})
        
        # Human/Nonhuman classification
        if self.task == 'HumanNonhuman':
            if 'human' in file_name:
                print('Human labeled (1)')
                return 1
            else:
                print('Nonhuman labeled (0)')
                return 0
        
        # Four-class classification
        elif self.task == 'FourClass':
            if 'human' in file_name:
                print('Human labeled (1)')
                return 1
            elif 'pet' in file_name:
                print('Pet labeled (2)')
                return 2
            elif 'irobot' in file_name:
                print('IRobot labeled (3)')
                return 3
            elif 'fan' in file_name or 'empty' in file_name or 'nomotion' in file_name:
                print('Empty/Fan labeled (0)')
                return 0
            else:
                print(f'Unrecognized class type for {file_name}')
        
        # Three-class classification
        elif self.task == 'ThreeClass':
            if 'human' in file_name:
                print('Human labeled (1)')
                return 1
            elif 'pet' in file_name:
                print('Pet labeled (2)')
                return 2
            elif 'irobot' in file_name:
                print('IRobot labeled (2)')
                return 2
            elif 'nomotion' in file_name or 'empty' in file_name:
                print('Empty labeled (0)')
                return 0
            else:
                print(f'Unrecognized class type for {file_name}')
        
        # Detection task (binary)
        elif self.task == 'Detection':
            if 'nomotion' in file_name or 'empty' in file_name:
                print('NoMotion labeled (0)')
                return 0
            elif any(x in file_name for x in ['human', 'fan', 'pet', 'irobot']):
                print('Motion labeled (1)')
                return 1
            else:
                print(f'Unrecognized class type for {file_name}')
        
        # Try to find a match in the mapping by checking each key in the file name
        for class_name, class_label in mapping.items():
            if class_name.lower() in file_name:
                print(f'{class_name} labeled ({class_label})')
                return class_label
        
        # Try parent directory name if file name doesn't match
        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
        for class_name, class_label in mapping.items():
            if class_name.lower() in parent_dir:
                print(f'{class_name} labeled from directory ({class_label})')
                return class_label
        
        print(f'No label determined for {file_path}')
        return None
    
    def __len__(self):
        """Get the length of the dataset."""
        return len(self.labels)
    
    def __getitem__(self, index):
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample.
            
        Returns:
            A tuple of (sample, label).
        """
        # 添加索引边界检查
        if index >= len(self.labels) or (hasattr(self, 'samples') and self.samples is not None and index >= self.samples.shape[0]):
            raise IndexError(f"Index {index} out of bounds for dataset with {len(self.labels)} labels and {self.samples.shape[0] if hasattr(self, 'samples') and self.samples is not None else 0} samples")
            
        sample = self.samples[index]
        label = self.labels[index]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
