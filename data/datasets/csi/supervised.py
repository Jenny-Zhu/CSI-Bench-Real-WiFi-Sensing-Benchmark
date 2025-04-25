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
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load CSI data from HDF5 files."""
        for dir_path in self.data_dir:
            h5_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.h5')]
            
            for file_path in h5_files:
                try:
                    with h5py.File(file_path, 'r') as f:
                        # Determine if this is a test or train file based on naming convention
                        is_test_file = 'test' in os.path.basename(file_path).lower()
                        
                        # Only process files matching the requested split
                        if (self.if_test == 1 and is_test_file) or (self.if_test == 0 and not is_test_file):
                            # Extract data and labels - customize based on actual structure
                            csi_data = np.array(f.get('csi_data', None))
                            labels = np.array(f.get('labels', None))
                            
                            if csi_data is not None and labels is not None:
                                # Process and add each example
                                for i in range(len(labels)):
                                    sample = csi_data[i]
                                    label = labels[i]
                                    
                                    # Normalize and convert to tensor
                                    sample = normalize_csi(sample)
                                    sample_tensor = torch.from_numpy(sample).float()
                                    label_tensor = torch.tensor(label).long()
                                    
                                    self.data.append(sample_tensor)
                                    self.labels.append(label_tensor)
                                    
                except Exception as e:
                    print(f"Error loading HDF5 file {file_path}: {e}")

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
        # Process single directory or list of directories
        for dir_path in self.data_dir:
            # If it's a directory, look for .mat files
            if os.path.isdir(dir_path):
                # Get all .mat files in directory
                file_list = [f for f in os.listdir(dir_path) if f.endswith('.mat')]
                
                # Filter files for ThreeClass if needed
                if self.task == 'ThreeClass':
                    file_list = [f for f in file_list if "Fan" not in f]
                
                # Process each file
                for filename in file_list:
                    file_path = os.path.join(dir_path, filename)
                    print(f"Processing file: {file_path}")
                    try:
                        # Load samples and convert to tensor
                        samples = mat73.loadmat(file_path)['X']
                        samples_tensor = torch.from_numpy(samples).float()
                        
                        # Get label for this file
                        label = self.generate_label(file_path)
                        if label is not None:
                            # Add samples and labels
                            self.samples.append(samples_tensor)
                            # Add same label for all samples
                            for i in range(samples_tensor.shape[0]):
                                self.labels.append(label)
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
        
        # Combine all samples and add channel dimension
        if len(self.samples) > 0:
            # Concatenate tensors and add channel dimension
            self.samples = torch.unsqueeze(torch.cat(self.samples, dim=0), dim=-3)
            print(f"Loaded {len(self.labels)} samples with shape {self.samples.shape}")
        else:
            # Create empty tensor to avoid errors
            self.samples = torch.zeros((0, 1, 250, 100))
    
    def generate_label(self, file_path):
        """Generate label based on file name and task.
        
        Args:
            file_path: Path to the data file.
            
        Returns:
            Label as an integer.
        """
        file_name = os.path.basename(file_path).lower()
        print(f"Generating label for file: {file_name}, task: {self.task}")
        
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
            elif 'fan' in file_name:
                print('Fan labeled (0)')
                return 0
            elif 'empty' in file_name or 'nomotion' in file_name:
                print('Empty labeled (0)')
                return 0
            else:
                print(f'Unrecognized class type for {file_name}')
                return None
        
        # Three-class classification (matches old implementation)
        elif self.task == 'ThreeClass':
            if 'human' in file_name:
                print('Human labeled (0)')
                return 0
            elif 'pet' in file_name:
                print('Pet labeled (1)')
                return 1
            elif 'irobot' in file_name:
                print('IRobot labeled (2)')
                return 2
            else:
                print(f'Unrecognized class type for {file_name}')
                return None
        
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
                return None
                
        # Detection and Classification
        elif self.task == 'DetectionandClassification':
            if 'nomotion' in file_name or 'empty' in file_name:
                print('NoMotion labeled (0)')
                return 0
            elif 'human' in file_name:
                print('Human Motion labeled (1)')
                return 1
            elif 'pet' in file_name:
                print('Pet Motion labeled (2)')
                return 2
            elif 'irobot' in file_name:
                print('IRobot Motion labeled (3)')
                return 3
            elif 'fan' in file_name:
                print('Fan Motion labeled (4)')
                return 4
            else:
                print(f'Unrecognized class type for {file_name}')
                return None
        
        # Human ID
        elif self.task == 'HumanID':
            tester_list = ['Andrew', 'Brain', 'Brendon', 'Dan']
            for ind, val in enumerate(tester_list):
                if val.lower() in file_name:
                    print(f'{val} labeled ({ind})')
                    return ind
            print(f'Unrecognized class type for {file_name}')
            return None
        
        # Human Motion
        elif self.task == 'HumanMotion':
            motion_list = ['Running', 'Sneaking', 'Walking']
            for ind, val in enumerate(motion_list):
                if val.lower() in file_name:
                    print(f'{val} labeled ({ind})')
                    return ind
            print(f'Unrecognized class type for {file_name}')
            return None
        
        # NTU Human ID
        elif self.task == 'NTUHumanID':
            tester_list = ['001', '002', '003', '004', '005',
                          '006', '007', '008', '009', '010',
                          '011', '012', '013', '014', '015']
            for ind, val in enumerate(tester_list):
                if val in file_name:
                    print(f'Person {val} labeled ({ind})')
                    return ind
            print(f'Unrecognized class type for {file_name}')
            return None
        
        # NTU HAR
        elif self.task == 'NTUHAR':
            activity_list = ['run', 'walk', 'box', 'circle', 'clean', 'fall']
            for ind, val in enumerate(activity_list):
                if val.lower() in file_name:
                    print(f'{val} labeled ({ind})')
                    return ind
            print(f'Unrecognized class type for {file_name}')
            return None
        
        # Widar
        elif self.task == 'Widar':
            activity_list = ['PP', 'Sw', 'Cl', 'Sl', 'DNH', 'DOH', 'DRH', 'DTH',
                           'DZH', 'DZ', 'DN', 'DO', 'Dr1', 'Dr2', 'Dr3', 'Dr4', 'Dr5',
                           'Dr6', 'Dr7', 'Dr8', 'Dr9', 'Dr10']
            for ind, val in enumerate(activity_list):
                if val in file_name:
                    print(f'Activity {val} labeled ({ind})')
                    return ind
            print(f'Unrecognized class type for {file_name}')
            return None
        
        # Fallback to parent directory
        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
        print(f"Checking parent directory: {parent_dir}")
        
        # Try to determine label from directory name
        # Use the same logic as above for each task type
        
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
        sample = self.samples[index]
        label = self.labels[index]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
