import os
import torch
import numpy as np
import scipy.io as sio
import mat73
from torch.utils.data import Dataset, random_split
from data.preprocessing.csi_preprocessing import normalize_csi

class CSIDatasetMAT(Dataset):
    """Dataset for CSI data in MAT format.
    
    This class can be used for training, validation, and testing based on the dataset_type parameter.
    """
    
    def __init__(self, data_dir, task='ThreeClass', transform=None, dataset_type='train'):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory or list of directories containing CSI data
            task: Task type ('ThreeClass', 'HumanNonhuman', etc.)
            transform: Transform to apply to the data
            dataset_type: Type of dataset ('train', 'validation', 'test')
        """
        self.transform = transform
        self.task = task
        self.dataset_type = dataset_type
        
        # Convert single directory to list
        if isinstance(data_dir, str):
            self.data_dir = [data_dir]
        else:
            self.data_dir = data_dir
        
        # Storage for data and labels
        self.samples = None
        self.labels = []
        self.file_paths = []  # Store file paths for test set metadata analysis
        
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
        
        # Create class name mapping (for test set analysis)
        self.idx_to_class = {}
        for task_name, class_dict in self.class_mappings.items():
            if task_name == self.task:
                self.idx_to_class = {v: k for k, v in class_dict.items()}
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load CSI data from MAT files."""
        # Process each directory
        all_samples = []
        
        for dir_path in self.data_dir:
            # Skip if not a directory
            if not os.path.isdir(dir_path):
                continue
                
            # Get all .mat files in directory
            file_list = [f for f in os.listdir(dir_path) if f.endswith('.mat')]
            
            # Filter files for ThreeClass if needed
            if self.task == 'ThreeClass':
                file_list = [f for f in file_list if "Fan" not in f]
            
            # Process each file
            for filename in file_list:
                file_path = os.path.join(dir_path, filename)
                log_prefix = f"[{self.dataset_type}]"
                print(f"{log_prefix} Processing file: {file_path}")
                try:
                    # Load samples and convert to tensor
                    samples = mat73.loadmat(file_path)['X']
                    samples_tensor = torch.from_numpy(samples).float()
                    
                    # Get label for this file
                    label = self.generate_label(file_path)
                    if label is not None:
                        # Add samples and labels
                        all_samples.append(samples_tensor)
                        # Add same label for all samples
                        for i in range(samples_tensor.shape[0]):
                            self.labels.append(label)
                            if self.dataset_type == 'test':
                                self.file_paths.append(file_path)
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
        
        # Combine all samples and add channel dimension
        if len(all_samples) > 0:
            # Concatenate tensors and add channel dimension
            self.samples = torch.unsqueeze(torch.cat(all_samples, dim=0), dim=-3)
            print(f"{log_prefix} Loaded {len(self.labels)} samples with shape {self.samples.shape}")
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
        
        # Human/Nonhuman classification
        if self.task == 'HumanNonhuman':
            if 'human' in file_name:
                return 1
            else:
                return 0
        
        # Four-class classification
        elif self.task == 'FourClass':
            if 'human' in file_name:
                return 1
            elif 'pet' in file_name:
                return 2
            elif 'irobot' in file_name:
                return 3
            elif 'fan' in file_name or 'empty' in file_name or 'nomotion' in file_name:
                return 0
            else:
                return None
        
        # Three-class classification
        elif self.task == 'ThreeClass':
            if 'human' in file_name:
                return 1
            elif 'pet' in file_name or 'irobot' in file_name or 'fan' in file_name:
                return 2
            elif 'empty' in file_name or 'nomotion' in file_name:
                return 0
            else:
                return None
        
        # Detection task (binary)
        elif self.task == 'Detection':
            if 'nomotion' in file_name or 'empty' in file_name:
                return 0
            elif any(x in file_name for x in ['human', 'fan', 'pet', 'irobot']):
                return 1
            else:
                return None
                
        # Detection and Classification
        elif self.task == 'DetectionandClassification':
            if 'nomotion' in file_name or 'empty' in file_name:
                return 0
            elif 'human' in file_name:
                return 1
            elif 'pet' in file_name:
                return 2
            elif 'irobot' in file_name:
                return 3
            elif 'fan' in file_name:
                return 4
            else:
                return None
        
        # Human ID
        elif self.task == 'HumanID':
            tester_list = ['Andrew', 'Brain', 'Brendon', 'Dan']
            for ind, val in enumerate(tester_list):
                if val.lower() in file_name:
                    return ind
            return None
        
        # Human Motion
        elif self.task == 'HumanMotion':
            motion_list = ['Running', 'Sneaking', 'Walking']
            for ind, val in enumerate(motion_list):
                if val.lower() in file_name:
                    return ind
            return None
        
        # NTU Human ID
        elif self.task == 'NTUHumanID':
            tester_list = ['001', '002', '003', '004', '005',
                          '006', '007', '008', '009', '010',
                          '011', '012', '013', '014', '015']
            for ind, val in enumerate(tester_list):
                if val in file_name:
                    return ind
            return None
        
        # NTU HAR
        elif self.task == 'NTUHAR':
            activity_list = ['run', 'walk', 'box', 'circle', 'clean', 'fall']
            for ind, val in enumerate(activity_list):
                if val.lower() in file_name:
                    return ind
            return None
        
        # Widar
        elif self.task == 'Widar':
            activity_list = ['PP', 'Sw', 'Cl', 'Sl', 'DNH', 'DOH', 'DRH', 'DTH',
                           'DZH', 'DZ', 'DN', 'DO', 'Dr1', 'Dr2', 'Dr3', 'Dr4', 'Dr5',
                           'Dr6', 'Dr7', 'Dr8', 'Dr9', 'Dr10']
            for ind, val in enumerate(activity_list):
                if val in file_name:
                    return ind
            return None
        
        # Fallback to parent directory
        parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
        
        # Try to determine label from directory name
        # Using the same logic as above for each task type
        
        print(f'No label determined for {file_path}')
        return None
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, index):
        """Return a sample from the dataset."""
        sample = self.samples[index]
        label = self.labels[index]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
    

    
    def get_metadata(self):
        """Get test set metadata (only useful when dataset_type='test')"""
        if self.dataset_type != 'test':
            return {"warning": "Metadata is only available for test datasets"}
            
        metadata = {
            "num_samples": len(self.labels),
            "class_distribution": self._get_class_distribution()
        }
        
        # Add class names to distribution
        class_distribution_with_names = {}
        for class_idx, count in metadata["class_distribution"].items():
            class_name = self.idx_to_class.get(class_idx, f"Class {class_idx}")
            class_distribution_with_names[class_name] = count
        
        metadata["class_distribution"] = class_distribution_with_names
        
        # Add file information
        if hasattr(self, 'file_paths') and self.file_paths:
            metadata["files"] = self.file_paths
            
        return metadata
    
    def _get_class_distribution(self):
        """Get class distribution in the dataset"""
        if not self.labels:
            return {}
            
        class_counts = {}
        for label in self.labels:
            label_item = label.item() if torch.is_tensor(label) else label
            class_counts[label_item] = class_counts.get(label_item, 0) + 1
        
        return class_counts
        
    def get_confusion_matrix_labels(self):
        """Get labels for confusion matrix visualization"""
        return [self.idx_to_class.get(i, f"Class {i}") for i in range(len(self.idx_to_class))]
