import os
import pandas as pd
import numpy as np
import json

class LabelMapper:
    """
    Utility class for handling label mapping between string labels and integers.
    This helps ensure consistent label mapping across training and evaluation.
    """
    def __init__(self, label_column='label', save_path=None):
        self.label_column = label_column
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.num_classes = 0
        self.save_path = save_path
    
    def fit(self, metadata_path=None, metadata_df=None, labels=None):
        """
        Create a mapping from unique labels to indices.
        
        Args:
            metadata_path: Path to metadata CSV file.
            metadata_df: Pandas DataFrame containing metadata.
            labels: List of labels to use for mapping.
        """
        if metadata_path is not None:
            # Load metadata from CSV
            metadata_df = pd.read_csv(metadata_path)
            unique_labels = metadata_df[self.label_column].unique()
        elif metadata_df is not None:
            # Use provided DataFrame
            unique_labels = metadata_df[self.label_column].unique()
        elif labels is not None:
            # Use provided labels list
            unique_labels = np.unique(labels)
        else:
            raise ValueError("Either metadata_path, metadata_df, or labels must be provided")
        
        # Sort labels for deterministic mapping
        unique_labels = sorted(unique_labels)
        
        # Create mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        
        print(f"Created label mapping with {self.num_classes} classes:")
        for label, idx in self.label_to_idx.items():
            print(f"  {label} -> {idx}")
        
        # Save mapping if path is provided
        if self.save_path:
            self.save(self.save_path)
        
        return self
    
    def transform(self, labels):
        """
        Transform string labels to indices.
        
        Args:
            labels: String label or list of string labels.
            
        Returns:
            Integer indices.
        """
        if isinstance(labels, (list, np.ndarray, pd.Series)):
            return np.array([self.label_to_idx.get(str(label), 0) for label in labels])
        else:
            return self.label_to_idx.get(str(labels), 0)
    
    def inverse_transform(self, indices):
        """
        Transform indices back to string labels.
        
        Args:
            indices: Integer index or list of integer indices.
            
        Returns:
            String labels.
        """
        if isinstance(indices, (list, np.ndarray)):
            return np.array([self.idx_to_label.get(idx, "unknown") for idx in indices])
        else:
            # Single index
            return self.idx_to_label.get(indices, "unknown")
    
    def save(self, path):
        """
        Save the label mapping to a JSON file.
        
        Args:
            path: Path to save the mapping.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        mapping = {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': {str(k): v for k, v in self.idx_to_label.items()},  # Convert int keys to str for JSON
            'num_classes': self.num_classes
        }
        
        with open(path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        print(f"Saved label mapping to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load a label mapping from a JSON file.
        
        Args:
            path: Path to the saved mapping.
            
        Returns:
            LabelMapper instance.
        """
        with open(path, 'r') as f:
            mapping = json.load(f)
        
        mapper = cls()
        mapper.label_to_idx = mapping['label_to_idx']
        mapper.idx_to_label = {int(k): v for k, v in mapping['idx_to_label'].items()}  # Convert str keys back to int
        mapper.num_classes = mapping['num_classes']
        
        return mapper

    def get_name(self, idx):
        """
        Get the name (label) for a given index.
        
        Args:
            idx: The index of the class
            
        Returns:
            The name (label) of the class
        """
        return self.idx_to_label.get(idx, "unknown")

def create_label_mapper_from_metadata(metadata_path, label_column='label', save_path=None):
    """
    Utility function to create a label mapper from metadata file.
    
    Args:
        metadata_path: Path to metadata CSV file.
        label_column: Column containing labels.
        save_path: Path to save the mapping.
        
    Returns:
        LabelMapper instance and number of classes.
    """
    mapper = LabelMapper(label_column=label_column, save_path=save_path)
    mapper.fit(metadata_path=metadata_path)
    
    return mapper, mapper.num_classes 