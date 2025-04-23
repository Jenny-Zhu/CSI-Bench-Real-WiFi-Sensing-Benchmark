"""
Base loading utilities and factory functions for all pipelines.
"""
import torch
import os
from enum import Enum

class DataType(Enum):
    CSI = 'csi'
    ACF = 'acf'

class TaskType(Enum):
    PRETRAINING = 'pretraining'
    SUPERVISED = 'supervised'
    META_LEARNING = 'meta_learning'

def get_data_loader(task_type, data_type, **kwargs):
    """
    Factory function to get the appropriate data loader based on task and data type.
    
    Args:
        task_type (str or TaskType): Type of task - 'pretraining', 'supervised', or 'meta_learning'
        data_type (str or DataType): Type of data - 'csi' or 'acf'
        **kwargs: Additional parameters to pass to the loader function
        
    Returns:
        Data loader or tuple of data loaders, depending on the task_type and data_type
    """
    # Convert string to enum if needed
    if isinstance(task_type, str):
        task_type = TaskType(task_type)
    if isinstance(data_type, str):
        data_type = DataType(data_type)
    
    if task_type == TaskType.PRETRAINING:
        if data_type == DataType.CSI:
            from .pretraining.data_loader import load_csi_data_unsupervised
            return load_csi_data_unsupervised(**kwargs)
        elif data_type == DataType.ACF:
            from .pretraining.data_loader import load_acf_data_unsupervised
            return load_acf_data_unsupervised(**kwargs)
    
    elif task_type == TaskType.SUPERVISED:
        if data_type == DataType.CSI:
            from .supervised.data_loader import load_data_supervised
            return load_data_supervised(**kwargs)
        elif data_type == DataType.ACF:
            from .supervised.data_loader import load_acf_supervised
            return load_acf_supervised(**kwargs)
    
    elif task_type == TaskType.META_LEARNING:
        if data_type == DataType.CSI:
            from .meta_learning.data_loader import load_csi_data_benchmark
            return load_csi_data_benchmark(**kwargs)
        elif data_type == DataType.ACF:
            raise NotImplementedError("ACF meta-learning data loader not implemented yet")
    
    raise ValueError(f"Unsupported combination of task_type={task_type} and data_type={data_type}")

def get_model_loader(task_type, data_type, **kwargs):
    """
    Factory function to get the appropriate model loader based on task and data type.
    
    Args:
        task_type (str or TaskType): Type of task - 'pretraining', 'supervised', or 'meta_learning'
        data_type (str or DataType): Type of data - 'csi' or 'acf'
        **kwargs: Additional parameters to pass to the loader function
        
    Returns:
        Model instance
    """
    # Convert string to enum if needed
    if isinstance(task_type, str):
        task_type = TaskType(task_type)
    if isinstance(data_type, str):
        data_type = DataType(data_type)
    
    if task_type == TaskType.PRETRAINING:
        if data_type == DataType.CSI:
            # Determine which unsupervised model to load
            if kwargs.get('joint', False):
                if kwargs.get('variable_shape', False):
                    from .pretraining.model_loader import load_model_unsupervised_joint_csi_var
                    return load_model_unsupervised_joint_csi_var(**kwargs)
                else:
                    from .pretraining.model_loader import load_model_unsupervised_joint
                    return load_model_unsupervised_joint(**kwargs)
            else:
                from .pretraining.model_loader import load_model_unsupervised
                return load_model_unsupervised(**kwargs)
        elif data_type == DataType.ACF:
            from .pretraining.model_loader import load_model_unsupervised_joint
            return load_model_unsupervised_joint(**kwargs)
    
    elif task_type == TaskType.SUPERVISED:
        # Check if we're loading a pretrained model or a model from scratch
        if kwargs.get('pretrained', False):
            from .supervised.model_loader import load_model_pretrained
            return load_model_pretrained(**kwargs)
        else:
            from .supervised.model_loader import load_model_scratch
            return load_model_scratch(**kwargs)
    
    elif task_type == TaskType.META_LEARNING:
        if data_type == DataType.CSI:
            from .meta_learning.model_loader import load_csi_model_benchmark
            return load_csi_model_benchmark(**kwargs)
        elif data_type == DataType.ACF:
            raise NotImplementedError("ACF meta-learning model loader not implemented yet")
    
    raise ValueError(f"Unsupported combination of task_type={task_type} and data_type={data_type}")

def variable_shape_collate_fn(batch):
    """
    Collate function for variable shape data in a batch.
    
    Args:
        batch: List of samples
        
    Returns:
        Batch as-is without stacking
    """
    # 'batch' is a list of items returned by __getitem__()
    # Each item has shape (T_i, F_i), which can be different
    # We just return the entire list as-is.
    return batch
