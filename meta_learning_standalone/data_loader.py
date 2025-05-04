"""
Standalone data loader for meta-learning.
This module wraps the functionality from meta_learning_data.py without circular imports.
"""
# Import directly from meta_learning_data.py
from meta_learning_data import (
    load_meta_learning_tasks,
    load_csi_data_benchmark,
    MetaTaskSampler
)
