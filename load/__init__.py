from .supervised.benchmark_loader import load_benchmark_supervised

# Optional imports that may be used by other modules
try:
    from .supervised.benchmark_dataset import BenchmarkCSIDataset, load_benchmark_datasets
except ImportError:
    pass

try:
    from .supervised.label_utils import LabelMapper, create_label_mapper_from_metadata
except ImportError:
    pass

# Factory functions for easier API access
from .base import (
    get_data_loader,
    get_model_loader
)

__all__ = [
    'load_csi_supervised',
    'load_model_pretrained',
    'fine_tune_model',
    'load_model_trained',
    'load_model_scratch',
    'load_benchmark_supervised',
    'get_data_loader',
    'get_model_loader'
]

# Dynamically add other exports if available
if 'BenchmarkCSIDataset' in globals():
    __all__.extend(['BenchmarkCSIDataset', 'load_benchmark_datasets'])
if 'LabelMapper' in globals():
    __all__.extend(['LabelMapper', 'create_label_mapper_from_metadata'])