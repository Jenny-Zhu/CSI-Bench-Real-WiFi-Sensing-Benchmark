# Global exports for backward compatibility
from .supervised.data_loader import (
    load_csi_supervised,
    load_acf_supervised
)

from .supervised.model_loader import (
    fine_tune_model,
    load_model_trained,
    load_model_scratch
)

from .meta_learning.data_loader import (
    load_csi_data_benchmark
)

from .meta_learning.model_loader import (
    load_csi_model_benchmark
)

# Factory functions for easier API access
from .base import (
    get_data_loader,
    get_model_loader
)

__all__ = [
    'load_csi_supervised',
    'load_acf_supervised',
    'load_model_pretrained',
    'fine_tune_model',
    'load_model_trained',
    'load_model_scratch',
    'load_csi_data_benchmark',
    'load_csi_model_benchmark',
    'get_data_loader',
    'get_model_loader'
]