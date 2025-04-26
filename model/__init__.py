"""
Model module initialization.
Provides access to all model classes for both supervised learning and meta-learning.
"""

# Import supervised models
from .supervised.models import (
    MLPClassifier,
    LSTMClassifier,
    ResNet18Classifier,
    TransformerClassifier,
    ViTClassifier
)

# Import meta-learning models
from .meta_learning.meta_model import (
    BaseMetaModel,
    CSIMetaModel,
    ACFMetaModel,
    CSI2DCNN,
    CSITransformer
)

# For backwards compatibility

# Define what gets imported with "from model import *"
__all__ = [
    # Supervised models
    'MLPClassifier',
    'LSTMClassifier',
    'ResNet18Classifier',
    'TransformerClassifier',
    'ViTClassifier',
    # Meta-learning models
    'BaseMetaModel',
    'CSIMetaModel',
    'ACFMetaModel',
    'CSI2DCNN',
    'CSITransformer'
]

# Provide model type mapping for convenience
MODEL_TYPES = {
    'mlp': MLPClassifier,
    'lstm': LSTMClassifier,
    'resnet18': ResNet18Classifier,
    'transformer': TransformerClassifier,
    'vit': ViTClassifier
}

# Meta model type mapping
META_MODEL_TYPES = {
    'mlp': 'mlp',
    'lstm': 'lstm',
    'resnet18': 'resnet18',
    'transformer': 'transformer',
    'vit': 'vit',
    'csi2dcnn': 'resnet18',  # Legacy mapping
    'csitransformer': 'transformer'  # Legacy mapping
}


def get_model(model_name, **kwargs):
    """
    Factory function to create a model instance

    Args:
        model_name (str): Name of the model to create
        **kwargs: Additional arguments for model initialization

    Returns:
        Model instance
    """
    model_name = model_name.lower()

    if model_name in MODEL_TYPES:
        return MODEL_TYPES[model_name](**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_name}. Available models: {list(MODEL_TYPES.keys())}")


def get_meta_model(model_name, **kwargs):
    """
    Factory function to create a meta-learning model instance

    Args:
        model_name (str): Name of the model to create
        **kwargs: Additional arguments for model initialization

    Returns:
        Meta-learning model instance
    """
    model_name = model_name.lower()

    if model_name in META_MODEL_TYPES:
        model_type = META_MODEL_TYPES[model_name]
        return BaseMetaModel(model_type=model_type, **kwargs)
    else:
        raise ValueError(f"Unknown meta-model type: {model_name}. Available models: {list(META_MODEL_TYPES.keys())}")
