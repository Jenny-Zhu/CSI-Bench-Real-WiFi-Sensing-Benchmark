"""
Model module initialization.
Provides access to all model classes for supervised learning.
"""

# Import supervised models
from .supervised.models import (
    MLPClassifier,
    LSTMClassifier,
    ResNet18Classifier,
    TransformerClassifier,
    ViTClassifier
)

# Define what gets imported with "from model import *"
__all__ = [
    # Supervised models
    'MLPClassifier',
    'LSTMClassifier',
    'ResNet18Classifier',
    'TransformerClassifier',
    'ViTClassifier'
]

# Provide model type mapping for convenience
MODEL_TYPES = {
    'mlp': MLPClassifier,
    'lstm': LSTMClassifier,
    'resnet18': ResNet18Classifier,
    'transformer': TransformerClassifier,
    'vit': ViTClassifier
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
