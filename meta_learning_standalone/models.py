"""
Standalone models for meta-learning.
This module wraps the functionality from meta_model.py without circular imports.
"""
# Import directly from meta_model.py
from meta_model import (
    BaseMetaModel,
    MLPClassifier,
    LSTMClassifier,
    ResNet18Classifier,
    TransformerClassifier,
    ViTClassifier
)
