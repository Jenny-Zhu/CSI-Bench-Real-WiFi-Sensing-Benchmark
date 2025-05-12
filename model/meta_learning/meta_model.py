import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Remove these imports
# from ..backbone.vit import ViTBackbone
# from ..backbone.cnn import CNNBackbone
# from ..backbone.hybrid import HybridBackbone

# Only keep the ClassificationHead if you need it
from ..base.heads import ClassificationHead

# Import the model classes we created
from .models import (
    MLPClassifier, LSTMClassifier, ResNet18Classifier, TransformerClassifier, ViTClassifier
)


class BaseMetaModel(nn.Module):
    def __init__(self,
                 model_type='vit',  # 'mlp', 'lstm', 'resnet18', 'transformer', 'vit'
                 win_len=250,
                 feature_size=98,
                 in_channels=1,
                 emb_dim=128,
                 num_classes=5,
                 dropout=0.1,
                 inner_lr=0.01,  # Learning rate for inner loop adaptation
                 **kwargs):
        super().__init__()

        self.model_type = model_type
        self.inner_lr = inner_lr

        # Initialize model based on type
        if model_type == 'mlp':
            self.model = MLPClassifier(
                win_len=win_len,
                feature_size=feature_size,
                num_classes=num_classes,
                emb_dim=emb_dim
            )
        elif model_type == 'lstm':
            self.model = LSTMClassifier(
                feature_size=feature_size,
                num_classes=num_classes,
                emb_dim=emb_dim
            )
        elif model_type == 'resnet18':
            self.model = ResNet18Classifier(
                win_len=win_len,
                feature_size=feature_size,
                num_classes=num_classes,
                emb_dim=emb_dim
            )
        elif model_type == 'transformer':
            self.model = TransformerClassifier(
                feature_size=feature_size,
                num_classes=num_classes,
                emb_dim=emb_dim
            )
        elif model_type == 'vit':
            self.model = ViTClassifier(
                win_len=win_len,
                feature_size=feature_size,
                in_channels=in_channels,
                emb_dim=emb_dim,
                num_classes=num_classes,
                dropout=dropout,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x):
        return self.model(x)

    def adapt(self, support_x, support_y, num_inner_steps=1):
        """
        Adapt model to support set (inner loop of MAML)

        Args:
            support_x: Support set inputs [N*K, C, H, W] where N is num_classes and K is shots
            support_y: Support set labels [N*K]
            num_inner_steps: Number of inner loop update steps

        Returns:
            Adapted model parameters
        """
        # Store the original parameters
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}

        # Inner loop adaptation
        for step in range(num_inner_steps):
            # Forward pass with current fast weights
            logits = self.forward_with_weights(support_x, fast_weights)

            # Compute loss
            loss = F.cross_entropy(logits, support_y)

            # Compute gradients
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)

            # Update fast weights
            fast_weights = {name: param - self.inner_lr * grad
                            for (name, param), grad in zip(fast_weights.items(), grads)}

        return fast_weights

    def forward_with_weights(self, x, weights):
        """
        Forward pass using specified weights

        Args:
            x: Input data
            weights: Dictionary of model parameters

        Returns:
            Model output using provided weights
        """
        # Create a temporary copy of the model
        model_copy = copy.deepcopy(self.model)

        # Replace model parameters with specified weights
        with torch.no_grad():
            for name, param in model_copy.named_parameters():
                if name in weights:
                    param.copy_(weights[name])

        # Forward pass with the copied model
        return model_copy(x)

    def meta_learning_forward(self, support_x, support_y, query_x):
        """
        MAML-style meta-learning forward pass

        Args:
            support_x: Support set inputs [N*K, C, H, W]
            support_y: Support set labels [N*K]
            query_x: Query set inputs [N*Q, C, H, W]

        Returns:
            Query set logits [N*Q, num_classes]
        """
        # Adapt to support set
        fast_weights = self.adapt(support_x, support_y)

        # Evaluate on query set using adapted weights
        logits = self.forward_with_weights(query_x, fast_weights)

        return logits

    def get_representation(self, x):
        """
        Get representation before classification head

        Args:
            x: Input data [B, C, H, W]

        Returns:
            Feature representation [B, emb_dim]
        """
        # This is model-specific, but for simplicity, we'll just return features
        # You might need to customize this based on your model architecture
        if hasattr(self.model, 'get_features'):
            return self.model.get_features(x)
        else:
            # For most models, you'll need a more specific implementation
            return None


# For backward compatibility
class CSIMetaModel(BaseMetaModel):
    """Specialized meta-learning model for CSI data"""

    def __init__(self, **kwargs):
        super().__init__(model_type='vit', **kwargs)


# Legacy models for backward compatibility

class CSI2DCNN(nn.Module):
    """Legacy 2D CNN model for CSI data, for backward compatibility"""

    def __init__(self, in_channels=1, num_classes=5, **kwargs):
        super().__init__()
        # Map to ResNet18 instead of CNN backbone
        self.model = BaseMetaModel(
            model_type='resnet18',
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )

    def forward(self, x):
        return self.model(x)

    def meta_learning_forward(self, support_x, support_y, query_x):
        return self.model.meta_learning_forward(support_x, support_y, query_x)


class CSITransformer(nn.Module):
    """Legacy Transformer model for CSI data, for backward compatibility"""

    def __init__(self, in_channels=1, num_classes=5, **kwargs):
        super().__init__()
        # Map to transformer
        self.model = BaseMetaModel(
            model_type='transformer',
            in_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )

    def forward(self, x):
        return self.model(x)

    def meta_learning_forward(self, support_x, support_y, query_x):
        return self.model.meta_learning_forward(support_x, support_y, query_x)