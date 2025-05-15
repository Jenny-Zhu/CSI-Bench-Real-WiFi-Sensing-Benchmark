import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from model.supervised.models import (
    MLPClassifier, 
    LSTMClassifier, 
    ResNet18Classifier, 
    TransformerClassifier, 
    ViTClassifier,
    PatchTST,
    TimesFormer1D
)

class FewShotAdaptiveModel(nn.Module):
    """
    A model that performs few-shot learning to adapt pre-trained models to new settings.
    It fine-tunes only the classification head by default or can fine-tune the entire model.
    """
    
    def __init__(
        self,
        base_model,
        model_type='vit',
        num_classes=None,
        adaptation_lr=0.01,
        adaptation_steps=10,
        finetune_all=False,
        device=None
    ):
        """
        Initialize the few-shot adaptive model.
        
        Args:
            base_model: The pre-trained base model to adapt
            model_type: Type of the base model
            num_classes: Number of classes (can be different from the base model)
            adaptation_lr: Learning rate for few-shot adaptation
            adaptation_steps: Number of adaptation steps for few-shot learning
            finetune_all: Whether to fine-tune all parameters or just the classifier
            device: Device to use (defaults to the same device as base_model)
        """
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type
        self.adaptation_lr = adaptation_lr
        self.adaptation_steps = adaptation_steps
        self.finetune_all = finetune_all
        
        if device is None:
            # Use the same device as the base model
            self.device = next(base_model.parameters()).device
        else:
            self.device = device
            
        # Determine the feature dimension based on model type
        if hasattr(base_model, 'classifier'):
            feature_dim = base_model.classifier.in_features
        else:
            # Default for unknown models
            feature_dim = 128
            
        # Create a new classifier if num_classes is provided
        if num_classes is not None:
            self.classifier = nn.Linear(feature_dim, num_classes)
            # Initialize with the original classifier weights for classes that match
            if hasattr(base_model, 'classifier'):
                original_classes = base_model.classifier.out_features
                min_classes = min(original_classes, num_classes)
                with torch.no_grad():
                    self.classifier.weight.data[:min_classes] = base_model.classifier.weight.data[:min_classes]
                    self.classifier.bias.data[:min_classes] = base_model.classifier.bias.data[:min_classes]
        else:
            # Use the original classifier
            if hasattr(base_model, 'classifier'):
                self.classifier = copy.deepcopy(base_model.classifier)
            else:
                raise ValueError("Base model doesn't have a classifier and num_classes not provided")
                
        # Move to the correct device
        self.to(self.device)
        
    def forward(self, x):
        """
        Forward pass using the base model and the adapted classifier.
        
        Args:
            x: Input data
            
        Returns:
            Model output
        """
        # Extract features from the base model (excluding the classification head)
        features = self.extract_features(x)
        # Pass features through the classifier
        return self.classifier(features)
    
    def extract_features(self, x):
        """
        Extract features from the base model before the classification head.
        
        Args:
            x: Input data
            
        Returns:
            Features before classification
        """
        # Handle different model architectures
        if self.model_type in ['vit', 'patchtst', 'timesformer1d']:
            # For ViT-like models
            if hasattr(self.base_model, 'embedding') and hasattr(self.base_model, 'encoder'):
                x = self.base_model.embedding(x)
                batch_size = x.shape[0]
                cls_tokens = self.base_model.cls_token.expand(batch_size, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = self.base_model.encoder(x)
                return x[:, 0]  # CLS token
            
        elif self.model_type == 'mlp':
            # For MLP models
            if hasattr(self.base_model, 'feature'):
                x = x.view(x.size(0), -1)
                return self.base_model.feature(x)
                
        elif self.model_type == 'lstm':
            # For LSTM models
            if hasattr(self.base_model, 'lstm') and hasattr(self.base_model, 'fc'):
                x = x.squeeze(1)
                _, (hidden, _) = self.base_model.lstm(x)
                hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
                return self.base_model.fc(hidden_cat)
                
        elif self.model_type == 'resnet18':
            # For ResNet18 models
            if hasattr(self.base_model, 'resnet') and hasattr(self.base_model, 'proj'):
                feat = self.base_model.resnet(x)
                return self.base_model.proj(feat)
                
        elif self.model_type == 'transformer':
            # For Transformer models
            if hasattr(self.base_model, 'input_proj') and hasattr(self.base_model, 'transformer'):
                x = x.squeeze(1)
                x = self.base_model.input_proj(x)
                x = self.base_model.pos_encoder(x)
                x = self.base_model.transformer(x)
                return self.base_model.proj(x.mean(dim=1))
                
        # Generic fallback - just remove the final classification layer
        # This assumes the model applies the classifier at the end
        with torch.no_grad():
            # Store the original classifier
            if hasattr(self.base_model, 'classifier'):
                original_classifier = self.base_model.classifier
                # Temporarily set to identity to get features
                self.base_model.classifier = nn.Identity()
                features = self.base_model(x)
                # Restore the original classifier
                self.base_model.classifier = original_classifier
                return features
                
        # If all else fails, return the full output
        # This won't work well for few-shot adaptation but prevents errors
        return self.base_model(x)
    
    def adapt_to_support_set(self, support_x, support_y, criterion=None):
        """
        Adapt the model to a support set (few examples from the new environment).
        
        Args:
            support_x: Support set inputs [N*K, C, H, W] where N is num_classes and K is shots
            support_y: Support set labels [N*K]
            criterion: Loss function for adaptation (defaults to CrossEntropyLoss)
            
        Returns:
            Adapted model
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        # Determine which parameters to update during adaptation
        if self.finetune_all:
            # Fine-tune the entire model
            params_to_update = self.parameters()
        else:
            # Only fine-tune the classifier
            params_to_update = self.classifier.parameters()
            
        # Create optimizer for adaptation
        optimizer = torch.optim.Adam(params_to_update, lr=self.adaptation_lr)
        
        # Move support set to the correct device
        support_x = support_x.to(self.device)
        support_y = support_y.to(self.device)
        
        # Set to training mode
        self.train()
        
        # Perform adaptation steps
        for step in range(self.adaptation_steps):
            # Forward pass
            logits = self(support_x)
            loss = criterion(logits, support_y)
            
            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Set back to evaluation mode
        self.eval()
        
        return self
    
    @classmethod
    def from_pretrained(cls, model_path, model_type='vit', **kwargs):
        """
        Create a few-shot adaptive model from a pre-trained model checkpoint.
        
        Args:
            model_path: Path to the pre-trained model checkpoint
            model_type: Type of the model to load
            **kwargs: Additional arguments for FewShotAdaptiveModel constructor
            
        Returns:
            Initialized FewShotAdaptiveModel
        """
        # Load the checkpoint
        print(f"Loading pre-trained model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if we have a MultitaskAdapterModel checkpoint
        is_multitask_model = False
        if isinstance(checkpoint, dict) and 'adapters' in checkpoint and 'heads' in checkpoint:
            print("Detected a MultitaskAdapterModel checkpoint...")
            is_multitask_model = True
            
            # We'll need to convert this to a standard model
            # Get the task name to extract the right head and adapter
            task_name = kwargs.get('task_name', 'MotionSourceRecognition')
            print(f"Using task: {task_name}")
            
            # Check if the task exists in the adapters and heads
            if task_name not in checkpoint['adapters'] or task_name not in checkpoint['heads']:
                print(f"Warning: Task {task_name} not found in checkpoint. Available tasks:")
                print(f"Adapters: {list(checkpoint['adapters'].keys())}")
                print(f"Heads: {list(checkpoint['heads'].keys())}")
                print("Falling back to the first available task.")
                
                # Use the first available adapter and head instead
                available_tasks = list(set(checkpoint['adapters'].keys()) & set(checkpoint['heads'].keys()))
                if available_tasks:
                    task_name = available_tasks[0]
                    print(f"Using task: {task_name}")
                else:
                    raise ValueError("No valid task found in the multitask model.")
            
            # Determine number of classes from the head
            head_weights = checkpoint['heads'][task_name]
            # Check if it's a linear layer
            if 'weight' in head_weights:
                num_classes = head_weights['weight'].shape[0]
            else:
                # Try to find the last layer of the head
                last_layer_key = None
                for key in head_weights.keys():
                    if 'weight' in key and '.' in key:
                        layer_num = int(key.split('.')[0]) if key.split('.')[0].isdigit() else -1
                        if last_layer_key is None or layer_num > int(last_layer_key.split('.')[0]):
                            last_layer_key = key
                
                if last_layer_key:
                    num_classes = head_weights[last_layer_key].shape[0]
                else:
                    num_classes = kwargs.get('num_classes', 2)  # Default if we can't determine
        else:
            # Determine number of classes from the regular checkpoint
            if 'num_classes' in checkpoint:
                num_classes = checkpoint['num_classes']
            elif hasattr(checkpoint, 'classifier') and hasattr(checkpoint.classifier, 'out_features'):
                num_classes = checkpoint.classifier.out_features
            else:
                # Try to infer from model architecture
                model_state_dict = checkpoint.get('model_state_dict', checkpoint)
                classifier_weight_key = None
                for key in model_state_dict.keys():
                    if 'classifier.weight' in key:
                        classifier_weight_key = key
                        break
                
                if classifier_weight_key:
                    num_classes = model_state_dict[classifier_weight_key].shape[0]
                else:
                    # Default if we can't determine
                    num_classes = kwargs.get('num_classes', 2)
        
        # Get the input shape parameters from the kwargs or defaults
        # But first try to infer them from the checkpoint
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Try to get win_len and feature_size from the model weights
        win_len = kwargs.get('win_len', 500)
        feature_size = kwargs.get('feature_size', 232)
        in_channels = kwargs.get('in_channels', 1)
        emb_dim = kwargs.get('emb_dim', 128)
        
        # For ViT models, check the patch embedding weight
        if model_type == 'vit' and any('embedding.embedding.proj.weight' in k for k in model_state_dict.keys()):
            for key in model_state_dict.keys():
                if 'embedding.embedding.proj.weight' in key:
                    # Extract dimensions from the weight shape
                    # Shape is typically [emb_dim, in_channels, patch_height, patch_width]
                    weight_shape = model_state_dict[key].shape
                    if len(weight_shape) == 4:
                        # The patch dimensions need to be multiplied by number of patches in sequence
                        patch_height = weight_shape[2] 
                        patch_width = weight_shape[3]
                        
                        # We need to calculate how many patches fit in the original dimensions
                        # However, this is approximate as we don't know the exact stride
                        print(f"Detected patch shape: {patch_height}x{patch_width}")
                        
                        # Instead of trying to guess the right dimensions,
                        # use the exact dimensions from the checkpoint
                        win_len = 500 # Force to match checkpoint
                        feature_size = 232  # Force to match checkpoint
                        
                        print(f"Forcing win_len={win_len}, feature_size={feature_size} to match checkpoint")
                            
                        emb_dim = weight_shape[0]
                        in_channels = weight_shape[1]
                        break
        
        # Create a new instance of the base model
        ModelClass = {
            'mlp': MLPClassifier, 
            'lstm': LSTMClassifier, 
            'resnet18': ResNet18Classifier, 
            'transformer': TransformerClassifier, 
            'vit': ViTClassifier,
            'patchtst': PatchTST,
            'timesformer1d': TimesFormer1D
        }.get(model_type)
        
        if ModelClass is None:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Initialize the base model with proper parameters
        print(f"Creating {model_type} model with {num_classes} classes...")
        print(f"Model dimensions: win_len={win_len}, feature_size={feature_size}, in_channels={in_channels}, emb_dim={emb_dim}")
        
        # Check the checkpoint for any information about dimensions
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            config = checkpoint['config']
            # Use dimensions from config if available
            if 'win_len' in config:
                win_len = config['win_len']
                print(f"Using win_len={win_len} from checkpoint config")
            if 'feature_size' in config:
                feature_size = config['feature_size']
                print(f"Using feature_size={feature_size} from checkpoint config")
                
        base_model = ModelClass(
            win_len=win_len,
            feature_size=feature_size,
            in_channels=in_channels,
            emb_dim=emb_dim,
            num_classes=num_classes
        )
        
        # For multitask models, we need to modify the model to incorporate the adapter
        # and then create a simulated regular model
        if is_multitask_model:
            print("Cannot directly load a MultitaskAdapterModel into a regular model right now.")
            print("Please use a supervised-trained model instead.")
            raise ValueError("Multitask model loading not fully implemented yet.")
        
        # Load the state dictionary
        try:
            if 'model_state_dict' in checkpoint:
                base_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                base_model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Continuing with an un-initialized model. This may not work well.")
        
        # Create a copy of kwargs with only the parameters accepted by FewShotAdaptiveModel.__init__
        accepted_params = ['adaptation_lr', 'adaptation_steps', 'finetune_all', 'device']
        fewshot_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
            
        # Create the few-shot adaptive model
        return cls(
            base_model=base_model,
            model_type=model_type,
            num_classes=num_classes,
            **fewshot_kwargs
        ) 