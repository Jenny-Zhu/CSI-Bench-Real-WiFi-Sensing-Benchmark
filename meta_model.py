import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torchvision.models import resnet18

class MLPClassifier(nn.Module):
    def __init__(self, win_len=232, feature_size=500, num_classes=2, emb_dim=128):
        super().__init__()
        input_size = win_len * feature_size
        self.feature = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, emb_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # Handle both single examples and batches
        if len(x.shape) > 3:  # [batch_size, channels, height, width]
            x = x.view(x.size(0), -1)
        else:  # [channels, height, width]
            x = x.view(1, -1)
        
        feat = self.feature(x)
        return self.classifier(feat)
    
    def get_features(self, x):
        if len(x.shape) > 3:  # [batch_size, channels, height, width]
            x = x.view(x.size(0), -1)
        else:  # [channels, height, width]
            x = x.view(1, -1)
        return self.feature(x)

class LSTMClassifier(nn.Module):
    def __init__(self, feature_size=500, hidden_size=128, num_layers=2, num_classes=2, emb_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # Handle different input formats: [batch, channels, seq_len, features] or [batch, seq_len, features]
        if len(x.shape) == 4:
            x = x.squeeze(1)  # Remove channel dimension
        
        # Run LSTM
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        feat = self.fc(hidden_cat)
        return self.classifier(feat)
    
    def get_features(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(1)
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden_cat)

class ResNet18Classifier(nn.Module):
    def __init__(self, win_len=232, feature_size=500, num_classes=2, emb_dim=128):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        # Modify first layer to accept 1 channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()
        self.proj = nn.Linear(512, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # Ensure input is 4D [batch_size, channels, height, width]
        if len(x.shape) == 3:  # [channels, height, width]
            x = x.unsqueeze(0)  # Add batch dimension
        elif len(x.shape) == 2:  # [height, width]
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Ensure channel dimension is present and correct
        if x.shape[1] != 1:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Forward through ResNet
        feat = self.resnet(x)
        feat = self.proj(feat)
        return self.classifier(feat)
    
    def get_features(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        
        if x.shape[1] != 1:
            x = x.unsqueeze(1)
            
        feat = self.resnet(x)
        return self.proj(feat)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, feature_size=500, d_model=128, nhead=8, num_layers=4, num_classes=2, emb_dim=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(feature_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        
    def forward(self, x):
        # Handle different input formats: [batch, channels, seq_len, features] or [batch, seq_len, features]
        if len(x.shape) == 4:
            x = x.squeeze(1)  # Remove channel dimension
        elif len(x.shape) == 3 and x.shape[0] == 1:
            # If it's [channels, seq_len, features]
            x = x.squeeze(0)  # Remove channel dimension
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Project to model dimension
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        feat = self.proj(x.mean(dim=1))  # Global average pooling
        return self.classifier(feat)
    
    def get_features(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(1)
        elif len(x.shape) == 3 and x.shape[0] == 1:
            x = x.squeeze(0).unsqueeze(0)
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.proj(x.mean(dim=1))

class ViTClassifier(nn.Module):
    def __init__(self, win_len=232, feature_size=500, in_channels=1, emb_dim=128, 
                 depth=6, num_heads=4, mlp_ratio=4.0, dropout=0.1, num_classes=2):
        super().__init__()
        # Calculate patch size - aim for ~100 patches
        patch_h = max(1, feature_size // 10)
        patch_w = max(1, win_len // 10)
        
        # Calculate number of patches
        self.num_patches = (feature_size // patch_h) * (win_len // patch_w)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, emb_dim, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        
        # Transformer layers
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, 
                                       dim_feedforward=int(emb_dim * mlp_ratio),
                                       dropout=dropout, batch_first=True)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        
        # Initialize position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        # Ensure input is 4D [batch_size, channels, height, width]
        if len(x.shape) == 3:  # [channels, height, width]
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Ensure channel dimension is present
        if x.shape[1] != 1:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Create patches
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_dim]
        
        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add position embeddings
        if x.size(1) != self.pos_embed.size(1):
            # Handle different sequence lengths with interpolation
            pos_embed = self.pos_embed.transpose(1, 2)
            pos_embed = F.interpolate(pos_embed, size=x.size(1), mode='linear')
            pos_embed = pos_embed.transpose(1, 2)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification from CLS token
        x = self.norm(x)
        return self.classifier(x[:, 0])
    
    def get_features(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        if x.shape[1] != 1:
            x = x.unsqueeze(1)
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed.transpose(1, 2)
            pos_embed = F.interpolate(pos_embed, size=x.size(1), mode='linear')
            pos_embed = pos_embed.transpose(1, 2)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0]  # Return CLS token features

class BaseMetaModel(nn.Module):
    def __init__(self,
                 model_type='vit',  # 'mlp', 'lstm', 'resnet18', 'transformer', 'vit'
                 win_len=232,
                 feature_size=500,
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
                emb_dim=emb_dim,
                dropout=dropout
            )
        elif model_type == 'vit':
            self.model = ViTClassifier(
                win_len=win_len,
                feature_size=feature_size,
                in_channels=in_channels,
                emb_dim=emb_dim,
                num_classes=num_classes,
                dropout=dropout
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
        if hasattr(self.model, 'get_features'):
            return self.model.get_features(x)
        else:
            return None 