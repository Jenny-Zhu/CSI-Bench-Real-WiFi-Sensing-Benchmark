import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class MLPClassifier(nn.Module):
    """Multi-layer Perceptron for WiFi sensing"""
    def __init__(self, win_len=250, feature_size=98, num_classes=2):
        super(MLPClassifier, self).__init__()
        input_size = win_len * feature_size
        
        self.fc = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Flatten input: [batch, channels, win_len, feature_size] -> [batch, win_len*feature_size]
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LSTMClassifier(nn.Module):
    """LSTM model for WiFi sensing"""
    def __init__(self, feature_size=98, hidden_size=256, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        # Input shape: [batch, channels, win_len, feature_size]
        # LSTM expects: [batch, win_len, feature_size]
        x = x.squeeze(1)  # Remove channel dimension
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the final hidden state from both directions
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Classification
        out = self.fc(hidden_cat)
        return out

class ResNet18Classifier(nn.Module):
    """Modified ResNet-18 for WiFi sensing"""
    def __init__(self, win_len=250, feature_size=98, num_classes=2):
        super(ResNet18Classifier, self).__init__()
        
        # Load pretrained ResNet-18
        self.resnet = resnet18(pretrained=False)
        
        # Modify first conv layer to accept single channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final fc layer
        self.resnet.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # ResNet forward pass
        return self.resnet(x)

class TransformerClassifier(nn.Module):
    """Transformer model for WiFi sensing"""
    def __init__(self, feature_size=98, d_model=256, nhead=8, 
                 num_layers=4, dropout=0.1, num_classes=2):
        super(TransformerClassifier, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(feature_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes)
        )
        
    def forward(self, x):
        # Input shape: [batch, channels, win_len, feature_size]
        # Transform to: [batch, win_len, feature_size]
        x = x.squeeze(1)
        
        # Project to d_model dimensions
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Global average pooling over sequence length
        x = x.mean(dim=1)
        
        # Classification
        return self.classifier(x)

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
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