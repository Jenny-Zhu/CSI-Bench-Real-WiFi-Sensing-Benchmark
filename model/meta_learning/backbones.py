import torch
import torch.nn as nn
from model.supervised.models import (
    MLPClassifier, LSTMClassifier, ResNet18Classifier, TransformerClassifier
)

class MLPBackbone(nn.Module):
    def __init__(self, win_len=250, feature_size=98, emb_dim=128, **kwargs):
        super().__init__()
        # Use the MLP up to the penultimate layer for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(win_len * feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.feature_extractor(x)

class LSTMBackbone(nn.Module):
    def __init__(self, feature_size=98, hidden_size=128, num_layers=2, emb_dim=128, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, emb_dim)

    def forward(self, x):
        x = x.squeeze(1)
        _, (hidden, _) = self.lstm(x)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden_cat)

class ResNet18Backbone(nn.Module):
    def __init__(self, win_len=250, feature_size=98, emb_dim=128, **kwargs):
        super().__init__()
        from torchvision.models import resnet18
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the final fc, add a new projection
        self.resnet.fc = nn.Identity()
        self.proj = nn.Linear(512, emb_dim)

    def forward(self, x):
        features = self.resnet(x)
        return self.proj(features)

class TransformerBackbone(nn.Module):
    def __init__(self, feature_size=98, d_model=128, nhead=8, num_layers=4, emb_dim=128, dropout=0.1, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(feature_size, d_model)
        self.pos_encoder = TransformerClassifier.PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, emb_dim)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.proj(x)