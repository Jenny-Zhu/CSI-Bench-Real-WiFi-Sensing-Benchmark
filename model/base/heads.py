import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    """Classification head for supervised learning"""
    
    def __init__(self, in_features, num_classes, hidden_dim=None, dropout=0.0):
        super().__init__()
        
        if hidden_dim is not None:
            self.fc = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Feature vector [B, in_features]
            
        Returns:
            Classification logits [B, num_classes]
        """
        return self.fc(x)

class ContrastiveHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, in_features, out_features, hidden_dim=None, temperature=0.1):
        super().__init__()
        
        self.temperature = temperature
        
        if hidden_dim is not None:
            self.projector = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.projector = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x1, x2=None):
        """
        Args:
            x1: Features from first view [B, in_features]
            x2: Features from second view [B, in_features], optional
            
        Returns:
            If x2 is provided, returns contrastive loss; otherwise, returns projected features
        """
        z1 = F.normalize(self.projector(x1), dim=1)
        
        if x2 is None:
            return z1
        
        z2 = F.normalize(self.projector(x2), dim=1)
        
        # Compute contrastive loss
        batch_size = z1.size(0)
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Elements on diagonal should have high similarity
        labels = torch.arange(batch_size, device=z1.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss, z1, z2

class ReconstructionHead(nn.Module):
    """Decoder head for reconstructing input"""
    
    def __init__(self, in_features, img_size, channels=1):
        super().__init__()
        
        self.img_size = img_size  # (H, W)
        self.channels = channels
        
        # Calculate intermediate dimensions for decoder
        hidden_dim = in_features * 2
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels * img_size[0] * img_size[1])
        )
    
    def forward(self, x):
        """
        Args:
            x: Feature vector [B, in_features]
            
        Returns:
            Reconstructed image [B, C, H, W]
        """
        B = x.size(0)
        # Decode and reshape to image shape
        x = self.decoder(x)
        x = x.view(B, self.channels, self.img_size[0], self.img_size[1])
        return x
