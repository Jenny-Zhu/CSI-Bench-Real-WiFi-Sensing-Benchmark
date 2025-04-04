import torch
import torch.nn.functional as F

def evaluate(fmodel, x, y):
    logits = fmodel(x)
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == y).float().mean().item()
    return acc

def preprocess_csi(raw_signal):
    # Placeholder for real preprocessing
    return raw_signal
