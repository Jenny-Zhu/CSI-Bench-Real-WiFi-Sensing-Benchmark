#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from load.supervised.benchmark_loader import load_benchmark_supervised
from model.multitask.models import MultiTaskAdapterModel
from model.supervised.models import TransformerClassifier
from scripts.train_supervised import MODEL_TYPES

def debug_multitask(task_name='MotionSourceRecognition', batch_size=32):
    print(f"Debug multitask learning with task: {task_name}")
    
    # Set data directory
    data_dir = 'wifi_benchmark_dataset'
    
    # Step 1: Load data
    print(f"\nStep 1: Loading data from {data_dir}...")
    try:
        data = load_benchmark_supervised(
            dataset_root=data_dir,
            task_name=task_name,
            batch_size=batch_size
        )
        print(f"Data loaded successfully!")
        print(f"Number of classes: {data['num_classes']}")
        print(f"Loaders available: {list(data['loaders'].keys())}")
        
        # Check train loader
        train_loader = data['loaders']['train']
        print(f"Training samples: {len(train_loader.dataset)}")
        
        # Get a batch to check the shape
        sample_x, sample_y = next(iter(train_loader))
        print(f"Sample batch shape: {sample_x.shape}, Labels shape: {sample_y.shape}")
        feature_size = sample_x.shape[-1]
        win_len = sample_x.shape[-2]
        print(f"Feature size: {feature_size}, Window length: {win_len}")
        
        # Check labels
        labels = sample_y.numpy()
        unique_labels = np.unique(labels)
        print(f"Unique labels in batch: {unique_labels}")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Step 2: Define model parameters
    print("\nStep 2: Setting up model parameters...")
    
    # Model parameters
    model_type = 'transformer'
    emb_dim = 128
    dropout = 0.1
    task_classes = {task_name: data['num_classes']}
    
    print(f"Model type: {model_type}")
    print(f"Embedding dimension: {emb_dim}")
    print(f"Dropout: {dropout}")
    print(f"Task classes: {task_classes}")
    
    # Step 3: Build the model
    print("\nStep 3: Building the model...")
    try:
        # Create backbone model
        ModelClass = MODEL_TYPES[model_type]
        model_kwargs = {
            'num_classes': data['num_classes'],
            'feature_size': feature_size,
            'd_model': emb_dim,
            'dropout': dropout
        }
        
        print(f"Creating {model_type} backbone with parameters: {model_kwargs}")
        backbone_model = ModelClass(**model_kwargs)
        print(f"Backbone created successfully!")
        
        # Extract transformer backbone
        class TransformerEmbedding(nn.Module):
            def __init__(self, cls_model):
                super().__init__()
                self.input_proj = cls_model.input_proj
                self.pos_encoder = cls_model.pos_encoder
                self.transformer = cls_model.transformer

            def forward(self, x):
                x = x.squeeze(1)
                x = self.input_proj(x)
                x = self.pos_encoder(x)
                x = self.transformer(x)
                return x.mean(dim=1)

        backbone = TransformerEmbedding(backbone_model)
        print("TransformerEmbedding created successfully!")
        
        # Attach config with both dict and attribute access
        class ConfigDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(f"No such attribute: {name}")

        # Get transformer layers
        if hasattr(backbone_model.transformer, 'layers'):
            num_layers = len(backbone_model.transformer.layers)
        else:
            # For PyTorch's TransformerEncoder
            num_layers = backbone_model.transformer.num_layers
        
        backbone.config = ConfigDict(
            model_type="bert",
            hidden_size=emb_dim,
            num_attention_heads=8,
            num_hidden_layers=num_layers,
            intermediate_size=emb_dim * 4,
            hidden_dropout_prob=dropout,
            use_return_dict=False
        )
        print(f"Backbone config attached: {backbone.config}")
        
        # Create multitask model
        model = MultiTaskAdapterModel(
            backbone, 
            task_classes,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.05
        )
        print("MultiTaskAdapterModel created successfully!")
        
        # Set active task
        model.set_active_task(task_name)
        print(f"Active task set to: {task_name}")
        
        # Test forward pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        sample_x = sample_x.to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(sample_x)
            print(f"Forward pass successful! Output shape: {outputs.shape}")
        
    except Exception as e:
        print(f"Error building model: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nDebug completed successfully!")
    print("All components working as expected.")

if __name__ == "__main__":
    debug_multitask() 