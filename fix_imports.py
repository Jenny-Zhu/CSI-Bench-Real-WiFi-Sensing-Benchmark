import os

def create_or_update_init(directory, content=None):
    """Create or update an __init__.py file in the specified directory"""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist, creating it...")
        os.makedirs(directory, exist_ok=True)
    
    init_file = os.path.join(directory, '__init__.py')
    
    if content is None:
        content = '# Module initialization\n'
    
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Created/updated {init_file}")

def main():
    # Fix load module structure
    create_or_update_init('load', '# Load module\n')
    
    # Fix load.supervised module
    create_or_update_init('load/supervised', '''# Supervised learning data loading utilities
# Import only what's needed to avoid circular imports
''')
    
    # Fix load.meta_learning module
    create_or_update_init('load/meta_learning', '''# Meta-learning data loading utilities
# Import only what's needed to avoid circular imports
''')
    
    # Fix engine module structure
    create_or_update_init('engine', '# Engine module for training and evaluation\n')
    
    # Fix engine.supervised module
    create_or_update_init('engine/supervised', '''# Supervised learning training utilities
# Import necessary classes and functions
''')
    
    # Fix engine.meta_learning module
    create_or_update_init('engine/meta_learning', '''# Meta-learning training utilities
# Import necessary classes and functions
''')
    
    # Fix model module structure
    create_or_update_init('model', '# Model definitions module\n')
    
    # Fix model.meta_learning module
    create_or_update_init('model/meta_learning', '''# Meta-learning model definitions
# Import necessary classes and functions
''')
    
    # Create a standalone module for meta-learning to avoid import issues
    os.makedirs('meta_learning_standalone', exist_ok=True)
    create_or_update_init('meta_learning_standalone', '''# Standalone meta-learning module
# This module contains all the necessary classes and functions for meta-learning
# without relying on the rest of the codebase to avoid import issues
''')
    
    # Create or update meta_learning_standalone wrappers
    with open('meta_learning_standalone/data_loader.py', 'w', encoding='utf-8') as f:
        f.write('''"""
Standalone data loader for meta-learning.
This module wraps the functionality from meta_learning_data.py without circular imports.
"""
# Import directly from meta_learning_data.py
from meta_learning_data import (
    load_meta_learning_tasks,
    load_csi_data_benchmark,
    MetaTaskSampler
)
''')
    
    with open('meta_learning_standalone/models.py', 'w', encoding='utf-8') as f:
        f.write('''"""
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
''')
    
    with open('meta_learning_standalone/trainer.py', 'w', encoding='utf-8') as f:
        f.write('''"""
Standalone trainer for meta-learning.
This module provides training utilities without circular imports.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

# Define the MAMLTrainer class directly here
class MAMLTrainer:
    def __init__(self, model, device, inner_lr=0.01, meta_lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, task_batch):
        """Perform a single meta-training step"""
        # Implementation moved to train_meta_standalone.py
        pass
    
    def evaluate(self, data_loader, num_adaptation_steps=5):
        """Evaluate the model on the given data loader"""
        # Implementation moved to train_meta_standalone.py
        pass
''')
    
    print("\nFixed import structure in key modules.")
    print("Now run fix_all_null_bytes.py to remove null bytes from the files.")
    print("Then you can use the train_meta_standalone.py script to train without import issues.")

if __name__ == "__main__":
    main() 