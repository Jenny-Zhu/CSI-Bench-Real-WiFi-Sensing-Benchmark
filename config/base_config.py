import os
import torch
import json
from datetime import datetime

class BaseConfig:
    """Base configuration class that all specific configs inherit from.
    Contains shared settings and utility methods.
    """
    
    def __init__(self, args):
        """Initialize configuration from command line args
        
        Args:
            args: Command line arguments namespace
        """
        # Basic settings
        self.mode = args.mode                    # Training mode
        self.data_type = args.data_type          # Data type: 'csi' or 'acf'
        self.model_type = args.model_type        # Model type: 'cnn', 'vit', 'transformer'
        
        # Data settings
        self.data_dir = args.data_dir            # List of data directories
        self.batch_size = getattr(args, 'batch_size', 16)  # Batch size
        
        # Device settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Output settings
        self.output_dir = getattr(args, 'output_dir', 'output')  # Output directory
        self.model_name = getattr(args, 'model_name', 'model')   # Model name
        
        # Create a unique run identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.mode}_{self.data_type}_{self.model_type}_{timestamp}"
        
        # Create output directories
        self.run_dir = os.path.join(self.output_dir, self.run_id)
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        self.log_dir = os.path.join(self.run_dir, 'logs')
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def save(self):
        """Save configuration to a JSON file in the run directory"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and k != 'device'}
        
        config_path = os.path.join(self.run_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        return config_path
    
    @classmethod
    def load(cls, config_path):
        """Load configuration from a JSON file
        
        Args:
            config_path: Path to the config JSON file
            
        Returns:
            config: Configuration object
        """
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create a minimal args object
        class Args:
            pass
        
        args = Args()
        args.mode = config_dict.get('mode', 'pretraining')
        args.data_type = config_dict.get('data_type', 'csi')
        args.model_type = config_dict.get('model_type', 'vit')
        
        # Create config object
        config = cls(args)
        
        # Update all attributes
        for key, value in config_dict.items():
            setattr(config, key, value)
        
        return config
    
    def __str__(self):
        """String representation of the configuration"""
        attrs = [f"  {k}={v}" for k, v in self.__dict__.items() 
                if not k.startswith('_') and k != 'device']
        return f"{self.__class__.__name__}(\n" + "\n".join(attrs) + "\n)"