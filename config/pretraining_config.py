from config.base_config import BaseConfig

class PretrainingConfig(BaseConfig):
    """Configuration for self-supervised pretraining"""
    
    def __init__(self, args):
        """Initialize pretraining configuration
        
        Args:
            args: Command line arguments
        """
        # Initialize base configuration
        super().__init__(args)
        
        # Model settings
        self.emb_size = getattr(args, 'emb_size', 128)        # Embedding size
        self.depth = getattr(args, 'depth', 6)                # Model depth
        self.in_channels = getattr(args, 'in_channels', 1)    # Input channels
        self.freq_out = getattr(args, 'freq_out', 10)         # Frequency output dimension
        
        # Training settings
        self.learning_rate = getattr(args, 'learning_rate', 1e-5)
        self.weight_decay = getattr(args, 'decay_rate', 1e-4)
        self.num_epochs = getattr(args, 'num_epochs', 100)
        self.patience = getattr(args, 'patience', 20)
        
        # CSI-specific settings
        if self.data_type == 'csi':
            self.time_seg = getattr(args, 'time_seg', 5)
            self.sample_rate = getattr(args, 'sample_rate', 100)
        
        # ACF-specific settings
        elif self.data_type == 'acf':
            self.win_len = getattr(args, 'win_len', 250)
            self.feature_size = getattr(args, 'feature_size', 98)
        
        # Data augmentation settings
        self.mask_ratio = getattr(args, 'mask_ratio', 0.1)    # Masking ratio for augmentation
        self.row_mask_ratio = getattr(args, 'row_mask_ratio', 0.1)  # Row masking ratio
        self.col_mask_ratio = getattr(args, 'col_mask_ratio', 0.1)  # Column masking ratio
        
        # Loss weights
        self.contrastive_weight = getattr(args, 'contrastive_weight', 1.0)
        self.reconstruction_weight = getattr(args, 'reconstruction_weight', 1.0)