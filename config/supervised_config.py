from config.base_config import BaseConfig

class SupervisedConfig(BaseConfig):
    """Configuration for supervised training"""
    
    def __init__(self, args):
        """Initialize supervised training configuration
        
        Args:
            args: Command line arguments
        """
        # Initialize base configuration
        super().__init__(args)
        
        # Model settings
        self.emb_size = getattr(args, 'emb_size', 128)
        self.depth = getattr(args, 'depth', 6)
        self.in_channels = getattr(args, 'in_channels', 1)
        
        # Training settings
        self.learning_rate = getattr(args, 'learning_rate', 1e-4)
        self.weight_decay = getattr(args, 'weight_decay', 1e-4)
        self.num_epochs = getattr(args, 'num_epochs', 100)
        self.patience = getattr(args, 'patience', 15)
        
        # Pretrained model settings
        self.pretrained = getattr(args, 'pretrained', False)
        self.pretrained_path = getattr(args, 'pretrained_path', None)
        self.freeze_backbone = getattr(args, 'freeze_backbone', False)
        
        # Data settings
        self.train_ratio = getattr(args, 'train_ratio', 0.8)
        self.val_ratio = getattr(args, 'val_ratio', 0.1)
        self.test_ratio = getattr(args, 'test_ratio', 0.1)
        
        # Task-specific settings
        self.task_name = getattr(args, 'task_name', 'default')
        
        # Set number of classes based on task
        self.num_classes = 2  # Default binary classification
        if self.task_name == 'HumanNonhuman':
            self.num_classes = 2
        elif self.task_name == 'FourClass':
            self.num_classes = 4
        elif self.task_name == 'NTUHumanID':
            self.num_classes = 15
        elif self.task_name == 'HumanMotion':
            self.num_classes = 3
        elif self.task_name == 'ThreeClass':
            self.num_classes = 3
        elif self.task_name == 'DetectionandClassification':
            self.num_classes = 5
        elif self.task_name == 'Detection':
            self.num_classes = 2
        
        # Input size settings
        if hasattr(args, 'resize_height') and hasattr(args, 'resize_width'):
            self.resize_height = args.resize_height
            self.resize_width = args.resize_width