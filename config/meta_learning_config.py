from config.base_config import BaseConfig

class MetaLearningConfig(BaseConfig):
    """Configuration for meta-learning training"""
    
    def __init__(self, args):
        """Initialize meta-learning configuration
        
        Args:
            args: Command line arguments
        """
        # Initialize base configuration
        super().__init__(args)
        
        # Meta-learning specific settings
        self.meta_method = getattr(args, 'meta_method', 'maml')
        self.k_shot = getattr(args, 'k_shot', 5)
        self.q_query = getattr(args, 'q_query', 15)
        self.steps = getattr(args, 'steps', 100)
        self.tasks_per_batch = getattr(args, 'tasks_per_batch', 4)
        
        # MAML-specific settings
        self.inner_lr = getattr(args, 'inner_lr', 0.01)
        self.meta_lr = getattr(args, 'meta_lr', 0.001)
        
        # Task settings
        self.task_name = getattr(args, 'task_name', 'goodbad')
        self.label_keywords = {}  # Will be populated based on task_name
        
        # Set label keywords based on task
        if self.task_name == 'goodbad':
            self.label_keywords = {'good': 0, 'bad': 1}
        elif self.task_name == 'motionempty':
            self.label_keywords = {'empty': 0, 'motion': 1}
        
        # Input size settings
        self.resize_height = getattr(args, 'resize_height', 64)
        self.resize_width = getattr(args, 'resize_width', 100)
        
        # Model settings
        self.model_dims = {}  # Will be populated during training once we know actual dimensions
