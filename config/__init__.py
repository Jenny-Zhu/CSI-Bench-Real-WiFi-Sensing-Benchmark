from config.base_config import BaseConfig
from config.pretraining_config import PretrainingConfig
from config.supervised_config import SupervisedConfig
from config.meta_learning_config import MetaLearningConfig

def get_config(args):
    """Factory function to create the appropriate config based on args.mode
    
    Args:
        args: Command line arguments namespace
        
    Returns:
        config: The appropriate configuration object instance
    """
    if args.mode == "pretraining":
        return PretrainingConfig(args)
    elif args.mode == "supervised":
        return SupervisedConfig(args)
    elif args.mode == "meta":
        return MetaLearningConfig(args)
    else:
        raise ValueError(f"Unsupported training mode: {args.mode}")