# Export trainers
from engine.supervised import TaskTrainer

# Factory function to get the appropriate trainer
def get_trainer(model, data_loader, config):
    """Factory function to create the appropriate trainer
    
    Args:
        model: The model to train
        data_loader: Data loader for training
        config: Configuration object
        
    Returns:
        trainer: An instance of the appropriate trainer class
    """
    if config.mode == "supervised":
        from engine.supervised import TaskTrainer
        return TaskTrainer(model, data_loader, config)
    
    raise ValueError(f"Unsupported training mode: {config.mode}")
