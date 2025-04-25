# Export trainers
from engine.supervised import TaskTrainer, TaskTrainerACF 
from engine.meta_learning import MetaTrainer

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
        if config.data_type == "csi":
            from engine.supervised import TaskTrainer
            return TaskTrainer(model, data_loader, config)
        elif config.data_type == "acf":
            from engine.supervised import TaskTrainerACF
            return TaskTrainerACF(model, data_loader, config)
    
    elif config.mode == "meta":
        from engine.meta_learning import MetaTrainer
        return MetaTrainer(model, data_loader, config)
    
    raise ValueError(f"Unsupported training mode: {config.mode}")
