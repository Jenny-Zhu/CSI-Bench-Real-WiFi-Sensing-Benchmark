# Export main model classes
from .supervised.vit import ViT_Parallel
from .meta_learning.meta_model import CSI2DCNN, CSITransformer

# For supervised learning
from .supervised.vit import ViT_Parallel
from .meta_learning.meta_model import CSI2DCNN, CSITransformer

# Convenience factory function - get appropriate model based on task and data type
def get_model(task_type, data_type, **kwargs):
    """
    Factory function to get appropriate model based on task and data type
    
    Args:
        task_type (str): 'supervised' or 'meta_learning'
        data_type (str): 'csi' or 'acf'
        **kwargs: Other parameters for the model
        
    Returns:
        Appropriate model instance
    """
    if task_type == 'supervised':
        if kwargs.get('model_name', '').lower() == 'vit':
            # Use ViT model directly
            return ViT_Parallel(**kwargs)
        elif data_type == 'csi':
            from .supervised.classifier import CSIClassifier
            return CSIClassifier(**kwargs)
        else:  # 'acf'
            from .supervised.classifier import ACFClassifier
            return ACFClassifier(**kwargs)
    
    elif task_type == 'meta_learning':
        if data_type == 'csi':
            from .meta_learning.meta_model import CSIMetaModel
            return CSIMetaModel(**kwargs)
        else:  # 'acf'
            from .meta_learning.meta_model import ACFMetaModel
            return ACFMetaModel(**kwargs)
    
    # 如果没有匹配的模型类型，抛出错误
    raise ValueError(f"Unsupported combination of task_type={task_type} and data_type={data_type}")
