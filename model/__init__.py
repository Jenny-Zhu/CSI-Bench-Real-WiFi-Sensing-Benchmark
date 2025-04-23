# Export main model classes
from .unified.vit_model import ViTUnified
from .unified.cnn_model import CNNUnified

# For backward compatibility - original model classes
from .pretraining.ssl_model import ViT_Parallel, ViT_MultiTask
from .meta_learning.meta_model import CSI2DCNN, CSITransformer

# Convenience factory function - get appropriate model based on task and data type
def get_model(task_type, data_type, **kwargs):
    """
    Factory function to get appropriate model based on task and data type
    
    Args:
        task_type (str): 'pretraining', 'supervised', or 'meta_learning'
        data_type (str): 'csi' or 'acf'
        **kwargs: Other parameters for the model
        
    Returns:
        Appropriate model instance
    """
    if task_type == 'pretraining':
        if kwargs.get('joint', False):
            from .pretraining.ssl_model import JointSSLModel
            return JointSSLModel(data_type=data_type, **kwargs)
        else:
            from .pretraining.ssl_model import SSLModel
            return SSLModel(data_type=data_type, **kwargs)
    
    elif task_type == 'supervised':
        if data_type == 'csi':
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
    
    # Return unified model as fallback
    return ViTUnified(task_type=task_type, data_type=data_type, **kwargs)
