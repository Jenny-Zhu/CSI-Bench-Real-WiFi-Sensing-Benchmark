import torch
import torch.nn as nn
from model.supervised import ViT_Parallel

# Constants or default scaling factors for patch sizes
PATCH_W_SCALE = 10
PATCH_H_SCALE = 2

def fine_tune_model(model, freeze_up_to_layer=2):
    """
    Freezes the first 'freeze_up_to_layer' blocks in model.encoder.
    For deeper Transformer with .layers,
    each block is enumerated and param.require_grad is set accordingly.
    
    Args:
        model (nn.Module): Model to freeze layers
        freeze_up_to_layer (int): Number of layers to freeze
        
    Returns:
        model (nn.Module): Model with frozen layers
    """
    # 检查模型类型并执行相应的冻结操作
    if hasattr(model, 'model') and hasattr(model.model, 'backbone'):
        # 新版本的ViT_Parallel
        for i, block in enumerate(model.model.backbone.layers):
            if i < freeze_up_to_layer:
                for param in block.parameters():
                    param.requires_grad = False
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        # 标准Transformer模型
        for i, block in enumerate(model.encoder.layers):
            if i < freeze_up_to_layer:
                for param in block.parameters():
                    param.requires_grad = False
    # 可以添加其他模型类型的冻结逻辑
    return model

def load_model_trained(checkpoint_path, model_name, task, win_len=250, feature_size=98, in_channels=1):
    """
    Loads a trained model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model_name (str): Name of the model architecture
        task (str): Name of the task
        win_len (int): Window length
        feature_size (int): Feature size
        in_channels (int): Number of input channels
        
    Returns:
        Model with loaded weights
    """
    # 获取类别数量
    classes = {
        'HumanNonhuman': 2, 
        'FourClass': 4, 
        'NTUHumanID': 15, 
        'NTUHAR': 6, 
        'HumanID': 4, 
        'Widar': 22,
        'HumanMotion': 3, 
        'ThreeClass': 3, 
        'DetectionandClassification': 5, 
        'Detection': 2
    }
    
    # 确保任务在支持列表中
    if task not in classes:
        raise ValueError(f"Task {task} not in supported task list: {list(classes.keys())}")
    
    # 创建模型
    model = load_model_scratch(model_name=model_name, task=task, win_len=win_len, feature_size=feature_size, in_channels=in_channels)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    
    return model

def load_model_scratch(model_name='ViT', task='ThreeClass', win_len=250, feature_size=98, in_channels=1):
    """
    Creates a new model from scratch based on the specified model name.
    
    Args:
        model_name (str): Name of the model architecture ('ViT' or other models that will be implemented)
        task (str): Name of the task
        win_len (int): Window length
        feature_size (int): Feature size
        in_channels (int): Number of input channels
        
    Returns:
        New model instance
    """
    # 支持的类别映射
    classes = {
        'HumanNonhuman': 2, 
        'FourClass': 4, 
        'NTUHumanID': 15, 
        'NTUHAR': 6, 
        'HumanID': 4, 
        'Widar': 22,
        'HumanMotion': 3, 
        'ThreeClass': 3, 
        'DetectionandClassification': 5, 
        'Detection': 2
    }
    
    # 确保任务在支持列表中
    if task not in classes:
        raise ValueError(f"Task {task} not in supported task list: {list(classes.keys())}")
    
    num_classes = classes[task]
    
    # 计算ViT的embedding size
    # emb_size = int((win_len / PATCH_W_SCALE) * (feature_size / PATCH_H_SCALE))
    emb_size = 128
    
    # 根据指定的模型名称创建相应的模型
    if model_name == 'ViT':
        print(f"Using model: ViT with emb_size={emb_size}, win_len={win_len}, feature_size={feature_size}")
        model = ViT_Parallel(
            win_len=win_len,
            feature_size=feature_size,
            emb_dim=emb_size,
            in_channels=in_channels,
            proj_dim=emb_size,
            num_classes=num_classes
        )
    # 预留其他模型的导入和实例化，这些模型将在model/文件夹中逐步实现
    elif model_name == 'MLP':
        from .models import MLPClassifier
        model = MLPClassifier(
            win_len=win_len,
            feature_size=feature_size,
            num_classes=num_classes
        )
    
    elif model_name == 'LSTM':
        from .models import LSTMClassifier
        model = LSTMClassifier(
            feature_size=feature_size,
            num_classes=num_classes
        )
    
    elif model_name == 'ResNet18':
        from .models import ResNet18Classifier
        model = ResNet18Classifier(
            win_len=win_len,
            feature_size=feature_size,
            num_classes=num_classes
        )
    
    elif model_name == 'Transformer':
        from .models import TransformerClassifier
        model = TransformerClassifier(
            feature_size=feature_size,
            num_classes=num_classes
        )
    
    else:
        supported_models = ['ViT', 'MLP', 'LeNet', 'ResNet18', 'LSTM', 'BiLSTM', 'GRUNet', 'Transformer']
        raise ValueError(f"Model {model_name} not in supported model list: {supported_models}")
    
    return model
