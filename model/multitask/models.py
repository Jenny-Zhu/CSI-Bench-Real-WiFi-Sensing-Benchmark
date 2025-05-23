import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional, Union
import math
from peft import get_peft_model, LoraConfig

class AdapterWrapper(nn.Module):
    """Wraps a PEFT model so forward(x) calls the pure-PyTorch backbone."""
    def __init__(self, peft_model):
        super().__init__()
        self.peft = peft_model

    def forward(self, x):
        # Directly delegate to the underlying backbone,
        # bypassing the HF-style input_ids signature.
        return self.peft.base_model(x)

class LoRALayer(nn.Module):
    """
    LoRA adapter layer for efficient fine-tuning
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        r: int = 8, 
        lora_alpha: int = 32, 
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        
        # LoRA components
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        
        # Scaling factor
        self.scaling = lora_alpha / r
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class TaskAdapter(nn.Module):
    """
    Task-specific adapter module
    """
    def __init__(
        self, 
        d_model: int, 
        dropout: float = 0.1,
        bottleneck_size: int = 64
    ):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_size)
        self.non_linear = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, hidden_states):
        residual = hidden_states
        x = self.down_proj(hidden_states)
        x = self.non_linear(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = x + residual
        x = self.layer_norm(x)
        return x


class TaskAdapters(nn.Module):
    """
    Collection of task-specific adapters
    """
    def __init__(
        self, 
        backbone, 
        task_names: List[str],
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.backbone = backbone
        self.task_names = task_names
        self.active_task = None
        
        hidden_size = backbone.config.hidden_size
        
        # Create LoRA adapters for each task
        self.loras = nn.ModuleDict()
        
        # For each task, create adapters for each layer
        for task in task_names:
            task_loras = nn.ModuleDict()
            
            # Apply LoRA to self-attention modules in each transformer layer
            for i in range(backbone.config.num_hidden_layers):
                layer_loras = nn.ModuleDict({
                    "q": LoRALayer(
                        hidden_size, hidden_size, 
                        r=lora_r, lora_alpha=lora_alpha, 
                        lora_dropout=lora_dropout
                    ),
                    "k": LoRALayer(
                        hidden_size, hidden_size,
                        r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout
                    ),
                    "v": LoRALayer(
                        hidden_size, hidden_size,
                        r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout
                    ),
                    "o": LoRALayer(
                        hidden_size, hidden_size,
                        r=lora_r, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout
                    )
                })
                task_loras[f"layer_{i}"] = layer_loras
                
            # Add final adapter after the transformer stack
            task_loras["output"] = TaskAdapter(hidden_size)
            
            self.loras[task] = task_loras
    
    def set_active_task(self, task_name: str):
        """Set the active task for inference"""
        if task_name not in self.task_names:
            raise ValueError(f"Task {task_name} not found in available tasks: {self.task_names}")
        self.active_task = task_name
    
    def get_active_adapters(self):
        """Get active task adapters"""
        if self.active_task is None:
            raise ValueError("No active task set. Call set_active_task first.")
        return self.loras[self.active_task]
    
    def apply_adapters(self, hidden_states):
        """Apply the active task's output adapter"""
        if self.active_task is None:
            raise ValueError("No active task set. Call set_active_task first.")
        
        return self.loras[self.active_task]["output"](hidden_states)
        
    def forward(self, x):
        """Apply backbone with LoRA adaptations for active task"""
        if self.active_task is None:
            raise ValueError("No active task set. Call set_active_task first.")
            
        # Forward through backbone
        hidden_states = self.backbone(x)
        
        # Apply final adapter
        output = self.apply_adapters(hidden_states)
        
        return output


class MultiTaskAdapterModel(nn.Module):
    """
    Multi-task model with task-specific adapters and heads
    """
    def __init__(
        self, 
        backbone, 
        task_classes: Dict[str, int],
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.backbone = backbone
        self.task_names = list(task_classes.keys())
        self.active_task = None
        
        # Task-specific adapters
        self.adapters = TaskAdapters(
            backbone, 
            self.task_names,
            lora_r=lora_r,
                lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # Task-specific classification heads
        hidden_size = backbone.config.hidden_size
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, num_classes)
            for task, num_classes in task_classes.items()
        })
        
        # Initialize heads
        for head in self.heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
    
    def set_active_task(self, task_name: str):
        """Set the active task for inference"""
        if task_name not in self.task_names:
            raise ValueError(f"Task {task_name} not found in available tasks: {self.task_names}")
        self.active_task = task_name
        self.adapters.set_active_task(task_name)
    
    def forward(self, x):
        """Forward pass for the active task"""
        if self.active_task is None:
            raise ValueError("No active task set. Call set_active_task first.")
        
        # Get adapted features
        features = self.adapters(x)
        
        # Apply task-specific head
        logits = self.heads[self.active_task](features)
        
        return logits


class PatchTSTTaskAdapters(nn.Module):
    """
    Collection of task-specific adapters for PatchTST model
    """
    def __init__(
        self, 
        backbone, 
        task_names: List[str],
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.backbone = backbone
        self.task_names = task_names
        self.active_task = None
        
        # Get hidden size from the backbone
        emb_dim = backbone.config.hidden_size
        
        # Create task-specific adapters
        self.task_adapters = nn.ModuleDict()
        
        for task in task_names:
            # Create adapter layers for each transformer block
            adapter_layers = nn.ModuleList()
            
            # For each transformer block in the backbone
            for i in range(backbone.config.num_hidden_layers):
                # Create an adapter for this layer
                adapter_block = nn.ModuleDict({
                    "attn_q": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    "attn_k": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    "attn_v": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    "attn_out": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    "mlp": TaskAdapter(emb_dim, dropout=lora_dropout)
                })
                adapter_layers.append(adapter_block)
            
            # Final output adapter for this task
            final_adapter = TaskAdapter(emb_dim, dropout=lora_dropout)
            
            # Store all adapters for this task as a ModuleDict
            task_module_dict = nn.ModuleDict()
            task_module_dict.add_module("output", final_adapter)
            task_module_dict.add_module("layers", adapter_layers)
            
            self.task_adapters[task] = task_module_dict
    
    def set_active_task(self, task_name: str):
        """Set the active task for inference"""
        if task_name not in self.task_names:
            raise ValueError(f"Task {task_name} not found in available tasks: {self.task_names}")
        self.active_task = task_name
    
    def forward(self, x):
        """Apply backbone and task-specific adapters"""
        if self.active_task is None:
            raise ValueError("No active task set. Call set_active_task first.")
        
        # Get active task adapters
        adapters = self.task_adapters[self.active_task]
        
        # Forward through backbone
        hidden_states = self.backbone(x)
        
        # Apply final adapter
        output = adapters["output"](hidden_states)
        
        return output


class PatchTSTAdapterModel(nn.Module):
    """
    Multi-task model with PatchTST backbone and task-specific adapters and heads
    """
    def __init__(
        self, 
        backbone, 
        task_classes: Dict[str, int],
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.backbone = backbone
        self.task_names = list(task_classes.keys())
        self.active_task = None
        
        # Prepare backbone config if it doesn't exist
        if not hasattr(backbone, 'config'):
            # Create config with necessary attributes
            class ConfigDict(dict):
                def __getattr__(self, name):
                    try:
                        return self[name]
                    except KeyError:
                        raise AttributeError(f"No such attribute: {name}")
            
            # Extract attributes from the transformer blocks
            num_layers = len(backbone.transformer.layers)
            emb_dim = backbone.emb_dim if hasattr(backbone, 'emb_dim') else backbone.transformer.layers[0].self_attn.embed_dim
            
            backbone.config = ConfigDict(
                hidden_size=emb_dim,
                num_hidden_layers=num_layers
            )
        
        # Task-specific adapters
        self.adapters = PatchTSTTaskAdapters(
            backbone, 
            self.task_names,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # Task-specific classification heads
        hidden_size = backbone.config.hidden_size
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, num_classes)
            for task, num_classes in task_classes.items()
        })
        
        # Initialize heads
        for head in self.heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
    
    def set_active_task(self, task_name: str):
        """Set the active task for inference"""
        if task_name not in self.task_names:
            raise ValueError(f"Task {task_name} not found in available tasks: {self.task_names}")
        self.active_task = task_name
        self.adapters.set_active_task(task_name)
    
    def forward(self, x):
        """Forward pass for the active task"""
        if self.active_task is None:
            raise ValueError("No active task set. Call set_active_task first.")
        
        # Get adapted features
        features = self.adapters(x)
        
        # Apply task-specific head
        logits = self.heads[self.active_task](features)
        
        return logits


class TimesFormerTaskAdapters(nn.Module):
    """
    Collection of task-specific adapters for TimesFormer-1D model
    """
    def __init__(
        self, 
        backbone, 
        task_names: List[str],
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.backbone = backbone
        self.task_names = task_names
        self.active_task = None
        
        # Get hidden size from the backbone
        emb_dim = backbone.config.hidden_size
        
        # Create task-specific adapters
        self.task_adapters = nn.ModuleDict()
        
        for task in task_names:
            # Create adapter layers for each transformer block
            adapter_layers = nn.ModuleList()
            
            # For each transformer block in the backbone
            for i in range(backbone.config.num_hidden_layers):
                # Create separate adapters for temporal and feature attention
                adapter_block = nn.ModuleDict({
                    # Temporal attention adapters
                    "temporal_q": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    "temporal_k": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    "temporal_v": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    "temporal_out": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    
                    # Feature attention adapters
                    "feature_q": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    "feature_k": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    "feature_v": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    "feature_out": LoRALayer(emb_dim, emb_dim, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                    
                    # MLP adapter
                    "mlp": TaskAdapter(emb_dim, dropout=lora_dropout)
        })
                adapter_layers.append(adapter_block)
            
            # Final output adapter for this task
            final_adapter = TaskAdapter(emb_dim, dropout=lora_dropout)
            
            # Store all adapters for this task as a ModuleDict
            task_module_dict = nn.ModuleDict()
            task_module_dict.add_module("output", final_adapter)
            task_module_dict.add_module("layers", adapter_layers)
            
            self.task_adapters[task] = task_module_dict
    
    def set_active_task(self, task_name: str):
        """Set the active task for inference"""
        if task_name not in self.task_names:
            raise ValueError(f"Task {task_name} not found in available tasks: {self.task_names}")
        self.active_task = task_name
    
    def forward(self, x):
        """Apply backbone and task-specific adapters"""
        if self.active_task is None:
            raise ValueError("No active task set. Call set_active_task first.")
        
        # Get active task adapters
        adapters = self.task_adapters[self.active_task]
        
        # Forward through backbone
        hidden_states = self.backbone(x)
        
        # Apply final adapter
        output = adapters["output"](hidden_states)
        
        return output


class TimesFormerAdapterModel(nn.Module):
    """
    Multi-task model with TimesFormer-1D backbone and task-specific adapters and heads
    """
    def __init__(
        self, 
        backbone, 
        task_classes: Dict[str, int],
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.backbone = backbone
        self.task_names = list(task_classes.keys())
        self.active_task = None

        # Prepare backbone config if it doesn't exist
        if not hasattr(backbone, 'config'):
            # Create config with necessary attributes
            class ConfigDict(dict):
                def __getattr__(self, name):
                    try:
                        return self[name]
                    except KeyError:
                        raise AttributeError(f"No such attribute: {name}")
            
            # Extract attributes from the transformer blocks
            num_layers = len(backbone.blocks)
            emb_dim = backbone.emb_dim if hasattr(backbone, 'emb_dim') else backbone.blocks[0].norm1.normalized_shape[0]
            
            backbone.config = ConfigDict(
                hidden_size=emb_dim,
                num_hidden_layers=num_layers
            )
        
        # Task-specific adapters
        self.adapters = TimesFormerTaskAdapters(
            backbone, 
            self.task_names,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        
        # Task-specific classification heads
        hidden_size = backbone.config.hidden_size
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, num_classes)
            for task, num_classes in task_classes.items()
        })
        
        # Initialize heads
        for head in self.heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
    
    def set_active_task(self, task_name: str):
        """Set the active task for inference"""
        if task_name not in self.task_names:
            raise ValueError(f"Task {task_name} not found in available tasks: {self.task_names}")
        self.active_task = task_name
        self.adapters.set_active_task(task_name)

    def forward(self, x):
        """Forward pass for the active task"""
        if self.active_task is None:
            raise ValueError("No active task set. Call set_active_task first.")
        
        # Get adapted features
        features = self.adapters(x)
        
        # Apply task-specific head
        logits = self.heads[self.active_task](features)
        
        return logits