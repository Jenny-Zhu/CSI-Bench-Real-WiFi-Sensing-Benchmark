import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class LSTMOptimizer(nn.Module):
    """
    LSTM-based meta-optimizer for few-shot learning.
    This optimizer uses an LSTM to learn how to update model parameters.
    
    Based on "Optimization as a Model for Few-Shot Learning"
    (Ravi & Larochelle, 2017)
    """
    
    def __init__(self, input_size, hidden_size=20, num_layers=1):
        """
        Initialize the LSTM optimizer.
        
        Args:
            input_size: Size of input features (gradient dimensions)
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
        """
        super(LSTMOptimizer, self).__init__()
        
        # LSTM for parameter updates
        self.lstm = nn.LSTM(
            input_size=input_size * 2,  # Gradient and parameter
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # Linear layer to produce update
        self.update_layer = nn.Linear(hidden_size, input_size)
        
        # Hidden state for LSTM
        self.hidden = None
    
    def reset_state(self, batch_size=1):
        """
        Reset the LSTM hidden state.
        
        Args:
            batch_size: Batch size for hidden state
        """
        self.hidden = None
    
    def forward(self, gradient, parameter):
        """
        Compute parameter update based on gradient and current parameter.
        
        Args:
            gradient: Parameter gradient
            parameter: Current parameter value
            
        Returns:
            Parameter update value
        """
        # Reshape inputs for LSTM
        batch_size = gradient.size(0)
        
        # Concatenate gradient and parameter
        lstm_input = torch.cat([gradient, parameter], dim=1)
        
        # Add sequence dimension
        lstm_input = lstm_input.unsqueeze(0)  # [1, batch_size, input_size*2]
        
        # Initialize hidden state if necessary
        if self.hidden is None:
            h0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=gradient.device)
            c0 = torch.zeros(1, batch_size, self.lstm.hidden_size, device=gradient.device)
            self.hidden = (h0, c0)
        
        # Forward through LSTM
        lstm_out, self.hidden = self.lstm(lstm_input, self.hidden)
        
        # Generate update
        update = self.update_layer(lstm_out.squeeze(0))
        
        return update

    def initialize_weights(self):
        """Initialize LSTM optimizer weights with appropriate scaling"""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Initialize update layer
        nn.init.xavier_uniform_(self.update_layer.weight, gain=0.01)
        nn.init.constant_(self.update_layer.bias, 0.0)

class CoordinateWiseLSTMOptimizer(nn.Module):
    """
    Coordinate-wise LSTM optimizer that applies the same LSTM cell
    to each parameter, treating each as an independent optimization problem.
    """
    
    def __init__(self, hidden_size=20, num_layers=1):
        """
        Initialize the coordinate-wise LSTM optimizer.
        
        Args:
            hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
        """
        super(CoordinateWiseLSTMOptimizer, self).__init__()
        
        # LSTM for parameter updates
        self.lstm = nn.LSTM(
            input_size=2,  # Gradient and parameter for a single coordinate
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        # Linear layer to produce update
        self.update_layer = nn.Linear(hidden_size, 1)
        
        # Whether to use coordinate-wise hidden states
        self.coord_wise_hidden = True
        self.hidden_dict = {}
    
    def reset_state(self):
        """Reset all LSTM hidden states"""
        self.hidden_dict = {}
    
    def forward(self, parameters, gradients):
        """
        Compute parameter updates based on gradients and current parameters.
        
        Args:
            parameters: Dictionary of current parameters
            gradients: Dictionary of parameter gradients
            
        Returns:
            Dictionary of parameter updates
        """
        updates = {}
        
        # Process each parameter separately
        for name, param in parameters.items():
            if name in gradients and gradients[name] is not None:
                grad = gradients[name]
                
                # Process each element in the parameter tensor independently
                param_flat = param.view(-1)
                grad_flat = grad.view(-1)
                update_flat = torch.zeros_like(param_flat)
                
                for i in range(len(param_flat)):
                    # Get the coordinate-wise hidden state or initialize it
                    hidden_key = f"{name}_{i}"
                    if hidden_key not in self.hidden_dict or self.hidden_dict[hidden_key] is None:
                        h0 = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=param.device)
                        c0 = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=param.device)
                        self.hidden_dict[hidden_key] = (h0, c0)
                    
                    # Prepare input for this coordinate [gradient, parameter]
                    coord_input = torch.stack([grad_flat[i], param_flat[i]]).view(1, 1, 2)
                    
                    # Forward through LSTM
                    lstm_out, self.hidden_dict[hidden_key] = self.lstm(
                        coord_input, 
                        self.hidden_dict[hidden_key]
                    )
                    
                    # Generate update
                    update = self.update_layer(lstm_out.squeeze())
                    update_flat[i] = update.item()
                
                # Reshape update back to parameter shape
                updates[name] = update_flat.view_as(param)
            else:
                updates[name] = torch.zeros_like(param)
        
        return updates
