import torch
import numpy as np
import random
from data.augmentation.base_augmentation import BaseAugmentation

class DataAugmentation(BaseAugmentation):
    """Data augmentation techniques for CSI data."""
    
    def __init__(self, device=None, level=0.01, modulation_factor=0.1, max_shift=0.1, 
                 temporal_max_shift=5, frequency_max_perturbation=0.05):
        """Initialize the CSI data augmentation.
        
        Args:
            device: The device to use for tensor operations.
            level: The noise level to add.
            modulation_factor: The factor for signal modulation.
            max_shift: The maximum phase shift.
            temporal_max_shift: The maximum temporal shift.
            frequency_max_perturbation: The maximum frequency perturbation.
        """
        super().__init__(device)
        self.level = level
        self.modulation_factor = modulation_factor
        self.max_shift = max_shift
        self.temporal_max_shift = temporal_max_shift
        self.frequency_max_perturbation = frequency_max_perturbation
    
    def add_noise(self, data):
        """Add Gaussian noise to the data.
        
        Args:
            data: The data to add noise to.
            
        Returns:
            The data with added noise.
        """
        noise = torch.randn(data.size(), device=self.device) * self.level
        return data + noise
    
    def signal_modulation(self, data):
        """Apply signal modulation to the data.
        
        Args:
            data: The data to modulate.
            
        Returns:
            The modulated data.
        """
        modulation = 1 + torch.randn(data.size(0), *data.size()[2:-1], 1, device=self.device) * self.modulation_factor
        data[:, 0, ...] *= modulation
        return data
    
    def phase_shift(self, data):
        """Apply phase shift to the data.
        
        Args:
            data: The data to phase shift.
            
        Returns:
            The phase-shifted data.
        """
        shift = torch.randn(data.size(0), *data.size()[2:-1], 1, device=self.device) * self.max_shift
        data[:, 1, ...] += shift
        return data
    
    def temporal_shift(self, data):
        """Apply temporal shift to the data.
        
        Args:
            data: The data to temporally shift.
            
        Returns:
            The temporally shifted data.
        """
        for i in range(data.size(0)):
            shift = torch.randint(-self.temporal_max_shift, self.temporal_max_shift + 1, (1,)).item()
            data[i] = torch.roll(data[i], shifts=shift, dims=-1)
        return data
    
    def frequency_perturbation(self, data):
        """Apply frequency perturbation to the data.
        
        Args:
            data: The data to perturb.
            
        Returns:
            The perturbed data.
        """
        perturbation_factor = 1 + torch.randn(data.size(), device=self.device) * self.frequency_max_perturbation
        for ch in range(data.size(1)):
            data_fft = torch.fft.fft(data[:, ch, ...], dim=3)
            data[:, ch, ...] = torch.fft.ifft(data_fft * perturbation_factor[:, ch, ...], dim=3).real
        return data
    
    def apply_augmentations(self, data):
        """Apply all augmentations to the data.
        
        Args:
            data: The data to augment.
            
        Returns:
            The augmented data.
        """
        data = self.add_noise(data)
        data = self.signal_modulation(data)
        data = self.phase_shift(data)
        data = self.temporal_shift(data)
        data = self.frequency_perturbation(data)
        return data
