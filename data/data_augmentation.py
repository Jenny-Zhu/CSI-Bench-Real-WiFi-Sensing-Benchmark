import numpy as np
import random
import torch
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomApply, ColorJitter, RandomGrayscale, GaussianBlur, ToTensor, Normalize
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision.transforms as transforms

class DataAugmentation:
    def __init__(self, device, level=0.01, modulation_factor=0.1, max_shift=0.1, temporal_max_shift=5,
                 frequency_max_perturbation=0.05):
        self.level = level
        self.modulation_factor = modulation_factor
        self.max_shift = max_shift
        self.temporal_max_shift = temporal_max_shift
        self.frequency_max_perturbation = frequency_max_perturbation
        self.device = device  # Specify device to create tensors directly on the right device

    def add_noise(self, data):
        noise = torch.randn(data.size(), device=self.device) * self.level
        return data + noise

    def signal_modulation(self, data):
        modulation = 1 + torch.randn(data.size(0), *data.size()[2:-1], 1, device=self.device) * self.modulation_factor
        data[:, 0, ...] *= modulation
        return data

    def phase_shift(self, data):
        shift = torch.randn(data.size(0), *data.size()[2:-1], 1, device=self.device) * self.max_shift
        data[:, 1, ...] += shift
        return data

    def temporal_shift(self, data):
        for i in range(data.size(0)):
            shift = torch.randint(-self.temporal_max_shift, self.temporal_max_shift + 1, (1,)).item()
            data[i] = torch.roll(data[i], shifts=shift, dims=-1)
        return data

    def frequency_perturbation(self, data):
        perturbation_factor = 1 + torch.randn(data.size(), device=self.device) * self.frequency_max_perturbation
        # Apply FFT, modify, and then apply inverse FFT
        for ch in range(data.size(1)):
            data_fft = torch.fft.fft(data[:, ch, ...], dim=3)
            data[:, ch, ...] = torch.fft.ifft(data_fft * perturbation_factor[:, ch, ...], dim=3).real
        return data
        # data_fft = torch.fft.fft(data, dim=3)
        # modified_fft = data_fft * perturbation_factor
        # data_ifft = torch.fft.ifft(modified_fft, dim=3).real
        # return data_ifft

    def apply_augmentations(self, data):
        data = self.add_noise(data)
        data = self.signal_modulation(data)
        data = self.phase_shift(data)
        data = self.temporal_shift(data)
        data = self.frequency_perturbation(data)
        return data


class DataAugmentationCPU:
    def __init__(self, level=0.01, modulation_factor=0.1, max_shift=0.1, temporal_max_shift=5,
                frequency_max_perturbation=0.05):
        self.level = level
        self.modulation_factor = modulation_factor
        self.max_shift = max_shift
        self.temporal_max_shift = temporal_max_shift
        self.frequency_max_perturbation = frequency_max_perturbation

    def add_noise(self, data):
        noise = np.random.normal(0, self.level, data.shape)
        return data + noise

    def signal_modulation(self, data):
        """
        Apply amplitude modulation only to the amplitude channel.
        """
        # Create modulation factor, should match the exact shape except the last dimension (time) which should be 1
        modulation = 1 + self.modulation_factor * np.random.randn(*data.shape[0:1], *data.shape[2:-1], 1)
        # Apply the modulation factor only to the amplitude channel
        data[:, 0, ...] *= modulation
        return data

    def phase_shift(self, data):
        """
        Apply a random phase shift only to the phase channel.
        """
        # Create a shift factor that matches the data dimensions except for the last one (time)
        shift = self.max_shift * np.random.randn(*data.shape[0:1], *data.shape[2:-1], 1)
        # Apply the shift only to the phase channel
        data[:, 1, ...] += shift
        return data

    def temporal_shift(self, data):
        for i in range(data.shape[0]):
            shift = np.random.randint(-self.temporal_max_shift, self.temporal_max_shift)
            data[i] = np.roll(data[i], shift, axis=-1)
        return data

    def frequency_perturbation(self, data):
        """
        Apply frequency perturbation to both amplitude and phase channels.
        """
        # Create a perturbation factor that matches the full dimensions of the data, specifically for the subcarriers
        perturbation_factor = 1 + self.frequency_max_perturbation * np.random.randn(*data.shape)

        # Apply the perturbation to all channels using the Fourier transform
        for ch in range(data.shape[1]):  # Loop over channels
            # Perform the Fourier transform, modify the frequency content, and perform the inverse Fourier transform
            data[:, ch, ...] = np.fft.ifft(np.fft.fft(data[:, ch, ...], axis=3) * perturbation_factor[:, ch, ...],
                                        axis=3).real
        return data

    def apply_augmentations(self, data):
        data = self.add_noise(data)
        data = self.signal_modulation(data)
        data = self.phase_shift(data)
        data = self.temporal_shift(data)
        data = self.frequency_perturbation(data)
        return data

# ### Test
# # Example usage
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# augmentor = DataAugmentation()
# data = np.random.randn(10, 2, 4, 4, 30, 100)  # Simulated example data
#
# # Original data stats
# print("Original Data Stats:")
# print("Mean:", np.mean(data))
# print("Std Dev:", np.std(data))
#
# # Augmented data
# augmented_data = augmentor.apply_augmentations(data)
#
# # Augmented data stats
# print("Augmented Data Stats:")
# print("Mean:", np.mean(augmented_data))
# print("Std Dev:", np.std(augmented_data))

class TensorToPILTransform:
    """Convert a tensor to a PIL Image."""
    def __call__(self, tensor):
        return to_pil_image(tensor)

class GaussianNoiseTransform:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # Generate noise on the same device as the input tensor
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        return tensor + noise

class NormalizeSample:
    def __call__(self, tensor):
        # Normalize each sample
        mean = tensor.mean(dim=(1, 2), keepdim=True)
        std = tensor.std(dim=(1, 2), keepdim=True)
        return (tensor - mean) / (std + 1e-6)

class RandomMaskTransform:
    def __init__(self, block_size, probability=0.5):
        """
        Args:
            block_size (int): Size of the square block to mask.
            probability (float): Probability with which to apply the mask.
        """
        self.block_size = block_size
        self.probability = probability

    def __call__(self, tensor):
        if random.random() < self.probability:
            # Generate random coordinates for the top-left corner of the block
            x = random.randint(0, tensor.size(1) - self.block_size)
            y = random.randint(0, tensor.size(2) - self.block_size)
            tensor[:, x:x+self.block_size, y:y+self.block_size] = 0
        return tensor


class AddRandomDCTermPerFeature:
    def __init__(self, min_value, max_value, num_features):
        """
        Args:
            min_value (float): The minimum value of the DC term to be added.
            max_value (float): The maximum value of the DC term to be added.
            num_features (int): The number of features in the data.
        """
        self.min_value = min_value
        self.max_value = max_value
        self.num_features = num_features

    def __call__(self, tensor):
        # Generate a random DC term for each feature
        dc_values = torch.FloatTensor(self.num_features).uniform_(self.min_value, self.max_value)
        dc_values = dc_values.view(1, 1, self.num_features).expand_as(tensor)  # Match the shape to input tensor
        return tensor + dc_values


class ShuffleFeatures:
    def __init__(self, num_features):
        """
        Args:
            num_features (int): The number of features in the data.
        """
        self.num_features = num_features

    def __call__(self, tensor):
        # Generate a random permutation of feature indices
        indices = torch.randperm(self.num_features)
        # Apply this permutation to the feature dimension
        return tensor[:, :, indices]


class DataAugmentACF:
    def __init__(self):
        # No need to pass feature_size during initialization
        pass

    def get_augmentation(self, feature_size):
        """Creates an augmentation pipeline using the computed feature_size."""
        return Compose([
            # Convert tensor to PIL to use certain transformations
            TensorToPILTransform(),

            # SimCLR transformations
            RandomHorizontalFlip(),
            RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            RandomGrayscale(p=0.2),

            # Convert back to tensor
            ToTensor(),

            # Custom transformations that work on tensors
            ShuffleFeatures(num_features=feature_size),
            RandomApply([GaussianNoiseTransform(std=0.1)], p=0.3),
            AddRandomDCTermPerFeature(min_value=-10, max_value=10, num_features=feature_size),
            RandomMaskTransform(block_size=10, probability=0.5),  # Apply random masking
            NormalizeSample()
        ])

    def __call__(self, data):
        # Calculate the feature size from the last dimension of the input data
        feature_size = data.shape[-1]

        # Build the pipeline using the computed feature_size
        augmentation_pipeline = self.get_augmentation(feature_size)

        # Apply the augmentation pipeline to the data and return the result
        return augmentation_pipeline(data)
