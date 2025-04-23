import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from scipy import signal

def check_sample_rate(data, expected_rate=None, plot=False, window_size=1000):
    """
    Check the effective sample rate of the data by analyzing frequency content.
    
    Args:
        data (ndarray): Input data array
        expected_rate (float, optional): Expected sample rate in Hz
        plot (bool): Whether to plot the frequency spectrum
        window_size (int): Window size for spectral analysis
        
    Returns:
        float: Estimated sample rate
    """
    # If data is multi-dimensional, use the first channel/dimension
    if data.ndim > 1:
        data_1d = data[0] if data.ndim == 2 else data[0, 0]
    else:
        data_1d = data
    
    # Compute power spectral density
    f, Pxx = signal.welch(data_1d, fs=1.0, nperseg=window_size)
    
    # Find dominant frequency
    peak_idx = np.argmax(Pxx)
    dominant_freq = f[peak_idx]
    
    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 6))
        plt.semilogy(f, Pxx)
        plt.xlabel('Normalized Frequency (cycles/sample)')
        plt.ylabel('Power Spectral Density')
        plt.title('Power Spectrum')
        plt.axvline(x=dominant_freq, color='r', linestyle='--', 
                    label=f'Dominant Freq: {dominant_freq:.4f}')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Compare with expected rate if provided
    if expected_rate is not None:
        print(f"Expected sample rate: {expected_rate} Hz")
        print(f"Dominant frequency: {dominant_freq} (normalized)")
        
        # If too different, warn
        if abs(dominant_freq - expected_rate) / expected_rate > 0.1:  # >10% difference
            print("WARNING: Dominant frequency differs significantly from expected rate")
    
    return dominant_freq

def adjust_sample_rate(data, current_rate, target_rate):
    """
    Resample data to adjust the sample rate.
    
    Args:
        data (ndarray): Input data array
        current_rate (float): Current sample rate
        target_rate (float): Target sample rate
        
    Returns:
        ndarray: Resampled data
    """
    # Calculate resampling factor
    factor = target_rate / current_rate
    
    # Handle multi-dimensional data
    original_shape = data.shape
    
    if data.ndim == 1:
        # For 1D data, simple resampling
        return signal.resample(data, int(len(data) * factor))
    else:
        # For multi-dimensional data, reshape and process each channel
        reshaped = False
        if data.ndim > 2:
            # Reshape to 2D: (channels, samples)
            original_shape = data.shape
            data = data.reshape(-1, original_shape[-1])
            reshaped = True
        
        # Process each channel
        num_channels = data.shape[0]
        num_samples = data.shape[1]
        resampled_length = int(num_samples * factor)
        
        resampled_data = np.zeros((num_channels, resampled_length))
        for i in range(num_channels):
            resampled_data[i] = signal.resample(data[i], resampled_length)
        
        # Reshape back if needed
        if reshaped:
            new_shape = list(original_shape)
            new_shape[-1] = resampled_length
            resampled_data = resampled_data.reshape(new_shape)
        
        return resampled_data
