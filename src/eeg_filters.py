"""
EEG Signal Filtering Module using MNE

This module provides comprehensive filtering for EEG signals to remove artifacts
and noise, including bandpass filtering and notch filtering for powerline noise.
"""

import numpy as np
import mne
from scipy import signal
from typing import Tuple, Optional, Union


class EEGFilter:
    """
    A comprehensive EEG filtering class using MNE library.
    
    Provides bandpass filtering (1-40 Hz) and notch filtering (50/60 Hz)
    to remove artifacts and noise from EEG signals.
    """
    
    def __init__(self, sampling_rate: int, notch_freq: Optional[float] = None):
        """
        Initialize the EEG filter.
        
        Args:
            sampling_rate: Sampling rate of the EEG data in Hz
            notch_freq: Frequency for notch filter (50 or 60 Hz). If None, auto-detect.
        """
        self.sampling_rate = sampling_rate
        self.notch_freq = notch_freq or self._detect_powerline_frequency()
        
        # Bandpass filter parameters (EEG frequency range)
        self.bandpass_low = 1.0  # Hz
        self.bandpass_high = 40.0  # Hz
        
        # Notch filter parameters
        self.notch_quality = 30.0  # Quality factor for notch filter
        
    def _detect_powerline_frequency(self) -> float:
        """
        Auto-detect powerline frequency based on common standards.
        Returns 50 Hz for most of the world, 60 Hz for North America.
        """
        # You can modify this logic based on your location
        # For now, defaulting to 50 Hz (European standard)
        return 50.0
    
    def apply_bandpass_filter(self, data: np.ndarray, 
                            low_freq: Optional[float] = None,
                            high_freq: Optional[float] = None) -> np.ndarray:
        """
        Apply bandpass filter to keep only EEG frequencies (1-40 Hz).
        
        Args:
            data: EEG signal data (1D array)
            low_freq: Lower cutoff frequency (default: 1 Hz)
            high_freq: Upper cutoff frequency (default: 40 Hz)
            
        Returns:
            Filtered EEG signal
        """
        low_freq = low_freq or self.bandpass_low
        high_freq = high_freq or self.bandpass_high
        
        # Create a simple MNE RawArray for filtering
        info = mne.create_info(['EEG'], self.sampling_rate, ch_types=['eeg'])
        raw = mne.io.RawArray(data.reshape(1, -1), info)
        
        # Apply bandpass filter
        raw.filter(l_freq=low_freq, h_freq=high_freq, 
                  method='iir', iir_params=dict(order=4, ftype='butter'))
        
        return raw.get_data()[0]
    
    def apply_notch_filter(self, data: np.ndarray, 
                          notch_freq: Optional[float] = None) -> np.ndarray:
        """
        Apply notch filter to remove powerline noise (50/60 Hz).
        
        Args:
            data: EEG signal data (1D array)
            notch_freq: Notch frequency (default: auto-detected)
            
        Returns:
            Filtered EEG signal
        """
        notch_freq = notch_freq or self.notch_freq
        
        # Create a simple MNE RawArray for filtering
        info = mne.create_info(['EEG'], self.sampling_rate, ch_types=['eeg'])
        raw = mne.io.RawArray(data.reshape(1, -1), info)
        
        # Apply notch filter
        raw.notch_filter(freqs=notch_freq, notch_widths=1.0)
        
        return raw.get_data()[0]
    
    def apply_comprehensive_filter(self, data: np.ndarray,
                                 bandpass: bool = True,
                                 notch: bool = True,
                                 detrend: bool = True) -> np.ndarray:
        """
        Apply comprehensive filtering pipeline.
        
        Args:
            data: EEG signal data (1D array)
            bandpass: Whether to apply bandpass filter
            notch: Whether to apply notch filter
            detrend: Whether to detrend the signal
            
        Returns:
            Fully filtered EEG signal
        """
        filtered_data = data.copy()
        
        # Detrend to remove slow drifts
        if detrend:
            filtered_data = signal.detrend(filtered_data)
        
        # Apply notch filter first (removes powerline noise)
        if notch:
            filtered_data = self.apply_notch_filter(filtered_data)
        
        # Apply bandpass filter (removes artifacts outside EEG range)
        if bandpass:
            filtered_data = self.apply_bandpass_filter(filtered_data)
        
        return filtered_data
    
    def apply_moving_artifact_removal(self, data: np.ndarray, 
                                    threshold: float = 3.0) -> np.ndarray:
        """
        Remove large artifacts caused by movement using statistical thresholding.
        
        Args:
            data: EEG signal data (1D array)
            threshold: Standard deviation threshold for artifact detection
            
        Returns:
            EEG signal with movement artifacts removed
        """
        # Calculate rolling statistics
        window_size = int(self.sampling_rate * 0.5)  # 0.5 second window
        if len(data) < window_size:
            return data
        
        # Use scipy's uniform_filter for efficient rolling statistics
        from scipy.ndimage import uniform_filter
        
        # Calculate rolling mean and std
        rolling_mean = uniform_filter(data, size=window_size, mode='constant')
        rolling_var = uniform_filter(data**2, size=window_size, mode='constant') - rolling_mean**2
        rolling_std = np.sqrt(np.maximum(rolling_var, 0))  # Ensure non-negative
        
        # Identify artifacts
        z_scores = np.abs(data - rolling_mean) / (rolling_std + 1e-8)
        artifact_mask = z_scores > threshold
        
        # Replace artifacts with interpolated values
        if np.any(artifact_mask):
            valid_indices = np.where(~artifact_mask)[0]
            if len(valid_indices) > 1:
                filtered_data = np.interp(
                    np.arange(len(data)), 
                    valid_indices, 
                    data[valid_indices]
                )
            else:
                filtered_data = data
        else:
            filtered_data = data
        
        return filtered_data


def create_eeg_filter(sampling_rate: int, notch_freq: Optional[float] = None) -> EEGFilter:
    """
    Factory function to create an EEG filter instance.
    
    Args:
        sampling_rate: Sampling rate of the EEG data
        notch_freq: Notch frequency (50 or 60 Hz)
        
    Returns:
        EEGFilter instance
    """
    return EEGFilter(sampling_rate, notch_freq)


def quick_filter_eeg(data: np.ndarray, sampling_rate: int, 
                    notch_freq: Optional[float] = None) -> np.ndarray:
    """
    Quick one-liner function to filter EEG data.
    
    Args:
        data: EEG signal data (1D array)
        sampling_rate: Sampling rate of the EEG data
        notch_freq: Notch frequency (50 or 60 Hz)
        
    Returns:
        Filtered EEG signal
    """
    filter_obj = create_eeg_filter(sampling_rate, notch_freq)
    return filter_obj.apply_comprehensive_filter(data)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    sampling_rate = 256
    duration = 10  # seconds
    t = np.linspace(0, duration, sampling_rate * duration)
    
    # Create synthetic EEG signal with noise
    eeg_signal = (
        10 * np.sin(2 * np.pi * 10 * t) +  # Alpha wave
        5 * np.sin(2 * np.pi * 20 * t) +   # Beta wave
        2 * np.sin(2 * np.pi * 50 * t) +   # Powerline noise
        0.5 * np.sin(2 * np.pi * 0.5 * t) + # Slow drift
        np.random.normal(0, 1, len(t))     # White noise
    )
    
    # Apply filtering
    filtered_signal = quick_filter_eeg(eeg_signal, sampling_rate)
    
    print(f"Original signal std: {np.std(eeg_signal):.3f}")
    print(f"Filtered signal std: {np.std(filtered_signal):.3f}")
    print("Filtering completed successfully!")
