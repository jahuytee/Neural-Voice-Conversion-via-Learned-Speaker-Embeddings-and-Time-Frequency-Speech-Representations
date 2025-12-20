"""
Feature extraction for neural voice conversion.

This module extracts acoustic features from preprocessed audio:
- Mel-spectrograms (perceptually-motivated frequency representation)
- Fundamental frequency (F0) - pitch tracking
- Aperiodicity measures

These features are the input to the neural network models.
"""

import numpy as np
import librosa
import pyworld as pw
from typing import Tuple, Optional, Dict
import torch


class FeatureExtractor:
    """
    Extract acoustic features for voice conversion.
    
    Signal Processing Pipeline:
    1. STFT (Short-Time Fourier Transform)
    2. Mel-filterbank projection
    3. Logarithmic compression
    4. F0 extraction using WORLD vocoder
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,  # 10ms
        n_mels: int = 80,
        fmin: float = 50.0,
        fmax: float = 8000.0,
        f0_min: float = 71.0,
        f0_max: float = 800.0
    ):
        """
        Initialize feature extractor.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            n_fft: FFT size for STFT
            hop_length: Hop size in samples (10ms = 160 samples @ 16kHz)
            n_mels: Number of mel filterbank channels
            fmin: Minimum frequency for mel filterbank
            fmax: Maximum frequency for mel filterbank
            f0_min: Minimum F0 for pitch tracking
            f0_max: Maximum F0 for pitch tracking
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.f0_min = f0_min
        self.f0_max = f0_max
        
        # Pre-compute mel filterbank for efficiency
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract log mel-spectrogram from audio.
        
        Signal Processing Steps:
        1. STFT: X[k,t] = Σ x[n]·w[n-tH]·e^(-j2πkn/N)
        2. Power spectrum: |X[k,t]|²
        3. Mel filterbank: M[m,t] = Σ mel_basis[m,k]·|X[k,t]|²
        4. Log compression: log(1 + C·M[m,t])
        
        Args:
            audio: Input audio signal (after preprocessing)
            
        Returns:
            Log mel-spectrogram (n_mels, n_frames)
            
        Mathematical Details:
            Mel scale: mel(f) = 2595·log10(1 + f/700)
            Triangular filters with 50% overlap
            Log dynamic range compression
        """
        # STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        
        # Power spectrum
        power_spec = np.abs(stft) ** 2
        
        # Apply mel filterbank
        mel_spec = np.dot(self.mel_basis, power_spec)
        
        # Log compression (with small epsilon to avoid log(0))
        log_mel = np.log(mel_spec + 1e-10)
        
        return log_mel
    
    def extract_f0_world(
        self,
        audio: np.ndarray,
        use_harvest: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fundamental frequency using WORLD vocoder.
        
        WORLD is a high-quality vocoder for speech analysis:
        - DIO/Harvest algorithms for F0 estimation
        - Robust to noise and vibrato
        - Used in state-of-the-art voice conversion
        
        Args:
            audio: Input audio (float64, normalized)
            use_harvest: Use Harvest (slower, higher quality) vs DIO
            
        Returns:
            Tuple of:
            - f0: Fundamental frequency contour (Hz)
            - time_axis: Time points for F0 values
            
        Algorithm (Harvest):
            1. Bandpass filter around harmonic regions
            2. Instantaneous frequency estimation
            3. F0 refinement using harmonics
            4. Temporal smoothing
        """
        # Convert to float64 for WORLD
        audio_f64 = audio.astype(np.float64)
        
        if use_harvest:
            # Harvest: slower but more accurate
            f0, time_axis = pw.harvest(
                audio_f64,
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=self.hop_length / self.sample_rate * 1000  # ms
            )
        else:
            # DIO: faster, slightly less accurate
            f0, time_axis = pw.dio(
                audio_f64,
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=self.hop_length / self.sample_rate * 1000
            )
            # Refine F0 with StoneMask
            f0 = pw.stonemask(audio_f64, f0, time_axis, self.sample_rate)
        
        return f0, time_axis
    
    def extract_spectral_envelope(self, audio: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """
        Extract spectral envelope using WORLD CheapTrick.
        
        The spectral envelope represents the formant structure
        and vocal tract characteristics.
        
        Args:
            audio: Input audio
            f0: F0 contour from extract_f0_world()
            
        Returns:
            Spectral envelope (n_fft//2+1, n_frames)
        """
        audio_f64 = audio.astype(np.float64)
        
        # Extract time axis
        _, time_axis = self.extract_f0_world(audio, use_harvest=False)
        
        # CheapTrick: spectral envelope estimation
        spectrogram = pw.cheaptrick(
            audio_f64,
            f0,
            time_axis,
            self.sample_rate,
            fft_size=self.n_fft
        )
        
        return spectrogram.T  # (freq, time)
    
    def extract_aperiodicity(
        self,
        audio: np.ndarray,
        f0: np.ndarray
    ) -> np.ndarray:
        """
        Extract aperiodicity using WORLD D4C.
        
        Aperiodicity measures indicate how much of the signal
        is noise-like vs. harmonic. Important for natural synthesis.
        
        Args:
            audio: Input audio
            f0: F0 contour
            
        Returns:
            Aperiodicity (n_fft//2+1, n_frames)
        """
        audio_f64 = audio.astype(np.float64)
        _, time_axis = self.extract_f0_world(audio, use_harvest=False)
        
        # D4C: aperiodicity estimation
        aperiodicity = pw.d4c(
            audio_f64,
            f0,
            time_axis,
            self.sample_rate,
            fft_size=self.n_fft
        )
        
        return aperiodicity.T  # (freq, time)
    
    def extract_all_features(
        self,
        audio: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract complete feature set for voice conversion.
        
        Args:
            audio: Preprocessed audio signal
            
        Returns:
            Dictionary containing:
            - 'mel': Log mel-spectrogram (80, T)
            - 'f0': F0 contour (T,)
            - 'f0_continuous': Interpolated F0 (no zeros)
            - 'vuv': Voiced/unvoiced flags (T,)
            - 'spectral_env': Spectral envelope (n_fft//2+1, T)
            - 'aperiodicity': Aperiodicity (n_fft//2+1, T)
        """
        features = {}
        
        # 1. Mel-spectrogram
        features['mel'] = self.extract_mel_spectrogram(audio)
        
        # 2. F0 extraction
        f0, time_axis = self.extract_f0_world(audio, use_harvest=True)
        features['f0'] = f0
        
        # 3. Voiced/unvoiced detection
        # F0 = 0 indicates unvoiced frame
        vuv = (f0 > 0).astype(np.float32)
        features['vuv'] = vuv
        
        # 4. Continuous F0 (interpolate unvoiced regions)
        f0_continuous = self._interpolate_f0(f0)
        features['f0_continuous'] = f0_continuous
        
        # 5. Spectral envelope
        features['spectral_env'] = self.extract_spectral_envelope(audio, f0)
        
        # 6. Aperiodicity
        features['aperiodicity'] = self.extract_aperiodicity(audio, f0)
        
        return features
    
    def _interpolate_f0(self, f0: np.ndarray) -> np.ndarray:
        """
        Interpolate F0 over unvoiced regions.
        
        This creates a continuous F0 contour useful for prosody modeling.
        
        Args:
            f0: Raw F0 with zeros for unvoiced frames
            
        Returns:
            Continuous F0 contour
        """
        # Find voiced frames
        voiced_indices = np.where(f0 > 0)[0]
        
        if len(voiced_indices) == 0:
            # No voiced frames, return zeros
            return f0
        
        # Interpolate
        f0_continuous = np.copy(f0)
        if len(voiced_indices) > 1:
            f0_continuous = np.interp(
                np.arange(len(f0)),
                voiced_indices,
                f0[voiced_indices]
            )
        
        return f0_continuous
    
    def features_to_torch(
        self,
        features: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert numpy features to PyTorch tensors.
        
        Args:
            features: Feature dictionary from extract_all_features()
            
        Returns:
            Dictionary with torch tensors
        """
        torch_features = {}
        for key, value in features.items():
            torch_features[key] = torch.from_numpy(value).float()
        
        return torch_features


def compute_feature_statistics(
    feature_list: list,
    feature_type: str = 'mel'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for feature normalization.
    
    This is computed over the entire training dataset
    and used to normalize features to zero-mean, unit-variance.
    
    Args:
        feature_list: List of feature arrays
        feature_type: Which feature ('mel', 'f0', etc.)
        
    Returns:
        Tuple of (mean, std) arrays
    """
    # Concatenate all features
    all_features = np.concatenate([f[feature_type] for f in feature_list], axis=-1)
    
    # Compute statistics along time axis
    mean = np.mean(all_features, axis=-1, keepdims=True)
    std = np.std(all_features, axis=-1, keepdims=True)
    
    # Avoid division by zero
    std = np.maximum(std, 1e-8)
    
    return mean, std


def normalize_features(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """
    Normalize features to zero-mean, unit-variance.
    
    Args:
        features: Input features
        mean: Precomputed mean
        std: Precomputed std
        
    Returns:
        Normalized features
    """
    return (features - mean) / std


if __name__ == "__main__":
    print("Feature Extraction Module for Neural Voice Conversion")
    print("=" * 60)
    print()
    print("This module extracts:")
    print("  1. Mel-spectrograms (80 channels, perceptual scale)")
    print("  2. F0 contour (WORLD vocoder, Harvest algorithm)")
    print("  3. Spectral envelope (vocal tract characteristics)")
    print("  4. Aperiodicity (harmonic structure)")
    print()
    print("Signal Processing:")
    print("  - STFT → Mel filterbank → Log compression")
    print("  - F0: DIO/Harvest + StoneMask refinement")
    print("  - Envelope: CheapTrick algorithm")
    print()
    print("Usage:")
    print("  extractor = FeatureExtractor(sample_rate=16000)")
    print("  features = extractor.extract_all_features(audio)")
