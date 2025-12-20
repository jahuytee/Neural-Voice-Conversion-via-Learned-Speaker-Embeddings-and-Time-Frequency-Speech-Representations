"""
Signal preprocessing for neural voice conversion.

This module implements DSP fundamentals:
- Resampling
- Pre-emphasis filtering  
- Frame segmentation
- Windowing
- Voice Activity Detection (VAD)
"""

import numpy as np
import librosa
import scipy.signal as scipy_signal
from typing import Tuple, Optional


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: Input audio signal
        orig_sr: Original sample rate
        target_sr: Target sample rate (default: 16kHz)
        
    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def apply_preemphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter to boost high frequencies.
    
    Transfer function: H(z) = 1 - α·z^(-1)
    
    This compensates for the -6dB/octave spectral tilt in speech
    and improves SNR of high-frequency components.
    
    Args:
        audio: Input audio signal
        coef: Pre-emphasis coefficient α (typically 0.95-0.97)
        
    Returns:
        Pre-emphasized audio
        
    Signal Processing:
        y[n] = x[n] - α·x[n-1]
    """
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


def inverse_preemphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Reverse pre-emphasis filter.
    
    Transfer function: H(z) = 1 / (1 - α·z^(-1))
    
    Args:
        audio: Pre-emphasized audio
        coef: Pre-emphasis coefficient used in forward pass
        
    Returns:
        De-emphasized audio
    """
    # Use lfilter for IIR de-emphasis
    return scipy_signal.lfilter([1], [1, -coef], audio)


def segment_into_frames(
    audio: np.ndarray,
    frame_length: int = 400,  # 25ms at 16kHz
    hop_length: int = 160,     # 10ms at 16kHz
    window: str = 'hann'
) -> np.ndarray:
    """
    Segment audio into overlapping frames with windowing.
    
    Args:
        audio: Input audio signal
        frame_length: Frame size in samples (default: 25ms = 400 samples @ 16kHz)
        hop_length: Hop size in samples (default: 10ms = 160 samples @ 16kHz)
        window: Window type ('hann', 'hamming', 'blackman')
        
    Returns:
        Framed audio (n_frames, frame_length)
        
    Signal Processing:
        - Overlap: (frame_length - hop_length) / frame_length
        - For default params: 60% overlap
        - Window applied: x_i[n] = audio[i·hop + n] · w[n]
    """
    # Create window function
    if window == 'hann':
        win = np.hanning(frame_length)
    elif window == 'hamming':
        win = np.hamming(frame_length)
    elif window == 'blackman':
        win = np.blackman(frame_length)
    else:
        win = np.ones(frame_length)
    
    # Calculate number of frames
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    
    # Segment into frames
    frames = np.zeros((n_frames, frame_length))
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= len(audio):
            frames[i] = audio[start:end] * win
    
    return frames


def voice_activity_detection(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_length: int = 400,
    hop_length: int = 160,
    energy_threshold: float = 0.02,
    zero_crossing_threshold: float = 0.3
) -> np.ndarray:
    """
    Simple energy-based Voice Activity Detection.
    
    Detects speech vs. silence/noise based on:
    1. Short-term energy
    2. Zero-crossing rate
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate in Hz
        frame_length: Frame size for analysis
        hop_length: Hop size
        energy_threshold: Normalized energy threshold (0-1)
        zero_crossing_threshold: ZCR threshold
        
    Returns:
        Boolean mask where True = voice activity
        
    Algorithm:
        E[i] = Σ(x_i[n]²) / frame_length
        ZCR[i] = Σ(|sign(x[n]) - sign(x[n-1])|) / (2·frame_length)
        VAD[i] = (E[i] > threshold) AND (ZCR[i] < threshold)
    """
    # Segment into frames
    frames = segment_into_frames(audio, frame_length, hop_length, window='none')
    n_frames = frames.shape[0]
    
    # Calculate frame energy
    energy = np.sum(frames ** 2, axis=1) / frame_length
    
    # Normalize energy
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    
    # Calculate zero-crossing rate
    zcr = np.zeros(n_frames)
    for i, frame in enumerate(frames):
        signs = np.sign(frame)
        zcr[i] = np.sum(np.abs(np.diff(signs))) / (2 * frame_length)
    
    # Voice activity decision
    vad = (energy > energy_threshold) & (zcr < zero_crossing_threshold)
    
    return vad


def trim_silence(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_length: int = 400,
    hop_length: int = 160,
    energy_threshold: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove leading and trailing silence from audio.
    
    Args:
        audio: Input audio
        sample_rate: Sample rate
        frame_length: Frame size for VAD
        hop_length: Hop size
        energy_threshold: Energy threshold for VAD
        
    Returns:
        Tuple of (trimmed_audio, vad_mask)
    """
    # Get VAD mask
    vad = voice_activity_detection(
        audio, sample_rate, frame_length, hop_length, energy_threshold
    )
    
    # Find first and last voice frames
    voice_frames = np.where(vad)[0]
    
    if len(voice_frames) == 0:
        # No voice detected, return original
        return audio, vad
    
    start_frame = voice_frames[0]
    end_frame = voice_frames[-1]
    
    # Convert to sample indices
    start_sample = start_frame * hop_length
    end_sample = min(end_frame * hop_length + frame_length, len(audio))
    
    trimmed = audio[start_sample:end_sample]
    
    return trimmed, vad


def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target RMS level in dB.
    
    Args:
        audio: Input audio
        target_level: Target RMS level in dB (default: -20 dB)
        
    Returns:
        Normalized audio
        
    Formula:
        RMS = sqrt(mean(x²))
        RMS_dB = 20·log10(RMS)
        gain = 10^((target_dB - RMS_dB) / 20)
    """
    # Calculate RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    # Calculate current level in dB
    current_level = 20 * np.log10(rms)
    
    # Calculate gain needed
    gain_db = target_level - current_level
    gain = 10 ** (gain_db / 20)
    
    # Apply gain
    normalized = audio * gain
    
    # Clip to prevent overflow
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def main_preprocessing_pipeline(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
    apply_vad: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline.
    
    Steps:
    1. Resample to target rate
    2. Normalize audio level
    3. Trim silence (optional)
    4. Apply pre-emphasis
    
    Args:
        audio: Input audio
        orig_sr: Original sample rate
        target_sr: Target sample rate
        apply_vad: Whether to trim silence
        normalize: Whether to normalize level
        
    Returns:
        Tuple of (processed_audio, metadata_dict)
    """
    metadata = {}
    
    # Step 1: Resample
    if orig_sr != target_sr:
        audio = resample_audio(audio, orig_sr, target_sr)
        metadata['resampled'] = True
        metadata['target_sr'] = target_sr
    
    # Step 2: Normalize
    if normalize:
        audio = normalize_audio(audio, target_level=-20.0)
        metadata['normalized'] = True
    
    # Step 3: Trim silence
    if apply_vad:
        audio, vad_mask = trim_silence(audio, target_sr)
        metadata['trimmed'] = True
        metadata['voice_frames'] = np.sum(vad_mask)
        metadata['total_frames'] = len(vad_mask)
    
    # Step 4: Pre-emphasis
    audio = apply_preemphasis(audio, coef=0.97)
    metadata['preemphasized'] = True
    
    return audio, metadata


if __name__ == "__main__":
    print("Signal Preprocessing Module for Neural Voice Conversion")
    print("=" * 60)
    print()
    print("Functions available:")
    print("  - resample_audio(): Change sample rate")
    print("  - apply_preemphasis(): Boost high frequencies")
    print("  - segment_into_frames(): Frame segmentation + windowing")
    print("  - voice_activity_detection(): Detect speech vs silence")
    print("  - main_preprocessing_pipeline(): Complete pipeline")
    print()
    print("All functions include detailed DSP mathematics in docstrings")
