"""
Evaluation metrics for voice conversion.

Implements:
- Mel Cepstral Distortion (MCD)
- F0 Root Mean Square Error (F0-RMSE)
- Speaker similarity (cosine similarity)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


def mel_cepstral_distortion(
    target_mel: np.ndarray,
    predicted_mel: np.ndarray
) -> float:
    """
    Compute Mel Cepstral Distortion (MCD).
    
    MCD = (10 / ln(10)) * sqrt(2 * Σ(c_i - c'_i)^2)
    
    Args:
        target_mel: Target mel-spectrogram (n_mels, time)
        predicted_mel: Predicted mel-spectrogram (n_mels, time)
        
    Returns:
        mcd: MCD in dB (lower is better, < 6.0 is good)
    """
    # Compute MEL cepstral coefficients (simplified version)
    diff = target_mel - predicted_mel
    
    # MCD formula
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2 * np.mean(diff ** 2))
    
    return mcd


def f0_rmse(
    target_f0: np.ndarray,
    predicted_f0: np.ndarray,
    voiced_mask: np.ndarray = None
) -> float:
    """
    Compute F0 Root Mean Square Error.
    
    Args:
        target_f0: Target F0 contour (time,)
        predicted_f0: Predicted F0 contour (time,)
        voiced_mask: Binary mask for voiced frames (time,)
        
    Returns:
        rmse: F0 RMSE in Hz (lower is better, < 15 Hz is good)
    """
    if voiced_mask is not None:
        # Only compute on voiced frames
        target_f0 = target_f0[voiced_mask > 0.5]
        predicted_f0 = predicted_f0[voiced_mask > 0.5]
    
    # RMSE
    rmse = np.sqrt(np.mean((target_f0 - predicted_f0) ** 2))
    
    return rmse


def speaker_similarity(
    converted_embedding: torch.Tensor,
    target_embedding: torch.Tensor
) -> float:
    """
    Compute speaker similarity using cosine similarity.
    
    Args:
        converted_embedding: Speaker embedding of converted audio (dim,)
        target_embedding: Target speaker embedding (dim,)
        
    Returns:
        similarity: Cosine similarity in [0, 1] (higher is better, > 0.8 is good)
    """
    similarity = F.cosine_similarity(
        converted_embedding.unsqueeze(0),
        target_embedding.unsqueeze(0),
        dim=1
    ).item()
    
    return similarity


def evaluate_conversion(
    target_mel: np.ndarray,
    converted_mel: np.ndarray,
    target_f0: np.ndarray = None,
    converted_f0: np.ndarray = None,
    target_speaker_emb: torch.Tensor = None,
    converted_speaker_emb: torch.Tensor = None
) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        target_mel: Target mel-spectrogram
        converted_mel: Converted mel-spectrogram
        target_f0: Target F0 contour (optional)
        converted_f0: Converted F0 contour (optional)
        target_speaker_emb: Target speaker embedding (optional)
        converted_speaker_emb: Converted speaker embedding (optional)
        
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {}
    
    # MCD
    metrics['mcd'] = mel_cepstral_distortion(target_mel, converted_mel)
    
    # F0 RMSE
    if target_f0 is not None and converted_f0 is not None:
        metrics['f0_rmse'] = f0_rmse(target_f0, converted_f0)
    
    # Speaker similarity
    if target_speaker_emb is not None and converted_speaker_emb is not None:
        metrics['speaker_similarity'] = speaker_similarity(
            converted_speaker_emb,
            target_speaker_emb
        )
    
    return metrics


def print_metrics(metrics: dict):
    """Print evaluation metrics in nice format."""
    print()
    print("="*60)
    print(" Evaluation Metrics".center(60))
    print("="*60)
    print()
    
    if 'mcd' in metrics:
        mcd = metrics['mcd']
        quality = "Excellent" if mcd < 5.0 else "Good" if mcd < 6.0 else "Fair" if mcd < 7.0 else "Poor"
        print(f"  Mel Cepstral Distortion (MCD): {mcd:.2f} dB ({quality})")
        print(f"    - Lower is better, < 6.0 dB is good")
    
    if 'f0_rmse' in metrics:
        f0 = metrics['f0_rmse']
        quality = "Excellent" if f0 < 10.0 else "Good" if f0 < 15.0 else "Fair" if f0 < 20.0 else "Poor"
        print(f"  F0 RMSE: {f0:.2f} Hz ({quality})")
        print(f"    - Lower is better, < 15 Hz is good")
    
    if 'speaker_similarity' in metrics:
        sim = metrics['speaker_similarity']
        quality = "Excellent" if sim > 0.9 else "Good" if sim > 0.8 else "Fair" if sim > 0.7 else "Poor"
        print(f"  Speaker Similarity: {sim:.3f} ({quality})")
        print(f"    - Higher is better, > 0.8 is good")
    
    print()
    print("="*60)


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("=" * 60)
    print()
    print("Available metrics:")
    print("  - Mel Cepstral Distortion (MCD): Audio quality")
    print("  - F0 RMSE: Prosody preservation")
    print("  - Speaker Similarity: Voice identity transfer")
    print()
    print("Quality thresholds:")
    print("  - MCD < 6.0 dB: Good")
    print("  - F0 RMSE < 15 Hz: Good")
    print("  - Speaker Similarity > 0.8: Good")
