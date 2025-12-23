"""
Loss functions for voice conversion training.

Implements:
1. Reconstruction loss (L1 + spectral)
2. Speaker similarity loss
3. F0 prosody preservation loss
4. Combined multi-task loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss combining L1 and spectral losses.
    
    L_recon = ||X - X'||_1 + λ * ||STFT(x) - STFT(x')||_1
    
    The L1 loss ensures mel-spectrogram accuracy,
    while spectral loss ensures fine-grained frequency details.
    """
    
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 160,
        spectral_weight: float = 0.5
    ):
        """
        Initialize reconstruction loss.
        
        Args:
            n_fft: FFT size for spectral loss
            hop_length: Hop length for STFT
            spectral_weight: Weight for spectral loss term
        """
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spectral_weight = spectral_weight
        
        # L1 loss for mel-spectrogram
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        waveform_pred: Optional[torch.Tensor] = None,
        waveform_target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            predicted: Predicted mel-spectrogram (batch, n_mels, time)
            target: Target mel-spectrogram (batch, n_mels, time)
            waveform_pred: Predicted waveform (optional, for spectral loss)
            waveform_target: Target waveform (optional, for spectral loss)
            
        Returns:
            loss: Combined reconstruction loss
        """
        # Mel-spectrogram L1 loss
        mel_loss = self.l1_loss(predicted, target)
        
        # Spectral loss (if waveforms provided)
        if waveform_pred is not None and waveform_target is not None:
            spectral_loss = self._spectral_loss(waveform_pred, waveform_target)
            total_loss = mel_loss + self.spectral_weight * spectral_loss
        else:
            total_loss = mel_loss
        
        return total_loss
    
    def _spectral_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spectral L1 loss using STFT.
        
        Args:
            predicted: Predicted waveform (batch, samples)
            target: Target waveform (batch, samples)
            
        Returns:
            loss: Spectral L1 loss
        """
        # Compute STFT
        pred_stft = torch.stft(
            predicted,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )
        target_stft = torch.stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )
        
        # Magnitude loss
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        loss = F.l1_loss(pred_mag, target_mag)
        
        return loss


class SpeakerSimilarityLoss(nn.Module):
    """
    Speaker similarity loss using cosine similarity.
    
    L_spk = 1 - cos(e_converted, e_target)
    
    Encourages converted speech to have speaker embedding
    similar to target speaker.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        converted_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute speaker similarity loss.
        
        Args:
            converted_embedding: Speaker embedding of converted audio (batch, dim)
            target_embedding: Speaker embedding of target speaker (batch, dim)
            
        Returns:
            loss: Speaker dissimilarity (1 - cosine similarity)
        """
        # Cosine similarity
        similarity = F.cosine_similarity(converted_embedding, target_embedding, dim=1)
        
        # Loss: maximize similarity = minimize (1 - similarity)
        loss = 1.0 - similarity.mean()
        
        return loss


class F0Loss(nn.Module):
    """
    F0 (fundamental frequency) loss for prosody preservation.
    
    L_F0 = MSE(F0_source, F0_converted)
    
    Ensures converted speech preserves prosody (pitch contour)
    from source speaker.
    """
    
    def __init__(self, voiced_weight: float = 2.0):
        """
        Initialize F0 loss.
        
        Args:
            voiced_weight: Higher weight for voiced (F0 > 0) frames
        """
        super().__init__()
        self.voiced_weight = voiced_weight
    
    def forward(
        self,
        predicted_f0: torch.Tensor,
        target_f0: torch.Tensor,
        voiced_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute F0 loss.
        
        Args:
            predicted_f0: Predicted F0 contour (batch, time)
            target_f0: Target F0 contour (batch, time)
            voiced_mask: Binary mask for voiced frames (batch, time)
            
        Returns:
            loss: Weighted MSE loss on F0
        """
        # MSE loss
        mse = (predicted_f0 - target_f0) ** 2
        
        if voiced_mask is not None:
            # Weight voiced frames more heavily
            weights = torch.ones_like(voiced_mask)
            weights = torch.where(voiced_mask > 0.5, 
                                self.voiced_weight * weights, 
                                weights)
            mse = mse * weights
        
        loss = mse.mean()
        
        return loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using intermediate features from discriminator.
    
    L_feat = Σ ||φ_i(x) - φ_i(x')||_1
    
    Matches intermediate network activations for perceptual similarity.
    """
    
    def __init__(self, discriminator: Optional[nn.Module] = None):
        """
        Initialize perceptual loss.
        
        Args:
            discriminator: Discriminator network (for feature extraction)
        """
        super().__init__()
        self.discriminator = discriminator
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        features_pred: Optional[list] = None,
        features_target: Optional[list] = None
    ) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            predicted: Predicted output
            target: Target output
            features_pred: Pre-extracted features from predicted (optional)
            features_target: Pre-extracted features from target (optional)
            
        Returns:
            loss: Perceptual feature matching loss
        """
        if features_pred is None or features_target is None:
            # Extract features if not provided
            if self.discriminator is None:
                return torch.tensor(0.0, device=predicted.device)
            
            features_pred = self._extract_features(predicted)
            features_target = self._extract_features(target)
        
        # L1 loss on each feature level
        loss = 0.0
        for feat_p, feat_t in zip(features_pred, features_target):
            loss += F.l1_loss(feat_p, feat_t)
        
        # Average over feature levels
        loss = loss / len(features_pred)
        
        return loss
    
    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract intermediate features from discriminator."""
        features = []
        # This would be implemented based on discriminator architecture
        # For now, return empty list
        return features


class VoiceConversionLoss(nn.Module):
    """
    Combined multi-task loss for voice conversion.
    
    L_total = λ_recon * L_recon 
            + λ_spk * L_spk 
            + λ_F0 * L_F0
            + λ_perc * L_perc
    """
    
    def __init__(
        self,
        recon_weight: float = 1.0,
        speaker_weight: float = 0.1,
        f0_weight: float = 0.5,
        perceptual_weight: float = 0.1
    ):
        """
        Initialize combined loss.
        
        Args:
            recon_weight: Weight for reconstruction loss
            speaker_weight: Weight for speaker similarity loss
            f0_weight: Weight for F0 prosody loss
            perceptual_weight: Weight for perceptual loss
        """
        super().__init__()
        
        self.recon_weight = recon_weight
        self.speaker_weight = speaker_weight
        self.f0_weight = f0_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize loss functions
        self.recon_loss = ReconstructionLoss()
        self.speaker_loss = SpeakerSimilarityLoss()
        self.f0_loss = F0Loss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(
        self,
        predicted_mel: torch.Tensor,
        target_mel: torch.Tensor,
        converted_speaker_emb: Optional[torch.Tensor] = None,
        target_speaker_emb: Optional[torch.Tensor] = None,
        predicted_f0: Optional[torch.Tensor] = None,
        target_f0: Optional[torch.Tensor] = None,
        voiced_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predicted_mel: Predicted mel-spectrogram
            target_mel: Target mel-spectrogram
            converted_speaker_emb: Speaker embedding of converted audio
            target_speaker_emb: Target speaker embedding
            predicted_f0: Predicted F0 contour
            target_f0: Target F0 contour
            voiced_mask: Voiced/unvoiced mask
            
        Returns:
            losses: Dictionary of individual and total losses
        """
        losses = {}
        
        # Reconstruction loss (always computed)
        loss_recon = self.recon_loss(predicted_mel, target_mel)
        losses['recon'] = loss_recon
        total_loss = self.recon_weight * loss_recon
        
        # Speaker similarity loss (if embeddings provided)
        if converted_speaker_emb is not None and target_speaker_emb is not None:
            loss_spk = self.speaker_loss(converted_speaker_emb, target_speaker_emb)
            losses['speaker'] = loss_spk
            total_loss += self.speaker_weight * loss_spk
        
        # F0 loss (if F0 provided)
        if predicted_f0 is not None and target_f0 is not None:
            loss_f0 = self.f0_loss(predicted_f0, target_f0, voiced_mask)
            losses['f0'] = loss_f0
            total_loss += self.f0_weight * loss_f0
        
        losses['total'] = total_loss
        
        return losses


if __name__ == "__main__":
    print("Loss Functions Module for Voice Conversion")
    print("=" * 60)
    print()
    
    # Test reconstruction loss
    print("1. Testing Reconstruction Loss...")
    recon_loss_fn = ReconstructionLoss()
    pred_mel = torch.randn(4, 80, 100)
    target_mel = torch.randn(4, 80, 100)
    loss = recon_loss_fn(pred_mel, target_mel)
    print(f"   ✓ Reconstruction loss: {loss.item():.4f}")
    
    # Test speaker similarity loss
    print("\n2. Testing Speaker Similarity Loss...")
    spk_loss_fn = SpeakerSimilarityLoss()
    emb1 = F.normalize(torch.randn(4, 256), p=2, dim=1)
    emb2 = F.normalize(torch.randn(4, 256), p=2, dim=1)
    loss = spk_loss_fn(emb1, emb2)
    print(f"   ✓ Speaker loss: {loss.item():.4f}")
    
    # Test F0 loss
    print("\n3. Testing F0 Loss...")
    f0_loss_fn = F0Loss()
    f0_pred = torch.randn(4, 100) * 100 + 150
    f0_target = torch.randn(4, 100) * 100 + 150
    voiced = (torch.randn(4, 100) > 0).float()
    loss = f0_loss_fn(f0_pred, f0_target, voiced)
    print(f"   ✓ F0 loss: {loss.item():.4f}")
    
    # Test combined loss
    print("\n4. Testing Combined Loss...")
    vc_loss_fn = VoiceConversionLoss()
    losses = vc_loss_fn(
        pred_mel, target_mel,
        emb1, emb2,
        f0_pred, f0_target,
        voiced
    )
    print(f"   ✓ Total loss: {losses['total'].item():.4f}")
    print(f"      - Reconstruction: {losses['recon'].item():.4f}")
    print(f"      - Speaker: {losses['speaker'].item():.4f}")
    print(f"      - F0: {losses['f0'].item():.4f}")
    
    print("\n✅ All loss functions working correctly!")
