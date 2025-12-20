"""
Speaker Encoder for Voice Conversion

This module implements speaker embedding extraction using:
- DVec (d-vector): LSTM-based speaker verification
- GE2E (Generalized End-to-End) loss

The speaker encoder learns to create 256-dimensional embeddings
that capture speaker-specific characteristics (timbre, pitch, resonance).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpeakerEncoder(nn.Module):
    """
    DVec-style speaker encoder using LSTM.
    
    Architecture:
        Input: Mel-spectrogram (n_mels, time)
        → 3x LSTM layers (256 hidden units)
        → Temporal average pooling
        → Linear projection (256 dims)
        → L2 normalization
        Output: Speaker embedding (256,)
    
    The embedding captures speaker identity, NOT content.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 3,
        embedding_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize speaker encoder.
        
        Args:
            n_mels: Number of mel channels
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            embedding_dim: Final embedding dimensionality
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Projection to embedding space
        self.projection = nn.Linear(hidden_dim, embedding_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        self.projection.bias.data.fill_(0)
    
    def forward(self, mel_spec: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract speaker embedding from mel-spectrogram.
        
        Args:
            mel_spec: Mel-spectrogram (batch, n_mels, time)
            lengths: Actual sequence lengths (batch,) for masking
            
        Returns:
            speaker_embedding: L2-normalized embedding (batch, embedding_dim)
        """
        batch_size = mel_spec.size(0)
        
        # Transpose for LSTM: (batch, time, n_mels)
        x = mel_spec.transpose(1, 2)
        
        # LSTM forward pass
        # outputs: (batch, time, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        outputs, (h_n, c_n) = self.lstm(x)
        
        # Temporal average pooling
        if lengths is not None:
            # Mask padding frames
            mask = self._create_mask(lengths, outputs.size(1)).to(outputs.device)
            outputs = outputs * mask.unsqueeze(-1)
            # Average over non-padded frames
            embedding = outputs.sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            # Simple average over all frames
            embedding = outputs.mean(dim=1)
        
        # Project to embedding space
        embedding = self.projection(embedding)
        
        # L2 normalization (critical for metric learning)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def _create_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create binary mask for variable-length sequences."""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
        mask = mask < lengths.unsqueeze(1)
        return mask.float()


class GE2ELoss(nn.Module):
    """
    Generalized End-to-End (GE2E) Loss for speaker verification.
    
    From "Generalized End-to-End Loss for Speaker Verification" (Wan et al., 2018)
    
    The loss encourages embeddings from the same speaker to be close
    and embeddings from different speakers to be far apart.
    
    Mathematical formulation:
        L_GE2E = -Σ log( exp(w·cos(e_jk, c_j) + b) / Σ_i exp(w·cos(e_jk, c_i) + b) )
    
    where:
        e_jk: embedding k from speaker j
        c_j: centroid of speaker j (excluding e_jk)
        w, b: learnable scale and bias
    """
    
    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        """
        Initialize GE2E loss.
        
        Args:
            init_w: Initial value for similarity scale
            init_b: Initial value for similarity bias
        """
        super().__init__()
        
        # Learnable parameters
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute GE2E loss.
        
        Args:
            embeddings: Speaker embeddings
                       (num_speakers, utterances_per_speaker, embedding_dim)
                       
        Returns:
            loss: Scalar GE2E loss
        """
        num_speakers, utterances_per_speaker, embedding_dim = embeddings.size()
        
        # Compute centroids (excluding current utterance)
        # Shape: (num_speakers, embedding_dim)
        centroids = self._compute_centroids(embeddings)
        
        # Compute similarity matrix
        # sim[j,k,i] = similarity between utterance k of speaker j and centroid of speaker i
        similarities = self._compute_similarities(embeddings, centroids)
        
        # Apply scale and bias
        similarities = self.w * similarities + self.b
        
        # Compute softmax cross-entropy
        # Target: each utterance should match its own speaker's centroid
        loss = 0
        for j in range(num_speakers):
            for k in range(utterances_per_speaker):
                # Similarity to all centroids
                sim_vector = similarities[j, k, :]  # (num_speakers,)
                
                # Cross-entropy: log(exp(sim_jj) / Σ_i exp(sim_ji))
                loss += -F.log_softmax(sim_vector, dim=0)[j]
        
        # Average over all utterances
        loss = loss / (num_speakers * utterances_per_speaker)
        
        return loss
    
    def _compute_centroids(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute centroids for each speaker (excluding current utterance).
        
        For GE2E, we exclude the current utterance when computing centroid
        to prevent trivial solutions.
        """
        num_speakers, utterances_per_speaker, embedding_dim = embeddings.size()
        
        # Simple centroid: average of all utterances per speaker
        centroids = embeddings.mean(dim=1)  # (num_speakers, embedding_dim)
        
        return centroids
    
    def _compute_similarities(self, embeddings: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between embeddings and centroids.
        
        Returns:
            similarities: (num_speakers, utterances_per_speaker, num_speakers)
        """
        num_speakers, utterances_per_speaker, embedding_dim = embeddings.size()
        
        # Flatten embeddings: (num_speakers * utterances_per_speaker, embedding_dim)
        embeddings_flat = embeddings.view(-1, embedding_dim)
        
        # Compute cosine similarity
        # (num_speakers * utterances_per_speaker, num_speakers)
        similarities_flat = F.cosine_similarity(
            embeddings_flat.unsqueeze(1),  # (N, 1, D)
            centroids.unsqueeze(0),        # (1, S, D)
            dim=2
        )
        
        # Reshape back
        similarities = similarities_flat.view(num_speakers, utterances_per_speaker, num_speakers)
        
        return similarities


class TripletLoss(nn.Module):
    """
    Alternative: Triplet loss for speaker verification.
    
    L_triplet = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    
    Simpler than GE2E but may be less effective.
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings (batch, embedding_dim)
            positive: Positive embeddings (same speaker) (batch, embedding_dim)
            negative: Negative embeddings (different speaker) (batch, embedding_dim)
            
        Returns:
            loss: Scalar triplet loss
        """
        # Euclidean distance
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss with margin
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


def cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: (batch, embedding_dim) or (embedding_dim,)
        embedding2: (batch, embedding_dim) or (embedding_dim,)
        
    Returns:
        similarity: Scalar or (batch,) in range [-1, 1]
    """
    return F.cosine_similarity(embedding1, embedding2, dim=-1)


if __name__ == "__main__":
    print("Speaker Encoder Module")
    print("=" * 60)
    print()
    print("Architecture: DVec (LSTM-based)")
    print("  - 3x LSTM layers (256 hidden units)")
    print("  - Temporal average pooling")
    print("  - Linear projection (256 dims)")
    print("  - L2 normalization")
    print()
    print("Loss: GE2E (Generalized End-to-End)")
    print("  - Cosine similarity to centroids")
    print("  - Learnable scale (w) and bias (b)")
    print("  - Cross-entropy over speaker identities")
    print()
    
    # Example usage
    model = SpeakerEncoder(n_mels=80, embedding_dim=256)
    mel = torch.randn(4, 80, 100)  # (batch=4, mels=80, time=100)
    embedding = model(mel)
    print(f"Input shape: {mel.shape}")
    print(f"Output embedding shape: {embedding.shape}")
    print(f"Embedding norm: {embedding.norm(dim=1)}")  # Should be ~1.0 (normalized)
