"""
Content Encoder for Voice Conversion

Extracts linguistic content (what is said) while removing speaker identity (who says it).

Uses Instance Normalization to remove speaker-specific statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentEncoder(nn.Module):
    """
    Content encoder using CNN with Instance Normalization.
    
    Architecture:
        Input: Mel-spectrogram (n_mels, time)
        → Conv1D + InstanceNorm + ReLU
        → Conv1D (stride=2) + InstanceNorm + ReLU  # Downsample
        → Conv1D (stride=2) + InstanceNorm + ReLU  # Downsample
        Output: Content embedding (512, time/4)
    
    Instance Normalization removes speaker-specific mean/std,
    forcing the network to learn content-only representations.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        channels: list = [128, 256, 512],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize content encoder.
        
        Args:
            n_mels: Number of mel channels
            channels: Channel dimensions for each conv layer
            kernel_size: Convolutional kernel size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_mels = n_mels
        self.channels = channels
        
        # Build convolutional layers
        layers = []
        in_channels = n_mels
        
        for i, out_channels in enumerate(channels):
            # Stride increases for downsampling
            stride = 2 if i > 0 else 1
            
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2
                ),
                nn.InstanceNorm1d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        
        # Output dimension
        self.output_dim = channels[-1]
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Extract content embedding from mel-spectrogram.
        
        Args:
            mel_spec: Mel-spectrogram (batch, n_mels, time)
            
        Returns:
            content: Content embedding (batch, 512, time/4)
                    Speaker identity removed by Instance Norm
        """
        # Convolutional encoding
        content = self.encoder(mel_spec)
        
        return content


class Decoder(nn.Module):
    """
    Decoder for voice conversion using Adaptive Instance Normalization (AdaIN).
    
    Combines:
    - Content embedding (what to say)
    - Speaker embedding (who should say it)
    
    To generate mel-spectrogram in target speaker's voice.
    """
    
    def __init__(
        self,
        content_dim: int = 512,
        speaker_dim: int = 256,
        n_mels: int = 80,
        channels: list = [512, 256, 128],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize decoder.
        
        Args:
            content_dim: Content embedding dimension
            speaker_dim: Speaker embedding dimension
            n_mels: Number of mel channels (output)
            channels: Channel dimensions for each deconv layer
            kernel_size: Kernel size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.content_dim = content_dim
        self.speaker_dim = speaker_dim
        self.n_mels = n_mels
        
        # Build decoder layers
        layers = []
        in_channels = content_dim
        
        for i, out_channels in enumerate(channels):
            # Transposed conv for upsampling
            layers.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size * 2,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1
                )
            )
            
            # AdaIN layer (conditioned on speaker)
            layers.append(AdaptiveInstanceNorm1d(out_channels, speaker_dim))
            
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
        
        # Final conv to mel channels
        layers.append(
            nn.Conv1d(channels[-1], n_mels, kernel_size=kernel_size, padding=kernel_size // 2)
        )
        layers.append(nn.Tanh())  # Normalize output range
        
        self.decoder = nn.ModuleList(layers)
    
    def forward(self, content: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """
        Generate mel-spectrogram from content + speaker.
        
        Args:
            content: Content embedding (batch, content_dim, time/4)
            speaker_embedding: Speaker embedding (batch, speaker_dim)
            
        Returns:
            mel_spec: Reconstructed mel-spectrogram (batch, n_mels, time)
        """
        x = content
        
        for layer in self.decoder:
            if isinstance(layer, AdaptiveInstanceNorm1d):
                # AdaIN needs speaker embedding
                x = layer(x, speaker_embedding)
            else:
                x = layer(x)
        
        return x


class AdaptiveInstanceNorm1d(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN).
    
    Applies instance normalization, then scales and shifts
    based on speaker embedding.
    
    AdaIN(x, s) = σ(s) * ((x - μ(x)) / σ(x)) + μ(s)
    
    where μ(s), σ(s) are learned from speaker embedding.
    """
    
    def __init__(self, num_features: int, speaker_dim: int):
        """
        Initialize AdaIN layer.
        
        Args:
            num_features: Number of features (channels)
            speaker_dim: Speaker embedding dimension
        """
        super().__init__()
        
        self.num_features = num_features
        
        # Instance normalization
        self.instance_norm = nn.InstanceNorm1d(num_features, affine=False)
        
        # Learn scale and bias from speaker embedding
        self.fc_scale = nn.Linear(speaker_dim, num_features)
        self.fc_bias = nn.Linear(speaker_dim, num_features)
        
        # Initialize
        nn.init.constant_(self.fc_scale.weight, 0)
        nn.init.constant_(self.fc_scale.bias, 1)  # Scale = 1 initially
        nn.init.constant_(self.fc_bias.weight, 0)
        nn.init.constant_(self.fc_bias.bias, 0)   # Bias = 0 initially
    
    def forward(self, x: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN.
        
        Args:
            x: Input features (batch, num_features, time)
            speaker_embedding: Speaker embedding (batch, speaker_dim)
            
        Returns:
            out: Normalized and conditioned features (batch, num_features, time)
        """
        # Instance normalization
        normalized = self.instance_norm(x)
        
        # Compute scale and bias from speaker embedding
        scale = self.fc_scale(speaker_embedding)  # (batch, num_features)
        bias = self.fc_bias(speaker_embedding)    # (batch, num_features)
        
        # Reshape for broadcasting: (batch, num_features, 1)
        scale = scale.unsqueeze(2)
        bias = bias.unsqueeze(2)
        
        # Apply affine transformation
        out = scale * normalized + bias
        
        return out


class VoiceConversionModel(nn.Module):
    """
    Complete voice conversion model.
    
    Combines:
    - Content encoder (extracts WHAT is said)
    - Speaker encoder (extracts WHO speaks) - frozen during VC training
    - Decoder (generates mel in target voice)
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        content_channels: list = [128, 256, 512],
        speaker_dim: int = 256,
        decoder_channels: list = [512, 256, 128]
    ):
        """
        Initialize complete VC model.
        
        Args:
            n_mels: Mel channels
            content_channels: Content encoder channels
            speaker_dim: Speaker embedding dimension
            decoder_channels: Decoder channels
        """
        super().__init__()
        
        # Content encoder
        self.content_encoder = ContentEncoder(
            n_mels=n_mels,
            channels=content_channels
        )
        
        # Decoder
        self.decoder = Decoder(
            content_dim=content_channels[-1],
            speaker_dim=speaker_dim,
            n_mels=n_mels,
            channels=decoder_channels
        )
    
    def forward(
        self,
        source_mel: torch.Tensor,
        target_speaker_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform voice conversion.
        
        Args:
            source_mel: Source mel-spectrogram (batch, n_mels, time)
            target_speaker_embedding: Target speaker (batch, speaker_dim)
            
        Returns:
            converted_mel: Mel in target speaker's voice (batch, n_mels, time')
        """
        # Extract content (removes speaker info)
        content = self.content_encoder(source_mel)
        
        # Generate with target speaker
        converted_mel = self.decoder(content, target_speaker_embedding)
        
        return converted_mel


if __name__ == "__main__":
    print("Content Encoder & Decoder Module")
    print("=" * 60)
    print()
    print("Content Encoder:")
    print("  - CNN with Instance Normalization")
    print("  - Removes speaker identity")
    print("  - Preserves linguistic content")
    print()
    print("Decoder:")
    print("  - Transposed CNN for upsampling")
    print("  - AdaIN: Conditions on speaker embedding")
    print("  - Generates mel-spectrogram")
    print()
    
    # Example usage
    content_enc = ContentEncoder(n_mels=80)
    decoder = Decoder(content_dim=512, speaker_dim=256, n_mels=80)
    
    mel = torch.randn(2, 80, 100)  # (batch=2, mels=80, time=100)
    speaker_emb = torch.randn(2, 256)  # (batch=2, speaker_dim=256)
    
    content = content_enc(mel)
    reconstructed = decoder(content, speaker_emb)
    
    print(f"Input mel shape: {mel.shape}")
    print(f"Content shape: {content.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Full model
    vc_model = VoiceConversionModel()
    converted = vc_model(mel, speaker_emb)
    print(f"Converted mel shape: {converted.shape}")
