"""
Voice conversion inference pipeline.

Load trained models and perform voice conversion.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import soundfile as sf

# Import our modules
from models.speaker_encoder import SpeakerEncoder
from models.voice_conversion import VoiceConversionModel
from signal_processing.preprocessing import main_preprocessing_pipeline
from signal_processing.features import FeatureExtractor


class VoiceConverter:
    """
    Voice conversion inference pipeline.
    
    Usage:
        converter = VoiceConverter(
            speaker_encoder_path='checkpoints/speaker_encoder.pt',
            vc_model_path='checkpoints/vc_model.pt'
        )
        
        # Convert voice
        output = converter.convert(
            source_audio='my_voice.wav',
            target_speaker_embedding='lebron.pt'
        )
    """
    
    def __init__(
        self,
        speaker_encoder_path: str,
        vc_model_path: str,
        device: str = None
    ):
        """
        Initialize voice converter.
        
        Args:
            speaker_encoder_path: Path to trained speaker encoder checkpoint
            vc_model_path: Path to trained VC model checkpoint
            device: Device to run on (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading models on device: {self.device}")
        
        # Load speaker encoder
        self.speaker_encoder = SpeakerEncoder(n_mels=80, embedding_dim=256)
        checkpoint = torch.load(speaker_encoder_path, map_location=self.device)
        self.speaker_encoder.load_state_dict(checkpoint['model_state_dict'])
        self.speaker_encoder = self.speaker_encoder.to(self.device)
        self.speaker_encoder.eval()
        print("✓ Speaker encoder loaded")
        
        # Load VC model
        self.vc_model = VoiceConversionModel(n_mels=80)
        checkpoint = torch.load(vc_model_path, map_location=self.device)
        self.vc_model.load_state_dict(checkpoint['model_state_dict'])
        self.vc_model = self.vc_model.to(self.device)
        self.vc_model.eval()
        print("✓ VC model loaded")
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(sample_rate=16000)
        
        print("✓ Voice converter ready!")
    
    def extract_speaker_embedding(
        self,
        audio_path: str,
        save_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract speaker embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            save_path: Optional path to save embedding
            
        Returns:
            embedding: Speaker embedding (256,)
        """
        # Load audio
        audio, sr = sf.read(audio_path)
        
        # Preprocess
        audio, sr = main_preprocessing_pipeline(
            audio, sr, target_sr=16000,
            apply_vad=True, normalize=True
        )
        
        # Extract mel-spectrogram
        mel = self.feature_extractor.extract_mel_spectrogram(audio)
        mel = torch.from_numpy(mel).unsqueeze(0).float().to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.speaker_encoder(mel)
        
        # Save if requested
        if save_path:
            torch.save(embedding.cpu(), save_path)
            print(f"✓ Speaker embedding saved to: {save_path}")
        
        return embedding
    
    def convert(
        self,
        source_audio: str,
        target_speaker_embedding: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Convert source audio to target speaker's voice.
        
        Args:
            source_audio: Path to source audio file
            target_speaker_embedding: Path to target speaker embedding (.pt file)
            output_path: Optional path to save converted audio
            
        Returns:
            converted_audio: Converted audio waveform
        """
        print(f"Converting: {source_audio}")
        
        # Load source audio
        audio, sr = sf.read(source_audio)
        audio, sr = main_preprocessing_pipeline(
            audio, sr, target_sr=16000,
            apply_vad=True, normalize=True
        )
        
        # Extract source mel-spectrogram
        source_mel = self.feature_extractor.extract_mel_spectrogram(audio)
        source_mel = torch.from_numpy(source_mel).unsqueeze(0).float().to(self.device)
        
        # Load target speaker embedding
        if isinstance(target_speaker_embedding, str):
            target_emb = torch.load(target_speaker_embedding, map_location=self.device)
        else:
            target_emb = target_speaker_embedding
        
        # Ensure correct shape
        if target_emb.dim() == 1:
            target_emb = target_emb.unsqueeze(0)
        
        # Voice conversion
        with torch.no_grad():
            converted_mel = self.vc_model(source_mel, target_emb)
        
        # Convert mel back to audio
        # Note: In practice, you'd use a vocoder (e.g., HiFi-GAN, WaveGlow)
        # For now, we'll use Griffin-Lim as a simple vocoder
        converted_mel_np = converted_mel.squeeze(0).cpu().numpy()
        converted_audio = self._mel_to_audio(converted_mel_np)
        
        # Save if requested
        if output_path:
            sf.write(output_path, converted_audio, 16000)
            print(f"✓ Converted audio saved to: {output_path}")
        
        print("✓ Conversion complete!")
        return converted_audio
    
    def _mel_to_audio(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Convert mel-spectrogram to audio using Griffin-Lim.
        
        Args:
            mel_spec: Mel-spectrogram (n_mels, time)
            
        Returns:
            audio: Audio waveform
        """
        import librosa
        
        # Inverse mel scaling
        mel_basis = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=80)
        mel_basis_inv = np.linalg.pinv(mel_basis)
        spec = np.dot(mel_basis_inv, np.exp(mel_spec))
        
        # Griffin-Lim
        audio = librosa.griffinlim(
            spec,
            n_iter=32,
            hop_length=160,
            win_length=1024
        )
        
        return audio
    
    def convert_batch(
        self,
        source_files: list,
        target_speaker_embedding: str,
        output_dir: str
    ):
        """
        Convert multiple audio files.
        
        Args:
            source_files: List of source audio file paths
            target_speaker_embedding: Target speaker embedding path
            output_dir: Output directory for converted files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting {len(source_files)} files...")
        
        for i, source_file in enumerate(source_files, 1):
            source_path = Path(source_file)
            output_file = output_path / f"converted_{source_path.stem}.wav"
            
            print(f"\n[{i}/{len(source_files)}] {source_path.name}")
            self.convert(
                source_audio=str(source_path),
                target_speaker_embedding=target_speaker_embedding,
                output_path=str(output_file)
            )
        
        print(f"\n✓ All files converted! Saved to: {output_dir}")


def quick_convert(
    source_audio: str,
    target_speaker: str,
    output_audio: str,
    speaker_encoder_path: str = "checkpoints/speaker_encoder/speaker_encoder_epoch_100.pt",
    vc_model_path: str = "checkpoints/voice_conversion/voice_conversion_epoch_200.pt"
):
    """
    Quick one-line voice conversion.
    
    Args:
        source_audio: Source audio file
        target_speaker: Target speaker embedding (.pt file)
        output_audio: Output audio file
        speaker_encoder_path: Speaker encoder checkpoint
        vc_model_path: VC model checkpoint
    """
    converter = VoiceConverter(speaker_encoder_path, vc_model_path)
    converter.convert(source_audio, target_speaker, output_audio)


if __name__ == "__main__":
    print("Voice Conversion Inference Pipeline")
    print("=" * 60)
    print()
    print("Usage:")
    print("  converter = VoiceConverter(")
    print("      speaker_encoder_path='checkpoints/speaker.pt',")
    print("      vc_model_path='checkpoints/vc.pt'")
    print("  )")
    print()
    print("  # Extract speaker embedding")
    print("  converter.extract_speaker_embedding(")
    print("      audio_path='lebron.wav',")
    print("      save_path='lebron_embedding.pt'")
    print("  )")
    print()
    print("  # Convert voice")
    print("  converter.convert(")
    print("      source_audio='my_voice.wav',")
    print("      target_speaker_embedding='lebron_embedding.pt',")
    print("      output_path='my_voice_as_lebron.wav'")
    print("  )")
