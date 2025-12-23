"""
PyTorch Dataset for multi-speaker voice conversion.

Handles:
1. Loading audio files from multiple speakers
2. Feature extraction and caching
3. Data augmentation
4. Batching for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import pickle
import random

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from signal_processing.preprocessing import main_preprocessing_pipeline
from signal_processing.features import FeatureExtractor


class VoiceConversionDataset(Dataset):
    """
    Dataset for multi-speaker voice conversion.
    
    Structure:
        data/
        ├── speaker1/
        │   ├── audio001.wav
        │   ├── audio002.wav
        │   └── ...
        ├── speaker2/
        │   └── ...
        └── ...
    """
    
    def __init__(
        self,
        data_root: str,
        sample_rate: int = 16000,
        segment_length: int = 16000,  # 1 second
        cache_dir: Optional[str] = None,
        preprocess: bool = True,
        augment: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing speaker subdirectories
            sample_rate: Target sample rate
            segment_length: Length of audio segments in samples
            cache_dir: Directory for caching features (None = no caching)
            preprocess: Whether to apply preprocessing
            augment: Whether to apply data augmentation
        """
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preprocess = preprocess
        self.augment = augment
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        
        # Create cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.audio_files, self.speaker_ids = self._load_dataset()
        self.num_speakers = len(set(self.speaker_ids))
        
        print(f"Loaded {len(self.audio_files)} audio files from {self.num_speakers} speakers")
    
    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        """
        Load audio file paths and speaker IDs.
        
        Returns:
            audio_files: List of audio file paths
            speaker_ids: List of speaker IDs (integers)
        """
        audio_files = []
        speaker_ids = []
        
        # Get all speaker directories
        speaker_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        for speaker_id, speaker_dir in enumerate(speaker_dirs):
            # Get all wav files in speaker directory
            wav_files = list(speaker_dir.glob("*.wav"))
            
            audio_files.extend(wav_files)
            speaker_ids.extend([speaker_id] * len(wav_files))
        
        return audio_files, speaker_ids
    
    def __len__(self) -> int:
        """Return number of audio files."""
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            sample: Dictionary containing:
                - mel: Mel-spectrogram (n_mels, time)
                - f0: F0 contour (time,)
                - speaker_id: Speaker ID (integer)
                - audio: Raw audio (for debugging)
        """
        audio_path = self.audio_files[idx]
        speaker_id = self.speaker_ids[idx]
        
        # Try to load from cache
        if self.cache_dir:
            cache_path = self.cache_dir / f"{audio_path.stem}_features.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    features = pickle.load(f)
                    return self._prepare_sample(features, speaker_id)
        
        # Load audio
        audio = self._load_audio(audio_path)
        
        # Extract features
        features = self.feature_extractor.extract_all_features(audio)
        
        # Cache features
        if self.cache_dir:
            cache_path = self.cache_dir / f"{audio_path.stem}_features.pkl"
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        
        return self._prepare_sample(features, speaker_id)
    
    def _load_audio(self, audio_path: Path) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            audio: Preprocessed audio array
        """
        import soundfile as sf
        
        # Load audio
        audio, sr = sf.read(str(audio_path))
        
        # Preprocess
        if self.preprocess:
            audio, _ = main_preprocessing_pipeline(
                audio, sr, self.sample_rate,
                apply_vad=True, normalize=True
            )
        
        # Random segment extraction
        if len(audio) > self.segment_length:
            start = random.randint(0, len(audio) - self.segment_length)
            audio = audio[start:start + self.segment_length]
        else:
            # Pad if too short
            audio = np.pad(audio, (0, max(0, self.segment_length - len(audio))))
        
        # Data augmentation
        if self.augment:
            audio = self._augment_audio(audio)
        
        return audio
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation.
        
        Args:
            audio: Input audio
            
        Returns:
            augmented: Augmented audio
        """
        # Random gain (volume adjustment)
        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            audio = audio * gain
        
        # Random noise injection
        if random.random() < 0.3:
            noise = np.random.randn(*audio.shape) * 0.005
            audio = audio + noise
        
        # Random time stretching (future enhancement)
        # Would require librosa or similar
        
        return audio
    
    def _prepare_sample(
        self,
        features: Dict[str, np.ndarray],
        speaker_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare sample for training.
        
        Args:
            features: Extracted features
            speaker_id: Speaker ID
            
        Returns:
            sample: Dictionary of tensors
        """
        # Convert to tensors
        sample = {
            'mel': torch.from_numpy(features['mel']).float(),
            'f0': torch.from_numpy(features['f0_continuous']).float(),
            'vuv': torch.from_numpy(features['vuv']).float(),
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long)
        }
        
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Handles variable-length sequences by padding.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        batched: Dictionary of batched tensors
    """
    # Get max length in batch
    max_len = max(sample['mel'].shape[1] for sample in batch)
    
    # Pad all samples to max length
    mels = []
    f0s = []
    vuvs = []
    speaker_ids = []
    
    for sample in batch:
        mel = sample['mel']
        f0 = sample['f0']
        vuv = sample['vuv']
        
        # Pad
        pad_len = max_len - mel.shape[1]
        if pad_len > 0:
            mel = torch.nn.functional.pad(mel, (0, pad_len))
            f0 = torch.nn.functional.pad(f0, (0, pad_len))
            vuv = torch.nn.functional.pad(vuv, (0, pad_len))
        
        mels.append(mel)
        f0s.append(f0)
        vuvs.append(vuv)
        speaker_ids.append(sample['speaker_id'])
    
    # Stack into batch
    batched = {
        'mel': torch.stack(mels),
        'f0': torch.stack(f0s),
        'vuv': torch.stack(vuvs),
        'speaker_id': torch.stack(speaker_ids)
    }
    
    return batched


def create_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory (optional)
        batch_size: Batch size
        num_workers: Number of worker processes
        cache_dir: Feature cache directory
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader (or None)
    """
    # Training dataset
    train_dataset = VoiceConversionDataset(
        data_root=train_dir,
        cache_dir=cache_dir,
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Validation dataset
    val_loader = None
    if val_dir:
        val_dataset = VoiceConversionDataset(
            data_root=val_dir,
            cache_dir=cache_dir,
            augment=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Voice Conversion Dataset Module")
    print("=" * 60)
    print()
    
    # Example usage
    print("Dataset Features:")
    print("  - Multi-speaker audio loading")
    print("  - Automatic preprocessing")
    print("  - Feature extraction and caching")
    print("  - Data augmentation (gain, noise)")
    print("  - Variable-length sequence handling")
    print()
    print("Usage:")
    print("  dataset = VoiceConversionDataset('data/train')")
    print("  train_loader, val_loader = create_dataloaders('data/train', 'data/val')")
