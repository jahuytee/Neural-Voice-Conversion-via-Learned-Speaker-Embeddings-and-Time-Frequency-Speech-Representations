"""
Extract speaker embedding from audio files.

Usage:
    python scripts/extract_speaker.py --audio lebron.wav --output lebron.pt
    python scripts/extract_speaker.py --audio "private_data/lebron/*.wav" --output lebron.pt
"""

import argparse
import torch
from pathlib import Path
from glob import glob

from inference import VoiceConverter


def extract_speaker_embedding(
    audio_files: list,
    output_path: str,
    speaker_encoder_path: str = "checkpoints/speaker_encoder/speaker_encoder_epoch_100.pt",
    vc_model_path: str = "checkpoints/voice_conversion/voice_conversion_epoch_200.pt"
):
    """
    Extract speaker embedding from audio file(s).
    
    If multiple files provided, averages their embeddings.
    
    Args:
        audio_files: List of audio file paths
        output_path: Output path for embedding (.pt file)
        speaker_encoder_path: Speaker encoder checkpoint
        vc_model_path: VC model checkpoint (needed for initialization)
    """
    print("="*70)
    print(" Speaker Embedding Extraction".center(70))
    print("="*70)
    print()
    
    # Load converter
    print("Loading models...")
    converter = VoiceConverter(speaker_encoder_path, vc_model_path)
    print()
    
    # Extract embeddings
    embeddings = []
    print(f"Extracting embeddings from {len(audio_files)} file(s)...")
    print()
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {Path(audio_file).name}")
        embedding = converter.extract_speaker_embedding(audio_file)
        embeddings.append(embedding)
    
    # Average embeddings if multiple files
    if len(embeddings) > 1:
        print()
        print(f"Averaging {len(embeddings)} embeddings...")
        avg_embedding = torch.stack(embeddings).mean(dim=0)
    else:
        avg_embedding = embeddings[0]
    
    # Save
    torch.save(avg_embedding.cpu(), output_path)
    print()
    print(f"✅ Speaker embedding saved to: {output_path}")
    print(f"   Shape: {avg_embedding.shape}")
    print(f"   Norm: {avg_embedding.norm().item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract speaker embedding from audio files"
    )
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Audio file(s). Can be single file or glob pattern (e.g., "data/*.wav")'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for speaker embedding (.pt file)'
    )
    parser.add_argument(
        '--speaker-encoder',
        type=str,
        default='checkpoints/speaker_encoder/speaker_encoder_epoch_100.pt',
        help='Speaker encoder checkpoint path'
    )
    parser.add_argument(
        '--vc-model',
        type=str,
        default='checkpoints/voice_conversion/voice_conversion_epoch_200.pt',
        help='VC model checkpoint path'
    )
    
    args = parser.parse_args()
    
    # Handle glob patterns
    audio_files = glob(args.audio)
    if not audio_files:
        # Try as single file
        audio_files = [args.audio]
    
    # Validate files exist
    for f in audio_files:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            exit(1)
    
    # Extract
    extract_speaker_embedding(
        audio_files=audio_files,
        output_path=args.output,
        speaker_encoder_path=args.speaker_encoder,
        vc_model_path=args.vc_model
    )
