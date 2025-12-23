"""
Voice conversion CLI tool.

Convert your voice to sound like a target speaker.

Usage:
    # Single file
    python convert.py --input my_voice.wav --target lebron.pt --output my_voice_as_lebron.wav
    
    # Batch processing
    python convert.py --input "recordings/*.wav" --target lebron.pt --output converted/
"""

import argparse
from pathlib import Path
from glob import glob

from inference import VoiceConverter


def main():
    parser = argparse.ArgumentParser(
        description="Voice Conversion - Convert your voice to a target speaker"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input audio file(s). Can be single file or glob pattern'
    )
    parser.add_argument(
        '--target',
        type=str,
        required=True,
        help='Target speaker embedding (.pt file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output audio file or directory (for batch processing)'
    )
    parser.add_argument(
        '--speaker-encoder',
        type=str,
        default='checkpoints/speaker_encoder/speaker_encoder_epoch_100.pt',
        help='Speaker encoder checkpoint'
    )
    parser.add_argument(
        '--vc-model',
        type=str,
        default='checkpoints/voice_conversion/voice_conversion_epoch_200.pt',
        help='VC model checkpoint'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print(" Voice Conversion".center(70))
    print("="*70)
    print()
    
    # Load models
    print("Loading models...")
    converter = VoiceConverter(args.speaker_encoder, args.vc_model)
    print()
    
    # Handle glob patterns
    input_files = glob(args.input)
    if not input_files:
        input_files = [args.input]
    
    # Validate
    for f in input_files:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            exit(1)
    
    if not Path(args.target).exists():
        print(f"Error: Target speaker embedding not found: {args.target}")
        exit(1)
    
    # Single file or batch?
    if len(input_files) == 1:
        # Single file conversion
        print(f"Converting: {input_files[0]}")
        print(f"Target speaker: {args.target}")
        print(f"Output: {args.output}")
        print()
        
        converter.convert(
            source_audio=input_files[0],
            target_speaker_embedding=args.target,
            output_path=args.output
        )
    else:
        # Batch conversion
        print(f"Batch converting {len(input_files)} files...")
        print(f"Target speaker: {args.target}")
        print(f"Output directory: {args.output}")
        print()
        
        converter.convert_batch(
            source_files=input_files,
            target_speaker_embedding=args.target,
            output_dir=args.output
        )
    
    print()
    print("="*70)
    print("✅ Conversion complete!".center(70))
    print("="*70)


if __name__ == "__main__":
    main()
