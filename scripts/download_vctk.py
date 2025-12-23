"""
Download and prepare VCTK dataset for training.

VCTK Corpus: 110 speakers, ~44 hours of speech
https://datashare.ed.ac.uk/handle/10283/3443
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse


def download_file(url: str, output_path: str):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_vctk(output_dir: str = "data"):
    """
    Download VCTK corpus.
    
    Args:
        output_dir: Output directory for dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(" Downloading VCTK Corpus".center(70))
    print("="*70)
    print()
    print("Dataset: VCTK Corpus 0.92")
    print("Speakers: 110")
    print("Duration: ~44 hours")
    print("Size: ~11 GB")
    print()
    
    # VCTK download URL
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    zip_path = output_path / "VCTK-Corpus-0.92.zip"
    
    # Download
    if zip_path.exists():
        print(f"✓ Archive already exists: {zip_path}")
    else:
        print("Downloading VCTK corpus...")
        print(f"URL: {url}")
        print(f"Destination: {zip_path}")
        print()
        
        try:
            download_file(url, str(zip_path))
            print()
            print("✓ Download complete!")
        except Exception as e:
            print(f"✗ Download failed: {e}")
            print()
            print("Alternative: Download manually from:")
            print("  https://datashare.ed.ac.uk/handle/10283/3443")
            return
    
    # Extract
    extract_path = output_path / "vctk"
    if extract_path.exists():
        print(f"✓ Dataset already extracted: {extract_path}")
    else:
        print()
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)
        print("✓ Extraction complete!")
    
    # Organize structure
    print()
    print("Organizing dataset structure...")
    organize_vctk(extract_path)
    
    print()
    print("="*70)
    print("✅ VCTK dataset ready!".center(70))
    print("="*70)
    print()
    print(f"Location: {extract_path}")
    print(f"Speakers: {len(list((extract_path / 'wav48_silence_trimmed').glob('p*')))}")
    print()
    print("Next steps:")
    print("  1. Train speaker encoder:")
    print(f"     python scripts/train_speaker_encoder.py --data {extract_path}")
    print()
    print("  2. Train voice conversion:")
    print("     python scripts/train_vc.py --data {extract_path} \\")
    print("       --speaker-encoder checkpoints/speaker_encoder_epoch_100.pt")


def organize_vctk(vctk_path: Path):
    """Organize VCTK into expected structure."""
    # VCTK structure is already good, just verify
    audio_dir = vctk_path / "wav48_silence_trimmed"
    
    if not audio_dir.exists():
        print("⚠ Warning: Expected audio directory not found")
        print(f"   Looking for: {audio_dir}")
        return
    
    # Count speakers
    speakers = list(audio_dir.glob("p*"))
    print(f"✓ Found {len(speakers)} speakers")
    
    # Verify some files exist
    total_files = sum(len(list(spk.glob("*.flac"))) for spk in speakers[:5])
    print(f"✓ Verified audio files exist (sampled {total_files} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download VCTK corpus")
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory (default: data/)'
    )
    
    args = parser.parse_args()
    
    download_vctk(args.output)
