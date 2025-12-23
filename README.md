# Neural Voice Conversion
## Learned Speaker Embeddings + Time-Frequency Speech Representations

**A signals + ML heavy implementation of neural voice conversion.**

⚠️ **IMPORTANT: Ethical Use Only**
This system is built for research and educational purposes. Use only with:
- Your own voice
- Voices with explicit written consent
- Open datasets (VCTK, LibriTTS)

**DO NOT use for impersonation, deception, or unauthorized voice cloning.**

---

## Project Overview

This project implements a complete neural voice conversion pipeline:

**Signal Processing (DSP)**:
- Pre-emphasis filtering
- STFT with perceptual mel-filterbanks
- F0 extraction (WORLD vocoder)
- Spectral envelope analysis

**Machine Learning**:
- Content encoder (CNN + Instance Norm)
- Speaker embeddings (DVec/x-vector)
- Conditional decoder
- HiFi-GAN neural vocoder

**Key Features**:
- Disentangled content/speaker representations
- Multi-algorithm F0 tracking
- Adversarial training for naturalness
- Real-time capable architecture

---

## Installation

```bash
# Clone repository
git clone https://github.com/jahuytee/Neural-Voice-Conversion-via-Learned-Speaker-Embeddings-and-Time-Frequency-Speech-Representations.git
cd Neural-Voice-Conversion-via-Learned-Speaker-Embeddings-and-Time-Frequency-Speech-Representations

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Prepare Data (Legal Datasets)

```bash
# Download VCTK Corpus
python scripts/download_vctk.py

# Preprocess audio
python scripts/preprocess_audio.py --dataset vctk --output data/processed
```

### 2. Extract Features

```bash
# Extract mel-spectrograms + F0
python scripts/extract_features.py --input data/processed --output features/
```

### 3. Train Speaker Encoder

```bash
# Train speaker verification model
python training/train_speaker.py --config configs/speaker_encoder.yaml
```

### 4. Train Voice Conversion

```bash
# End-to-end VC training
python training/train_vc.py --config configs/vc_model.yaml
```

### 5. Convert Voice

```bash
# Perform voice conversion
python scripts/inference.py \
  --source path/to/source.wav \
  --target_speaker speaker_id \
  --output converted.wav
```

---

## Architecture

```
Source Audio
    ↓
[Preprocessing] → Pre-emphasis, Framing, VAD
    ↓
[Feature Extraction] → Mel-spectrogram (80 bins) + F0
    ↓
[Content Encoder] → Instance-normalized CNN → z_content
    ↓
[Speaker Encoder] → LSTM/TDNN → z_speaker (256-dim)
    ↓
[Decoder] → Conditional generation → Reconstructed Mel
    ↓
[Neural Vocoder] → HiFi-GAN → Output Audio
```

---

## Project Structure

```
voice-conversion/
├── signal_processing/       # DSP modules
│   ├── preprocessing.py     # Pre-emphasis, framing
│   ├── features.py          # Mel, F0 extraction
│   └── world_wrapper.py     # WORLD vocoder
├── models/                  # Neural network architectures
│   ├── content_encoder.py
│   ├── speaker_encoder.py
│   ├── decoder.py
│   └── vocoder.py
├── training/                # Training scripts
│   ├── dataset.py
│   ├── train_speaker.py
│   ├── train_vc.py
│   └── losses.py
├── evaluation/              # Metrics and testing
│   ├── metrics.py           # MCD, F0-RMSE
│   └── visualize.py
├── configs/                 # Configuration files
├── scripts/                 # Utility scripts
└── notebooks/               # Jupyter demos
```

---

## Technical Details

### Signal Processing
- **Sample Rate**: 16 kHz
- **Frame Length**: 25 ms (400 samples)
- **Hop Length**: 10 ms (160 samples)
- **FFT Size**: 1024
- **Mel Filters**: 80 (50-8000 Hz)
- **F0 Range**: 71-800 Hz

### Model Architecture
- **Content Encoder**: 3-layer CNN (512 dims)
- **Speaker Encoder**: 3-layer LSTM (256 dims)
- **Decoder**: 4-layer transpose CNN
- **Vocoder**: HiFi-GAN V1

### Training
- **Optimizer**: AdamW (β₁=0.9, β₂=0.999)
- **Learning Rate**: 2e-4 with cosine annealing
- **Batch Size**: 16
- **Total Steps**: ~200k

---

## Evaluation

**Objective Metrics**:
- Mel Cepstral Distortion (MCD) < 6.0 dB
- F0 RMSE < 15 Hz
- Speaker similarity > 0.8

**Subjective Metrics**:
- Mean Opinion Score (MOS) for quality
- Speaker similarity rating

---

## Legal & Ethical Usage

✅ **Permitted Uses**:
- Personal voice cloning (your own voice)
- Research with consented participants
- Training on open datasets (VCTK, LibriTTS, CMU Arctic)

❌ **Prohibited Uses**:
- Celebrity voice cloning without consent
- Impersonation or deception
- Commercial use without proper licensing
- Violation of right of publicity laws

**Disclaimer**: Users are solely responsible for compliance with applicable laws including right of publicity, biometric privacy regulations (GDPR, CCPA, BIPA), and deepfake disclosure requirements.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{voice_conversion_2024,
  author = {Jason Tran},
  title = {Neural Voice Conversion via Learned Speaker Embeddings and Time-Frequency Speech Representations},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/jahuytee/Neural-Voice-Conversion-via-Learned-Speaker-Embeddings-and-Time-Frequency-Speech-Representations}
}
```

---

## License

MIT License - See LICENSE file for details

**Additional Terms**: This software may not be used for unauthorized voice impersonation, identity fraud, or violation of celebrity rights of publicity.

---

## Acknowledgments

- VCTK Corpus (University of Edinburgh)
- LibriTTS Dataset
- WORLD Vocoder
- HiFi-GAN authors

---

## Contact

For questions or collaboration: [Your Email]

**Note**: Private experiments with celebrity voices are the user's legal responsibility. This public repository demonstrates the technical system using only legal, consented data.
