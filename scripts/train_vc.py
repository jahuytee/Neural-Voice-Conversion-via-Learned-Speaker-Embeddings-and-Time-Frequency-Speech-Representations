"""
Training script for voice conversion model.

Trains complete VC system (content encoder + decoder) with frozen speaker encoder.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
from tqdm import tqdm

# Import our modules
from models.speaker_encoder import SpeakerEncoder
from models.voice_conversion import VoiceConversionModel
from training.dataset import create_dataloaders
from training.losses import VoiceConversionLoss
from training.train_utils import TrainingMonitor


def train_voice_conversion(
    data_dir: str,
    speaker_encoder_path: str,
    val_dir: str = None,
    output_dir: str = "checkpoints/voice_conversion",
    num_epochs: int = 200,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = None,
    resume: str = None
):
    """
    Train voice conversion model.
    
    Args:
        data_dir: Training data directory
        speaker_encoder_path: Path to trained speaker encoder checkpoint
        val_dir: Validation data directory
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: Device to train on
        resume: Path to checkpoint to resume from
    """
    # Setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(" Training Voice Conversion Model".center(70))
    print("="*70)
    print()
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print()
    
    # Load speaker encoder (frozen)
    print(f"Loading speaker encoder from: {speaker_encoder_path}")
    speaker_encoder = SpeakerEncoder(n_mels=80, embedding_dim=256)
    checkpoint = torch.load(speaker_encoder_path, map_location=device)
    speaker_encoder.load_state_dict(checkpoint['model_state_dict'])
    speaker_encoder = speaker_encoder.to(device)
    speaker_encoder.eval()
    
    # Freeze speaker encoder
    for param in speaker_encoder.parameters():
        param.requires_grad = False
    
    print("✓ Speaker encoder loaded and frozen")
    print()
    
    # Create VC model
    vc_model = VoiceConversionModel(
        n_mels=80,
        content_channels=[128, 256, 512],
        speaker_dim=256,
        decoder_channels=[512, 256, 128]
    )
    vc_model = vc_model.to(device)
    
    trainable_params = sum(p.numel() for p in vc_model.parameters() if p.requires_grad)
    print(f"VC Model trainable parameters: {trainable_params:,}")
    print()
    
    # Loss function
    loss_fn = VoiceConversionLoss(
        recon_weight=1.0,
        speaker_weight=0.1,
        f0_weight=0.5
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        vc_model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if resume:
        print(f"Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume)
        vc_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        print()
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        train_dir=data_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        num_workers=4
    )
    print(f"Training batches: {len(train_loader)}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
    print()
    
    # Training monitor
    monitor = TrainingMonitor("voice_conversion", save_dir=output_dir)
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        vc_model.train()
        total_loss = 0
        total_recon = 0
        total_spk = 0
        total_f0 = 0
        
        print(f"Epoch {epoch+1}/{num_epochs} " + "-"*50)
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1}",
            colour="#006400"
        )
        
        for batch in progress_bar:
            # Move to device
            mel = batch['mel'].to(device)  # (batch, n_mels, time)
            f0 = batch['f0'].to(device)
            vuv = batch['vuv'].to(device)
            
            # Extract source speaker embeddings (frozen)
            with torch.no_grad():
                source_speaker_emb = speaker_encoder(mel)
            
            # Voice conversion (self-reconstruction for now)
            # In full training, you'd have source and target pairs
            converted_mel = vc_model(mel, source_speaker_emb)
            
            # Extract converted speaker embedding for loss
            with torch.no_grad():
                converted_speaker_emb = speaker_encoder(converted_mel)
            
            # Compute losses
            losses = loss_fn(
                predicted_mel=converted_mel,
                target_mel=mel,
                converted_speaker_emb=converted_speaker_emb,
                target_speaker_emb=source_speaker_emb,
                predicted_f0=f0,  # Would extract from converted in real scenario
                target_f0=f0,
                voiced_mask=vuv
            )
            
            loss = losses['total']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(vc_model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_recon += losses['recon'].item()
            if 'speaker' in losses:
                total_spk += losses['speaker'].item()
            if 'f0' in losses:
                total_f0 += losses['f0'].item()
            
            monitor.log_batch(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{losses['recon'].item():.4f}"
            })
        
        # Epoch summary
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        avg_spk = total_spk / len(train_loader) if total_spk > 0 else 0
        avg_f0 = total_f0 / len(train_loader) if total_f0 > 0 else 0
        
        print(f"↪ Avg Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, Spk: {avg_spk:.4f}, F0: {avg_f0:.4f})")
        
        # Validation
        val_loss = None
        if val_loader and (epoch + 1) % 5 == 0:
            vc_model.eval()
            val_total = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", colour="#FFA500"):
                    mel = batch['mel'].to(device)
                    f0 = batch['f0'].to(device)
                    vuv = batch['vuv'].to(device)
                    
                    source_speaker_emb = speaker_encoder(mel)
                    converted_mel = vc_model(mel, source_speaker_emb)
                    converted_speaker_emb = speaker_encoder(converted_mel)
                    
                    losses = loss_fn(
                        predicted_mel=converted_mel,
                        target_mel=mel,
                        converted_speaker_emb=converted_speaker_emb,
                        target_speaker_emb=source_speaker_emb,
                        predicted_f0=f0,
                        target_f0=f0,
                        voiced_mask=vuv
                    )
                    
                    val_total += losses['total'].item()
            
            val_loss = val_total / len(val_loader)
            print(f"↪ Validation Loss: {val_loss:.4f}")
        
        # Log epoch
        monitor.log_epoch(epoch, avg_loss, val_loss)
        
        # Learning rate step
        scheduler.step()
        print(f"↪ Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            monitor.save_checkpoint(epoch, vc_model, optimizer, {
                'scheduler_state': scheduler.state_dict()
            })
        
        # Plot losses
        if (epoch + 1) % 5 == 0:
            monitor.plot_losses(show=False, save=True)
        
        print()
    
    # Final save
    print("Training complete!")
    monitor.save_checkpoint(num_epochs - 1, vc_model, optimizer)
    monitor.plot_losses(show=False, save=True)
    
    print(f"\nCheckpoints saved to: {output_dir}")
    print("✅ Voice conversion training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train voice conversion model")
    parser.add_argument('--data', type=str, required=True, help='Training data directory')
    parser.add_argument('--speaker-encoder', type=str, required=True, help='Trained speaker encoder checkpoint')
    parser.add_argument('--val', type=str, default=None, help='Validation data directory')
    parser.add_argument('--output', type=str, default='checkpoints/voice_conversion', help='Output directory')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train_voice_conversion(
        data_dir=args.data,
        speaker_encoder_path=args.speaker_encoder,
        val_dir=args.val,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        resume=args.resume
    )
