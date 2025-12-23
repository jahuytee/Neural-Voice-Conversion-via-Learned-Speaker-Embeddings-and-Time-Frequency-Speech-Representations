"""
Training script for speaker encoder.

Trains the speaker encoder using GE2E loss on multi-speaker dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
from tqdm import tqdm
from tabulate import tabulate

# Import our modules
from models.speaker_encoder import SpeakerEncoder, GE2ELoss
from training.dataset import VoiceConversionDataset, create_dataloaders
from training.train_utils import TrainingMonitor, validate


def train_speaker_encoder(
    data_dir: str,
    val_dir: str = None,
    output_dir: str = "checkpoints/speaker_encoder",
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    device: str = None,
    resume: str = None
):
    """
    Train speaker encoder with GE2E loss.
    
    Args:
        data_dir: Training data directory
        val_dir: Validation data directory
        output_dir: Output directory for checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: Device to train on (cuda/cpu)
        resume: Path to checkpoint to resume from
    """
    # Setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(" Training Speaker Encoder".center(70))
    print("="*70)
    print()
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print()
    
    # Create model
    model = SpeakerEncoder(n_mels=80, embedding_dim=256)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Loss function
    loss_fn = GE2ELoss()
    loss_fn = loss_fn.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
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
        model.load_state_dict(checkpoint['model_state_dict'])
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
    monitor = TrainingMonitor("speaker_encoder", save_dir=output_dir)
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        
        print(f"Epoch {epoch+1}/{num_epochs} " + "-"*50)
        
        # Progress bar (like CBOW style)
        progress_bar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1}",
            colour="#006400"  # Dark green
        )
        
        for batch in progress_bar:
            # Move to device
            mel = batch['mel'].to(device)  # (batch, n_mels, time)
            speaker_ids = batch['speaker_id'].to(device)
            
            # Organize into (num_speakers, utterances_per_speaker, ...)
            # For GE2E, we need multiple utterances per speaker
            unique_speakers = torch.unique(speaker_ids)
            num_speakers = len(unique_speakers)
            
            # Skip if not enough speakers in batch
            if num_speakers < 2:
                continue
            
            # Group by speaker
            embeddings_list = []
            for spk_id in unique_speakers:
                mask = speaker_ids == spk_id
                spk_mels = mel[mask]
                
                # Get embeddings for this speaker
                spk_embeddings = model(spk_mels)  # (n_utterances, embedding_dim)
                embeddings_list.append(spk_embeddings)
            
            # Pad to same number of utterances
            max_utterances = max(e.shape[0] for e in embeddings_list)
            padded_embeddings = []
            for emb in embeddings_list:
                if emb.shape[0] < max_utterances:
                    # Repeat last embedding to pad
                    padding = emb[-1:].repeat(max_utterances - emb.shape[0], 1)
                    emb = torch.cat([emb, padding], dim=0)
                padded_embeddings.append(emb[:max_utterances])
            
            # Stack: (num_speakers, max_utterances, embedding_dim)
            embeddings = torch.stack(padded_embeddings)
            
            # Compute GE2E loss
            loss = loss_fn(embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            monitor.log_batch(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Epoch summary
        avg_loss = total_loss / len(train_loader)
        print(f"↪ Total Loss: {total_loss:.4f}, Avg Loss: {avg_loss:.4f}")
        
        # Validation
        val_loss = None
        if val_loader and (epoch + 1) % 5 == 0:
            model.eval()
            val_total = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", colour="#FFA500"):
                    mel = batch['mel'].to(device)
                    speaker_ids = batch['speaker_id'].to(device)
                    
                    # Similar processing as training
                    unique_speakers = torch.unique(speaker_ids)
                    if len(unique_speakers) < 2:
                        continue
                    
                    embeddings_list = []
                    for spk_id in unique_speakers:
                        mask = speaker_ids == spk_id
                        spk_mels = mel[mask]
                        spk_embeddings = model(spk_mels)
                        embeddings_list.append(spk_embeddings)
                    
                    max_utterances = max(e.shape[0] for e in embeddings_list)
                    padded_embeddings = []
                    for emb in embeddings_list:
                        if emb.shape[0] < max_utterances:
                            padding = emb[-1:].repeat(max_utterances - emb.shape[0], 1)
                            emb = torch.cat([emb, padding], dim=0)
                        padded_embeddings.append(emb[:max_utterances])
                    
                    embeddings = torch.stack(padded_embeddings)
                    loss = loss_fn(embeddings)
                    val_total += loss.item()
            
            val_loss = val_total / len(val_loader)
            print(f"↪ Validation Loss: {val_loss:.4f}")
        
        # Log epoch
        monitor.log_epoch(epoch, avg_loss, val_loss)
        
        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"↪ Learning rate: {current_lr:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            monitor.save_checkpoint(epoch, model, optimizer, {
                'loss_fn_state': loss_fn.state_dict(),
                'scheduler_state': scheduler.state_dict()
            })
        
        # Plot losses
        if (epoch + 1) % 5 == 0:
            monitor.plot_losses(show=False, save=True)
        
        print()
    
    # Final save
    print("Training complete!")
    monitor.save_checkpoint(num_epochs - 1, model, optimizer)
    monitor.plot_losses(show=False, save=True)
    
    print(f"\nCheckpoints saved to: {output_dir}")
    print("✅ Speaker encoder training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train speaker encoder")
    parser.add_argument('--data', type=str, required=True, help='Training data directory')
    parser.add_argument('--val', type=str, default=None, help='Validation data directory')
    parser.add_argument('--output', type=str, default='checkpoints/speaker_encoder', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train_speaker_encoder(
        data_dir=args.data,
        val_dir=args.val,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        resume=args.resume
    )
