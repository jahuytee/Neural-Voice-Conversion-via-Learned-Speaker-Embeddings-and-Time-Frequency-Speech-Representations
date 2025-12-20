"""
Training utilities with progress tracking (similar to CBOW assignment).

Includes:
- Training loops with tqdm progress bars
- Loss tracking and plotting
- Validation metrics
- Checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TrainingMonitor:
    """
    Monitor training progress with nice displays.
    Similar to your CBOW assignment style.
    """
    
    def __init__(self, model_name: str, save_dir: str = "checkpoints"):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Loss tracking (like your batch_losses)
        self.batch_losses = []
        self.batch_indices = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []
        self.global_batch_count = 0
        
        # Metrics
        self.metrics = {
            "train": {"losses": []},
            "val": {"losses": [], "accuracies": []}
        }
    
    def log_batch(self, loss: float):
        """Log batch loss (called every training step)."""
        self.batch_losses.append(loss)
        self.batch_indices.append(self.global_batch_count)
        self.global_batch_count += 1
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None):
        """Log epoch metrics."""
        self.epoch_train_losses.append(train_loss)
        self.metrics["train"]["losses"].append(train_loss)
        
        if val_loss is not None:
            self.epoch_val_losses.append(val_loss)
            self.metrics["val"]["losses"].append(val_loss)
        
        # Print table (like your CBOW)
        table_data = [
            ["Epoch", epoch + 1],
            ["Train Loss", f"{train_loss:.4f}"]
        ]
        if val_loss is not None:
            table_data.append(["Val Loss", f"{val_loss:.4f}"])
        
        print(tabulate(table_data, tablefmt="simple_grid"))
    
    def plot_losses(self, show: bool = True, save: bool = True):
        """Plot training curves (like your plot_loss function)."""
        plt.figure(figsize=(12, 5))
        
        # Batch losses
        plt.subplot(1, 2, 1)
        plt.plot(self.batch_indices, self.batch_losses, alpha=0.6, linewidth=0.5)
        # Smoothed curve
        window = min(100, len(self.batch_losses) // 10)
        if window > 1:
            smoothed = np.convolve(
                self.batch_losses,
                np.ones(window) / window,
                mode='valid'
            )
            plt.plot(smoothed, color='red', linewidth=2, label='Smoothed')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss (Per Batch)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Epoch losses
        plt.subplot(1,2, 2)
        epochs = range(1, len(self.epoch_train_losses) + 1)
        plt.plot(epochs, self.epoch_train_losses, 'o-', label='Train Loss')
        if self.epoch_val_losses:
            plt.plot(epochs, self.epoch_val_losses, 's-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress (Per Epoch)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"{self.model_name}_loss_curves.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Loss plot saved to: {save_path}")
        
        if show:
            plt.show()
        plt.close()
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        additional_data: Optional[Dict] = None
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': self.epoch_train_losses[-1] if self.epoch_train_losses else None,
            'metrics': self.metrics,
            'batch_count': self.global_batch_count
        }
        
        if additional_data:
            checkpoint.update(additional_data)
        
        save_path = self.save_dir / f"{self.model_name}_epoch_{epoch+1}.pt"
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved: {save_path}")


def train_speaker_encoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 16,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    max_norm: float = 2.0,
    checkpoint_every: int = 5,
    validate_every: int = 2,
    model_name: str = "speaker_encoder"
):
    """
    Train speaker encoder with progress bars (CBOW style).
    
    Args:
        model: Speaker encoder model
        train_loader: Training data loader
        val_loader: Validation data loader
        loss_fn: Loss function (e.g., TripletLoss)
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on
        max_norm: Gradient clipping threshold
        checkpoint_every: Save checkpoint every N epochs
        validate_every: Run validation every N epochs
        model_name: Name for saving checkpoints
    """
    model = model.to(device)
    monitor = TrainingMonitor(model_name)
    
    print(f"Training batches available: {len(train_loader)}")
    print(f"Device: {device}")
    print()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        print(f"Epoch {epoch+1} {'-'*54}")
        
        # Progress bar (like your CBOW)
        progress_bar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch+1}",
            leave=True,
            colour="#006400"  # Dark green, like yours
        )
        
        for batch_data in progress_bar:
            # Move to device
            if isinstance(batch_data, (list, tuple)):
                batch_data = [x.to(device) for x in batch_data]
            else:
                batch_data = batch_data.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            loss = loss_fn(outputs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (like your CBOW)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            monitor.log_batch(loss.item())
            
            # Update progress bar with current loss
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Epoch summary
        avg_train_loss = total_loss / len(train_loader)
        print(f"↪ Total Loss: {total_loss:.4f}, Avg Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        val_loss = None
        if val_loader and (epoch + 1) % validate_every == 0:
            val_loss = validate(model, val_loader, loss_fn, device)
            print(f"↪ Validation Loss: {val_loss:.4f}")
        
        # Log epoch
        monitor.log_epoch(epoch, avg_train_loss, val_loss)
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            monitor.save_checkpoint(epoch, model, optimizer)
        
        # Plot losses periodically
        if (epoch + 1) % validate_every == 0:
            monitor.plot_losses(show=False, save=True)
    
    # Final save
    monitor.save_checkpoint(num_epochs - 1, model, optimizer)
    monitor.plot_losses(show=True, save=True)
    
    print("\n" + "=" * 70)
    print(" Training Complete!".center(70))
    print("=" * 70)
    
    return monitor.metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: str
) -> float:
    """
    Run validation.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation", colour="#FFA500"):
            if isinstance(batch_data, (list, tuple)):
                batch_data = [x.to(device) for x in batch_data]
            else:
                batch_data = batch_data.to(device)
            
            outputs = model(batch_data)
            loss = loss_fn(outputs)
            total_loss += loss.item()
    
    model.train()  # Back to training mode
    return total_loss / len(val_loader)


def train_voice_conversion(
    content_encoder: nn.Module,
    speaker_encoder: nn.Module,
    decoder: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
):
    """
    Train complete voice conversion model.
    
    Similar structure to speaker encoder training,
    but with content/speaker disentanglement.
    """
    content_encoder = content_encoder.to(device)
    decoder = decoder.to(device)
    speaker_encoder = speaker_encoder.to(device)
    speaker_encoder.eval()  # Freeze speaker encoder
    
    monitor = TrainingMonitor("voice_conversion")
    
    print(f"Training Voice Conversion Model")
    print(f"Batches: {len(train_loader)}, Device: {device}")
    print()
    
    for epoch in range(num_epochs):
        content_encoder.train()
        decoder.train()
        total_loss = 0
        
        print(f"Epoch {epoch+1} {'-'*54}")
        
        progress_bar = tqdm(
            train_loader,
            desc=f"VC Training Epoch {epoch+1}",
            colour="#006400"
        )
        
        for mel_source, mel_target, speaker_id in progress_bar:
            mel_source = mel_source.to(device)
            mel_target = mel_target.to(device)
            speaker_id = speaker_id.to(device)
            
            # Extract content (what is said)
            z_content = content_encoder(mel_source)
            
            # Extract speaker embedding (who says it)
            with torch.no_grad():
                z_speaker = speaker_encoder(speaker_id)
            
            # Reconstruct with target speaker
            mel_reconstructed = decoder(z_content, z_speaker)
            
            # Compute loss
            loss = loss_fn(mel_reconstructed, mel_target)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(content_encoder.parameters()) + list(decoder.parameters()),
                max_norm=2.0
            )
            optimizer.step()
            
            total_loss += loss.item()
            monitor.log_batch(loss.item())
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        monitor.log_epoch(epoch, avg_loss)
        
        if (epoch + 1) % 5 == 0:
            monitor.save_checkpoint(
                epoch, decoder, optimizer,
                {'content_encoder': content_encoder.state_dict()}
            )
            monitor.plot_losses(show=False, save=True)
    
    return monitor.metrics


if __name__ == "__main__":
    print("Training Infrastructure for Voice Conversion")
    print("=" * 60)
    print()
    print("Features:")
    print("  ✓ tqdm progress bars (like CBOW)")
    print("  ✓ Batch and epoch loss tracking")
    print("  ✓ Gradient clipping")
    print("  ✓ Automatic checkpointing")
    print("  ✓ Loss curve plotting")
    print("  ✓ Validation monitoring")
    print()
    print("Functions:")
    print("  - train_speaker_encoder()")
    print("  - train_voice_conversion()")
    print("  - TrainingMonitor class")
