import torch
from torch.amp import autocast, GradScaler
import time
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file, load_file

from tacit.models.dit import TACITModel


class Trainer:
    """
    Trainer class with optimizations for high GPU utilization:
    - Automatic Mixed Precision (AMP) for faster computation
    - Proper gradient handling
    - Non-blocking data transfers
    """

    def __init__(self, model=None, optimizer=None, device=None, learning_rate=1e-4,
                 use_amp=True):
        """
        Args:
            model: TACITModel instance (creates new one if None)
            optimizer: Optimizer instance (creates Adam if None)
            device: Torch device (auto-detects if None)
            learning_rate: Learning rate for Adam optimizer
            use_amp: Enable Automatic Mixed Precision (recommended for A100)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.model = model if model is not None else TACITModel()
        self.model = self.model.to(self.device)

        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        # Mixed precision training setup
        self.use_amp = use_amp and device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler('cuda')
            print("Mixed Precision (AMP) enabled for faster training")
        else:
            self.scaler = None

    def compute_loss(self, x0, x1):
        """Compute flow matching loss."""
        t = torch.rand(x0.shape[0], device=self.device)
        t_expanded = t.reshape(-1, 1, 1, 1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        v_pred = self.model(xt, t)
        loss = (v_pred - (x1 - x0)) ** 2
        loss = loss.mean()
        return loss

    def train_step(self, x0, x1):
        """
        Single training step with proper gradient handling and AMP.

        The correct order is:
        1. Zero gradients (clear old gradients FIRST)
        2. Forward pass
        3. Backward pass (compute new gradients)
        4. Optimizer step (update weights)
        """
        # 1. Zero gradients FIRST (set_to_none=True is more efficient)
        self.optimizer.zero_grad(set_to_none=True)

        # Transfer to GPU with non_blocking for async copy
        x0 = x0.to(self.device, non_blocking=True)
        x1 = x1.to(self.device, non_blocking=True)

        if self.use_amp:
            # Mixed precision forward pass
            with autocast('cuda'):
                loss = self.compute_loss(x0, x1)

            # Scaled backward pass (prevents gradient underflow in FP16)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            loss = self.compute_loss(x0, x1)
            loss.backward()
            self.optimizer.step()

        return loss.item()


def load_checkpoint(trainer, checkpoint_path):
    """Load model weights from a safetensors checkpoint."""
    state_dict = load_file(checkpoint_path)
    trainer.model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {checkpoint_path}")


def train(trainer, dataloader, num_epochs, checkpoint_dir='./checkpoints',
          log_every=500, checkpoint_every=5):
    """
    Main training loop.

    Args:
        trainer: Trainer instance
        dataloader: DataLoader with maze pairs
        num_epochs: Number of epochs to train
        checkpoint_dir: Directory to save checkpoints
        log_every: Log average loss every N batches
        checkpoint_every: Save checkpoint every N epochs

    Returns:
        all_losses: List of all batch losses
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    all_losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_start = time.time()
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader),
                           desc=f'Epoch {epoch+1}/{num_epochs}')

        for idx_batch, (x0, x1) in progress_bar:
            loss = trainer.train_step(x0, x1)
            epoch_losses.append(loss)

            progress_bar.set_postfix({'loss': f'{loss:.6f}'})

            if idx_batch % log_every == 0 and idx_batch > 0:
                avg_recent = sum(epoch_losses[-log_every:]) / log_every
                print(f'\n  Batch {idx_batch}: avg loss = {avg_recent:.6e}')

        epoch_time = time.time() - epoch_start
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        all_losses.extend(epoch_losses)

        # Calculate throughput
        samples_per_sec = len(dataloader) * dataloader.batch_size / epoch_time
        print(f'\nEpoch {epoch+1} complete: avg loss = {epoch_avg_loss:.6e}, '
              f'time = {epoch_time:.1f}s, throughput = {samples_per_sec:.0f} samples/s')

        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f'tacit_epoch_{epoch+1}.safetensors'
            save_file(trainer.model.state_dict(), str(checkpoint_path))
            print(f'  Checkpoint saved: {checkpoint_path.name}')

    return all_losses
