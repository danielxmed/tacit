import torch
import time
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file, load_file

from tacit.models.dit import TACITModel


class Trainer:
    def __init__(self, model=None, optimizer=None, device=None, learning_rate=1e-4):
        """
        Args:
            model: TACITModel instance (creates new one if None)
            optimizer: optimizer instance (creates Adam if None)
            device: torch device (auto-detects if None)
            learning_rate: learning rate for Adam optimizer
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.model = model if model is not None else TACITModel()
        self.model = self.model.to(self.device)

        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

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
        """Single training step."""
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        loss = self.compute_loss(x0, x1)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
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
        num_epochs: number of epochs to train
        checkpoint_dir: directory to save checkpoints
        log_every: log average loss every N batches
        checkpoint_every: save checkpoint every N epochs

    Returns:
        all_losses: list of all batch losses
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

            progress_bar.set_postfix({'loss': f'{loss:.4f}'})

            if idx_batch % log_every == 0 and idx_batch > 0:
                avg_recent = sum(epoch_losses[-log_every:]) / log_every
                print(f'\n  Batch {idx_batch}: avg loss = {avg_recent:.4f}')

        epoch_time = time.time() - epoch_start
        epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
        all_losses.extend(epoch_losses)

        print(f'\nEpoch {epoch+1} complete: avg loss = {epoch_avg_loss:.4f}, time = {epoch_time:.1f}s')

        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f'tacit_epoch_{epoch+1}.safetensors'
            save_file(trainer.model.state_dict(), str(checkpoint_path))
            print(f'  Checkpoint saved: {checkpoint_path.name}')

    return all_losses
