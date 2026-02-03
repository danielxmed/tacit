#!/usr/bin/env python
"""
Train TACIT model on maze dataset.

Usage:
    python scripts/train.py --data_dir ./data --epochs 50
    python scripts/train.py --data_dir ./data --checkpoint ./checkpoints/tacit_epoch_10.safetensors
"""

import argparse
import torch

from tacit.models.dit import TACITModel
from tacit.data.dataset import create_dataloader
from tacit.training.trainer import Trainer, train, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Train TACIT model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to maze dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_batches', type=int, default=None,
                        help='Number of .npz files to use (None = all)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader workers')
    parser.add_argument('--checkpoint_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=500,
                        help='Log average loss every N batches')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create model and trainer
    model = TACITModel()
    trainer = Trainer(model=model, device=device, learning_rate=args.lr)

    # Load checkpoint if provided
    if args.checkpoint:
        load_checkpoint(trainer, args.checkpoint)

    # Create dataloader
    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_workers=args.num_workers
    )

    # Train
    losses = train(
        trainer=trainer,
        dataloader=dataloader,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every
    )

    print(f"\nTraining complete. Final avg loss: {sum(losses[-100:]) / 100:.4f}")


if __name__ == '__main__':
    main()
