#!/usr/bin/env python
"""
Train TACIT model on maze dataset.

Optimized for high GPU utilization with:
- torch.compile() for graph optimization
- Automatic Mixed Precision (AMP)
- Increased batch size and workers

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
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size (default 256 for better GPU utilization)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_batches', type=int, default=None,
                        help='Number of .npz files to use (None = all)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers (default 8 for parallel loading)')
    parser.add_argument('--checkpoint_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=500,
                        help='Log average loss every N batches')
    parser.add_argument('--no_compile', action='store_true',
                        help='Disable torch.compile() optimization')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable Automatic Mixed Precision')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Create model
    model = TACITModel()

    # Apply torch.compile for graph optimization (PyTorch 2.0+)
    # This fuses operations and reduces memory transfers
    if not args.no_compile and hasattr(torch, 'compile'):
        print("Applying torch.compile() for optimized execution...")
        try:
            model = torch.compile(model)
            print("Model compiled successfully")
        except Exception as e:
            print(f"Warning: torch.compile() failed ({e}), continuing without compilation")

    # Create trainer with AMP support
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        use_amp=not args.no_amp
    )

    # Load checkpoint if provided
    if args.checkpoint:
        load_checkpoint(trainer, args.checkpoint)

    # Create optimized dataloader
    print(f"\nDataLoader configuration:")
    print(f"  batch_size: {args.batch_size}")
    print(f"  num_workers: {args.num_workers}")

    dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_workers=args.num_workers
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    losses = train(
        trainer=trainer,
        dataloader=dataloader,
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every
    )

    print(f"\nTraining complete. Final avg loss: {sum(losses[-100:]) / min(100, len(losses)):.4f}")


if __name__ == '__main__':
    main()
