#!/usr/bin/env python
"""
Generate maze solutions using a trained TACIT model.

Usage:
    python scripts/sample.py --checkpoint ./checkpoints/tacit_epoch_50.safetensors --data_dir ./data
    python scripts/sample.py --checkpoint ./checkpoints/tacit_epoch_50.safetensors --data_dir ./data --num_samples 8
"""

import argparse
import torch
from safetensors.torch import load_file

from tacit.models.dit import TACITModel
from tacit.data.dataset import MazeDataset
from tacit.inference.sampling import visualize_predictions


def load_checkpoint_flexible(model, checkpoint_path):
    """
    Load checkpoint handling both compiled and non-compiled model formats.

    torch.compile() adds '_orig_mod.' prefix to all state dict keys.
    This function handles both formats transparently.
    """
    state_dict = load_file(checkpoint_path)

    # Check if this is from a compiled model (has _orig_mod. prefix)
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith('_orig_mod.'):
        # Strip the prefix from all keys
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Sample from trained TACIT model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to maze dataset (for visualization)')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to visualize')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of Euler sampling steps')
    parser.add_argument('--num_batches', type=int, default=1,
                        help='Number of dataset batches to load')
    parser.add_argument('--output', type=str, default='./predictions.png',
                        help='Path to save output figure')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = TACITModel()
    load_checkpoint_flexible(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    # Load dataset
    dataset = MazeDataset(args.data_dir, num_batches=args.num_batches)

    # Visualize predictions
    visualize_predictions(
        model=model,
        dataset=dataset,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=device,
        save_path=args.output
    )


if __name__ == '__main__':
    main()
