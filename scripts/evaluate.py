#!/usr/bin/env python
"""
Evaluate and compare TACIT model checkpoints on fixed samples.

Computes metrics and generates side-by-side visualizations for scientific comparison.

Usage:
    python scripts/evaluate.py --checkpoints ./checkpoints/tacit_epoch_10.safetensors ./checkpoints/tacit_epoch_50.safetensors
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from safetensors.torch import load_file

from tacit.models.dit import TACITModel
from tacit.inference.sampling import sample_euler_method


def load_checkpoint_flexible(model, checkpoint_path):
    """Load checkpoint handling both compiled and non-compiled formats."""
    state_dict = load_file(checkpoint_path)
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith('_orig_mod.'):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


def load_fixed_samples(data_dir: str, num_samples: int, seed: int = 42):
    """Load fixed samples using a deterministic seed."""
    np.random.seed(seed)

    data_path = Path(data_dir)
    batch_files = sorted(data_path.glob('batch_*.npz'))

    if not batch_files:
        raise ValueError(f"No batch files found in {data_dir}")

    # Load first batch
    batch_data = np.load(batch_files[0])
    inputs = batch_data['inputs']
    outputs = batch_data['outputs']

    # Select random indices (deterministic due to seed)
    indices = np.random.choice(len(inputs), size=num_samples, replace=False)

    samples = []
    for idx in indices:
        input_img = inputs[idx]
        output_img = outputs[idx]

        input_tensor = torch.from_numpy(input_img.copy()).permute(2, 0, 1).float() / 255.0
        output_tensor = torch.from_numpy(output_img.copy()).permute(2, 0, 1).float() / 255.0

        samples.append((input_tensor, output_tensor))

    return samples


def compute_metrics(prediction: torch.Tensor, ground_truth: torch.Tensor) -> dict:
    """Compute evaluation metrics between prediction and ground truth."""
    pred = prediction.cpu()
    gt = ground_truth.cpu()

    # MSE - Mean Squared Error
    mse = torch.mean((pred - gt) ** 2).item()

    # MAE - Mean Absolute Error
    mae = torch.mean(torch.abs(pred - gt)).item()

    # PSNR - Peak Signal-to-Noise Ratio
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

    # Path accuracy: check red channel (solution path)
    # Threshold to binary and compute IoU
    pred_path = (pred[0] > 0.5).float()  # Red channel
    gt_path = (gt[0] > 0.5).float()

    intersection = (pred_path * gt_path).sum()
    union = ((pred_path + gt_path) > 0).float().sum()
    path_iou = (intersection / union).item() if union > 0 else 0.0

    return {
        'mse': mse,
        'mae': mae,
        'psnr': psnr,
        'path_iou': path_iou
    }


def evaluate_checkpoint(checkpoint_path: str, samples: list, device: torch.device,
                        num_steps: int = 10) -> tuple:
    """Evaluate a single checkpoint on all samples."""
    model = TACITModel()
    load_checkpoint_flexible(model, checkpoint_path)
    model = model.to(device)
    model.eval()

    predictions = []
    all_metrics = []

    with torch.no_grad():
        for x0, x1 in samples:
            x0_batch = x0.unsqueeze(0).to(device)
            pred = sample_euler_method(model, x0_batch, num_steps)
            pred = torch.clamp(pred[0].cpu(), 0, 1)
            predictions.append(pred)

            metrics = compute_metrics(pred, x1)
            all_metrics.append(metrics)

    # Aggregate metrics
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in all_metrics]),
        'mae': np.mean([m['mae'] for m in all_metrics]),
        'psnr': np.mean([m['psnr'] for m in all_metrics]),
        'path_iou': np.mean([m['path_iou'] for m in all_metrics]),
    }

    return predictions, all_metrics, avg_metrics


def create_comparison_figure(samples: list, results: dict, output_path: str):
    """Create side-by-side comparison figure."""
    num_samples = len(samples)
    num_checkpoints = len(results)

    # Columns: Problem, Ground Truth, then one per checkpoint
    num_cols = 2 + num_checkpoints

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(3 * num_cols, 3 * num_samples))

    # Column titles
    col_titles = ['Problem', 'Ground Truth'] + [Path(cp).stem.replace('tacit_', '') for cp in results.keys()]

    for i, (x0, x1) in enumerate(samples):
        # Problem
        axes[i, 0].imshow(x0.permute(1, 2, 0).numpy())
        if i == 0:
            axes[i, 0].set_title(col_titles[0], fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # Ground Truth
        axes[i, 1].imshow(x1.permute(1, 2, 0).numpy())
        if i == 0:
            axes[i, 1].set_title(col_titles[1], fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')

        # Predictions from each checkpoint
        for j, (cp_path, (preds, metrics, _)) in enumerate(results.items()):
            col_idx = 2 + j
            pred_img = preds[i].permute(1, 2, 0).numpy()
            axes[i, col_idx].imshow(pred_img)

            if i == 0:
                axes[i, col_idx].set_title(col_titles[col_idx], fontsize=12, fontweight='bold')

            # Add metrics as text
            m = metrics[i]
            axes[i, col_idx].text(0.02, 0.98, f"IoU: {m['path_iou']:.2f}\nPSNR: {m['psnr']:.1f}",
                                  transform=axes[i, col_idx].transAxes,
                                  fontsize=8, verticalalignment='top',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[i, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison figure saved to: {output_path}")


def print_metrics_table(results: dict):
    """Print metrics table in a nice format."""
    print("\n" + "=" * 70)
    print("EVALUATION METRICS (averaged over all samples)")
    print("=" * 70)
    print(f"{'Checkpoint':<25} {'MSE':>10} {'MAE':>10} {'PSNR':>10} {'Path IoU':>10}")
    print("-" * 70)

    for cp_path, (_, _, avg_metrics) in results.items():
        name = Path(cp_path).stem.replace('tacit_', '')
        print(f"{name:<25} {avg_metrics['mse']:>10.6f} {avg_metrics['mae']:>10.6f} "
              f"{avg_metrics['psnr']:>10.2f} {avg_metrics['path_iou']:>10.3f}")

    print("=" * 70)
    print("\nMetrics explanation:")
    print("  MSE      : Mean Squared Error (lower is better)")
    print("  MAE      : Mean Absolute Error (lower is better)")
    print("  PSNR     : Peak Signal-to-Noise Ratio in dB (higher is better)")
    print("  Path IoU : Intersection over Union for solution path (higher is better)")
    print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate TACIT model checkpoints')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Paths to model checkpoints to compare')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to maze dataset')
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Number of samples to evaluate')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of Euler sampling steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible sample selection')
    parser.add_argument('--output', type=str, default='./evaluation.png',
                        help='Path to save comparison figure')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Euler steps: {args.num_steps}")

    # Load fixed samples
    print(f"\nLoading samples from {args.data_dir}...")
    samples = load_fixed_samples(args.data_dir, args.num_samples, args.seed)
    print(f"Loaded {len(samples)} fixed samples")

    # Evaluate each checkpoint
    results = {}
    for cp_path in args.checkpoints:
        print(f"\nEvaluating: {cp_path}")
        preds, metrics, avg_metrics = evaluate_checkpoint(
            cp_path, samples, device, args.num_steps
        )
        results[cp_path] = (preds, metrics, avg_metrics)

    # Print metrics table
    print_metrics_table(results)

    # Create comparison figure
    create_comparison_figure(samples, results, args.output)


if __name__ == '__main__':
    main()
