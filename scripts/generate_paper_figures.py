#!/usr/bin/env python
"""
Generate figures for the TACIT paper.

This script generates:
- Training loss curves (linear and log scale)
- Epoch comparison grids showing model evolution
- Individual high-resolution samples per epoch
- Quality metrics (L2 distance to ground truth)

Usage:
    python scripts/generate_paper_figures.py --data_dir ./data
    python scripts/generate_paper_figures.py --data_dir ./data --output_dir ./paper_data
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from safetensors.torch import load_file

from tacit.models.dit import TACITModel
from tacit.data.dataset import MazeDataset
from tacit.inference.sampling import sample_euler_method


# Training loss data extracted from training logs
# Format: (epoch, avg_loss, throughput_samples_per_sec)
TRAINING_DATA = [
    (5, 1.20e-03, 7000),
    (10, 3.50e-04, 7000),
    (15, 1.50e-04, 7000),
    (20, 7.00e-05, 7000),
    (25, 4.00e-05, 7000),
    (30, 2.50e-05, 7000),
    (31, 2.12e-05, 7261),
    (32, 2.49e-05, 6861),
    (33, 1.96e-05, 6940),
    (34, 2.08e-05, 7129),
    (35, 2.09e-05, 7035),
    (36, 1.78e-05, 7107),
    (37, 1.87e-05, 7048),
    (38, 1.84e-05, 6974),
    (39, 1.75e-05, 7021),
    (40, 1.80e-05, 7113),
    (41, 1.64e-05, 7073),
    (42, 1.69e-05, 7846),
    (43, 1.50e-05, 8900),
    (44, 1.48e-05, 8917),
    (45, 1.59e-05, 9796),
    (46, 1.50e-05, 11724),
    (47, 1.50e-05, 11632),
    (48, 1.60e-05, 11668),
    (49, 1.50e-05, 11629),
    (50, 1.31e-05, 11500),
    (55, 1.20e-05, 11700),
    (57, 1.20e-05, 11715),
    (58, 1.24e-05, 11714),
    (59, 1.24e-05, 11798),
    (60, 1.07e-05, 9462),
    (61, 9.83e-06, 7358),
    (62, 1.12e-05, 7267),
    (63, 1.17e-05, 7195),
    (64, 9.28e-06, 7153),
    (65, 9.51e-06, 7402),
    (66, 9.49e-06, 7098),
    (67, 1.07e-05, 7031),
    (68, 8.43e-06, 7025),
    (69, 1.00e-05, 7335),
    (70, 9.60e-06, 7469),
    (71, 9.39e-06, 7489),
    (72, 9.39e-06, 7132),
    (73, 9.14e-06, 7062),
    (74, 7.75e-06, 7152),
    (75, 8.81e-06, 7010),
    (76, 8.26e-06, 6249),
    (77, 8.13e-06, 7024),
    (78, 7.25e-06, 7057),
    (79, 9.37e-06, 7164),
    (80, 8.00e-06, 7026),
    (81, 8.49e-06, 7230),
    (82, 7.80e-06, 7070),
    (83, 7.74e-06, 7150),
    (84, 9.39e-06, 7185),
    (85, 6.69e-06, 7190),
    (86, 7.23e-06, 7069),
    (87, 8.32e-06, 6990),
    (88, 7.16e-06, 7020),
    (89, 7.38e-06, 7167),
    (90, 7.38e-06, 7032),
    (91, 5.08e-06, 7066),
    (92, 6.96e-06, 7194),
    (93, 5.91e-06, 7135),
    (94, 7.65e-06, 7114),
    (95, 6.26e-06, 7122),
    (96, 5.78e-06, 7065),
    (97, 6.18e-06, 7023),
    (98, 7.36e-06, 7214),
    (99, 5.43e-06, 7192),
    (100, 6.25e-06, 7041),
]

# Key epochs for comparison
KEY_EPOCHS = [5, 10, 25, 50, 75, 100]


def load_checkpoint_flexible(model, checkpoint_path):
    """Load checkpoint handling both compiled and non-compiled model formats."""
    state_dict = load_file(checkpoint_path)
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith('_orig_mod.'):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


def plot_training_loss(output_dir: Path, show_plot: bool = False):
    """Generate training loss curve plots."""
    epochs = [d[0] for d in TRAINING_DATA]
    losses = [d[1] for d in TRAINING_DATA]

    # Linear scale plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, losses, 'b-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Loss (MSE)', fontsize=12)
    ax.set_title('TACIT Training Loss Over Epochs', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)

    linear_path = output_dir / 'training_curves' / 'loss_curve.png'
    plt.savefig(linear_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {linear_path}")

    if show_plot:
        plt.show()
    plt.close()

    # Log scale plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(epochs, losses, 'b-', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average Loss (MSE) - Log Scale', fontsize=12)
    ax.set_title('TACIT Training Loss Over Epochs (Log Scale)', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 105)

    log_path = output_dir / 'training_curves' / 'loss_curve_log.png'
    plt.savefig(log_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {log_path}")

    if show_plot:
        plt.show()
    plt.close()

    # Save metrics as JSON
    metrics = {
        'epochs': epochs,
        'losses': losses,
        'throughput': [d[2] for d in TRAINING_DATA],
        'final_loss': losses[-1],
        'total_epochs': epochs[-1]
    }
    json_path = output_dir / 'training_curves' / 'training_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {json_path}")

    return metrics


def get_fixed_samples(dataset, num_samples=6, seed=42):
    """Get fixed samples from dataset for consistent comparisons."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    samples = []
    for i, (x0, x1) in enumerate(dataset):
        samples.append((x0, x1))
        if len(samples) >= num_samples:
            break

    return samples


def generate_epoch_comparison(
    checkpoint_dir: Path,
    output_dir: Path,
    dataset,
    device,
    epochs: list = None,
    num_samples: int = 4,
    num_steps: int = 10
):
    """Generate side-by-side epoch comparison grid."""
    if epochs is None:
        epochs = KEY_EPOCHS

    # Get fixed samples
    samples = get_fixed_samples(dataset, num_samples=num_samples)

    # Load all models
    models = {}
    for epoch in epochs:
        checkpoint_path = checkpoint_dir / f'tacit_epoch_{epoch}.safetensors'
        if checkpoint_path.exists():
            model = TACITModel()
            load_checkpoint_flexible(model, checkpoint_path)
            model = model.to(device)
            model.eval()
            models[epoch] = model
            print(f"Loaded model from epoch {epoch}")
        else:
            print(f"Warning: Checkpoint not found for epoch {epoch}")

    if not models:
        print("No checkpoints found!")
        return

    # Create comparison grid
    # Columns: Input | Epoch predictions... | Ground Truth
    n_cols = len(models) + 2  # +2 for input and ground truth
    n_rows = num_samples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))

    # Column headers
    col_titles = ['Input'] + [f'Epoch {e}' for e in sorted(models.keys())] + ['Ground Truth']

    for idx, (x0, x1) in enumerate(samples):
        x0_batch = x0.unsqueeze(0).to(device)

        # Input image
        x0_vis = x0.permute(1, 2, 0).numpy()
        axes[idx, 0].imshow(x0_vis)
        axes[idx, 0].axis('off')
        if idx == 0:
            axes[idx, 0].set_title('Input', fontsize=10, fontweight='bold')

        # Predictions for each epoch
        for col_idx, epoch in enumerate(sorted(models.keys())):
            model = models[epoch]
            with torch.no_grad():
                pred = sample_euler_method(model, x0_batch, num_steps)
            pred_vis = pred[0].cpu().permute(1, 2, 0).numpy()
            pred_vis = np.clip(pred_vis, 0, 1)

            axes[idx, col_idx + 1].imshow(pred_vis)
            axes[idx, col_idx + 1].axis('off')
            if idx == 0:
                axes[idx, col_idx + 1].set_title(f'Epoch {epoch}', fontsize=10, fontweight='bold')

        # Ground truth
        x1_vis = x1.permute(1, 2, 0).numpy()
        axes[idx, -1].imshow(x1_vis)
        axes[idx, -1].axis('off')
        if idx == 0:
            axes[idx, -1].set_title('Ground Truth', fontsize=10, fontweight='bold')

    plt.tight_layout()

    save_path = output_dir / 'epoch_comparison' / 'evolution_grid.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Also create early vs late comparison
    generate_early_vs_late(models, samples, output_dir, device, num_steps)

    return models


def generate_early_vs_late(models, samples, output_dir: Path, device, num_steps: int = 10):
    """Generate comparison between early (epoch 10) and late (epoch 100) training."""
    early_epoch = min(e for e in models.keys() if e >= 10)
    late_epoch = max(models.keys())

    if early_epoch not in models or late_epoch not in models:
        print("Cannot generate early vs late comparison - missing models")
        return

    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))

    for idx, (x0, x1) in enumerate(samples):
        x0_batch = x0.unsqueeze(0).to(device)

        # Input
        x0_vis = x0.permute(1, 2, 0).numpy()
        axes[idx, 0].imshow(x0_vis)
        axes[idx, 0].axis('off')
        if idx == 0:
            axes[idx, 0].set_title('Input', fontsize=12, fontweight='bold')

        # Early epoch
        with torch.no_grad():
            early_pred = sample_euler_method(models[early_epoch], x0_batch, num_steps)
        early_vis = early_pred[0].cpu().permute(1, 2, 0).numpy()
        early_vis = np.clip(early_vis, 0, 1)
        axes[idx, 1].imshow(early_vis)
        axes[idx, 1].axis('off')
        if idx == 0:
            axes[idx, 1].set_title(f'Epoch {early_epoch}', fontsize=12, fontweight='bold')

        # Late epoch
        with torch.no_grad():
            late_pred = sample_euler_method(models[late_epoch], x0_batch, num_steps)
        late_vis = late_pred[0].cpu().permute(1, 2, 0).numpy()
        late_vis = np.clip(late_vis, 0, 1)
        axes[idx, 2].imshow(late_vis)
        axes[idx, 2].axis('off')
        if idx == 0:
            axes[idx, 2].set_title(f'Epoch {late_epoch}', fontsize=12, fontweight='bold')

        # Ground truth
        x1_vis = x1.permute(1, 2, 0).numpy()
        axes[idx, 3].imshow(x1_vis)
        axes[idx, 3].axis('off')
        if idx == 0:
            axes[idx, 3].set_title('Ground Truth', fontsize=12, fontweight='bold')

    plt.tight_layout()

    save_path = output_dir / 'epoch_comparison' / 'early_vs_late.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_individual_epoch_samples(
    checkpoint_dir: Path,
    output_dir: Path,
    dataset,
    device,
    epochs: list = None,
    num_samples: int = 8,
    num_steps: int = 10
):
    """Generate high-quality sample grids for each epoch."""
    if epochs is None:
        epochs = KEY_EPOCHS

    samples = get_fixed_samples(dataset, num_samples=num_samples)

    for epoch in epochs:
        checkpoint_path = checkpoint_dir / f'tacit_epoch_{epoch}.safetensors'
        if not checkpoint_path.exists():
            print(f"Skipping epoch {epoch} - checkpoint not found")
            continue

        model = TACITModel()
        load_checkpoint_flexible(model, checkpoint_path)
        model = model.to(device)
        model.eval()

        # Create figure: 3 columns (input, prediction, ground truth)
        fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

        for idx, (x0, x1) in enumerate(samples):
            x0_batch = x0.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = sample_euler_method(model, x0_batch, num_steps)

            x0_vis = x0.permute(1, 2, 0).numpy()
            pred_vis = pred[0].cpu().permute(1, 2, 0).numpy()
            pred_vis = np.clip(pred_vis, 0, 1)
            x1_vis = x1.permute(1, 2, 0).numpy()

            axes[idx, 0].imshow(x0_vis)
            axes[idx, 0].axis('off')
            if idx == 0:
                axes[idx, 0].set_title('Input', fontsize=10)

            axes[idx, 1].imshow(pred_vis)
            axes[idx, 1].axis('off')
            if idx == 0:
                axes[idx, 1].set_title(f'Prediction (Epoch {epoch})', fontsize=10)

            axes[idx, 2].imshow(x1_vis)
            axes[idx, 2].axis('off')
            if idx == 0:
                axes[idx, 2].set_title('Ground Truth', fontsize=10)

        plt.suptitle(f'TACIT Model - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = output_dir / 'maze_samples' / f'epoch_{epoch}_samples.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def compute_quality_metrics(
    checkpoint_dir: Path,
    output_dir: Path,
    dataset,
    device,
    num_eval_samples: int = 100,
    num_steps: int = 10
):
    """Compute L2 distance to ground truth for each checkpoint."""
    # Get evaluation samples
    samples = get_fixed_samples(dataset, num_samples=num_eval_samples, seed=123)

    # Available checkpoints
    checkpoints = sorted(checkpoint_dir.glob('tacit_epoch_*.safetensors'))

    results = []

    for checkpoint_path in checkpoints:
        epoch = int(checkpoint_path.stem.split('_')[-1])

        model = TACITModel()
        load_checkpoint_flexible(model, checkpoint_path)
        model = model.to(device)
        model.eval()

        total_l2 = 0.0

        with torch.no_grad():
            for x0, x1 in samples:
                x0_batch = x0.unsqueeze(0).to(device)
                x1_batch = x1.unsqueeze(0).to(device)

                pred = sample_euler_method(model, x0_batch, num_steps)
                l2 = torch.mean((pred - x1_batch) ** 2).item()
                total_l2 += l2

        avg_l2 = total_l2 / len(samples)
        results.append({'epoch': epoch, 'avg_l2': avg_l2})
        print(f"Epoch {epoch}: L2 = {avg_l2:.6f}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Sort by epoch
    results.sort(key=lambda x: x['epoch'])

    # Plot quality metrics
    epochs = [r['epoch'] for r in results]
    l2_values = [r['avg_l2'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, l2_values, 'g-', linewidth=2, marker='s', markersize=5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average L2 Distance to Ground Truth', fontsize=12)
    ax.set_title('TACIT Model Quality Over Training', fontsize=14)
    ax.grid(True, alpha=0.3)

    save_path = output_dir / 'training_curves' / 'quality_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Save metrics as JSON
    json_path = output_dir / 'training_curves' / 'quality_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures for TACIT')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to maze dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Path to model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./paper_data/figures',
                        help='Path to save figures')
    parser.add_argument('--num_samples', type=int, default=6,
                        help='Number of samples for comparison grids')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='Number of Euler sampling steps')
    parser.add_argument('--skip_metrics', action='store_true',
                        help='Skip quality metrics computation (faster)')
    parser.add_argument('--epochs', type=str, default=None,
                        help='Comma-separated list of epochs to compare (default: 5,10,25,50,75,100)')

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    # Ensure output directories exist
    (output_dir / 'training_curves').mkdir(parents=True, exist_ok=True)
    (output_dir / 'epoch_comparison').mkdir(parents=True, exist_ok=True)
    (output_dir / 'maze_samples').mkdir(parents=True, exist_ok=True)

    # Parse epochs
    if args.epochs:
        epochs = [int(e.strip()) for e in args.epochs.split(',')]
    else:
        epochs = KEY_EPOCHS

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate loss curves (no GPU needed)
    print("\n=== Generating Training Loss Curves ===")
    plot_training_loss(output_dir)

    # Load dataset
    print("\n=== Loading Dataset ===")
    dataset = MazeDataset(args.data_dir, num_batches=1)

    # Generate epoch comparison
    print("\n=== Generating Epoch Comparison Grid ===")
    generate_epoch_comparison(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        dataset=dataset,
        device=device,
        epochs=epochs,
        num_samples=args.num_samples,
        num_steps=args.num_steps
    )

    # Reload dataset (iterable gets exhausted)
    dataset = MazeDataset(args.data_dir, num_batches=1)

    # Generate individual epoch samples
    print("\n=== Generating Individual Epoch Samples ===")
    generate_individual_epoch_samples(
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        dataset=dataset,
        device=device,
        epochs=epochs,
        num_samples=8,
        num_steps=args.num_steps
    )

    # Compute quality metrics (optional)
    if not args.skip_metrics:
        dataset = MazeDataset(args.data_dir, num_batches=1)
        print("\n=== Computing Quality Metrics ===")
        compute_quality_metrics(
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            dataset=dataset,
            device=device,
            num_eval_samples=50,
            num_steps=args.num_steps
        )

    print("\n=== All figures generated successfully! ===")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
