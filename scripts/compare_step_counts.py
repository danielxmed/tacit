#!/usr/bin/env python
"""
Investigate the effect of different step counts on solution quality.

This script analyzes:
- How quality varies with number of Euler steps
- Convergence behavior across different step counts
- Minimum steps needed for acceptable quality

Usage:
    python scripts/compare_step_counts.py --checkpoint checkpoints/tacit_epoch_100.safetensors
    python scripts/compare_step_counts.py --checkpoint checkpoints/tacit_epoch_100.safetensors --step_counts 5 10 20 50 100 200
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict

from tacit.models.dit import TACITModel
from tacit.interpretability.utils import (
    load_checkpoint_flexible,
    sample_euler_with_trajectory,
    tensor_to_image,
    setup_output_dirs,
    save_figure,
    load_samples_from_data_dir,
    compute_mse,
    compute_psnr,
    compute_red_channel_metrics
)


def load_test_mazes(test_maze_dir: str, num_samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Load test mazes from directory."""
    test_dir = Path(test_maze_dir)
    samples = []

    for i in range(num_samples):
        input_path = test_dir / f'maze_{i:04d}_input.npy'
        solution_path = test_dir / f'maze_{i:04d}_solution.npy'

        if not input_path.exists() or not solution_path.exists():
            break

        input_tensor = torch.from_numpy(np.load(input_path))
        solution_tensor = torch.from_numpy(np.load(solution_path))
        samples.append((input_tensor, solution_tensor))

    return samples


def compare_quality_across_steps(
    model: torch.nn.Module,
    x0: torch.Tensor,
    x1_gt: torch.Tensor,
    step_counts: List[int],
    device: torch.device
) -> Dict:
    """
    Compare final quality metrics across different step counts.

    Args:
        model: TACITModel instance
        x0: Input tensor (3, H, W)
        x1_gt: Ground truth tensor
        step_counts: List of step counts to test
        device: Torch device

    Returns:
        Dictionary with metrics for each step count
    """
    results = {}

    x0_batch = x0.unsqueeze(0).to(device)

    for num_steps in step_counts:
        # Get final prediction
        _, x_final = sample_euler_with_trajectory(model, x0_batch, num_steps)
        x_final = x_final.cpu()

        # Compute metrics
        mse = compute_mse(x_final, x1_gt)
        psnr = compute_psnr(mse)
        red_metrics = compute_red_channel_metrics(x_final, x1_gt)

        results[num_steps] = {
            'mse': mse,
            'psnr': psnr,
            'iou': red_metrics['iou'],
            'recall': red_metrics['recall'],
            'precision': red_metrics['precision'],
            'prediction': x_final.squeeze(0)
        }

    return results


def compute_convergence_curve(
    trajectory: List[Tuple[float, torch.Tensor]],
    x1_gt: torch.Tensor
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute MSE and IoU at each step of the trajectory.

    Args:
        trajectory: List of (timestep, state_tensor)
        x1_gt: Ground truth tensor

    Returns:
        timesteps, mse_values, iou_values
    """
    timesteps = []
    mse_values = []
    iou_values = []

    for t, x_t in trajectory:
        timesteps.append(t)
        mse_values.append(compute_mse(x_t, x1_gt))

        red_metrics = compute_red_channel_metrics(x_t, x1_gt)
        iou_values.append(red_metrics['iou'])

    return timesteps, mse_values, iou_values


def find_sufficient_steps(
    convergence_data: Dict[int, List[Dict]],
    threshold_iou: float = 0.9,
    threshold_mse_ratio: float = 1.1
) -> Dict[str, int]:
    """
    Find minimum number of steps needed for acceptable quality.

    Two criteria:
    1. IoU threshold: minimum steps to achieve IoU > threshold
    2. MSE ratio: minimum steps where MSE is within ratio of best

    Args:
        convergence_data: Dictionary mapping step counts to metrics lists
        threshold_iou: Target IoU threshold
        threshold_mse_ratio: Acceptable MSE ratio vs best

    Returns:
        Dictionary with sufficient step counts for each criterion
    """
    step_counts = sorted(convergence_data.keys())

    # Compute mean final metrics for each step count
    final_metrics = {}
    for steps in step_counts:
        sample_metrics = convergence_data[steps]
        mean_iou = np.mean([m[-1]['iou'] for m in sample_metrics])
        mean_mse = np.mean([m[-1]['mse'] for m in sample_metrics])
        final_metrics[steps] = {'iou': mean_iou, 'mse': mean_mse}

    # Find best MSE (usually at highest step count)
    best_mse = min(final_metrics[s]['mse'] for s in step_counts)

    # Find sufficient steps for IoU
    sufficient_iou = None
    for steps in step_counts:
        if final_metrics[steps]['iou'] >= threshold_iou:
            sufficient_iou = steps
            break

    # Find sufficient steps for MSE
    sufficient_mse = None
    for steps in step_counts:
        if final_metrics[steps]['mse'] <= best_mse * threshold_mse_ratio:
            sufficient_mse = steps
            break

    return {
        'iou_threshold': sufficient_iou or step_counts[-1],
        'mse_ratio': sufficient_mse or step_counts[-1],
        'threshold_iou_used': threshold_iou,
        'threshold_mse_ratio_used': threshold_mse_ratio,
        'best_mse': best_mse
    }


def plot_visual_comparison(
    sample_results: Dict[int, Dict],
    x0: torch.Tensor,
    x1_gt: torch.Tensor,
    step_counts: List[int],
    output_path: Path,
    sample_id: int
) -> None:
    """Plot visual comparison across different step counts."""
    num_cols = len(step_counts) + 2  # +2 for input and GT

    fig, axes = plt.subplots(1, num_cols, figsize=(2.5 * num_cols, 3))

    # Input
    axes[0].imshow(tensor_to_image(x0))
    axes[0].set_title('Input', fontsize=10)
    axes[0].axis('off')

    # Predictions for each step count
    for i, steps in enumerate(step_counts):
        pred = sample_results[steps]['prediction']
        axes[i + 1].imshow(tensor_to_image(pred))
        iou = sample_results[steps]['iou']
        axes[i + 1].set_title(f'{steps} steps\nIoU={iou:.3f}', fontsize=9)
        axes[i + 1].axis('off')

    # Ground truth
    axes[-1].imshow(tensor_to_image(x1_gt))
    axes[-1].set_title('Ground Truth', fontsize=10)
    axes[-1].axis('off')

    plt.suptitle(f'Sample {sample_id}: Effect of Step Count', fontsize=11)
    plt.tight_layout()
    save_figure(fig, output_path)


def plot_convergence_curves(
    all_convergence: Dict[int, List[Dict]],
    output_path: Path
) -> None:
    """Plot convergence curves for different step counts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    step_counts = sorted(all_convergence.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(step_counts)))

    # MSE convergence
    for steps, color in zip(step_counts, colors):
        sample_metrics = all_convergence[steps]
        # Average across samples
        all_t = []
        all_mse = []
        for metrics in sample_metrics:
            t = [m['t'] for m in metrics]
            mse = [m['mse'] for m in metrics]
            all_t.append(t)
            all_mse.append(mse)

        # Interpolate to common grid
        common_t = np.linspace(0, 1, 101)
        interp_mse = []
        for t, mse in zip(all_t, all_mse):
            interp_mse.append(np.interp(common_t, t, mse))

        mean_mse = np.mean(interp_mse, axis=0)
        axes[0].plot(common_t, mean_mse, color=color, linewidth=2, label=f'{steps} steps')

    axes[0].set_xlabel('Time (t)', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('MSE Convergence', fontsize=13)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)

    # IoU convergence
    for steps, color in zip(step_counts, colors):
        sample_metrics = all_convergence[steps]
        all_t = []
        all_iou = []
        for metrics in sample_metrics:
            t = [m['t'] for m in metrics]
            iou = [m['iou'] for m in metrics]
            all_t.append(t)
            all_iou.append(iou)

        common_t = np.linspace(0, 1, 101)
        interp_iou = []
        for t, iou in zip(all_t, all_iou):
            interp_iou.append(np.interp(common_t, t, iou))

        mean_iou = np.mean(interp_iou, axis=0)
        axes[1].plot(common_t, mean_iou, color=color, linewidth=2, label=f'{steps} steps')

    axes[1].set_xlabel('Time (t)', fontsize=12)
    axes[1].set_ylabel('Path IoU', fontsize=12)
    axes[1].set_title('Path IoU Convergence', fontsize=13)
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_quality_vs_steps(
    final_metrics: Dict[int, Dict],
    output_path: Path
) -> None:
    """Plot final quality metrics vs number of steps."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    step_counts = sorted(final_metrics.keys())

    # MSE
    mse_means = [final_metrics[s]['mse']['mean'] for s in step_counts]
    mse_stds = [final_metrics[s]['mse']['std'] for s in step_counts]

    axes[0].errorbar(step_counts, mse_means, yerr=mse_stds, fmt='o-', capsize=5,
                    color='tab:blue', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Steps', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Final MSE vs Steps', fontsize=13)
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)

    # PSNR
    psnr_means = [final_metrics[s]['psnr']['mean'] for s in step_counts]
    psnr_stds = [final_metrics[s]['psnr']['std'] for s in step_counts]

    axes[1].errorbar(step_counts, psnr_means, yerr=psnr_stds, fmt='o-', capsize=5,
                    color='tab:green', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Steps', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_title('Final PSNR vs Steps', fontsize=13)
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    # IoU
    iou_means = [final_metrics[s]['iou']['mean'] for s in step_counts]
    iou_stds = [final_metrics[s]['iou']['std'] for s in step_counts]

    axes[2].errorbar(step_counts, iou_means, yerr=iou_stds, fmt='o-', capsize=5,
                    color='tab:red', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Steps', fontsize=12)
    axes[2].set_ylabel('Path IoU', fontsize=12)
    axes[2].set_title('Final Path IoU vs Steps', fontsize=13)
    axes[2].set_xscale('log')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_sufficient_steps_analysis(
    final_metrics: Dict[int, Dict],
    sufficient_info: Dict,
    output_path: Path
) -> None:
    """Plot analysis of sufficient step counts."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    step_counts = sorted(final_metrics.keys())
    iou_means = [final_metrics[s]['iou']['mean'] for s in step_counts]
    mse_means = [final_metrics[s]['mse']['mean'] for s in step_counts]

    # Plot IoU curve
    ax.plot(step_counts, iou_means, 'o-', color='tab:blue', linewidth=2.5,
            markersize=10, label='Path IoU')

    # Mark IoU threshold
    ax.axhline(sufficient_info['threshold_iou_used'], color='tab:blue',
               linestyle='--', alpha=0.5, label=f"IoU threshold ({sufficient_info['threshold_iou_used']})")

    # Mark sufficient steps for IoU
    suff_iou = sufficient_info['iou_threshold']
    ax.axvline(suff_iou, color='tab:green', linestyle='-.',
               label=f'Sufficient steps (IoU): {suff_iou}')

    ax.set_xlabel('Number of Steps', fontsize=12)
    ax.set_ylabel('Path IoU', fontsize=12)
    ax.set_title('Step Count Sufficiency Analysis', fontsize=13)
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text annotation
    ax.annotate(f'Min steps for IoU > {sufficient_info["threshold_iou_used"]}: {suff_iou}',
                xy=(suff_iou, sufficient_info['threshold_iou_used']),
                xytext=(suff_iou * 2, sufficient_info['threshold_iou_used'] - 0.15),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    save_figure(fig, output_path)


def create_summary_visual_grid(
    all_sample_results: List[Dict[int, Dict]],
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    step_counts: List[int],
    output_path: Path,
    max_samples: int = 6
) -> None:
    """Create summary grid showing multiple samples across step counts."""
    n_samples = min(len(samples), max_samples)
    n_cols = len(step_counts) + 2  # input + steps + GT

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(2.2 * n_cols, 2.2 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    col_titles = ['Input'] + [f'{s} steps' for s in step_counts] + ['Ground Truth']

    for row in range(n_samples):
        x0, x1_gt = samples[row]
        sample_results = all_sample_results[row]

        # Input
        axes[row, 0].imshow(tensor_to_image(x0))
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title(col_titles[0], fontsize=10)

        # Predictions
        for col, steps in enumerate(step_counts):
            pred = sample_results[steps]['prediction']
            axes[row, col + 1].imshow(tensor_to_image(pred))
            axes[row, col + 1].axis('off')
            if row == 0:
                axes[row, col + 1].set_title(col_titles[col + 1], fontsize=10)

        # Ground truth
        axes[row, -1].imshow(tensor_to_image(x1_gt))
        axes[row, -1].axis('off')
        if row == 0:
            axes[row, -1].set_title(col_titles[-1], fontsize=10)

    plt.suptitle('Visual Comparison: Effect of Step Count', fontsize=12, y=1.01)
    plt.tight_layout()
    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Compare effect of step counts on TACIT solution quality'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Path to data directory'
    )
    parser.add_argument(
        '--test_maze_dir', type=str, default='paper_data/test_mazes',
        help='Path to test mazes'
    )
    parser.add_argument(
        '--output_dir', type=str, default='paper_data/interpretability/step_comparison',
        help='Output directory'
    )
    parser.add_argument(
        '--num_samples', type=int, default=20,
        help='Number of samples to analyze'
    )
    parser.add_argument(
        '--step_counts', type=int, nargs='+', default=[5, 10, 20, 50, 100],
        help='Step counts to compare'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup output directories
    paths = setup_output_dirs(args.output_dir, ['samples'])

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = TACITModel()
    load_checkpoint_flexible(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    # Load samples
    print("Loading samples...")
    test_maze_dir = Path(args.test_maze_dir)
    if test_maze_dir.exists() and (test_maze_dir / 'maze_0000_input.npy').exists():
        print(f"  Using test mazes from: {test_maze_dir}")
        samples = load_test_mazes(str(test_maze_dir), args.num_samples)
    elif args.data_dir:
        print(f"  Using data directory: {args.data_dir}")
        samples = load_samples_from_data_dir(args.data_dir, args.num_samples, args.seed)
    else:
        raise ValueError("No data source available")

    print(f"Loaded {len(samples)} samples")
    print(f"Testing step counts: {args.step_counts}")

    # Analyze each sample
    all_sample_results = []
    all_convergence = {steps: [] for steps in args.step_counts}

    print("\nAnalyzing samples...")

    for sample_id, (x0, x1_gt) in enumerate(samples):
        # Compare quality across step counts
        sample_results = compare_quality_across_steps(
            model, x0, x1_gt, args.step_counts, device
        )
        all_sample_results.append(sample_results)

        # Get convergence curves for each step count
        x0_batch = x0.unsqueeze(0).to(device)

        for steps in args.step_counts:
            trajectory, _ = sample_euler_with_trajectory(model, x0_batch, steps)
            trajectory = [(t, x.cpu()) for t, x in trajectory]

            t_vals, mse_vals, iou_vals = compute_convergence_curve(trajectory, x1_gt)

            metrics_list = [{'t': t, 'mse': m, 'iou': i}
                          for t, m, i in zip(t_vals, mse_vals, iou_vals)]
            all_convergence[steps].append(metrics_list)

        # Save individual visual comparison (first few samples)
        if sample_id < 5:
            plot_visual_comparison(
                sample_results, x0, x1_gt, args.step_counts,
                paths['samples'] / f'sample_{sample_id:02d}_comparison.png',
                sample_id
            )

        if (sample_id + 1) % 5 == 0:
            print(f"  Processed {sample_id + 1}/{len(samples)} samples")

    # Compute aggregate metrics
    print("\nComputing aggregate metrics...")

    final_metrics = {}
    for steps in args.step_counts:
        mse_vals = [all_sample_results[i][steps]['mse'] for i in range(len(samples))]
        psnr_vals = [all_sample_results[i][steps]['psnr'] for i in range(len(samples))]
        iou_vals = [all_sample_results[i][steps]['iou'] for i in range(len(samples))]

        final_metrics[steps] = {
            'mse': {'mean': np.mean(mse_vals), 'std': np.std(mse_vals)},
            'psnr': {'mean': np.mean(psnr_vals), 'std': np.std(psnr_vals)},
            'iou': {'mean': np.mean(iou_vals), 'std': np.std(iou_vals)}
        }

    # Find sufficient steps
    sufficient_info = find_sufficient_steps(all_convergence)

    # Create visualizations
    print("Creating visualizations...")

    # Summary visual grid
    create_summary_visual_grid(
        all_sample_results, samples, args.step_counts,
        paths['root'] / 'visual_comparison.png'
    )

    # Convergence curves
    plot_convergence_curves(
        all_convergence,
        paths['root'] / 'convergence_curves.png'
    )

    # Quality vs steps
    plot_quality_vs_steps(
        final_metrics,
        paths['root'] / 'quality_vs_steps.png'
    )

    # Sufficient steps analysis
    plot_sufficient_steps_analysis(
        final_metrics,
        sufficient_info,
        paths['root'] / 'sufficient_steps_analysis.png'
    )

    # Save metrics
    print("Saving metrics...")

    # Convert to JSON-serializable format
    json_metrics = {
        'config': {
            'num_samples': len(samples),
            'step_counts': args.step_counts,
            'seed': args.seed
        },
        'final_metrics': final_metrics,
        'sufficient_steps': sufficient_info,
        'per_sample': []
    }

    for i, sample_results in enumerate(all_sample_results):
        sample_data = {'sample_id': i}
        for steps in args.step_counts:
            sample_data[f'steps_{steps}'] = {
                'mse': sample_results[steps]['mse'],
                'psnr': sample_results[steps]['psnr'],
                'iou': sample_results[steps]['iou']
            }
        json_metrics['per_sample'].append(sample_data)

    with open(paths['root'] / 'step_count_metrics.json', 'w') as f:
        json.dump(json_metrics, f, indent=2)

    print(f"\nComplete! Results saved to: {args.output_dir}")
    print("\nSummary of final metrics (mean +/- std):")
    print("-" * 60)
    print(f"{'Steps':<10} {'MSE':<20} {'PSNR (dB)':<15} {'IoU':<15}")
    print("-" * 60)
    for steps in args.step_counts:
        m = final_metrics[steps]
        print(f"{steps:<10} {m['mse']['mean']:.6f} +/- {m['mse']['std']:.6f}  "
              f"{m['psnr']['mean']:.2f} +/- {m['psnr']['std']:.2f}  "
              f"{m['iou']['mean']:.3f} +/- {m['iou']['std']:.3f}")
    print("-" * 60)
    print(f"\nSufficient steps for IoU > {sufficient_info['threshold_iou_used']}: "
          f"{sufficient_info['iou_threshold']}")


if __name__ == '__main__':
    main()
