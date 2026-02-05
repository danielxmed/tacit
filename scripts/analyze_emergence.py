#!/usr/bin/env python
"""
Analyze how the solution path emerges during the transformation process.

This script provides quantitative analysis of path emergence, detecting
phase transitions and computing statistics about the temporal evolution
of the solution.

Usage:
    python scripts/analyze_emergence.py --checkpoint checkpoints/tacit_epoch_100.safetensors
    python scripts/analyze_emergence.py --checkpoint checkpoints/tacit_epoch_100.safetensors --num_samples 50 --num_steps 100
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import interp1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from tacit.models.dit import TACITModel
from tacit.interpretability.utils import (
    load_checkpoint_flexible,
    sample_euler_with_trajectory,
    extract_red_path_mask,
    compute_red_channel_metrics,
    tensor_to_image,
    setup_output_dirs,
    save_figure,
    load_samples_from_data_dir
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


def analyze_emergence_trajectory(
    trajectory: List[Tuple[float, torch.Tensor]],
    x1_gt: torch.Tensor,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Analyze how the red path emerges along the trajectory.

    Args:
        trajectory: List of (timestep, state_tensor)
        x1_gt: Ground truth solution
        threshold: Red detection threshold

    Returns:
        DataFrame with columns: t, red_count, red_fraction, iou, recall, precision
    """
    rows = []

    for t, x_t in trajectory:
        metrics = compute_red_channel_metrics(x_t, x1_gt, threshold)
        rows.append({
            't': t,
            'red_count': metrics['red_count'],
            'red_fraction': metrics['red_fraction'],
            'iou': metrics['iou'],
            'recall': metrics['recall'],
            'precision': metrics['precision']
        })

    return pd.DataFrame(rows)


def detect_phase_transition(
    emergence_df: pd.DataFrame,
    metric: str = 'recall'
) -> Dict[str, float]:
    """
    Detect phase transition points in the emergence curve.

    Uses simple threshold-based detection:
    - t_onset: First time metric exceeds 10%
    - t_midpoint: Time when metric crosses 50%
    - t_completion: First time metric exceeds 90%

    Args:
        emergence_df: DataFrame with emergence data
        metric: Which metric to analyze ('recall', 'iou', 'red_fraction')

    Returns:
        Dictionary with transition points and width
    """
    t = emergence_df['t'].values
    values = emergence_df[metric].values

    # Find transition points
    t_onset = None
    t_midpoint = None
    t_completion = None

    for i, (ti, vi) in enumerate(zip(t, values)):
        if t_onset is None and vi > 0.1:
            t_onset = ti
        if t_midpoint is None and vi > 0.5:
            t_midpoint = ti
        if t_completion is None and vi > 0.9:
            t_completion = ti

    # Default to endpoints if not found
    if t_onset is None:
        t_onset = t[-1]
    if t_midpoint is None:
        t_midpoint = t[-1]
    if t_completion is None:
        t_completion = t[-1]

    # Compute transition width
    transition_width = t_completion - t_onset if t_completion and t_onset else 1.0

    return {
        't_onset': t_onset,
        't_midpoint': t_midpoint,
        't_completion': t_completion,
        'transition_width': transition_width
    }


def compute_emergence_rate(
    emergence_df: pd.DataFrame,
    metric: str = 'recall'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the rate of emergence (derivative) of a metric.

    Args:
        emergence_df: DataFrame with emergence data
        metric: Which metric to differentiate

    Returns:
        t_centers: Timesteps at derivative centers
        rates: Rate of change at each timestep
    """
    t = emergence_df['t'].values
    values = emergence_df[metric].values

    if len(t) < 2:
        return np.array([]), np.array([])

    # Simple finite difference
    dt = np.diff(t)
    dv = np.diff(values)
    rates = dv / np.where(dt > 0, dt, 1e-6)

    # Center timesteps
    t_centers = (t[:-1] + t[1:]) / 2

    # Smooth if scipy available
    if HAS_SCIPY and len(rates) > 5:
        rates = gaussian_filter1d(rates, sigma=1)

    return t_centers, rates


def compute_emergence_statistics(
    all_emergence_data: List[pd.DataFrame],
    timesteps: np.ndarray
) -> Dict:
    """
    Compute statistics across multiple samples.

    Args:
        all_emergence_data: List of emergence DataFrames
        timesteps: Common timesteps to interpolate to

    Returns:
        Dictionary with mean, std, CI for each metric
    """
    metrics = ['recall', 'iou', 'precision', 'red_fraction']
    stats = {}

    for metric in metrics:
        all_values = []

        for df in all_emergence_data:
            # Interpolate to common timesteps
            if HAS_SCIPY:
                interp = interp1d(df['t'].values, df[metric].values,
                                kind='linear', fill_value='extrapolate')
                values = interp(timesteps)
            else:
                # Simple nearest neighbor interpolation
                values = np.interp(timesteps, df['t'].values, df[metric].values)

            all_values.append(values)

        all_values = np.array(all_values)

        # Compute statistics
        mean = np.mean(all_values, axis=0)
        std = np.std(all_values, axis=0)
        n = len(all_values)
        ci_95 = 1.96 * std / np.sqrt(n)

        stats[metric] = {
            'mean': mean.tolist(),
            'std': std.tolist(),
            'ci_95_lower': (mean - ci_95).tolist(),
            'ci_95_upper': (mean + ci_95).tolist()
        }

    return stats


def plot_emergence_curves(
    all_emergence_data: List[pd.DataFrame],
    output_path: Path,
    title: str = 'Path Emergence Curves'
) -> None:
    """Plot all individual emergence curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('recall', 'Path Recall'),
        ('iou', 'Path IoU'),
        ('precision', 'Path Precision'),
        ('red_fraction', 'Red Pixel Fraction')
    ]

    for ax, (metric, label) in zip(axes.flat, metrics):
        for i, df in enumerate(all_emergence_data):
            alpha = 0.3 if len(all_emergence_data) > 10 else 0.5
            ax.plot(df['t'], df[metric], alpha=alpha, linewidth=0.8)

        ax.set_xlabel('Time (t)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_path)


def plot_emergence_with_ci(
    timesteps: np.ndarray,
    stats: Dict,
    output_path: Path
) -> None:
    """Plot mean emergence curves with confidence intervals."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('recall', 'Path Recall', 'tab:blue'),
        ('iou', 'Path IoU', 'tab:orange'),
        ('precision', 'Path Precision', 'tab:green'),
        ('red_fraction', 'Red Pixel Fraction', 'tab:red')
    ]

    for ax, (metric, label, color) in zip(axes.flat, metrics):
        mean = np.array(stats[metric]['mean'])
        ci_lower = np.array(stats[metric]['ci_95_lower'])
        ci_upper = np.array(stats[metric]['ci_95_upper'])

        ax.plot(timesteps, mean, color=color, linewidth=2, label='Mean')
        ax.fill_between(timesteps, ci_lower, ci_upper, color=color, alpha=0.2, label='95% CI')

        ax.set_xlabel('Time (t)')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

    plt.suptitle('Mean Emergence Curves with 95% Confidence Interval', fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_path)


def plot_emergence_rate(
    all_emergence_data: List[pd.DataFrame],
    output_path: Path
) -> None:
    """Plot the rate of emergence (derivative)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    all_t_centers = []
    all_rates = []

    for df in all_emergence_data:
        t_centers, rates = compute_emergence_rate(df, 'recall')
        if len(rates) > 0:
            ax.plot(t_centers, rates, alpha=0.3, linewidth=0.8, color='tab:blue')
            all_t_centers.append(t_centers)
            all_rates.append(rates)

    # Plot mean rate
    if all_rates:
        # Interpolate to common grid
        common_t = np.linspace(0, 1, 50)
        interp_rates = []
        for t_c, r in zip(all_t_centers, all_rates):
            if len(t_c) > 1:
                interp_rates.append(np.interp(common_t, t_c, r))

        if interp_rates:
            mean_rate = np.mean(interp_rates, axis=0)
            ax.plot(common_t, mean_rate, color='tab:red', linewidth=2, label='Mean Rate')

    ax.set_xlabel('Time (t)')
    ax.set_ylabel('d(Recall)/dt')
    ax.set_title('Emergence Rate (Derivative of Recall)')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_phase_transitions(
    all_transitions: List[Dict],
    output_path: Path
) -> None:
    """Plot histogram of phase transition points."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    t_onsets = [t['t_onset'] for t in all_transitions if t['t_onset'] is not None]
    t_midpoints = [t['t_midpoint'] for t in all_transitions if t['t_midpoint'] is not None]
    t_completions = [t['t_completion'] for t in all_transitions if t['t_completion'] is not None]

    bins = np.linspace(0, 1, 21)

    axes[0].hist(t_onsets, bins=bins, edgecolor='black', alpha=0.7, color='tab:green')
    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f't_onset (10% recall)\nmean={np.mean(t_onsets):.3f}')
    axes[0].axvline(np.mean(t_onsets), color='red', linestyle='--', label='Mean')

    axes[1].hist(t_midpoints, bins=bins, edgecolor='black', alpha=0.7, color='tab:blue')
    axes[1].set_xlabel('Time (t)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f't_midpoint (50% recall)\nmean={np.mean(t_midpoints):.3f}')
    axes[1].axvline(np.mean(t_midpoints), color='red', linestyle='--', label='Mean')

    axes[2].hist(t_completions, bins=bins, edgecolor='black', alpha=0.7, color='tab:orange')
    axes[2].set_xlabel('Time (t)')
    axes[2].set_ylabel('Count')
    axes[2].set_title(f't_completion (90% recall)\nmean={np.mean(t_completions):.3f}')
    axes[2].axvline(np.mean(t_completions), color='red', linestyle='--', label='Mean')

    plt.suptitle('Phase Transition Point Distribution', fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_path)


def create_paper_figure(
    timesteps: np.ndarray,
    stats: Dict,
    all_transitions: List[Dict],
    output_path: Path
) -> None:
    """Create a publication-ready composite figure."""
    fig = plt.figure(figsize=(14, 5))

    # Left panel: Emergence curves with CI
    ax1 = fig.add_subplot(1, 2, 1)

    mean_recall = np.array(stats['recall']['mean'])
    ci_lower = np.array(stats['recall']['ci_95_lower'])
    ci_upper = np.array(stats['recall']['ci_95_upper'])

    ax1.plot(timesteps, mean_recall, color='tab:blue', linewidth=2.5, label='Recall')
    ax1.fill_between(timesteps, ci_lower, ci_upper, color='tab:blue', alpha=0.2)

    mean_iou = np.array(stats['iou']['mean'])
    ax1.plot(timesteps, mean_iou, color='tab:orange', linewidth=2.5, linestyle='--', label='IoU')

    ax1.set_xlabel('Time (t)', fontsize=12)
    ax1.set_ylabel('Metric Value', fontsize=12)
    ax1.set_title('(a) Path Emergence During Transformation', fontsize=13)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right panel: Transition point distribution
    ax2 = fig.add_subplot(1, 2, 2)

    t_midpoints = [t['t_midpoint'] for t in all_transitions if t['t_midpoint'] is not None]
    widths = [t['transition_width'] for t in all_transitions]

    ax2.scatter(t_midpoints, widths, alpha=0.6, s=50)
    ax2.axhline(np.mean(widths), color='red', linestyle='--', label=f'Mean width: {np.mean(widths):.3f}')
    ax2.axvline(np.mean(t_midpoints), color='blue', linestyle='--', label=f'Mean midpoint: {np.mean(t_midpoints):.3f}')

    ax2.set_xlabel('Transition Midpoint (t)', fontsize=12)
    ax2.set_ylabel('Transition Width', fontsize=12)
    ax2.set_title('(b) Phase Transition Characteristics', fontsize=13)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze path emergence during TACIT transformation'
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
        '--output_dir', type=str, default='paper_data/interpretability/emergence',
        help='Output directory'
    )
    parser.add_argument(
        '--num_samples', type=int, default=50,
        help='Number of samples for statistics'
    )
    parser.add_argument(
        '--num_steps', type=int, default=100,
        help='Number of Euler steps (higher = finer resolution)'
    )
    parser.add_argument(
        '--red_threshold', type=float, default=0.5,
        help='Threshold for red pixel detection'
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
    paths = setup_output_dirs(args.output_dir, ['curves', 'transitions', 'metrics'])

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

    # Analyze each sample
    all_emergence_data = []
    all_transitions = []

    print(f"\nAnalyzing emergence with {args.num_steps} steps...")

    for sample_id, (x0, x1_gt) in enumerate(samples):
        x0_batch = x0.unsqueeze(0).to(device)

        # Get trajectory
        trajectory, _ = sample_euler_with_trajectory(model, x0_batch, args.num_steps)
        trajectory = [(t, x.cpu()) for t, x in trajectory]

        # Analyze emergence
        emergence_df = analyze_emergence_trajectory(trajectory, x1_gt, args.red_threshold)
        all_emergence_data.append(emergence_df)

        # Detect phase transitions
        transitions = detect_phase_transition(emergence_df, 'recall')
        transitions['sample_id'] = sample_id
        all_transitions.append(transitions)

        if (sample_id + 1) % 10 == 0:
            print(f"  Processed {sample_id + 1}/{len(samples)} samples")

    # Compute statistics
    print("\nComputing statistics...")
    common_timesteps = np.linspace(0, 1, args.num_steps + 1)
    stats = compute_emergence_statistics(all_emergence_data, common_timesteps)

    # Create visualizations
    print("Creating visualizations...")

    # Individual emergence curves
    plot_emergence_curves(
        all_emergence_data,
        paths['curves'] / 'emergence_curves_all.png',
        f'Path Emergence Curves (n={len(samples)})'
    )

    # Mean with CI
    plot_emergence_with_ci(
        common_timesteps,
        stats,
        paths['curves'] / 'emergence_mean_with_ci.png'
    )

    # Emergence rate
    plot_emergence_rate(
        all_emergence_data,
        paths['curves'] / 'emergence_rate.png'
    )

    # Phase transitions
    plot_phase_transitions(
        all_transitions,
        paths['transitions'] / 'phase_transition_histogram.png'
    )

    # Paper figure
    create_paper_figure(
        common_timesteps,
        stats,
        all_transitions,
        paths['root'] / 'paper_figure_emergence.png'
    )

    # Save data
    print("Saving metrics...")

    # Emergence data as CSV
    for i, df in enumerate(all_emergence_data):
        df['sample_id'] = i
    combined_df = pd.concat(all_emergence_data, ignore_index=True)
    combined_df.to_csv(paths['metrics'] / 'emergence_data_full.csv', index=False)

    # Transition points
    transitions_df = pd.DataFrame(all_transitions)
    transitions_df.to_csv(paths['metrics'] / 'transition_points.csv', index=False)

    # Summary statistics
    summary = {
        'config': {
            'num_samples': len(samples),
            'num_steps': args.num_steps,
            'red_threshold': args.red_threshold,
            'seed': args.seed
        },
        'timesteps': common_timesteps.tolist(),
        'statistics': stats,
        'transition_summary': {
            't_onset': {
                'mean': np.mean([t['t_onset'] for t in all_transitions]),
                'std': np.std([t['t_onset'] for t in all_transitions])
            },
            't_midpoint': {
                'mean': np.mean([t['t_midpoint'] for t in all_transitions]),
                'std': np.std([t['t_midpoint'] for t in all_transitions])
            },
            't_completion': {
                'mean': np.mean([t['t_completion'] for t in all_transitions]),
                'std': np.std([t['t_completion'] for t in all_transitions])
            },
            'transition_width': {
                'mean': np.mean([t['transition_width'] for t in all_transitions]),
                'std': np.std([t['transition_width'] for t in all_transitions])
            }
        }
    }

    with open(paths['metrics'] / 'summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nComplete! Results saved to: {args.output_dir}")
    print("\nKey findings:")
    print(f"  - Mean onset (10%):     t = {summary['transition_summary']['t_onset']['mean']:.3f}")
    print(f"  - Mean midpoint (50%):  t = {summary['transition_summary']['t_midpoint']['mean']:.3f}")
    print(f"  - Mean completion (90%): t = {summary['transition_summary']['t_completion']['mean']:.3f}")
    print(f"  - Mean transition width: {summary['transition_summary']['transition_width']['mean']:.3f}")


if __name__ == '__main__':
    main()
