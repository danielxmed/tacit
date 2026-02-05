#!/usr/bin/env python
"""
Spatial analysis of where the model focuses during maze solving.

This script analyzes the spatial patterns of emergence:
- Which parts of the path emerge first (start, middle, end)?
- Does the model work locally or globally?
- Are there distinctive spatial patterns across samples?

Usage:
    python scripts/analyze_spatial.py --checkpoint checkpoints/tacit_epoch_100.safetensors
    python scripts/analyze_spatial.py --checkpoint checkpoints/tacit_epoch_100.safetensors --num_samples 30 --num_steps 50
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from tacit.models.dit import TACITModel
from tacit.interpretability.utils import (
    load_checkpoint_flexible,
    sample_euler_with_trajectory,
    extract_red_path_mask,
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


def compute_pixel_change_heatmap(
    x_t_prev: torch.Tensor,
    x_t_curr: torch.Tensor
) -> np.ndarray:
    """
    Compute heatmap of pixel changes between consecutive states.

    Args:
        x_t_prev: Previous state tensor (3, H, W)
        x_t_curr: Current state tensor (3, H, W)

    Returns:
        Heatmap (H, W) of L2 norm of changes per pixel
    """
    if x_t_prev.dim() == 4:
        x_t_prev = x_t_prev.squeeze(0)
    if x_t_curr.dim() == 4:
        x_t_curr = x_t_curr.squeeze(0)

    # L2 norm across channels
    diff = x_t_curr - x_t_prev
    heatmap = torch.sqrt((diff ** 2).sum(dim=0))

    return heatmap.cpu().numpy()


def segment_solution_path(
    x1_gt: torch.Tensor,
    fractions: List[float] = [0.33, 0.34, 0.33]
) -> Dict[str, torch.Tensor]:
    """
    Segment the ground truth path into start, middle, and end regions.

    Uses distance from start (top-left) to segment the path.

    Args:
        x1_gt: Ground truth solution tensor (3, H, W)
        fractions: Relative sizes of start/middle/end segments

    Returns:
        Dictionary with 'start', 'middle', 'end' masks
    """
    if x1_gt.dim() == 4:
        x1_gt = x1_gt.squeeze(0)

    # Get path mask
    path_mask = extract_red_path_mask(x1_gt, threshold=0.5)
    path_points = torch.nonzero(path_mask, as_tuple=False)

    if len(path_points) == 0:
        # No path found, return empty masks
        H, W = path_mask.shape
        empty = torch.zeros((H, W), dtype=torch.bool)
        return {'start': empty, 'middle': empty, 'end': empty}

    # Compute distance from top-left corner (start)
    # In maze convention: start is (1, 1), exit is (size-2, size-2)
    H, W = path_mask.shape

    # Use manhattan distance from start (top-left region)
    distances = path_points[:, 0].float() + path_points[:, 1].float()

    # Sort path points by distance
    sorted_indices = torch.argsort(distances)
    sorted_points = path_points[sorted_indices]

    # Segment by fractions
    n_points = len(sorted_points)
    n_start = int(n_points * fractions[0])
    n_middle = int(n_points * fractions[1])

    start_points = sorted_points[:n_start]
    middle_points = sorted_points[n_start:n_start + n_middle]
    end_points = sorted_points[n_start + n_middle:]

    # Create masks
    start_mask = torch.zeros((H, W), dtype=torch.bool)
    middle_mask = torch.zeros((H, W), dtype=torch.bool)
    end_mask = torch.zeros((H, W), dtype=torch.bool)

    if len(start_points) > 0:
        start_mask[start_points[:, 0], start_points[:, 1]] = True
    if len(middle_points) > 0:
        middle_mask[middle_points[:, 0], middle_points[:, 1]] = True
    if len(end_points) > 0:
        end_mask[end_points[:, 0], end_points[:, 1]] = True

    return {
        'start': start_mask,
        'middle': middle_mask,
        'end': end_mask
    }


def analyze_emergence_by_segment(
    trajectory: List[Tuple[float, torch.Tensor]],
    segment_masks: Dict[str, torch.Tensor],
    x1_gt: torch.Tensor,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Analyze how each segment (start/middle/end) emerges over time.

    Args:
        trajectory: List of (timestep, state_tensor)
        segment_masks: Dictionary with segment masks from segment_solution_path
        x1_gt: Ground truth solution
        threshold: Red detection threshold

    Returns:
        DataFrame with columns: t, start_recall, middle_recall, end_recall
    """
    rows = []

    for t, x_t in trajectory:
        if x_t.dim() == 4:
            x_t = x_t.squeeze(0)

        pred_mask = extract_red_path_mask(x_t, threshold)

        row = {'t': t}

        for segment_name, segment_mask in segment_masks.items():
            # Count correctly predicted pixels in this segment
            segment_gt = segment_mask.sum().item()
            if segment_gt > 0:
                segment_correct = (pred_mask & segment_mask).sum().item()
                recall = segment_correct / segment_gt
            else:
                recall = 0.0

            row[f'{segment_name}_recall'] = recall

        rows.append(row)

    return pd.DataFrame(rows)


def compute_spatial_emergence_order(
    trajectory: List[Tuple[float, torch.Tensor]],
    x1_gt: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute when each pixel of the solution first appears.

    Args:
        trajectory: List of (timestep, state_tensor)
        x1_gt: Ground truth solution
        threshold: Red detection threshold

    Returns:
        first_appearance_map: (H, W) array with timestep of first correct prediction
        gt_mask: Ground truth path mask
    """
    if x1_gt.dim() == 4:
        x1_gt = x1_gt.squeeze(0)

    gt_mask = extract_red_path_mask(x1_gt, threshold)
    H, W = gt_mask.shape

    # Initialize with infinity (never appeared)
    first_appearance = np.full((H, W), np.inf)

    for t, x_t in trajectory:
        if x_t.dim() == 4:
            x_t = x_t.squeeze(0)

        pred_mask = extract_red_path_mask(x_t, threshold)

        # Find pixels that are correctly predicted and haven't been recorded yet
        correct = (pred_mask & gt_mask).numpy()
        new_correct = correct & (first_appearance == np.inf)

        first_appearance[new_correct] = t

    return first_appearance, gt_mask.numpy()


def classify_emergence_pattern(
    segment_analysis: pd.DataFrame,
    threshold: float = 0.1
) -> str:
    """
    Classify the emergence pattern based on segment analysis.

    Categories:
    - 'start_first': Start region emerges first (forward solving)
    - 'end_first': End region emerges first (backward solving)
    - 'simultaneous': All regions emerge together (global processing)
    - 'middle_first': Middle region emerges first (unusual pattern)

    Args:
        segment_analysis: DataFrame with segment recall over time
        threshold: Recall threshold to consider "emerged"

    Returns:
        Pattern classification string
    """
    # Find when each segment crosses threshold
    segments = ['start', 'middle', 'end']
    onset_times = {}

    for segment in segments:
        col = f'{segment}_recall'
        if col in segment_analysis.columns:
            crossed = segment_analysis[segment_analysis[col] > threshold]
            if len(crossed) > 0:
                onset_times[segment] = crossed['t'].iloc[0]
            else:
                onset_times[segment] = float('inf')

    # Classify based on relative timing
    if not onset_times:
        return 'unknown'

    sorted_segments = sorted(onset_times.keys(), key=lambda s: onset_times[s])

    # Check if times are very close (simultaneous)
    times = list(onset_times.values())
    time_range = max(times) - min(times) if times and min(times) < float('inf') else 0

    if time_range < 0.1:  # Within 10% of total time
        return 'simultaneous'

    first_segment = sorted_segments[0]
    if first_segment == 'start':
        return 'start_first'
    elif first_segment == 'end':
        return 'end_first'
    else:
        return 'middle_first'


def plot_change_heatmap_sequence(
    trajectory: List[Tuple[float, torch.Tensor]],
    x1_gt: torch.Tensor,
    output_path: Path,
    sample_id: int
) -> None:
    """Plot sequence of change heatmaps."""
    # Select key timesteps
    n_frames = min(6, len(trajectory) - 1)
    step = max(1, (len(trajectory) - 1) // n_frames)
    indices = list(range(0, len(trajectory) - 1, step))[:n_frames]

    fig, axes = plt.subplots(1, n_frames + 1, figsize=(3 * (n_frames + 1), 3))

    for i, idx in enumerate(indices):
        t_prev, x_prev = trajectory[idx]
        t_curr, x_curr = trajectory[idx + 1]

        heatmap = compute_pixel_change_heatmap(x_prev, x_curr)

        im = axes[i].imshow(heatmap, cmap='hot', vmin=0)
        axes[i].set_title(f't={t_prev:.2f} to {t_curr:.2f}', fontsize=9)
        axes[i].axis('off')

    # Show ground truth
    gt_img = tensor_to_image(x1_gt)
    axes[-1].imshow(gt_img)
    axes[-1].set_title('Ground Truth', fontsize=9)
    axes[-1].axis('off')

    plt.suptitle(f'Sample {sample_id}: Pixel Change Heatmaps', fontsize=11)
    plt.tight_layout()
    save_figure(fig, output_path)


def plot_first_appearance_map(
    first_appearance: np.ndarray,
    gt_mask: np.ndarray,
    output_path: Path,
    sample_id: int
) -> None:
    """Plot map showing when each pixel first appears."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # First appearance map (only for path pixels)
    appearance_masked = np.ma.masked_where(~gt_mask | (first_appearance == np.inf), first_appearance)

    im = axes[0].imshow(appearance_masked, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title(f'Sample {sample_id}: First Appearance Time', fontsize=11)
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], label='Time (t)')

    # Show path with segments colored by time
    axes[1].imshow(appearance_masked, cmap='cool', vmin=0, vmax=1)
    axes[1].set_title('Path Emergence Order', fontsize=11)
    axes[1].axis('off')

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_segment_emergence(
    all_segment_data: List[pd.DataFrame],
    output_path: Path
) -> None:
    """Plot aggregate segment emergence curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'start': 'tab:green', 'middle': 'tab:blue', 'end': 'tab:red'}
    segments = ['start', 'middle', 'end']

    # Compute mean and std for each segment
    common_t = np.linspace(0, 1, 101)

    for segment in segments:
        col = f'{segment}_recall'
        all_values = []

        for df in all_segment_data:
            if col in df.columns:
                interp_values = np.interp(common_t, df['t'].values, df[col].values)
                all_values.append(interp_values)

        if all_values:
            all_values = np.array(all_values)
            mean = np.mean(all_values, axis=0)
            std = np.std(all_values, axis=0)

            ax.plot(common_t, mean, color=colors[segment], linewidth=2, label=f'{segment.capitalize()}')
            ax.fill_between(common_t, mean - std, mean + std, color=colors[segment], alpha=0.2)

    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Segment Recall', fontsize=12)
    ax.set_title('Path Segment Emergence Over Time', fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_pattern_distribution(
    pattern_counts: Dict[str, int],
    output_path: Path
) -> None:
    """Plot distribution of emergence patterns."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    patterns = list(pattern_counts.keys())
    counts = list(pattern_counts.values())

    colors = {
        'start_first': 'tab:green',
        'end_first': 'tab:red',
        'simultaneous': 'tab:blue',
        'middle_first': 'tab:orange',
        'unknown': 'tab:gray'
    }

    bar_colors = [colors.get(p, 'tab:gray') for p in patterns]

    ax.bar(patterns, counts, color=bar_colors, edgecolor='black')
    ax.set_xlabel('Emergence Pattern', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Spatial Emergence Patterns', fontsize=13)

    # Add count labels
    for i, (p, c) in enumerate(zip(patterns, counts)):
        ax.text(i, c + 0.5, str(c), ha='center', fontsize=11)

    plt.tight_layout()
    save_figure(fig, output_path)


def create_aggregate_heatmap(
    all_first_appearances: List[Tuple[np.ndarray, np.ndarray]],
    output_path: Path
) -> None:
    """Create aggregate heatmap of emergence timing across samples."""
    # Collect all first appearance maps
    H, W = all_first_appearances[0][0].shape

    # Average emergence time per pixel position
    sum_times = np.zeros((H, W))
    count_times = np.zeros((H, W))

    for first_app, gt_mask in all_first_appearances:
        valid = gt_mask & (first_app < np.inf)
        sum_times[valid] += first_app[valid]
        count_times[valid] += 1

    mean_times = np.divide(sum_times, count_times, out=np.full((H, W), np.nan), where=count_times > 0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    im = ax.imshow(mean_times, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Aggregate Mean First Appearance Time', fontsize=13)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Mean Time (t)')

    plt.tight_layout()
    save_figure(fig, output_path)


def create_paper_figure_spatial(
    all_segment_data: List[pd.DataFrame],
    pattern_counts: Dict[str, int],
    output_path: Path
) -> None:
    """Create publication-ready composite figure for spatial analysis."""
    fig = plt.figure(figsize=(14, 5))

    # Left panel: Segment emergence curves
    ax1 = fig.add_subplot(1, 2, 1)

    colors = {'start': 'tab:green', 'middle': 'tab:blue', 'end': 'tab:red'}
    segments = ['start', 'middle', 'end']
    common_t = np.linspace(0, 1, 101)

    for segment in segments:
        col = f'{segment}_recall'
        all_values = []

        for df in all_segment_data:
            if col in df.columns:
                interp_values = np.interp(common_t, df['t'].values, df[col].values)
                all_values.append(interp_values)

        if all_values:
            all_values = np.array(all_values)
            mean = np.mean(all_values, axis=0)
            ci = 1.96 * np.std(all_values, axis=0) / np.sqrt(len(all_values))

            ax1.plot(common_t, mean, color=colors[segment], linewidth=2.5, label=f'{segment.capitalize()}')
            ax1.fill_between(common_t, mean - ci, mean + ci, color=colors[segment], alpha=0.2)

    ax1.set_xlabel('Time (t)', fontsize=12)
    ax1.set_ylabel('Segment Recall', fontsize=12)
    ax1.set_title('(a) Path Segment Emergence', fontsize=13)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Right panel: Pattern distribution
    ax2 = fig.add_subplot(1, 2, 2)

    pattern_labels = {
        'start_first': 'Start-First\n(Forward)',
        'end_first': 'End-First\n(Backward)',
        'simultaneous': 'Simultaneous\n(Global)',
        'middle_first': 'Middle-First',
        'unknown': 'Unknown'
    }

    patterns = list(pattern_counts.keys())
    counts = list(pattern_counts.values())
    labels = [pattern_labels.get(p, p) for p in patterns]

    colors_bar = {
        'start_first': 'tab:green',
        'end_first': 'tab:red',
        'simultaneous': 'tab:blue',
        'middle_first': 'tab:orange',
        'unknown': 'tab:gray'
    }
    bar_colors = [colors_bar.get(p, 'tab:gray') for p in patterns]

    ax2.bar(labels, counts, color=bar_colors, edgecolor='black')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('(b) Spatial Emergence Patterns', fontsize=13)

    for i, c in enumerate(counts):
        ax2.text(i, c + 0.3, str(c), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Spatial analysis of path emergence in TACIT'
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
        '--output_dir', type=str, default='paper_data/interpretability/spatial',
        help='Output directory'
    )
    parser.add_argument(
        '--num_samples', type=int, default=30,
        help='Number of samples to analyze'
    )
    parser.add_argument(
        '--num_steps', type=int, default=50,
        help='Number of Euler steps'
    )
    parser.add_argument(
        '--segment_fractions', type=float, nargs=3, default=[0.33, 0.34, 0.33],
        help='Fractions for start/middle/end segments'
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
    paths = setup_output_dirs(args.output_dir,
                             ['heatmaps', 'segments', 'emergence_order', 'patterns'])

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
    all_segment_data = []
    all_first_appearances = []
    all_patterns = []

    print(f"\nAnalyzing spatial emergence with {args.num_steps} steps...")

    for sample_id, (x0, x1_gt) in enumerate(samples):
        x0_batch = x0.unsqueeze(0).to(device)

        # Get trajectory
        trajectory, _ = sample_euler_with_trajectory(model, x0_batch, args.num_steps)
        trajectory = [(t, x.cpu()) for t, x in trajectory]

        # Segment the ground truth path
        segment_masks = segment_solution_path(x1_gt, args.segment_fractions)

        # Analyze segment emergence
        segment_df = analyze_emergence_by_segment(trajectory, segment_masks, x1_gt)
        all_segment_data.append(segment_df)

        # Compute first appearance map
        first_appearance, gt_mask = compute_spatial_emergence_order(trajectory, x1_gt)
        all_first_appearances.append((first_appearance, gt_mask))

        # Classify pattern
        pattern = classify_emergence_pattern(segment_df)
        all_patterns.append(pattern)

        # Create individual visualizations (only for first few samples)
        if sample_id < 5:
            # Change heatmap sequence
            heatmap_path = paths['heatmaps'] / f'sample_{sample_id:02d}_change_sequence.png'
            plot_change_heatmap_sequence(trajectory, x1_gt, heatmap_path, sample_id)

            # First appearance map
            appearance_path = paths['emergence_order'] / f'first_appearance_sample_{sample_id:02d}.png'
            plot_first_appearance_map(first_appearance, gt_mask, appearance_path, sample_id)

        if (sample_id + 1) % 10 == 0:
            print(f"  Processed {sample_id + 1}/{len(samples)} samples")

    # Aggregate visualizations
    print("\nCreating aggregate visualizations...")

    # Segment emergence aggregate
    plot_segment_emergence(
        all_segment_data,
        paths['segments'] / 'segment_emergence_aggregate.png'
    )

    # Pattern distribution
    pattern_counts = {}
    for p in all_patterns:
        pattern_counts[p] = pattern_counts.get(p, 0) + 1

    plot_pattern_distribution(
        pattern_counts,
        paths['patterns'] / 'pattern_distribution.png'
    )

    # Aggregate heatmap
    create_aggregate_heatmap(
        all_first_appearances,
        paths['heatmaps'] / 'aggregate_heatmap.png'
    )

    # Paper figure
    create_paper_figure_spatial(
        all_segment_data,
        pattern_counts,
        paths['root'] / 'paper_figure_spatial.png'
    )

    # Save metrics
    print("Saving metrics...")

    # Pattern classification
    pattern_data = {
        'sample_id': list(range(len(all_patterns))),
        'pattern': all_patterns
    }
    pd.DataFrame(pattern_data).to_csv(
        paths['patterns'] / 'pattern_classification.csv', index=False
    )

    # Summary
    summary = {
        'config': {
            'num_samples': len(samples),
            'num_steps': args.num_steps,
            'segment_fractions': args.segment_fractions,
            'seed': args.seed
        },
        'pattern_counts': pattern_counts,
        'dominant_pattern': max(pattern_counts.keys(), key=lambda k: pattern_counts[k])
    }

    with open(paths['patterns'] / 'pattern_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nComplete! Results saved to: {args.output_dir}")
    print("\nPattern distribution:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_patterns)
        print(f"  {pattern}: {count} ({pct:.1f}%)")
    print(f"\nDominant pattern: {summary['dominant_pattern']}")


if __name__ == '__main__':
    main()
