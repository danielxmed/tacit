#!/usr/bin/env python
"""
Generate step-by-step visualizations of the TACIT model's transformation process.

This is the main interpretability script that captures and visualizes intermediate
states during the flow matching transformation from problem to solution.

Usage:
    python scripts/generate_step_by_step.py --checkpoint checkpoints/tacit_epoch_100.safetensors --num_samples 8
    python scripts/generate_step_by_step.py --checkpoint checkpoints/tacit_epoch_100.safetensors --num_steps 10 20 50 100 --create_gif
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

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
    """
    Load test mazes from the generate_test_mazes.py output directory.

    Args:
        test_maze_dir: Path to directory with maze_XXXX_input.npy files
        num_samples: Maximum number of samples to load

    Returns:
        List of (input_tensor, solution_tensor) tuples
    """
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


def create_step_grid(
    x0: torch.Tensor,
    x1_gt: torch.Tensor,
    trajectory: List[Tuple[float, torch.Tensor]],
    output_path: Path,
    sample_id: int,
    num_steps: int
) -> None:
    """
    Create a grid visualization showing the transformation at each captured step.

    The grid shows: Problem | t=0.0 | t=0.1 | ... | t=1.0 | Ground Truth

    Args:
        x0: Input (problem) tensor (3, H, W)
        x1_gt: Ground truth solution tensor (3, H, W)
        trajectory: List of (timestep, state_tensor) from sampling
        output_path: Path to save the figure
        sample_id: Sample identifier for title
        num_steps: Number of Euler steps used
    """
    num_frames = len(trajectory)
    # Add 2 columns: one for problem label, one for GT
    num_cols = num_frames + 1  # trajectory already includes initial state, add GT

    fig, axes = plt.subplots(1, num_cols, figsize=(2 * num_cols, 2.5))

    # Plot trajectory frames
    for i, (t, x_t) in enumerate(trajectory):
        img = tensor_to_image(x_t)
        axes[i].imshow(img)
        if i == 0:
            axes[i].set_title(f'Input\nt=0.00', fontsize=9)
        else:
            axes[i].set_title(f't={t:.2f}', fontsize=9)
        axes[i].axis('off')

    # Plot ground truth
    gt_img = tensor_to_image(x1_gt)
    axes[-1].imshow(gt_img)
    axes[-1].set_title('Ground\nTruth', fontsize=9)
    axes[-1].axis('off')

    plt.suptitle(f'Sample {sample_id} - {num_steps} steps', fontsize=11, y=1.02)
    plt.tight_layout()

    save_figure(fig, output_path)


def create_trajectory_gif(
    x0: torch.Tensor,
    x1_gt: torch.Tensor,
    trajectory: List[Tuple[float, torch.Tensor]],
    output_path: Path,
    duration_ms: int = 200
) -> None:
    """
    Create an animated GIF showing the transformation process.

    Args:
        x0: Input tensor
        x1_gt: Ground truth tensor
        trajectory: List of (timestep, state_tensor)
        output_path: Path to save the GIF
        duration_ms: Duration per frame in milliseconds
    """
    if not HAS_IMAGEIO:
        print("Warning: imageio not available, skipping GIF creation")
        return

    frames = []

    def fig_to_array(fig):
        """Convert matplotlib figure to numpy array."""
        fig.canvas.draw()
        # Use buffer_rgba() which works across matplotlib versions
        buf = np.asarray(fig.canvas.buffer_rgba())
        # Convert RGBA to RGB
        return buf[:, :, :3].copy()

    for t, x_t in trajectory:
        # Create frame with label
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        img = tensor_to_image(x_t)
        ax.imshow(img)
        ax.set_title(f't = {t:.3f}', fontsize=12)
        ax.axis('off')

        # Convert figure to image array
        frame = fig_to_array(fig)
        frames.append(frame)
        plt.close(fig)

    # Add ground truth as final frame (held longer)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    gt_img = tensor_to_image(x1_gt)
    ax.imshow(gt_img)
    ax.set_title('Ground Truth', fontsize=12)
    ax.axis('off')
    frame = fig_to_array(fig)
    # Repeat GT frame for longer display
    for _ in range(3):
        frames.append(frame)
    plt.close(fig)

    # Save GIF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, duration=duration_ms / 1000.0, loop=0)


def compute_trajectory_metrics(
    trajectory: List[Tuple[float, torch.Tensor]],
    x1_gt: torch.Tensor
) -> Dict:
    """
    Compute metrics at each step of the trajectory.

    Args:
        trajectory: List of (timestep, state_tensor)
        x1_gt: Ground truth solution tensor

    Returns:
        Dictionary with timesteps and corresponding metrics
    """
    timesteps = []
    mse_values = []
    psnr_values = []
    iou_values = []
    red_fractions = []

    for t, x_t in trajectory:
        timesteps.append(t)
        mse = compute_mse(x_t, x1_gt)
        mse_values.append(mse)
        psnr_values.append(compute_psnr(mse))

        red_metrics = compute_red_channel_metrics(x_t, x1_gt)
        iou_values.append(red_metrics['iou'])
        red_fractions.append(red_metrics['red_fraction'])

    return {
        'timesteps': timesteps,
        'mse': mse_values,
        'psnr': psnr_values,
        'path_iou': iou_values,
        'red_fraction': red_fractions
    }


def create_summary_grid(
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    all_trajectories: Dict[int, List[Tuple[float, torch.Tensor]]],
    output_path: Path,
    num_steps: int
) -> None:
    """
    Create a summary grid showing multiple samples at key timesteps.

    Rows: Different samples
    Columns: t=0, t=0.25, t=0.5, t=0.75, t=1.0, GT

    Args:
        samples: List of (input, gt) tensor pairs
        all_trajectories: Dict mapping sample_id to trajectory
        output_path: Path to save figure
        num_steps: Number of steps used
    """
    num_samples = len(samples)
    # Select key timesteps: 0, 0.25, 0.5, 0.75, 1.0 + GT
    key_t = [0.0, 0.25, 0.5, 0.75, 1.0]
    num_cols = len(key_t) + 1  # +1 for GT

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(2 * num_cols, 2 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    col_titles = [f't={t:.2f}' for t in key_t] + ['GT']

    for row, (sample_id, (x0, x1_gt)) in enumerate(zip(all_trajectories.keys(), samples)):
        trajectory = all_trajectories[sample_id]

        # Find frames closest to key timesteps
        for col, target_t in enumerate(key_t):
            # Find closest frame
            closest_idx = min(range(len(trajectory)),
                            key=lambda i: abs(trajectory[i][0] - target_t))
            _, x_t = trajectory[closest_idx]

            img = tensor_to_image(x_t)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')

            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=10)

        # Ground truth column
        gt_img = tensor_to_image(x1_gt)
        axes[row, -1].imshow(gt_img)
        axes[row, -1].axis('off')
        if row == 0:
            axes[row, -1].set_title('GT', fontsize=10)

    plt.suptitle(f'Step-by-Step Transformation ({num_steps} steps)', fontsize=12, y=1.01)
    plt.tight_layout()

    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate step-by-step visualizations of TACIT transformation'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Path to data directory with batch_*.npz files (optional if test_mazes exist)'
    )
    parser.add_argument(
        '--test_maze_dir', type=str, default='paper_data/test_mazes',
        help='Path to test mazes from generate_test_mazes.py'
    )
    parser.add_argument(
        '--output_dir', type=str, default='paper_data/interpretability/step_by_step',
        help='Output directory'
    )
    parser.add_argument(
        '--num_samples', type=int, default=8,
        help='Number of samples to process'
    )
    parser.add_argument(
        '--num_steps', type=int, nargs='+', default=[10, 20, 50, 100],
        help='Number of Euler steps to test'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--create_gif', action='store_true',
        help='Generate animated GIFs'
    )
    parser.add_argument(
        '--gif_duration', type=int, default=200,
        help='Duration per frame in GIF (ms)'
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup output directories
    subdirs = ['grids', 'metrics']
    if args.create_gif:
        subdirs.append('gifs')

    paths = setup_output_dirs(args.output_dir, subdirs)

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
        raise ValueError(
            "No data source available. Either run generate_test_mazes.py first "
            "or provide --data_dir with batch_*.npz files"
        )

    print(f"Loaded {len(samples)} samples")

    # Process each step count
    all_metrics = {}

    for num_steps in args.num_steps:
        print(f"\nProcessing with {num_steps} steps...")

        all_trajectories = {}
        step_metrics = []

        for sample_id, (x0, x1_gt) in enumerate(samples):
            # Move to device
            x0_batch = x0.unsqueeze(0).to(device)

            # Get trajectory
            trajectory, x_final = sample_euler_with_trajectory(
                model, x0_batch, num_steps=num_steps
            )

            # Convert trajectory tensors to CPU
            trajectory = [(t, x.cpu()) for t, x in trajectory]
            all_trajectories[sample_id] = trajectory

            # Create grid visualization
            grid_path = paths['grids'] / f'sample_{sample_id:02d}_steps_{num_steps}.png'
            create_step_grid(x0, x1_gt, trajectory, grid_path, sample_id, num_steps)

            # Create GIF if requested
            if args.create_gif and HAS_IMAGEIO:
                gif_path = paths['gifs'] / f'sample_{sample_id:02d}_steps_{num_steps}.gif'
                create_trajectory_gif(x0, x1_gt, trajectory, gif_path, args.gif_duration)

            # Compute metrics
            metrics = compute_trajectory_metrics(trajectory, x1_gt)
            metrics['sample_id'] = sample_id
            step_metrics.append(metrics)

            print(f"  Sample {sample_id}: final MSE={metrics['mse'][-1]:.6f}, "
                  f"final IoU={metrics['path_iou'][-1]:.3f}")

        # Create summary grid
        summary_path = paths['root'] / f'summary_grid_steps_{num_steps}.png'
        create_summary_grid(samples, all_trajectories, summary_path, num_steps)

        all_metrics[num_steps] = step_metrics

    # Save all metrics
    metrics_path = paths['metrics'] / 'trajectory_metrics.json'

    # Convert to JSON-serializable format
    json_metrics = {}
    for num_steps, step_metrics in all_metrics.items():
        json_metrics[str(num_steps)] = []
        for m in step_metrics:
            json_metrics[str(num_steps)].append({
                'sample_id': m['sample_id'],
                'timesteps': m['timesteps'],
                'mse': m['mse'],
                'psnr': m['psnr'],
                'path_iou': m['path_iou'],
                'red_fraction': m['red_fraction']
            })

    with open(metrics_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)

    print(f"\nComplete! Results saved to: {args.output_dir}")
    print(f"  - Grid visualizations: {paths['grids']}")
    if args.create_gif:
        print(f"  - GIF animations: {paths['gifs']}")
    print(f"  - Metrics: {metrics_path}")


if __name__ == '__main__':
    main()
