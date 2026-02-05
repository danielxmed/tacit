#!/usr/bin/env python
"""
Generate a controlled set of test mazes for interpretability analysis.

This script creates maze problem-solution pairs in a standardized format,
with both image (PNG) and tensor (NPY) outputs for flexibility in analysis.

Usage:
    python scripts/generate_test_mazes.py --num_mazes 20 --seed 42
    python scripts/generate_test_mazes.py --sizes 11 15 21 --num_mazes 5 --output_dir paper_data/test_mazes
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

from tacit.data.generation import (
    set_seed,
    generate_maze,
    solve_maze,
    render_maze
)


def generate_and_save_maze(
    maze_id: int,
    size: int,
    output_dir: Path,
    output_size: int = 64
) -> dict:
    """
    Generate a single maze and save all outputs.

    Args:
        maze_id: Unique identifier for this maze
        size: Logical maze size (must be odd)
        output_dir: Directory to save outputs
        output_size: Image resolution (64x64 default)

    Returns:
        Metadata dictionary for this maze
    """
    # Generate maze structure
    maze = generate_maze(size)

    # Solve to find path
    path = solve_maze(maze)

    # Render images
    input_img = render_maze(maze, path=None, output_size=output_size)
    solution_img = render_maze(maze, path=path, output_size=output_size)

    # Save PNG images
    Image.fromarray(input_img).save(output_dir / f'maze_{maze_id:04d}_input.png')
    Image.fromarray(solution_img).save(output_dir / f'maze_{maze_id:04d}_solution.png')

    # Save as numpy tensors (channels-first, normalized to [0, 1])
    input_tensor = input_img.astype(np.float32).transpose(2, 0, 1) / 255.0
    solution_tensor = solution_img.astype(np.float32).transpose(2, 0, 1) / 255.0

    np.save(output_dir / f'maze_{maze_id:04d}_input.npy', input_tensor)
    np.save(output_dir / f'maze_{maze_id:04d}_solution.npy', solution_tensor)

    # Compute metadata
    metadata = {
        'id': maze_id,
        'size': size,
        'path_length': len(path),
        'actual_maze_size': maze.shape[0],
        'output_size': output_size
    }

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Generate test mazes for interpretability analysis'
    )
    parser.add_argument(
        '--output_dir', type=str, default='paper_data/test_mazes',
        help='Output directory (default: paper_data/test_mazes)'
    )
    parser.add_argument(
        '--num_mazes', type=int, default=20,
        help='Number of mazes to generate (default: 20)'
    )
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=[11, 15, 21, 25, 31],
        help='Maze sizes to use (default: 11 15 21 25 31)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output_size', type=int, default=64,
        help='Output image size in pixels (default: 64)'
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_mazes} test mazes")
    print(f"Sizes: {args.sizes}")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    all_metadata = {
        'config': {
            'num_mazes': args.num_mazes,
            'sizes': args.sizes,
            'seed': args.seed,
            'output_size': args.output_size
        },
        'mazes': []
    }

    for i in range(args.num_mazes):
        # Cycle through sizes for variety
        size = args.sizes[i % len(args.sizes)]

        # Generate and save
        maze_metadata = generate_and_save_maze(
            maze_id=i,
            size=size,
            output_dir=output_dir,
            output_size=args.output_size
        )

        all_metadata['mazes'].append(maze_metadata)

        if (i + 1) % 5 == 0 or i == args.num_mazes - 1:
            print(f"Generated {i + 1}/{args.num_mazes} mazes")

    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print("-" * 50)
    print(f"Complete! Files saved to: {output_dir}")
    print(f"  - {args.num_mazes} input images (maze_XXXX_input.png)")
    print(f"  - {args.num_mazes} solution images (maze_XXXX_solution.png)")
    print(f"  - {args.num_mazes * 2} numpy tensors (*.npy)")
    print(f"  - metadata.json")

    # Print size distribution
    size_counts = {}
    for m in all_metadata['mazes']:
        size_counts[m['size']] = size_counts.get(m['size'], 0) + 1
    print(f"\nSize distribution: {size_counts}")

    # Print path length statistics
    path_lengths = [m['path_length'] for m in all_metadata['mazes']]
    print(f"Path lengths: min={min(path_lengths)}, max={max(path_lengths)}, "
          f"mean={np.mean(path_lengths):.1f}")


if __name__ == '__main__':
    main()
