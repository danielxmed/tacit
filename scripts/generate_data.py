#!/usr/bin/env python
"""
Generate maze dataset for TACIT training.

Usage:
    python scripts/generate_data.py --total 100000 --save_dir ./data
    python scripts/generate_data.py --total 1000000 --batch_size 10000
"""

import argparse
from tacit.data.generation import generate_dataset, set_seed


def main():
    parser = argparse.ArgumentParser(description='Generate maze dataset')
    parser.add_argument('--total', type=int, default=100000,
                        help='Total number of maze pairs to generate')
    parser.add_argument('--batch_size', type=int, default=10000,
                        help='Number of pairs per .npz file')
    parser.add_argument('--output_size', type=int, default=64,
                        help='Image resolution (default: 64)')
    parser.add_argument('--save_dir', type=str, default='./data',
                        help='Directory to save dataset')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    generate_dataset(
        total_size=args.total,
        batch_size=args.batch_size,
        output_size=args.output_size,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
