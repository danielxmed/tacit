import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from pathlib import Path
import random


class MazeDataset(IterableDataset):
    """
    Iterable dataset that loads maze pairs from .npz batch files efficiently.

    Instead of random access (which causes constant file reloading), this:
    1. Shuffles the order of .npz files each epoch
    2. Loads one file at a time (single decompression)
    3. Shuffles samples within each file
    4. Yields samples sequentially

    This gives good shuffling while avoiding the I/O bottleneck.
    """

    def __init__(self, data_dir: str, num_batches: int = None):
        """
        Args:
            data_dir: Path to directory containing batch_XXXX.npz files
            num_batches: How many batches to use (None = all)
        """
        self.data_dir = Path(data_dir)
        self.batch_files = sorted(self.data_dir.glob('batch_*.npz'))

        if num_batches is not None:
            self.batch_files = self.batch_files[:num_batches]

        if len(self.batch_files) == 0:
            raise ValueError(f"No batch files found in {data_dir}")

        # Load first batch to get samples per file
        first_batch = np.load(self.batch_files[0])
        self.samples_per_file = len(first_batch['inputs'])
        first_batch.close()

        self.total_samples = len(self.batch_files) * self.samples_per_file

        print(f"Dataset: {self.total_samples:,} samples from {len(self.batch_files)} batches")

    def __len__(self):
        return self.total_samples

    def _get_worker_files(self):
        """Distribute files across workers for parallel loading."""
        worker_info = torch.utils.data.get_worker_info()

        # Shuffle files for this epoch (use same seed across workers for consistency)
        files = list(self.batch_files)
        random.shuffle(files)

        if worker_info is None:
            # Single worker - process all files
            return files
        else:
            # Multi-worker - split files among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # Each worker gets every nth file
            return files[worker_id::num_workers]

    def __iter__(self):
        """Iterate through all samples, loading one file at a time."""
        worker_files = self._get_worker_files()

        for batch_file in worker_files:
            # Load entire file (single decompression)
            batch_data = np.load(batch_file)
            inputs = batch_data['inputs']
            outputs = batch_data['outputs']

            # Shuffle indices within this file
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)

            # Yield samples
            for idx in indices:
                input_img = inputs[idx]
                output_img = outputs[idx]

                # Convert to tensor: uint8 [0,255] -> float32 [0,1]
                # Transpose: (H, W, C) -> (C, H, W)
                input_tensor = torch.from_numpy(input_img.copy()).permute(2, 0, 1).float() / 255.0
                output_tensor = torch.from_numpy(output_img.copy()).permute(2, 0, 1).float() / 255.0

                yield input_tensor, output_tensor


def create_dataloader(data_dir: str,
                      batch_size: int = 256,
                      num_batches: int = None,
                      shuffle: bool = True,  # Ignored - shuffling is built into IterableDataset
                      num_workers: int = 8) -> DataLoader:
    """
    Creates a DataLoader optimized for GPU training.

    Args:
        data_dir: Path to maze dataset
        batch_size: Samples per training batch (default 256 for better GPU utilization)
        num_batches: How many .npz files to use (None = all)
        shuffle: Ignored (shuffling is handled internally by the IterableDataset)
        num_workers: Parallel data loading processes (default 8)

    Returns:
        DataLoader configured for efficient GPU training
    """
    dataset = MazeDataset(data_dir, num_batches=num_batches)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Must be False for IterableDataset (shuffling done internally)
        num_workers=num_workers,
        pin_memory=True,           # Faster GPU transfer
        drop_last=True,            # Avoid irregular batch sizes
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,  # Increased prefetch
    )

    return loader
