import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class MazeDataset(Dataset):
    """
    Dataset that loads maze pairs from .npz batch files.

    Loads batches lazily to avoid memory issues with 1M samples.
    """

    def __init__(self, data_dir: str, num_batches: int = None):
        """
        Args:
            data_dir: path to directory containing batch_XXXX.npz files
            num_batches: how many batches to use (None = all)
        """
        self.data_dir = Path(data_dir)
        self.batch_files = sorted(self.data_dir.glob('batch_*.npz'))

        if num_batches is not None:
            self.batch_files = self.batch_files[:num_batches]

        if len(self.batch_files) == 0:
            raise ValueError(f"No batch files found in {data_dir}")

        # Load first batch to get batch size
        first_batch = np.load(self.batch_files[0])
        self.batch_size = len(first_batch['inputs'])
        first_batch.close()

        self.total_samples = len(self.batch_files) * self.batch_size

        # Cache for current loaded batch (avoid reloading constantly)
        self._cached_batch_idx = -1
        self._cached_inputs = None
        self._cached_outputs = None

        print(f"Dataset: {self.total_samples:,} samples from {len(self.batch_files)} batches")

    def __len__(self):
        return self.total_samples

    def _load_batch(self, batch_idx: int):
        """Load a batch into cache."""
        if batch_idx != self._cached_batch_idx:
            batch_data = np.load(self.batch_files[batch_idx])
            self._cached_inputs = batch_data['inputs']
            self._cached_outputs = batch_data['outputs']
            self._cached_batch_idx = batch_idx
            batch_data.close()

    def __getitem__(self, idx: int):
        """
        Returns a single (input, output) pair as tensors.

        Images are converted from uint8 [0, 255] to float32 [0, 1]
        and from (H, W, C) to (C, H, W) for PyTorch.
        """
        batch_idx = idx // self.batch_size
        sample_idx = idx % self.batch_size

        self._load_batch(batch_idx)

        # Get images as numpy arrays
        input_img = self._cached_inputs[sample_idx]
        output_img = self._cached_outputs[sample_idx]

        # Convert to tensor: uint8 [0,255] -> float32 [0,1]
        # Transpose: (H, W, C) -> (C, H, W)
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
        output_tensor = torch.from_numpy(output_img).permute(2, 0, 1).float() / 255.0

        return input_tensor, output_tensor


def create_dataloader(data_dir: str,
                      batch_size: int = 64,
                      num_batches: int = None,
                      shuffle: bool = True,
                      num_workers: int = 2) -> DataLoader:
    """
    Creates a DataLoader for training.

    Args:
        data_dir: path to maze dataset
        batch_size: samples per training batch
        num_batches: how many .npz files to use (None = all)
        shuffle: randomize order each epoch
        num_workers: parallel data loading processes
    """
    dataset = MazeDataset(data_dir, num_batches=num_batches)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True    # Avoid weird batch sizes at the end
    )

    return loader
