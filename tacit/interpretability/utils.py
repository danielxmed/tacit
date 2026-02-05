"""
Shared utility functions for TACIT interpretability analysis.

This module provides common functionality used across all interpretability scripts,
including model loading, trajectory sampling, path detection, and visualization helpers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from safetensors.torch import load_file


def load_checkpoint_flexible(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """
    Load checkpoint handling both compiled and non-compiled model formats.

    Handles the '_orig_mod.' prefix that torch.compile adds to state dict keys.

    Args:
        model: TACITModel instance to load weights into
        checkpoint_path: Path to safetensors checkpoint file

    Returns:
        Model with loaded weights
    """
    state_dict = load_file(checkpoint_path)
    sample_key = next(iter(state_dict.keys()))
    if sample_key.startswith('_orig_mod.'):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


def sample_euler_with_trajectory(
    model: torch.nn.Module,
    x0: torch.Tensor,
    num_steps: int = 10,
    capture_every: int = 1
) -> Tuple[List[Tuple[float, torch.Tensor]], torch.Tensor]:
    """
    Euler sampling that captures intermediate states during the transformation.

    This is the core function for interpretability analysis - it allows us to
    observe how the model transforms the input step by step.

    Args:
        model: TACITModel instance in eval mode
        x0: Input tensor (batch, 3, 64, 64) - the unsolved maze
        num_steps: Number of Euler integration steps
        capture_every: Capture state every N steps (1 = capture all)

    Returns:
        intermediates: List of (timestep, state_tensor) for each captured step
        final: Final output tensor x_1
    """
    step_delta_t = 1 / num_steps
    x = x0.clone()
    t = 0.0
    intermediates = [(0.0, x.clone())]

    with torch.no_grad():
        for step in range(num_steps):
            effective_t = torch.ones(x0.shape[0], device=x0.device) * t
            v_pred = model(x, effective_t)
            x = x + (v_pred * step_delta_t)
            t += step_delta_t

            if (step + 1) % capture_every == 0:
                intermediates.append((t, x.clone()))

    return intermediates, x


def extract_red_path_mask(
    image_tensor: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Extract binary mask of red pixels (solution path) from image tensor.

    The solution path in TACIT mazes is marked in red (high R, low G, low B).

    Args:
        image_tensor: Image tensor (3, H, W) with values in [0, 1]
        threshold: Threshold for red channel detection

    Returns:
        Binary mask (H, W) where True = red pixel (part of solution)
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
    # Red pixels: R > threshold, G < 0.3, B < 0.3
    red_mask = (r > threshold) & (g < 0.3) & (b < 0.3)
    return red_mask


def detect_red_pixels(
    image_tensor: torch.Tensor,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, float]:
    """
    Detect red pixels and compute redness intensity.

    Args:
        image_tensor: Image tensor (3, H, W) with values in [0, 1]
        threshold: Threshold for red channel detection

    Returns:
        mask: Binary mask (H, W) where True = red pixel
        intensity: Average "redness" value across the image
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)

    r, g, b = image_tensor[0], image_tensor[1], image_tensor[2]
    # Red pixels: R > threshold, G and B low
    red_mask = (r > threshold) & (g < 0.3) & (b < 0.3)

    # Redness intensity: R - average of G and B, clamped to positive
    intensity = (r - (g + b) / 2).clamp(0).mean()

    return red_mask, intensity.item()


def compute_red_channel_metrics(
    x_t: torch.Tensor,
    x1_gt: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute metrics about red channel (solution path) at current state.

    Args:
        x_t: Current state tensor (3, H, W) or (1, 3, H, W)
        x1_gt: Ground truth solution tensor
        threshold: Threshold for red detection

    Returns:
        Dictionary with:
        - red_count: Number of red pixels in x_t
        - red_fraction: Fraction of image that is red
        - gt_red_count: Number of red pixels in ground truth
        - iou: Intersection over Union with ground truth path
        - recall: Fraction of GT path pixels found
        - precision: Fraction of predicted red that matches GT
    """
    if x_t.dim() == 4:
        x_t = x_t.squeeze(0)
    if x1_gt.dim() == 4:
        x1_gt = x1_gt.squeeze(0)

    pred_mask = extract_red_path_mask(x_t, threshold)
    gt_mask = extract_red_path_mask(x1_gt, threshold)

    pred_count = pred_mask.sum().item()
    gt_count = gt_mask.sum().item()

    intersection = (pred_mask & gt_mask).sum().item()
    union = (pred_mask | gt_mask).sum().item()

    iou = intersection / union if union > 0 else 0.0
    recall = intersection / gt_count if gt_count > 0 else 0.0
    precision = intersection / pred_count if pred_count > 0 else 0.0

    total_pixels = x_t.shape[1] * x_t.shape[2]

    return {
        'red_count': pred_count,
        'red_fraction': pred_count / total_pixels,
        'gt_red_count': gt_count,
        'iou': iou,
        'recall': recall,
        'precision': precision
    }


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy image array.

    Args:
        tensor: Image tensor (C, H, W) or (1, C, H, W) with values in [0, 1]

    Returns:
        Numpy array (H, W, C) with values in [0, 1], suitable for plt.imshow
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    # Clamp to valid range
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy and transpose from (C, H, W) to (H, W, C)
    img = tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))

    return img


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Convert numpy image array to PyTorch tensor.

    Args:
        image: Numpy array (H, W, C) with values in [0, 255] or [0, 1]

    Returns:
        Tensor (C, H, W) with values in [0, 1]
    """
    # Handle uint8 images
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # Transpose from (H, W, C) to (C, H, W)
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()

    return tensor


def setup_output_dirs(output_dir: Union[str, Path], subdirs: List[str]) -> Dict[str, Path]:
    """
    Create output directory structure for interpretability analysis.

    Args:
        output_dir: Base output directory path
        subdirs: List of subdirectory names to create

    Returns:
        Dictionary mapping subdirectory names to their Path objects
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {'root': output_dir}
    for subdir in subdirs:
        path = output_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = path

    return paths


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    dpi: int = 150,
    close: bool = True
) -> None:
    """
    Save matplotlib figure with consistent settings.

    Args:
        fig: Matplotlib figure to save
        path: Output file path
        dpi: Resolution in dots per inch
        close: Whether to close the figure after saving
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')

    if close:
        plt.close(fig)


def load_samples_from_data_dir(
    data_dir: str,
    num_samples: int,
    seed: int = 42
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load maze samples from data directory with deterministic selection.

    Args:
        data_dir: Path to directory containing batch_*.npz files
        num_samples: Number of samples to load
        seed: Random seed for reproducible sample selection

    Returns:
        List of (input_tensor, output_tensor) tuples
    """
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
    available = min(len(inputs), num_samples)
    indices = np.random.choice(len(inputs), size=available, replace=False)

    samples = []
    for idx in indices:
        input_img = inputs[idx]
        output_img = outputs[idx]

        input_tensor = torch.from_numpy(input_img.copy()).permute(2, 0, 1).float() / 255.0
        output_tensor = torch.from_numpy(output_img.copy()).permute(2, 0, 1).float() / 255.0

        samples.append((input_tensor, output_tensor))

    return samples


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute Mean Squared Error between prediction and target."""
    if pred.dim() == 4:
        pred = pred.squeeze(0)
    if target.dim() == 4:
        target = target.squeeze(0)
    return torch.mean((pred - target) ** 2).item()


def compute_psnr(mse: float) -> float:
    """Compute Peak Signal-to-Noise Ratio from MSE."""
    if mse > 0:
        return 10 * np.log10(1.0 / mse)
    return float('inf')
