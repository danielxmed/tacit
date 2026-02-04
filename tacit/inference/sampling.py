import torch
import numpy as np
import matplotlib.pyplot as plt


def sample_euler_method(model, x0, num_steps=10):
    """
    Generate solution using Euler method sampling.

    Args:
        model: TACITModel instance
        x0: input tensor (batch, 3, 64, 64)
        num_steps: number of Euler steps

    Returns:
        x: generated output tensor
    """
    step_delta_t = 1 / num_steps
    x = x0
    t = 0.0

    with torch.no_grad():
        for step in range(num_steps):
            effective_t = torch.ones(x0.shape[0], device=x0.device)
            effective_t = effective_t * t
            v_pred = model(x, effective_t)
            x = x + (v_pred * step_delta_t)
            t += step_delta_t

    return x


def visualize_predictions(model, dataset, num_samples=4, num_steps=10, device=None,
                          save_path=None):
    """
    Visualize model predictions compared to ground truth.

    Args:
        model: TACITModel instance
        dataset: MazeDataset instance (IterableDataset)
        num_samples: number of samples to visualize
        num_steps: Euler sampling steps
        device: torch device (auto-detects if None)
        save_path: if provided, save figure to this path instead of showing
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Collect samples from the IterableDataset
    samples = []
    for i, (x0, x1) in enumerate(dataset):
        samples.append((x0, x1))
        if len(samples) >= num_samples:
            break

    if len(samples) < num_samples:
        print(f"Warning: only got {len(samples)} samples (requested {num_samples})")
        num_samples = len(samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

    # Handle single sample case (axes is 1D)
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        x0, x1 = samples[i]

        x0_batch = x0.unsqueeze(0).to(device)

        x1_pred = sample_euler_method(model, x0_batch, num_steps)

        x0_vis = x0.permute(1, 2, 0).numpy()
        x1_vis = x1.permute(1, 2, 0).numpy()
        x1_pred_vis = x1_pred[0].cpu().permute(1, 2, 0).numpy()
        x1_pred_vis = np.clip(x1_pred_vis, 0, 1)

        axes[i, 0].imshow(x0_vis)
        axes[i, 0].set_title('Problem')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(x1_vis)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(x1_pred_vis)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    model.train()
