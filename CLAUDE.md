# CLAUDE.md - AI Assistant Guide for TACIT

## Project Overview

**TACIT** (Task-Agnostic Continuous Image-to-Image Translation) is a machine learning research project implementing a diffusion-based transformer model for solving maze problems. The model learns to transform images of unsolved mazes into solved mazes using a diffusion process.

## Repository Structure

```
tacit/
├── LICENSE                                    # Apache 2.0 license
├── CLAUDE.md                                  # This file
└── notebooks/
    ├── 1_TACIT_maze_data_generator.ipynb     # Dataset generation (1M samples)
    └── 2_TACIT_maze_training.ipynb           # Model training and evaluation
```

## Technology Stack

- **Deep Learning Framework**: PyTorch (with CUDA support)
- **Numerical Computing**: NumPy
- **Image Processing**: PIL/Pillow
- **Visualization**: Matplotlib
- **Model Serialization**: SafeTensors
- **Execution Environment**: Google Colab (GPU: NVIDIA Tesla T4)
- **Progress Tracking**: tqdm

## Key Components

### Notebook 1: Data Generator (`1_TACIT_maze_data_generator.ipynb`)

Generates 1 million maze problem-solution pairs:

| Component | Description |
|-----------|-------------|
| `generate_maze()` | Creates mazes using iterative DFS/backtracking |
| `solve_maze()` | Finds shortest path using BFS |
| `render_maze()` | Converts maze arrays to 64x64 RGB images |
| `generate_maze_pair()` | Creates unsolved/solved image pairs |

**Output**: 100 compressed .npz files, each containing 10,000 samples (~120MB per file)

### Notebook 2: Training (`2_TACIT_maze_training.ipynb`)

Trains the diffusion transformer model:

| Class | Purpose |
|-------|---------|
| `MazeDataset` | PyTorch Dataset with lazy batch loading |
| `TimestepEmbedder` | Sinusoidal timestep encoding (256-dim) |
| `PatchEmbed` | 8x8 patch embedding to 384-dim |
| `DiTBlock` | Transformer block with adaptive LayerNorm |
| `FinalLayer` | Reconstructs patches back to images |
| `TACITModel` | Complete architecture (8 transformer blocks) |
| `Trainer` | Training loop with Adam optimizer |

## Model Architecture

```
Input (bs, 3, 64, 64) + timestep
    ↓
PatchEmbed → (bs, 64, 384) [8x8 patches]
    ↓
+ 2D Sinusoidal Position Embedding
    ↓
8× DiTBlock (384 hidden, 6 heads, adaLN)
    ↓
FinalLayer
    ↓
Output (bs, 3, 64, 64)
```

**Key Hyperparameters**:
- Hidden dimension: 384
- Transformer blocks: 8
- Attention heads: 6
- Patch size: 8×8
- Learning rate: 1e-4
- Inference steps: 10 (Euler sampling)

## Development Workflow

### Running the Pipeline

1. **Generate Dataset** (Notebook 1):
   - Mount Google Drive
   - Run all cells to generate 1M maze pairs
   - Output saved to `/content/drive/MyDrive/notebooks_tacit/maze_dataset/`

2. **Train Model** (Notebook 2):
   - Mount Google Drive and load dataset
   - Initialize model and trainer
   - Train with checkpointing every 5 epochs
   - Evaluate using `euler_sample()` function

### Directory Conventions

- **Dataset**: `notebooks_tacit/maze_dataset/` (Google Drive)
- **Checkpoints**: `notebooks_tacit/checkpoints/` (Google Drive)
- **Checkpoint format**: `checkpoint_epoch_N.safetensors`

## Code Conventions

### Naming Standards

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `DiTBlock`, `TACITModel` |
| Functions | snake_case | `generate_maze`, `euler_sample` |
| Constants | UPPER_SNAKE_CASE | `DATA_DIR`, `CHECKPOINT_DIR` |
| Variables | snake_case | `hidden_size`, `num_heads` |

### Documentation Standards

- Use comprehensive docstrings with `Args:` and `Returns:` sections
- Include type hints for function parameters and return values
- Comment complex algorithmic sections

### Image Format Conventions

- **Resolution**: 64×64 pixels, RGB
- **Storage format**: uint8 [0, 255] in .npz files
- **Training format**: float32 [0, 1], channels-first (C, H, W)
- **Color scheme**:
  - White (255, 255, 255): Paths
  - Black (0, 0, 0): Walls
  - Green (0, 255, 0): Entry/Exit points
  - Red (255, 0, 0): Solution path

## Important Implementation Details

### Diffusion Training Objective

The model learns to predict the velocity field (direction from problem to solution):
```python
# Interpolation
x_t = (1 - t) * x_0 + t * x_1  # x_0=input, x_1=target

# Target velocity
velocity = x_1 - x_0

# Loss: MSE between predicted and target velocity
```

### Memory Management

- **Lazy loading**: Dataset only loads one batch at a time
- **Batch caching**: Keeps current batch in memory for efficiency
- **Pin memory**: Enabled for faster GPU transfer
- **Drop last**: Ensures consistent batch sizes

### Checkpoint Management

```python
# Saving
safetensors.torch.save_file(model.state_dict(), path)

# Loading
load_checkpoint(model, optimizer, checkpoint_path)
```

## Common Tasks for AI Assistants

### Adding New Maze Sizes

Modify the `maze_sizes` list in Notebook 1:
```python
maze_sizes = [11, 15, 21, 25, 31]  # Must be odd numbers
```

### Adjusting Model Capacity

In Notebook 2, modify `TACITModel` initialization:
```python
model = TACITModel(
    hidden_size=384,      # Increase for more capacity
    depth=8,              # More transformer blocks
    num_heads=6,          # More attention heads
    patch_size=8          # Smaller = more patches
)
```

### Changing Training Parameters

```python
trainer = Trainer(model, learning_rate=1e-4)  # Adjust LR
# In train(): batch_size, num_epochs, checkpoint_every
```

## Testing and Validation

### Quick Model Verification

```python
# Test forward pass
model = TACITModel()
x = torch.randn(2, 3, 64, 64)
t = torch.rand(2)
out = model(x, t)
assert out.shape == (2, 3, 64, 64)
```

### Visual Evaluation

Use `visualize_predictions()` to compare:
- Input (unsolved maze)
- Model prediction
- Ground truth (solved maze)

## File Size Reference

| File | Approximate Size |
|------|------------------|
| Each .npz batch | ~120 MB |
| Full dataset (100 batches) | ~12 GB |
| Model checkpoint | ~50 MB |

## Dependencies

```python
# Core
torch>=2.0
numpy
Pillow
matplotlib
tqdm

# Model saving
safetensors

# Environment (Colab)
google-colab  # For Drive mounting
```

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
