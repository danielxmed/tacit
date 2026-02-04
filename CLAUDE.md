# CLAUDE.md - AI Assistant Guide for TACIT

## Project Overview

**TACIT** (Transformation-Aware Capturing of Implicit Thought) is a machine learning research project implementing a diffusion-based transformer model for image-to-image reasoning tasks. The current demonstration uses maze-solving: the model learns to transform images of unsolved mazes into solved mazes using a diffusion process.

## Repository Structure

```
tacit/
├── tacit/                        # Main Python package
│   ├── __init__.py               # Package exports (TACITModel, Trainer, sample_euler_method)
│   ├── models/
│   │   └── dit.py                # DiTBlock, TACITModel, embeddings
│   ├── data/
│   │   ├── generation.py         # Maze generation and rendering
│   │   └── dataset.py            # MazeDataset, create_dataloader
│   ├── training/
│   │   └── trainer.py            # Trainer class, training loop
│   └── inference/
│       └── sampling.py           # Euler sampling, visualization
├── scripts/                      # Entry point scripts
│   ├── generate_data.py          # Dataset generation
│   ├── train.py                  # Model training (with torch.compile, AMP)
│   ├── sample.py                 # Inference/sampling
│   ├── evaluate.py               # Evaluation with PSNR, Path IoU metrics
│   └── generate_paper_figures.py # Paper figure generation
├── paper_data/                   # Paper figures and metrics
│   ├── figures/
│   │   ├── training_curves/      # Loss curves, quality metrics
│   │   ├── epoch_comparison/     # Evolution grids
│   │   └── maze_samples/         # Per-epoch sample visualizations
│   ├── reports/                  # Training summaries
│   └── README.md
├── data/                         # Dataset directory (local, gitignored)
├── checkpoints/                  # Model checkpoints (local, gitignored)
├── notebooks/                    # Original Jupyter notebooks (reference)
│   ├── 1_TACIT_maze_data_generator.ipynb
│   └── 2_TACIT_maze_training.ipynb
├── pyproject.toml                # Package configuration
├── requirements.txt
├── .gitignore
├── LICENSE
├── CLAUDE.md
└── README.md
```

## Technology Stack

- **Deep Learning Framework**: PyTorch 2.0+ (with CUDA support, torch.compile, AMP)
- **Numerical Computing**: NumPy
- **Image Processing**: PIL/Pillow
- **Visualization**: Matplotlib
- **Model Serialization**: SafeTensors
- **Progress Tracking**: tqdm

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in editable mode (recommended for development)
pip install -e .
```

## Key Components

### Data Generation (`tacit/data/generation.py`)

Generates maze problem-solution pairs:

| Component | Description |
|-----------|-------------|
| `generate_maze()` | Creates mazes using iterative DFS/backtracking |
| `solve_maze()` | Finds shortest path using BFS |
| `render_maze()` | Converts maze arrays to 64x64 RGB images |
| `generate_maze_pair()` | Creates unsolved/solved image pairs |
| `generate_dataset()` | Generates full dataset in batches |

**Output**: Compressed .npz files, each containing 10,000 samples (~120MB per file)

### Model (`tacit/models/dit.py`)

DiT-based architecture:

| Class | Purpose |
|-------|---------|
| `TimestepEmbedder` | Sinusoidal timestep encoding (256-dim) |
| `PatchEmbed` | 8x8 patch embedding to 384-dim |
| `DiTBlock` | Transformer block with adaptive LayerNorm |
| `FinalLayer` | Reconstructs patches back to images |
| `TACITModel` | Complete architecture (8 transformer blocks) |

### Training (`tacit/training/trainer.py`)

| Class/Function | Purpose |
|----------------|---------|
| `Trainer` | Training loop with Adam optimizer, AMP support |
| `train()` | Main training function with checkpointing |
| `load_checkpoint()` | Load model from safetensors file |

### Dataset (`tacit/data/dataset.py`)

| Class | Purpose |
|-------|---------|
| `MazeDataset` | PyTorch Dataset with lazy batch loading |
| `create_dataloader()` | Creates DataLoader for training |

### Inference (`tacit/inference/sampling.py`)

| Function | Purpose |
|----------|---------|
| `sample_euler_method()` | Euler sampling for diffusion inference |
| `visualize_predictions()` | Create comparison visualizations |

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
- Default batch size: 256
- Inference steps: 10 (Euler sampling)

## Development Workflow

### Running the Pipeline

1. **Generate Dataset**:
   ```bash
   python scripts/generate_data.py --total 1000000 --save_dir ./data
   ```

2. **Train Model**:
   ```bash
   # Standard training (uses torch.compile and AMP by default)
   python scripts/train.py --data_dir ./data --epochs 50

   # Resume from checkpoint
   python scripts/train.py --data_dir ./data --checkpoint ./checkpoints/tacit_epoch_25.safetensors --epochs 100

   # Disable optimizations if needed
   python scripts/train.py --data_dir ./data --no_compile --no_amp
   ```

3. **Sample/Visualize**:
   ```bash
   python scripts/sample.py --checkpoint ./checkpoints/tacit_epoch_50.safetensors --data_dir ./data
   ```

4. **Evaluate Model**:
   ```bash
   # Compare multiple checkpoints
   python scripts/evaluate.py --checkpoints ./checkpoints/tacit_epoch_10.safetensors ./checkpoints/tacit_epoch_50.safetensors
   ```

5. **Generate Paper Figures**:
   ```bash
   python scripts/generate_paper_figures.py --data_dir ./data --checkpoint_dir ./checkpoints
   ```

### Directory Conventions

- **Dataset**: `./data/` (local, gitignored)
- **Checkpoints**: `./checkpoints/` (local, gitignored)
- **Checkpoint format**: `tacit_epoch_N.safetensors`
- **Paper figures**: `./paper_data/figures/`

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

- **Resolution**: 64x64 pixels, RGB
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

### Training Optimizations

The training script includes several performance optimizations:
- **torch.compile()**: Graph optimization for fused operations (PyTorch 2.0+)
- **Automatic Mixed Precision (AMP)**: FP16 training with gradient scaling
- **Optimized DataLoader**: 8 workers, pinned memory, prefetching

### Memory Management

- **Lazy loading**: Dataset only loads one batch at a time
- **Batch caching**: Keeps current batch in memory for efficiency
- **Pin memory**: Enabled for faster GPU transfer
- **Drop last**: Ensures consistent batch sizes

### Checkpoint Management

```python
# Saving (handles compiled models)
safetensors.torch.save_file(model.state_dict(), path)

# Loading (handles _orig_mod. prefix from compiled models)
load_checkpoint(trainer, checkpoint_path)
```

## Evaluation Metrics

The `evaluate.py` script computes:

| Metric | Description | Better |
|--------|-------------|--------|
| MSE | Mean Squared Error | Lower |
| MAE | Mean Absolute Error | Lower |
| PSNR | Peak Signal-to-Noise Ratio (dB) | Higher |
| Path IoU | Intersection over Union for solution path | Higher |

## Common Tasks for AI Assistants

### Adding New Maze Sizes

Modify `generate_maze_pair()` in `tacit/data/generation.py`:
```python
size = random.choice([11, 15, 21, 25, 31])  # Must be odd numbers
```

### Adjusting Model Capacity

Modify `TACITModel` initialization in `tacit/models/dit.py`:
```python
model = TACITModel(
    hidden_size=384,      # Increase for more capacity
    num_blocks=8,         # More transformer blocks
    num_heads=6,          # More attention heads
    patch_size=8          # Smaller = more patches
)
```

### Changing Training Parameters

Via CLI:
```bash
python scripts/train.py --lr 1e-4 --batch_size 256 --epochs 50 --num_workers 8
```

Or programmatically:
```python
from tacit import TACITModel, Trainer
from tacit.data import create_dataloader

model = TACITModel()
trainer = Trainer(model, learning_rate=1e-4, use_amp=True)
dataloader = create_dataloader('./data', batch_size=256)
```

### Using as a Package

```python
from tacit import TACITModel, Trainer, sample_euler_method
from tacit.data import MazeDataset, create_dataloader

# Create and use model
model = TACITModel()
dataloader = create_dataloader('./data', batch_size=64)
trainer = Trainer(model=model, learning_rate=1e-4)
```

## Testing and Validation

### Quick Model Verification

```python
import torch
from tacit import TACITModel

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

### Quantitative Evaluation

```bash
python scripts/evaluate.py \
    --checkpoints ./checkpoints/tacit_epoch_*.safetensors \
    --num_samples 100 \
    --output ./evaluation.png
```

## File Size Reference

| File | Approximate Size |
|------|------------------|
| Each .npz batch | ~120 MB |
| Full dataset (100 batches) | ~12 GB |
| Model checkpoint | ~50 MB |

## Dependencies

Core dependencies (see `requirements.txt`):
```
torch>=2.0
numpy
Pillow
matplotlib
tqdm
safetensors
```

Development dependencies:
```
pytest
```

## Git Conventions

The `.gitignore` excludes:
- `data/*.npz` - Large dataset files
- `checkpoints/*.safetensors` - Model checkpoints
- `__pycache__/`, `*.pyc` - Python cache
- `.ipynb_checkpoints/` - Jupyter artifacts
- IDE files (`.vscode/`, `.idea/`)

## License

This project is licensed under the Apache License 2.0. See `LICENSE` for details.
