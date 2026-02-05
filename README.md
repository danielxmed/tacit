# TACIT

TACIT (Transformation-Aware Capturing of Implicit Thought) is a diffusion-based transformer model for image-to-image reasoning tasks. The project demonstrates maze-solving: the model learns to transform images of unsolved mazes into solved mazes using a diffusion process.

## Key Finding: Simultaneous Emergence

Our interpretability analysis revealed that the model exhibits **simultaneous emergence** of the solution path:
- The solution appears through a sharp phase transition at t* ≈ 0.70
- All path segments (start, middle, end) emerge at the same timestep
- This is unlike traditional sequential algorithms (BFS, DFS, A*)
- Analogous to the human "eureka moment" in problem-solving

See `paper_data/reports/` for detailed analysis.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Generate Dataset

```bash
# Generate 100K maze pairs (for testing)
python scripts/generate_data.py --total 100000 --save_dir ./data

# Generate 1M maze pairs (full dataset)
python scripts/generate_data.py --total 1000000 --batch_size 10000 --save_dir ./data
```

### 2. Train Model

```bash
# Train for 50 epochs (uses torch.compile and AMP by default)
python scripts/train.py --data_dir ./data --epochs 50

# Resume from checkpoint
python scripts/train.py --data_dir ./data --checkpoint ./checkpoints/tacit_epoch_15.safetensors --epochs 100
```

### 3. Sample from Model

```bash
python scripts/sample.py --checkpoint ./checkpoints/tacit_epoch_50.safetensors --data_dir ./data
```

### 4. Evaluate Model

```bash
python scripts/evaluate.py --checkpoints ./checkpoints/tacit_epoch_*.safetensors --num_samples 100
```

### 5. Interpretability Analysis

```bash
# Generate test mazes
python scripts/generate_test_mazes.py --output_dir ./paper_data/test_mazes

# Step-by-step visualization with GIFs
python scripts/generate_step_by_step.py --checkpoint ./checkpoints/tacit_epoch_100.safetensors

# Phase transition analysis
python scripts/analyze_emergence.py --checkpoint ./checkpoints/tacit_epoch_100.safetensors

# Spatial emergence patterns
python scripts/analyze_spatial.py --checkpoint ./checkpoints/tacit_epoch_100.safetensors

# Step count convergence
python scripts/compare_step_counts.py --checkpoint ./checkpoints/tacit_epoch_100.safetensors
```

## Repository Structure

```
tacit/
├── tacit/                        # Main package
│   ├── models/                   # Model architecture (DiT blocks, TACITModel)
│   ├── data/                     # Data pipeline (generation, dataset)
│   ├── training/                 # Training code (Trainer class)
│   ├── inference/                # Sampling and visualization
│   └── interpretability/         # Analysis utilities
├── scripts/                      # Entry points
│   ├── generate_data.py          # Dataset generation
│   ├── train.py                  # Model training
│   ├── sample.py                 # Inference/sampling
│   ├── evaluate.py               # Evaluation metrics
│   ├── generate_paper_figures.py # Paper figure generation
│   ├── generate_test_mazes.py    # Test maze generation
│   ├── generate_step_by_step.py  # Step-by-step visualization
│   ├── analyze_emergence.py      # Phase transition analysis
│   ├── analyze_spatial.py        # Spatial pattern analysis
│   └── compare_step_counts.py    # Step count analysis
├── paper_data/                   # Research outputs
│   ├── figures/                  # Training visualizations
│   ├── interpretability/         # Analysis outputs
│   ├── reports/                  # Research analysis reports
│   └── test_mazes/               # Deterministic test set
├── paper_draft/                  # LaTeX paper source
├── data/                         # Dataset (local, gitignored)
├── checkpoints/                  # Model checkpoints (local, gitignored)
└── notebooks/                    # Reference Jupyter notebooks
```

## Usage as Python Package

```python
from tacit import TACITModel, Trainer, sample_euler_method
from tacit.data import MazeDataset, create_dataloader

# Create model
model = TACITModel()

# Load dataset
dataloader = create_dataloader('./data', batch_size=64)

# Train
trainer = Trainer(model=model, learning_rate=1e-4)
# ... training loop
```

## Model Architecture

- **Architecture**: DiT (Diffusion Transformer) with adaptive LayerNorm
- **Hidden dimension**: 384
- **Transformer blocks**: 8
- **Attention heads**: 6
- **Patch size**: 8x8
- **Image resolution**: 64x64 RGB

## Research Reports

Detailed analysis available in `paper_data/reports/`:
- `training_summary.md` - Training dynamics and convergence
- `phase_transition_analysis.md` - Mathematical analysis of the phase transition
- `spatial_emergence_analysis.md` - Spatial patterns of solution emergence
- `philosophical_synthesis.md` - Theoretical implications

## License

Apache License 2.0
