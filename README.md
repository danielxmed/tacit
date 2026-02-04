# TACIT

TACIT (Transformation-Aware Capturing of Implicit Thought) is a diffusion-based transformer model for image-to-image reasoning tasks.

> **Note**: This repository is under active development. Documentation will be expanded as the project matures.

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
# Train for 50 epochs
python scripts/train.py --data_dir ./data --epochs 50

# Resume from checkpoint
python scripts/train.py --data_dir ./data --checkpoint ./checkpoints/tacit_epoch_15.safetensors --epochs 100
```

### 3. Sample from Model

```bash
python scripts/sample.py --checkpoint ./checkpoints/tacit_epoch_50.safetensors --data_dir ./data
```

## Repository Structure

```
tacit/
├── tacit/                    # Main package
│   ├── models/               # Model architecture
│   │   └── dit.py            # DiT blocks, embeddings, TACITModel
│   ├── data/                 # Data pipeline
│   │   ├── generation.py     # Maze generation and rendering
│   │   └── dataset.py        # PyTorch Dataset and DataLoader
│   ├── training/             # Training code
│   │   └── trainer.py        # Trainer class and training loop
│   └── inference/            # Inference code
│       └── sampling.py       # Euler sampling and visualization
├── scripts/                  # Entry points
│   ├── generate_data.py      # Dataset generation script
│   ├── train.py              # Training script
│   └── sample.py             # Inference script
├── data/                     # Dataset directory (local)
├── checkpoints/              # Model checkpoints
├── notebooks/                # Original Jupyter notebooks
├── requirements.txt
└── README.md
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

## License

Apache License 2.0
