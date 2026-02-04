# TACIT Paper Data

This directory contains figures, metrics, and documentation for the TACIT interpretability paper.

## Directory Structure

```
paper_data/
├── figures/
│   ├── epoch_comparison/
│   │   ├── evolution_grid.png      # Model evolution across epochs 5-100
│   │   └── early_vs_late.png       # Direct comparison: epoch 10 vs 100
│   ├── training_curves/
│   │   ├── loss_curve.png          # Training loss (linear scale)
│   │   ├── loss_curve_log.png      # Training loss (log scale)
│   │   ├── quality_metrics.png     # L2 distance to ground truth
│   │   ├── training_metrics.json   # Raw loss/throughput data
│   │   └── quality_metrics.json    # Raw L2 evaluation data
│   └── maze_samples/
│       ├── epoch_5_samples.png     # Samples at epoch 5
│       ├── epoch_10_samples.png    # Samples at epoch 10
│       ├── epoch_25_samples.png    # Samples at epoch 25
│       ├── epoch_50_samples.png    # Samples at epoch 50
│       ├── epoch_75_samples.png    # Samples at epoch 75
│       └── epoch_100_samples.png   # Samples at epoch 100
├── reports/
│   └── training_summary.md         # Complete training analysis
└── README.md                       # This file
```

## Figure Descriptions

### Training Curves

| File | Description |
|------|-------------|
| `loss_curve.png` | Shows training loss convergence from ~1.2e-03 to ~6.25e-06 over 100 epochs |
| `loss_curve_log.png` | Same data in log scale, better visualizes the learning dynamics |
| `quality_metrics.png` | L2 distance between predictions and ground truth for each checkpoint |

### Epoch Comparison

| File | Description |
|------|-------------|
| `evolution_grid.png` | 6 maze samples showing predictions from epochs 5, 10, 25, 50, 75, 100 plus ground truth |
| `early_vs_late.png` | Direct comparison between early training (epoch 10) and final model (epoch 100) |

### Maze Samples

Individual high-resolution (300 DPI) sample grids for each checkpoint, showing:
- Input (unsolved maze)
- Prediction (model output)
- Ground Truth (correct solution)

## Regenerating Figures

To regenerate all figures from checkpoints:

```bash
python scripts/generate_paper_figures.py \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --output_dir ./paper_data/figures \
    --num_samples 6
```

Options:
- `--epochs 5,25,50,100` - Customize which epochs to compare
- `--skip_metrics` - Skip L2 quality computation (faster)
- `--num_samples N` - Number of maze samples in comparison grids

## Key Metrics

| Metric | Value |
|--------|-------|
| Final Training Loss | 6.25e-06 |
| Final L2 Distance | 0.0014 |
| Total Improvement (Loss) | 192x |
| Total Improvement (L2) | 22.7x |
| Training Epochs | 100 |
| Best Checkpoint | epoch_100 |

## Usage in Paper

Recommended figures for the paper:

1. **Training dynamics**: `loss_curve_log.png` (shows learning phases clearly)
2. **Model evolution**: `evolution_grid.png` (demonstrates learning progression)
3. **Quality improvement**: `quality_metrics.png` (quantitative improvement)
4. **Final results**: `epoch_100_samples.png` (high-quality final outputs)

All figures are 300 DPI, suitable for publication.

## License

Apache License 2.0 - See main repository LICENSE file.
