# TACIT Paper Data

This directory contains figures, metrics, analysis outputs, and documentation for the TACIT interpretability research.

## Directory Structure

```
paper_data/
├── figures/
│   ├── epoch_comparison/           # Model evolution visualizations
│   │   ├── evolution_grid.png      # Model evolution across epochs 5-100
│   │   └── early_vs_late.png       # Direct comparison: epoch 10 vs 100
│   ├── training_curves/            # Training metrics
│   │   ├── loss_curve.png          # Training loss (linear scale)
│   │   ├── loss_curve_log.png      # Training loss (log scale)
│   │   ├── quality_metrics.png     # L2 distance to ground truth
│   │   ├── training_metrics.json   # Raw loss/throughput data
│   │   └── quality_metrics.json    # Raw L2 evaluation data
│   └── maze_samples/               # Per-epoch sample visualizations
│       └── epoch_N_samples.png     # Samples at each checkpoint
├── interpretability/               # Interpretability analysis outputs
│   ├── emergence/                  # Phase transition analysis
│   │   ├── curves/                 # Emergence curves over time
│   │   ├── metrics/                # Statistics and CSV data
│   │   ├── transitions/            # Phase transition histograms
│   │   └── paper_figure_emergence.png
│   ├── spatial/                    # Spatial pattern analysis
│   │   ├── emergence_order/        # First appearance visualizations
│   │   ├── heatmaps/               # Spatial activation maps
│   │   ├── patterns/               # Pattern classification
│   │   ├── segments/               # Segment emergence analysis
│   │   └── paper_figure_spatial.png
│   ├── step_by_step/               # Step-by-step transformation
│   │   ├── grids/                  # Step grids (PNG)
│   │   ├── gifs/                   # Animated transformations (GIF)
│   │   ├── metrics/                # Quality metrics per step
│   │   └── summary_grid_steps_N.png
│   └── step_comparison/            # Step count convergence
│       ├── samples/                # Sample comparisons
│       ├── convergence_curves.png  # Convergence over steps
│       ├── quality_vs_steps.png    # Quality metrics vs step count
│       └── step_count_metrics.json # Quantitative data
├── reports/                        # Research analysis reports
│   ├── training_summary.md         # Training dynamics analysis
│   ├── phase_transition_analysis.md # Mathematical phase transition analysis
│   ├── spatial_emergence_analysis.md # Spatial patterns of emergence
│   ├── philosophical_synthesis.md  # Theoretical implications
│   ├── citation_context_verification.md
│   └── bibliography_verification.md
├── test_mazes/                     # Deterministic test maze set
│   ├── maze_NNNN_input.npy         # Input mazes
│   └── maze_NNNN_solution.npy      # Ground truth solutions
├── architecture_diagram_prompt.md  # Architecture reference
└── README.md                       # This file
```

## Figure Descriptions

### Training Curves (`figures/training_curves/`)

| File | Description |
|------|-------------|
| `loss_curve.png` | Training loss convergence from ~1.2e-03 to ~6.25e-06 over 100 epochs |
| `loss_curve_log.png` | Same data in log scale, visualizes learning dynamics |
| `quality_metrics.png` | L2 distance between predictions and ground truth |

### Epoch Comparison (`figures/epoch_comparison/`)

| File | Description |
|------|-------------|
| `evolution_grid.png` | 6 maze samples showing predictions from epochs 5, 10, 25, 50, 75, 100 plus ground truth |
| `early_vs_late.png` | Direct comparison between early training (epoch 10) and final model (epoch 100) |

### Maze Samples (`figures/maze_samples/`)

Individual high-resolution (300 DPI) sample grids for each checkpoint, showing:
- Input (unsolved maze)
- Prediction (model output)
- Ground Truth (correct solution)

## Interpretability Analysis

### Emergence Analysis (`interpretability/emergence/`)

Quantitative analysis of how the solution path emerges during transformation:

| Output | Description |
|--------|-------------|
| `curves/emergence_curves_all.png` | IoU/recall curves for all samples |
| `curves/emergence_mean_with_ci.png` | Mean curve with confidence interval |
| `curves/emergence_rate.png` | Rate of emergence (derivative) |
| `metrics/summary_statistics.json` | Aggregate statistics |
| `metrics/emergence_data_full.csv` | Full emergence data |
| `metrics/transition_points.csv` | Detected phase transition points |
| `transitions/phase_transition_histogram.png` | Distribution of transition times |
| `paper_figure_emergence.png` | Publication-ready figure |

**Key Finding**: Phase transition at t* ≈ 0.70 with transition width Δt ≈ 0.02

### Spatial Analysis (`interpretability/spatial/`)

Analysis of spatial patterns in solution emergence:

| Output | Description |
|--------|-------------|
| `emergence_order/first_appearance_*.png` | First pixel appearance per sample |
| `heatmaps/aggregate_heatmap.png` | Aggregate emergence heatmap |
| `heatmaps/sample_*_change_sequence.png` | Per-sample change sequences |
| `patterns/pattern_classification.csv` | Pattern type classification |
| `patterns/pattern_distribution.png` | Distribution of emergence patterns |
| `patterns/pattern_summary.json` | Pattern statistics |
| `segments/segment_emergence_aggregate.png` | Start/middle/end segment comparison |
| `paper_figure_spatial.png` | Publication-ready figure |

**Key Finding**: All path segments (start, middle, end) emerge simultaneously

### Step-by-Step Visualization (`interpretability/step_by_step/`)

Visualizations of the transformation process:

| Output | Description |
|--------|-------------|
| `grids/sample_*_steps_*.png` | Step grid for each sample |
| `gifs/sample_*_steps_*.gif` | Animated transformation GIFs |
| `metrics/trajectory_metrics.json` | Quality metrics along trajectory |
| `summary_grid_steps_*.png` | Summary grids for different step counts |

### Step Comparison (`interpretability/step_comparison/`)

Analysis of convergence with different Euler step counts:

| Output | Description |
|--------|-------------|
| `convergence_curves.png` | Quality convergence curves |
| `quality_vs_steps.png` | Final quality vs step count |
| `sufficient_steps_analysis.png` | Minimum steps analysis |
| `visual_comparison.png` | Visual comparison grid |
| `samples/*.png` | Per-sample comparisons |
| `step_count_metrics.json` | Quantitative metrics |

**Key Finding**: 10 steps sufficient for >95% quality; diminishing returns beyond 20 steps

## Regenerating Outputs

### Training Figures

```bash
python scripts/generate_paper_figures.py \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints \
    --output_dir ./paper_data/figures \
    --num_samples 6
```

### Interpretability Analysis

```bash
# Step-by-step visualization
python scripts/generate_step_by_step.py \
    --checkpoint ./checkpoints/tacit_epoch_100.safetensors \
    --output_dir ./paper_data/interpretability/step_by_step

# Emergence analysis
python scripts/analyze_emergence.py \
    --checkpoint ./checkpoints/tacit_epoch_100.safetensors \
    --output_dir ./paper_data/interpretability/emergence

# Spatial analysis
python scripts/analyze_spatial.py \
    --checkpoint ./checkpoints/tacit_epoch_100.safetensors \
    --output_dir ./paper_data/interpretability/spatial

# Step comparison
python scripts/compare_step_counts.py \
    --checkpoint ./checkpoints/tacit_epoch_100.safetensors \
    --output_dir ./paper_data/interpretability/step_comparison
```

## Key Metrics Summary

| Metric | Value |
|--------|-------|
| Final Training Loss | 6.25e-06 |
| Final L2 Distance | 0.0014 |
| Phase Transition Time | t* ≈ 0.70 |
| Transition Width | Δt ≈ 0.02 |
| Final Path IoU | 0.9706 ± 0.071 |
| Recommended Euler Steps | 10-20 |

## Research Reports

| Report | Description |
|--------|-------------|
| `training_summary.md` | Complete training analysis with loss curves and convergence |
| `phase_transition_analysis.md` | Mathematical analysis of the sharp phase transition |
| `spatial_emergence_analysis.md` | Analysis of spatial patterns (simultaneous emergence) |
| `philosophical_synthesis.md` | Theoretical implications for AI and cognitive science |

## Usage in Paper

Recommended figures:

1. **Training dynamics**: `figures/training_curves/loss_curve_log.png`
2. **Model evolution**: `figures/epoch_comparison/evolution_grid.png`
3. **Phase transition**: `interpretability/emergence/paper_figure_emergence.png`
4. **Spatial patterns**: `interpretability/spatial/paper_figure_spatial.png`
5. **Step-by-step**: `interpretability/step_by_step/summary_grid_steps_20.png`

All figures are 300 DPI, suitable for publication.

## License

Apache License 2.0 - See main repository LICENSE file.
