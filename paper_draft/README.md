# TACIT Paper Draft

This directory contains the LaTeX source for the TACIT research paper.

## Files

- `main.tex` - Main paper source
- `Makefile` - Build automation

## Building the PDF

```bash
# Using make
make

# Or directly with pdflatex
pdflatex main.tex
pdflatex main.tex  # Run twice for cross-references
```

## Key Research Findings

The interpretability analysis revealed two significant findings:

### 1. Phase Transition Phenomenon

The solution path emerges through a sharp phase transition, not gradually:
- **Critical time**: t* ≈ 0.70 (consistent across 100% of samples)
- **Transition width**: Δt ≈ 0.02 (only 2% of the total process)
- **Recall jumps**: From ~0% to ~99.6% in this narrow window
- **Final IoU**: 0.9706 ± 0.071

### 2. Simultaneous Emergence

All parts of the solution path appear at the same timestep:
- Start, middle, and end segments emerge together
- Unlike traditional algorithms (BFS, DFS, A*) that explore sequentially
- Suggests holistic "insight-like" computation
- Analogous to the human "eureka moment"

## Available Figures

### Training Figures (`../paper_data/figures/`)

| Figure | File | Description |
|--------|------|-------------|
| Training loss | `training_curves/loss_curve_log.png` | Log scale loss (recommended) |
| Quality metrics | `training_curves/quality_metrics.png` | L2 distance over epochs |
| Model evolution | `epoch_comparison/evolution_grid.png` | 6 samples across epochs |
| Sample outputs | `maze_samples/epoch_N_samples.png` | Per-epoch samples |

### Interpretability Figures (`../paper_data/interpretability/`)

| Figure | File | Description |
|--------|------|-------------|
| Phase transition | `emergence/paper_figure_emergence.png` | Emergence curves with transition point |
| Emergence rate | `emergence/curves/emergence_rate.png` | Derivative showing sharp peak |
| Spatial patterns | `spatial/paper_figure_spatial.png` | Segment emergence comparison |
| Heatmaps | `spatial/heatmaps/aggregate_heatmap.png` | Aggregate spatial patterns |
| Step-by-step | `step_by_step/summary_grid_steps_20.png` | Transformation visualization |
| Animated GIFs | `step_by_step/gifs/*.gif` | Animated transformations |
| Step comparison | `step_comparison/visual_comparison.png` | Different step counts |

## Recommended Figure Selection

For the paper, consider including:

1. **Figure 1 - Model Overview**:
   - Architecture diagram (to be created with TikZ)
   - Example input/output pair

2. **Figure 2 - Training Dynamics**:
   - `training_curves/loss_curve_log.png`
   - `epoch_comparison/evolution_grid.png`

3. **Figure 3 - Phase Transition** (KEY RESULT):
   - `emergence/paper_figure_emergence.png`
   - Shows the sharp transition at t* ≈ 0.70

4. **Figure 4 - Simultaneous Emergence** (KEY RESULT):
   - `spatial/paper_figure_spatial.png`
   - Demonstrates all segments emerging together

5. **Figure 5 - Step-by-Step Transformation**:
   - `step_by_step/summary_grid_steps_20.png`
   - Visual illustration of the process

## Including Figures in LaTeX

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{../paper_data/interpretability/emergence/paper_figure_emergence.png}
    \caption{Phase transition in solution emergence. The solution path appears through a sharp transition at $t^* \approx 0.70$ with width $\Delta t \approx 0.02$.}
    \label{fig:phase-transition}
\end{figure}
```

## Research Reports

Detailed analysis available in `../paper_data/reports/`:

| Report | Content |
|--------|---------|
| `phase_transition_analysis.md` | Mathematical analysis of the phase transition |
| `spatial_emergence_analysis.md` | Spatial patterns (simultaneous emergence) |
| `philosophical_synthesis.md` | Theoretical implications |
| `training_summary.md` | Training dynamics |

## TODO

- [ ] Add actual figure includes with `\includegraphics`
- [ ] Add formal bibliography (BibTeX)
- [ ] Add architecture diagram (TikZ or external)
- [ ] Include phase transition analysis in methods/results
- [ ] Discuss simultaneous emergence in discussion section
- [ ] Add step-by-step inference visualization figure

## arXiv Submission

For arXiv submission:
1. Ensure all figures are included (copy to paper_draft/ or use relative paths)
2. Use `arxiv` package if needed for formatting
3. Include all source files in submission
4. Verify PDF renders correctly
5. Include supplementary GIFs as separate files or link to repository
