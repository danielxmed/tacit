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

## Figures

Figures from training are available in `../paper_data/figures/`:

### Training Curves
- `training_curves/loss_curve.png` - Linear scale loss
- `training_curves/loss_curve_log.png` - Log scale loss (recommended)
- `training_curves/quality_metrics.png` - L2 distance over epochs

### Epoch Comparisons
- `epoch_comparison/evolution_grid.png` - 6 samples across epochs
- `epoch_comparison/early_vs_late.png` - Epoch 10 vs 100

### Sample Outputs
- `maze_samples/epoch_N_samples.png` - Samples at each checkpoint

## TODO

- [ ] Add actual figure includes with `\includegraphics`
- [ ] Add formal bibliography (BibTeX)
- [ ] Add architecture diagram (TikZ or external)
- [ ] Run additional experiments as needed
- [ ] Add step-by-step inference visualization figure

## arXiv Submission

For arXiv submission:
1. Ensure all figures are included
2. Use `arxiv` package if needed for formatting
3. Include all source files in submission
4. Verify PDF renders correctly
