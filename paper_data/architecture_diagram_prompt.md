# TACIT Architecture Diagram - AI Generation Prompt


---

## Main Prompt

Create a clean, professional technical diagram showing a neural network architecture for an academic machine learning paper. White background, minimalist style similar to the original "Attention Is All You Need" transformer diagram.

**Title at top**: "TACIT Architecture"

**Vertical flow from top to bottom with these components**:

### Top Section - Input
- Two small images side by side: a maze (black and white grid with green dots) and a clock/dial showing "t"
- Label: "Input: Maze Image (64×64×3) + Timestep t ∈ [0,1]"

### Second Row - Patch Embedding
- Rectangle labeled "Patch Embedding"
- Inside: "Conv2D (8×8, stride=8)"
- Arrow pointing down with annotation "64 patches × 384-dim"

### Third Row - Split into Two Branches
**Left branch**: Box labeled "2D Sinusoidal Position Embedding" with "(64 × 384)" below
**Right branch**: Box labeled "Timestep Embedding" with "Sinusoidal → MLP → 384-dim" below

Both branches have arrows converging back together with a "+" symbol

### Fourth Row - Main Transformer Stack
- Large rounded rectangle containing:
- Header: "Transformer Blocks ×8"
- Inside, show one block detail:
  - "Adaptive LayerNorm" (with small arrow from right showing timestep conditioning)
  - "Multi-Head Self-Attention (6 heads)"
  - Curved residual connection arrow
  - "Adaptive LayerNorm"
  - "MLP (384→1536→384)"
  - Curved residual connection arrow
- Show "×8" prominently to indicate repetition

### Fifth Row - Final Layer
- Rectangle labeled "Final Layer"
- Inside: "AdaLN → Linear → Reshape"
- Arrow from right showing timestep conditioning

### Bottom Section - Output
- Small image showing a maze with red solution path
- Label: "Output: Predicted Velocity v(x,t)"
- Subtitle: "(64×64×3)"

---

## Visual Style Requirements

1. **Color scheme**:
   - Main flow arrows: Dark blue (#2563EB)
   - Timestep conditioning arrows: Orange (#F97316)
   - Boxes: Light gray fill (#F3F4F6) with dark gray border (#374151)
   - Text: Black (#111827)
   - Residual connections: Dashed gray lines

2. **Typography**:
   - Clean sans-serif font (like Helvetica or Arial)
   - Component names in bold
   - Dimensions/specs in regular weight

3. **Layout**:
   - Centered, vertical flow
   - Consistent spacing between components
   - Clear visual hierarchy

4. **Special elements**:
   - Small maze images at input/output showing the transformation
   - The "×8" for transformer blocks should be prominent
   - Timestep conditioning should visually branch from the right side into each conditioned component

---

## Alternative Text-Based Description (for manual creation)

```
                    ┌─────────────────────────────────┐
                    │  INPUT                          │
                    │  Maze (64×64×3) + Timestep t    │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │  PATCH EMBEDDING                │
                    │  Conv2D(8×8) → 64 patches       │
                    └────────────────┬────────────────┘
                                     │
                         ┌───────────┴───────────┐
                         ▼                       ▼
              ┌──────────────────┐   ┌──────────────────┐
              │  Position Embed  │   │  Timestep Embed  │
              │  (2D Sinusoidal) │   │  (MLP: 256→384)  │
              └────────┬─────────┘   └────────┬─────────┘
                       │                      │
                       └──────────┬───────────┘
                                  │ (+)
                                  ▼
              ┌───────────────────────────────────────────┐
              │                                           │
              │  ╔═══════════════════════════════════╗   │
              │  ║  TRANSFORMER BLOCK                ║   │
              │  ║  ┌─────────────────────────────┐  ║   │
              │  ║  │ Adaptive LayerNorm ◄─── t   │  ║   │
              │  ║  │ Multi-Head Attention (6h)   │  ║   │
              │  ║  │ + Residual                  │  ║   │
              │  ║  │ Adaptive LayerNorm ◄─── t   │  ║   │
              │  ║  │ MLP (384→1536→384)          │  ║   │
              │  ║  │ + Residual                  │  ║   │
              │  ║  └─────────────────────────────┘  ║   │
              │  ╚═══════════════════════════════════╝   │
              │                                     ×8   │
              └────────────────────┬──────────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────────┐
              │  FINAL LAYER                            │
              │  AdaLN ◄─── t  →  Linear  →  Reshape   │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │  OUTPUT                         │
                    │  Velocity Field v(x,t)          │
                    │  (64×64×3)                      │
                    └─────────────────────────────────┘
```

---

## Key Points to Emphasize

1. The timestep `t` conditions multiple components (shown with orange arrows)
2. The 8 transformer blocks are the computational core
3. Adaptive LayerNorm (adaLN) is the conditioning mechanism
4. The output is a velocity field, not the final solution directly
5. This is a Diffusion Transformer (DiT) adapted for image-to-image transformation

---

## Dimensions Reference

| Component | Input Dim | Output Dim |
|-----------|-----------|------------|
| Input Image | - | 64×64×3 |
| Patch Embed | 64×64×3 | 64×384 |
| Pos Embed | - | 64×384 |
| Timestep Embed | 1 | 384 |
| Transformer | 64×384 | 64×384 |
| Final Layer | 64×384 | 64×64×3 |

**Total Parameters**: ~20M
