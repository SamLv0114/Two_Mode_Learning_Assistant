---
id: lora
name: Low-rank adaptation (LoRA)
triggers: lora, qlora, low-rank adaptation, low-rank adapter, low-rank adapters, low-rank update
---
## Often left open
- The rank r and the scale α; the update is multiplied by α/r.
- Which weight matrices get an adapter. The original LoRA paper adapts only the attention query and value projections.
- Whether dropout is applied on the adapter path, and whether the update is merged into the base weight after training.

## Common mistakes
- Both factors initialised randomly. LoRA initialises A randomly (Gaussian) and B to zero, so BA = 0 and the adapted layer starts exactly at the base layer.
- The scale applied as r/α instead of α/r, or left out.
- The frozen base weight W0 changed by the update step. Only A and B are trained.
- Shapes swapped. For W0 of shape (d_out, d_in), B is (d_out, r) and A is (r, d_in), so B(Ax) has the shape of W0x.

## Self-checks
- At initialisation the adapted layer's output equals the base layer's output exactly.
- After training, W0 equals a copy taken before training, while A and B differ from their saved initial values.
- The merged weight W0 + (α/r)·BA gives the same output as the unmerged forward pass.
- The sizes of A and B added up equal r·(d_in + d_out) per adapted matrix.
