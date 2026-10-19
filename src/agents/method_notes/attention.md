---
id: attention
name: Attention and Transformer layers
triggers: attention, self-attention, scaled dot-product attention, multi-head attention, attention mechanism, causal mask, causal attention, positional encoding, transformer, vision transformer
generic: true
---
## Often left open
- The number of heads h and the head size; d_k = d_model / h is the usual split.
- Whether layer normalisation comes before each sub-layer (pre-LN) or after the residual addition (post-LN, as in the original Transformer).
- The positional encoding: sinusoidal, learned, or rotary.

## Common mistakes
- Scores scaled by sqrt(d_model) instead of sqrt(d_k), the per-head key dimension.
- The softmax taken over the query axis instead of the key axis. Each query's weights over the keys must sum to 1.
- The mask applied after the softmax, or as 0 instead of a large negative value. Masked scores are set to −inf (or a large negative number) before the softmax.
- A causal mask that lets position t see later positions. Position t may attend only to positions ≤ t.
- A softmax computed without subtracting the row maximum, which overflows for large scores.
- Heads concatenated without the final output projection.

## Self-checks
- Every row of the attention weights sums to 1, and masked positions get weight 0.
- With a causal mask, changing a later token leaves every earlier output unchanged.
- If all keys are identical, attention is uniform and each output is the mean of the values.
- The output has one row per query and the model width as its last dimension.
