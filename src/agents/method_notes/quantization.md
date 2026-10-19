---
id: quantization
name: Quantization
triggers: quantization, quantisation, int8, int4, 4-bit, 8-bit, low-bit, post-training quantization, quantization-aware training, straight-through estimator, weight quantization
excludes: vector quantization, vector-quantized, vq-vae, vqvae, residual quantization, product quantization, quantum
---
## Often left open
- The bit width, and whether quantization is symmetric or asymmetric (with a zero-point).
- Per-tensor or per-channel scales.
- How the clipping range is chosen: min/max, a percentile, or learned.

## Common mistakes
- Off-by-one integer ranges. Signed 8-bit is −128..127, and symmetric schemes often use −127..127 so that zero maps to zero exactly.
- The zero-point left out when dequantizing. x̂ = s·(q − z).
- Per-channel scales computed over the wrong axis. For a weight matrix they belong to its output channels.
- In quantization-aware training, rounding treated as having zero gradient. The straight-through estimator passes the gradient through unchanged inside the clipping range and blocks it outside.

## Self-checks
- The quantize-then-dequantize error is at most s/2 for values inside the range.
- Values outside the range are clipped to its ends.
- Zero is represented exactly when the scheme is meant to guarantee it.
