---
id: contrastive
name: Contrastive representation learning (InfoNCE, SimCLR, CLIP)
triggers: contrastive learning, contrastive loss, contrastive objective, infonce, info-nce, nt-xent, simclr, moco, contrastive language-image, contrastive pre-training, contrastive pretraining
---
## Often left open
- The temperature τ: fixed (as in SimCLR) or learned (as in CLIP).
- The batch size, which sets the number of negatives, and the projection head.

## Common mistakes
- Similarities computed on unnormalised embeddings when the method uses cosine similarity. L2-normalise first.
- The temperature applied after the softmax. The logits are divided by τ before it.
- In SimCLR's NT-Xent loss, an example's similarity with itself left among the candidates. It must be masked out.
- Positives not aligned with the labels. For a batch of pairs the positives lie on the diagonal of the similarity matrix.
- A CLIP-style loss computed in one direction only. CLIP averages the image-to-text and text-to-image cross-entropies.

## Self-checks
- If all embeddings are identical, the loss is log N for a CLIP-style batch of N pairs, and log(2N − 1) for NT-Xent.
- With well-separated positive pairs and a small τ, the loss approaches 0.
- Cosine similarities lie in [−1, 1], and each embedding's similarity with itself is 1.
