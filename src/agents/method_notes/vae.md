---
id: vae
name: Variational autoencoders
triggers: variational autoencoder, variational auto-encoder, vae, reparameterization trick, reparameterisation trick, beta-vae
---
## Often left open
- The latent size.
- The decoder's likelihood: Gaussian (a squared-error reconstruction term) or Bernoulli (binary cross-entropy).
- The KL weight β (1 in a standard VAE).

## Common mistakes
- z sampled directly from N(μ, σ²), which blocks gradients. Use z = μ + σ·ε with ε ~ N(0, I).
- The encoder's second output used as σ in one place and as log σ² in another.
- The KL term with a flipped sign or a dropped constant. KL(q ‖ N(0, I)) = −½·Σ(1 + log σ² − μ² − σ²).
- Reconstruction summed over dimensions but KL averaged (or the reverse), which silently re-weights the two terms.

## Self-checks
- The KL term is 0 when μ = 0 and log σ² = 0, and positive otherwise.
- Hand-derived gradients with respect to μ and log σ² match finite differences.
- The loss decreases over training on toy data.
