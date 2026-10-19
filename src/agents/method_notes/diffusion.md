---
id: diffusion
name: Denoising diffusion models
triggers: diffusion model, denoising diffusion, ddpm, ddim, score-based, score matching, noise schedule, reverse diffusion, latent diffusion, diffusion-based, diffusion transformer, diffusion policy
---
## Often left open
- The number of steps T and the noise schedule β_1..β_T. DDPM used a linear schedule from 1e-4 to 0.02 over T = 1000; a cosine schedule is a common alternative.
- What the network predicts: the added noise ε, the clean sample x0, or v.
- The sampling variance σ_t²: β_t or the posterior variance β̃_t.

## Common mistakes
- α_t (= 1 − β_t) confused with ᾱ_t, the product of α_s for s ≤ t. The forward process uses ᾱ_t: x_t = sqrt(ᾱ_t)·x0 + sqrt(1 − ᾱ_t)·ε.
- Off-by-one timesteps. The schedule arrays and the sampled t must use the same convention (1..T or 0..T−1).
- Noise added at the final sampling step. The last step returns the mean without noise.
- The wrong training target. When the network predicts ε, the target is the noise that was added, not x0 or x_t.
- One t shared by the whole batch instead of an independent t per example.

## Self-checks
- ᾱ_T is close to 0, so x_T has mean close to 0 and standard deviation close to 1.
- Inverting the forward process with the true noise recovers x0: (x_t − sqrt(1 − ᾱ_t)·ε) / sqrt(ᾱ_t) equals x0.
- ᾱ_t decreases strictly as t increases.
