---
id: policy_gradient
name: Policy-gradient reinforcement learning (REINFORCE, PPO, GRPO)
triggers: policy gradient, policy optimization, ppo, proximal policy optimization, reinforce, actor-critic, generalized advantage estimation, gae, grpo, group relative policy optimization, rlvr, verifiable rewards
---
## Often left open
- The discount γ and GAE λ (PPO used γ = 0.99 and λ = 0.95), the clip range ε (0.2 in PPO), and the number of epochs per batch.
- Whether advantages are normalised per batch.
- For GRPO-style methods: the group size, how rewards are normalised within a group, and the weight of the KL penalty to the reference policy.

## Common mistakes
- The objective optimised with the wrong sign. A loss to minimise is the negative of the surrogate objective.
- The probability ratio computed from log-probabilities of a different action, or of the current policy for both terms. The old log-probabilities are fixed numbers recorded when the data was collected.
- Clipping written as clip(r, 1−ε, 1+ε)·A alone instead of min(r·A, clip(r, 1−ε, 1+ε)·A).
- Returns or advantages that bootstrap across an episode boundary. The value after a terminal state is 0.
- A baseline that depends on the action taken, which biases the gradient.

## Self-checks
- In the first update of each batch the ratio is exactly 1 for every sample.
- For a hand-written reward sequence, the discounted returns match a direct calculation.
- When the ratio is beyond the clip range in the direction the advantage favours, that sample's gradient is 0.
- On a toy bandit or chain task, the average reward increases over updates.
