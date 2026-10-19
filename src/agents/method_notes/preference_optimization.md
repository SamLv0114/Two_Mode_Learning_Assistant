---
id: preference_optimization
name: Preference optimization (RLHF reward models, DPO)
triggers: direct preference optimization, dpo, preference optimization, rlhf, reinforcement learning from human feedback, reward model, bradley-terry, preference pairs, preference data
---
## Often left open
- β, the strength of the implicit constraint toward the reference model.
- Which model is the frozen reference; usually the supervised fine-tuned model.
- Whether a response's log-probability is the sum over its tokens (as in DPO) or a length-normalised average (used by some variants).

## Common mistakes
- Log-probabilities that include the prompt tokens. Only the response tokens are scored.
- The reference model updated during training. It stays frozen.
- Chosen and rejected responses swapped, which trains toward the dispreferred answer.
- The DPO loss written without the reference terms. It is −log σ(β·[(log πθ(y_w|x) − log π_ref(y_w|x)) − (log πθ(y_l|x) − log π_ref(y_l|x))]).
- For a Bradley-Terry reward model: the loss applied to each reward separately instead of to the difference r(chosen) − r(rejected).

## Self-checks
- When the policy equals the reference, the DPO loss is log 2 ≈ 0.693 for every pair.
- After a gradient step, the chosen-minus-rejected margin increases on the training pair.
- With a fixed positive margin, a larger β gives a lower loss.
