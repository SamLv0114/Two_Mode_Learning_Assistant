---
id: mixture_of_experts
name: Mixture of experts
triggers: mixture of experts, mixture-of-experts, moe, sparse experts, expert routing, router, gating network, top-k routing, switch transformer, load balancing
---
## Often left open
- The number of experts and how many each token is sent to: top-1 as in Switch Transformer, top-2 as in GShard.
- Whether the selected gate values are renormalised to sum to 1.
- The expert capacity, and what happens to tokens over capacity (dropped, or passed on through the residual connection).

## Common mistakes
- Every expert run on every token and the outputs averaged. That is a dense ensemble, not sparse routing.
- The auxiliary load-balancing loss left out, so routing collapses onto a few experts.
- Gate values renormalised over the selected experts when the paper uses the full softmax values, or the reverse.
- The router's softmax taken over tokens instead of over experts.

## Self-checks
- Each token is processed by exactly k experts, with non-negative gate weights.
- With perfectly uniform routing, Switch Transformer's load-balancing term N·Σ f_i·P_i equals 1 before it is multiplied by its coefficient.
- Expert usage counts add up to (number of tokens) × k when no token is dropped.
