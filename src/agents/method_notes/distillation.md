---
id: distillation
name: Knowledge distillation
triggers: knowledge distillation, distillation loss, teacher model, student model, teacher-student, soft targets, soft labels, self-distillation
excludes: mean teacher, dataset distillation, data distillation
---
## Often left open
- The temperature T.
- The weight between the soft (teacher) loss and the hard-label loss, and whether the hard-label loss is used at all.

## Common mistakes
- The temperature applied to only one of the teacher and the student. Both are softened with the same T.
- The soft loss not multiplied by T². Without it the soft loss's gradients shrink by 1/T² relative to the hard loss (Hinton et al., 2015).
- KL divergence with the roles swapped. The standard soft loss is KL(teacher ‖ student), which equals the cross-entropy of the student against the teacher's softened probabilities minus a constant.
- The teacher updated during distillation. It stays fixed.

## Self-checks
- When the student's logits equal the teacher's, the soft loss is 0.
- The soft loss is never negative.
- Softened probabilities sum to 1 and become more uniform as T grows.
