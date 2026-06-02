# Darwin Family Paper — arXiv:2605.14386

## BibTeX
```
@article{kim2026darwinfamily,
  title={Darwin Family: MRI-Trust-Weighted Evolutionary Merging for Training-Free Scaling of Language-Model Reasoning},
  author={Kim, Taebong and Hong, Youngsik and Kim, Minsik and Choi, Sunyoung and Jang, Jaewon and Shin, Junghoon and Kim, Minseo},
  journal={arXiv preprint arXiv:2605.14386},
  year={2026},
  month={May}
}
```

## Headline result
**Darwin-27B-Opus: 86.9% on GPQA Diamond, #6 out of 1,252 evaluated models** — outperforming its fully trained foundation model with zero gradient updates.

## TL;DR
Training-free evolutionary merging of LLMs. Two parents (sharing a pretrained base) are weight-space recombined via a 14-dim genome. CMA-ES evolves the genome across generations. Output is one merged model, same size as a parent. No training involved.

## Three key contributions

### 1. 14-dim adaptive merge genome
`g = (γ, α_attn, α_ffn, α_emb, ρA, ρB, r0, r1, r2, r3, r4, r5, τ, λ)`
- Core (6): global ratio + 3 per-component ratios + 2 parent densities
- Block-level (6): r0..r5 — 6 independent layer-block merge ratios
- Hyper (2): τ (MRI-Trust), λ (regularization)

### 2. MRI-Trust Fusion
`r_final(T) = τ · r_MRI(T) + (1 - τ) · r_genome(T)`

`r_MRI(T) = MRI_B(T) / (MRI_A(T) + MRI_B(T))`
`MRI(T) = α · Static(T) + (1 - α) · Probe(T)`, α = 0.5

- **Static term**: normalized entropy + variance + capped ℓ2-norm (no calibration data needed)
- **Probe term**: cosine distance between reasoning-conditioned and generic activations (requires small calibration set)
- **τ**: paper converges to 0.35-0.55 empirically across scales

### 3. Architecture Mapper
`Comp(i,j) = 0.5·Type + 0.3·Dim + 0.2·Param` (β1, β2, β3 fixed)

- Constrained greedy matching under minimum compatibility threshold
- Tensors below threshold are SKIPPED (parent A's value kept)
- Enables cross-architecture merge: demonstrated Transformer + Mamba

## Merge kernel
```
θM(T) = θbase(T) + (1 - r_final(T)) · ΔA(T) + r_final(T) · ΔB(T)
```

Where ΔA = θA - θbase, ΔB = θB - θbase. Algebraically this simplifies to:
**`θM(T) = (1 - r_final(T)) · θA(T) + r_final(T) · θB(T)`** — a pure convex combination.

## CMA-ES evolution loop
1. Sample N candidate genomes from multivariate Gaussian
2. For each: build child (full merge) → score on benchmark → fitness
3. Update Gaussian distribution toward higher-fitness candidates
4. Repeat 20-50 generations
5. Keep best-genome child

Population: 20-50. Recombination weights weighted by fitness ranking.

## Recursive evolution ("Family")
Take best child, merge with third parent → grandchild. Merge grandchild with fourth → great-grandchild. Each generation can introduce new capabilities. Paper's 27B flagship was multi-generation.

## Constraints (paper is strict)
- 2 parents per merge (not 3)
- 1 child output (single dense model, NOT MoE)
- Same size as a parent
- Training-free (no gradient updates)
- Single tensor, per-tensor blending

## Why this matters
Reasoning emerges during pretraining and is invariant under post-training. So you can recombine frozen pretrained models' reasoning abilities without expensive RLHF. The paper proves this with the 86.9% GPQA result.

## What the paper doesn't cover
- 3+ parent merges (paper is strictly 2-parent; multi-generation is the workaround)
- MoE output (the child is one dense model)
- Modality heads (text body only — vision/audio/etc. attach separately)
- Calibration data generation (Probe term needs it; the paper doesn't specify how to build it)
