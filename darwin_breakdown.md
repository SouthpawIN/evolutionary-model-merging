# Darwin Family — Evolutionary LLM Merging

*A breakdown of arXiv:2605.14386 (Kim et al., May 2026) — how the genetic
algorithm recombines frozen LLMs into stronger children, and how we're
applying it to OmniSenter.*

![Robot chimp with DNA helix and neural HUD — our visual for the
Darwin/evolution/AI project theme]
*Visual generated for the OmniSenter project — the robotic primate
contemplating its own genetic code and technological augmentation.*

---

## 1. What the paper claims

The Darwin Family paper asks a simple question with a striking answer:

> Can you recombine two pretrained LLMs — **without any training** — into a
> single model that's *better than either parent*?

Their answer: **yes.** Their headline result is **Darwin-27B-Opus at 86.9%
on GPQA Diamond** (a hard graduate-level reasoning benchmark), ranking #6
out of 1,252 evaluated models. That beats the fully-trained foundation
model. No gradient updates. No RLHF. No distillation. Just weight-space
recombination.

The mechanism is a **genetic algorithm over a 14-dimensional merge
genome**, with CMA-ES as the optimizer and MRI-Trust Fusion as the
per-tensor mixing rule. The child model is one dense network the same
size as a single parent.

## 2. The 14-dimensional genome

The genome is a vector that controls *how* two parents are blended:

```
g = (γ, α_attn, α_ffn, α_emb, ρA, ρB, r0, r1, r2, r3, r4, r5, τ, λ)
```

| Group | Parameters | Role |
|---|---|---|
| **Core (6)** | `γ, α_attn, α_ffn, α_emb, ρA, ρB` | Global ratio, per-component ratios (attention vs FFN vs embedding), parent densities |
| **Block (6)** | `r0..r5` | Six independent layer-block merge ratios |
| **Hyper (2)** | `τ, λ` | MRI-Trust coefficient, regularization |

Different genome values → different child. The genome is what "evolves."

## 3. MRI-Trust Fusion (per-tensor)

For every weight tensor `T` in the model, the merge kernel is:

```
θM(T) = (1 - r_final(T)) · θA(T) + r_final(T) · θB(T)
```

`r_final` is itself a blend of two signals:

```
r_final(T) = τ · r_MRI(T) + (1 - τ) · r_genome(T)
```

- **`r_MRI(T)`** — from a diagnostic. `MRI(T) = α·Static(T) + (1-α)·Probe(T)`,
  with `α = 0.5` paper-fixed.
  - **Static**: normalized entropy + variance + capped ℓ2-norm of the
    tensor (no calibration data needed)
  - **Probe**: cosine distance between reasoning-conditioned and generic
    activations (requires a small calibration set)
  - The diagnostic answers: "which parent's weights are more important
    for this tensor?"
- **`r_genome(T)`** — derived from the genome's ρ values. The
  evolutionary prior.
- **`τ`** — single trust parameter. Paper converges to **0.35-0.55**
  empirically across model scales. `τ=0` is pure evolution; `τ=1` is
  pure diagnostic.

The two signals correct each other: the diagnostic catches tensors
where one parent is clearly better, while the genome provides a global
tendency that smooths over diagnostic noise.

## 4. Architecture Mapper (cross-architecture)

For parents with **different architectures** (e.g., Transformer + Mamba,
or Qwen2 + Qwen3 with different hidden dims), the Mapper finds layer
correspondences:

```
Comp(i, j) = 0.5 · Type(i,j) + 0.3 · Dim(i,j) + 0.2 · Param(i,j)
```

- **Type**: do they play the same functional role (e.g., both are
  attention layers)?
- **Dim**: do dimensions match?
- **Param**: are parameter shapes similar?

Tensors below a compatibility threshold are **skipped** — parent A's
value is kept. The paper demonstrated a Transformer + Mamba
cross-architecture merge via this mechanism.

## 5. CMA-ES evolution loop

This is the "Darwin" part. CMA-ES (Covariance Matrix Adaptation
Evolution Strategy) is a population-based optimizer that searches the
genome space:

1. **Sample N candidate genomes** from a multivariate Gaussian (start
   broad, narrow over generations)
2. **Build N children** — one full merge per candidate genome
3. **Score each child** on a real benchmark → fitness
4. **Update the Gaussian** — shift it toward higher-fitness candidates
5. **Repeat** for 20-50 generations

After evolution, keep the best-genome child. The paper showed scores
climb generation over generation as CMA-ES finds better merge weights.

Population: 20-50. Each generation's `O(N × merge_cost + N ×
benchmark_cost)` is the main compute expense.

## 6. The "Family" — recursive evolution

The trick that makes the *family* work: take an evolved child, merge
it with a third parent → **grandchild**. Merge the grandchild with a
fourth → **great-grandchild**. Each generation can introduce new
capabilities. The paper's 27B flagship was evolved across multiple
generations this way.

This is how OmniSenter builds out: OmniLance 6B is the first
generation, OmniStep 6B is a parallel second-generation merge with
different parents, and OmniSenter 9A3B composes them as experts.

## 7. Constraints (the paper is strict)

1. **2 parents per merge** (not 3)
2. **1 child output** — single dense model, NOT MoE
3. **Same size as a parent** — interpolation, not stacking
4. **Training-free** — no gradient updates, no fine-tuning in the
   merge itself
5. **CMA-ES, not random** — the genome search is principled

## 8. What this means for OmniSenter

We're applying this paper to the OmniSenter project. The architecture
is a **two-level MoE hierarchy** where the paper-exact weight merge
happens within each 6B sub-model, and routing happens between them:

| Model | Total | Active | Composition |
|---|---|---|---|
| **OmniLance 6B** | 6B | 3B | Paper-exact Darwin merge: Omni 3B + Lance 3B → 3B child, MoE-routed with parent copies |
| **OmniStep 6B** | 6B | 3B | Paper-exact Darwin merge: Omni 3B + ACE-Step 3B → 3B child, MoE-routed with parent copies |
| **OmniSenter 12A6B** | 12B | 6B | Hierarchical MoE: routes between OmniLance 6B and OmniStep 6B sub-models (one 6B active per token) |

The naming convention `XAYB` = X total B, Y active B per token.
The 2-parent merges are named `6B` (6B total = 2 parents of 3B each,
3B active = one parent routed at the sub-model level). The combined
OmniSenter is `12A6B` (12B total = two 6B sub-models, 6B active = one
sub-model routed at the top level).

Within each sub-model, the paper-exact Darwin 2-parent weight merge
produces the 3B child. The sub-model also retains copies of both
parents so the MoE router can choose between the merged child and
either parent at inference time. This gives the system three
reasoning modes per sub-model (merged / parent A / parent B) plus
the cross-sub-model routing at the OmniSenter level.

**Modality heads** are attached as **separate components** after
the text merges work:
- Vision in: Omni's NaViT (or Lance's Qwen2.5-VL vision)
- Audio in: Omni's Whisper (Omni only)
- Speech out: Omni's talker + token2wav
- Image/video gen: Lance's DiT (separate inference path)
- Music gen: ACE-Step DiT + UMT5 (separate inference path)

## 9. Paper-exact pitfalls (compiled from this project)

These are the bugs we hit, so you don't:

1. **`lm_head` extraction**: parents wrap `lm_head` outside the
   `model.*` prefix. Omni uses `thinker.lm_head.weight`, Lance uses
   `language_model.lm_head.weight`. Vanilla Qwen2 uses `lm_head.weight`.
   Standard `model.*` extraction patterns miss all of them.
2. **No random projection for cross-arch**: Architecture Mapper SKIPS
   dim-mismatched tensors. A random linear projection introduces noise
   that destroys the merge.
3. **No extra scaling**: the `(1 - gamma + gamma · alpha)` factor that
   shows up in some merge scripts is NOT in the paper. γ, α are genome
   values that feed into `r_final` — not post-merge scaling.
4. **Multi-way merges must sum to 1.0**: the convex combination
   constraint. A 3-way formula like `(1-r)·ρ_A·A + (1-r)·ρ_B·B + r·ρ_C·C`
   does NOT sum to 1.
5. **Tied embeddings**: Omni-derived 3B models use
   `tie_word_embeddings: true`. If you write a separate `lm_head` in
   the merged checkpoint, llama.cpp behavior is ambiguous.
6. **GGUF conversion**: Lance and ACE-Step have non-text-LLM tensors
   (`_moe_gen`, `q_norm`, `k_norm`, `latent_pos_embed`, `vae2llm`,
   `llm2vae`, `time_embedder`) that llama.cpp's converter doesn't
   know. Filter them out before conversion.
7. **Tokenizer files**: must be copied from the source parent (not in
   safetensors).

## 10. The full `evolutionary-model-merging` skill

The Darwin Family methodology is packaged as a reusable skill at
**`/home/sovthpaw/.hermes/skills/mlops/evolutionary-model-merging/`**
and mirrored to **`github.com/SouthpawIN/evolutionary-model-merging`**.

The skill includes:

- **`SKILL.md`** — the procedure, formulas, pitfalls, and workflow
- **`references/darwin-paper-summary.md`** — paper notes, BibTeX,
  headline result
- **`references/parent-model-architectures.md`** — Qwen2.5-Omni,
  Qwen2.5-VL, Lance, ACE-Step specs and tensor names
- **`references/merge-pitfalls.md`** — extended bug list with
  reproduction details
- **`scripts/paper_exact_2parent_merge.py`** — clean paper-exact
  2-parent reference implementation
- **`scripts/cma_es_evolution.py`** — CMA-ES evolution loop (the
  paper's actual method)
- **`scripts/filter_for_gguf.py`** — filter non-text-LLM tensors
  before GGUF conversion
- **`scripts/real_benchmark.py`** — 10-question real-inference
  benchmark template (against llama-server :port)

To install the skill in another Hermes Agent instance:

```bash
gh repo clone SouthpawIN/evolutionary-model-merging ~/.hermes/skills/mlops/evolutionary-model-merging
```

## 11. 1M context length (YaRN)

Goal: extend the merged OmniLance 6B from 32k → 1M tokens via YaRN
scaling on RoPE. YaRN (Yet another RoPE extensioN) handles long-context
extension by:

1. Changing `rope_theta` to a larger value
2. Adding `rope_scaling` config with `type: yarn` and a `factor`
3. Setting `max_position_embeddings` to 1048576
4. Verifying with passkey retrieval at multiple context lengths

This is a config-level change, no retraining needed for inference. For
training, YaRN has a 400-step warmup procedure that's documented in
the original paper.

## 12. What we're building this weekend

1. **OmniLance 6B** with CMA-ES evolution (paper-faithful, 2-4 hours
   compute, target 4-6/10 on the reasoning benchmark)
2. **OmniStep 6B** with CMA-ES (Omni + ACE-Step via Mapper)
3. **OmniSenter 12A6B** as the two-level MoE composition (routes
   between OmniLance 6B and OmniStep 6B sub-models)
4. **YaRN 1M context** on the best child
5. **Recursive family evolution** — child N becomes a parent for
   child N+1

## 13. Model sizes at a glance

| Component | Total params | Active per token |
|---|---|---|
| Omni-3B (parent) | 3B | 3B |
| Lance 3B (parent) | 3B | 3B |
| ACE-Step LM (parent) | 4B | 4B |
| **OmniLance 6B** (sub-model) | 6B | 3B |
| **OmniStep 6B** (sub-model) | 6B | 3B |
| **OmniSenter 12A6B** (composed) | 12B | 6B |

## References

- **Paper**: Kim, T. et al. (2026). *Darwin Family: MRI-Trust-Weighted
  Evolutionary Merging for Training-Free Scaling of Language-Model
  Reasoning.* arXiv:2605.14386.
- **PDF**: https://arxiv.org/pdf/2605.14386
- **GitHub**: https://github.com/SouthpawIN/evolutionary-model-merging
- **Project directory**: `/home/sovthpaw/Models/senter-omni/omni-sender/`
- **Source models**:
  - Qwen2.5-Omni-3B: `model_type: qwen2_5_omni`, hidden=2048, 36L
  - Lance 3B: `model_type: qwen2_5_vl`, hidden=2048, 36L
  - ACE-Step LM: Qwen3-4B, hidden=2560, 36L, vocab=217204

---

*Prepared for the Senter Dev Discord server. This is an active research
project — the methodology is being replicated exactly per the paper, then
applied to the OmniSenter multimodal intelligence.*
