---
name: evolutionary-model-merging
description: Training-free weight-space recombination of LLMs using evolutionary algorithms. Implements the Darwin Family paper (arXiv:2605.14386) methodology — MRI-Trust Fusion, Architecture Mapper, CMA-ES genome search. Use when merging 2+ parent LLMs into a single dense child model without gradient updates, when the user asks to "test the paper" or "follow the Darwin methodology", or when working on OmniSenter / OmniLance / OmniStep style multi-parent merges.
version: 1
created: 2026-06-02
---

# Evolutionary Model Merging (Darwin Family methodology)

Paper-faithful training-free LLM merging. When the user invokes this skill, they are **testing a paper** — do NOT modify, improvise, or "improve" the methodology. Replicate the paper exactly, then report honest results. A negative result ("the paper's formula doesn't transfer to these parents without CMA-ES evolution") is a valid finding.

## When to use

- Merging 2+ parent LLMs into a single dense child (same size as a parent, NOT MoE stacking)
- Training-free, weight-space only (no gradient updates)
- The user said "follow the paper", "do not modify the process", or named a specific paper
- The project uses names like OmniSenter / OmniLance / OmniStep with "6A3B" / "6B" / "9A3B" suffixes

## When NOT to use

- The user wants MoE stacking (this paper produces a single dense child)
- The user wants to add modality heads — attach those as separate components AFTER the text merge works
- The user wants fine-tuning / RLHF / DPO — those are separate processes

## Core paper-exact formulas (do not modify)

```
Merge kernel:   θM(T) = (1 - r_final(T)) · θA(T) + r_final(T) · θB(T)
MRI-Trust:      r_final(T) = τ · r_MRI(T) + (1 - τ) · r_genome(T)
MRI ratio:      r_MRI(T)  = MRI_B(T) / (MRI_A(T) + MRI_B(T))
MRI composite:  MRI(T)    = α · Static(T) + (1 - α) · Probe(T),  α = 0.5 (paper-fixed)
Genome:         14-dim = (γ, α_attn, α_ffn, α_emb, ρA, ρB, r0, r1, r2, r3, r4, r5, τ, λ)
CMA-ES:         evolves the 14-dim genome over 20-50 generations, N=20-50 population
```

The merge is a **pure convex combination**. No extra scaling, no random projection, no invented terms.

## Critical pitfalls (compiled from this session)

### 1. lm_head extraction
Parents wrap `lm_head` OUTSIDE the `model.*` prefix:
- Omni-3B: `thinker.lm_head.weight` (NOT `thinker.model.lm_head.*`)
- Lance 3B: `language_model.lm_head.weight` (NOT `language_model.model.lm_head.*`)
- Vanilla Qwen2: `lm_head.weight`

**Extract explicitly** — the standard `model.*` patterns miss it. If you strip lm_head, the merged model is mathematically broken.

### 2. No random projection for cross-architecture
When parent dims don't match (e.g., Omni 2048 vs ACE-Step 2560), the paper's Architecture Mapper **SKIPS** dim-mismatched tensors (Comp(i,j) below threshold). Do NOT use a random linear projection to "align" — that introduces noise that destroys the merge.

### 3. No extra scaling
A common bug: `m = blend * (1 - gamma + gamma * alpha)`. The `(1 - gamma + gamma * alpha)` factor is NOT in the paper. γ, α are genome values that feed into r_final — they are not post-merge scaling. The correct formula is just `(1-r)·A + r·B`.

### 4. Multi-way merges must sum to 1.0
For 3+ parents extending the paper, the convex combination must satisfy Σ wᵢ = 1.0. A common bug: `(1-r)·ρ_A·A + (1-r)·ρ_B·B + r·ρ_C·C` does NOT sum to 1. The correct form: `(1-r)·((ρ_A·A + ρ_B·B)/(ρ_A+ρ_B)) + r·C`.

### 5. Tied embeddings
Omni-derived 3B models use `tie_word_embeddings: true` — embed_tokens doubles as lm_head. If you write a separate lm_head in the merged checkpoint, llama.cpp behavior with `tie_word_embeddings: true` is ambiguous. Match the parent's config: Qwen2-Omni = `true`, vanilla Qwen2 = depends on size.

### 6. GGUF conversion filter patterns
Lance 3B and ACE-Step have non-text-LLM tensors that llama.cpp's GGUF converter doesn't know:
- `_moe_gen` (Lance's per-layer MoE generation expert twins)
- `q_norm`, `k_norm` (Qwen3 per-head norms — not in Qwen2)
- `latent_pos_embed`, `vae2llm`, `llm2vae`, `time_embedder` (Lance's VAE/diffusion adapters)

**Filter these out** before GGUF conversion. See `scripts/filter_for_gguf.py`.

### 7. Tokenizer files must be copied
The GGUF converter reads `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, `added_tokens.json` from the model directory. They are NOT in safetensors — copy them from the source parent.

## Workflow (paper-faithful)

1. **Verify parent compatibility**
   - Read configs: `architectures`, `model_type`, `hidden_size`, `num_hidden_layers`, `vocab_size`, `intermediate_size`
   - Same dims → 100% Architecture Mapper match, easy merge
   - Different dims → Mapper skips mismatched, use parent A as base

2. **Establish a baseline**
   - Benchmark the **original unmerged parent** with your eval pipeline BEFORE merging
   - Our results: original Omni-3B Q4_K_M = 9/10 on a 10-question reasoning benchmark
   - If the pipeline doesn't reach ~9/10 on the baseline, fix the pipeline first — the merge will only degrade from there

3. **Extract text bodies**
   - Strip encoders (vision, audio), strip decoders (talker, diffusion heads)
   - Keep the text LLM core (layers.*, embed_tokens, layernorms)
   - **Keep lm_head** (see pitfall #1)

4. **Implement the paper's exact formulas** (no inventions)
   - MRI-Trust Fusion with τ
   - MRI = α·Static + (1-α)·Probe, α = 0.5
   - Per-tensor r_final from MRI-Trust
   - Convex combination merge kernel

5. **CMA-ES evolution (the actual paper)**
   - Initialize genome from paper's recommended starting values
   - Population 20-50, generations 20-50
   - Each generation: sample candidates → merge → score → update
   - **A single fixed-genome merge is NOT the paper** — naive merges degrade parent performance

6. **Convert to GGUF, serve, benchmark**
   - Filter non-text-LLM tensors
   - Copy tokenizer files
   - Set `tie_word_embeddings` correctly
   - Serve with llama-server, run real benchmark

7. **Honest reporting**
   - If the merge scores below the original parent, that's a valid finding
   - Explain WHY (no evolution, no calibration data, parents too dissimilar, etc.)
   - Do NOT pretend a broken model works

## Reference architecture facts (OmniSenter project)

- **Qwen2.5-Omni-3B**: `model_type: qwen2_5_omni`, text body hidden=2048, 36L, vocab=151936, intermediate=11008
- **Lance 3B** (bytedance-research): `model_type: qwen2_5_vl`, text body hidden=2048, 36L, vocab=151936, intermediate=11008 — same shape as Omni text body
- **ACE-Step LM**: Qwen3-4B, hidden=2560, 36L, vocab=217204 — different shape, Architecture Mapper skips
- **Omni text body 1:1 mergeable with Lance text body** — same dims, same layer count
- **Omni is NOT Qwen2.5-VL** — it's a separate model with its own vision (NaViT) + audio (Whisper) + speech output (talker + token2wav) stack

## User preference (this project)

When the user says "follow the paper", "do not modify the process", or invokes this skill, they mean it **literally**. They are testing a paper as part of a research project. Replicate the methodology exactly, then report honest results — even if results are negative. Do not:
- Add "improvements" or shortcuts
- Use random projections to "fix" cross-arch merges
- Skip CMA-ES in favor of a fixed genome
- Pretend a broken model works
- Add a "scaling factor" that's not in the paper

## Support files

- `references/darwin-paper-summary.md` — paper notes (formulas, headline results, key terms)
- `references/parent-model-architectures.md` — Qwen2.5-Omni, Qwen2.5-VL, Lance, ACE-Step specs
- `references/merge-pitfalls.md` — extended bug list with reproduction details
- `scripts/paper_exact_2parent_merge.py` — clean paper-exact 2-parent reference implementation
- `scripts/filter_for_gguf.py` — filter non-text-LLM tensors before GGUF conversion
- `scripts/real_benchmark.py` — 10-question real-inference benchmark template (vs llama-server :port)
- `scripts/cma_es_evolution.py` — CMA-ES evolution loop (the paper's actual method)
