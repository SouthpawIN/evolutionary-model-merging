# Merge Pitfalls — extended bug list with reproduction

This is a session-specific log of bugs we hit and how to avoid them. Use as a checklist before each merge.

## Pitfall 1: lm_head dropped during extraction
**Symptom**: Merged model produces `???` or empty string or repeats one token.
**Cause**: Extractor's prefix patterns (`model.*`, `thinker.model.*`) miss `lm_head` which lives at `thinker.lm_head.weight` (Omni) or `language_model.lm_head.weight` (Lance).
**Detection**: After extract, count tensors. Omni should have ~434 text tensors INCLUDING `lm_head.weight`. If `lm_head.weight` is missing, the merge will be broken.
**Fix**: Add explicit extraction:
```python
elif k in ("thinker.lm_head.weight", "language_model.lm_head.weight", "lm_head.weight"):
    o["lm_head.weight"] = v
```

## Pitfall 2: Random projection for cross-architecture
**Symptom**: 3-parent merge outputs garbage. 2-parent merge works.
**Cause**: Used `torch.randn(2048, 2560) / sqrt(2560)` to project ACE-Step's 2560-dim weights to 2048-dim. Random projection is just noise.
**Detection**: Check if the merge produced tensors with std significantly different from the parents. If yes, the projection corrupted the weights.
**Fix**: Do NOT project. The paper's Architecture Mapper SKIPS dim-mismatched tensors. Let parent A's value be kept for those tensors. ACE-Step will only contribute on shape-matched tensors (mostly biases, 1D norms).

## Pitfall 3: Extra scaling factor
**Symptom**: Model produces low-quality but coherent text. Score 2/10 when 9/10 expected.
**Cause**: `m = (rho_o*A + rho_l*B) * (1 - gamma + gamma * alpha)`. The `(1 - gamma + gamma * alpha)` factor is NOT in the paper. γ and α are genome values that feed into r_final, not post-merge scaling.
**Detection**: Compare merged norm weight std to parent std. If merged is ~65% of parent, the bug is active.
**Fix**: Use the paper-exact `m = (1 - r) * A + r * B`. No post-multiplication.

## Pitfall 4: 3-way merge not summing to 1.0
**Symptom**: 3-parent merge produces weights with magnitude ~0.5 of parents. Model is broken.
**Cause**: `m = (1-r)·ρ_A·A + (1-r)·ρ_B·B + r·ρ_C·C`. With ρ_A=ρ_B=0.45, ρ_C=0.10, r=0.5: sum = 0.225+0.225+0.05 = 0.5. Weights are scaled to 50%.
**Fix**: For 3 parents, normalize: `m = (1-r)·((ρ_A·A + ρ_B·B)/(ρ_A+ρ_B)) + r·C`. Sum = 1.0.

## Pitfall 5: tie_word_embeddings mismatch
**Symptom**: Model produces wrong tokens (e.g., "The" repeated). llama.cpp loads the safetensors lm_head but the config says tie.
**Cause**: Qwen2-Omni config has `tie_word_embeddings: false` (because Omni's text body has a separate lm_head). But after our merge, if the merged model uses `tie_word_embeddings: true` (Qwen2 vanilla default), the model skips our averaged lm_head and uses embed_tokens directly.
**Detection**: If the model produces wrong tokens but the stats look fine, this is likely the issue.
**Fix**: Check the original parent's `tie_word_embeddings` setting. Omni text body = `false` (in `thinker_config.text_config.tie_word_embeddings`). Use that.

## Pitfall 6: GGUF converter doesn't know Lance's _moe_gen / Qwen3's q_norm
**Symptom**: `ValueError: Can not map tensor 'layers.0.input_layernorm_moe_gen.weight'`
**Cause**: Lance has per-layer MoE generation expert twins (`_moe_gen` suffix). ACE-Step's Qwen3 base has per-head `q_norm` and `k_norm` (not in Qwen2). llama.cpp's GGUF converter doesn't know these.
**Fix**: Filter before GGUF conversion:
```python
DROP_PATTERNS = [
    "_moe_gen",         # Lance per-layer MoE gen expert
    ".q_norm.",         # Qwen3 per-head Q norm
    ".k_norm.",         # Qwen3 per-head K norm
    "latent_pos_embed", # Lance latent-space PE
    "vae2llm",          # Lance VAE→LLM adapter
    "llm2vae",          # Lance LLM→VAE adapter
    "time_embedder",    # Lance diffusion timestep embedder
]
```

## Pitfall 7: Tokenizer files not copied
**Symptom**: `TypeError: expected str, bytes or os.PathLike object, not NoneType` during GGUF conversion.
**Cause**: GGUF converter needs `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`, `added_tokens.json` in the model directory. They're not in safetensors.
**Fix**: Copy from the source parent after filter:
```bash
cp /path/to/Qwen2.5-Omni-3B/{tokenizer.json,tokenizer_config.json,vocab.json,merges.txt,special_tokens_map.json,added_tokens.json} /path/to/filtered_ckpt/
```

## Pitfall 8: Treating "tie" as truth
**Symptom**: gen_0 worked at 2/10 without lm_head. The new OmniLance with lm_head is 0/10 (`???`).
**Cause**: gen_0 didn't have lm_head in safetensors, so llama.cpp's Qwen2 loader fell back to `tie_word_embeddings: true` and used embed_tokens. The new model has BOTH lm_head (averaged, bad) AND `tie_word_embeddings: false` config — uses the averaged lm_head which is bad.
**Fix**: Either (a) don't put lm_head in safetensors, let llama.cpp tie; or (b) put lm_head AND set `tie_word_embeddings: false` to use it (but then the lm_head must be correct). Option (a) is safer.

## Pitfall 9: Benchmarking proxies, not real inference
**Symptom**: Proxy fitness says "0.609" and converges. Real benchmark says 0/10.
**Cause**: Proxy fitness was a mathematical combination of layer statistics, not actual model output.
**Fix**: Always benchmark real inference. Use the 10-question reasoning template (MMLU, GSM8K, etc.) against llama-server.

## Pitfall 10: Single fixed-genome merge without CMA-ES
**Symptom**: All scores worse than original parent.
**Cause**: The paper's claim is that CMA-ES evolution finds genome values that produce BETTER children. A single fixed-genome merge is not the paper — it's just one sample from the search space.
**Fix**: Implement CMA-ES. Even a small population (10-20) over 10-20 generations helps. Without it, naive merging of 3B parents with similar but non-identical distributions degrades performance.

## Detection checklist (run before declaring merge done)

1. `lm_head.weight` present in safetensors? (Or intentionally omitted for tied)
2. Filter patterns applied? (No `_moe_gen`, no Qwen3 `q_norm`)
3. Tokenizer files present in checkpoint dir?
4. `tie_word_embeddings` in config matches what's in safetensors?
5. Compare merged norm.weight std to parent std — should be similar (not 0.5x or 2x)
6. Compare merged embed_tokens std to parent std — should be similar
7. Compare merged lm_head (if present) std to parent std — should be similar
8. Run real benchmark BEFORE declaring success
9. Compare to original parent baseline — if merged is much worse, debug
10. If using cross-arch merge (different hidden_dim), verify Architecture Mapper skipped mismatched tensors (not random-projected)
