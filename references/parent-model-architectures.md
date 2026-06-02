# Parent Model Architectures (OmniSenter project)

## Qwen2.5-Omni-3B
- **HF**: `Qwen/Qwen2.5-Omni-3B`
- **Architecture**: `Qwen2_5OmniModel`
- **Model type**: `qwen2_5_omni`
- **Text body** (what we merge):
  - `hidden_size`: 2048
  - `num_hidden_layers`: 36
  - `intermediate_size`: 11008
  - `num_attention_heads`: 16
  - `num_key_value_heads`: 2
  - `vocab_size`: 151936
  - `max_position_embeddings`: 32768
  - `tie_word_embeddings`: false
  - `rope_scaling`: mrope with section [16, 24, 24]
- **Modality stack** (attached separately, NOT merged):
  - **Thinker**: text LLM (the part we merge)
  - **Vision encoder** (NaViT-style): depth=32, hidden=1280, patch=14
  - **Audio encoder** (Whisper-style): 32 layers, d_model=1280
  - **Talker**: 24 layers, hidden=896, vocab=8448 (TTS codec tokens)
  - **Token2Wav**: BigVGAN vocoder + DiT
- **Tensors to extract** (prefixes):
  - `thinker.model.*` → strip `thinker.model.` → e.g., `embed_tokens.weight`
  - `thinker.lm_head.weight` → `lm_head.weight` (CRITICAL — easy to miss)
- **Tensors to drop**: visual.*, audio.*, talker.*, token2wav.*

## Qwen2.5-VL-3B (Lance 3B is built on this)
- **HF**: `Qwen/Qwen2.5-VL-3B-Instruct` (for reference)
- **Architecture**: `Qwen2_5_VLForConditionalGeneration`
- **Model type**: `qwen2_5_vl`
- **Text body** (same shape as Omni):
  - hidden=2048, 36L, intermediate=11008, vocab=151936, heads=16, kv=2
  - `tie_word_embeddings`: true
- **Modality stack**:
  - **Vision encoder**: depth=32, hidden=1280, patch=14, out_hidden=2048
  - **No audio**, **no speech output**

## Lance 3B (bytedance-research/Lance)
- **HF**: `bytedance-research/Lance`
- **Architecture**: `Qwen2_5_VLForConditionalGeneration` (built on Qwen2.5-VL-3B)
- **Model type**: `qwen2_5_vl`
- **Text body shape**: identical to Omni text body (hidden=2048, 36L, vocab=151936)
- **Unique tensors** (Lance-only, not in Omni):
  - `_moe_gen` (every layer has a paired `_moe_gen` weight — pre-trained dual experts for image/video generation)
  - `vae2llm`, `llm2vae` (VAE↔LLM adapters for diffusion conditioning)
  - `time_embedder` (diffusion timestep embedder)
  - `latent_pos_embed` (latent-space positional embedding)
  - These need to be FILTERED OUT before GGUF conversion (llama.cpp doesn't know them)
- **Tensors to extract** (prefixes):
  - `language_model.model.*` → strip `language_model.model.` → e.g., `embed_tokens.weight`
  - `language_model.lm_head.weight` → `lm_head.weight` (CRITICAL)

## ACE-Step LM (Qwen3-4B text body)
- **HF**: `ACE-Step/ACE-Step-v1-3.5B` includes `ace-step-5Hz-lm-4B` (Qwen3-4B base)
- **Architecture**: Qwen3-based text LLM
- **Text body**:
  - `hidden_size`: 2560 (NOT 2048 — different from Omni/Lance)
  - `num_hidden_layers`: 36
  - `vocab_size`: 217204 (NOT 151936 — different from Omni/Lance)
- **Architecture Mapper verdict** for Omni (2048) + ACE-Step (2560):
  - Type: both transformer, high compatibility
  - Dim: 2048 vs 2560, low compatibility (β2=0.3)
  - Comp = 0.5·1 + 0.3·0.8 + 0.2·0.5 ≈ 0.8 — marginal pass
  - Tensors with shape mismatches get SKIPPED (use parent A)
- **Per-tensor merge only works on shape-matched tensors** (very few — mostly biases, layernorms with no dim suffix)
- **lm_head shape mismatch** (Omni 151936×2048 vs ACE 217204×2560) — skip, use Omni

## Modality heads (for "expert-on-base" architecture)
- **Vision in**: Omni's NaViT (model_type `qwen2_5_omni_vision_encoder`) — output_dim=2048
- **Audio in**: Omni's Whisper-style audio encoder (model_type `qwen2_5_omni_audio_encoder`) — output_dim=2048
- **Speech out**: Omni's talker (model_type `qwen2_5_omni_talker`) + token2wav (BigVGAN + DiT)
- **Image/Video gen**: Lance's DiT (NOT in our downloaded weights — only text body was pulled; need full Lance if you want image/video generation)
- **Music gen**: ACE-Step DiT (downloaded separately at `senter-omni/ace-step-lora/weights/ace_step_transformer`) + UMT5 lyric encoder

## Compatibility matrix (text body merge)

| Parent A \ Parent B | Omni-3B | Lance 3B | ACE-Step LM |
|---|---|---|---|
| **Omni-3B** | — | 100% (same shape) | partial (dim mismatch, vocab mismatch) |
| **Lance 3B** | 100% | — | partial |
| **ACE-Step LM** | partial | partial | — |

**Clean merges**: Omni↔Lance (any direction)
**Hard merges**: anything with ACE-Step (Architecture Mapper mostly skips, ACE contributes very little)
