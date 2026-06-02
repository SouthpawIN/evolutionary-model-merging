#!/usr/bin/env python3
"""
Filter non-text-LLM tensors before GGUF conversion.

Lance 3B and ACE-Step (Qwen3-based) have tensors that llama.cpp's GGUF converter
doesn't know how to map. This script strips them out and saves a clean text-LLM
checkpoint ready for conversion.

Usage:
  python3 filter_for_gguf.py <input_dir> <output_dir>
"""
import os, sys, json, gc
from pathlib import Path
import safetensors.torch as st
import torch

# Tensors to drop — non-text-LLM modality heads / cross-arch artifacts
DROP_PATTERNS = [
    "_moe_gen",         # Lance per-layer MoE gen expert twins
    ".q_norm.",         # Qwen3 per-head Q norm (not in Qwen2)
    ".k_norm.",         # Qwen3 per-head K norm (not in Qwen2)
    "latent_pos_embed", # Lance latent-space PE (diffusion)
    "vae2llm",          # Lance VAE→LLM adapter
    "llm2vae",          # Lance LLM→VAE adapter
    "time_embedder",    # Lance diffusion timestep embedder
]

def should_drop(k):
    return any(p in k for p in DROP_PATTERNS)

def main():
    if len(sys.argv) != 3:
        print("Usage: filter_for_gguf.py <input_dir> <output_dir>")
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)

    kept, dropped = {}, []
    for f in sorted(src.glob("*.safetensors")):
        print(f"  reading {f.name}")
        d = st.load_file(str(f), device="cpu")
        for k, v in d.items():
            (dropped if should_drop(k) else kept).__setitem__(k, v)
    print(f"  kept: {len(kept)}, dropped: {len(dropped)}")
    print(f"  patterns: {DROP_PATTERNS}")

    SHARD = 5 * 1024**3
    bf16 = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in kept.items()}
    shards, cur, sz = [], {}, 0
    for k, v in bf16.items():
        vs = v.numel() * v.element_size()
        if sz + vs > SHARD and cur:
            shards.append(cur); cur = {}; sz = 0
        cur[k] = v; sz += vs
    if cur: shards.append(cur)
    n = len(shards)
    wm = {}
    for i, s in enumerate(shards, 1):
        fn = f"model-{i:05d}-of-{n:05d}.safetensors"
        st.save_file(s, str(dst / fn))
        for k in s: wm[k] = fn
        print(f"  wrote {fn} ({sum(v.numel() * v.element_size() for v in s.values()) / 1e9:.2f}GB)")
    total = sum(v.numel() * v.element_size() for v in bf16.values())
    with open(dst / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": wm}, f, indent=2)

    # Copy config and tokenizer files
    import shutil
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
            print(f"  copied {f}")
    print(f"Done → {dst}")

if __name__ == "__main__":
    main()
