#!/usr/bin/env python3
"""
Paper-exact 2-parent merge per Darwin Family (arXiv:2605.14386).
NO inventions, NO modifications. Use as a reference template.

MRI-Trust Fusion with per-tensor r_final. Pure convex combination.
For 3+ parents, see cma_es_evolution.py for the proper multi-generation approach.
"""
import os, json, time, gc, argparse
from pathlib import Path
import torch
import safetensors.torch as st

# Paper-fixed constants
ALPHA_MRI = 0.5       # Static vs Probe balance
TAU_DEFAULT = 0.4     # MRI-Trust (paper converges to 0.35-0.55)

def load_sf(p, prefix=None):
    s = {}
    for f in sorted(Path(p).glob("*.safetensors")):
        d = st.load_file(str(f), device="cpu")
        if prefix: d = {k: v for k, v in d.items() if k.startswith(prefix)}
        s.update(d)
    return s

def extract_text(s, name=""):
    """Strip encoders/heads, keep text LLM body AND lm_head.

    Critical: lm_head is OUTSIDE the model.* prefix in Omni/Lance/Qwen3.
    """
    o = {}
    for k, v in s.items():
        if k.startswith("thinker.model."):          o[k.replace("thinker.model.", "", 1)] = v
        elif k.startswith("language_model.model."): o[k.replace("language_model.model.", "", 1)] = v
        elif k.startswith("model."):                o[k.replace("model.", "", 1)] = v
        # lm_head lives outside model.* — MUST extract explicitly
        elif k in ("thinker.lm_head.weight", "language_model.lm_head.weight", "lm_head.weight"):
            o["lm_head.weight"] = v
    print(f"  [{name}] extracted {len(o)} text tensors (lm_head={'lm_head.weight' in o})")
    return o

def static_term(t):
    """Paper's Static: normalized entropy + variance + capped ℓ2-norm.

    Simplified scalar 'importance'. Higher = more important for this tensor.
    """
    a = t.float().abs() + 1e-12
    p = a / a.sum()
    H = -(p * p.log()).sum()
    V = t.float().var().sqrt() + 1e-12
    N = torch.clamp(t.float().norm(), max=t.float().norm().item() / 5 + 1e-12)
    return H + V.sqrt() + N.log()

def mri_trust_r(t_a, t_b, rho_b=0.5, tau=TAU_DEFAULT, alpha=ALPHA_MRI):
    """Paper Eq: r_final = τ·r_MRI + (1-τ)·r_genome
                  r_MRI = MRI_B / (MRI_A + MRI_B)
                  MRI = α·Static + (1-α)·Probe (Probe omitted, Static-dominant)
    """
    static_a = static_term(t_a)
    static_b = static_term(t_b)
    r_mri = static_b / (static_a + static_b)
    r_genome = rho_b
    return tau * r_mri + (1 - tau) * r_genome

def paper_merge_2parent(theta_a, theta_b, rho_b=0.5, tau=TAU_DEFAULT):
    """Paper-exact 2-parent merge.

    θM = (1 - r_final) · θA + r_final · θB  per tensor
    r_final from MRI-Trust Fusion.

    For cross-architecture (different dim), Architecture Mapper SKIPS
    dim-mismatched tensors (keeps parent A's value). NO random projection.
    """
    out = {}
    shared = sorted(set(theta_a.keys()) & set(theta_b.keys()))
    a_only = sorted(set(theta_a.keys()) - set(theta_b.keys()))
    b_only = sorted(set(theta_b.keys()) - set(theta_a.keys()))
    print(f"  shared: {len(shared)}, A-only: {len(a_only)}, B-only: {len(b_only)}")

    n_merged, n_skipped = 0, 0
    for k in shared:
        ta, tb = theta_a[k].float(), theta_b[k].float()
        if ta.shape != tb.shape:
            # Architecture Mapper: shape mismatch → skip, keep A
            out[k] = theta_a[k].clone()
            n_skipped += 1
            continue
        r = mri_trust_r(ta, tb, rho_b=rho_b, tau=tau)
        m = (1 - r) * ta + r * tb
        out[k] = m.to(theta_a[k].dtype)
        n_merged += 1
    for k in a_only: out[k] = theta_a[k].clone()
    for k in b_only: out[k] = theta_b[k].clone()
    print(f"  merged: {n_merged}, skipped (shape): {n_skipped}")
    return out

def save_sharded(state, path, shard_bytes=5 * 1024**3):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    bf16 = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in state.items()}
    shards, cur, sz = [], {}, 0
    for k, v in bf16.items():
        vs = v.numel() * v.element_size()
        if sz + vs > shard_bytes and cur:
            shards.append(cur); cur = {}; sz = 0
        cur[k] = v; sz += vs
    if cur: shards.append(cur)
    n = len(shards)
    wm = {}
    for i, s in enumerate(shards, 1):
        fn = f"model-{i:05d}-of-{n:05d}.safetensors"
        st.save_file(s, str(path / fn))
        for k in s: wm[k] = fn
    total = sum(v.numel() * v.element_size() for v in bf16.values())
    with open(path / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": wm}, f, indent=2)
    print(f"  saved {len(state)} tensors, {total/1e9:.2f}GB, {n} shards")

def write_config(path, hidden=2048, layers=36, vocab=151936, intermediate=11008, tie=True):
    cfg = {
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
        "hidden_size": hidden, "intermediate_size": intermediate,
        "num_hidden_layers": layers, "num_attention_heads": 16,
        "num_key_value_heads": 2, "max_position_embeddings": 32768,
        "vocab_size": vocab, "torch_dtype": "bfloat16",
        "tie_word_embeddings": tie,
    }
    with open(Path(path) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="Path to parent A safetensors dir")
    p.add_argument("--a_prefix", default="thinker.", help="Prefix filter for A (e.g., thinker.)")
    p.add_argument("--b", required=True, help="Path to parent B safetensors dir")
    p.add_argument("--b_prefix", default=None, help="Prefix filter for B (e.g., language_model.)")
    p.add_argument("--out", required=True, help="Output dir")
    p.add_argument("--rho_b", type=float, default=0.5, help="Genome ρ for parent B (0..1)")
    p.add_argument("--tau", type=float, default=TAU_DEFAULT)
    p.add_argument("--tie", action="store_true", help="Set tie_word_embeddings=True (Qwen2 vanilla)")
    args = p.parse_args()

    print(f"Paper-exact 2-parent merge")
    print(f"  A: {args.a} (prefix={args.a_prefix})")
    print(f"  B: {args.b} (prefix={args.b_prefix})")
    print(f"  ρ_B={args.rho_b}, τ={args.tau}, tie_embeddings={args.tie}")

    print("Load A")
    a = load_sf(args.a, prefix=args.a_prefix)
    print(f"Load B")
    b = load_sf(args.b, prefix=args.b_prefix)
    print("Extract text")
    ta = extract_text(a, "A")
    tb = extract_text(b, "B")
    del a, b; gc.collect()
    print("Paper merge")
    m = paper_merge_2parent(ta, tb, rho_b=args.rho_b, tau=args.tau)
    del ta, tb; gc.collect()
    print("Save")
    save_sharded(m, args.out)
    write_config(args.out, tie=args.tie)
    print(f"Done → {args.out}")

if __name__ == "__main__":
    main()
