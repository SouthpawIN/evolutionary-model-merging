#!/usr/bin/env python3
"""
CMA-ES evolution for the 14-dim merge genome (Darwin Family paper, actual method).

This is the paper's real approach. A single fixed-genome merge is NOT the paper —
CMA-ES evolves the genome over many generations to find values that produce children
better than either parent.

Run paper_exact_2parent_merge.py for each candidate genome, score on a real
benchmark, and let CMA-ES update its distribution toward higher-fitness candidates.

Requirements:
  pip install cma
  llama-server running with merged checkpoints (or use a fast scoring proxy)

Usage:
  python3 cma_es_evolution.py --port 11434 --generations 10 --population 8
"""
import argparse, time, json, subprocess, sys
import numpy as np

# Try importing cma; if not installed, fall back to simple random search with logging
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("cma not installed — install with `pip install cma` for proper CMA-ES. Falling back to random search.")

# Genome bounds (each dim in [0, 1])
GENOME_BOUNDS = {
    "gamma":      (0.0, 1.0),
    "alpha_attn": (0.0, 1.0),
    "alpha_ffn":  (0.0, 1.0),
    "alpha_emb":  (0.0, 1.0),
    "rho_a":      (0.0, 1.0),
    "rho_b":      (0.0, 1.0),
    "r0":         (0.0, 1.0), "r1": (0.0, 1.0), "r2": (0.0, 1.0),
    "r3":         (0.0, 1.0), "r4": (0.0, 1.0), "r5": (0.0, 1.0),
    "tau":        (0.0, 1.0),
    "lambda_reg": (0.0, 1.0),
}
GENOME_KEYS = list(GENOME_BOUNDS.keys())
N_DIMS = len(GENOME_KEYS)

def genome_to_kwargs(g_vec):
    return {k: float(v) for k, v in zip(GENOME_KEYS, g_vec)}

def random_genome(rng):
    return np.array([rng.uniform(lo, hi) for lo, hi in GENOME_BOUNDS.values()])

def evaluate_genome(g_vec, port, eval_script=None):
    """Build merged child with this genome, run benchmark, return score.

    For efficiency in CMA-ES, you might use a fast proxy (e.g., perplexity on a
    held-out set) instead of full benchmark. But the paper uses real benchmark
    scores, so prefer that for the final generation.
    """
    kwargs = genome_to_kwargs(g_vec)
    # 1. Build merged child
    # TODO: call paper_exact_2parent_merge.py with --rho_b and --tau from kwargs
    # 2. Convert to GGUF, serve on a new port
    # 3. Run real_benchmark.py against that port
    # 4. Return score
    #
    # For now this is a stub — wire up your specific pipeline here.
    # Reference: scripts/paper_exact_2parent_merge.py + filter + convert + serve + benchmark
    raise NotImplementedError("Wire this up to your merge + benchmark pipeline")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=11434, help="llama-server port")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.2, help="Initial CMA-ES sigma")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="cma_es_log.json", help="Path to write evolution log")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    if HAS_CMA:
        # CMA-ES over 14-dim unit-cube (will be rescaled to bounds)
        x0 = np.full(N_DIMS, 0.5)
        es = cma.CMAEvolutionStrategy(x0, args.sigma, {
            "bounds": [0, 1],
            "popsize": args.population,
            "verbose": -9,
        })
        log = {"method": "cma-es", "generations": [], "best_genome": None, "best_fitness": -1}
    else:
        # Random search fallback
        es = None
        log = {"method": "random-search", "generations": [], "best_genome": None, "best_fitness": -1}

    t0 = time.time()
    for gen in range(args.generations):
        gen_log = {"gen": gen, "candidates": []}
        if HAS_CMA:
            candidates = es.ask()
        else:
            candidates = [random_genome(rng) for _ in range(args.population)]

        for i, c in enumerate(candidates):
            try:
                score = evaluate_genome(c, args.port)
            except NotImplementedError as e:
                print(str(e))
                sys.exit(1)
            except Exception as e:
                print(f"  [gen {gen} cand {i}] ERR: {e}")
                score = 0
            gen_log["candidates"].append({"genome": list(c), "fitness": score})
            if score > log["best_fitness"]:
                log["best_genome"] = list(c)
                log["best_fitness"] = score

        if HAS_CMA:
            fitnesses = [-c["fitness"] for c in gen_log["candidates"]]
            es.tell(candidates, fitnesses)
        log["generations"].append(gen_log)
        print(f"gen {gen}: best={log['best_fitness']}/{len(gen_log['candidates'])}")

        with open(args.out, "w") as f:
            json.dump(log, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Best fitness: {log['best_fitness']}")
    print(f"Best genome: {genome_to_kwargs(log['best_genome'])}")
    print(f"Log: {args.out}")

if __name__ == "__main__":
    main()
