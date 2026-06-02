#!/usr/bin/env python3
"""
Real-inference benchmark for llama-server. Use to evaluate merged models.

Compares model output to expected substrings. Always run this BEFORE declaring
a merge successful — proxy fitness functions don't reflect real generation quality.

Usage:
  python3 real_benchmark.py --url http://localhost:11434 --model omni-sender-gen_0-f16.gguf
"""
import argparse, time, json
import requests

DEFAULT_TESTS = [
    ("General knowledge", "What is the capital of France? One word.", "Paris"),
    ("Reasoning",         "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Yes or no.", "Yes"),
    ("Math",              "What is 17 × 24?", "408"),
    ("Code",              "Write a Python function that returns the sum of a list.", "def"),
    ("Comprehension",     "What planet is known as the Red Planet? One word.", "Mars"),
    ("Logic",             "I have a sister. My sister has a brother. Who am I to her? One word.", "brother"),
    ("Factual",           "What is the chemical symbol for gold?", "Au"),
    ("Coding",            "In Python, what does len([1,2,3]) return? Number only.", "3"),
    ("Science",           "How many planets are in our solar system? Number only.", "8"),
    ("Reasoning",         "What comes next: 2, 4, 8, 16, ?", "32"),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--model", required=True)
    p.add_argument("--max_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--tests", default=None, help="Path to JSON tests file [{category, prompt, expected}]")
    p.add_argument("--out", default=None, help="Write results JSON to this path")
    args = p.parse_args()

    if args.tests:
        with open(args.tests) as f:
            tests = [(t["category"], t["prompt"], t["expected"]) for t in json.load(f)]
    else:
        tests = DEFAULT_TESTS

    # Health check
    h = requests.get(f"{args.url}/health", timeout=10)
    if h.status_code != 200 or h.json().get("status") != "ok":
        print(f"server unhealthy: {h.text}")
        return

    api = f"{args.url}/v1/chat/completions"
    hdr = {"Content-Type": "application/json"}
    results, score = [], 0
    t0 = time.time()
    for i, (cat, q, exp) in enumerate(tests):
        try:
            r = requests.post(api, headers=hdr, json={
                "model": args.model,
                "messages": [{"role": "user", "content": q}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }, timeout=30)
            ans = r.json()["choices"][0]["message"]["content"].strip()
            ok = exp.lower() in ans.lower()
            score += ok
            results.append({"category": cat, "ok": ok, "answer": ans[:120], "expected": exp})
            print(f"[{i+1}/{len(tests)}] {'✓' if ok else '✗'} {cat}: {ans[:60]!r}")
        except Exception as e:
            results.append({"category": cat, "ok": False, "error": str(e), "expected": exp})
            print(f"[{i+1}/{len(tests)}] ✗ {cat}: ERR {e}")
    elapsed = time.time() - t0
    print(f"\n=== {args.model}: {score}/{len(tests)} in {elapsed:.1f}s ===")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "score": score, "total": len(tests),
                       "elapsed_s": elapsed, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
