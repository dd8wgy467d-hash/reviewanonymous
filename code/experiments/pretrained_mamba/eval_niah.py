"""Mamba-2 (MAMBA2_SIZE, default 130m) on niah_single_1 over the (L, d) grid.

    uv run python eval_niah.py --layout A --tau inf  --out results/niah_A_asis.json
    uv run python eval_niah.py --layout B --tau 2048 --out results/niah_B_tau2048.json

One pass per haystack and L gives the prefix state; each (needle, d) runs from it. --haystacks runs
a subset (one cluster job each); merge_parts.py joins them.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mamba2_jax as m
import niah

N_GEN = 16
SEED = 0
D_DEFAULT = [0, 25, 50, 100, 250, 500, 750, 1025, 2050, 3075, 4100, 8200]     # multiples of 25: whole haystack lines
L_DEFAULT = [0, 25, 2050, 8200, 32775, 131075]


def prefix_states(params, pb, layout, h, Ls, tau):
    """State after the prompt-independent prefix of haystack h, for every L in Ls."""
    states = {}
    for L in Ls:
        ids = pb.prefix(layout, h, L)
        states[L] = m.run(params, np.array([ids]), m.init_state(1), tau)[1] if ids else m.init_state(1)
    return states


def stack_states(states):
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=1), *states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["A", "B", "C"], required=True,
                    help="A: instruction first; B: instruction just before the needle; C: no instruction")
    ap.add_argument("--tau", type=float, default=math.inf)
    ap.add_argument("--d", type=int, nargs="+", default=D_DEFAULT)
    ap.add_argument("--L", type=int, nargs="+", default=L_DEFAULT)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--n-haystacks", type=int, default=10)
    ap.add_argument("--haystacks", type=int, nargs="+", default=None,
                    help="run only these haystacks (default: all); the output is then a part")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--append", action="store_true",
                    help="merge into an existing --out file (same settings), replacing the given d")
    args = ap.parse_args()
    assert args.n_prompts % args.n_haystacks == 0
    per_h = args.n_prompts // args.n_haystacks

    params = m.load_params()
    tok = m.load_tokenizer()
    pb = niah.PromptBuilder(tok, SEED)
    prompts = niah.make_prompts(args.n_prompts, SEED)
    Ls = sorted(args.L)

    t0 = time.time()
    records = []
    hs = sorted(args.haystacks) if args.haystacks is not None else list(range(args.n_haystacks))
    assert not (args.append and args.haystacks is not None), "parts are joined by merge_parts.py"
    for h in hs:
        t1 = time.time()
        batch = stack_states([s for _, s in sorted(prefix_states(params, pb, args.layout, h, Ls, args.tau).items())])
        print(f"haystack {h}: prefix passes {time.time() - t1:.0f}s", flush=True)
        for i in range(h * per_h, (h + 1) * per_h):
            p = prompts[i]
            t1 = time.time()
            for d in args.d:
                ids = np.tile(np.array([pb.suffix(args.layout, p, h, d)]), (len(Ls), 1))
                logits, st = m.run(params, ids, batch, args.tau)
                gen = m.generate(params, logits, st, N_GEN, args.tau)
                for j, L in enumerate(Ls):
                    text = tok.decode(gen[j].tolist())
                    records.append({"d": d, "L": L, "prompt": i, "haystack": h, "key": p["key"],
                                    "value": p["value"], "output": text, "correct": p["value"] in text})
            acc = np.mean([r["correct"] for r in records if r["prompt"] == i])
            print(f"  prompt {i:3d}: {acc:.2f} of (d, L) correct ({time.time() - t1:.1f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"layout": args.layout, "tau": args.tau if math.isfinite(args.tau) else "inf", "d": args.d,
            "L": Ls, "n_prompts": args.n_prompts, "n_haystacks": args.n_haystacks, "seed": SEED,
            "n_generated_tokens": N_GEN, "model": f"state-spaces/mamba2-{m.SIZE}", "haystacks": hs,
            "segment": m.SEGMENT, "backend": jax.default_backend(),
            "seconds": round(time.time() - t0)}
    if args.append and args.out.exists():
        # merge with an earlier run of the same variant: new (d, L) points replace old ones, the grids
        # become the unions (an analysis uses the (d, L) points every run has)
        old = json.loads(args.out.read_text())
        for k in ("layout", "tau", "n_prompts", "n_haystacks", "seed", "n_generated_tokens"):
            assert old["meta"][k] == meta[k], f"cannot append: {k} differs"
        new = {(d, L) for d in args.d for L in Ls}
        records = [r for r in old["records"] if (r["d"], r["L"]) not in new] + records
        meta["d"] = sorted(set(old["meta"]["d"]) | set(args.d))
        meta["L"] = sorted(set(old["meta"]["L"]) | set(Ls))
        meta["seconds"] += old["meta"]["seconds"]
    args.out.write_text(json.dumps({"meta": meta, "records": records}, indent=1))
    print("accuracy (rows d, columns L):")
    for d in meta["d"]:
        accs = [np.mean([r["correct"] for r in records if r["d"] == d and r["L"] == L]) for L in Ls]
        print(f"  d={d:5d} " + " ".join(f"{a:.2f}" for a in accs))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
