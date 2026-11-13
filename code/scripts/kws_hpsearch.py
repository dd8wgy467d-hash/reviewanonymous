"""Random search over the NFSM training hyperparameters on KWS Stream (nfsm-64x4, 4 layers).

    python scripts/kws_hpsearch.py sample            # results/kws_hpsearch/configs.json
    python scripts/kws_hpsearch.py commands          # the two stages of each configuration
    python scripts/kws_hpsearch.py table [--ref DIR] # results/kws_hpsearch/summary.md

15 draws, seed 42 each: lr log-uniform in [1e-4, 3e-3], batch in {16, 32, 64, 128}, nu log-uniform
in [0.25, 2], wd log-uniform in [1e-6, 1e-2], in both stages. Configuration i writes to
$HPSEARCH_DIR/c<i> (default results/kws_hpsearch).
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("HPSEARCH_DIR", os.path.join(ROOT, "results", "kws_hpsearch"))
TOKEN = "nfsm-64x4,4,last,42"
N, SAMPLE_SEED = 15, 0
TEST_LENS_HP = (101, 200, 400, 800, 1600)
CHANCE = 1 / 11


def loguniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def sample() -> list:
    rng = np.random.default_rng(SAMPLE_SEED)
    lr = loguniform(rng, 1e-4, 3e-3, N)
    batch = rng.choice([16, 32, 64, 128], N)
    nu = loguniform(rng, 0.25, 2.0, N)
    wd = loguniform(rng, 1e-6, 1e-2, N)
    return [{"id": f"c{i:02d}", "token": TOKEN, "lr": float(f"{lr[i]:.3g}"), "batch": int(batch[i]),
             "nu": float(f"{nu[i]:.3g}"), "wd": float(f"{wd[i]:.3g}")} for i in range(N)]


def load_configs() -> list:
    with open(os.path.join(OUT, "configs.json")) as f:
        return json.load(f)["configs"]


def hp_args(c) -> str:
    return f"--lr {c['lr']} --batch {c['batch']} --nu {c['nu']} --wd {c['wd']}"


def part(d, suite):
    p = os.path.join(d, suite, "parts", TOKEN.replace(",", "_") + ".json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def row_of(rec_s, rec_k) -> dict:
    accs = {T: rec_s["acc"][rec_s["lens"].index(T)] for T in TEST_LENS_HP if T in rec_s["lens"]}
    h = rec_s["history"] or {}
    return {"acc": accs, "val": h.get("best_val_acc"), "steps": h.get("steps_run"),
            "digits_steps": (rec_k or {}).get("history", {}).get("steps_run"),
            "minutes": (rec_s.get("train_seconds", 0) + (rec_k or {}).get("train_seconds", 0)) / 60,
            "digits_test": (rec_k or {}).get("test_acc")}


def fmt(v, pct=True):
    return "-" if v is None else (f"{100 * v:.1f}" if pct else f"{v}")


def table(ref_dir) -> str:
    cfgs = load_configs()
    head = ("| config | lr | batch | ν | wd | " + " | ".join(f"{T / 100:g} s" for T in TEST_LENS_HP)
            + " | best val | KWS Digits test | steps (digits + stream) | wall time (min) | learned |")
    lines = ["# NFSM random search on KWS Stream: test accuracy (%, filtered noise, 11 classes)", "",
             head, "|---|" + "---:|" * (len(TEST_LENS_HP) + 9)]
    done = []
    for c in cfgs:
        d = os.path.join(OUT, c["id"])
        s, k = part(d, "kws_stream"), part(d, "kws")
        pre = f"| {c['id']} | {c['lr']:.2g} | {c['batch']} | {c['nu']:.2g} | {c['wd']:.1e} | "
        if s is None:
            lines.append(pre + " | ".join("-" for _ in TEST_LENS_HP) + " | - | "
                         + fmt(k and k.get("test_acc")) + " | - | - | not finished |")
            continue
        r = row_of(s, k)
        done.append(r)
        learned = min(r["acc"].values()) > 2 * CHANCE
        lines.append(pre + " | ".join(fmt(r["acc"].get(T)) for T in TEST_LENS_HP)
                     + f" | {fmt(r['val'])} | {fmt(r['digits_test'])} | {r['digits_steps']} + {r['steps']}"
                     f" | {r['minutes']:.0f} | {'yes' if learned else '**no (chance)**'} |")
    if done:
        for name, f in (("median", np.median), ("min", np.min), ("max", np.max)):
            vals = [f([r["acc"][T] for r in done if T in r["acc"]]) for T in TEST_LENS_HP]
            lines.append(f"| **{name}** ({len(done)} configs) | | | | | "
                         + " | ".join(fmt(v) for v in vals)
                         + f" | {fmt(f([r['val'] for r in done]))} | {fmt(f([r['digits_test'] for r in done]))}"
                         f" | | {f([r['minutes'] for r in done]):.0f} | |")
    if ref_dir:
        refs = []
        for seed in (42, 43, 44):
            p = os.path.join(ref_dir, f"nfsm-64x4_4_last_{seed}.json")
            if os.path.exists(p):
                with open(p) as f:
                    refs.append(json.load(f))
        if refs:
            vals = [[r["acc"][r["lens"].index(T)] for r in refs] for T in TEST_LENS_HP]
            lines.append(f"| **paper default** (seeds 42-44, median) | 1e-3 | 64 | 0.5 | 1.0e-04 | "
                         + " | ".join(fmt(np.median(v)) for v in vals) + " | | | 30000 + 45000 | | yes |")
    lines += ["", f"Learned: accuracy above {200 * CHANCE:.0f}% (twice chance, 1/11) at every length."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(text)
    print(text)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("sample", "commands", "table"))
    p.add_argument("--ref", help="folder of the default runs' part JSONs scored on the same lengths")
    a = p.parse_args()
    if a.action == "sample":
        os.makedirs(OUT, exist_ok=True)
        cfgs = sample()
        with open(os.path.join(OUT, "configs.json"), "w") as f:
            json.dump({"sampling_seed": SAMPLE_SEED, "token": TOKEN,
                       "space": {"lr": "log-uniform [1e-4, 3e-3]", "batch": "uniform {16, 32, 64, 128}",
                                 "nu": "log-uniform [0.25, 2.0]", "wd": "log-uniform [1e-6, 1e-2]"},
                       "configs": cfgs}, f, indent=1)
        for c in cfgs:
            print(c)
    elif a.action == "commands":
        lens = " ".join(map(str, TEST_LENS_HP))
        for c in load_configs():
            print(f"{c['id']}\t-m experiments.kws.run --job {TOKEN} {hp_args(c)} --no-wandb\t"
                  f"-m experiments.kws_stream.run --job {TOKEN} {hp_args(c)} --test-lens {lens} "
                  f"--ood-filter --no-wandb")
    else:
        table(a.ref)


if __name__ == "__main__":
    sys.exit(main())
