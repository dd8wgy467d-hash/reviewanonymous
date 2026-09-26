"""The configuration table of the wall-clock benchmark, one row per Slurm array task.

    python -m experiments.appendix.walltime.configs      # (re)write results/walltime/configs.csv

Experiment `length`: L = 2^6 .. 2^17; NFSM with one head of 16 states and with one head of 5,
Mamba and AUSSM at d = 16 with d_state = 16, PD-SSM at N = 16; sequential and parallel; F and FB.
The dense reference at d = 16 (parallel only, F and FB) is appended after these rows, so that the
ids of the first 320 configurations do not change.
Experiment `state`: L = 2^12, parallel only, F and FB; d = 2 .. 256 for NFSM (one head of d
states), Mamba and AUSSM (d_inner = d, d_state = 16), PD-SSM (N = d) and the dense reference.
The cheap `state` rows come first, then `length` by increasing L. No JAX needed.
"""

import csv
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(_ROOT, "results", "walltime")
CONFIGS_CSV = os.path.join(RESULTS_DIR, "configs.csv")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")

B = 16
M = 256
D_STATE = 16
LENGTH_L = [2 ** k for k in range(6, 18)]
LENGTH_D = 16
LENGTH_MODELS = [("nfsm", LENGTH_D), ("nfsm", 5), ("mamba", LENGTH_D), ("aussm", LENGTH_D),
                 ("pdssm", LENGTH_D)]
LENGTH_EXTRA = [("dense", LENGTH_D)]                  # parallel only; rows appended at the end
STATE_L = 2 ** 12
STATE_D = [2 ** k for k in range(1, 9)]
STATE_MODELS = ["nfsm", "mamba", "aussm", "pdssm", "dense"]
FIELDS = ["config_id", "experiment", "model", "impl", "mode", "L", "d", "d_state", "B", "m",
          "merge_size", "dtype"]
INT_FIELDS = ("config_id", "L", "d", "B", "m", "merge_size")


def merge_size(model, d):
    """Scalars combined per merge of the parallel scan."""
    return {"nfsm": d, "pdssm": 3 * d, "mamba": 2 * d * D_STATE, "aussm": 2 * d * D_STATE,
            "dense": d * d + d}[model]


def _row(experiment, model, impl, mode, L, d):
    return {"experiment": experiment, "model": model, "impl": impl, "mode": mode, "L": L, "d": d,
            "d_state": D_STATE if model in ("mamba", "aussm") else "", "B": B, "m": M,
            "merge_size": merge_size(model, d),
            "dtype": "float32 (complex64 state)" if model in ("aussm", "pdssm") else "float32"}


def rows():
    out = []
    for d in STATE_D:
        for model in STATE_MODELS:
            for mode in ("FB", "F"):
                out.append(_row("state", model, "parallel", mode, STATE_L, d))
    for L in LENGTH_L:
        for model, d in LENGTH_MODELS:
            for impl in ("parallel", "sequential"):
                for mode in ("FB", "F"):
                    out.append(_row("length", model, impl, mode, L, d))
    for L in LENGTH_L:
        for model, d in LENGTH_EXTRA:
            for mode in ("FB", "F"):
                out.append(_row("length", model, "parallel", mode, L, d))
    return [{"config_id": i, **r} for i, r in enumerate(out)]


def write(path=CONFIGS_CSV):
    table = rows()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(table)
    return table


def read(path=CONFIGS_CSV):
    with open(path, newline="") as f:
        table = list(csv.DictReader(f))
    for r in table:
        for k in INT_FIELDS:
            r[k] = int(r[k])
        r["d_state"] = int(r["d_state"]) if r["d_state"] else None
    return table


if __name__ == "__main__":
    table = write()
    print(f"wrote {CONFIGS_CSV}")
    print(len(table))
