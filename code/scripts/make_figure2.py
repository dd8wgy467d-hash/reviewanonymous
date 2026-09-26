"""Rebuild figure_2.tex (the failing-length table) from the failing-length results.

A cell shows the best (or median) L_max over the seeds that learn the task (accuracy at or above the bar at
the training length), with [min;max] below it and the count of such seeds; a cross marks a
configuration no seed learns, and -- a configuration not measured yet.

The table lives in the paper tree, not in this repository. Set NFSM_ASSETS_DIR to the
directory holding it (default: ./assets)."""

import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FSA_COLS = [("Z2", "regular"), ("Z16", "regular"), ("S3", "factored"), ("S4", "factored"),
            ("A5", "factored"), ("M11", "factored"), ("DFF5", "regular"), ("FF", "regular")]
TSO_COLS = ["question-3", "question-4", "question-5"]
CROSS = r"$\times$"
STAT = "median"                                # "best" (max over seeds that learn) or "median"


def load(suite):
    out = {}
    for p in glob.glob(os.path.join(ROOT, "results", suite, "faillen", "*.json")):
        with open(p) as f:
            r = json.load(f)
        out[r["token"]] = r
    return out


def fmt(n):
    for size, tag in ((1 << 20, "M"), (1 << 10, "k")):
        if n >= size and n % size == 0:
            return f"{n // size}{tag}"
    return str(n)


def cell(rec, start):
    """(main, range, seeds) for one configuration."""
    if rec is None or not rec["seeds"]:
        return "--", "", ""
    vals = []                                   # (length, capped?) for the seeds that learn it
    for v in rec["seeds"].values():
        L, cap = v["failing_length"], v.get("cap", rec["cap"])
        if L is None:
            vals.append((cap, True))
        elif L > start:
            vals.append((L, False))
    n = len(rec["seeds"])
    if not vals:
        return CROSS, "", f"0/{n}"
    lengths = [x[0] for x in vals]
    med = max(lengths) if STAT == "best" else statistics.median(lengths)
    capped = {x[0] for x in vals if x[1]}
    # dagger: the transition table extracted from every seed that learns the task is exactly the
    # target automaton (FSA only; TSO runs carry no extraction)
    exact = [v["exact_machine"] for v in rec["seeds"].values() if v["exact_machine"] is not None]
    dagger = bool(exact) and sum(map(bool, exact)) >= len(vals)
    def tex(x, mark=False):
        return (r"$" + (r"\geq " if x in capped else "") + fmt(int(x))
                + (r"^{\dagger}" if mark else "") + r"$")
    return tex(med, dagger), f"[{tex(min(lengths))};{tex(max(lengths))}]", f"{len(vals)}/{n}"


fsa, tso = load("fsa"), load("tso")
rows = [("nfsm", "anchored", 1, r"\RTa"), ("mamba", "free", 4, r"\RTb"),
        ("aussm", "free", 4, r"\RTc"), ("pdssm", "free", 4, r"\RTd")]
left, right = [], []
for cellname, arm, layers, y in rows:
    macro = r"\RTcellO" if cellname == "nfsm" else r"\RTcell"
    for i, (task, rung) in enumerate(FSA_COLS):
        rec = fsa.get(f"{task},{rung if cellname == 'nfsm' else 'fixed'},{cellname},{layers},{arm}")
        main, rng, seeds = cell(rec, 64)
        left.append(f"{macro}{{{i}}}{{{y}}}{{{main}}}" +
                    (f"{{{seeds}}}" if cellname == "nfsm" else f"{{{rng}}}{{{seeds}}}"))
    macro = r"\RTcellOR" if cellname == "nfsm" else r"\RTcellR"
    for i, task in enumerate(TSO_COLS):
        rec = tso.get(f"{task},{'exact' if cellname == 'nfsm' else 'fixed'},{cellname},"
                      f"{2 if cellname == 'nfsm' else 4},{arm}")
        main, rng, seeds = cell(rec, 256)
        right.append(f"{macro}{{{i}}}{{{y}}}{{{main}}}" +
                     (f"{{{seeds}}}" if cellname == "nfsm" else f"{{{rng}}}{{{seeds}}}"))

ASSETS = os.environ.get("NFSM_ASSETS_DIR") or os.path.join(ROOT, "assets")
path = os.path.join(ASSETS, "figure_2.tex")
with open(path) as f:
    lines = f.read().split("\n")
cells, out = left + right, []
for line in lines:                              # replace the cell lines in place, keeping geometry
    if line.startswith("\\RTcell"):
        out.append(cells.pop(0))
    else:
        out.append(line)
assert not cells, f"{len(cells)} generated cells had no slot in the file"
with open(path, "w") as f:
    f.write("\n".join(out))
print(f"wrote {path}: {len(left)} left cells, {len(right)} right cells")
for line in left + right:
    print("  " + line)
