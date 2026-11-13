"""Fill figure_3.tex (per-seed failing length). Rows are matched by their comment
"% <task>/<model>"; deltaproduct-nhK selects n_h = K (default 2). NFSM_ASSETS_DIR points to the table.
"""

import glob
import json
import os
import re
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = ["42", "43", "44", "45", "46"]
CROSS, TBD = r"$\times$", r"\tbd"
FSA = {"Z2": "regular", "Z16": "regular", "S3": "factored", "S4": "factored",
       "A5": "factored", "M11": "factored", "DFF5": "regular", "FF": "regular"}


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


def tex(x, capped):
    return "$" + (r"\geq " if capped else "") + fmt(int(x)) + "$"


def row(rec, start):
    """Seven cells: one per seed, then the median and range over the seeds that learn the task."""
    if rec is None:
        return [TBD] * 7
    cells, vals = [], []                         # vals: (length, capped?) of the seeds that learn it
    for s in SEEDS:
        v = rec["seeds"].get(s)
        if v is None:
            cells.append(TBD)
            continue
        L, cap = v["failing_length"], v.get("cap", rec["cap"])
        if L is None:
            vals.append((cap, True))
        elif L > start:
            vals.append((L, False))
        else:                                    # fails at the training length: never learned
            cells.append(CROSS)
            continue
        cells.append(tex(*vals[-1]))
    if not vals:
        return cells + [CROSS, ""]
    capped = {x for x, c in vals if c}
    lengths = [x for x, _ in vals]
    med = statistics.median(lengths)
    lo, hi = min(lengths), max(lengths)
    return cells + [tex(med, med in capped), f"[{tex(lo, lo in capped)};\\,{tex(hi, hi in capped)}]"]


fsa, tso = load("fsa"), load("tso")


def _baseline(model):
    """(cell, rung) of a baseline row: deltaproduct is n_h = 2 unless written deltaproduct-nhK."""
    if model.startswith("deltaproduct"):
        return "deltaproduct", (model.split("-", 1)[1] if "-" in model else "nh2")
    return model, "fixed"


def lookup(task, model):
    model, rung = _baseline(model) if model != "nfsm" else (model, None)
    if task.startswith("TSO"):
        q = f"question-{task[3:]}"
        tok = (f"{q},exact,nfsm,2,free" if model == "nfsm" else f"{q},{rung},{model},4,free")
        return tso.get(tok), 256
    tok = (f"{task},{FSA[task]},nfsm,1,free" if model == "nfsm" else f"{task},{rung},{model},4,free")
    return fsa.get(tok), 64


ASSETS = os.environ.get("NFSM_ASSETS_DIR") or os.path.join(ROOT, "assets")
path = os.path.join(ASSETS, "figure_3.tex")
with open(path) as f:
    lines = f.read().split("\n")
pat = re.compile(r"^(.*?&.*?)& .* \\\\ % (\w+)/([\w-]+)\s*$")
n = 0
for i, line in enumerate(lines):
    m = pat.match(line)
    if not m:
        continue
    lead, task, model = m.groups()
    lines[i] = lead + "& " + " & ".join(row(*lookup(task, model))) + rf" \\ % {task}/{model}"
    n += 1
with open(path, "w") as f:
    f.write("\n".join(lines))
print(f"wrote {path}: {n} rows")
