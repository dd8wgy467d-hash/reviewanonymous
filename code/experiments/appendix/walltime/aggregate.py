"""Merge the raw JSONs of the wall-clock benchmark into the CSV and the summary; never measures.

    python -m experiments.appendix.walltime.aggregate

Reads results/walltime/configs.csv and the raw JSONs of three runs, a later one replacing an
earlier one for the same configuration:
  raw/          no chunking
  raw_mem95/    no chunking, rerun with a larger XLA_PYTHON_CLIENT_MEM_FRACTION (optional)
  raw_chunked/  the model's default time-chunking budget (configurations out of memory without it)
The memory fraction of every run is read from its JSON.
and writes results/walltime.csv (one row per configuration, status `missing` where no JSON was
written) and results/walltime_summary.md. Needs numpy only. Figures: plots.py, from the CSV.
"""

import csv
import json
import os
import sys
from collections import Counter

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.appendix.walltime.configs import (CONFIGS_CSV, LENGTH_D, LENGTH_EXTRA, LENGTH_MODELS,
                                                   RESULTS_DIR, STATE_L, read)

CSV_OUT = os.path.join(_ROOT, "results", "walltime.csv")
SUMMARY_OUT = os.path.join(_ROOT, "results", "walltime_summary.md")
N_TIMED = 10
COLUMNS = (["config_id", "slurm_job_id", "experiment", "model", "impl", "mode", "L", "d",
            "d_state", "B", "m", "merge_size", "compile_ms"]
           + [f"t{i}" for i in range(1, N_TIMED + 1)]
           + ["median_ms", "iqr_ms", "min_ms", "max_ms", "peak_mem_mb", "mem_method", "n_traces",
              "status", "node", "gpu", "gpu_uuid", "gpu_mem_gb", "cuda", "jax_version",
              "flax_version", "dtype", "mem_fraction", "chunked", "chunk_len", "mem_at_cap",
              "source"])
SOURCES = ["raw", "raw_mem95", "raw_chunked"]          # increasing precedence
DEFAULT_FRACTION = 0.75                                # JAX's XLA_PYTHON_CLIENT_MEM_FRACTION
NAMES = {"nfsm": "NFSM", "mamba": "Mamba---", "aussm": "AUSSM", "pdssm": "PD-SSM",
         "dense": "dense"}
REF_L, REF_D = STATE_L, LENGTH_D
SLOPE_L_MIN, SLOPE_D_MIN = 2 ** 10, 16


def load(configs=CONFIGS_CSV, results_dir=RESULTS_DIR, sources=SOURCES):
    """Every configuration, merged with its JSON from the latest source that has one."""
    recs = []
    for cfg in read(configs):
        rec = {**cfg, "status": "missing", "reason": "no raw JSON", "source": ""}
        for src in sources:
            path = os.path.join(results_dir, src, f"{cfg['config_id']}.json")
            if os.path.exists(path):
                with open(path) as f:
                    rec = {**cfg, **json.load(f), "source": src}
        env = rec.get("env") or {}
        rec["mem_fraction"] = float(env.get("XLA_PYTHON_CLIENT_MEM_FRACTION") or DEFAULT_FRACTION)
        rec["chunked"] = bool(rec.get("chunked", False))
        if rec.get("chunk_len") is None and rec["status"] != "missing" and not rec["chunked"]:
            rec["chunk_len"] = rec["L"]                        # asserted when chunking was off
        cap = rec["mem_fraction"] * (rec.get("gpu_mem_gb") or 0) * 1024
        used = (rec.get("peak_mem_mb") or 0) + (rec.get("mem_idle_mb") or 0)
        rec["mem_at_cap"] = bool(rec["status"] == "ok" and cap and used >= 0.99 * cap)
        recs.append(rec)
    return recs


def write_csv(recs, path=CSV_OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in recs:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in COLUMNS])


# --- queries ------------------------------------------------------------------------------------

def _sel(recs, **kw):
    return [r for r in recs if all(r.get(k) == v for k, v in kw.items())]


def _median(recs, **kw):
    hits = [r for r in _sel(recs, **kw) if r["status"] == "ok"]
    return hits[0]["median_ms"] if hits else None


def _series(recs, x, y="median_ms", **kw):
    pts = sorted((r[x], r[y]) for r in _sel(recs, **kw) if r["status"] == "ok" and r.get(y))
    return [p[0] for p in pts], [p[1] for p in pts]


def _slope(xs, ys, x_min):
    pts = [(a, b) for a, b in zip(xs, ys) if a >= x_min and b and b > 0]
    if len(pts) < 2:
        return None, len(pts)
    a, b = np.log2([p[0] for p in pts]), np.log2([p[1] for p in pts])
    return float(np.polyfit(a, b, 1)[0]), len(pts)


def _label(model, d):
    return f"{NAMES[model]} (d={d})"


def _fmt(v, spec=".3g"):
    return "n/a" if v is None else format(v, spec)


def _pow2(L):
    return f"2^{int(np.log2(L))}"


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


# --- summary ------------------------------------------------------------------------------------

def crossover(recs, model, d, mode):
    """Smallest L from which the parallel scan is faster at every measured length."""
    par = dict(zip(*_series(recs, "L", experiment="length", model=model, d=d, impl="parallel",
                            mode=mode)))
    seq = dict(zip(*_series(recs, "L", experiment="length", model=model, d=d, impl="sequential",
                            mode=mode)))
    common = sorted(set(par) & set(seq))
    if not common:
        return "n/a (no length measured with both)"
    faster = [par[L] < seq[L] for L in common]
    if not any(faster):
        return f"none: sequential faster at every L in {_pow2(common[0])}..{_pow2(common[-1])}"
    i = len(common)
    while i > 0 and faster[i - 1]:
        i -= 1
    note = "" if all(faster[i:]) and not any(faster[:i]) else " (not monotone below it)"
    if i == 0:
        return f"<= {_pow2(common[0])} (parallel faster at every measured L){note}"
    if i == len(common):
        return f"none at the largest common L ({_pow2(common[-1])}){note}"
    return f"{_pow2(common[i])}{note}"


def summary(all_recs):
    lines = ["# Wall-clock measurements", ""]
    status = Counter(r["status"] + (" (chunked)" if r["chunked"] else "") for r in all_recs)
    lines += [f"{len(all_recs)} configurations: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())), ""]
    ok = [r for r in all_recs if r["status"] == "ok"]
    ran = [r for r in all_recs if r["status"] != "missing"]
    recs = [r for r in all_recs if not r["chunked"]]       # every statistic below is without chunking

    lines += ["## Sources", ""]
    for src in SOURCES:
        ids = [r["config_id"] for r in all_recs if r["source"] == src]
        lines.append(f"- `{src}` ({len(ids)}): " + (", ".join(map(str, ids)) if len(ids) < 60 else "the others"))
    at_cap = [r for r in all_recs if r["mem_at_cap"]]
    lines.append(f"- peak memory at the fraction cap, so a lower bound ({len(at_cap)}): "
                 + (", ".join(f"{r['config_id']} ({r['mem_fraction']})" for r in at_cap) or "none"))
    lines += [""]

    lines += ["## Hardware, software and environment", ""]
    for key in ("gpu", "gpu_mem_gb", "node", "cuda", "jax_version", "flax_version", "python",
                "interpreter", "dtype", "B", "m", "n_warmup", "n_timed", "seed", "mem_fraction",
                "chunked"):
        vals = Counter(str(r.get(key)) for r in ran)
        lines.append(f"- {key}: " + "; ".join(f"{v} ({n})" for v, n in sorted(vals.items())))
    uuids = {r.get("gpu_uuid") for r in ran if r.get("gpu_uuid")}
    lines.append(f"- distinct GPUs (UUID): {len(uuids)}")
    envs = Counter(json.dumps(r.get("env"), sort_keys=True) for r in ran
                   if isinstance(r.get("env"), dict))
    for env, n in envs.items():
        e = {k: v for k, v in json.loads(env).items() if k != "CUDA_VISIBLE_DEVICES"}
        lines.append(f"- environment ({n} jobs): `{json.dumps(e)}`")
    for method, n in Counter(r.get("mem_method") for r in ok).items():
        lines.append(f"- peak-memory method ({n} jobs): {method}")
    if len({(r.get('gpu'), r.get('gpu_mem_gb')) for r in ran}) > 1:
        lines.append("- **DEVIATION: the jobs did not all run on the same GPU model / memory size.**")
    lines += ["", "Notes on the setup:",
              "- No Slurm node features were available, so no `--constraint` could be set; "
              "every result records the GPU name, memory and a stable UUID surrogate (above).",
              "- XLA_PYTHON_CLIENT_MEM_FRACTION caps the memory XLA may use even without "
              "preallocation (JAX's default is 0.75); `oom` means over the cap of that run. "
              "Fractions used, by source: " + "; ".join(
                  f"`{src}` " + ", ".join(f"{fr} ({n})" for fr, n in sorted(Counter(
                      r["mem_fraction"] for r in all_recs if r["source"] == src).items()))
                  for src in SOURCES if any(r["source"] == src for r in all_recs))
              + ". A peak equal to the cap is a lower bound (listed under Sources).",
              "- Chunked runs use each model's default budget (NFSM theta 2^26, PD-SSM dictionary "
              "2^26, Mamba / AUSSM state 2^25 elements, in both scans; the dense reference 2^26 "
              "elements of d x d) and are left out of every statistic below except their own table.",
              "- Peak memory is the nvidia-smi high-water mark of XLA's memory pool during the timed "
              "calls minus the idle value after CUDA initialisation; the pool is not returned, so it "
              "includes compilation, autotuning and warm-up.",
              "- The `sequential` implementations loop only the recurrence: each layer's per-step "
              "input projections (NFSM's d^2 logits, the SSMs' drives) are computed for all steps at "
              "once, as in the model code.", ""]

    lines += ["## Crossover length (parallel faster than sequential)", ""]
    rows = [[_label(m, d), crossover(recs, m, d, "F"), crossover(recs, m, d, "FB")]
            for m, d in LENGTH_MODELS]
    lines += _table(["model", "F", "FB"], rows) + [""]

    lines += [f"## Log-log slopes of the median time", "",
              f"(a) against L, over L >= {_pow2(SLOPE_L_MIN)} (points used in brackets)", ""]
    rows, slopes_a = [], {}
    for m, d in LENGTH_MODELS + LENGTH_EXTRA:
        row = [_label(m, d)]
        for impl in ("parallel", "sequential"):
            for mode in ("F", "FB"):
                s, n = _slope(*_series(recs, "L", experiment="length", model=m, d=d, impl=impl,
                                       mode=mode), SLOPE_L_MIN)
                slopes_a[(m, d, impl, mode)] = s
                row.append(f"{_fmt(s, '.2f')} [{n}]")
        rows.append(row)
    lines += _table(["model", "parallel F", "parallel FB", "sequential F", "sequential FB"], rows)
    lines += ["", f"(b) against d at L = {_pow2(STATE_L)}, parallel, over d >= {SLOPE_D_MIN}", ""]
    rows, slopes_b = [], {}
    for m in ("nfsm", "mamba", "aussm", "pdssm", "dense"):
        row = [NAMES[m]]
        for mode in ("F", "FB"):
            s, n = _slope(*_series(recs, "d", experiment="state", model=m, mode=mode), SLOPE_D_MIN)
            slopes_b[(m, mode)] = s
            row.append(f"{_fmt(s, '.2f')} [{n}]")
        rows.append(row)
    lines += _table(["model", "F", "FB"], rows) + [""]

    lines += [f"## Time ratios at L = {_pow2(REF_L)}, d = {REF_D} (length experiment, parallel)", ""]
    rows = []
    for other in ("mamba", "aussm", "pdssm"):
        row = [f"NFSM / {NAMES[other]}"]
        for mode in ("F", "FB"):
            a = _median(recs, experiment="length", model="nfsm", d=REF_D, impl="parallel", mode=mode, L=REF_L)
            b = _median(recs, experiment="length", model=other, d=REF_D, impl="parallel", mode=mode, L=REF_L)
            row.append(_fmt(a / b if a and b else None))
        rows.append(row)
    lines += _table(["ratio", "F", "FB"], rows) + [""]

    lines += [f"## Dense merges / NFSM at L = {_pow2(STATE_L)} (state experiment)", ""]
    rows = []
    for d in sorted({r["d"] for r in recs if r["experiment"] == "state"}):
        row = [d]
        for mode in ("F", "FB"):
            a = _median(recs, experiment="state", model="dense", d=d, mode=mode)
            b = _median(recs, experiment="state", model="nfsm", d=d, mode=mode)
            row.append(_fmt(a / b if a and b else None))
        rows.append(row)
    lines += _table(["d", "F", "FB"], rows) + [""]

    lines += ["## Peak memory (MB) and longest L that fits (length experiment, FB)", "",
              "`>=` marks a peak at the memory cap (a lower bound)."]
    lines += [""]
    rows = []
    for m, d in LENGTH_MODELS + LENGTH_EXTRA:
        row = [_label(m, d)]
        for L in (2 ** 12, 2 ** 15):
            hit = _sel(all_recs, experiment="length", model=m, d=d, impl="parallel", mode="FB", L=L)
            r = hit[0] if hit else None
            if r is None:
                row.append("n/a")
            elif r["status"] != "ok":
                row.append(r["status"])
            else:
                row.append(("chunked " if r["chunked"] else "") + (">= " if r["mem_at_cap"] else "")
                           + _fmt(r.get("peak_mem_mb"), ".0f"))
        for impl in ("parallel", "sequential"):
            Ls, _ = _series(recs, "L", experiment="length", model=m, d=d, impl=impl, mode="FB")
            Lc, _ = _series(all_recs, "L", experiment="length", model=m, d=d, impl=impl, mode="FB")
            row.append(f"{_pow2(max(Ls)) if Ls else 'none'} / {_pow2(max(Lc)) if Lc else 'none'}")
        rows.append(row)
    lines += _table(["model", f"parallel {_pow2(2 ** 12)}", f"parallel {_pow2(2 ** 15)}",
                     "longest L parallel (no chunking / chunked)",
                     "longest L sequential (no chunking / chunked)"], rows) + [""]

    chunked = sorted((r for r in all_recs if r["chunked"]), key=lambda r: r["config_id"])
    lines += [f"## Chunked runs ({len(chunked)}; out of memory without chunking at fraction 0.95)", ""]
    rows = [[r["config_id"], r["experiment"], NAMES[r["model"]], r["d"], r["impl"], r["mode"],
             _pow2(r["L"]), r.get("chunk_len", "n/a"), r["status"], _fmt(r.get("median_ms")),
             ((">= " if r["mem_at_cap"] else "") + _fmt(r.get("peak_mem_mb"), ".0f")) if r["status"] == "ok" else ""]
            for r in chunked]
    lines += _table(["id", "experiment", "model", "d", "impl", "mode", "L", "chunk length", "status",
                     "median ms", "peak MB"], rows) + [""]

    lines += ["## Compile times (ms; first call, not included in any timing above)", ""]
    groups = {}
    for r in (r for r in ok if not r["chunked"]):
        groups.setdefault((r["experiment"], r["model"], r["d"] if r["experiment"] == "length" else "all",
                           r["impl"], r["mode"]), []).append(r["compile_ms"])
    rows = [[*k[:2], k[2], *k[3:], len(v), _fmt(min(v), ".0f"), _fmt(float(np.median(v)), ".0f"),
             _fmt(max(v), ".0f")] for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))]
    lines += _table(["experiment", "model", "d", "impl", "mode", "n", "min", "median", "max"], rows)
    lines += [""]

    bad = [r for r in all_recs if r["status"] != "ok"]
    lines += [f"## Configurations not ok ({len(bad)})", ""]
    if bad:
        rows = [[r["config_id"], r["experiment"], NAMES[r["model"]], r["d"], r["impl"], r["mode"],
                 r["L"], r["status"], (r.get("reason") or "").replace("\n", " ").replace("|", "/")[:160]]
                for r in bad]
        lines += _table(["id", "experiment", "model", "d", "impl", "mode", "L", "status", "reason"], rows)
    else:
        lines.append("None.")
    lines += [""]

    lines += ["## Sanity checks (deviations are reported, not corrected)", ""]
    dev = []
    traces = [r for r in all_recs if r.get("n_traces") not in (None, 1)]
    if traces:
        dev.append(f"n_traces != 1 in configurations {[r['config_id'] for r in traces]}")
    spread = [r for r in ok if r["iqr_ms"] > 0.1 * r["median_ms"]]
    if spread:
        worst = sorted(spread, key=lambda r: -r["iqr_ms"] / r["median_ms"])[:10]
        dev.append(f"IQR > 10 % of the median in {len(spread)} configuration(s); worst: " + ", ".join(
            f"{r['config_id']} ({NAMES[r['model']]} {r['impl']} {r['mode']} L={r['L']} d={r['d']}: "
            f"{100 * r['iqr_ms'] / r['median_ms']:.0f} %)" for r in worst))
    for (m, d, impl, mode), s in slopes_a.items():
        if impl == "sequential" and s is not None and not 0.8 <= s <= 1.2:
            dev.append(f"sequential {mode} slope in L of {_label(m, d)} is {s:.2f} (expected ~1)")
        if impl == "parallel" and s is not None:
            s_seq = slopes_a.get((m, d, "sequential", mode))
            if s_seq is not None and s >= s_seq:
                dev.append(f"parallel {mode} slope in L of {_label(m, d)} ({s:.2f}) is not below the "
                           f"sequential one ({s_seq:.2f})")
    par_fb = [slopes_a.get((m, REF_D, "parallel", "FB")) for m in ("nfsm", "mamba", "aussm", "pdssm")]
    if all(s is not None for s in par_fb):
        lines.append(f"- Parallel FB slopes in L at d = {REF_D} (NFSM, Mamba---, AUSSM, PD-SSM): "
                     + ", ".join(f"{s:.2f}" for s in par_fb) + f" (spread {max(par_fb) - min(par_fb):.2f})")
    sb = {m: slopes_b.get((m, "FB")) for m in ("nfsm", "mamba", "aussm", "pdssm", "dense")}
    if all(v is not None for v in sb.values()):
        lines.append("- FB slopes in d: " + ", ".join(f"{NAMES[m]} {v:.2f}" for m, v in sb.items()))
        if not min(sb["nfsm"], sb["pdssm"]) > max(sb["mamba"], sb["aussm"]):
            dev.append("in d, NFSM and PD-SSM do not both grow faster than Mamba--- and AUSSM")
        if not sb["dense"] > max(sb[m] for m in ("nfsm", "mamba", "aussm", "pdssm")):
            dev.append("in d, the dense merges do not grow fastest")
    ratio = {}
    for m in ("nfsm", "mamba", "aussm", "pdssm"):
        f = _median(recs, experiment="length", model=m, d=REF_D, impl="parallel", mode="F", L=REF_L)
        fb = _median(recs, experiment="length", model=m, d=REF_D, impl="parallel", mode="FB", L=REF_L)
        ratio[m] = fb / f if f and fb else None
    if all(v is not None for v in ratio.values()):
        lines.append(f"- FB / F at L = {_pow2(REF_L)}, d = {REF_D}, parallel: "
                     + ", ".join(f"{NAMES[m]} {v:.2f}" for m, v in ratio.items()))
        if ratio["pdssm"] < max(ratio.values()):
            dev.append("PD-SSM's FB / F ratio is not the largest of the four models")
    for m, d in (("nfsm", REF_D), ("pdssm", REF_D)):
        s, n = _slope(*_series(recs, "L", y="peak_mem_mb", experiment="length", model=m, d=d,
                               impl="parallel", mode="FB"), SLOPE_L_MIN)
        lines.append(f"- peak-memory slope in L of {_label(m, d)} (parallel FB, L >= "
                     f"{_pow2(SLOPE_L_MIN)}): {_fmt(s, '.2f')} [{n}] (B L d^2 predicts 1)")
    lines += [""] + ([f"- **Deviation:** {d}" for d in dev] if dev else ["- No deviation flagged."])
    return "\n".join(lines) + "\n"


def main():
    recs = load()
    write_csv(recs)
    with open(SUMMARY_OUT, "w") as f:
        f.write(summary(recs))
    status = Counter(r["status"] for r in recs)
    print(f"[walltime] {dict(status)} -> {CSV_OUT}, {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
