"""Figures of the wall-clock benchmark, from results/walltime.csv alone (numpy and matplotlib).

    python -m experiments.appendix.walltime.plots [--csv results/walltime.csv] [--out-dir figures]
        [--width 5.5] [--height-cm 10] [--with-forward] [--nfsm-label NFSM]

figures/walltime.pdf, three log-log panels of equal width, one colour per model, one shared
legend: (a) and (b) on the top row with a shared y axis, (c) centred on the bottom row.
  (a) time per call against L (length experiment, d = LENGTH_D, FB): solid parallel, dotted sequential,
      median with the interquartile range of the 10 timed calls as a band; --with-forward adds F
      as thin lines.
  (b) time per call against d at L = 2^12 (state experiment, parallel, FB).
  (c) peak memory (GB) against L (length experiment, d = LENGTH_D, parallel, FB).
Points of configurations that only fit with time chunking (CSV column `chunked`) get a red outer
circle, in every panel. LaTeX typesetting is used when `latex` is on the PATH.
"""

import argparse
import csv
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.appendix.parallelism.drift_plots import _plt
from experiments.appendix.walltime.configs import LENGTH_D

CSV_IN = os.path.join(_ROOT, "results", "walltime.csv")
OUT_DIR = os.path.join(_ROOT, "figures")
MODELS = ["nfsm", "mamba", "aussm", "pdssm"]        # the dense reference stays in the CSV,
                                                    # it is not plotted
COLOR = {"nfsm": "#d94801", "mamba": "#2171b5", "aussm": "#238b45", "pdssm": "#6a51a3",
         "dense": "#525252"}
MARKER = {"nfsm": "o", "mamba": "s", "aussm": "^", "pdssm": "D", "dense": "v"}
STATE_L = 2 ** 12


def load(path=CSV_IN):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r["status"] != "ok":
            continue
        times = np.array([float(r[f"t{i}"]) for i in range(1, 11)])
        out.append({"experiment": r["experiment"], "model": r["model"], "impl": r["impl"],
                    "mode": r["mode"], "L": int(r["L"]), "d": int(r["d"]),
                    "median": float(r["median_ms"]), "q25": float(np.percentile(times, 25)),
                    "q75": float(np.percentile(times, 75)),
                    "mem": float(r["peak_mem_mb"]) if r["peak_mem_mb"] else np.nan,
                    "chunked": r.get("chunked") == "True"})
    return out


def _name(model, nfsm_label):
    import matplotlib.pyplot as plt
    if model == "mamba":
        return r"Mamba$^{\text{---}}$" if plt.rcParams["text.usetex"] else r"Mamba$^{-}$"
    return {"nfsm": nfsm_label, "aussm": "AUSSM", "pdssm": "PD-SSM",
            "dense": "dense merges"}[model]


def _curve(rows, x, **kw):
    pts = sorted((r for r in rows if all(r[k] == v for k, v in kw.items())), key=lambda r: r[x])
    return (np.array([r[x] for r in pts]), np.array([r["median"] for r in pts]),
            np.array([r["q25"] for r in pts]), np.array([r["q75"] for r in pts]),
            np.array([r["mem"] for r in pts]), np.array([r["chunked"] for r in pts], bool))


CHUNKED = dict(linestyle="none", marker="o", markersize=7, markerfacecolor="none",
               markeredgecolor="#e41a1c", markeredgewidth=0.9)


def _circle_chunked(ax, xs, ys, chunked):
    """A red outer circle around the points that only fit with time chunking."""
    if chunked is not None and chunked.any():
        ax.plot(xs[chunked], ys[chunked], zorder=3, **CHUNKED)


def _draw(ax, xs, ys, lo, hi, model, ls="-", lw=1.1, band=True, marker=True, chunked=None):
    if xs.size == 0:
        return
    if band:
        ax.fill_between(xs, lo, hi, color=COLOR[model], alpha=0.2, linewidth=0, zorder=1)
    ax.plot(xs, ys, color=COLOR[model], linestyle=ls, linewidth=lw, zorder=2,
            marker=MARKER[model] if marker else None, markersize=3.2, markeredgewidth=0)
    if marker:
        _circle_chunked(ax, xs, ys, chunked)


def _log2_axis(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2, numticks=20))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"$2^{{{int(round(np.log2(v)))}}}$"))
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.6)


def _thin_ticks(ax, every=2):
    for i, lab in enumerate(ax.xaxis.get_ticklabels()):
        lab.set_visible(i % every == 0)


def _save(fig, plt, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[walltime plot] {path}")


def _memory_panel(ax, rows):
    """Peak memory (GB) against L: length experiment, d = LENGTH_D, parallel, FB."""
    for model in MODELS:
        xs, _, _, _, mem, chk = _curve(rows, "L", experiment="length", model=model, d=LENGTH_D,
                                       impl="parallel", mode="FB")
        keep = np.isfinite(mem) & (mem > 0)
        if keep.any():
            gb = mem[keep] / 1024
            ax.plot(xs[keep], gb, color=COLOR[model], linewidth=1.1, marker=MARKER[model],
                    markersize=3.2, markeredgewidth=0, zorder=2)
            _circle_chunked(ax, xs[keep], gb, chk[keep])


def figure(rows, path, width, height, with_forward, nfsm_label):
    """(a) time against L, (b) time against d, (c) peak memory against L, one shared legend."""
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy
    fig = plt.figure(figsize=(width, height))
    # 2 x 4 equal columns: time panels on columns 0-1 and 2-3, memory centred on columns 1-2, so the
    # three panels have the same width
    grid = fig.add_gridspec(2, 4, wspace=0.1, hspace=0.45, top=0.97)
    ax = fig.add_subplot(grid[0, 0:2])
    bx = fig.add_subplot(grid[0, 2:4], sharey=ax)
    cx = fig.add_subplot(grid[1, 1:3])
    for model in MODELS:
        for impl, ls in (("parallel", "-"), ("sequential", ":")):
            xs, ys, lo, hi, _, chk = _curve(rows, "L", experiment="length", model=model,
                                            d=LENGTH_D, impl=impl, mode="FB")
            _draw(ax, xs, ys, lo, hi, model, ls=ls, chunked=chk)
            if with_forward:
                xs, ys, lo, hi, _, _ = _curve(rows, "L", experiment="length", model=model,
                                              d=LENGTH_D, impl=impl, mode="F")
                _draw(ax, xs, ys, lo, hi, model, ls=ls, lw=0.5, band=False, marker=False)
    for model in MODELS:
        xs, ys, lo, hi, _, chk = _curve(rows, "d", experiment="state", model=model, L=STATE_L,
                                        impl="parallel", mode="FB")
        _draw(bx, xs, ys, lo, hi, model, chunked=chk)
    _memory_panel(cx, rows)
    for a, xlabel, ylabel in ((ax, r"sequence length $L$", "time per call (ms)"),
                              (bx, r"state size $d$", None),
                              (cx, r"sequence length $L$", "peak memory (GB)")):
        _log2_axis(a)
        a.set_xlabel(xlabel)
        if ylabel:
            a.set_ylabel(ylabel)
    bx.tick_params(axis="y", which="both", labelleft=False)
    _thin_ticks(ax)
    _thin_ticks(cx)

    present = [m for m in MODELS if any(r["model"] == m for r in rows)]
    handles = [Line2D([0], [0], color=COLOR[m], marker=MARKER[m], markersize=4, linewidth=1.1,
                      label=_name(m, nfsm_label)) for m in present]
    handles += [Line2D([0], [0], color="k", linestyle="-", linewidth=1.1, label="parallel"),
                Line2D([0], [0], color="k", linestyle=":", linewidth=1.1, label="sequential")]
    if with_forward:
        handles.append(Line2D([0], [0], color="k", linewidth=0.5, label="forward only"))
    if any(r["chunked"] for r in rows):
        handles.append(Line2D([0], [0], label="chunked", **CHUNKED))
    models, styles = handles[:len(present)], handles[len(present):]
    row_pt = 1.7 * plt.rcParams["legend.fontsize"]                 # one legend row, in points
    for row, lift in ((models, row_pt), (styles, 0.0)):           # two rows, each centred
        fig.legend(handles=row, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
                   ncol=len(row), columnspacing=1.2, handlelength=1.8, borderaxespad=0.0,
                   bbox_transform=offset_copy(fig.transFigure, fig=fig, y=lift, units="points"))
    _save(fig, plt, path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", default=CSV_IN)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--width", type=float, default=5.5, help="inches (ICLR text width)")
    p.add_argument("--height-cm", type=float, default=10.0)
    p.add_argument("--with-forward", action="store_true")
    p.add_argument("--nfsm-label", default="NFSM")
    args = p.parse_args()
    figure(load(args.csv), os.path.join(args.out_dir, "walltime.pdf"), args.width,
           args.height_cm / 2.54, args.with_forward, args.nfsm_label)


if __name__ == "__main__":
    main()
