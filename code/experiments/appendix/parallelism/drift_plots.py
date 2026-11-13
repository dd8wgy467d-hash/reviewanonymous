"""Figures of the scan drift, from the JSONs of drift.py (numpy and matplotlib only).

    python -m experiments.appendix.parallelism.drift_plots [--dtypes float32 float16]
        [--dir results/parallelism/drift] [--band seeds|words] [--logx] [--width 6.75]

e_t against t, mean over seeds with a min-max band, averaged over step pairs and bins of t.
DeltaProduct is drawn from `--deltaproduct` (default deltaproduct-h1, one head of 32).
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DRIFT_DIR = os.path.join(_ROOT, "results", "parallelism", "drift")
MODELS = ["mamba", "aussm", "pdssm", "mingru", "deltaproduct", "nfsm"]
import sys                            # noqa: E402
sys.path.insert(0, _ROOT)
from nfsm import palette as P         # noqa: E402  one style per model, shared by every figure
NAMES = {"mamba": "Mamba", "aussm": "AUSSM", "pdssm": "PD-SSM", "mingru": "minGRU",
         "deltaproduct": "DeltaProduct", "nfsm": "Neural-FSM"}
N_MARKS = 9                           # markers per curve, staggered between models so they do not stack
EXACT = ("nfsm",)                     # e_t = 0 at every step: drawn on the zero strip below the log axis
RHO = {"mamba": r"$\rho<1$", "aussm": r"$\rho=1$", "pdssm": r"$\rho<1$",       # spectral radius
       "mingru": r"$\rho<1$", "deltaproduct": r"$\rho=1$", "nfsm": r"$e_t = 0$"}
DTYPE_LABEL = {"float32": "fp32", "float16": "fp16"}
DTYPE_ORDER = ["float16", "float32"]                   # left to right
# drawing order, bottom to top; Mamba, which sits just above minGRU in fp32, goes last
Z_ORDER = ["mingru", "deltaproduct", "pdssm", "aussm", "mamba"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rc = {"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
          "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42}
    if shutil.which("latex"):
        rc.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}"})
    else:                             # Computer Modern without a TeX install
        rc.update({"mathtext.fontset": "cm", "font.serif": ["cmr10"],
                   "axes.formatter.use_mathtext": True, "axes.unicode_minus": False})
    plt.rcParams.update(rc)
    return plt


def _name(model):
    """The paper's name: Mamba and minGRU with negative eigenvalues carry a superscript em dash."""
    if model not in ("mamba", "mingru"):
        return NAMES[model]
    import matplotlib.pyplot as plt
    return NAMES[model] + (r"$^{\text{---}}$" if plt.rcParams["text.usetex"] else r"$^{-}$")


def _marks(n, k):
    """Indices of N_MARKS markers along a curve of n points, shifted by model k."""
    step = max(n // N_MARKS, 1)
    return list(range((k * step) // len(MODELS) + step // 2, n, step))


def _arr(v):
    return np.array([np.nan if x is None else x for x in v], np.float64)


DELTAPRODUCT_VARIANT = "deltaproduct-h1"               # the drift.py run drawn as "deltaproduct"


def load(directory, dtype):
    """{model: {"mean": [S, T], "min": [S, T], "max": [S, T]}} of the per-seed JSONs."""
    out = {}
    for model in MODELS:
        recs = []
        run = DELTAPRODUCT_VARIANT if model == "deltaproduct" else model
        for path in sorted(glob.glob(os.path.join(directory, dtype, f"{run}_seed*.json"))):
            with open(path) as f:
                recs.append(json.load(f))
        if recs:
            out[model] = {k: np.stack([_arr(r[f"diff_{k}"]) for r in recs])
                          for k in ("mean", "min", "max")}
            out[model]["seeds"] = [r["seed"] for r in recs]
    return out


def _positive(v):
    return np.where(np.isfinite(v) & (v > 0), v, np.nan)


def _pairs(v):
    """[S, T] -> [S, T // 2]: mean over step pairs (1, 2), (3, 4), ... (odd and even steps round differently)."""
    S, T = v.shape
    with np.errstate(all="ignore"):
        return np.nanmean(v[:, :T - T % 2].reshape(S, T // 2, 2), -1)


def _bin(t, *series, bins=256, logx=False):
    """Average every series over `bins` bins of t (linear or log); empty bins dropped."""
    edges = (np.geomspace(t[0], t[-1] + 1, bins + 1) if logx else np.linspace(t[0], t[-1] + 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, bins - 1)
    keep = np.bincount(idx, minlength=bins) > 0
    with np.errstate(all="ignore"):
        out = [np.array([np.nanmean(v[idx == b]) for b in range(bins)])[keep] for v in (t, *series)]
    return out


def _curve(d, band, pair=True):
    """(t, line, lo, hi): mean over seeds of the batch mean and its band."""
    f = _pairs if pair else (lambda v: v)
    with np.errstate(all="ignore"):
        mean = f(d["mean"])
        line = np.nanmean(mean, 0)
        if band == "seeds":
            lo, hi = np.nanmin(mean, 0), np.nanmax(mean, 0)
        else:
            lo, hi = np.nanmin(f(d["min"]), 0), np.nanmax(f(d["max"]), 0)
    t = 2 * np.arange(line.size) + 1.5 if pair else np.arange(1, line.size + 1)
    return t, line, np.where(np.isfinite(_positive(lo)), lo, line), hi


def make_figure(data_by_dtype, out_path, band="seeds", logx=False, width=6.75, height=2.3,
                pair=True, bins=256):
    """One panel per dtype (float16 left, float32 right), shared y axis."""
    plt = _plt()
    from matplotlib.lines import Line2D
    dtypes = sorted(data_by_dtype, key=lambda dt: DTYPE_ORDER.index(dt) if dt in DTYPE_ORDER else 9)
    models = [m for m in MODELS if any(m in data_by_dtype[dt] for dt in dtypes)]
    exact = [m for m in models if m in EXACT]
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2 if exact else 1, len(dtypes), wspace=0.06, hspace=0.06,
                          height_ratios=[1, 0.07] if exact else [1])
    axes, zaxes = [], []
    for j in range(len(dtypes)):
        axes.append(fig.add_subplot(gs[0, j], sharey=axes[0] if axes else None))
        if exact:
            zaxes.append(fig.add_subplot(gs[1, j], sharex=axes[j]))
    for j, (ax, dtype) in enumerate(zip(axes, dtypes)):
        T_max = 1
        for model in models:
            d = data_by_dtype[dtype].get(model)
            if d is None:
                continue
            k = MODELS.index(model)
            t, line, lo, hi = _curve(d, band, pair)
            if bins and model not in EXACT:
                t, line, lo, hi = _bin(t, line, lo, hi, bins=bins, logx=logx)
            T_max = max(T_max, d["mean"].shape[1])
            if model in EXACT:
                zero = np.nanmax(d["max"], 0) == 0          # every word of every seed agrees exactly
                tz = np.flatnonzero(zero) + 1
                zaxes[j].plot(tz, np.zeros(zero.sum()), rasterized=True,
                              **P.line_kw(model, markevery=_marks(tz.size, k)))
                if not zero.all():
                    print(f"[drift plot] {model} {dtype}: {int((~zero).sum())} steps with e_t > 0")
                continue
            z = 2 + 2 * (Z_ORDER.index(model) if model in Z_ORDER else len(Z_ORDER))
            ax.fill_between(t, _positive(lo), _positive(hi), color=P.color(model), alpha=0.25,
                            linewidth=0, rasterized=True, zorder=z)
            ax.plot(t, _positive(line), rasterized=True, zorder=z + 1,
                    **P.line_kw(model, lw=1.1, markevery=_marks(t.size, k)))
        ax.set_yscale("log")
        if logx:
            ax.set_xscale("log")
        ax.set_xlim(1, T_max)
        ax.set_title(DTYPE_LABEL.get(dtype, dtype), fontsize=9, pad=3)
        ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.6)
        if j:
            ax.tick_params(axis="y", which="both", labelleft=False)
        if not exact:
            ax.set_xlabel(r"step $t$")
            continue
        # the zero strip: a broken axis under the log axis, holding only e_t = 0
        zx = zaxes[j]
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        zx.spines["top"].set_visible(False)
        zx.set_ylim(-1, 1)
        zx.set_yticks([0], ["$0$"] if j == 0 else [""])
        zx.tick_params(axis="y", which="minor", left=False)
        zx.set_xlabel(r"step $t$")
        zx.grid(True, axis="x", which="major", linestyle=":", linewidth=0.5, alpha=0.6)
        kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=5, linestyle="none", color="k",
                  mec="k", mew=0.8, clip_on=False)
        ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kw)
        zx.plot([0, 1], [1, 1], transform=zx.transAxes, **kw)
    axes[0].set_ylabel(r"$\|h_t^{\mathrm{par}} - h_t^{\mathrm{seq}}\|_\infty$")

    # legend rows: the rho < 1 baselines, then the rho = 1 ones (matplotlib fills by column)
    rows = [[m for m in models if RHO[m] == RHO["mamba"]],
            [m for m in models if RHO[m] == RHO["aussm"]] + exact]        # the NFSM ends row 2
    rows = [r for r in rows if r]
    ncol = max(map(len, rows))
    blank = Line2D([], [], linestyle="none", label=" ")
    grid = [[Line2D([0], [0], **P.line_kw(m, lw=1.2), label=f"{_name(m)} ({RHO[m]})") for m in r] + [blank] * (ncol - len(r))
            for r in rows]
    handles = [grid[i][j] for j in range(ncol) for i in range(len(grid))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.97), frameon=False,
               ncol=ncol, columnspacing=1.2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[drift plot] {'+'.join(DTYPE_LABEL.get(dt, dt) for dt in dtypes)}: "
          f"{', '.join(models)} -> {out_path}")


def main():
    global DELTAPRODUCT_VARIANT
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=DRIFT_DIR)
    p.add_argument("--dtypes", nargs="+", default=["float32", "float16"])
    p.add_argument("--band", choices=["seeds", "words"], default="seeds")
    p.add_argument("--logx", action="store_true")
    p.add_argument("--per-step", action="store_true",
                   help="plot every step instead of the mean over consecutive step pairs")
    p.add_argument("--deltaproduct", default=DELTAPRODUCT_VARIANT,
                   help="drift.py run drawn as DeltaProduct (deltaproduct-h1: one head; deltaproduct: 8 heads)")
    p.add_argument("--bins", type=int, default=256,
                   help="average each curve over this many bins of t (0: no binning)")
    p.add_argument("--width", type=float, default=6.75)
    p.add_argument("--height", type=float, default=2.3)
    args = p.parse_args()
    DELTAPRODUCT_VARIANT = args.deltaproduct
    data = {}
    for dtype in args.dtypes:
        data[dtype] = load(args.dir, dtype)
        if not data[dtype]:
            print(f"[drift plot] no JSON under {os.path.join(args.dir, dtype)}")
            del data[dtype]
    kw = dict(band=args.band, logx=args.logx, width=args.width, height=args.height,
              pair=not args.per_step, bins=args.bins)
    fig_dir = os.path.join(args.dir, "figures")
    if len(data) > 1:
        make_figure(data, os.path.join(fig_dir, "drift_combined.pdf"), **kw)
    else:
        for dtype in data:
            make_figure({dtype: data[dtype]}, os.path.join(fig_dir, f"drift_{dtype}.pdf"), **kw)


if __name__ == "__main__":
    main()
