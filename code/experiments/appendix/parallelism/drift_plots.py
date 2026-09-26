"""Figures of the scan drift (drift.py), from its JSONs alone: numpy and matplotlib, no JAX.

    python -m experiments.appendix.parallelism.drift_plots [--dtypes float32 float16]
        [--dir results/parallelism/drift] [--band seeds|words] [--logx] [--width 6.75]

One figure per dtype, <dir>/figures/drift_<dtype>.pdf. Left (3/4 of the width): e_t against t,
one colour and line style per baseline; the line is the mean over seeds of the batch mean, the
band is the min-max over seeds of the batch mean (--band seeds) or over every word of every seed
(--band words). Right (1/4): per baseline, a box plot of the batch mean at every step and seed,
whiskers at the min and max. The y axis is shared and logarithmic; exact agreement (e_t = 0) is
left out of it. With both dtypes, drift_combined.pdf also puts them on the same axes, float32 in
a lighter shade of each baseline's colour, and pairs the boxes. LaTeX typesetting is used when
`latex` is on the PATH.
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DRIFT_DIR = os.path.join(_ROOT, "results", "parallelism", "drift")
MODELS = ["mamba", "aussm", "pdssm"]
STYLE = {                             # label, colour (the dark end of plots.py's ramps), line style
    "mamba": ("Mamba", "#2171b5", "--"),
    "aussm": ("AUSSM", "#238b45", "-."),
    "pdssm": ("PD-SSM", "#6a51a3", "-"),
}
RHO = {"mamba": r"$\rho<1$", "aussm": r"$\rho=1$", "pdssm": r"$\rho<1$"}   # spectral radius
DTYPE_LABEL = {"float32": "fp32", "float16": "fp16"}


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
    """The paper's name: Mamba with negative eigenvalues is Mamba with a superscript em dash."""
    if model != "mamba":
        return STYLE[model][0]
    import matplotlib.pyplot as plt
    return r"Mamba$^{\text{---}}$" if plt.rcParams["text.usetex"] else r"Mamba$^{-}$"


def _arr(v):
    return np.array([np.nan if x is None else x for x in v], np.float64)


def load(directory, dtype):
    """{model: {"mean": [S, T], "min": [S, T], "max": [S, T]}} of the per-seed JSONs."""
    out = {}
    for model in MODELS:
        recs = []
        for path in sorted(glob.glob(os.path.join(directory, dtype, f"{model}_seed*.json"))):
            with open(path) as f:
                recs.append(json.load(f))
        if recs:
            out[model] = {k: np.stack([_arr(r[f"diff_{k}"]) for r in recs])
                          for k in ("mean", "min", "max")}
            out[model]["seeds"] = [r["seed"] for r in recs]
    return out


def _positive(v):
    return np.where(np.isfinite(v) & (v > 0), v, np.nan)


def _shade(color, dtype, combined):
    """The baseline's colour; lightened for float32 when both dtypes share the figure."""
    from matplotlib.colors import to_rgb
    if not combined or dtype != "float32":
        return color
    return tuple(1 - 0.5 * (1 - c) for c in to_rgb(color))


def _curve(d, band):
    """(line, lo, hi) over t: mean over seeds of the batch mean, and its band."""
    with np.errstate(all="ignore"):
        line = np.nanmean(d["mean"], 0)
        if band == "seeds":
            lo, hi = np.nanmin(d["mean"], 0), np.nanmax(d["mean"], 0)
        else:
            lo, hi = np.nanmin(d["min"], 0), np.nanmax(d["max"], 0)
    return line, np.where(np.isfinite(_positive(lo)), lo, line), hi


def make_figure(data_by_dtype, out_path, band="seeds", logx=False, width=6.75, height=2.3):
    """`data_by_dtype` {dtype: load(...)}; one dtype, or float32 and float16 on the same axes
    (colour and line style per baseline, lighter shade for float32; paired boxes)."""
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(width, height), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 1], "wspace": 0.04})
    dtypes = list(data_by_dtype)
    combined = len(dtypes) > 1
    models = [m for m in MODELS if any(m in data_by_dtype[dt] for dt in dtypes)]
    box_w = 0.6 / len(dtypes)
    T_max = 1
    for j, dtype in enumerate(dtypes):
        for i, model in enumerate(models):
            d = data_by_dtype[dtype].get(model)
            if d is None:
                continue
            _, base, ls = STYLE[model]
            color = _shade(base, dtype, combined)
            line, lo, hi = _curve(d, band)
            t = np.arange(1, line.size + 1)
            T_max = max(T_max, line.size)
            ax.fill_between(t, _positive(lo), _positive(hi), color=color, alpha=0.18,
                            linewidth=0, rasterized=True, zorder=1)
            ax.plot(t, _positive(line), color=color, linestyle=ls, linewidth=0.8,
                    rasterized=True, zorder=2)
            vals = _positive(d["mean"].ravel())
            pos = i + 1 + (j - (len(dtypes) - 1) / 2) * box_w
            bp = bx.boxplot([vals[np.isfinite(vals)]], positions=[pos], whis=(0, 100),
                            showfliers=False, widths=box_w * 0.85, patch_artist=True,
                            medianprops={"color": "k", "linewidth": 0.8},
                            whiskerprops={"linewidth": 0.7, "color": color},
                            capprops={"linewidth": 0.7, "color": color})
            bp["boxes"][0].set(facecolor=color, alpha=0.6, edgecolor=base, linewidth=0.8)
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(1, T_max)
    ax.set_xlabel(r"step $t$")
    ax.set_ylabel(r"$\|h_t^{\mathrm{par}} - h_t^{\mathrm{seq}}\|_\infty$")
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, alpha=0.6)

    bx.set_xticks(range(1, len(models) + 1), [_name(m) for m in models], rotation=30,
                  ha="right", rotation_mode="anchor", fontsize=7.5)
    bx.set_xlim(0.5, len(models) + 0.5)
    bx.tick_params(axis="y", which="both", left=False)
    bx.grid(True, axis="y", which="major", linestyle=":", linewidth=0.5, alpha=0.6)

    handles = [Line2D([0], [0], color=STYLE[m][1], linestyle=STYLE[m][2], linewidth=1.2,
                      label=f"{_name(m)} ({RHO[m]})") for m in models]
    if combined:
        handles += [Line2D([0], [0], color=_shade("#404040", dt, True), linewidth=4,
                           label=DTYPE_LABEL.get(dt, dt)) for dt in dtypes]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              ncol=len(handles), borderaxespad=0.2, columnspacing=1.2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[drift plot] {'+'.join(DTYPE_LABEL.get(dt, dt) for dt in dtypes)}: "
          f"{', '.join(models)} -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=DRIFT_DIR)
    p.add_argument("--dtypes", nargs="+", default=["float32", "float16"])
    p.add_argument("--band", choices=["seeds", "words"], default="seeds")
    p.add_argument("--logx", action="store_true")
    p.add_argument("--width", type=float, default=6.75)
    p.add_argument("--height", type=float, default=2.3)
    args = p.parse_args()
    data = {}
    for dtype in args.dtypes:
        data[dtype] = load(args.dir, dtype)
        if not data[dtype]:
            print(f"[drift plot] no JSON under {os.path.join(args.dir, dtype)}")
            del data[dtype]
    kw = dict(band=args.band, logx=args.logx, width=args.width, height=args.height)
    fig_dir = os.path.join(args.dir, "figures")
    for dtype in data:
        make_figure({dtype: data[dtype]}, os.path.join(fig_dir, f"drift_{dtype}.pdf"), **kw)
    if len(data) > 1:
        make_figure(data, os.path.join(fig_dir, "drift_combined.pdf"), **kw)


if __name__ == "__main__":
    main()
