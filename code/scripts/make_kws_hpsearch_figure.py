"""Figure of the hyperparameter search on KWS Stream (results/kws_hpsearch/).

    python scripts/make_kws_hpsearch_figure.py [--out DIR]

Writes DIR/hpsearch.{pdf,png} (default figures/kws_hpsearch/). A: accuracy against length, one line
per configuration, and the default (seeds 42-44). B: mean accuracy against each hyperparameter.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
from analyse_kws_stream import FPS, REF_DARK, set_style  # noqa: E402

DEFAULT_C = "#F2A65A"                          # pastel orange: the paper default
HP = os.path.join(ROOT, "results", "kws_hpsearch")
DEFAULT = {"lr": 1e-3, "batch": 64, "nu": 0.5, "wd": 1e-4}
SPACE = {"lr": (1e-4, 3e-3), "batch": (16, 128), "nu": (0.25, 2.0), "wd": (1e-6, 1e-2)}
LABELS = {"lr": "learning rate", "batch": "batch size", "nu": r"sampling temperature $\nu$",
          "wd": "weight decay"}


def load():
    cfgs = json.load(open(os.path.join(HP, "configs.json")))["configs"]
    runs = []
    for c in cfgs:
        part = glob.glob(os.path.join(HP, c["id"], "kws_stream", "parts", "*.json"))[0]
        d = json.load(open(part))
        runs.append({**c, "lens": np.array(d["lens"]), "acc": 100 * np.array(d["acc"])})
    ref = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HP, "ref_filtered", "*.json")))]
    ref_acc = 100 * np.array([r["acc"] for r in ref])
    return runs, np.array(ref[0]["lens"]), ref_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "kws_hpsearch"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    set_style()

    runs, ref_lens, ref_acc = load()
    from nfsm.palette import nfsm_ramp
    cmap = nfsm_ramp()                         # shades of the NFSM hue, light: low learning rate
    norm = LogNorm(*SPACE["lr"])
    ylim = (64, 97)

    fig = plt.figure(figsize=(5.5, 2.05))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.22)
    axA = fig.add_subplot(outer[0])
    inner = outer[1].subgridspec(2, 2, hspace=0.8, wspace=0.12)
    axB = [fig.add_subplot(inner[i // 2, i % 2]) for i in range(4)]

    # A: accuracy against length
    for r in sorted(runs, key=lambda r: r["lr"]):
        axA.plot(r["lens"] / FPS, r["acc"], color=cmap(norm(r["lr"])), lw=0.9, marker="o", ms=1.8)
    x = ref_lens / FPS
    axA.fill_between(x, ref_acc.min(0), ref_acc.max(0), color=DEFAULT_C, alpha=0.35, lw=0, zorder=9)
    axA.plot(x, np.median(ref_acc, 0), color=DEFAULT_C, lw=1.6, ls=(0, (3, 1.5)), zorder=10)
    axA.set_xscale("log", base=2)
    axA.set_xticks(x, ["1", "2", "4", "8", "16"])
    axA.minorticks_off()
    axA.set_xlabel("stream length (s)")
    axA.set_ylabel(r"test accuracy (\%)" if mpl.rcParams["text.usetex"] else "test accuracy (%)")
    axA.set_ylim(*ylim)

    # B: mean accuracy against each hyperparameter
    mean = np.array([r["acc"].mean() for r in runs])
    for k, (ax, name) in enumerate(zip(axB, ("lr", "batch", "nu", "wd"))):
        v = np.array([r[name] for r in runs])
        ax.axvline(DEFAULT[name], color=DEFAULT_C, lw=1.4, ls=(0, (3, 1.5)), zorder=2)
        ax.scatter(v, mean, s=9, c=[cmap(norm(r["lr"])) for r in runs], edgecolor="black", lw=0.3,
                   zorder=3)
        ax.set_xscale("log", base=2 if name == "batch" else 10)
        lo, hi = SPACE[name]
        ax.set_xlim(lo / 1.3, hi * 1.3)
        if name == "batch":
            ax.set_xticks([16, 32, 64, 128], ["16", "32", "64", "128"])
        elif name == "nu":
            ax.set_xticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1", "2"])
        ax.minorticks_off()
        ax.set_xlabel(LABELS[name], labelpad=1)
        ax.set_ylim(*ylim)
        ax.set_yticks([70, 80, 90])
        if k % 2:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(r"mean acc.\ (\%)" if mpl.rcParams["text.usetex"] else "mean acc. (%)",
                          labelpad=2)

    fig.canvas.draw()
    top = axA.get_position().y1 + 0.03
    for ax, tag, dx in ((axA, "A", 0.075), (axB[0], "B", 0.09)):
        fig.text(ax.get_position().x0 - dx, top, tag, fontsize=9, fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"hpsearch.{ext}"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
