"""Failing length of float32 checkpoints re-scored in lower precision.

    python scripts/make_lowprec_faillen_figure.py [--out DIR]

Reads results/{fsa,tso}/faillen_{fp32,bf16,fp16}/ and writes DIR/lowprec_faillen.{pdf,png}: L_m per
seed and arithmetic (top) and its ratio to fp32 (bottom).
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
from analyse_kws_stream import set_style  # noqa: E402
from nfsm import palette as P  # noqa: E402

PANELS = (("fsa", "S3", r"$S_3$"), ("fsa", "S4", r"$S_4$"), ("fsa", "FF", "FF"),
          ("fsa", "Z16", r"$\mathbb{Z}_{16}$"), ("tso", "question-4", "TSO question-4"))
MODELS = (("nfsm", None, "NFSM"), ("mingru", "fixed", r"minGRU$^-$"),
          ("deltaproduct", "nh1", r"DP $n_h{=}1$"), ("deltaproduct", "nh2", r"DP $n_h{=}2$"),
          ("deltaproduct", "nh4", r"DP $n_h{=}4$"))
FORMATS = (("fp32", "o", 0.0), ("bf16", "^", 0.25), ("fp16", "s", 0.5))   # marker, x offset
LABEL = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


def name(task, cell, rung):
    if cell == "nfsm":
        return f"{task}_{'factored' if task in ('S3', 'S4') else 'regular'}_nfsm_1_free"
    return f"{task}_{rung}_{cell}_4_free"


def load(suite, fmt, nm):
    """{seed: (L_m, holds)}: holds = True when the run passed at its ceiling (L_m is then the ceiling)."""
    p = os.path.join(ROOT, "results", suite, f"faillen_{fmt}", nm + ".json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        rec = json.load(f)
    return {int(s): ((v["cap"], True) if v["failing_length"] is None else (v["failing_length"], False))
            for s, v in rec["seeds"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "lowprec_faillen"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    set_style()
    if mpl.rcParams["text.usetex"]:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}"

    fig, axs = plt.subplots(2, len(PANELS), figsize=(7.2, 3.6), sharey="row",
                            gridspec_kw={"height_ratios": [1.45, 1], "hspace": 0.12, "wspace": 0.08})
    rng = np.random.default_rng(0)
    for j, (suite, task, title) in enumerate(PANELS):
        top, bot = axs[0, j], axs[1, j]
        ticks, labels = [], []
        for i, (cell, rung, mlabel) in enumerate(MODELS):
            data = {f: load(suite, f, name(task, cell, rung)) for f, _, _ in FORMATS}
            if not data["fp32"]:
                continue
            ticks.append(i + 0.25)
            labels.append(mlabel)
            c, e = P.color(cell), P.dark(cell)
            if cell == "deltaproduct":            # three n_h: lighter to darker steps of the same hue
                c = {"nh1": "#DDD2F3", "nh2": P.color(cell), "nh4": "#8E78CC"}[rung]
            for f, mk, dx in FORMATS:
                runs = data[f]
                if not runs:
                    continue
                L = np.array([runs[s][0] for s in sorted(runs)], float)
                held = np.array([runs[s][1] for s in sorted(runs)])
                x = i + dx + rng.uniform(-0.05, 0.05, len(L))
                top.scatter(x, L, s=11, marker=mk, color=c, edgecolor=e, lw=0.5, zorder=3)
                for xi, Li in zip(x[held], L[held]):
                    top.annotate("", (xi, Li * 2.6), (xi, Li), zorder=3,
                                 arrowprops=dict(arrowstyle="-|>", color=e, lw=0.6, mutation_scale=5))
                top.plot([i + dx - 0.1, i + dx + 0.1], [np.median(L)] * 2, color=e, lw=1.2, zorder=4)
                if f != "fp32" and not held.all():
                    ref = data["fp32"]
                    both = [s for s in sorted(runs) if s in ref and not runs[s][1] and not ref[s][1]]
                    r = np.array([runs[s][0] / ref[s][0] for s in both])
                    xb = i + dx + rng.uniform(-0.05, 0.05, len(r))
                    bot.scatter(xb, r, s=11, marker=mk, color=c, edgecolor=e, lw=0.5, zorder=3)
            if all(v[1] for r in data.values() for v in r.values()):
                top.text(i + 0.25, 2.6e6, "holds in\nall formats", ha="center", va="bottom", fontsize=5.5,
                         color=e)
        top.set_title(title, fontsize=7.5, pad=12)
        top.set_yscale("log")
        top.set_ylim(40, 1.5e7)
        top.set_yticks([1e2, 1e3, 1e4, 1e5, 1 << 20],
                       ["$10^2$", "$10^3$", "$10^4$", "$10^5$", "$2^{20}$"])
        top.axhline(1 << 20, color=P.REF, lw=0.5, ls=":", zorder=1)
        bot.set_yscale("log", base=2)
        bot.set_ylim(1 / 32, 2.5)
        bot.set_yticks([1 / 16, 1 / 4, 1, 2], ["1/16", "1/4", "1", "2"])
        bot.axhline(1, color=P.REF, lw=0.6, zorder=1)
        bot.axhspan(0.9, 1.1, color="#EDEDED", lw=0, zorder=0)
        for ax in (top, bot):
            ax.set_xlim(-0.45, len(MODELS) - 0.05)
            ax.minorticks_off()
            ax.set_xticks(ticks, labels if ax is bot else [""] * len(ticks), rotation=40, ha="right",
                          rotation_mode="anchor", fontsize=6)
    axs[0, 0].set_ylabel(r"failing length $L_m$", labelpad=1)
    axs[1, 0].set_ylabel(r"$L_m$ / $L_m$(float32)", labelpad=1)
    handles = [Line2D([], [], ls="", marker=mk, ms=4, mfc="#D0D0D0", mec="black", mew=0.5, label=LABEL[f])
               for f, mk, _ in FORMATS]
    handles.append(Line2D([], [], color="black", lw=1.2, label="median over seeds"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, 1.04))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"lowprec_faillen.{ext}"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
