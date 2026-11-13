"""Figure of the head-size search (results/fsa_headsearch/).

    python scripts/make_fsa_headsearch_figure.py [--out DIR]

Writes DIR/headsearch.{pdf,png} (default figures/fsa_headsearch/). A: accuracy against length, one
panel per task, one line per configuration (representable or not) and the paper heads. B: mean
accuracy against head statistics. Representable runs that failed within 100k steps and solved the
task in a 1M-step rerun (results/fsa_headsearch_long/) are drawn with open markers.
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
from nfsm.palette import REF, REF_DARK, STYLE  # noqa: E402

HS = os.environ.get("HEADSEARCH_DIR", os.path.join(ROOT, "results", "fsa_headsearch"))
HS_LONG = HS + "_long"                         # reruns with 10x the step budget
DEFAULT_C = "#F2A65A"                          # pastel orange: the paper default (as in Fig. 8)
DEFAULT_DARK = "#A86416"
YES_C, YES_DARK = STYLE["nfsm"][1], STYLE["nfsm"][2]
TASKS = ("S3", "S4", "Z2", "FF", "Z16")
TASK_TEX = {"S3": r"$S_3$", "S4": r"$S_4$", "Z2": r"$\mathbb{Z}_2$", "FF": "FF", "Z16": r"$\mathbb{Z}_{16}$"}
TASK_MARK = {"S3": "o", "S4": "s", "Z2": "^", "FF": "D", "Z16": "v"}


def _part(root, token):
    p = os.path.join(root, "fsa", "parts", token.replace(",", "_") + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {"lens": np.array(d["lens"]), "acc": 100 * np.array(d["seq_acc"]),
            "exact": (d.get("extraction") or {}).get("exact", False)}


def load():
    """Finished runs; a failed run whose 10x rerun solved it is replaced by the rerun (extended)."""
    cfgs = json.load(open(os.path.join(HS, "configs.json")))["configs"]
    runs = []
    cfgs = [c for c in cfgs if c["task"] in TASKS]
    for c in cfgs:
        r = _part(HS, c["token"])
        if r is None:
            continue
        long = _part(HS_LONG, c["token"])
        if r["acc"][-1] < 100 and long is not None and long["acc"][-1] == 100:
            r, ext = long, True
        else:
            ext = False
        runs.append({**c, **r, "extended": ext})
    return len(cfgs), runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figures", "fsa_headsearch"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    set_style()
    tex = mpl.rcParams["text.usetex"]
    if tex:
        mpl.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}"     # \mathbb{Z}_2
    pct = r"(\%)" if tex else "(%)"

    total, runs = load()
    search = [r for r in runs if not r.get("reference")]
    ref = []                     # the paper's heads: one seed per task, the first that solves it
    for task in TASKS:
        rr = sorted((r for r in runs if r.get("reference") and r["task"] == task),
                    key=lambda r: r["token"])
        if rr:
            ref.append(next((r for r in rr if r["acc"][-1] == 100), rr[0]))

    fig = plt.figure(figsize=(5.5, 2.8))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.42)
    innerA = outer[0].subgridspec(2, 3, hspace=0.55, wspace=0.12)
    innerB = outer[1].subgridspec(2, 2, hspace=0.75, wspace=0.25)
    axA = [fig.add_subplot(innerA[i // 3, i % 3]) for i in range(5)]
    axL = fig.add_subplot(innerA[1, 2])         # the sixth slot holds the legend
    axL.axis("off")
    axB = [fig.add_subplot(innerB[i // 2, i % 2]) for i in range(4)]
    ylim = (-4, 104)

    # A: accuracy against length, one panel per task
    for k, (ax, task) in enumerate(zip(axA, TASKS)):
        for r in sorted((r for r in search if r["task"] == task), key=lambda r: r["representable"]):
            c = YES_C if r["representable"] else REF
            ax.plot(r["lens"], r["acc"], color=c, lw=0.8, marker="o", alpha=0.9,
                    ms=2.6 if r["extended"] else 1.6, mfc="white" if r["extended"] else c,
                    mec=c, mew=0.6, zorder=3 if r["representable"] else 2)
        mine = [r for r in search if r["task"] == task]
        if mine:                       # solved: 100% at the longest test length
            ok = [r["acc"][-1] == 100 for r in mine]
            rep = [o for o, r in zip(ok, mine) if r["representable"]]
            n_ext = sum(o and r["extended"] for o, r in zip(ok, mine))
            blue = (f"{sum(rep) - n_ext}+{n_ext}/{len(rep)}" if n_ext else f"{sum(rep)}/{len(rep)}")
            ax.text(0.5, 0.56, blue, transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=7, color=YES_DARK, zorder=12)
            ax.text(0.5, 0.54, f"{sum(ok)}/{len(ok)}", transform=ax.transAxes, ha="center",
                    va="top", fontsize=7, color=REF_DARK, zorder=12)
        rr = [r for r in ref if r["task"] == task]
        if rr:
            ax.plot(rr[0]["lens"], rr[0]["acc"], color=DEFAULT_C, lw=1.5, ls=(0, (3, 1.5)), zorder=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 256, 1024, 4096, 16384], ["64", "", "1k", "", "16k"])
        ax.set_xlim(45, 23000)
        ax.minorticks_off()
        ax.set_ylim(*ylim)
        ax.set_yticks([0, 50, 100])
        ax.set_title(TASK_TEX[task] if tex else task, fontsize=7.5, pad=2)
        if k % 3:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(f"seq. acc. {pct}", labelpad=1)
        if k >= 2:
            ax.set_xlabel("test length", labelpad=1)
        else:
            ax.set_xticklabels([])

    # B: mean accuracy against head statistics
    feats = (("number of heads", lambda h: len(h), False, (0.5, 5.5), [1, 2, 3, 4, 5], None),
             ("total states", lambda h: sum(h), True, (1.6, 100), [2, 8, 32], None),
             ("largest head", lambda h: max(h), True, (1.6, 20), [2, 4, 8, 16], None),
             ("joint states", lambda h: float(np.prod(h)), True, (1.6, 2 ** 21), [2, 2 ** 7, 2 ** 14, 2 ** 20],
              ["2", "$2^{7}$", "$2^{14}$", "$2^{20}$"]))
    for k, (ax, (label, f, logx, xlim, ticks, ticklabels)) in enumerate(zip(axB, feats)):
        for r in search:
            c, e = (YES_C, YES_DARK) if r["representable"] else (REF, REF_DARK)
            ax.scatter(f(r["heads"]), r["acc"].mean(), s=9, marker=TASK_MARK[r["task"]],
                       color="white" if r["extended"] else c, edgecolor=e, lw=0.6 if r["extended"] else 0.4,
                       zorder=5 if r["extended"] else 3)
        for task in TASKS:            # the paper's heads: the single seed kept above
            rr = [r for r in ref if r["task"] == task]
            if rr:
                ax.scatter(f(rr[0]["heads"]), rr[0]["acc"].mean(), s=16,
                           marker=TASK_MARK[task], color=DEFAULT_C, edgecolor=DEFAULT_DARK, lw=0.5,
                           zorder=4)
        if logx:
            ax.set_xscale("log", base=2)
        ax.minorticks_off()
        ax.set_xlim(*xlim)
        ax.set_xticks(ticks, ticklabels or [str(t) for t in ticks])
        ax.set_xlabel(label, labelpad=1)
        ax.set_ylim(*ylim)
        ax.set_yticks([0, 50, 100])
        if k % 2:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(f"mean acc. {pct}", labelpad=1)

    handles = [Line2D([], [], color=YES_C, lw=1, label="representable"),
               Line2D([], [], color=REF, lw=1, label="not representable"),
               Line2D([], [], color=DEFAULT_C, lw=1.5, ls=(0, (3, 1.5)), label="paper heads")]
    if any(r["extended"] for r in search):
        handles.append(Line2D([], [], color=YES_C, lw=0.8, marker="o", ms=2.6, mfc="white", mec=YES_C,
                              mew=0.6, label="solved with 10x steps"))
    handles += [Line2D([], [], ls="", marker=TASK_MARK[t], ms=3.5, mfc="#D0D0D0", mec="black", mew=0.5,
                       label=TASK_TEX[t] if tex else t) for t in TASKS]
    axL.legend(handles=handles, loc="center left", ncol=1, frameon=False, fontsize=5.8, handlelength=1.4,
               labelspacing=0.3, handletextpad=0.4, borderaxespad=0, bbox_to_anchor=(-0.12, 0.5))

    fig.canvas.draw()
    top = axA[0].get_position().y1 + 0.05
    for ax, tag, dx in ((axA[0], "A", 0.07), (axB[0], "B", 0.07)):
        fig.text(ax.get_position().x0 - dx, top, tag, fontsize=9, fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"headsearch.{ext}"))
    print(f"wrote {args.out}: {len(search)} search runs, {len(ref)} reference runs of {total}")


if __name__ == "__main__":
    main()
