"""Figures of the KWS Stream suite, from results/kws_stream/parts/*.json.

    python scripts/analyse_kws_stream.py [--out DIR]

Writes to DIR (default figures/kws_stream/) accuracy_vs_length.{pdf,png}, digit_accuracy_around_word.{pdf,png}
(accuracy against the noise before and after the digit) and summary.txt. Curves: median over the
three seeds, band: range. Sweeps on filtered test noise (parts_ood/) replace the default ones when
every seed has them.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FPS = 100                                     # MFCC frames per second
from nfsm import palette as P                  # noqa: E402  one style per model, shared by every figure
MODELS = (                                    # (config, label, fill, stroke): pastel, dark step (nfsm/palette.py)
    ("nfsm-64x4,4,last", r"Neural-FSM", P.color("nfsm"), P.dark("nfsm")),
    ("mingru-256,4,last", r"minGRU$^-$", P.color("mingru"), P.dark("mingru")),
    ("mamba-256,4,last", r"Mamba$^-$", P.color("mamba"), P.dark("mamba")),
    ("aussm-256,4,last", r"AUSSM", P.color("aussm"), P.dark("aussm")),
    ("deltaproduct-256,4,last", r"DeltaProduct", P.color("deltaproduct"), P.dark("deltaproduct")),
    # PD-SSM evaluated with the test-time decay floor tau = 2048 (token pdssm-256f2048, same checkpoint
    # as pdssm-256); left out until every sweep reaches 819200
    # ("pdssm-256f2048,4,last", r"PD-SSM", P.color("pdssm"), P.dark("pdssm")),
)
KEY = {r"Neural-FSM": "nfsm", r"minGRU$^-$": "mingru", r"Mamba$^-$": "mamba", r"AUSSM": "aussm",
       r"DeltaProduct": "deltaproduct", r"PD-SSM": "pdssm"}   # label -> palette key
REF = "#9A9A9A"
REF_DARK = "#5F5F5F"
CHANCE = 100 / 11                             # % of streams: one of 11 answers (10 digits, none)
BELOW_CHANCE = "#C0392B"                      # red hatching between 0 and chance (asked for, not a series)
LS_ND, LS_KDEF, LS_COLL = "-", (0, (4, 1.6)), (0, (1, 1.3))   # theory patterns (schematic)
# each model drawn in the line style of the pattern it follows: NFSM non-definite, the contracting
# baselines k-definite, the others collapsing
MODEL_LS = {"nfsm": LS_ND, "mingru": LS_KDEF, "mamba": LS_KDEF, "aussm": LS_COLL,
            "deltaproduct": LS_COLL, "pdssm": LS_KDEF}
TRAIN_LS = (0, (3, 2))                        # dashes: training-length lines up to the strip's arrow
TRAIN_MAX, WORD = 400, 101                    # longest training stream and one digit clip (frames)


def set_style():
    import matplotlib as mpl
    usetex = shutil.which("latex") is not None
    mpl.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if usetex else ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 7.5, "font.size": 7.5, "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.8, "lines.linewidth": 1.3,
        "xtick.direction": "out", "ytick.direction": "out",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def load(parts_dir):
    """{config: [part records]}, with the filtered-noise sweeps swapped in where complete (listed in "ood_keys")."""
    groups = {}
    for p in sorted(glob.glob(os.path.join(parts_dir, "*.json"))):
        with open(p) as f:
            rec = json.load(f)
        groups.setdefault(rec["token"].rsplit(",", 1)[0], []).append(rec)
    ood = {}
    for p in sorted(glob.glob(os.path.join(os.path.dirname(parts_dir), "parts_ood", "*.json"))):
        with open(p) as f:
            r = json.load(f)
        ood[r["token"]] = r
    for recs in groups.values():
        for key in ("prefix", "suffix"):
            alt = [ood.get(r["token"], {}).get(key) for r in recs]
            if all(a and set(r.get(key, {}).get("lens", [])) <= set(a["lens"]) for a, r in zip(alt, recs)):
                for a, r in zip(alt, recs):
                    r[key] = a
                    r.setdefault("ood_keys", []).append(key)
    return groups


def by_length(recs):
    """(lengths in frames, [seeds, lengths] accuracy over all streams), lengths common to all seeds."""
    lens = [T for T in recs[0]["lens"] if all(T in r["lens"] for r in recs)]
    return lens, np.array([[r["acc"][r["lens"].index(T)] for T in lens] for r in recs])


def by_hold(recs, edges):
    """[seeds, bins] digit-stream accuracy per hold bin, pooled over lengths; nan for an empty bin."""
    out = []
    for r in recs:
        lens = [T for T in r["lens"] if str(T) in r["hold"]]     # preliminary parts may lack holds
        if not lens:
            out.append([np.nan] * (len(edges) - 1))
            continue
        hold = np.concatenate([r["hold"][str(T)] for T in lens])
        ok = np.concatenate([r["correct"][str(T)] for T in lens])
        out.append([ok[(hold >= lo) & (hold < hi)].mean() if ((hold >= lo) & (hold < hi)).any()
                    else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
    return np.array(out)


def placed(recs, key):
    """(seeds, noise around the word in s, accuracy [seeds, n]) of the "prefix" or "suffix" sweeps, or None."""
    recs = [r for r in recs if r.get(key, {}).get("lens")]
    lens = [T for T in (recs[0][key]["lens"] if recs else []) if T > WORD
            and all(T in r[key]["lens"] for r in recs)]
    if not lens:
        return None
    acc = np.array([[r[key]["acc"][r[key]["lens"].index(T)] for T in lens] for r in recs])
    return len(recs), (np.array(lens) - WORD) / FPS, acc


def hold_edges(groups):
    """Doubling bins 0-100, 100-200, ... up to the longest tested length."""
    longest = max(max(r["lens"]) for recs in groups.values() for r in recs)
    edges = [0, 100]
    while edges[-1] < longest:
        edges.append(2 * edges[-1])
    return np.array(edges)


def place_labels(ax, ends, chance):
    """Label each curve right of its last point, labels at least GAP apart and off the chance line."""
    GAP = 7.0
    ends = sorted(ends, key=lambda e: e[0])
    ys = [e[0] for e in ends]
    for i in range(len(ys)):
        if chance is not None and abs(ys[i] - 100 * chance) < GAP / 2:
            ys[i] = 100 * chance + (GAP / 2 if ys[i] >= 100 * chance else -GAP / 2)
        if i and ys[i] - ys[i - 1] < GAP:
            ys[i] = ys[i - 1] + GAP
    x_text = 1.15 * max(e[1] for e in ends)         # one column right of the longest curve
    for (y, x, label, stroke), yl in zip(ends, ys):
        ax.annotate(label, (x, y), xytext=(x_text, yl), textcoords="data", color=stroke,
                    fontsize=7, va="center", ha="left", annotation_clip=False)


def below_chance(ax, y):
    """Light red hatching from 0 up to chance (y in %), behind the curves."""
    import matplotlib as mpl
    with mpl.rc_context({"hatch.linewidth": 0.4, "hatch.color": BELOW_CHANCE}):
        ax.axhspan(0, y, facecolor="none", edgecolor=BELOW_CHANCE, hatch="////", lw=0, alpha=0.45,
                   zorder=0)


def chance_line(ax, y):
    """Dashed horizontal line at chance (y in %), labelled at its left end, red hatching below."""
    below_chance(ax, y)
    ax.axhline(y, color=REF, lw=0.8, ls="--")
    ax.annotate("chance", (0.01, y), xycoords=("axes fraction", "data"), xytext=(0, 1.5),
                textcoords="offset points", ha="left", va="bottom", fontsize=6.5)


def log_ticks(ticks):
    """Drop 4096 when 8192 is also a tick: the two labels would touch on a log axis."""
    return [t for t in ticks if not (t == 4096 and 8192 in ticks)]


def train_line(ax, x, text, side, ls=":"):
    """Vertical line at x (dotted by default) with `text` written along it, on the given side."""
    ax.axvline(x, color=REF, lw=0.8, ls=ls)
    if not text:
        return
    ax.annotate(text, (x, 0.52), xycoords=("data", "axes fraction"),   # centred above the chance line
                xytext=(-2 if side == "left" else 2, 0), textcoords="offset points",
                rotation=90, ha="right" if side == "left" else "left", va="center", fontsize=6)


def band(ax, x, v, fill, stroke, label=None):
    """Median over seeds with markers and the seed range as a band; returns (x, median)."""
    keep = ~np.isnan(v).any(axis=0)
    xs, v = x[keep], 100 * v[:, keep]
    top = KEY[label] == "nfsm"                         # the NFSM line above every other model
    ax.fill_between(xs, v.min(0), v.max(0), color=fill, alpha=0.25, lw=0, zorder=1)
    ax.plot(xs, np.median(v, 0), **P.line_kw(KEY[label], linestyle=MODEL_LS[KEY[label]],
                                             zorder=5 if top else 3))
    return xs, np.median(v, 0)


def draw(ax, curves, chance, xlabel, ylabel):
    """Draw curves [(label, fill, stroke, x, values [seeds, n])] as median and seed range."""
    from matplotlib.ticker import FixedLocator, NullLocator
    chance_line(ax, 100 * chance)
    ends = []
    for label, fill, stroke, x, v in curves:
        xs, med = band(ax, x, v, fill, stroke, label)
        ends.append([med[-1], xs[-1], label, stroke])
    place_labels(ax, ends, chance)
    ax.set_xscale("log", base=2)
    lo, hi = min(c[3].min() for c in curves), max(c[3].max() for c in curves)
    ticks = log_ticks([t for t in (1, 4, 16, 64, 256, 1024, 4096, 8192) if lo * 0.9 <= t <= hi * 1.1])
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_ylim(0, 100)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


NOISE_FILL, WORD_FILL, READ_FILL = "#EDEDED", "#FCE5C7", "#CEE4F7"   # haystack, needle, question cells
TOKEN_EDGE, BRACKET = "#999999", "#555555"


def draw_stream_strip(ax, x_word, x_limits, half_width=0.035, readout_from=0.85, end=1.0):
    """Schematic stream (noise, digit, noise, read-out) with brackets for the noise before and after.
    Returns the height of the training-range arrow.
    """
    from matplotlib.patches import FancyBboxPatch
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    parts = [("noise", 0.0, x_word - half_width, NOISE_FILL),
             ("digit", x_word - half_width, x_word + half_width, WORD_FILL),
             ("noise", x_word + half_width, readout_from, NOISE_FILL),
             ("readout", readout_from, end, READ_FILL)]
    y, h, gap = 0.5, 0.42, 0.003
    for label, a, b, fill in parts:
        ax.add_patch(FancyBboxPatch((a + gap, y), b - a - 2 * gap, h,
                                    boxstyle="round,pad=0,rounding_size=0.008", mutation_aspect=6,
                                    facecolor=fill, edgecolor=TOKEN_EDGE, lw=0.5))
        ax.text((a + b) / 2, y + h / 2, label, ha="center", va="center", fontsize=6.5)
    lo, hi = x_limits                                             # the digit moves along the stream
    ax.annotate("", (lo, y + h + 0.17), (hi, y + h + 0.17),
                arrowprops=dict(arrowstyle="<->", color=BRACKET, lw=0.8, shrinkA=0, shrinkB=0),
                annotation_clip=False)
    ax.text(x_word, y + h + 0.24, "random position during training", ha="center", va="bottom", fontsize=6.5)
    for (_, a, b, _), sym in ((parts[0], r"$t_{\mathrm{before}}$"), (parts[2], r"$t_{\mathrm{after}}$")):
        yb = y - 0.1
        ax.plot([a + gap, a + gap, b - gap, b - gap], [yb + 0.06, yb, yb, yb + 0.06], color=BRACKET,
                lw=0.8, solid_capstyle="butt")
        ax.text((a + b) / 2, yb - 0.06, sym, ha="center", va="top", fontsize=6.5)
    return y + h + 0.17


def draw_around_word(ax, models, before, after):
    """Digit accuracy against the noise before (left) and after (right) the word, symmetric log axis."""
    from matplotlib.ticker import FixedLocator, NullLocator
    ax.axvline(0, color="#8A8A8A", lw=0.8)                      # the word
    for side in (-1, 1):                                         # most noise on one side in training
        train_line(ax, side * (TRAIN_MAX - WORD) / FPS, None,
                   "left" if side < 0 else "right", ls=TRAIN_LS)
    from matplotlib.lines import Line2D
    reach, handles = [1.0], []
    for key, label, fill, stroke in models:
        if key not in before and key not in after:
            continue
        if key in before:
            x, v = before[key]
            band(ax, -x, v, fill, stroke, label)
            reach.append(x.max())
        if key in after:
            x, v = after[key]
            xs, _ = band(ax, x, v, fill, stroke, label)
            reach.append(xs.max())
        handles.append(Line2D([], [], **P.line_kw(KEY[label], linestyle=MODEL_LS[KEY[label]]), label=label))
    ax.set_xscale("symlog", base=2, linthresh=1, linscale=0.5)
    span = max(reach)
    ax.set_xlim(-span * 1.4, span * 1.4)
    ticks = log_ticks([t for t in (1, 4, 16, 64, 256, 1024, 4096, 8192) if t <= span * 1.1])
    ticks = [-t for t in ticks[::-1]] + [0] + ticks
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{abs(t):g}" for t in ticks])
    # from 0, so curves below chance stay visible, with red hatching below chance
    ax.set_ylim(0, 100)
    below_chance(ax, CHANCE)
    # 0 ("forgot": the answer has settled on "none") gets a diagonal tick from the corner, added on the
    # panel that carries the y labels (forgot_tick), so that it clears "chance"
    yt = [CHANCE] + list(range(20, 101, 20))
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.set_yticklabels(["chance"] + [f"{t}" for t in yt[1:]])
    ax.set_ylabel(r"accuracy (\%)")
    ax.yaxis.set_label_coords(-0.108, 0.55)          # at the left edge, in line with the L_post label of panel B
    for side, text in ((-1, r"noise before the digit, $t_{\mathrm{before}}$ (s)"),
                       (1, r"noise after the digit, $t_{\mathrm{after}}$ (s)")):
        ax.annotate(text, (side * np.sqrt(span), 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -15), textcoords="offset points", ha="center", va="top",
                    fontsize=7.5)
    # brackets under the two halves: which side of the word has no noise
    from matplotlib.transforms import offset_copy
    x0 = ax.transAxes.inverted().transform(ax.transData.transform((0, 0)))[0]
    gap = 0.012
    below = offset_copy(ax.transAxes, fig=ax.figure, y=-31, units="points")   # under the captions
    for a, b, text in ((0.0, x0 - gap, "no noise after"), (x0 + gap, 1.0, "no noise before")):
        tick = offset_copy(below, fig=ax.figure, y=4, units="points")
        ax.plot([a, a], [0, 0], transform=tick, color="#555555", lw=0.8, clip_on=False)
        ax.plot([a, a, b, b], [0, 0, 0, 0], transform=below, color="#555555", lw=0.8, clip_on=False)
        for x in (a, b):
            ax.annotate("", (x, 0), xycoords=below, xytext=(x, 0),
                        textcoords=tick, arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8,
                                                         shrinkA=0, shrinkB=0),
                        annotation_clip=False)
        ax.annotate(text, ((a + b) / 2, 0), xycoords=below, xytext=(0, -2),
                    textcoords="offset points", ha="center", va="top", fontsize=7)
    return handles


def forgot_tick(ax, d=5.0):
    """Label of y = 0 ("forgot"): a diagonal tick from the lower left corner, the word horizontal at its end."""
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Affine2D, ScaledTranslation
    # points, from the lower left corner of the axes
    pts = Affine2D().scale(1 / 72) + ax.figure.dpi_scale_trans + ScaledTranslation(0, 0, ax.transAxes)
    ax.add_artist(Line2D([0, -d], [0, -d], transform=pts, color="black", clip_on=False,
                         lw=mpl.rcParams["ytick.major.width"]))
    ax.text(-d - 1.5, -d, "forgot", transform=pts, ha="right", va="center",
            fontsize=mpl.rcParams["ytick.labelsize"])


LEGEND_TOP = 22.2                                # points under the schematic axis


def draw_schematic(ax, span=2.0 ** 10):
    """The three ideal behaviours on the same axis: collapsing, k-definite and non-definite."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator
    from matplotlib.transforms import offset_copy
    ax.axvline(0, color="#8A8A8A", lw=0.8)
    below_chance(ax, CHANCE)
    x = np.concatenate([-np.geomspace(span * 1.4, 0.3, 300), np.geomspace(0.3, span * 1.4, 300)])
    low, K, ELL = CHANCE, 2.0 ** 2, 2.0 ** 6

    def drop(t, at, top, floor=low):                  # smooth fall from top to floor around t = at
        return floor + (top - floor) / (1 + np.exp(3.0 * (np.log2(np.maximum(t, 1e-9)) - np.log2(at))))
    # tops ordered as in the results (the collapsing and k-definite baselines peak above the NFSM),
    # just far enough apart to tell the curves apart
    styles = (("non-definite", np.full_like(x, 92.5), LS_ND),
              # past k the word is forgotten and the answer settles on "none": digit accuracy 0
              # (drawn a hair above the axis line so the curve stays visible)
              (r"$k$-definite", np.where(x < 0, 95.0, drop(x, K, 95.0, floor=1.5)), LS_KDEF),
              ("collapsing", drop(np.abs(x), ELL, 97.5), LS_COLL))
    handles = []
    for label, y, ls in styles:
        ax.plot(x, y, color=REF_DARK, lw=1.2, ls=ls, zorder=3)
        handles.append(Line2D([], [], color=REF_DARK, lw=1.2, ls=ls, label=label))
    ax.set_xscale("symlog", base=2, linthresh=1, linscale=0.5)
    ax.set_xlim(-span * 1.4, span * 1.4)
    ax.xaxis.set_major_locator(FixedLocator([-ELL, 0, K, ELL]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([r"$\ell$", "0", r"$k$", r"$\ell$"])
    ax.tick_params(labelleft=False)                   # y axis shared with the data panel
    for side, text in ((-1, r"$t_{\mathrm{before}}$"), (1, r"$t_{\mathrm{after}}$")):
        ax.annotate(text, (side * np.sqrt(span), 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -13), textcoords="offset points", ha="center", va="top", fontsize=7.5)
    # two rows: non-definite and collapsing side by side, k-definite centred below them
    kw = dict(frameon=False, fontsize=7, handlelength=2.2, borderaxespad=0, borderpad=0,
              columnspacing=0.9, handletextpad=0.5)
    top = ax.legend(handles=[handles[0], handles[2]], ncol=2, **kw)
    ax.add_artist(top)
    bottom = ax.legend(handles=[handles[1]], **kw)
    ax._theory_legends = (top, bottom)
    place_theory_legend(ax, 0.5, "upper center")


def place_theory_legend(ax, x, loc):
    """Place the two-row theory legend under the axis at `x`."""
    from matplotlib.transforms import offset_copy
    top, bottom = ax._theory_legends
    ext = top.get_window_extent(ax.figure.canvas.get_renderer())
    w, h = (v * 72 / ax.figure.dpi for v in (ext.width, ext.height))          # points
    top.set_loc(loc)
    top.set_bbox_to_anchor((x, 0), transform=offset_copy(ax.transAxes, fig=ax.figure,
                                                         y=-LEGEND_TOP, units="points"))
    bottom.set_loc("upper center")
    bottom.set_bbox_to_anchor((x, 0), transform=offset_copy(
        ax.transAxes, fig=ax.figure, x=w / 2 if loc == "upper left" else 0,
        y=-LEGEND_TOP - h - 0.35, units="points"))


PANEL_W = 394 / 72        # inches: 5.5 in text width minus the 1 pt border of figures/real_experiments.tex
LETTER_H = 6.5            # points: cap height of the bold \\small panel letter in figures/real_experiments.tex
PAD = 0.02                # inches of white around the saved figure (savefig.pad_inches)


def fit_width(fig, width=PANEL_W, pad=PAD, steps=4):
    """Set the figure width so the saved figure (tight box plus `pad`) is exactly `width` inches."""
    for _ in range(steps):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + width - (bb.width + 2 * pad), h)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--parts", default=os.path.join(ROOT, "results", "kws_stream", "parts"))
    p.add_argument("--out", default=os.path.join(ROOT, "figures", "kws_stream"))
    args = p.parse_args()
    set_style()
    import matplotlib.pyplot as plt
    os.makedirs(args.out, exist_ok=True)
    groups = load(args.parts)
    models = [m for m in MODELS if m[0] in groups]
    lines = []

    curves, x = [], None
    lines.append("test accuracy (%) over all streams against stream length: median [min; max]")
    for key, label, fill, stroke in models:
        lens, acc = by_length(groups[key])
        curves.append((label, fill, stroke, np.array(lens) / FPS, acc))
        lines.append(f"{key} ({len(groups[key])} seeds)")
        lines += [f"  {T / FPS:g} s: {100 * np.median(a):.1f} [{100 * a.min():.1f}; {100 * a.max():.1f}]"
                  for T, a in zip(lens, acc.T)]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    draw(ax, curves, 1 / 11, "stream length (s)", r"test accuracy (\%)")
    train_line(ax, TRAIN_MAX / FPS, "max training length", "left")     # longest training stream
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"accuracy_vs_length.{ext}"))
    plt.close(fig)

    edges = hold_edges(groups)
    mids = np.sqrt(np.maximum(edges[:-1], edges[1:] / 2) * edges[1:]) / FPS
    lines += ["", "digit-stream accuracy (%) against noise after the word (hold), random positions, "
              "all lengths pooled (not drawn)"]
    for key, label, fill, stroke in models:
        acc = by_hold(groups[key], edges)
        lines.append(f"{key} ({len(groups[key])} seeds)")
        lines += [f"  {lo / FPS:g}-{hi / FPS:g} s: {100 * np.median(a):.1f} "
                  f"[{100 * a.min():.1f}; {100 * a.max():.1f}]"
                  for lo, hi, a in zip(edges[:-1], edges[1:], acc.T) if not np.isnan(a).any()]
    before, after = {}, {}
    for side, key_json, title in ((before, "prefix", "word at the end, against noise before the word"),
                                  (after, "suffix", "word at the start, against noise after the word")):
        lines += ["", f"digit accuracy (%) with the {title}"]
        for key, label, fill, stroke in models:
            got = placed(groups[key], key_json)
            if got is None:
                continue
            side[key] = got[1:]
            ood = all(key_json in r.get("ood_keys", []) for r in groups[key])
            lines.append(f"{key} ({got[0]} seeds{', filtered test noise (--ood-filter)' if ood else ''})")
            lines += [f"  {x:g} s: {100 * np.median(a):.1f} [{100 * a.min():.1f}; {100 * a.max():.1f}]"
                      for x, a in zip(got[1], got[2].T)]
    fig = plt.figure(figsize=(7.2, 0.5 + 0.82))                  # strip + panel
    # prediction on the left (it carries the shared y axis), results on the right, as in the heat map
    grid = fig.add_gridspec(2, 2, height_ratios=(0.5, 0.82), width_ratios=(0.22, 1), hspace=0.1,
                            wspace=0.08)
    pred = fig.add_subplot(grid[1, 0])
    ax = fig.add_subplot(grid[1, 1], sharey=pred)
    handles = draw_around_word(ax, models, before, after)
    draw_schematic(pred)
    forgot_tick(pred)
    pred.tick_params(labelleft=True)
    ax.tick_params(labelleft=False)
    pred.set_ylabel(ax.get_ylabel())
    ax.set_ylabel("")
    key = fig.add_subplot(grid[0, 0])                 # model legend above the prediction panel
    key.axis("off")
    key.legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.5), ncol=2, frameon=False,
               fontsize=6.5, handlelength=1.8, labelspacing=0.3, columnspacing=0.4, borderaxespad=0,
               handletextpad=0.4)
    from matplotlib.transforms import offset_copy
    fig.canvas.draw()                                            # fixes the axis transform
    to_axes = lambda x: ax.transAxes.inverted().transform(ax.transData.transform((x, 0)))[0]
    x_word = to_axes(0)
    x_limits = [to_axes(side * (TRAIN_MAX - WORD) / FPS) for side in (-1, 1)]
    strip = fig.add_subplot(grid[0, 1])
    y_arrow = draw_stream_strip(strip, x_word, x_limits)
    from matplotlib.patches import ConnectionPatch
    for xf in x_limits:              # the panel's max-training-length lines, up to the arrow's ends
        fig.add_artist(ConnectionPatch((xf, 1), (xf, y_arrow), "axes fraction", "axes fraction",
                                       axesA=ax, axesB=strip, color=REF, lw=0.8, ls=TRAIN_LS,
                                       zorder=0.5))
    fit_width(fig)
    # y label flush with the figure's left edge (the legend's), in line with the L_post label of panel B
    # the y label stays beside the tick labels; the legend starts at the label's left edge (the figure's
    # left edge, in line with the L_post label of panel B); the prediction panel widens until the legend
    # ends at its right edge
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ticks_x0 = min(t.get_window_extent(r).x0 for t in pred.get_yticklabels() if t.get_text())
    ticks_w = (pred.get_window_extent(r).x0 - ticks_x0) * 72 / fig.dpi          # points
    fs, pad, gap = 6, 0.3, 2.0
    box_w = fs * (1 + 2 * pad)                                                  # rotated box width, points
    pred.annotate("theory", xy=(0, 0.5), xycoords="axes fraction", rotation=90,
                  xytext=(-(ticks_w + gap + box_w / 2), 0), textcoords="offset points",
                  ha="center", va="center", fontsize=fs, annotation_clip=False,
                  bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.25", fc="white", ec="#9A9A9A",
                            lw=0.5))
    pred.yaxis.labelpad += box_w + 2 * gap
    ratio, y_leg = 0.22, 0.5
    for _ in range(4):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = key.transAxes.inverted()
        x_lab = inv.transform((pred.yaxis.label.get_window_extent(r).x0, 0))[0]
        key.get_legend().set_bbox_to_anchor((x_lab, y_leg), transform=key.transAxes)
        # theory legend: from the same left edge
        x_lab_pred = pred.transAxes.inverted().transform((pred.yaxis.label.get_window_extent(r).x0, 0))[0]
        place_theory_legend(pred, x_lab_pred, "upper left")
        fig.canvas.draw()
        leg = key.get_legend().get_window_extent(r)
        ratio *= (leg.x1 - pred.get_window_extent(r).x0) / pred.get_window_extent(r).width
        # vertical centre of the legend halfway between the bottom of the letter A (cap height
        # LETTER_H points under the top of the saved image) and the top of the y label
        top = max(a.get_tightbbox(r).y1 for a in fig.axes if a is not key) + PAD * fig.dpi
        a_bottom = top - LETTER_H * fig.dpi / 72
        lab_top = pred.yaxis.label.get_window_extent(r).y1
        y_leg = key.transAxes.inverted().transform((0, (a_bottom + lab_top) / 2))[1]
        grid.set_width_ratios((ratio, 1))
        fit_width(fig)
    fit_width(fig, steps=8)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"digit_accuracy_around_word.{ext}"), pad_inches=PAD)
    plt.close(fig)

    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
