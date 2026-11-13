"""Figure style and drawing helpers of the RULER figures."""
from __future__ import annotations

import shutil

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

FULL_W, HALF_W, ROW_H = 7.2, 3.5, 2.4

# role colours: (fill, stroke/markers/text). One role per variant, the same in every figure.
SERIES = {
    "asis": ("#1F77B4", "#17598A"),       # Mamba blue of the paper
    "tau2048": ("#E8620A", "#A63A00"),    # the paper's orange (#D94801 family)
    "tau512": ("#2CA02C", "#1E701E"),
}
LINE_GREY = "#8A8A8A"
REF_DARK = "#555555"      # black!67: neutral key glyphs, brackets
TOKEN_EDGE = "#999999"    # black!40, the paper's token outline
DECCOL = "#145A96"        # deccol (RGB 20, 90, 150)
# accuracy 1 in the heat maps: the Mamba-2 teal of nfsm/palette.py, kept apart from Mamba- and AUSSM
import os, sys                                                  # noqa: E401,E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nfsm.palette import MAMBA2                                 # noqa: E402
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", MAMBA2])
TAU = REF_DARK            # the L_post = tau line: neutral, the series hues belong to the models
CELL = {"A": "#CEE4F7", "B": "#CEECD6", "C": "#FCE5C7", "D": "#F5D3E5", "E": "#E4E1F3"}   # cellA..E


def set_style():
    usetex = shutil.which("latex") is not None
    mpl.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if usetex else ["DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 7.5, "font.size": 9, "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.8, "lines.linewidth": 1.3,
        "xtick.direction": "out", "ytick.direction": "out",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": True, "grid.color": "#CCCCCC", "grid.linestyle": ":", "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
    })


def shades(variant: str, n: int) -> list[str]:
    """n shades of a variant hue, light to dark."""
    fill, dark = (np.array(to_rgb(c)) for c in SERIES[variant])
    light = 0.7 * fill + 0.3
    if n == 1:
        return [to_hex(dark)]
    pts = [light + (fill - light) * t * 2 if t < 0.5 else fill + (dark - fill) * (t - 0.5) * 2
           for t in np.linspace(0, 1, n)]
    return [to_hex(np.clip(p, 0, 1)) for p in pts]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k / n."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # min/max guard against rounding at k = 0 and k = n (the interval always contains p)
    return min(p, max(0.0, centre - half)), max(p, min(1.0, centre + half))


def spread_labels(ys: list[float], gap: float) -> list[float]:
    """Move label heights apart by at least `gap`, keeping their order, centred on the originals."""
    order = np.argsort(ys)
    pos = np.array(ys, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    shift = (np.array(ys, float)[order] - pos).mean()
    pos = pos + shift
    out = np.empty_like(pos)
    out[order] = pos
    return out.tolist()


def draw_prompt(ax, x0=0.0, x1=1.0, fontsize=6.5, layout="A"):
    """Schematic prompt strip of layout A, B or C, with brackets for L and d."""
    from matplotlib.patches import FancyBboxPatch

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    instr, hay_l = ("instruction", 0.19, CELL["E"], "black"), ("haystack", 0.28, "#EDEDED", "black")
    parts = {"A": [instr, hay_l], "B": [hay_l, instr], "C": [hay_l]}[layout] + [
        ("needle", 0.12, CELL["C"], "black"), ("haystack", 0.22, "#EDEDED", "black"),
        ("question", 0.16, CELL["A"], "black")]
    scale = (x1 - x0) / sum(w for _, w, _, _ in parts)
    x, y, h = x0, 0.5, 0.42
    spans = []
    for label, w, fill, ink in parts:
        w *= scale
        gap = 0.004
        ax.add_patch(FancyBboxPatch((x + gap, y), w - 2 * gap, h, boxstyle="round,pad=0,rounding_size=0.012",
                                    mutation_aspect=4, facecolor=fill, edgecolor=TOKEN_EDGE, lw=0.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color=ink)
        spans.append((x, x + w))
        x += w
    i = [p[0] for p in parts].index("needle")
    for (a, b), sym in ((spans[i - 1] if layout == "A" else spans[0], r"$L_{\mathrm{pre}}$ tokens"),
                        (spans[i + 1], r"$L_{\mathrm{post}}$ tokens")):
        yb = y - 0.1
        ax.plot([a + 0.004, a + 0.004, b - 0.004, b - 0.004], [yb + 0.06, yb, yb, yb + 0.06], color=REF_DARK,
                lw=0.8, solid_capstyle="butt")
        ax.text((a + b) / 2, yb - 0.06, sym, ha="center", va="top", fontsize=fontsize, color="black")


def panel_title(ax, title, ppl=None, letter=None, box=None, letter_at=None, box_left=False):
    """Panel title, row letter and an optional boxed 'ppl = ...' inside the panel."""
    kw = dict(textcoords="offset points", ha="left", va="baseline", fontsize=8)
    ax._panel_title = ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), **kw)
    if letter:
        txt = rf"\textbf{{{letter}}}" if mpl.rcParams["text.usetex"] else letter
        # letter_at: another artist whose left edge sets the letter's x (default: this panel's y label)
        ax.annotate(txt, xy=(0, 1), xycoords=(letter_at or ax.yaxis.label, "axes fraction"), xytext=(0, 5),
                    fontweight="bold", **kw)
    if ppl is not None or box is not None:
        fs, pad = 6, 0.3
        off = 4 + pad * fs                     # gap + box pad, in points
        ax.annotate(rf"ppl $= {ppl:.2f}$" if box is None else box, xy=(0 if box_left else 0.5, 0),
                    xycoords="axes fraction", xytext=(off if box_left else 0, off), textcoords="offset points",
                    ha="left" if box_left else "center", va="bottom", fontsize=fs,
                    zorder=6, bbox=dict(boxstyle=f"round,pad={pad},rounding_size=0.25", fc="white",
                                        ec=TOKEN_EDGE, lw=0.5))


def fit_titles(fig, pad_pt=2.0, max_iter=6):
    """Widen the figure (height unchanged) until no panel title runs past its panel's right spine."""
    for _ in range(max_iter):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        need = 1.0
        for ax in fig.axes:
            tt = getattr(ax, "_panel_title", None)
            if tt is None or not ax.get_visible():
                continue
            tb, ab = tt.get_window_extent(r), ax.get_window_extent(r)
            pad = pad_pt * fig.dpi / 72
            need = max(need, (tb.width + pad) / ab.width)
        if need <= 1.0:
            return
        w, h = fig.get_size_inches()
        fig.set_size_inches(w * min(need, 1.3) * 1.01, h)


def curved_text(ax, x, y, groups, offset_pt=2.0, angle_deg=-45.0, gap_em=0.3, **kw):
    """Write `groups` of glyphs along the curve (x, y), `offset_pt` to its left. Call after layout."""
    fs = kw.pop("fontsize", mpl.rcParams["font.size"])
    groups = [[g if isinstance(g, tuple) else (g, 1.0, 0.0) for g in word] for word in groups]
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72
    P = ax.transData.transform(np.c_[x, y])
    t = np.gradient(P, axis=0)
    t /= np.hypot(*t.T)[:, None]
    Q = P + offset_pt * px * np.c_[-t[:, 1], t[:, 0]]
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
    at = lambda u: np.c_[np.interp(u, s, Q[:, 0]), np.interp(u, s, Q[:, 1])][0]

    def width(g):
        tt = ax.text(0, 0, g[0], fontsize=fs * g[1], **kw)
        w = tt.get_window_extent(r).width
        tt.remove()
        return w

    widths = [[width(g) for g in word] for word in groups]
    gap = gap_em * fs * px
    total = sum(map(sum, widths)) + gap * (len(groups) - 1)
    ang = np.degrees(np.arctan2(t[:, 1], t[:, 0]))
    u = s[np.argmin(np.abs(ang - angle_deg))] - total / 2
    for word, ws in zip(groups, widths):
        for g, w in zip(word, ws):
            a, b = at(u), at(u + w)
            rot = np.arctan2(*(b - a)[::-1])
            xy = at(u + w / 2) + g[2] * fs * px * np.r_[-np.sin(rot), np.cos(rot)]
            ax.text(*ax.transAxes.inverted().transform(xy), g[0], transform=ax.transAxes, rotation=np.degrees(rot),
                    rotation_mode="anchor", ha="center", va="baseline", fontsize=fs * g[1], **kw)
            u += w
        u += gap


def separate_blocks(fig, blocks, lw=0.8):
    """Black rules between blocks of axes. Call after layout."""
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bbs = [Bbox.union([ax.get_tightbbox(r) for ax in blk]) for blk in blocks]
    full = Bbox.union([ax.get_tightbbox(r) for ax in fig.axes if ax.get_visible()])
    inv = fig.transFigure.inverted()
    for a, b in zip(bbs, bbs[1:]):
        (x0, y), (x1, _) = inv.transform([(full.x0, (a.y0 + b.y1) / 2), (full.x1, 0)])
        fig.lines.append(Line2D([x0, x1], [y, y], transform=fig.transFigure, figure=fig, color="black", lw=lw))


def save(fig, path_stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_stem}.{ext}")
