"""Draw the head-state trajectory figure from its plot JSON. Standalone: needs matplotlib only.

    python plot_trajectory.py head_trajectory_plot.json [--out head_trajectory] \\
        [--usetex --latex-preamble '\\usepackage{times}']

The JSON is written by experiments.tso.trajectory next to the figure: the words of the stream, the
person each word names (or null), the names, and one entry per row {group, label, values, raw}:
values are person indices (n = idle / unset) or, for raw rows, a head's state index.
"""

import argparse
import json

WIDTH = 5.5                                  # inches: ICLR text width
FONT = 7                                     # pt, every text in the figure
PERSON = ("#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#F0E442", "#D55E00")
EMPTY = "#EDEDED"
SHADES = ("#F7F7F7", "#D9D9D9", "#B8B8B8", "#969696", "#737373", "#555555")


def draw(data: dict, out: str, usetex: bool = False, preamble: str = "") -> None:
    """Write out.pdf and out.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if usetex:                                        # the local LaTeX, its preamble sets the font
        plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "text.latex.preamble": preamble, "font.size": FONT})
        bold = {"fontweight": "bold"}
    else:                                             # matplotlib's own Computer Modern
        plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.serif": ["cmr10"],
                             "mathtext.fontset": "cm", "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
                             "pdf.fonttype": 42, "font.size": FONT})
        bold = {"fontfamily": "cmb10"}
    words, who, n = data["tokens"], data["token_person"], len(data["names"])
    T = len(words)
    initial = [nm[0].upper() for nm in data["names"]]
    groups = []
    for r in data["rows"]:
        if not groups or groups[-1][0] != r["group"]:
            groups.append((r["group"], []))
        groups[-1][1].append(r)

    def extent(texts, **kw):
        """Largest (width, height) in inches of `texts` as drawn at FONT."""
        f = plt.figure()
        rd = f.canvas.get_renderer()
        ext = [f.text(0, 0, t, fontsize=FONT, **kw).get_window_extent(rd) for t in texts]
        plt.close(f)
        return max(e.width for e in ext) / f.dpi, max(e.height for e in ext) / f.dpi

    # margins in inches, measured from the text they hold
    label_w, text_h = extent([r["label"] for r in data["rows"]])
    tick_w = max([extent([w for w, p in zip(words, who) if p is None])[0]]
                 + ([extent([w for w, p in zip(words, who) if p is not None], **bold)[0]]
                    if any(p is not None for p in who) else []))
    pad, row_h, gap = 0.04, 0.118, 0.07
    top, bottom = text_h + 2 * pad, tick_w + pad
    left, right = text_h + pad + label_w + pad, 0.01
    grid_h = row_h * len(data["rows"]) + gap * (len(groups) - 1)
    height = top + grid_h + bottom
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([left / WIDTH, bottom / height, (WIDTH - left - right) / WIDTH, grid_h / height])
    col_w = (WIDTH - left - right) / T                # inches per column
    unit = row_h / col_w                              # one row, in column units

    y = 0.0
    for gi, (group, rows) in enumerate(groups):
        y0 = y
        for r in rows:
            vals, raw = r["values"], r.get("raw", False)
            for t in range(T):
                v = vals[t]
                face = SHADES[v] if raw else (EMPTY if v >= n else PERSON[v])
                ax.add_patch(plt.Rectangle((t - 0.5, y), 1, unit, facecolor=face, edgecolor="white", lw=0.35))
                if (raw or v < n) and (t == 0 or v != vals[t - 1]):   # a label where a run starts
                    ax.text(t, y + unit / 2, str(v) if raw else initial[v], ha="center",
                            va="center_baseline", fontsize=FONT,
                            color="white" if raw and face in SHADES[3:] else "#222222")
            ax.text(-0.5 - pad / col_w, y + unit / 2, r["label"], ha="right", va="center", fontsize=FONT)
            y += unit
        ax.text(-0.5 - (left - text_h / 2) / col_w, (y0 + y) / 2, group, ha="center", va="center",
                rotation=90, fontsize=FONT, **bold)
        if gi < len(groups) - 1:
            ax.axhline(y + gap / col_w / 2, color="#333333", lw=0.5)
            y += gap / col_w
    for t in range(1, T):                             # sentence boundaries
        if words[t - 1] == ".":
            ax.axvline(t - 0.5, color="#555555", lw=0.4, ls=(0, (1.5, 1.5)))
    ax.add_patch(plt.Rectangle((T - 1.5, 0), 1, y, fill=False, edgecolor="black", lw=1.0,  # the `?` column
                               clip_on=False, zorder=5))
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(y, 0)
    ax.set_xticks(range(T))
    ax.set_xticklabels(words, rotation=90, fontsize=FONT)
    for tick, p in zip(ax.get_xticklabels(), who):
        if p is not None:
            tick.set_color(PERSON[p])
            tick.set(**bold)
    ax.set_yticks([])
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PERSON[i], edgecolor="none") for i in range(n)]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=EMPTY, edgecolor="none"))
    labels = list(data["names"]) + ["idle / unset"]
    if any(r.get("raw") for r in data["rows"]):
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=SHADES[3], edgecolor="none"))
        labels.append("untracked: raw state")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               fontsize=FONT, frameon=False, handlelength=0.9, handleheight=0.9, handletextpad=0.3,
               columnspacing=0.8, borderaxespad=0.0, borderpad=0.0)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"wrote {out}.pdf|png", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="draw the head-state trajectory figure from its plot JSON")
    p.add_argument("json")
    p.add_argument("--out", default="head_trajectory", help="output path without extension")
    p.add_argument("--usetex", action="store_true", help="typeset every text with the local LaTeX")
    p.add_argument("--latex-preamble", default="", help=r"e.g. '\usepackage{times}'")
    args = p.parse_args()
    with open(args.json) as f:
        draw(json.load(f), args.out, usetex=args.usetex, preamble=args.latex_preamble)
