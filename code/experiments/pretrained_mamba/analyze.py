"""Tables and figures of the RULER experiment.

    uv run python analyze.py [results_dir]

Writes results/tables.md (accuracy with 95% Wilson intervals, perplexity) and
results/fig_accuracy_heatmap.{pdf,png} for every model size found.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plotting as P

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
LAYOUTS = [("A", "instruction first", "instruction first"),
           ("B", "instruction before the needle", "instruction before the needle"),
           ("C", "no instruction", "no instruction")]
SIZE_DIRS = ["mamba2_370m", "mamba2_780m"]     # other sizes, under results/, drawn below 130M
FIGS = Path(__file__).resolve().parents[2] / "figures" / "pretrained_mamba"     # figures for the paper
MODEL = "Mamba2 " + (RES.name.split("_", 1)[1].upper() if RES.name.startswith("mamba2_") else "130M")
# name, markdown label, LaTeX label, tau (None: model as is)
VARIANTS = [("asis", MODEL, MODEL, None), ("tau4096", "τ = 4096", r"$\tau = 4096$", 4096),
            ("tau2048", "τ = 2048", r"$\tau = 2048$", 2048), ("tau1024", "τ = 1024", r"$\tau = 1024$", 1024),
            ("tau512", "τ = 512", r"$\tau = 512$", 512),
            ("tau128", "τ = 128", r"$\tau = 128$", 128)]
FIG_LAYOUTS = ("A", "B", "C")           # figure blocks, top to bottom; a layout without 130M results is left out
D_FIG_MIN = 25                         # d = 0 is left out of the figures (tables keep it)
D_HEAT_LABEL = {25, 100, 500, 2050, 8200}     # labelled L_post ticks in fig_accuracy_heatmap
ASSUME_ZERO_D = (16400, 32775)     # floors not run there (780M) are drawn as 0 in the heat map
FIG_TAUS = (None, 4096, 2048, 1024, 512)   # variants drawn in the figures (tables have all of VARIANTS)
FIG_VARIANTS = [v for v in VARIANTS if v[3] in FIG_TAUS]


def L_tex(L):
    """L in tick-label form: 0, 25, 2k, 8k, ... (inside math)."""
    return str(L) if L < 1024 else rf"{L // 1024}\mathrm{{k}}"


def fmt_L(L):
    return f"${L_tex(L)}$"


def count(res, d, L, n):
    return int(sum(res[(d, L, i)] for i in range(n)))


def tables(runs, ds, Ls, n, meta):
    Lmax = max(Ls)
    out = [f"# Results\n\nRULER niah_single_1, needle d tokens before the question, {n} needles over "
           f"{meta['n_haystacks']} sampled noise haystacks (seed {meta['seed']}), greedy decoding of "
           f"{meta['n_generated_tokens']} tokens; a needle is answered when the output contains its 7-digit "
           f"value. Layout A: instruction · L haystack · needle · d haystack · question. Layout B: L haystack · "
           f"instruction · needle · d haystack · question. Brackets: 95% Wilson interval over the {n} needles.\n"]
    for layout, title, _ in LAYOUTS:
        out.append(f"## Layout {layout}: {title}\n")
        out.append("### Accuracy\n")
        for name, label, _, _ in VARIANTS:
            res = runs.get((layout, name))
            if res is None:
                continue
            out.append(f"**{label}**\n\n| d \\ L | " + " | ".join(f"{L:,}" for L in Ls) + " |")
            out.append("|---" * (len(Ls) + 1) + "|")
            for d in ds:
                cells = []
                for L in Ls:
                    k = count(res, d, L, n)
                    lo, hi = P.wilson(k, n)
                    cells.append(f"{k / n:.2f} [{lo:.2f}, {hi:.2f}]")
                out.append(f"| {d} | " + " | ".join(cells) + " |")
            out.append("")
        out.append(f"### Needles whose outcome differs from the outcome at L = {Lmax:,}\n")
        out.append(f"Out of {n}. Zero from some L on means the outcome no longer depends on L there.\n")
        out.append("| variant | d | " + " | ".join(f"{L:,}" for L in Ls) + " |")
        out.append("|---" * (len(Ls) + 2) + "|")
        for name, label, _, _ in VARIANTS:
            res = runs.get((layout, name))
            if res is None:
                continue
            for d in ds:
                flips = [sum(res[(d, L, i)] != res[(d, Lmax, i)] for i in range(n)) for L in Ls]
                out.append(f"| {label} | {d} | " + " | ".join(str(f) for f in flips) + " |")
        out.append("")
        out.append(f"### Horizon at L = {Lmax:,}\n")
        out.append("Largest d with accuracy ≥ 0.5, and the first larger d of the grid, in tokens and in units "
                   "of τ.\n")
        out.append("| variant | largest d with accuracy ≥ 0.5 | next d | in units of τ |\n|---|---|---|---|")
        for name, label, _, tau in VARIANTS:
            res = runs.get((layout, name))
            if res is None or tau is None:
                continue
            good = [d for d in ds if count(res, d, Lmax, n) / n >= 0.5]
            last = max(good) if good else None
            nxt = min([d for d in ds if last is not None and d > last], default=None)
            ratio = (f"{last / tau:.2f} to {nxt / tau:.2f}" if last is not None and nxt is not None else "")
            out.append(f"| {label} | {last} | {nxt} | {ratio} |")
        out.append("")
    ppl_path = RES / "perplexity.json"
    if ppl_path.exists():
        ppl = json.loads(ppl_path.read_text())
        out.append(f"## Perplexity\n\n{ppl['text']}: {ppl['n_segments']} independent segments of "
                   f"{ppl['segment']} tokens ({ppl['n_predicted_tokens']} predicted tokens).\n")
        out.append("| variant | perplexity | change |\n|---|---|---|")
        base = ppl["ppl"]["asis"]
        for name, label, _, _ in VARIANTS:
            if name in ppl["ppl"]:
                v = ppl["ppl"][name]
                out.append(f"| {label} | {v:.2f} | {v - base:+.2f} |")
        out.append("")
    return out


def load_ppl(res=None):
    path = (res or RES) / "perplexity.json"
    return json.loads(path.read_text())["ppl"] if path.exists() else {}


def model_label(res):
    """results/ holds 130M; results/mamba2_<size>/ another size."""
    return "Mamba2 " + (res.name.split("_", 1)[1].upper() if res.name.startswith("mamba2_") else "130M")


def load_model(res):
    """Runs, grids and perplexities of one model's results folder (None if it has no runs)."""
    runs, metas = {}, {}
    for layout, _, _ in LAYOUTS:
        for name, _, _, _ in VARIANTS:
            path = res / f"niah_{layout}_{name}.json"
            if path.exists():
                data = json.loads(path.read_text())
                runs[(layout, name)] = {(r["d"], r["L"], r["prompt"]): r["correct"] for r in data["records"]}
                metas[(layout, name)] = data["meta"]
    if not runs:
        return None
    meta = next(iter(metas.values()))
    n = meta["n_prompts"]
    assert all(mm["n_prompts"] == n for mm in metas.values())
    ds = sorted(set.intersection(*(set(mm["d"]) for mm in metas.values())))
    Ls = sorted(set.intersection(*(set(mm["L"]) for mm in metas.values())))
    # figures start at L = 25 (one haystack line) when every run has it; the tables keep L = 0
    Ls_fig = [L for L in Ls if L > 0] if 25 in Ls else Ls
    # each run's own grid (the heat map draws every run on its full grid; line plots use the common one)
    grids = {k: (sorted(L for L in mm["L"] if L > 0), sorted(d for d in mm["d"] if d >= D_FIG_MIN))
             for k, mm in metas.items()}
    return {"label": model_label(res), "runs": runs, "meta": meta, "n": n, "ds": ds, "Ls": Ls,
            "Ls_fig": Ls_fig, "ppl": load_ppl(res), "grids": grids}


def grid_figure(models, side=False, layouts=FIG_LAYOUTS, width=None, variants=None, wspace=0.18,
                pred_left=False, panel_h=0.95):
    """Figure with one block per layout and one row of panels per model. Returns (fig, key axis, rows),
    and the side axes with side=True. Sizes in inches.
    """
    nv, nm = len(variants or FIG_VARIANTS), len(models)
    # side panels sit as close as the other panels; the y labels of the rows below them go under them
    c0 = 1 if side else 0                                       # first column of the model panels
    STRIP, TITLE, PANEL, BETWEEN, AFTER, MARGIN = 0.40, 0.24, panel_h, 0.40, 0.58, 0.05   # inches
    if pred_left:                                               # main-text figure: a tighter key row
        STRIP = 0.26
    heights, idx = [], []                                       # idx[b] = (strip row, [panel rows])
    for b in range(len(layouts)):
        r0 = len(heights)
        # below the first row, the schematics' x labels sit above the next row's titles: more room
        gaps = [AFTER if side else BETWEEN] + [BETWEEN] * (nm - 2)
        heights += [STRIP, TITLE] + [h for g in gaps[:nm - 1] for h in (PANEL, g)] + [PANEL]
        idx.append((r0, [r0 + 2 + 2 * mi for mi in range(nm)]))
        heights += [AFTER if b < len(layouts) - 1 else AFTER - 0.2]
    H = sum(heights) + 2 * MARGIN
    fig = plt.figure(figsize=(width or P.FULL_W, H))
    ncol = nv + 2 * c0
    widths = ([1] + [1] * nv + [1]) if side else [1] * nv
    if side and pred_left:
        c0, ncol, widths = 3, nv + 3, [1, 1, 0.25] + [1] * nv
    gs = fig.add_gridspec(len(heights), ncol, height_ratios=heights, width_ratios=widths, hspace=0,
                          wspace=wspace, left=0.08, right=0.99, top=1 - MARGIN / H, bottom=MARGIN / H)
    rows, ax_key, sides = {}, None, {}
    fig._blocks = []                                            # axes of each block, for rules between them
    for b, layout in enumerate(layouts):
        rs, rp = idx[b]
        # strip over four panels, key over the rest; pred_left: key over the side panels, strip over the models
        s_cols, k_cols = (slice(c0, None), slice(0, 2)) if (side and pred_left) else \
            (slice(0, c0 + 3), slice(c0 + 3, None))
        ax_p = fig.add_subplot(gs[rs, s_cols])
        P.draw_prompt(ax_p, 0.0, 1.0, layout=layout)
        fig._blocks.append([ax_p])
        if b == 0:
            ax_key = fig.add_subplot(gs[rs, k_cols])
            ax_key.axis("off")
        if side:
            sides[layout] = (fig.add_subplot(gs[rp[0], 0]),
                             fig.add_subplot(gs[rp[0], 1 if pred_left else ncol - 1]))
        first = None
        for mi in range(nm):
            axes = []
            for j in range(nv):
                ax = fig.add_subplot(gs[rp[mi], c0 + j], sharey=first)
                first = first or ax
                axes.append(ax)
                if j:
                    ax.tick_params(labelleft=False)
            rows[(layout, mi)] = axes
            fig._blocks[-1] += axes
        fig._blocks[-1] += list(sides.get(layout, ()))
    return (fig, ax_key, rows, sides) if side else (fig, ax_key, rows)


def fit_width(fig, width, pad, steps=4):
    """Set the figure width so the saved figure (tight box plus `pad`) is exactly `width` inches."""
    for _ in range(steps):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + width - (bb.width + 2 * pad), h)


def fig_accuracy_heatmap(models, out_stem, layouts=None, letters=True, rules=False, width=None, taus=None,
                         xticks=None, units=True, wspace=0.18, tau_on_lines=False,
                         pred_left=False, side_box="schematic", panel_h=0.95):
    """Accuracy over the (L_pre, L_post) grid, one block per layout, one row per model, one panel per
    tau; dotted line at L_post = tau. Writes out_stem.pdf and out_stem.png.
    """
    layouts = layouts or [l for l in FIG_LAYOUTS if (l, "asis") in models[0]["runs"]]
    variants = [v for v in VARIANTS if v[3] in (taus or FIG_TAUS)]
    fig, ax_key, rows, sides = grid_figure(models, side=True, layouts=layouts, width=width, variants=variants,
                                           wspace=wspace, pred_left=pred_left, panel_h=panel_h)
    tok = " (tokens)" if units else ""
    nm = len(models)
    im = None
    # common log axes: the union of every run's grid, cells centred on the measured points
    allL = sorted({L for m in models for g in m["grids"].values() for L in g[0]})
    alld = sorted({d for m in models for g in m["grids"].values() for d in g[1]})

    def edges(v):
        """Cell edges on a log axis: geometric midpoints, half a step beyond the ends."""
        lv = np.log(np.asarray(v, float))
        mid = (lv[1:] + lv[:-1]) / 2
        return np.exp(np.concatenate([[lv[0] - (mid[0] - lv[0])], mid, [lv[-1] + (lv[-1] - mid[-1])]]))

    XL, YL = edges(allL), edges(alld)
    xlim, ylim = (XL[0], XL[-1]), (YL[0], YL[-1])
    xt = xticks or [25, 500, 8200, 131075]
    xt_lab = [fmt_L(L) for L in xt]
    yt = sorted(D_HEAT_LABEL | {32775})

    def frame(ax):
        """Log axes, shared limits and ticks."""
        ax.grid(False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks(xt, xt_lab)
        ax.set_yticks(yt, [f"${d}$" for d in yt])
        ax.minorticks_off()
        ax.tick_params(length=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)

    # schematic panels (first row of each block): two ideal automata on the same axes
    L_TOT, K_HOR = (6150 if width else 10250), 1025   # illustrative: total length at which the state collapses, horizon
    gx, gy = np.geomspace(*xlim, 400), np.geomspace(*ylim, 400)
    collapse = (gx[None, :] + gy[:, None] <= L_TOT).astype(float)
    definite = (np.ones_like(gx)[None, :] * (gy[:, None] <= K_HOR)).astype(float)
    curves = []                        # L_pre + L_post = const. labels, placed once the layout is final
    for layout in layouts:
        axl, axr = sides[layout]
        for ax, mat, title in ((axl, collapse, "collapsing"), (axr, definite, r"$k$-definite")):
            frame(ax)
            ax.pcolormesh(gx, gy, mat, cmap=P.SEQ_CMAP, vmin=0, vmax=1, shading="nearest", zorder=1,
                          rasterized=True)          # 400 x 400 cells: an image, not 160,000 vector quads
            P.panel_title(ax, title, box=side_box, box_left=bool(width))
            ax.set_xlabel(r"$L_{\mathrm{pre}}$" + tok)
        axl.set_ylabel(r"$L_{\mathrm{post}}$" + tok)
        axr.tick_params(labelleft=False)
        Lc = np.geomspace(xlim[0], L_TOT - ylim[0], 300)
        axl.plot(Lc, L_TOT - Lc, color=P.REF_DARK, lw=0.8, ls="--", zorder=3)
        curves.append((axl, Lc, L_TOT - Lc))
        axr.axhline(K_HOR, color=P.REF_DARK, lw=0.8, ls="--", zorder=3)
        axr.text(np.sqrt(xlim[0] * xlim[1]), K_HOR * 1.35, r"horizon $k$", va="bottom", ha="center",
                 fontsize=6.5, color=P.REF_DARK, zorder=4)

    for b, layout in enumerate(layouts):
        for mi, mdl in enumerate(models):
            n = mdl["n"]
            letter = "abcdefghijklmnopqrstuvwxyz"[b * nm + mi]
            for j, (ax, (name, _, tex, tau)) in enumerate(zip(rows[(layout, mi)], variants)):
                P.panel_title(ax, mdl["label"] if tau is None else tex, mdl["ppl"].get(name),
                              letter.upper() if j == 0 and letters else None,
                              # rows with a schematic panel: letter at the far left; others: at their own y label
                              letter_at=sides[layout][0].yaxis.label if mi == 0 else None)
                frame(ax)
                res = mdl["runs"].get((layout, name))
                if res is not None:
                    Ls, ds = mdl["grids"][(layout, name)]
                    # floors at L_post = 16400 and 32775 that were not run (780M): drawn as 0. Every
                    # floor run there (130M, 370M: 24,000 outcomes) answered no needle, and every
                    # floor horizon is below 4100
                    assumed = [d for d in ASSUME_ZERO_D if tau is not None and d not in ds]
                    ds = sorted(ds + assumed)
                    acc = np.array([[0.0 if d in assumed else count(res, d, L, n) / n for L in Ls] for d in ds])
                    im = ax.pcolormesh(edges(Ls), edges(ds), acc, cmap=P.SEQ_CMAP, vmin=0, vmax=1,
                                       shading="flat", zorder=1,
                                       rasterized=True)   # vector cells show seams in some PDF viewers
                    if tau is not None:
                        ax.axhline(tau, color=P.TAU, lw=0.9, ls=":", zorder=3)
                        if tau_on_lines:
                            ax.text(np.sqrt(xlim[0] * xlim[1]), tau * 1.35, r"$L_{\mathrm{post}} = \tau$",
                                    va="bottom", ha="center", fontsize=6.5, color=P.TAU,
                                    zorder=4)
                if mi == nm - 1:
                    ax.set_xlabel(r"$L_{\mathrm{pre}}$" + tok)
                else:
                    ax.tick_params(labelbottom=False)
                if j == 0 and mi > 0:
                    ax.set_ylabel(r"$L_{\mathrm{post}}$" + tok)
                else:
                    # first row: the schematic panel on the left carries the shared L_post axis
                    ax.tick_params(labelleft=False)
    # colour bar in the key slot
    ax_key.axis("on")
    for sp in ax_key.spines.values():
        sp.set_visible(False)
    ax_key.set_xticks([])
    ax_key.set_yticks([])
    cax = ax_key.inset_axes([0.08, 0.6, 0.84, 0.16] if tau_on_lines else [0.08, 0.6, 0.55, 0.16])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1] if tau_on_lines else [0, 0.5, 1])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label("accuracy", fontsize=7, labelpad=-7 if tau_on_lines else None)
    from matplotlib.lines import Line2D
    if not tau_on_lines:
        ax_key.legend(handles=[Line2D([], [], color=P.TAU, lw=0.9, ls=":", label=r"$L_{\mathrm{post}} = \tau$")],
                      loc="center left", bbox_to_anchor=(0.7, 0.68), frameon=False, fontsize=6.5)
    pad = 0.02 if width else matplotlib.rcParams["savefig.pad_inches"]
    if width:
        fit_width(fig, width, pad)
    else:
        P.fit_titles(fig)
    # every title centred over its panel (main figure: not widened to fit them)
    for ax in fig.axes:
        tt = getattr(ax, "_panel_title", None)
        if tt is not None:
            tt.xy = (0.5, 1)
            tt.set_ha("center")
    sub = lambda w: [(c, 0.7, -0.17) for c in w]          # subscript letters, placed one by one
    words = [[r"$L$"] + sub("pre"), [r"$+$"], [r"$L$"] + sub("post"), [r"$=$"], [r"$C$"]]
    for ax, cx, cy in curves:
        # narrow panels (fixed width): centred further down the curve, so the label clears the left spine
        P.curved_text(ax, cx, cy, words, offset_pt=3.0, fontsize=6, color=P.REF_DARK, zorder=4,
                      angle_deg=(-34.0 if panel_h < 0.72 else -42.0 if panel_h < 0.9 else -62.0) if width else -45.0)
    if rules:
        P.separate_blocks(fig, fig._blocks)
    with matplotlib.rc_context({"savefig.pad_inches": pad}):
        P.save(fig, out_stem)
    plt.close(fig)


def main():
    """Tables for RES; from results/ (130M), the figures include every size found."""
    P.set_style()
    mdl = load_model(RES)
    assert mdl, "no results"
    lines = tables(mdl["runs"], mdl["ds"], mdl["Ls"], mdl["n"], mdl["meta"])
    (RES / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))
    models = [mdl]
    if not RES.name.startswith("mamba2_"):
        models += [m for m in (load_model(RES / s) for s in SIZE_DIRS) if m]
    fig_accuracy_heatmap(models, RES / "fig_accuracy_heatmap", rules=True)
    if len(models) > 1:
        # paper figures: main text = layout A, largest model, with the schematics; appendix = everything
        FIGS.mkdir(parents=True, exist_ok=True)
        # niah_main goes in figures/real_experiments at 100%: 394 pt wide (5.5 in text width minus the 1 pt
        # border there); tau = 4096 left out, three L_pre ticks and no "(tokens)" so that six panels fit with
        # readable labels
        fig_accuracy_heatmap(models[-1:], FIGS / "niah_main", layouts=["A"], letters=False, width=394 / 72,
                             taus=(None, 2048, 1024, 512), xticks=[25, 2050, 131075], units=False,
                             tau_on_lines=True, pred_left=True, side_box="theory",
                             panel_h=0.6)
        fig_accuracy_heatmap(models, FIGS / "niah_appendix", rules=True)


if __name__ == "__main__":
    main()
