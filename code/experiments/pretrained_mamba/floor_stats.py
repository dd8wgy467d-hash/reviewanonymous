"""How often the decay bound is active: the fraction of (layer, head, token) steps with
a_t > -1/tau on the haystack, in the model as released and in each bounded model.

    uv run python floor_stats.py              # results/floor_stats.*, fig_floor_binding.*
    uv run python floor_stats.py --sweep      # results/floor_sweep.json
    uv run python floor_stats.py --plot-only
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mamba2_jax as m
import niah

N_HAY = 32775        # 1,311 noise lines
TAUS = (4096.0, 2048.0, 1024.0, 512.0, 128.0)
FIG_TAUS = (4096.0, 2048.0, 1024.0, 512.0)   # drawn in fig_floor_binding
RES = m.RESULTS
SWEEP_TAUS = tuple(float(2 ** k) for k in range(4, 21))     # 16 ... 1,048,576


def log_decays(params, tokens, tau):
    """Unclamped a_t for every layer, head and token of `tokens` (1, T): (L, T, H), from the model with floor tau."""
    state, out = m.init_state(1), []
    for s in range(0, tokens.shape[1], m.SEGMENT):
        seg = tokens[:, s:s + m.SEGMENT]
        n = seg.shape[1]
        seg = np.pad(seg, ((0, 0), (0, -(-n // m.CHUNK) * m.CHUNK - n)))
        state, a = m.forward_segment(params, jnp.asarray(seg), state, jnp.int32(n), jnp.float32(tau),
                                     decays=True)
        out.append(np.asarray(a[:, 0, :n]))
    return np.concatenate(out, axis=1)


def sweep():
    params = m.load_params()
    pb = niah.PromptBuilder(m.load_tokenizer())
    tokens = np.array([pb.prefix("A", 0, N_HAY)])
    a = log_decays(params, tokens, math.inf)[:, len(pb.instruction):, :]          # (L, T, H)
    out = {"tokens": f"haystack, {N_HAY} tokens after the instruction", "model": f"mamba2-{m.SIZE}",
           "taus": [int(t) for t in SWEEP_TAUS],
           "fraction of steps that bind": [round(float((a > -1.0 / t).mean()), 5) for t in SWEEP_TAUS],
           "heads binding on >50% of steps": [int(((a > -1.0 / t).mean(axis=1) > 0.5).sum()) for t in SWEEP_TAUS],
           "n_heads": m.N_LAYER * m.N_HEADS}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "floor_sweep.json").write_text(json.dumps(out, indent=1))
    for t, f in zip(out["taus"], out["fraction of steps that bind"]):
        print(f"tau {t:8d}: {100 * f:6.2f}%", flush=True)


def main():
    params = m.load_params()
    pb = niah.PromptBuilder(m.load_tokenizer())
    tokens = np.array([pb.prefix("A", 0, N_HAY)])          # instruction + sampled haystack 0
    skip = len(pb.instruction)

    summary = {"tokens": f"haystack, {N_HAY} tokens after the instruction", "n_heads": m.N_LAYER * m.N_HEADS}
    arrays = {}
    md = [f"# How often the decay floor binds\n\nHaystack of {N_HAY:,} noise tokens (after the instruction); "
          f"{m.N_LAYER} layers × {m.N_HEADS} heads = {m.N_LAYER * m.N_HEADS} heads. A step binds when "
          f"a_t = Δ_t A > −1/τ, i.e. when ρ_t = exp(a_t) > exp(−1/τ).\n"]

    for model_tau in (math.inf,) + TAUS:
        a = log_decays(params, tokens, model_tau)[:, skip:, :]          # (L, T, H)
        name = "asis" if math.isinf(model_tau) else f"tau{int(model_tau)}"
        arrays[f"median_time_{name}"] = np.median(-1.0 / a, axis=1)   # (L, H)
        thresholds = TAUS if math.isinf(model_tau) else (model_tau,)
        for tau in thresholds:
            frac = (a > -1.0 / tau).mean(axis=1)                       # (L, H)
            arrays[f"bind_{name}_at{int(tau)}"] = frac
            summary[f"{name} model, threshold tau={int(tau)}"] = {
                "fraction of steps that bind": round(float(frac.mean()), 4),
                "heads with any step binding": int((frac > 0).sum()),
                "heads binding on >50% of steps": int((frac > 0.5).sum()),
                "heads binding on >99% of steps": int((frac > 0.99).sum()),
                "layers of the heads binding on >50% of steps": sorted({int(l) for l in np.nonzero(frac > 0.5)[0]}),
            }
        if math.isinf(model_tau):
            t = arrays["median_time_asis"]
            summary["asis median memory time per head (tokens), percentiles 50/90/99/max"] = [
                round(float(np.percentile(t, q)), 1) for q in (50, 90, 99, 100)]
            summary["asis heads with median memory time > 512"] = int((t > 512).sum())
            summary["asis heads with median memory time > 2048"] = int((t > 2048).sum())

    md.append("| model | threshold | steps that bind | heads with any binding step | heads binding > 50% of steps | heads binding > 99% of steps |")
    md.append("|---|---|---|---|---|---|")
    for k, v in summary.items():
        if isinstance(v, dict):
            model, thr = k.split(" model, threshold ")
            label = "as is" if model == "asis" else f"τ = {model[3:]} (own pass)"
            md.append(f"| {label} | {thr.replace('tau=', 'τ = ')} | {100 * v['fraction of steps that bind']:.2f}% | "
                      f"{v['heads with any step binding']} | {v['heads binding on >50% of steps']} | "
                      f"{v['heads binding on >99% of steps']} |")
    p = summary["asis median memory time per head (tokens), percentiles 50/90/99/max"]
    md.append(f"\nModel as is, median memory time −1/a_t per head: median {p[0]}, 90th percentile {p[1]}, "
              f"99th percentile {p[2]}, max {p[3]} tokens; {summary['asis heads with median memory time > 512']} "
              f"heads above 512 and {summary['asis heads with median memory time > 2048']} above 2,048.\n")
    for k in [f"asis model, threshold tau={int(t)}" for t in TAUS]:
        md.append(f"Layers of the heads that bind on more than half the steps ({k.split(', ')[1].replace('tau=', 'τ = ')}, "
                  f"as is): {summary[k]['layers of the heads binding on >50% of steps']}.\n")

    RES.mkdir(parents=True, exist_ok=True)
    (RES / "floor_stats.json").write_text(json.dumps(summary, indent=1))
    (RES / "floor_stats.md").write_text("\n".join(md))
    np.savez(RES / "floor_stats.npz", **arrays)
    print("\n".join(md))
    plot(arrays)


SIZE_DIRS = ["mamba2_370m", "mamba2_780m"]      # other sizes, under results/, drawn below 130M


def plot(arrays=None):
    """Heat maps of the active fraction per layer and head, one row per model size, one column per tau."""
    import plotting as P

    P.set_style()
    base = RES if not RES.name.startswith("mamba2_") else RES.parent
    rows = []
    for d in [base] + [base / s for s in SIZE_DIRS]:
        if (d / "floor_stats.npz").exists() and (RES == base or d == RES):
            with np.load(d / "floor_stats.npz") as npz:
                name = "130M" if d == base else d.name.split("_", 1)[1].upper()
                sw = d / "floor_sweep.json"
                rows.append((f"Mamba2 {name}", {k: npz[k] for k in npz.files},
                             json.loads(sw.read_text()) if sw.exists() else None))
    nm, nt = len(rows), len(FIG_TAUS)
    fig = plt.figure(figsize=(P.FULL_W, 0.55 + nm * 1.6))
    gs = fig.add_gridspec(1 + nm, nt + 1, height_ratios=[0.3] + [1] * nm, width_ratios=[1] * nt + [1.3],
                          hspace=0.62, wspace=0.2)
    ax_key = fig.add_subplot(gs[0, 1:4])
    ax_key.axis("off")
    im = None
    for mi, (label, arr, sw) in enumerate(rows):
        for j, tau in enumerate(FIG_TAUS):
            ax = fig.add_subplot(gs[1 + mi, j])
            ax.grid(False)
            m = arr[f"bind_asis_at{int(tau)}"]
            im = ax.imshow(m, cmap=P.SEQ_CMAP, vmin=0, vmax=1, aspect="auto", origin="lower",
                           interpolation="nearest")
            nl, nh = m.shape
            # panel summary: share of all (layer, head, step) triples where the floor binds
            # same gap (4 pt) to the top and right spines: text offset = gap + box pad (0.3 em of 7.5 pt)
            ax.annotate(rf"${100 * m.mean():.1f}\%$", xy=(1, 1), xycoords="axes fraction",
                        xytext=(-(4 + 0.3 * 7.5), -(4 + 0.3 * 7.5)), textcoords="offset points",
                        ha="right", va="top", fontsize=7.5, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.3", fc="white", ec=P.TOKEN_EDGE, lw=0.5))
            ax.set_xticks([0, nh // 2, nh - 1], ["$1$", f"${nh // 2 + 1}$", f"${nh}$"])
            ax.set_yticks([0, nl // 2, nl - 1], ["$1$", f"${nl // 2 + 1}$", f"${nl}$"])
            ax.tick_params(length=2)
            if j == 0:
                ax.set_ylabel("layer")
            else:
                ax.tick_params(labelleft=False)
            if mi == nm - 1:
                ax.set_xlabel("head")
            P.panel_title(ax, rf"$\tau = {int(tau)}$",
                          letter="ABCDEFGH"[mi] if j == 0 else None)
        # last column: share of all steps that bind against tau (sweep), the panels' values marked
        ax = fig.add_subplot(gs[1 + mi, nt])
        col = P.SERIES["tau2048"][1]
        if sw is not None:
            ax.plot(sw["taus"], [100 * f for f in sw["fraction of steps that bind"]], color=col, lw=1.1,
                    marker="o", ms=2.2, zorder=3)
        pts = [100 * arr[f"bind_asis_at{int(t)}"].mean() for t in FIG_TAUS]
        ax.plot(FIG_TAUS, pts, ls="none", marker="o", ms=4.5, mfc="white", mec=col, mew=1.0, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks([2 ** 4, 2 ** 8, 2 ** 12, 2 ** 16, 2 ** 20],
                      ["$2^{4}$", "$2^{8}$", "$2^{12}$", "$2^{16}$", "$2^{20}$"])
        ax.minorticks_off()
        ax.set_xlim(2 ** 3.5, 2 ** 20.5)
        ax.set_ylim(0, 47)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(r"steps clamped (\%)" if plt.rcParams["text.usetex"] else "steps clamped (%)")
        ax.tick_params(length=2)
        if mi == nm - 1:
            ax.set_xlabel(r"$\tau$")
        P.panel_title(ax, f"{label}, all heads")
    cax = ax_key.inset_axes([0.0, 0.45, 1.0, 0.25])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_label(r"fraction of haystack steps where the floor binds ($a_t > -1/\tau$); box: all heads", fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.08 if nm > 1 else 0.15)
    P.fit_titles(fig)
    P.save(fig, RES / "fig_floor_binding")
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if "--sweep" in sys.argv:
        sweep()
    elif "--plot-only" in sys.argv:
        plot()
    else:
        main()
