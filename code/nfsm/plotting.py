"""Matplotlib setup (headless, serif, LaTeX when available) and figure saving."""

import os
import shutil

HAS_LATEX = shutil.which("latex") is not None


def plt_setup(rc=None):
    """Configure matplotlib and return pyplot; `rc` overrides individual rcParams."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = {"text.usetex": HAS_LATEX, "font.family": "serif", "font.size": 11,
            "axes.labelsize": 12, "axes.titlesize": 13, "legend.fontsize": 9}
    base.update(rc or {})
    plt.rcParams.update(base)
    return plt


def save_fig(fig, plt, out_path, dpi=300, tight=True):
    """Save `fig` to out_path (creating its directory) and close it."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}", flush=True)
