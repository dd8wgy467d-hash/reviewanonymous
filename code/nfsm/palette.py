"""One style per model for the figures: a pastel colour, a darker edge of the same hue and a
marker. Lines are solid, since dashes mean something else in the figures. Pairs that are close under
colour blindness differ by marker. Mamba-2 (pretrained, heat maps only) is teal.
"""

STYLE = {   # key: (label for matplotlib text, pastel, dark, marker, marker size)
    "nfsm":         ("Neural-FSM",       "#0F62FE", "#0A3FA8", "o", 3.2),
    "mamba":        ("Mamba",        "#E68AA3", "#A8475F", "^", 3.6),
    "mingru":       ("minGRU",       "#F2AE5E", "#A86416", "s", 2.8),
    "aussm":        ("AUSSM",        "#4C9C6C", "#2E6B47", "D", 2.6),
    "deltaproduct": ("DeltaProduct", "#B9A5E6", "#5B3FA8", "v", 3.6),
    "pdssm":        ("PD-SSM",       "#8A6A50", "#5A4230", "p", 3.6),
}
MODELS = list(STYLE)

MAMBA2 = "#74BFC6"          # pretrained Mamba-2: accuracy 1 in the heat maps (white -> MAMBA2)
REF = "#9A9A9A"             # reference / floor
REF_DARK = "#5F5F5F"


def color(model):
    return STYLE[model][1]


def dark(model):
    return STYLE[model][2]


def marker(model):
    return STYLE[model][3], STYLE[model][4]


def line_kw(model, markers=True, **kw):
    """Keyword arguments for ax.plot: pastel line, marker with a dark edge."""
    out = dict(color=color(model), linestyle="-", lw=1.3)
    if markers:
        m, ms = marker(model)
        out.update(marker=m, ms=ms, mfc=color(model), mec=dark(model), mew=0.5)
    out.update(kw)
    return out


def nfsm_ramp(n=None):
    """Shades of the NFSM hue, light to dark: a colormap, or n colours when n is given."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("nfsm", ["#C2D6FF", STYLE["nfsm"][1], "#0A2E78"])
    return cmap if n is None else [cmap(i / max(n - 1, 1)) for i in range(n)]


def mamba2_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("mamba2", ["#FFFFFF", MAMBA2])
