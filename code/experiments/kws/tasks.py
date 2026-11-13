"""KWS suite: spoken-digit classification from 101 MFCC frames, 11 classes (ten digits and
noise). Only the clip class is supervised.

Backbone with a linear input projection, m = 128, recipe defaults, batch 64.

Job token: cell,layers,readout,seed, e.g. nfsm-64x4,4,last,42. nfsm-KxN: K heads of N states;
cell-W: width W (mamba and aussm also cell-WsN with d_state N); deltaproduct: n_h = 2, 8 heads;
pdssm-W: W complex states.
"""

import os
import re

SUITE = "kws"
WANDB_PROJECT = "nfsm-kws"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get("KWS_DATA_DIR", os.path.join(ROOT, "data", "kws"))

SR, N_FRAMES, N_MFCC = 16_000, 101, 13
CLASSES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "noise")

M = 128                                  # model width
NFSM_HEADS = (16, 16, 16, 16)            # NFSM states per head; baselines use their sum as width
LAYERS = (3,)
CONV_KERNEL = 2
BATCH = 64
STEPS = 30_000
VAL_EVERY = 250
EVAL_BATCH = 1024
SEEDS = (42, 43, 44)
CELLS = ("nfsm", "mingru", "mamba", "aussm", "deltaproduct")
READOUTS = ("last", "mean")


def _heads(cell: str):
    """(registry cell, head sizes) of a token's cell field, or None if it is not a known cell."""
    if cell in CELLS:
        return cell, (NFSM_HEADS if cell == "nfsm" else (sum(NFSM_HEADS),))
    m = re.fullmatch(r"nfsm-(\d+)x(\d+)", cell)
    if m:
        return "nfsm", (int(m.group(2)),) * int(m.group(1))
    m = re.fullmatch(r"(mingru|mamba|aussm|deltaproduct|pdssm)-(\d+)", cell)
    if m:
        return m.group(1), (int(m.group(2)),)
    m = re.fullmatch(r"pdssm-(\d+)f(\d+)", cell)          # PD-SSM with a decay floor (see _decay_floor)
    if m:
        return "pdssm", (int(m.group(1)),)
    m = re.fullmatch(r"(mamba|aussm)-(\d+)s(\d+)", cell)
    return (m.group(1), (int(m.group(2)),)) if m else None


def _d_state(cell: str):
    """States per channel named by a cell-WsN field, else None (the backbone default)."""
    m = re.fullmatch(r"(mamba|aussm)-\d+s(\d+)", cell)
    return int(m.group(2)) if m else None


def _decay_floor(cell: str):
    """tau of a pdssm-<d>f<tau> field (|D_t| capped at exp(-1/tau) at test time), else None."""
    m = re.fullmatch(r"pdssm-\d+f(\d+)", cell)
    return int(m.group(1)) if m else None


def parse_token(token: str) -> dict:
    """cell,layers,readout,seed -> job dict."""
    cell, layers, readout, seed = token.split(",")
    if _heads(cell) is None or readout not in READOUTS:
        raise ValueError(f"bad token {token!r}")
    d_state, floor = _d_state(cell), _decay_floor(cell)
    cell, heads = _heads(cell)
    job = {"token": token, "cell": cell, "layers": int(layers), "readout": readout,
            "seed": int(seed), "head_sizes": list(heads), "m": M, "n_features": N_MFCC,
            "n_classes": len(CLASSES), "conv_kernel": CONV_KERNEL}
    if d_state is not None:
        job["d_state"] = d_state
    if floor is not None:
        job["pdssm_decay_floor"] = floor
    return job


def iter_jobs(cells=None, layers=None, readouts=None, seeds=None) -> list:
    """Job tokens; NFSM only unless baselines are named in `cells`; readout `last` by default."""
    return [f"{c},{n},{r},{s}" for c in cells or ("nfsm",) for n in layers or LAYERS
            for r in readouts or ("last",) for s in seeds or SEEDS]
