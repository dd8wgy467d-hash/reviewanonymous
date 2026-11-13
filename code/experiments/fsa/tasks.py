"""FSA suite: group and monoid word problems on one NFSM layer.

    task  structure                   letters                  classes  NFSM rungs (head sizes)
    Z2    cyclic Z_2                  0, 1                     2        regular (2)
    Z16   cyclic Z_16                 0..15                    16       regular (16)
    S3    symmetric S_3               adjacent transpositions  6        regular (6), factored (3, 2)
    S4    symmetric S_4               adjacent transpositions  24       regular (24), factored (4, 3, 2)
    A5    alternating A_5             adjacent 3-cycles        60       regular (60), factored (5, 12)
    M11   Mathieu M11                 2 generators, inverses   7920     factored (11, 11, 11, 11)
    DFF5  flip-flop, <= 4 identities  identity, reset, set     3        regular (3)
    FF    flip-flop, 90% identities   identity, reset, set     3        regular (3)

regular: one head of |Q| states; factored: the smallest separating family of quotient heads.
Baselines run 4 layers of width 256 (rung `fixed`); for deltaproduct the rung is the number of
Householder steps (nh1, nh2, nh4), 8 heads of 32. A rung h<n1>-<n2>-... sets the NFSM head sizes
(free arm only; scripts/fsa_headsearch.py).

Job token: task,rung,cell,layers,arm,seed, e.g. S4,factored,nfsm,1,free,42.
"""

import re

import numpy as np

SUITE = "fsa"
WANDB_PROJECT = "nfsm-fsa"
TRAIN_LEN = 64
BATCH = 256
CONV_KERNEL = 0
STEPS = 100_000
SWEEP_LENS = tuple(2 ** k for k in range(3, 16))          # 8 .. 32768
SEEDS = (42, 43, 44, 45, 46)
ARMS = ("free",)
CELLS = ("nfsm", "mamba", "aussm", "pdssm", "mingru", "deltaproduct")
# Recorded as not learned without a finished run (results/fsa/deltaproduct_M11_cancelled.md): left
# out of the default sweep so the dispatcher does not relaunch them; --tasks M11 still lists them.
NOT_LEARNED = {("M11", "deltaproduct")}
DP_RUNGS = ("nh1", "nh2", "nh4")          # deltaproduct: n_h Householder steps per token
BASELINE_WIDTH, BASELINE_LAYERS = 256, 4

TASKS = {"Z2": ("modular", 2), "Z16": ("modular", 16), "S3": ("perm", 3, "S"),
         "S4": ("perm", 4, "S"), "A5": ("perm", 5, "A"), "M11": ("mathieu",),
         "DFF5": ("monoid", "dff5"), "FF": ("monoid", "ff")}
RUNGS = {"Z2": ("regular",), "Z16": ("regular",), "S3": ("regular", "factored"),
         "S4": ("regular", "factored"), "A5": ("regular", "factored"), "M11": ("factored",),
         "DFF5": ("regular",), "FF": ("regular",)}
FACTORED_HEADS = {"S3": (("point", 0), ("parity",)),
                  "S4": (("point", 0), ("pairing",), ("parity",)),
                  "A5": (("point", 0), ("coset", 5)),
                  "M11": tuple(("point", i) for i in range(4))}


def n_letters(task: str) -> int:
    kind, *arg = TASKS[task]
    if kind == "modular":
        return arg[0]
    if kind == "perm":
        from nfsm.data.groups import permutation_alphabet_size
        return permutation_alphabet_size(*arg)
    if kind == "mathieu":
        return 4
    from nfsm.data.monoids import monoid_alphabet_size
    return monoid_alphabet_size(*arg)


def n_classes(task: str) -> int:
    kind, *arg = TASKS[task]
    if kind == "modular":
        return arg[0]
    if kind == "perm":
        from nfsm.data.groups import permutation_num_classes
        return permutation_num_classes(*arg)
    if kind == "mathieu":
        return 7920
    from nfsm.data.monoids import monoid_order
    return monoid_order(*arg)


def next_class(task: str) -> np.ndarray:
    """[A, C] int32: the class letter a sends class c to; class 0 is the empty prefix."""
    kind, *arg = TASKS[task]
    if kind == "modular":
        n = arg[0]
        return ((np.arange(n)[None, :] + np.arange(n)[:, None]) % n).astype(np.int32)
    if kind == "perm":
        from nfsm.data.groups import permutation_next_class
        return permutation_next_class(*arg)
    if kind == "mathieu":
        from nfsm.data.groups import mathieu_next_class
        return mathieu_next_class()
    from nfsm.data.monoids import monoid_next_class
    return monoid_next_class(*arg)


HEADS_RUNG = re.compile(r"h\d+(-\d+)*$")      # explicit head sizes, e.g. h4-5


def head_sizes(task: str, rung: str) -> tuple:
    if HEADS_RUNG.match(rung):
        return tuple(int(n) for n in rung[1:].split("-"))
    if rung == "fixed" or rung in DP_RUNGS:
        return (BASELINE_WIDTH,)
    if rung == "regular":
        return (n_classes(task),)
    if task == "M11":
        return (11,) * len(FACTORED_HEADS[task])
    from nfsm.data.groups import permutation_quotient_size
    _, g, kind = TASKS[task]
    return tuple(permutation_quotient_size(s, g, kind) for s in FACTORED_HEADS[task])


def data_fn(job: dict):
    """fn(key, B, T) -> (token ids [B, T], one-hot classes [B, T, C])."""
    kind, *arg = TASKS[job["task"]]
    if kind == "modular":
        from nfsm.data.synthetic import generate_modular_count_batch
        return lambda key, B, T: generate_modular_count_batch(key, B, arg[0], T)
    if kind == "perm":
        from nfsm.data.synthetic import generate_permutation_batch
        return lambda key, B, T: generate_permutation_batch(key, B, arg[0], T, arg[1])
    if kind == "mathieu":
        from nfsm.data.synthetic import generate_mathieu_batch
        return lambda key, B, T: generate_mathieu_batch(key, B, T)
    from nfsm.data.monoids import generate_monoid_batch
    return lambda key, B, T: generate_monoid_batch(key, B, arg[0], T)


EVAL_DRAW = {"FF": "ff01"}                 # FF is scored on a sparser draw than it trains on


def eval_data_fn(job: dict):
    """The data a run is scored on: as `data_fn`, but FF is drawn with 1% non-identity letters."""
    name = EVAL_DRAW.get(job["task"])
    if name is None:
        return data_fn(job)
    from nfsm.data.monoids import generate_monoid_batch
    return lambda key, B, T: generate_monoid_batch(key, B, name, T)


def _rungs(task: str, cell: str) -> tuple:
    return RUNGS[task] if cell == "nfsm" else DP_RUNGS if cell == "deltaproduct" else ("fixed",)


def parse_token(token: str) -> dict:
    """task,rung,cell,layers,arm,seed -> job dict."""
    task, rung, cell, layers, arm, seed = token.split(",")
    if task not in TASKS or cell not in CELLS or arm not in ARMS:
        raise ValueError(f"bad token {token!r}")
    if HEADS_RUNG.match(rung):
        if cell != "nfsm" or arm != "free":
            raise ValueError(f"rung {rung!r} is for the NFSM free arm only")
    elif rung not in _rungs(task, cell):
        raise ValueError(f"rung {rung!r} is not defined for {task} / {cell}")
    job = {"token": token, "task": task, "rung": rung, "cell": cell, "layers": int(layers),
           "arm": arm, "seed": int(seed), "head_sizes": list(head_sizes(task, rung)),
           "vocab": n_letters(task), "n_classes": n_classes(task), "conv_kernel": CONV_KERNEL,
           "val_batch": 128 if task == "M11" else 512}
    if cell == "deltaproduct":
        job["n_householder"] = int(rung[2:])
    return job


def iter_jobs(tasks=None, cells=None, arms=None, seeds=None) -> list:
    """Job tokens; NFSM only unless baselines are named in `cells`."""
    out = []
    for cell in cells or ("nfsm",):
        for task in tasks or [t for t in TASKS if (t, cell) not in NOT_LEARNED]:
            for rung in _rungs(task, cell):
                for arm in (arms or ARMS):
                    if cell != "nfsm" and arm != "free":
                        continue
                    layers = 1 if cell == "nfsm" else BASELINE_LAYERS
                    out += [f"{task},{rung},{cell},{layers},{arm},{s}" for s in seeds or SEEDS]
    return out
