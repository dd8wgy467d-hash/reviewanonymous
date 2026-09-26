"""The FSA suite: group and monoid word problems on one NFSM layer (pure Python / numpy).

    task  structure                   letters                  classes  NFSM rungs (head sizes)
    Z2    cyclic Z_2                  0, 1                     2        regular (2)
    Z16   cyclic Z_16                 0..15                    16       regular (16)
    S3    symmetric S_3               adjacent transpositions  6        regular (6), factored (3, 2)
    S4    symmetric S_4               adjacent transpositions  24       regular (24), factored (4, 3, 2)
    A5    alternating A_5             adjacent 3-cycles        60       regular (60), factored (5, 12)
    M11   Mathieu M11                 2 generators, inverses   7920     factored (11, 11, 11, 11)
    DFF5  flip-flop, <= 4 identities  identity, reset, set     3        regular (3)
    FF    flip-flop, 90% identities   identity, reset, set     3        regular (3)

regular is one head of |Q| states; factored is the smallest family of quotient heads that
separates the classes (nfsm.data.groups). The baselines (mamba, aussm, pdssm) run the free arm at
width 256 over 4 layers under the rung name `fixed`.

Job token: task,rung,cell,layers,arm,seed, e.g. S4,factored,nfsm,1,free,42.
"""

import numpy as np

SUITE = "fsa"
WANDB_PROJECT = "nfsm-fsa"
TRAIN_LEN = 64
BATCH = 256
CONV_KERNEL = 0
STEPS = 100_000
SWEEP_LENS = tuple(2 ** k for k in range(3, 16))          # 8 .. 32768
SEEDS = (42, 43, 44, 45, 46)
ARMS = ("anchored", "free")
CELLS = ("nfsm", "mamba", "aussm", "pdssm")
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


def _factored(task: str) -> tuple:
    """Verified (code, table) per head of the factored family."""
    from nfsm.data.groups import mathieu_quotient_codes, permutation_quotient_codes
    if task == "M11":
        return mathieu_quotient_codes(FACTORED_HEADS[task])
    _, g, kind = TASKS[task]
    return permutation_quotient_codes(g, kind, FACTORED_HEADS[task])


def head_sizes(task: str, rung: str) -> tuple:
    if rung == "fixed":
        return (BASELINE_WIDTH,)
    if rung == "regular":
        return (n_classes(task),)
    if task == "M11":
        return (11,) * len(FACTORED_HEADS[task])
    from nfsm.data.groups import permutation_quotient_size
    _, g, kind = TASKS[task]
    return tuple(permutation_quotient_size(s, g, kind) for s in FACTORED_HEADS[task])


def anchor_codes(job: dict) -> tuple:
    """One [C] code per head: the state the head stands in for each class."""
    if job["rung"] == "regular":
        return (np.arange(job["n_classes"], dtype=np.int32),)
    return tuple(code for code, _ in _factored(job["task"]))


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


def parse_token(token: str) -> dict:
    """task,rung,cell,layers,arm,seed -> job dict."""
    task, rung, cell, layers, arm, seed = token.split(",")
    if task not in TASKS or cell not in CELLS or arm not in ARMS:
        raise ValueError(f"bad token {token!r}")
    if rung not in (RUNGS[task] if cell == "nfsm" else ("fixed",)):
        raise ValueError(f"rung {rung!r} is not defined for {task} / {cell}")
    if cell != "nfsm" and arm != "free":
        raise ValueError("only NFSM has an anchored arm")
    return {"token": token, "task": task, "rung": rung, "cell": cell, "layers": int(layers),
            "arm": arm, "seed": int(seed), "head_sizes": list(head_sizes(task, rung)),
            "vocab": n_letters(task), "n_classes": n_classes(task), "conv_kernel": CONV_KERNEL,
            "val_batch": 128 if task == "M11" else 512}


def iter_jobs(tasks=None, cells=None, arms=None, seeds=None) -> list:
    """Job tokens; NFSM only unless baselines are named in `cells`."""
    out = []
    for cell in cells or ("nfsm",):
        for task in tasks or TASKS:
            for rung in (RUNGS[task] if cell == "nfsm" else ("fixed",)):
                for arm in (arms or ARMS):
                    if cell != "nfsm" and arm != "free":
                        continue
                    layers = 1 if cell == "nfsm" else BASELINE_LAYERS
                    out += [f"{task},{rung},{cell},{layers},{arm},{s}" for s in seeds or SEEDS]
    return out
