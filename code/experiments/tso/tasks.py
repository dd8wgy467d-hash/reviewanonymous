"""TSO suite (Tracking Shuffled Objects): state tracking over English sentences.

    swap-N      alice swaps with bob . ...                   assignment starts at the identity
    parcel-N    alice has the key . ... then swap sentences  assignment declared first
    question-N  a parcel stream, then `who has the <item> ?`, answered at the last token

N = 3, 4, 5. The target is the machine state at every token. The NFSM uses 2 and 3 layers (an item
head needs the parse head of the layer below); rungs `minimal` (swap), `exact` (parcel, question)
and `holders` (top layer with the item heads only). Baselines run 4 layers of width 256; mingru and
deltaproduct run on question only unless --tasks says otherwise.

Job token: task,rung,cell,layers,arm,seed, e.g. parcel-4,exact,nfsm,2,free,42.
"""

SUITE = "tso"
WANDB_PROJECT = "nfsm-tso"
TRAIN_LEN = 256
BATCH = 64
CONV_KERNEL = 2
STEPS = 100_000
SWEEP_LENS = tuple(2 ** k for k in range(6, 16))          # 64 .. 32768
SEEDS = (42, 43, 44, 45, 46)
ARMS = ("free",)
CELLS = ("nfsm", "mamba", "aussm", "pdssm", "mingru", "deltaproduct")
DP_RUNGS = ("nh1", "nh2", "nh4")          # deltaproduct: n_h Householder steps per token
NFSM_LAYERS = (2, 3)
BASELINE_WIDTH, BASELINE_LAYERS = 256, 4
TASKS = tuple(f"{t}-{n}" for t in ("swap", "parcel", "question") for n in (3, 4, 5))
QUESTION_ONLY = ("mingru", "deltaproduct")      # default sweep: question-N only


def _task(task: str):
    """task-N -> (data module, N)."""
    from nfsm.data.tso import TASKS as MODULES
    name, n = task.split("-")
    return MODULES[name], int(n)


def _rung(task: str) -> str:
    return "minimal" if task.startswith("swap") else "exact"


def data_fn(job: dict):
    """fn(key, B, T) -> (token ids, one-hot states[, answer])."""
    mod, n = _task(job["task"])
    return lambda key, B, T: mod.generate(key, B, n, T)


def parse_token(token: str) -> dict:
    """task,rung,cell,layers,arm,seed -> job dict."""
    task, rung, cell, layers, arm, seed = token.split(",")
    if task not in TASKS or cell not in CELLS or arm not in ARMS:
        raise ValueError(f"bad token {token!r}")
    allowed = ({_rung(task)} | ({"holders"} if not task.startswith("swap") else set())
               if cell == "nfsm" else set(DP_RUNGS) if cell == "deltaproduct" else {"fixed"})
    if rung not in allowed:
        raise ValueError(f"rung {rung!r} is not defined for {task} / {cell}")
    mod, n = _task(task)
    extra = {}
    if rung == "holders":
        full = list(mod.head_sizes(n))
        extra = {"head_sizes": full[1:], "layer_head_sizes": [full] * (int(layers) - 1) + [full[1:]]}
    if cell == "deltaproduct":
        extra["n_householder"] = int(rung[2:])
    return {"token": token, "task": task, "rung": rung, "cell": cell, "layers": int(layers),
            "arm": arm, "seed": int(seed),
            "head_sizes": list(mod.head_sizes(n) if cell == "nfsm" else (BASELINE_WIDTH,)),
            "vocab": mod.alphabet(n), "n_classes": mod.n_states(n) + mod.n_answers(n),
            "conv_kernel": CONV_KERNEL, "val_batch": 512, **extra}


def iter_jobs(tasks=None, cells=None, arms=None, seeds=None) -> list:
    """Job tokens; NFSM only unless baselines are named in `cells`."""
    out = []
    for cell in cells or ("nfsm",):
        default = [t for t in TASKS if t.startswith("question")] if cell in QUESTION_ONLY else TASKS
        for task in tasks or default:
            rungs = ((_rung(task),) if cell == "nfsm" else DP_RUNGS if cell == "deltaproduct"
                     else ("fixed",))
            for rung in rungs:
                for layers in (NFSM_LAYERS if cell == "nfsm" else (BASELINE_LAYERS,)):
                    for arm in (arms or ARMS):
                        if cell != "nfsm" and arm != "free":
                            continue
                        out += [f"{task},{rung},{cell},{layers},{arm},{s}" for s in seeds or SEEDS]
    return out
