"""Exact failing length of a trained run, by bisection on the sequence length.

L_m is the smallest length where accuracy on 512 sequences drops below 0.9 (baselines) or 1.0
(NFSM). The search starts at the training length, doubles up to the cap, then bisects.

  python -m experiments.faillen fsa --list-jobs
  python -m experiments.faillen fsa --job "S4,factored,nfsm,1,free"     # all seeds of one config
  python -m experiments.faillen fsa --aggregate

Writes results/<suite>/faillen/<config>.json. --precision {fp32, fp64, bf16, fp16} re-scores the
float32 checkpoints in that arithmetic, on the same batches (fp32: no TF32), and writes to
results/<suite>/faillen_<precision>/. Overflow (inf or NaN logits) counts as wrong.
"""

import argparse
import glob
import math
import importlib
import json
import os
import statistics
import time

from experiments import runner

BATCH = 512                                  # sequences scored at every length
CAP = 1 << 20                                # length ceiling, clipped per task by the output size


class MemoryCeiling(Exception):
    """A single sequence of this length does not fit on the GPU."""


def bar_for(cell: str) -> float:
    return 1.0 if cell == "nfsm" else 0.9


def _accuracy(model, data_fn, seed: int, length: int, budget: int) -> float:
    """Answer accuracy where the task has answers, else sequence accuracy, on BATCH fresh sequences."""
    import jax
    import numpy as np
    from nfsm.eval import make_acc_fn
    acc_fn = make_acc_fn(model)
    chunk = max(1, min(BATCH, budget // length))
    while True:                                  # a shared GPU may not fit the chunk: halve and retry
        try:
            sums, done = np.zeros(3), 0
            while done < BATCH:
                bs = min(chunk, BATCH - done)
                batch = data_fn(jax.random.PRNGKey(seed * 10_000_019 + length * 1_009 + done), bs, length)
                sums += np.asarray(acc_fn(model, *batch)) * bs
                done += bs
            break
        except Exception as exc:
            if not any(m in str(exc) for m in ("RESOURCE_EXHAUSTED", "Out of memory",
                                               "All configs failed during profiling")):
                raise
            if chunk == 1:
                raise MemoryCeiling(length) from exc
            chunk = max(1, chunk // 2)
            print(f"[retry] length {length}: out of memory, chunk -> {chunk}", flush=True)
    _, seq, ans = sums / BATCH
    return float(seq if ans != ans else ans)


def failing_length(evaluate, bar: float, cap: int, start: int):
    """Smallest length >= start with accuracy < bar (doubling, then bisection); None past the cap.
    Assumes accuracy does not increase with length.
    """
    low = None                                                  # largest length known to pass
    length = start
    while length <= cap:
        if evaluate(length) < bar:
            break
        low, length = length, length * 2
    else:
        return None                                             # the cap itself passes
    high = length                                               # first length known to fail
    if low is None:
        return high                                             # fails at the training length already
    while high - low > 1:
        mid = (low + high) // 2
        if evaluate(mid) >= bar:
            low = mid
        else:
            high = mid
    return high


PRECISIONS = ("fp32", "fp64", "bf16", "fp16")
LOW = {"bf16": "bfloat16", "fp16": "float16"}


def set_precision(precision: str) -> None:
    """Global JAX settings for `precision`; call before the first computation."""
    import jax
    jax.config.update("jax_default_matmul_precision", "highest")
    if precision == "fp64":
        jax.config.update("jax_enable_x64", True)


def cast_model(model, precision: str):
    """Cast float parameters in place: to float64 / complex128 (fp64), or to bf16 / fp16 (no complex)."""
    if precision not in ("fp64", *LOW):
        return model
    import jax
    import jax.numpy as jnp
    from flax import nnx
    state = nnx.state(model, nnx.Param)
    if precision in LOW:
        low = jnp.dtype(LOW[precision])
        bad = [v.dtype for v in jax.tree.leaves(state) if jnp.issubdtype(v.dtype, jnp.complexfloating)]
        if bad:
            raise ValueError(f"{precision}: complex parameters ({bad[0]}) have no 16-bit type")
        state = jax.tree.map(lambda v: v.astype(low) if v.dtype == jnp.float32 else v, state)
        nnx.update(model, state)
        for _, mod in model.iter_modules():      # norms would otherwise reduce in float32
            if hasattr(mod, "force_float32_reductions"):
                mod.force_float32_reductions = False
        return model
    up = {jnp.dtype("float32"): jnp.float64, jnp.dtype("complex64"): jnp.complex128}
    state = jax.tree.map(lambda v: v.astype(up.get(v.dtype, v.dtype)), state)
    nnx.update(model, state)
    return model


def run_job(suite, token: str, cap: int = None, seeds=None, cap_first: bool = False,
            precision: str = None) -> dict:
    """L_m for every seed of one configuration, saving after each seed. cap_first tests the cap first,
    which settles runs that hold in one evaluation.
    """
    task, rung, cell, layers, arm = token.split(",")
    bar = bar_for(cell)
    start = suite.TRAIN_LEN
    # one sequence's logits [T, n_classes] must fit comfortably on a shared GPU (~1 GiB)
    fits = (1 << 30) // (4 * suite.parse_token(f"{token},{suite.SEEDS[0]}")["n_classes"])
    cap = min(cap or CAP, 1 << int(math.log2(max(fits, start))))
    kind = f"faillen_{precision}" if precision else "faillen"
    path = os.path.join(runner.out_dir(suite, kind), runner.job_name(token) + ".json")
    out = {"token": token, "task": task, "rung": rung, "cell": cell, "layers": int(layers),
           "arm": arm, "batch": BATCH, "bar": bar, "cap": cap, "start": start, "seeds": {}}
    if precision:
        out["precision"] = precision
    for seed in (seeds or suite.SEEDS):
        run_token = f"{token},{seed}"
        name = runner.job_name(run_token)
        ckpt = os.path.join(runner.out_dir(suite, "checkpoints"), name + ".msgpack")
        if not os.path.exists(ckpt):
            print(f"[skip] {run_token}: no checkpoint", flush=True)
            continue
        job = suite.parse_token(run_token)
        model = cast_model(runner.load_checkpoint(suite, run_token), precision)
        data_fn = getattr(suite, "eval_data_fn", suite.data_fn)(job)
        budget = runner.token_budget(job)
        trace, t0 = {}, time.time()

        def evaluate(length):
            if length not in trace:
                trace[length] = _accuracy(model, data_fn, seed, length, budget)
            return trace[length]

        reached = cap
        try:
            if cap_first:                        # the longest length that fits, tested first
                while True:
                    try:
                        evaluate(reached)
                        break
                    except MemoryCeiling:
                        reached //= 2
                        if reached < start:
                            raise
                        print(f"[{run_token}] cap does not fit, trying {reached}", flush=True)
                if trace[reached] >= bar:
                    L = None                     # passes at the ceiling: holds at every length below
                else:
                    L = failing_length(evaluate, bar, reached, start)
            else:
                L = failing_length(evaluate, bar, cap, start)
        except MemoryCeiling as ceiling:         # cannot evaluate this length: report what we reached
            L, reached = None, max((k for k in trace if trace[k] >= bar), default=start)
            print(f"[{run_token}] length {ceiling.args[0]} does not fit; ceiling {reached}", flush=True)
        part = os.path.join(runner.out_dir(suite, "parts"), name + ".json")
        exact = None
        if os.path.exists(part):
            with open(part) as f:
                exact = json.load(f).get("extraction", {}).get("exact")
        out["seeds"][str(seed)] = {"failing_length": L, "cap": reached, "exact_machine": exact,
                                   "evaluations": {str(k): round(v, 4) for k, v in sorted(trace.items())},
                                   "seconds": round(time.time() - t0, 1)}
        print(f"[{run_token}] L_m = {L if L is not None else f'>{reached}'}  "
              f"({len(trace)} evaluations, {time.time() - t0:.0f}s)", flush=True)
        _write(path, out)                        # after every seed: a timeout keeps what finished
    return out


def _write(path: str, out: dict) -> None:
    """Merge this run's seeds into the part on disk, under a lock."""
    lock = path + ".lock"
    for _ in range(600):                         # O_EXCL create is atomic on the shared filesystem
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > 60:
                    os.remove(lock)              # a job died holding it
            except FileNotFoundError:
                pass
            time.sleep(0.2)
    try:
        _merge_write(path, out)
    finally:
        try:
            os.remove(lock)
        except FileNotFoundError:
            pass


def _merge_write(path: str, out: dict) -> None:
    rec = dict(out)
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        merged = dict(old.get("seeds", {}))
        merged.update(out["seeds"])
        rec["seeds"] = merged
    rec["cap"] = max([v.get("cap", out["cap"]) for v in rec["seeds"].values()] or [out["cap"]])
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, path)                        # atomic: parallel per-seed jobs never see a half file


def list_jobs(suite, tasks=None, cells=None, arms=None) -> list:
    """Configuration tokens (no seed) that have at least one checkpoint on disk."""
    ckpt_dir = runner.out_dir(suite, "checkpoints")
    seen = []
    for run_token in suite.iter_jobs(tasks=tasks, cells=cells, arms=arms):
        token = run_token.rsplit(",", 1)[0]
        if token in seen:
            continue
        if glob.glob(os.path.join(ckpt_dir, runner.job_name(token) + "_*.msgpack")):
            seen.append(token)
    return seen


def aggregate(suite) -> str:
    """Summarise the faillen JSONs: median L_m per configuration."""
    rows = []
    for path in sorted(glob.glob(os.path.join(runner.out_dir(suite, "faillen"), "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        per_seed = [(v["failing_length"], v.get("cap", rec["cap"])) for v in rec["seeds"].values()]
        if not per_seed:
            continue
        shown = [str(L) if L is not None else f">{c}" for L, c in per_seed]
        med = statistics.median([L if L is not None else c * 2 for L, c in per_seed])
        med = f">{rec['cap']}" if med > rec["cap"] else f"{med:g}"
        ex = [v["exact_machine"] for v in rec["seeds"].values() if v["exact_machine"] is not None]
        rows.append(f"| {rec['token']} | {rec['bar']:g} | {med} | {', '.join(shown)} | "
                    f"{f'{sum(map(bool, ex))}/{len(ex)}' if ex else '-'} |")
    text = "\n".join([f"# {suite.SUITE}: failing length L_m (batch {BATCH}, bisection from "
                      f"the training length {suite.TRAIN_LEN})", "",
                      "| config | bar | median L_m | per seed | exact machine |",
                      "|---|---:|---:|---|---:|"] + rows) + "\n"
    with open(os.path.join(runner.RESULTS_ROOT, suite.SUITE, "faillen_summary.md"), "w") as f:
        f.write(text)
    print(text, flush=True)
    return text


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="exact failing length by bisection")
    p.add_argument("suite", choices=("fsa", "tso"))
    p.add_argument("--list-jobs", action="store_true")
    p.add_argument("--job", metavar="TOKEN", help="one configuration: task,rung,cell,layers,arm")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--cells", nargs="+")
    p.add_argument("--arms", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--cap", type=int, help="length ceiling (default 2^20)")
    p.add_argument("--precision", choices=PRECISIONS,
                   help="re-score in this arithmetic (default: JAX's float32 defaults)")
    p.add_argument("--cap-first", action="store_true",
                   help="test the cap first; only search below it if that fails")
    args = p.parse_args()
    suite = importlib.import_module(f"experiments.{args.suite}.tasks")
    if args.list_jobs:
        for token in list_jobs(suite, args.tasks, args.cells, args.arms):
            print(token)
    elif args.job:
        if args.precision:
            set_precision(args.precision)
        run_job(suite, args.job, cap=args.cap, seeds=args.seeds, cap_first=args.cap_first,
                precision=args.precision)
    elif args.aggregate:
        aggregate(suite)
    else:
        p.error("one of --list-jobs, --job, --aggregate is required")
