"""Per-step disagreement of the associative scan with the sequential scan, on the hidden state.

    python -m experiments.appendix.parallelism.drift --list-jobs | --job TOKEN | --all

Token: model,dtype,seed. For each baseline (Mamba, AU-SSM, PD-SSM), number format and seed, the
cell (init from the seed) reads a batch of B = 64 modular-count words of length T = 2^15 (sampled
from the seed). Its transition drives (a_t, b_t) are computed once, in float32, and rounded to the
scan dtype; both scans then run on the same drives, so the only difference between them is the
order of the operations. Per word and per step,

    e_t = max_i |h_t^par[i] - h_t^seq[i]|        (i over the state coordinates, complex modulus)

and each job writes the batch mean, median, min and max of e_t for every t, together with the
batch mean of max_i |h_t^seq[i]| (the state scale), to results/parallelism/drift/<dtype>/.

AU-SSM and PD-SSM have complex states and there is no complex float16. Their float16 runs carry
the real and imaginary parts as two float16 arrays, with the complex products written out
(complex_emulated = true in the JSON); their float32 runs use complex64 and the cell's own scans.
Mamba always uses the cell's own scans. Figures: drift_plots.py.
"""

import argparse
import json
import os
import sys
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from nfsm import device
device.configure()

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from nfsm.cells import AUSSM, PDSSM, Mamba
from nfsm.cells.ssm import (_affine_parallel_scan, _affine_sequential_scan, _pd_parallel_scan,
                            _pd_sequential_scan, _scatter_last)
from nfsm.data.synthetic import generate_modular_count_batch

T = 2 ** 15
B = 64
BATCH_CHUNK = 8                       # words scanned at once (bounds the associative scan's memory)
D = 32                                # state width, as the main precision figure
M = 256                               # cell input width
TASK_N = 5                            # modular-count symbols driving the input
MODELS = ["mamba", "aussm", "pdssm"]
DTYPES = {"float32": jnp.float32, "float16": jnp.float16}
SEEDS = [0, 1, 2, 3, 4]
DRIFT_DIR = os.path.join(_ROOT, "results", "parallelism", "drift")


def make_batch(seed):
    """[B, T, M]: modular-count words from `seed` through a fixed random projection."""
    k_sym, k_proj = jax.random.split(jax.random.PRNGKey(seed + 12345))
    symbols, _ = generate_modular_count_batch(k_sym, B, TASK_N, T, False)
    return symbols @ (jax.random.normal(k_proj, (TASK_N, M)) * TASK_N ** -0.5)


def make_cell(model, seed):
    return {"mamba": Mamba, "aussm": AUSSM, "pdssm": PDSSM}[model](D, M, nnx.Rngs(seed))


# --- drives, as the cells compute them ---

def _selective_drive(cell, x):
    """(a, drive) [b, T, d, n] of Mamba / AU-SSM, from the block input x [b, T, M]."""
    u, _ = jnp.split(cell.in_proj(x), 2, axis=-1)
    u = cell.conv(u) if cell.conv is not None else jax.nn.silu(u)
    a, bus, _ = cell._drive(u)
    return a, bus


def _pd_drive(cell, x):
    """(idx, diag, b) [b, T, N] of PD-SSM (the deterministic pass: no surrogate term)."""
    diag, b = cell._diag_drive(x)
    return cell._idx_over_time(x), diag, b


# --- complex arithmetic on (re, im) pairs, for float16 ---

def _cmul(p, q):
    return p[0] * q[0] - p[1] * q[1], p[0] * q[1] + p[1] * q[0]


def _cadd(p, q):
    return p[0] + q[0], p[1] + q[1]


def _pair(z, dtype):
    return jnp.real(z).astype(dtype), jnp.imag(z).astype(dtype)


def _pair_affine_parallel(a, b):
    """`_affine_parallel_scan` on (re, im) pairs."""
    def op(left, right):
        (al, bl), (ar, br) = (left[:2], left[2:]), (right[:2], right[2:])
        return (*_cmul(al, ar), *_cadd(_cmul(ar, bl), br))
    out = jax.lax.associative_scan(op, (*a, *b), axis=1)
    return out[2], out[3]


def _pair_affine_sequential(a, b):
    """`_affine_sequential_scan` on (re, im) pairs."""
    def step(h, inp):
        h = _cadd(_cmul(inp[:2], h), inp[2:])
        return h, h
    h0 = (jnp.zeros_like(b[0][:, 0]),) * 2
    _, h = jax.lax.scan(step, h0, tuple(jnp.moveaxis(v, 1, 0) for v in (*a, *b)))
    return tuple(jnp.moveaxis(v, 0, 1) for v in h)


def _pair_pd_apply(idx, diag, v):
    prod = _cmul(diag, v)
    return _scatter_last(idx, prod[0]), _scatter_last(idx, prod[1])


def _pair_pd_parallel(idx, diag, b):
    """`_pd_parallel_scan` on (re, im) pairs."""
    def op(left, right):
        il, dl, bl = left[0], left[1:3], left[3:]
        ir, dr, br = right[0], right[1:3], right[3:]
        take = lambda v: jnp.take_along_axis(v, il, axis=-1)          # noqa: E731
        return (take(ir), *_cmul((take(dr[0]), take(dr[1])), dl),
                *_cadd(_pair_pd_apply(ir, dr, bl), br))
    out = jax.lax.associative_scan(op, (idx, *diag, *b), axis=1)
    return out[3], out[4]


def _pair_pd_sequential(idx, diag, b):
    """`_pd_sequential_scan` on (re, im) pairs."""
    def step(h, inp):
        h = _cadd(_pair_pd_apply(inp[0], inp[1:3], h), inp[3:])
        return h, h
    h0 = (jnp.zeros_like(b[0][:, 0]),) * 2
    _, h = jax.lax.scan(step, h0, tuple(jnp.moveaxis(v, 1, 0) for v in (idx, *diag, *b)))
    return tuple(jnp.moveaxis(v, 0, 1) for v in h)


# --- the two scans on one batch chunk ---

def _states(cell, x, model, dtype):
    """(h_par, h_seq), each an array or an (re, im) pair, [b, T, state...]."""
    emulate = dtype == jnp.float16 and model != "mamba"
    if model == "pdssm":
        idx, diag, b = _pd_drive(cell, x)
        if emulate:
            diag, b = _pair(diag, dtype), _pair(b, dtype)
            return _pair_pd_parallel(idx, diag, b), _pair_pd_sequential(idx, diag, b)
        return _pd_parallel_scan(idx, diag, b), _pd_sequential_scan(idx, diag, b)
    a, bus = _selective_drive(cell, x)
    if emulate:
        a, bus = _pair(a, dtype), _pair(bus, dtype)
        return _pair_affine_parallel(a, bus), _pair_affine_sequential(a, bus)
    if model == "mamba":
        a, bus = a.astype(dtype), bus.astype(dtype)
    return _affine_parallel_scan(a, bus), _affine_sequential_scan(a, bus)


def _modulus(h):
    """|h| in float32, for a real array, a complex array or an (re, im) pair."""
    if isinstance(h, tuple):
        re, im = (v.astype(jnp.float32) for v in h)
        return jnp.sqrt(re * re + im * im)
    return jnp.abs(h.astype(jnp.complex64 if jnp.iscomplexobj(h) else jnp.float32))


def _sub(p, q):
    if isinstance(p, tuple):
        return tuple(a.astype(jnp.float32) - b.astype(jnp.float32) for a, b in zip(p, q))
    up = jnp.complex64 if jnp.iscomplexobj(p) else jnp.float32
    return p.astype(up) - q.astype(up)


def per_step_fn(model, dtype):
    """Jitted (cell, x [b, T, M]) -> (e_t [b, T], max_i |h_t^seq[i]| [b, T])."""
    def fn(cell, x):
        h_par, h_seq = _states(cell, x, model, dtype)
        state_axes = lambda v: tuple(range(2, v.ndim))                  # noqa: E731
        diff = _modulus(_sub(h_par, h_seq))
        scale = _modulus(h_seq)
        return jnp.max(diff, axis=state_axes(diff)), jnp.max(scale, axis=state_axes(scale))
    return nnx.jit(fn)


# --- jobs ---

def _stats(e):
    """Batch statistics of [B, T] per-word values, NaN where a step has no finite value."""
    finite = np.where(np.isfinite(e), e, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {"mean": np.nanmean(finite, 0), "median": np.nanmedian(finite, 0),
                "min": np.nanmin(finite, 0), "max": np.nanmax(finite, 0)}


def _to_list(v):
    return [None if not np.isfinite(x) else float(f"{x:.6g}") for x in np.asarray(v, np.float64)]


def part_path(model, dtype_name, seed):
    return os.path.join(DRIFT_DIR, dtype_name, f"{model}_seed{seed}.json")


def run_job(token):
    model, dtype_name, seed = token.split(",")
    seed, dtype = int(seed), DTYPES[dtype_name]
    t0 = time.perf_counter()
    cell, x = make_cell(model, seed), make_batch(seed)
    fn = per_step_fn(model, dtype)
    errs, scales = [], []
    for i in range(0, B, BATCH_CHUNK):
        e, s = fn(cell, x[i:i + BATCH_CHUNK])
        errs.append(np.asarray(e, np.float32))
        scales.append(np.asarray(s, np.float32))
    e, s = np.concatenate(errs), np.concatenate(scales)
    stats = _stats(e)
    rec = {"model": model, "dtype": dtype_name, "seed": seed, "T": T, "B": B, "d": D, "m": M,
           "task": f"Z{TASK_N} modular count", "drive_dtype": "float32",
           "complex_emulated": dtype == jnp.float16 and model != "mamba",
           "metric": "e_t = max_i |h_t^par[i] - h_t^seq[i]| per word; batch statistics per t",
           "n_nonfinite": int((~np.isfinite(e)).sum()),
           "max_overall": float(np.nanmax(np.where(np.isfinite(e), e, np.nan))),
           "mean_overall": float(np.nanmean(np.where(np.isfinite(e), e, np.nan))),
           "exact_fraction": float((e == 0).mean()),
           "seconds": round(time.perf_counter() - t0, 2),
           **{f"diff_{k}": _to_list(v) for k, v in stats.items()},
           "state_mean": _to_list(_stats(s)["mean"])}
    path = part_path(model, dtype_name, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f)
    print(f"[drift] {token}: max {rec['max_overall']:.3e} mean {rec['mean_overall']:.3e} "
          f"nonfinite {rec['n_nonfinite']} in {rec['seconds']} s -> {path}", flush=True)
    return rec


def iter_jobs(models, dtypes, seeds):
    for dtype_name in dtypes:
        for model in models:
            for seed in seeds:
                yield f"{model},{dtype_name},{seed}"


def main():
    p = argparse.ArgumentParser(description="Per-step sequential vs associative scan drift.")
    p.add_argument("--list-jobs", action="store_true")
    p.add_argument("--job", metavar="TOKEN")
    p.add_argument("--all", action="store_true", help="every job in this process")
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--dtypes", nargs="+", choices=list(DTYPES), default=list(DTYPES))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = p.parse_args()
    jobs = list(iter_jobs(args.models, args.dtypes, args.seeds))
    if args.list_jobs:
        print("\n".join(jobs))
    elif args.job:
        device.print_devices("drift")
        run_job(args.job)
    elif args.all:
        device.print_devices("drift")
        for tok in jobs:
            run_job(tok)
    else:
        p.error("choose --list-jobs, --job TOKEN or --all")


if __name__ == "__main__":
    main()
