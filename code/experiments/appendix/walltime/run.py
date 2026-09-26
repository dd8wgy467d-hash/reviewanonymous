"""Wall-clock time and peak memory of one recurrent layer, without time chunking.

    python -m experiments.appendix.walltime.run [--config-id N] [--set L=64 B=2] [--raw-dir DIR]

With --chunked, the model's default time-chunking budget is used instead (for configurations
that do not fit without it); `chunked` and `chunk_len` are recorded.

One configuration per process: a row of results/walltime/configs.csv, by default row
$SLURM_ARRAY_TASK_ID. The layer becomes a pure function of (params, rest, key, x) through
nnx.split / nnx.merge, jitted with jax.jit. The first call compiles it, 5 calls warm it up and 10
calls are timed, each one synchronised. The job asserts that there is no time chunking, that the
function is traced once, and that JAX_LOG_COMPILES logs exactly one compilation of it. The result,
with status ok | oom | failed, goes to results/walltime/raw/<config_id>.json.

Modes: F is the forward pass (NFSM and PD-SSM deterministic); FB is jax.value_and_grad of
mean(out^2) with respect to the parameters (NFSM with Gumbel noise from a key passed as an argument,
PD-SSM with its surrogate term).

Peak memory: JAX exposes no allocator statistics, so this is measured from outside, as the maximum
of nvidia-smi's memory.used on this job's GPU -- sampled every 50 ms from the start of the timed
calls, plus one reading right after them -- minus its value after CUDA initialisation and before
the layer is built. With XLA_PYTHON_CLIENT_PREALLOCATE false the allocator grows a pool and does
not return it, so this is the high-water mark of that pool, compilation and warm-up included.
"""

import argparse
import json
import logging
import os
import platform
import hashlib
import re
import socket
import subprocess
import sys
import threading
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_LOG_COMPILES", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from experiments.appendix.walltime.configs import CONFIGS_CSV, INT_FIELDS, RAW_DIR, read

SEED = 0
NO_BUDGET = 2 ** 62
N_WARMUP = 5
N_TIMED = 10
SAMPLE_MS = 50
FN_NAME = "timed_call"
ENV_KEYS = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_ALLOCATOR", "JAX_LOG_COMPILES", "XLA_FLAGS", "JAX_PLATFORMS",
            "CUDA_VISIBLE_DEVICES", "NFSM_PY", "VIRTUAL_ENV", "LOADEDMODULES")


# --- GPU: identity and memory -------------------------------------------------------------------

def _smi(*args):
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                              timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def anon(value: str) -> str:
    """A stable, non-identifying surrogate for `value`.

    Node names, GPU UUIDs and interpreter paths reach the aggregated summary, which is shared
    alongside the paper. They are only ever compared for equality there (how many distinct GPUs,
    did every job run on the same node), so a digest preserves every check while carrying no
    site, account or user name.
    """
    return "anon-" + hashlib.sha256(value.encode()).hexdigest()[:12] if value else ""


def gpu_info():
    """This job's GPU: name, uuid, memory (GB), driver and CUDA versions; empty without nvidia-smi."""
    lines = _smi("--query-gpu=index,name,uuid,memory.total,driver_version",
                 "--format=csv,noheader,nounits").strip().splitlines()
    rows = [[c.strip() for c in ln.split(",")] for ln in lines if ln.strip()]
    if not rows:
        return {"gpu": "", "gpu_uuid": "", "gpu_mem_gb": None, "driver": "", "cuda_driver": ""}
    pick = rows[0]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]
    for r in rows if len(rows) > 1 else []:
        if visible in (r[0], r[2]):
            pick = r
    cuda = re.search(r"CUDA Version\s*:\s*([\d.]+)", _smi("-q"))
    return {"gpu": pick[1], "gpu_uuid": pick[2], "gpu_mem_gb": round(float(pick[3]) / 1024, 1),
            "driver": pick[4], "cuda_driver": cuda.group(1) if cuda else ""}


def memory_used_mb(uuid):
    out = _smi(f"--id={uuid}", "--query-gpu=memory.used", "--format=csv,noheader,nounits").strip()
    try:
        return float(out.splitlines()[0])
    except (IndexError, ValueError):
        return None


class MemorySampler:
    """nvidia-smi memory.used of one GPU every SAMPLE_MS ms, as (perf_counter, MB) pairs."""

    def __init__(self, uuid):
        self.uuid, self.samples, self.proc = uuid, [], None

    def __enter__(self):
        if self.uuid:
            self.proc = subprocess.Popen(
                ["nvidia-smi", f"--id={self.uuid}", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", f"--loop-ms={SAMPLE_MS}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
            deadline = time.perf_counter() + 5
            while not self.samples and time.perf_counter() < deadline:
                time.sleep(0.01)
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append((time.perf_counter(), float(line.strip())))
            except ValueError:
                pass

    def __exit__(self, *exc):
        if self.proc is not None:
            time.sleep(3 * SAMPLE_MS / 1000)
            self.proc.terminate()
            self.proc.wait()
            self.thread.join(timeout=2)
        return False

    def max_since(self, t0):
        """Largest sample read at or after t0 (lines arrive buffered, so read times lag)."""
        vals = [mb for t, mb in self.samples if t >= t0]
        return max(vals) if vals else None


class CompileLog(logging.Handler):
    """The JAX_LOG_COMPILES records (tracing and compilation messages)."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        msg = record.getMessage()
        if "Compiling" in msg or "Finished" in msg:
            self.lines.append(msg[:1000])

    def count(self, prefix):
        return sum(1 for ln in self.lines if ln.startswith(f"{prefix} jit({FN_NAME}) "))

    def ours(self):
        return [ln for ln in self.lines if f"jit({FN_NAME})" in ln or f" {FN_NAME} for jit" in ln]


# --- the layers ---------------------------------------------------------------------------------

import jax
import jax.numpy as jnp
import flax
from flax import nnx

import nfsm.cells.ssm as ssm
from nfsm.cells import AUSSM, NFSM, PDSSM, Mamba
from nfsm.cells.nfsm import THETA_BUDGET, _chunk_len, _from_chunks, _to_chunks
from nfsm.cells.ssm import DICT_BUDGET

DEFAULT_BUDGETS = {"nfsm": THETA_BUDGET, "pdssm": DICT_BUDGET, "mamba": ssm.STATE_BUDGET,
                   "aussm": ssm.STATE_BUDGET,   # the models' own time-chunking budgets (--chunked)
                   "dense": DICT_BUDGET}        # dense reference: PD-SSM's budget of d x d elements
ssm.STATE_BUDGET = NO_BUDGET          # read by Mamba / AUSSM at trace time: no time chunking


def _dense_op(left, right):
    return right[0] @ left[0], (right[0] @ left[1][..., None])[..., 0] + right[1]


class DenseAffine(nnx.Module):
    """h_t = A_t h_{t-1} + b_t from h_0 = 0, dense A_t = (I + tanh(A_proj x_t) / d) / 2 (norm at
    most 1) and b_t = b_proj x_t, by an associative scan merging (A2 A1, A2 b1 + b2). When B C d^2
    exceeds `budget`, time is processed C steps at a time (rematerialised), the state carried over."""

    def __init__(self, d, m, rngs, budget=NO_BUDGET):
        self.d, self.budget = d, budget
        self.A_proj = nnx.Linear(m, d * d, rngs=rngs)
        self.b_proj = nnx.Linear(m, d, rngs=rngs)

    def _coeffs(self, x):
        B, T, d = x.shape[0], x.shape[1], self.d
        A = 0.5 * (jnp.eye(d, dtype=x.dtype) + jnp.tanh(self.A_proj(x)).reshape(B, T, d, d) / d)
        return A, self.b_proj(x)

    def chunk_len(self, B, T):
        return _chunk_len(T, B * self.d * self.d, self.budget)

    def __call__(self, x):
        B, T = x.shape[0], x.shape[1]
        C = self.chunk_len(B, T)
        if C >= T:
            return jax.lax.associative_scan(_dense_op, self._coeffs(x), axis=1)[1]

        @jax.checkpoint
        def body(h_prev, x_c):
            cum_A, h = jax.lax.associative_scan(_dense_op, self._coeffs(x_c), axis=1)
            h = h + (cum_A @ h_prev[:, None, :, None])[..., 0]
            return h[:, -1], h

        h0 = jnp.zeros((B, self.d), x.dtype)
        _, h = jax.lax.scan(body, h0, _to_chunks(x, C))
        return _from_chunks(h)


def build_layer(cfg, chunked=False):
    """(layer, chunk length, budget). Without `chunked`, no time chunking (asserted); with it, the
    model's own default budget, and the chunk length it implies is recorded."""
    model, d, L, B, m = cfg["model"], cfg["d"], cfg["L"], cfg["B"], cfg["m"]
    budget = DEFAULT_BUDGETS.get(model, NO_BUDGET) if chunked else NO_BUDGET
    if model == "nfsm":
        layer = NFSM((d,), m, nnx.Rngs(params=SEED, noise=SEED + 1), theta_budget=budget)
        chunk = _chunk_len(L, layer._spec.theta_dim * B, layer.theta_budget)
    elif model == "pdssm":
        layer = PDSSM(d, m, nnx.Rngs(SEED), dict_budget=budget)
        chunk = _chunk_len(L, B * layer.N * layer.N, layer.dict_budget)
    elif model in ("mamba", "aussm"):
        ssm.STATE_BUDGET = budget                         # read at trace time, after this
        cls = Mamba if model == "mamba" else AUSSM
        layer = cls(d, m, nnx.Rngs(SEED), d_state=cfg["d_state"], d_conv=0)
        chunk = ssm._scan_chunk_len(B, L, d * cfg["d_state"], ssm.STATE_BUDGET)
    elif model == "dense":
        assert cfg["impl"] == "parallel", "the dense reference has only a parallel scan"
        layer = DenseAffine(d, m, nnx.Rngs(SEED), budget=budget)
        chunk = layer.chunk_len(B, L)
    else:
        raise ValueError(f"unknown model {model!r}")
    if not chunked:
        assert chunk == L, f"time chunking would happen: chunk length {chunk} < L = {L}"
    return layer, chunk, budget


def make_timed_fn(cfg, graphdef, counter):
    """The jitted pure function of (params, rest, key, x): out (F) or (loss, grads) (FB)."""
    model, parallel, fb = cfg["model"], cfg["impl"] == "parallel", cfg["mode"] == "FB"

    def apply(params, rest, key, x):
        layer = nnx.merge(graphdef, params, rest)
        if model == "nfsm":
            if fb:
                layer.rngs = nnx.Rngs(noise=key)
            return layer(x, deterministic=not fb, scan_impl="flat" if parallel else "sequential")
        if model == "pdssm":
            return layer(x, deterministic=not fb, parallel_mode=parallel)
        if model == "dense":
            return layer(x)
        return layer(x, parallel_mode=parallel)

    def timed_call(params, rest, key, x):
        counter[0] += 1                                     # Python side effect: runs only when traced
        if not fb:
            return apply(params, rest, key, x)
        return jax.value_and_grad(lambda p: jnp.mean(apply(p, rest, key, x) ** 2))(params)

    assert timed_call.__name__ == FN_NAME
    return jax.jit(timed_call)


def _sync(out):
    if isinstance(out, tuple):
        float(out[0])
    for leaf in jax.tree_util.tree_leaves(out):
        leaf.block_until_ready()


def _is_oom(e):
    s = f"{type(e).__name__}: {e}"
    return any(k in s for k in ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory", "OOM"))


# --- one configuration --------------------------------------------------------------------------

def measure(cfg, rec, log, gpu_uuid):
    jnp.zeros(()).block_until_ready()                       # CUDA context, before the layer
    rec["mem_idle_mb"] = memory_used_mb(gpu_uuid) if gpu_uuid else None
    rec["devices"] = [str(dv) for dv in jax.devices()]

    layer, rec["chunk_len"], rec["chunk_budget"] = build_layer(cfg, rec["chunked"])
    graphdef, params, rest = nnx.split(layer, nnx.Param, ...)
    rng = np.random.default_rng(SEED)
    x = jnp.asarray(rng.standard_normal((cfg["B"], cfg["L"], cfg["m"]), dtype=np.float32))
    keys = [jnp.asarray(k) for k in np.asarray(jax.random.split(jax.random.PRNGKey(SEED + 2),
                                                                1 + N_WARMUP + N_TIMED))]
    counter = [0]
    fn = make_timed_fn(cfg, graphdef, counter)

    with MemorySampler(gpu_uuid) as sampler:
        t0 = time.perf_counter()
        _sync(fn(params, rest, keys[0], x))
        rec["compile_ms"] = (time.perf_counter() - t0) * 1e3
        assert counter[0] == 1, f"n_traces = {counter[0]} after the compilation call, expected 1"
        for i in range(N_WARMUP):
            _sync(fn(params, rest, keys[1 + i], x))
        assert counter[0] == 1, f"n_traces = {counter[0]} after warm-up (retraced)"
        times = []
        t_start = time.perf_counter()
        for i in range(N_TIMED):
            t = time.perf_counter()
            _sync(fn(params, rest, keys[1 + N_WARMUP + i], x))
            times.append((time.perf_counter() - t) * 1e3)
        t_end = time.perf_counter()
        time.sleep(2 * SAMPLE_MS / 1000)
    rec["n_traces"] = counter[0]
    assert counter[0] == 1, f"n_traces = {counter[0]} after the timed calls (retraced)"

    q25, med, q75 = np.percentile(times, [25, 50, 75])
    rec.update({"times_ms": times, **{f"t{i + 1}": v for i, v in enumerate(times)},
                "median_ms": float(med), "iqr_ms": float(q75 - q25),
                "min_ms": float(np.min(times)), "max_ms": float(np.max(times))})

    end_used = memory_used_mb(gpu_uuid) if gpu_uuid else None
    during = [v for v in (sampler.max_since(t_start), end_used) if v is not None]
    rec["mem_samples"] = len(sampler.samples)
    rec["mem_max_all_mb"] = max((mb for _, mb in sampler.samples), default=None)
    rec["mem_end_mb"] = end_used
    rec["timed_window_ms"] = (t_end - t_start) * 1e3
    if during and rec["mem_idle_mb"] is not None:
        rec["peak_mem_mb"] = max(during) - rec["mem_idle_mb"]
        rec["mem_method"] = (f"nvidia-smi memory.used sampled every {SAMPLE_MS} ms from the start of "
                             "the timed calls, plus one reading right after them; max minus the "
                             "value after CUDA init and before building the layer")
    else:
        rec["peak_mem_mb"] = None
        rec["mem_method"] = "unavailable (no nvidia-smi samples)"

    n_compiling, n_finished = log.count("Compiling"), log.count("Finished XLA compilation of")
    rec["compile_log"] = log.ours()
    rec["n_compile_logs"] = n_compiling
    assert n_compiling == 1, (f"JAX_LOG_COMPILES shows {n_compiling} compilation(s) of {FN_NAME} "
                              f"({n_finished} finished), expected exactly 1")


def run(cfg, raw_dir, chunked=False):
    log = CompileLog()
    logging.getLogger("jax").addHandler(log)
    gpu = gpu_info()
    gpu_uuid = gpu["gpu_uuid"]                       # real: nvidia-smi is queried with it
    gpu = {**gpu, "gpu_uuid": anon(gpu_uuid)}        # recorded: a surrogate, see anon()
    job = os.environ.get("SLURM_ARRAY_JOB_ID")
    rec = {**cfg, "status": "failed", "reason": "",
           "slurm_job_id": (f"{job}_{os.environ.get('SLURM_ARRAY_TASK_ID')}" if job
                            else os.environ.get("SLURM_JOB_ID", "local")),
           "slurm_raw_job_id": os.environ.get("SLURM_JOB_ID", ""),
           "node": anon(os.environ.get("SLURMD_NODENAME", socket.gethostname())), **gpu,
           "cuda": f"driver {gpu['cuda_driver'] or '?'}; runtime {_cuda_runtime()}",
           "jax_version": jax.__version__, "flax_version": flax.__version__,
           "python": platform.python_version(), "interpreter": os.path.basename(sys.executable),
           "env": {k: os.environ.get(k) for k in ENV_KEYS},
           "n_warmup": N_WARMUP, "n_timed": N_TIMED, "seed": SEED, "no_chunk_budget": NO_BUDGET,
           "chunked": chunked,
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"[walltime] config {cfg['config_id']}: {cfg}", flush=True)
    try:
        measure(cfg, rec, log, gpu_uuid)
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "oom" if _is_oom(e) else "failed"
        rec["reason"] = f"{type(e).__name__}: {str(e)[:2000]}"
        rec["traceback"] = traceback.format_exc()[-4000:]
        rec.setdefault("compile_log", log.ours())
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{cfg['config_id']}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[walltime] config {cfg['config_id']}: {rec['status']} "
          f"median {rec.get('median_ms')} ms, peak {rec.get('peak_mem_mb')} MB, "
          f"compile {rec.get('compile_ms')} ms {rec['reason'][:300]} -> {path}", flush=True)
    return rec


def _cuda_runtime():
    try:
        import importlib.metadata as md
        return md.version("nvidia-cuda-runtime-cu12")
    except Exception:
        return "?"


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config-id", type=int, default=None, help="default: $SLURM_ARRAY_TASK_ID")
    p.add_argument("--configs", default=CONFIGS_CSV)
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--chunked", action="store_true",
                   help="use the model's default time-chunking budget instead of none")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override fields of the row (tests only, e.g. L=64 B=2)")
    args = p.parse_args()
    cid = args.config_id if args.config_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    table = {r["config_id"]: r for r in read(args.configs)}
    cfg = dict(table[cid])
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = int(v) if k in INT_FIELDS or k == "d_state" else v
    rec = run(cfg, args.raw_dir, args.chunked)
    sys.exit(0 if rec["status"] in ("ok", "oom") else 1)


if __name__ == "__main__":
    main()
