# Wall-clock measurements

This benchmark measures the time and peak memory of one recurrent layer for NFSM, Mamba---,
AUSSM, PD-SSM and a scan with dense matrix merges. Every configuration runs as its own Slurm
array task, in a fresh process, on one GPU node (the reported runs used an A100 80 GB).

| file | role |
|---|---|
| `configs.py` | the configuration table, `results/walltime/configs.csv` (344 rows) |
| `run.py` | one configuration: writes `results/walltime/<raw dir>/<config_id>.json` |
| `aggregate.py` | raw JSONs to `results/walltime.csv` and `results/walltime_summary.md` (never measures) |
| `plots.py` | CSV to `figures/walltime.pdf`: time against L and time against d on the top row, memory against L centred below (numpy and matplotlib only) |

The two experiments:

- **length**: L = 2^6 .. 2^17. NFSM with one head of 16 states and with one head of 5, Mamba and
  AUSSM at d = 16 with d_state = 16, and PD-SSM at N = 16. Both implementations, modes F and FB.
  The dense reference at d = 16 is added with its parallel scan only (ids 320–343).
- **state**: L = 2^12, parallel implementation only, modes F and FB. d = 2 .. 256 for NFSM,
  Mamba, AUSSM, PD-SSM, and the dense reference h_t = A_t h_{t-1} + b_t (d × d matrices merged by
  matrix products).

In every job, B = 16, m = 256, float32 (complex64 states for AUSSM and PD-SSM), and seed 0. The
layer becomes a pure function of (params, rest, key, x), jitted once. The job makes 1 compilation
call, 5 warm-up calls and 10 timed calls, each synchronised. It asserts that the function is
traced once and that `JAX_LOG_COMPILES` logs exactly one compilation. Without `--chunked`, it also
asserts that no time chunking happens.

## Run

A configuration that runs out of memory without chunking is rerun chunked, and the chunked run
replaces it in the aggregate:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 bash scripts/dispatch_walltime.sh                 # all 344, no chunking
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 bash scripts/dispatch_walltime.sh --ids <oom ids> \
    --raw-dir results/walltime/raw_chunked --chunked                                  # the ones that ran out of memory
python -m experiments.appendix.walltime.aggregate
python -m experiments.appendix.walltime.plots [--with-forward] [--nfsm-label NFSM]    # 5.5 in x 10 cm (ICLR)
```

- `aggregate.py` reads `raw/`, then `raw_mem95/` (if present), then `raw_chunked/`. For each
  configuration the latest JSON wins.
- Chunked runs use each model's default budget, in both implementations: NFSM theta 2^26, PD-SSM
  dictionary 2^26, Mamba and AUSSM state 2^25 elements, and the dense reference 2^26 elements of
  d × d. Chunked and unchunked outputs and gradients agree to float32 round-off. Chunked runs are
  left out of the no-chunking statistics, and `plots.py` circles them in red.
- JAX exposes no allocator statistics, so peak memory is the nvidia-smi high-water mark of XLA's
  memory pool from the start of the timed calls, minus the idle value after CUDA initialisation.
  The pool is never returned, so this includes compilation and warm-up. The value is approximate.
- `XLA_PYTHON_CLIENT_MEM_FRACTION` caps what fits, and each JSON records it. A peak equal to the
  cap (`mem_at_cap`) is a lower bound.
- `plots.py` uses LaTeX when `latex` is on the PATH, so copy `results/walltime.csv` and run it
  locally.
