# Neural finite state machines

This is the code for the submission *Length-Independent State Tracking Under a Parallel Scan*.

The neural finite state machine (NFSM) is a recurrent block whose reachable states are a finite
set of code points. At each step it applies a transition table read off input-dependent logits:
column j of `theta_x` sends state j to `argmax_l theta_x[l, j]`. Each step is therefore an index
map, and a sequence of steps is composed by a parallel prefix scan. A layer holds K such heads,
of possibly different sizes.

## Layout

```
nfsm/                     the method
  cells/nfsm.py           NFSM cell: argmax transitions, Gumbel exploration, straight-through surrogate, scans
  cells/ssm.py            baselines: Mamba (eigenvalues in (-1, 1]), AUSSM, PD-SSM
  cells/mingru.py         baseline: minGRU with negative eigenvalues (decay 1 - 2 z_t in (-1, 1))
  cells/delta.py          baseline: DeltaProduct[-1, 1] with n_h Householder steps (n_h = 1: DeltaNet)
  backbone.py             shared model: embedding -> L x [norm, conv, cell, MLP] -> read-out
  training/               the recipe (TrainConfig), the training loop, loss weights, optimizer
  eval.py                 accuracy and the length sweep
  data/                   group and monoid word problems, TSO streams
  palette.py              one style per model, shared by the figures
experiments/
  runner.py               command line shared by the two suites
  faillen.py              exact failing length L_max of a trained run, by bisection (tab:results, tab:seeds);
                          --precision re-scores a float32 checkpoint in bf16, fp16 or strict fp32
  fsa/                    algebraic state tracking: Z2, Z16, S3, S4, A5, M11, DFF5, FF
  tso/                    Tracking Shuffled Objects: swap, parcel, question
  kws/                    spoken digits (Speech Commands v0.02, MFCC), one-second clips: stage 1 of KWS Stream
  kws_stream/             a spoken digit inside long recorded noise, only the final class supervised (Section 7.2)
  pretrained_mamba/       released Mamba-2 checkpoints on RULER needle retrieval, as is and with a decay floor
                          (Section 7.2; own environment, see its README)
  appendix/parallelism/   parallel vs sequential scan drift on the baselines (fig:seqpar)
  appendix/ssms_fp/       the rotation SSM under finite precision (fig:drift)
  appendix/walltime/      wall-clock and peak memory of one layer (fig:walltime)
scripts/                  slurm template, dispatchers, environment, figure and summary builders, clean-up
  fsa_headsearch.py       random NFSM head sizes on the algebraic tasks (Appendix H.6)
  kws_hpsearch.py         random training hyperparameters on KWS Stream (Appendix I.1)
  analyse_kws_stream.py   KWS Stream figures; make_*_figure.py build the other new figures
```

## Training

Both suites share the model, the recipe (`nfsm.training.TrainConfig`), checkpointing and logging.
They differ only in their tasks, training length, batch, conv taps and W&B project.

The loss is a cross-entropy at every position. Past each sequence's first error t* its weight
decays as exp(−3 (t − t*)). Each position is also weighted by the inverse frequency of its class
in the batch.

The NFSM transition learns through a straight-through surrogate on the column it actually used,
with Gumbel sampling at temperature 0.5. An exploration term, β × excess at t*, keeps the used
columns flippable.

The training steps use AdamW with lr 1e-3, a cosine decay to 1e-4, weight decay 1e-4 and dropout
0.1. Training stops after 40 consecutive checks at validation sequence accuracy 1.0, with a
100k-step budget, and the checkpoint with the best validation accuracy is kept. Both use the
network's own output. The loss is the one above, with β = 0.03 (the `free` field of job tokens).

## Setup

```bash
uv sync --extra cuda      # GPU cluster; --extra mps on Apple silicon, --extra cpu elsewhere
```

The scripts use `.venv/bin/python` when it exists (`scripts/nfsm_env.sh`).

The dispatchers need bash 4 or newer (they use `mapfile` and associative arrays).

`scripts/slurm_gpu.sh` is a site-neutral SLURM template. Scheduler settings are read from the
environment rather than hard-coded, so export `SBATCH_ACCOUNT`, `SBATCH_PARTITION` and
`SBATCH_QOS` as your cluster requires before dispatching. If your site provides a conda
environment instead of a local `.venv`, export `NFSM_CONDA_ENV=<name>`.

Logging to Weights & Biases is on by default; pass `--no-wandb` (or set `WANDB_MODE=offline`) to
run without it.

## Run

```bash
python -m experiments.fsa.run --list-jobs                       # job tokens: task,rung,cell,layers,arm,seed
python -m experiments.fsa.run --job S4,factored,nfsm,1,free,42  # one configuration
bash scripts/dispatch.sh fsa [--dry-run] [--tasks S4 A5] [--arms free] [--cells mamba]
STEPS=25000 bash scripts/dispatch.sh tso                         # STEPS overrides the step budget
python -m experiments.fsa.run --aggregate                       # results/fsa/summary.md and seq_acc.pdf
bash scripts/dispatch_faillen.sh fsa                             # failing lengths (tab:results, tab:seeds)
bash scripts/dispatch_drift.sh                                   # appendix: per-step h_t, par vs seq scan
bash scripts/dispatch_walltime.sh                                # appendix: wall-clock, one array task per config
bash scripts/dispatch_ssms_fp.sh                                 # appendix: the finite-precision rotation SSM
bash scripts/clean.sh                                            # remove results/, figures/, logs/, caches
```

All outputs go under `results/`.

## Paper tables and figures

The LaTeX sources live in the paper tree, not here. `make_figure2.py` and `make_figure3.py` fill
the two failing-length tables in place; point `NFSM_ASSETS_DIR` at the directory holding them.

```bash
NFSM_ASSETS_DIR=../paper/assets python scripts/make_figure2.py   # failing length per task and model
NFSM_ASSETS_DIR=../paper/assets python scripts/make_figure3.py   # per-seed failing length
python scripts/paper_summary.py     # results/paper_summary.md: FSA extraction, TSO heads, sanity checks
python -m experiments.tso.trajectory --token question-4,exact,nfsm,2,free,42   # TSO head trajectory
```
