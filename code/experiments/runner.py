"""Command-line runner shared by the FSA and TSO suites.

    python -m experiments.<suite>.run --list-jobs [--tasks ...] [--cells ...] [--arms ...] [--seeds ...]
    python -m experiments.<suite>.run --job TOKEN [--steps N] [--no-wandb]
    python -m experiments.<suite>.run --aggregate

A job trains one configuration (nfsm.training.run_training), saves the restored best model to
results/<suite>/checkpoints/<name>.msgpack, measures its accuracy at every length of the suite's
sweep and writes results/<suite>/parts/<name>.json, where <name> is the token with commas replaced
by underscores. --aggregate writes results/<suite>/summary.md and seq_acc.pdf.

A suite module (experiments/<suite>/tasks.py) provides SUITE, WANDB_PROJECT, TRAIN_LEN, BATCH,
SWEEP_LENS, STEPS, iter_jobs(**filters) -> tokens, parse_token(token) -> job dict,
data_fn(job) -> fn(key, B, T) and anchor_codes(job).
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS_ROOT = os.environ.get("NFSM_RESULTS_DIR", os.path.join(ROOT, "results"))


def out_dir(suite, kind: str) -> str:
    d = os.path.join(RESULTS_ROOT, suite.SUITE, kind)
    os.makedirs(d, exist_ok=True)
    return d


def job_name(token: str) -> str:
    return token.replace(",", "_")


def build_model(job: dict):
    """An untrained Backbone of the shape a job dict describes (seeded by job['seed'])."""
    import jax
    from flax import nnx
    from nfsm.backbone import Backbone, ModelConfig
    cfg = ModelConfig(vocab=job["vocab"], n_classes=job["n_classes"],
                      head_sizes=tuple(job["head_sizes"]), cell=job["cell"],
                      n_layers=job["layers"], conv_kernel=job["conv_kernel"],
                      n_dict=16,                          # PD-SSM dictionary size K (state N = width)
                      layer_head_sizes=(tuple(tuple(s) for s in job["layer_head_sizes"])
                                        if job.get("layer_head_sizes") else None))
    return Backbone(cfg, rngs=nnx.Rngs(jax.random.PRNGKey(job["seed"])))


def train_config(suite, job: dict, steps: int = None, use_wandb: bool = True):
    from nfsm.training import TrainConfig
    return TrainConfig(steps=int(steps or suite.STEPS), batch_size=suite.BATCH,
                       seq_len=suite.TRAIN_LEN, val_batch=job["val_batch"],
                       early_stop_acc=1.0 if job["cell"] == "nfsm" else 0.99,
                       anchor_injective=job.get("anchor_injective", True),
                       use_wandb=use_wandb, wandb_project=suite.WANDB_PROJECT)


def token_budget(job: dict) -> int:
    """Tokens per evaluation chunk: bounded by the cell's per-token state and the read-out width."""
    budget = {"aussm": 8 * 1024, "mamba": 16 * 1024}.get(job["cell"], 64 * 1024) // job["layers"]
    return max(1, min(budget, (64 << 20) // job["n_classes"]))


def save_checkpoint(model, path: str) -> str:
    import flax.serialization as fser
    from flax import nnx
    with open(path, "wb") as f:
        f.write(fser.to_bytes(nnx.to_pure_dict(nnx.state(model, nnx.Param))))
    return path


def load_checkpoint(suite, token: str):
    """The model a job token names, with its saved weights."""
    import flax.serialization as fser
    from flax import nnx
    model = build_model(suite.parse_token(token))
    state = nnx.state(model, nnx.Param)
    with open(os.path.join(out_dir(suite, "checkpoints"), job_name(token) + ".msgpack"), "rb") as f:
        pure = fser.from_bytes(nnx.to_pure_dict(state), f.read())
    nnx.replace_by_pure_dict(state, pure)
    nnx.update(model, state)
    return model


def run_job(suite, token: str, steps: int = None, use_wandb: bool = True, extract=None) -> dict:
    """Train, checkpoint, sweep (and `extract(model, job)` for NFSM if given); write the part JSON."""
    from dataclasses import asdict
    from nfsm.eval import sweep_accuracy
    from nfsm.training import run_training

    job = suite.parse_token(token)
    name = job_name(token)
    print(f"[job] {token}  heads={job.get('layer_head_sizes') or job['head_sizes']}  "
          f"classes={job['n_classes']}", flush=True)
    model = build_model(job)
    cfg = train_config(suite, job, steps, use_wandb)
    data_fn = suite.data_fn(job)
    t0 = time.time()
    history = run_training(model, cfg, data_fn, job["seed"],
                           anchor_codes=suite.anchor_codes(job) if job["arm"] == "anchored" else None,
                           run_name=name, group=name.rsplit("_", 1)[0], config=job)
    seconds = time.time() - t0
    ckpt = save_checkpoint(model, os.path.join(out_dir(suite, "checkpoints"), name + ".msgpack"))
    sweep = sweep_accuracy(model, data_fn, job["seed"], suite.SWEEP_LENS, token_budget(job))
    rec = {**job, "params": model.param_count(), "recipe": asdict(cfg),
           "train_seconds": round(seconds, 1), "ckpt": os.path.relpath(ckpt, ROOT),
           "lens": list(suite.SWEEP_LENS), **sweep, "history": history}
    if extract is not None and job["cell"] == "nfsm":
        rec["extraction"] = extract(model, job)
    if use_wandb:
        import wandb
        for L, s in zip(suite.SWEEP_LENS, sweep["seq_acc"]):
            wandb.summary[f"sweep/seq_acc@{L}"] = s
        wandb.finish()
    with open(os.path.join(out_dir(suite, "parts"), name + ".json"), "w") as f:
        json.dump(rec, f)
    print(f"[done] {token}  seq_acc@{suite.TRAIN_LEN}="
          f"{sweep['seq_acc'][list(suite.SWEEP_LENS).index(suite.TRAIN_LEN)]:.4f}  "
          f"seq_acc@{suite.SWEEP_LENS[-1]}={sweep['seq_acc'][-1]:.4f}  "
          f"answer_acc@{suite.SWEEP_LENS[-1]}={sweep['answer_acc'][-1]:.4f}  ({seconds:.0f}s)"
          + (f"  extraction={rec['extraction']}" if "extraction" in rec else ""), flush=True)
    return rec


def solved_acc(rec: dict, i: int) -> float:
    """Accuracy at sweep index i: sequence accuracy, or its min with the answer accuracy if any."""
    a = rec["answer_acc"][i]
    return rec["seq_acc"][i] if a != a else min(rec["seq_acc"][i], a)


def failing_length(rec: dict):
    """Smallest swept length whose `solved_acc` is below 1.0 (NFSM) or 0.9, None if none."""
    bar = 1.0 if rec["cell"] == "nfsm" else 0.9
    return next((L for i, L in enumerate(rec["lens"]) if solved_acc(rec, i) < bar), None)


def aggregate(suite) -> str:
    """Summarise the part JSONs by configuration (all fields of the token but the seed)."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(out_dir(suite, "parts"), "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        groups.setdefault(rec["token"].rsplit(",", 1)[0], []).append(rec)
    lines = [f"# {suite.SUITE}: solved = sequence (and answer) accuracy 1.0 at T = "
             f"{suite.SWEEP_LENS[-1]}", "",
             "| config | solved | median failing length | exact machine |",
             "|---|---:|---:|---:|"]
    for key, recs in groups.items():
        solved = sum(solved_acc(r, -1) == 1.0 for r in recs)
        fails = [failing_length(r) or 2 * suite.SWEEP_LENS[-1] for r in recs]
        med = statistics.median(fails)
        med = f">{suite.SWEEP_LENS[-1]}" if med > suite.SWEEP_LENS[-1] else f"{med:g}"
        ext = [r["extraction"]["exact"] for r in recs if "extraction" in r]
        lines.append(f"| {key} | {solved}/{len(recs)} | {med} | "
                     f"{f'{sum(ext)}/{len(ext)}' if ext else '-'} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS_ROOT, suite.SUITE, "summary.md"), "w") as f:
        f.write(text)
    _plot(suite, groups)
    print(text, flush=True)
    return text


def _plot(suite, groups) -> None:
    """Mean sequence accuracy against length, one panel per task, one line per configuration."""
    from nfsm.plotting import plt_setup, save_fig
    plt = plt_setup()
    tasks = sorted({key.split(",")[0] for key in groups})
    if not tasks:
        return
    cols = min(4, len(tasks))
    rows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for ax, task in zip(axes.flat, tasks):
        for key, recs in groups.items():
            if key.split(",")[0] != task:
                continue
            mean = [sum(r["seq_acc"][i] for r in recs) / len(recs) for i in range(len(recs[0]["lens"]))]
            ax.plot(recs[0]["lens"], mean, marker="o", ms=3, label=",".join(key.split(",")[1:]))
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(task)
        ax.set_xlabel("length")
        ax.legend(fontsize=6)
    for ax in list(axes.flat)[len(tasks):]:
        ax.axis("off")
    axes[0][0].set_ylabel("sequence accuracy")
    save_fig(fig, plt, os.path.join(RESULTS_ROOT, suite.SUITE, "seq_acc.pdf"))


def main(suite, extract=None) -> None:
    """Parse the command line and run it for `suite`."""
    p = argparse.ArgumentParser(description=f"{suite.SUITE} suite")
    p.add_argument("--list-jobs", action="store_true", help="print the job tokens")
    p.add_argument("--job", metavar="TOKEN", help="train one configuration")
    p.add_argument("--aggregate", action="store_true", help="summarise the part JSONs")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--cells", nargs="+")
    p.add_argument("--arms", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--steps", type=int, help="step budget (default: the suite's)")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    from nfsm import device
    device.configure()
    if args.job is None:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if args.list_jobs:
        print("\n".join(suite.iter_jobs(tasks=args.tasks, cells=args.cells, arms=args.arms,
                                        seeds=args.seeds)))
    elif args.job:
        device.print_devices(suite.SUITE)
        run_job(suite, args.job, steps=args.steps, use_wandb=not args.no_wandb, extract=extract)
    elif args.aggregate:
        aggregate(suite)
    else:
        p.error("choose --list-jobs, --job TOKEN or --aggregate")
