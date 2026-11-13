"""KWS entry point (stage 1 of KWS Stream).

    python -m experiments.kws.run --list-jobs [--cells ...] [--layers ...] [--readouts ...] [--seeds ...]
    python -m experiments.kws.run --job TOKEN [--steps N] [--lr LR] [--batch B] [--nu NU] [--wd WD] [--no-wandb]
    python -m experiments.kws.run --aggregate

A job trains, keeps the best validation checkpoint, scores the test split and writes
results/kws/checkpoints/<name>.msgpack and results/kws/parts/<name>.json. Data:
$KWS_DATA_DIR/digits.npz (default data/kws/).
"""

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import runner                      # noqa: E402
from experiments.kws import tasks                   # noqa: E402


def load_data(path: str = None) -> dict:
    import numpy as np
    with np.load(path or os.path.join(tasks.DATA_DIR, "digits.npz")) as f:
        return {k: f[k] for k in f.files}


def make_predict_fn(model, tokens: int = None):
    """predict(model, x [N, T, F]) -> classes [N], deterministic, in batches."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    graphdef, _ = nnx.split(model)
    tokens = tokens or tasks.EVAL_BATCH * tasks.N_FRAMES

    @jax.jit
    def _pred(state, x):
        return jnp.argmax(nnx.merge(graphdef, state)(x), -1)

    def predict(model_ref, x):
        _, state = nnx.split(model_ref)
        b = max(1, tokens // x.shape[1])
        return np.concatenate([np.asarray(_pred(state, jnp.asarray(x[i:i + b])))
                               for i in range(0, len(x), b)])

    return predict


def make_eval_fn(model):
    """eval(model, x, y) -> (accuracy, confusion [C, C]) over the whole split, deterministic."""
    import numpy as np
    predict = make_predict_fn(model)

    def evaluate(model_ref, x, y):
        pred = predict(model_ref, x)
        C = len(tasks.CLASSES)
        conf = np.zeros((C, C), np.int64)
        np.add.at(conf, (np.asarray(y), pred), 1)
        return float(np.mean(pred == np.asarray(y))), conf

    return evaluate


def make_loss(cfg):
    """CE of the clip's class + (NFSM) free_explore * c(t) * mean exploration excess."""
    import jax.numpy as jnp
    import optax

    def loss_fn(model, batch):
        x, y, scale = batch
        logits, excess = model(x, training=True, return_explore=True)
        ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        exc = jnp.mean(excess)
        return ce + cfg.free_explore * scale * exc, (ce, exc)

    return loss_fn


def array_sampler(x, y, batch: int, seed: int):
    """sample(step) -> (x, y) batch, drawn without replacement through shuffled epochs."""
    import numpy as np
    rng = np.random.default_rng(seed)
    state = {"order": rng.permutation(len(y)), "pos": 0}

    def sample(step):
        if state["pos"] + batch > len(state["order"]):
            state["order"], state["pos"] = rng.permutation(len(y)), 0
        idx = state["order"][state["pos"]:state["pos"] + batch]
        state["pos"] += batch
        return x[idx], y[idx]

    return sample


def train(model, cfg, sample, validate, seed: int, run_name: str, job: dict,
          select_from: int = 0) -> dict:
    """Train in place and restore the best validation checkpoint (checks from select_from on)."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from nfsm.training.optim import make_optimizer
    from nfsm.training.trainer import explore_scale, make_train_step

    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=run_name, group=run_name.rsplit("_", 1)[0],
                   config={**asdict(cfg), **job})
    tx = make_optimizer(cfg.lr, cfg.steps, cfg.weight_decay, cfg.grad_clip, eps=cfg.adam_eps,
                        b1=cfg.adam_b1, b2=cfg.adam_b2)
    step_fn = make_train_step(model, tx, make_loss(cfg))
    opt_state = tx.init(nnx.state(model, nnx.Param))
    history = {k: [] for k in ("step", "loss", "ce", "excess", "val_acc")}
    best = {"acc": -1.0, "step": None, "params": None}
    step = -1
    for step in range(cfg.steps):
        xb, yb = sample(step)
        opt_state, loss, (ce, exc) = step_fn(
            model, opt_state, (jnp.asarray(xb), jnp.asarray(yb),
                               jnp.asarray(explore_scale(cfg, step), jnp.float32)))
        if step % cfg.val_every:
            continue
        loss, ce, exc = float(loss), float(ce), float(exc)
        if not np.isfinite(loss):
            print(f"[{step}] non-finite loss; stopping", flush=True)
            break
        v_acc = validate(model)
        for k, v in (("step", step), ("loss", loss), ("ce", ce), ("excess", exc), ("val_acc", v_acc)):
            history[k].append(v)
        if step >= select_from and v_acc >= best["acc"]:
            best = {"acc": v_acc, "step": step,
                    "params": jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))}
        if cfg.use_wandb:
            wandb.log({"train/loss": loss, "train/ce": ce, "train/excess": exc, "val/acc": v_acc},
                      step=step)
        if step % cfg.print_every == 0:
            print(f"[{step}] loss {loss:.4f} ce {ce:.4f} exc {exc:.3f}  val acc {v_acc:.4f}  "
                  f"(best {best['acc']:.4f} at {best['step']})", flush=True)
    if best["params"] is not None:
        nnx.update(model, best["params"])
    history.update(steps_run=step + 1, best_step=best["step"], best_val_acc=best["acc"])
    return history


def apply_hparams(job: dict, cfg, hp: dict = None) -> None:
    """Override lr, batch, nu (NFSM Gumbel temperature), wd and pdssm_init; None keeps the default.
    Recorded in job["hparams"].
    """
    hp = {k: v for k, v in (hp or {}).items() if v is not None}
    if not hp:
        return
    if "lr" in hp:
        cfg.lr = float(hp["lr"])
    if "wd" in hp:
        cfg.weight_decay = float(hp["wd"])
    if "batch" in hp:
        cfg.batch_size = int(hp["batch"])
    if "nu" in hp:
        job["noise_scale"] = float(hp["nu"])
    if "pdssm_init" in hp:
        job["pdssm_mag_init"] = hp["pdssm_init"]
    job["hparams"] = hp


def run_job(token: str, steps: int = None, use_wandb: bool = True, hp: dict = None) -> dict:
    from nfsm.training import TrainConfig
    from experiments.kws.model import build_model

    job = tasks.parse_token(token)
    name = runner.job_name(token)
    data = load_data()
    print(f"[job] {token}  heads={job['head_sizes']}  train/val/test = {len(data['y_train'])}/"
          f"{len(data['y_val'])}/{len(data['y_test'])}", flush=True)
    cfg = TrainConfig(steps=int(steps or tasks.STEPS), batch_size=tasks.BATCH,
                      seq_len=tasks.N_FRAMES, val_every=tasks.VAL_EVERY, print_every=tasks.VAL_EVERY * 4,
                      use_wandb=use_wandb, wandb_project=tasks.WANDB_PROJECT)
    apply_hparams(job, cfg, hp)
    model = build_model(job)
    evaluate = make_eval_fn(model)
    sample = array_sampler(data["x_train"], data["y_train"], cfg.batch_size, job["seed"])
    t0 = time.time()
    history = train(model, cfg, sample, lambda mm: evaluate(mm, data["x_val"], data["y_val"])[0],
                    job["seed"], name, job)
    seconds = time.time() - t0
    ckpt = runner.save_checkpoint(model, os.path.join(runner.out_dir(tasks, "checkpoints"),
                                                      name + ".msgpack"))
    test_acc, conf = evaluate(model, data["x_test"], data["y_test"])
    val_acc, _ = evaluate(model, data["x_val"], data["y_val"])
    rec = {**job, "params": model.param_count(), "recipe": asdict(cfg),
           "train_seconds": round(seconds, 1), "ckpt": os.path.relpath(ckpt, ROOT),
           "val_acc": val_acc, "test_acc": test_acc, "test_confusion": conf.tolist(),
           "classes": list(tasks.CLASSES), "history": history}
    if use_wandb:
        import wandb
        wandb.summary["test/acc"] = test_acc
        wandb.finish()
    with open(os.path.join(runner.out_dir(tasks, "parts"), name + ".json"), "w") as f:
        json.dump(rec, f)
    print(f"[done] {token}  val {val_acc:.4f}  test {test_acc:.4f}  best step {history['best_step']}"
          f"  params {rec['params']}  ({seconds:.0f}s)", flush=True)
    return rec


def aggregate() -> str:
    """Test accuracy per configuration (token without the seed): mean [min; max] over seeds."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(runner.out_dir(tasks, "parts"), "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        groups.setdefault(rec["token"].rsplit(",", 1)[0], []).append(rec)
    lines = ["# KWS Digits (11 classes): test accuracy, mean [min; max] over seeds", "",
             "| cell,layers,readout | seeds | params | test acc (%) | val acc (%) | train time (min) |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, recs in sorted(groups.items()):
        t = [100 * r["test_acc"] for r in recs]
        v = [100 * r["val_acc"] for r in recs]
        mins = sum(r["train_seconds"] for r in recs) / len(recs) / 60
        lines.append(f"| {key} | {len(recs)} | {recs[0]['params']:,} | {sum(t) / len(t):.2f} "
                     f"[{min(t):.2f}; {max(t):.2f}] | {sum(v) / len(v):.2f} | {mins:.0f} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(runner.RESULTS_ROOT, tasks.SUITE, "summary.md"), "w") as f:
        f.write(text)
    print(text, flush=True)
    return text


def add_hparam_args(p) -> None:
    """--lr, --batch, --nu, --wd: optional overrides of the training recipe (default: unchanged)."""
    p.add_argument("--lr", type=float, help="peak learning rate (default: TrainConfig's 1e-3)")
    p.add_argument("--batch", type=int, help="batch size (default: the suite's BATCH)")
    p.add_argument("--nu", type=float, help="NFSM Gumbel temperature (default: ModelConfig's 0.5)")
    p.add_argument("--wd", type=float, help="weight decay (default: TrainConfig's 1e-4)")
    p.add_argument("--pdssm-init", choices=("paper", "mamba"),
                   help="PD-SSM initial |D| (default: paper, uniform in [0.9, 0.999])")


def hparams(args) -> dict:
    return {"lr": args.lr, "batch": args.batch, "nu": args.nu, "wd": args.wd, "pdssm_init": args.pdssm_init}


def main() -> None:
    p = argparse.ArgumentParser(description="KWS Digits suite")
    p.add_argument("--list-jobs", action="store_true", help="print the job tokens")
    p.add_argument("--job", metavar="TOKEN", help="train one configuration")
    p.add_argument("--aggregate", action="store_true", help="summarise the part JSONs")
    p.add_argument("--cells", nargs="+")
    p.add_argument("--layers", nargs="+", type=int)
    p.add_argument("--readouts", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--steps", type=int, help=f"step budget (default {tasks.STEPS})")
    add_hparam_args(p)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    from nfsm import device
    device.configure()
    if args.job is None:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if args.list_jobs:
        print("\n".join(tasks.iter_jobs(cells=args.cells, layers=args.layers,
                                        readouts=args.readouts, seeds=args.seeds)))
    elif args.job:
        device.print_devices(tasks.SUITE)
        run_job(args.job, steps=args.steps, use_wandb=not args.no_wandb, hp=hparams(args))
    elif args.aggregate:
        aggregate()
    else:
        p.error("choose --list-jobs, --job TOKEN or --aggregate")


if __name__ == "__main__":
    main()
