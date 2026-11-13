"""KWS Stream entry point.

    python -m experiments.kws_stream.run --list-jobs [--cells ...] [--layers ...] [--readouts ...] [--seeds ...]
    python -m experiments.kws_stream.run --job TOKEN [--steps N] [--lr LR] [--batch B] [--nu NU] [--wd WD]
                                         [--test-lens T ...] [--ood-filter] [--no-wandb]
    python -m experiments.kws_stream.run --sweep TOKEN
    python -m experiments.kws_stream.run --aggregate

A job starts from the KWS checkpoint of the same token, trains with the length curriculum, keeps the
best validation checkpoint, scores the test streams at TEST_LENS and writes
results/kws_stream/parts/<name>.json. --sweep adds missing lengths; --prefix-sweep and --suffix-sweep
put the digit at the end or at the start of the stream (keys "prefix", "suffix"); with --ood-filter
they use the filtered test noise and write parts_ood/. Figures: scripts/analyse_kws_stream.py.
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
from experiments.kws_stream import tasks            # noqa: E402

HOLD_BINS = (0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800,
             409600, 819200)                                      # frames after the word


def prefetch(items, depth: int = 2):
    """Yield the items of an iterator in order, computing the next `depth` in a thread."""
    import queue
    import threading
    q, end = queue.Queue(maxsize=depth), object()

    def work():
        try:
            for item in items:
                q.put(item)
            q.put(end)
        except BaseException as e:                     # re-raised in the consumer
            q.put(e)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is end:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def score(predict, model, streams, split: str, T: int, seed: int, place: str = "random") -> dict:
    """Accuracy on the fixed streams of length T, and the hold of each digit stream."""
    import numpy as np
    chunk = max(1, tasks.EVAL_TOKENS // T, min(64, 2 ** 22 // T))   # streams generated at once
    ys, preds, holds = [], [], []
    for x, y, h in prefetch(streams.fixed_set(split, T, seed, chunk=chunk, place=place)):
        preds.append(predict(model, x)), ys.append(y), holds.append(h)
    y, pred, hold = np.concatenate(ys), np.concatenate(preds), np.concatenate(holds)
    ok = pred == y
    digit = y != tasks.NONE
    return {"acc": float(ok.mean()), "acc_digit": float(ok[digit].mean()),
            "acc_none": float(ok[~digit].mean()) if (~digit).any() else float("nan"),
            "hold": hold[digit].tolist(), "correct": ok[digit].astype(int).tolist()}


def init_path(token: str) -> str:
    return os.path.join(runner.RESULTS_ROOT, tasks.INIT_SUITE, "checkpoints",
                        runner.job_name(token) + ".msgpack")


def load_init(model, token: str, path: str = None) -> str:
    """Load the weights at `path` (default: the KWS checkpoint of `token`) into `model`."""
    import flax.serialization as fser
    from flax import nnx
    path = path or init_path(token)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint for {token}: {path}")
    state = nnx.state(model, nnx.Param)
    with open(path, "rb") as f:
        pure = fser.from_bytes(nnx.to_pure_dict(state), f.read())
    nnx.replace_by_pure_dict(state, pure)
    nnx.update(model, state)
    return path


def run_job(token: str, steps: int = None, use_wandb: bool = True, hp: dict = None,
            test_lens=None, ood: bool = False) -> dict:
    """Train one job; hp overrides the recipe, test_lens the swept lengths, ood filters the test noise."""
    import numpy as np
    from nfsm.training import TrainConfig
    from experiments.kws.model import build_model
    from experiments.kws.run import apply_hparams, make_predict_fn, train
    from experiments.kws_stream.data import Streams

    job = tasks.parse_token(token)
    name = runner.job_name(token)
    test_lens = tuple(test_lens or tasks.TEST_LENS)
    streams = Streams(ood_filter=ood)
    h = job["head_sizes"]
    print(f"[job] {token}  heads={len(h)} x {h[0]}  words train/val/test = "
          f"{len(streams.labels['train'])}/{len(streams.labels['val'])}/{len(streams.labels['test'])}",
          flush=True)
    cfg = TrainConfig(steps=int(steps or tasks.STEPS), batch_size=tasks.BATCH,
                      seq_len=max(tasks.TRAIN_LENS), val_every=tasks.VAL_EVERY,
                      print_every=tasks.VAL_EVERY * 4, use_wandb=use_wandb,
                      wandb_project=tasks.WANDB_PROJECT)
    apply_hparams(job, cfg, hp)
    model = build_model(job)
    init = load_init(model, token)
    print(f"[init] {os.path.relpath(init, ROOT)}", flush=True)
    predict = make_predict_fn(model, tasks.EVAL_TOKENS)
    rng = np.random.default_rng(job["seed"])

    def sample(step):
        T = int(rng.choice(tasks.unlocked(step)))
        x, y, _ = streams.batch(rng, "train", T, cfg.batch_size)
        return x, y

    val_sets = {T: [(x, y) for x, y, _ in streams.fixed_set("val", T, tasks.TEST_SEED, 1024)]
                for T in tasks.VAL_LENS}

    def validate(m):
        accs = []
        for T, chunks in val_sets.items():
            ok = np.concatenate([predict(m, x) == y for x, y in chunks])
            accs.append(ok.mean())
        return float(np.mean(accs))

    t0 = time.time()
    history = train(model, cfg, sample, validate, job["seed"], name, job,
                    select_from=tasks.full_range_step())
    seconds = time.time() - t0
    ckpt = runner.save_checkpoint(model, os.path.join(runner.out_dir(tasks, "checkpoints"),
                                                      name + ".msgpack"))
    sweep = sweep_lengths(predict, model, streams, test_lens)
    rec = {**job, "params": model.param_count(), "recipe": asdict(cfg),
           "init": os.path.relpath(init, ROOT),
           "curriculum": {"start": tasks.CURRICULUM_START, "every": tasks.CURRICULUM_EVERY,
                          "full_range_step": tasks.full_range_step()},
           "train_seconds": round(seconds, 1), "ckpt": os.path.relpath(ckpt, ROOT),
           "classes": list(tasks.CLASSES), "lens": list(test_lens),
           "acc": [sweep[T]["acc"] for T in test_lens],
           "acc_digit": [sweep[T]["acc_digit"] for T in test_lens],
           "acc_none": [sweep[T]["acc_none"] for T in test_lens],
           "hold": {str(T): sweep[T]["hold"] for T in test_lens},
           "correct": {str(T): sweep[T]["correct"] for T in test_lens},
           "history": history}
    if ood:
        rec["ood_filter"] = streams.ood
    if use_wandb:
        import wandb
        for T in test_lens:
            wandb.summary[f"test/acc@{T}"] = sweep[T]["acc"]
        wandb.finish()
    with open(os.path.join(runner.out_dir(tasks, "parts"), name + ".json"), "w") as f:
        json.dump(rec, f)
    print(f"[done] {token}  acc@{tasks.TEST_LENS[0]} {rec['acc'][0]:.4f}  "
          f"acc@{test_lens[-1]} {rec['acc'][-1]:.4f}  ({seconds:.0f}s training)", flush=True)
    return rec


def sweep_lengths(predict, model, streams, lens) -> dict:
    """{T: score} on the fixed test streams at each length of `lens`."""
    sweep = {}
    for T in lens:
        t1 = time.time()
        sweep[T] = score(predict, model, streams, "test", T, tasks.TEST_SEED)
        print(f"[test] T={T}  acc {sweep[T]['acc']:.4f}  digit {sweep[T]['acc_digit']:.4f}  "
              f"none {sweep[T]['acc_none']:.4f}  ({time.time() - t1:.0f}s)", flush=True)
    return sweep


def load_part(token: str) -> tuple:
    """(path, record) of a job part; a record without test lengths if only the checkpoint exists."""
    name = runner.job_name(token)
    part = os.path.join(runner.out_dir(tasks, "parts"), name + ".json")
    if os.path.exists(part):
        with open(part) as f:
            return part, json.load(f)
    ckpt = os.path.join(runner.out_dir(tasks, "checkpoints"), name + ".msgpack")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no part JSON and no checkpoint for {token!r}")
    print(f"[part] {token}: no part JSON, starting one from {os.path.relpath(ckpt, ROOT)}", flush=True)
    return part, {**tasks.parse_token(token), "ckpt": os.path.relpath(ckpt, ROOT),
                  "classes": list(tasks.CLASSES), "lens": [], "acc": [], "acc_digit": [],
                  "acc_none": [], "hold": {}, "correct": {}, "history": None}


def extend_sweep(token: str) -> dict:
    """Score a finished checkpoint at the missing lengths of TEST_LENS, saving after each one."""
    import numpy as np
    from experiments.kws.model import build_model
    from experiments.kws.run import make_predict_fn
    from experiments.kws_stream.data import Streams
    part, rec = load_part(token)
    missing = [T for T in tasks.TEST_LENS if T not in rec["lens"]]
    print(f"[sweep] {token}: adding {missing}", flush=True)
    if not missing:
        return rec
    model = build_model(tasks.parse_token(token))
    load_init(model, token, os.path.join(runner.out_dir(tasks, "checkpoints"),
                                         runner.job_name(token) + ".msgpack"))
    predict, streams = make_predict_fn(model, tasks.EVAL_TOKENS), Streams()
    for T in missing:
        s = sweep_lengths(predict, model, streams, [T])[T]
        rec["lens"].append(T)
        for k in ("acc", "acc_digit", "acc_none"):
            rec[k].append(s[k])
        rec["hold"][str(T)], rec["correct"][str(T)] = s["hold"], s["correct"]
        order = np.argsort(rec["lens"])
        for k in ("lens", "acc", "acc_digit", "acc_none"):
            rec[k] = [rec[k][i] for i in order]
        with open(part, "w") as f:
            json.dump(rec, f)
    print(f"[done] {token}  acc@{rec['lens'][-1]} {rec['acc'][-1]:.4f}", flush=True)
    return rec


PLACE_KEYS = {"end": "prefix", "start": "suffix"}   # part JSON key of each word placement


def place_sweep(token: str, place: str, ood: bool = False) -> dict:
    """Score a finished checkpoint with the digit at the end (key "prefix") or at the start (key
    "suffix") at the missing lengths. With ood, filtered test noise and parts_ood/.
    """
    import numpy as np
    from experiments.kws.model import build_model
    from experiments.kws.run import make_predict_fn
    from experiments.kws_stream.data import Streams
    name = runner.job_name(token)
    if ood:
        part = os.path.join(runner.out_dir(tasks, "parts_ood"), name + ".json")
        rec = {"token": token}
        if os.path.exists(part):
            with open(part) as f:
                rec = json.load(f)
    else:
        part, rec = load_part(token)
    key = PLACE_KEYS[place]
    pre = rec.setdefault(key, {"lens": [], "acc": []})
    missing = [T for T in tasks.TEST_LENS if T not in pre["lens"]]
    print(f"[{key}] {token}: adding {missing}", flush=True)
    if missing:
        model = build_model(tasks.parse_token(token))
        load_init(model, token, os.path.join(runner.out_dir(tasks, "checkpoints"), name + ".msgpack"))
        predict, streams = make_predict_fn(model, tasks.EVAL_TOKENS), Streams(ood_filter=ood)
        if ood:
            rec["ood_filter"] = streams.ood
            print(f"[{key}] out-of-distribution filter: {streams.ood}", flush=True)
        for T in missing:
            t1 = time.time()
            acc = score(predict, model, streams, "test", T, tasks.TEST_SEED, place=place)["acc_digit"]
            pre["lens"].append(T), pre["acc"].append(acc)
            print(f"[{key}] T={T}  noise {'before' if place == 'end' else 'after'} the word "
                  f"{T - tasks.WORD} frames  acc {acc:.4f}  "
                  f"({time.time() - t1:.0f}s)", flush=True)
            # rewrite after every length, so a time limit keeps what was scored
            order = np.argsort(pre["lens"])
            pre["lens"] = [pre["lens"][i] for i in order]
            pre["acc"] = [pre["acc"][i] for i in order]
            with open(part, "w") as f:
                json.dump(rec, f)
    print(f"[done] {token}  {key} acc@{pre['lens'][-1]} {pre['acc'][-1]:.4f}", flush=True)
    return rec


def hold_curve(rec: dict) -> list:
    """Accuracy of the digit streams per hold bin, pooled over all test lengths (nan if empty)."""
    import numpy as np
    if not rec["lens"]:
        return [float("nan")] * (len(HOLD_BINS) - 1)
    hold = np.concatenate([rec["hold"][str(T)] for T in rec["lens"]])
    ok = np.concatenate([rec["correct"][str(T)] for T in rec["lens"]])
    out = []
    for lo, hi in zip(HOLD_BINS[:-1], HOLD_BINS[1:]):
        sel = (hold >= lo) & (hold < hi)
        out.append(float(ok[sel].mean()) if sel.any() else float("nan"))
    return out


def aggregate() -> str:
    """Accuracy against length and against hold, mean [min; max] over seeds, per configuration."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(runner.out_dir(tasks, "parts"), "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        groups.setdefault(rec["token"].rsplit(",", 1)[0], []).append(rec)

    def cell(vals):
        vals = [100 * v for v in vals]
        return f"{sum(vals) / len(vals):.1f} [{min(vals):.1f}; {max(vals):.1f}]"

    lens = tasks.TEST_LENS
    lines = ["# KWS Stream: test accuracy (%) against stream length T (frames, 100 per second)", "",
             "| config | seeds | " + " | ".join(f"T={T}" for T in lens) + " |",
             "|---|---:|" + "---:|" * len(lens)]
    for key, recs in sorted(groups.items()):
        lines.append(f"| {key} | {len(recs)} | " + " | ".join(
            cell([r["acc"][r["lens"].index(T)] for r in recs])
            if all(T in r["lens"] for r in recs) else "-" for T in lens) + " |")
    bins = [f"{lo}-{hi}" for lo, hi in zip(HOLD_BINS[:-1], HOLD_BINS[1:])]
    lines += ["", "## Digit streams against hold (frames after the word, all lengths pooled)", "",
              "| config | " + " | ".join(bins) + " |", "|---|" + "---:|" * len(bins)]
    for key, recs in sorted(groups.items()):
        curves = [hold_curve(r) for r in recs]
        lines.append(f"| {key} | " + " | ".join(
            cell([c[i] for c in curves]) if all(c[i] == c[i] for c in curves) else "-"
            for i in range(len(bins))) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(runner.RESULTS_ROOT, tasks.SUITE, "summary.md"), "w") as f:
        f.write(text)
    print(text, flush=True)
    return text


def main() -> None:
    from experiments.kws.run import add_hparam_args, hparams
    p = argparse.ArgumentParser(description="KWS Stream suite")
    p.add_argument("--list-jobs", action="store_true", help="print the job tokens")
    p.add_argument("--job", metavar="TOKEN", help="train one configuration")
    p.add_argument("--sweep", metavar="TOKEN", help="add the missing test lengths to a finished job")
    p.add_argument("--prefix-sweep", metavar="TOKEN",
                   help="score a finished job on streams with the word at the end")
    p.add_argument("--suffix-sweep", metavar="TOKEN",
                   help="score a finished job on streams with the word at the start")
    p.add_argument("--ood-filter", action="store_true",
                   help="with --prefix-sweep/--suffix-sweep: no test noise crop over an out-of-"
                        "distribution frame (tasks.py) and write to parts_ood/; with --job: the "
                        "same filter in the test sweep")
    p.add_argument("--test-lens", nargs="+", type=int,
                   help="with --job: test only these lengths (default TEST_LENS)")
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
    if args.job is None and args.sweep is None and args.prefix_sweep is None \
            and args.suffix_sweep is None:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if args.list_jobs:
        print("\n".join(tasks.iter_jobs(cells=args.cells, layers=args.layers,
                                        readouts=args.readouts, seeds=args.seeds)))
    elif args.job:
        device.print_devices(tasks.SUITE)
        run_job(args.job, steps=args.steps, use_wandb=not args.no_wandb, hp=hparams(args),
                test_lens=args.test_lens, ood=args.ood_filter)
    elif args.sweep:
        device.print_devices(tasks.SUITE)
        extend_sweep(args.sweep)
    elif args.prefix_sweep:
        device.print_devices(tasks.SUITE)
        place_sweep(args.prefix_sweep, "end", ood=args.ood_filter)
    elif args.suffix_sweep:
        device.print_devices(tasks.SUITE)
        place_sweep(args.suffix_sweep, "start", ood=args.ood_filter)
    elif args.aggregate:
        aggregate()
    else:
        p.error("choose --list-jobs, --job TOKEN or --aggregate")


if __name__ == "__main__":
    main()
