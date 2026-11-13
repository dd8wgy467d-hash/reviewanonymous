"""Read a trained TSO NFSM as a finite machine and check it against the task.

With a causal conv of width k, a layer's step depends on a window of the last k contexts (symbol and
lower-layer states). We read each layer's tables on the windows the data reaches, bottom up, and
check tau_w o c = c o delta_w, c mapping task states to joint head states.

  python -m experiments.tso.extract --job "question-3,exact,nfsm,2,free,42"
  python -m experiments.tso.extract --all [--arms free]

Writes results/tso/extraction/<token>.json.
"""

import argparse
import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from experiments import runner
from experiments.tso import tasks

BATCH, LENGTH, BATCHES = 64, 512, 12        # sequences per pass, their length, passes


def _run(model, x):
    """Per-layer noise-free head states [L][B, T, K] for token ids [B, T]."""
    h = model.embed_tokens(x)
    states = []
    for layer in model.layers:
        states.append(np.asarray(layer.cell.states(layer.cell_input(h))))
        h, _, _ = layer(h)
    return states


def _stream(model, upto, tokens, below):
    """Stream entering layer `upto` [N, m] for symbols [N] and lower states `below` [N, l, K]."""
    h = model.embed_tokens(jnp.asarray(tokens))
    for j in range(upto):
        layer = model.layers[j]
        h = h + layer.cell.readout(jnp.asarray(below[:, j]))
        h = h + layer.mlp(layer.norm_mlp(h))
    return h


def extract_machine(model, job: dict, seed: int = 0, chunk: int = 4096,
                    max_checked: int = 20000) -> dict:
    """Check that the model is a finite machine on its joint head states and that it computes the task:
    the read-out names one task state per joint state, the window decides each step, the steps project
    onto the task automaton, and the tables read from the logits reproduce them. Checked on the
    (state, window) pairs the task produces, sampled until their count saturates.
    """
    data_fn = tasks.data_fn(job)
    L = len(list(model.layers))
    cls_of, trans, q_delta, allowed = {}, {}, {}, {}
    code_conflicts = readout_errors = nondeterministic = q_violations = 0
    answer_ok = answer_n = 0
    new_per_batch = []
    for b in range(BATCHES):
        batch = data_fn(jax.random.PRNGKey(seed * 7919 + b), BATCH, LENGTH)
        x, y = batch[0], batch[1]
        ans = batch[2] if len(batch) > 2 else None
        q = np.asarray(jnp.argmax(y, -1))
        st = np.stack(_run(model, x), axis=2)                   # [B, T, L, K]
        logits = model(x)
        pred = np.asarray(jnp.argmax(logits[..., :y.shape[-1]], -1))
        if ans is not None:
            answer_ok += int(np.sum(np.asarray(jnp.argmax(logits[:, -1, y.shape[-1]:], -1))
                                    == np.asarray(ans)))
            answer_n += int(np.asarray(ans).shape[0])
        xs = np.asarray(x)
        before = len(trans)
        for i in range(xs.shape[0]):
            joints = [tuple(st[i, t].reshape(-1)) for t in range(xs.shape[1])]
            for t in range(xs.shape[1]):
                j, qt = joints[t], int(q[i, t])
                if cls_of.setdefault(j, qt) != qt:
                    code_conflicts += 1                 # one joint state, two task states
                if pred[i, t] != qt:
                    readout_errors += 1                 # the head misnames the state
                if t:
                    w = (int(xs[i, t - 1]), int(xs[i, t]))
                    key = (joints[t - 1], w)
                    if trans.setdefault(key, j) != j:
                        nondeterministic += 1
                    qkey = (int(q[i, t - 1]), w)
                    if q_delta.setdefault(qkey, qt) != qt:
                        q_violations += 1
                    allowed.setdefault(int(q[i, t - 1]), set()).add(w)
        new_per_batch.append(len(trans) - before)

    # the tables read from the logits must reproduce every step seen
    keys = sorted(trans)[:max_checked]
    table_errors = 0
    if keys:
        K = len(job["head_sizes"])
        z_prev = np.array([k[0] for k in keys], np.int32).reshape(-1, L, K)
        z_cur = np.array([trans[k] for k in keys], np.int32).reshape(-1, L, K)
        x_prev = np.array([k[1][0] for k in keys], np.int32)
        x_cur = np.array([k[1][1] for k in keys], np.int32)
        for l in range(L):
            layer = list(model.layers)[l]
            for i in range(0, len(keys), chunk):
                sl = slice(i, i + chunk)
                h = jnp.stack([_stream(model, l, x_prev[sl], z_prev[sl]),
                               _stream(model, l, x_cur[sl], z_cur[sl])], axis=1)
                maps = np.asarray(layer.cell.transition_maps(layer.cell_input(h)))[:, 1]
                got = np.take_along_axis(maps, z_prev[sl, l][:, :, None], axis=2)[:, :, 0]
                table_errors += int(np.sum(got != z_cur[sl, l]))

    exact = (code_conflicts == 0 and readout_errors == 0 and nondeterministic == 0
             and q_violations == 0 and table_errors == 0)
    return {"exact": bool(exact), "task_states": len(set(cls_of.values())),
            "joint_states": len(cls_of), "windows": len({k[1] for k in trans}),
            "transitions": len(trans), "code_conflicts": int(code_conflicts), "readout_errors": int(readout_errors),
            "nondeterministic": int(nondeterministic), "homomorphism_violations": int(q_violations),
            "table_errors": int(table_errors), "transitions_checked": len(keys),
            "answer_acc": (answer_ok / answer_n) if answer_n else None,
            "new_transitions_per_batch": new_per_batch, "layers": L}


def run_job(token: str) -> dict:
    job = tasks.parse_token(token)
    model = runner.load_checkpoint(tasks, token)
    rec = {"token": token, **extract_machine(model, job, seed=job["seed"])}
    path = os.path.join(runner.out_dir(tasks, "extraction"), runner.job_name(token) + ".json")
    with open(path, "w") as f:
        json.dump(rec, f)
    print(f"[{token}] exact={rec['exact']} task_states={rec['task_states']} "
          f"joint={rec['joint_states']} windows={rec['windows']} transitions={rec['transitions']} "
          f"code={rec['code_conflicts']} readout={rec['readout_errors']} "
          f"nondet={rec['nondeterministic']} "
          f"homo={rec['homomorphism_violations']} tables={rec['table_errors']}"
          f"/{rec['transitions_checked']} answer={rec['answer_acc']} "
          f"new/batch={rec['new_transitions_per_batch']}", flush=True)
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="extract and check a TSO NFSM's tables")
    p.add_argument("--job", metavar="TOKEN")
    p.add_argument("--all", action="store_true", help="every NFSM checkpoint on disk")
    p.add_argument("--arms", nargs="+", default=["free"])
    args = p.parse_args()
    if args.job:
        run_job(args.job)
    elif args.all:
        for path in sorted(glob.glob(os.path.join(runner.out_dir(tasks, "parts"), "*_nfsm_*.json"))):
            with open(path) as f:
                token = json.load(f)["token"]
            if token.split(",")[4] in args.arms:
                run_job(token)
    else:
        p.error("one of --job, --all is required")
