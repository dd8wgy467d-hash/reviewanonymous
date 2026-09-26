"""Head states of a trained two-layer TSO NFSM along one short `question` stream, as one figure.

    python -m experiments.tso.trajectory --token question-4,exact,nfsm,2,free,42 [--length 50]

Each head's bare state index is given a meaning from --streams streams of --stream-len words drawn
from the real generator: the task register (parse, or the holder of one item) its state determines
best, and the majority value of that register in each state; the purity is the share of positions
that majority covers. Cells are coloured by that value, the ground-truth registers below them.
The drawn stream is the first one (keys seed + 101, seed + 102, ...) with at least --min-swaps swaps
that move the asked item and an answer other than the item's first holder. Everything runs noise
free, and the states drawn are checked against the full forward pass (experiments.tso.visualize).

Writes results/tso/figures/<name>/head_trajectory.pdf, head_trajectory.json (the analysis) and
head_trajectory_plot.json (all the figure needs: redraw it with plot_trajectory.py, matplotlib only).
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments import runner  # noqa: E402
from experiments.tso import tasks as TT  # noqa: E402
from experiments.tso.plot_trajectory import draw  # noqa: E402
from experiments.tso.visualize import Step, Words, check, run  # noqa: E402



def register_values(task: str, y: np.ndarray) -> list:
    """[R] arrays [B, T] of person-valued registers: parse (the sentence's first name, N when idle),
    then the holder of each item (N when unset)."""
    mod, n = TT._task(task)
    q = np.argmax(y[..., :mod.n_states(n)], -1)
    codes = mod.head_codes(n)
    parse = codes[0][q]
    return [np.where(parse == 0, n, parse - 1)] + [c[q] for c in codes[1:]]


def head_mapping(states: np.ndarray, regs: list, n: int, size: int) -> list:
    """Per head of states [B, T, K]: (register, purity, majority value per state, usage per state).
    A state never visited maps to n (drawn as idle / unset)."""
    out = []
    for k in range(states.shape[-1]):
        s = states[..., k].reshape(-1)
        best = None
        for r, reg in enumerate(regs):
            joint = np.zeros((size, n + 1), np.int64)
            np.add.at(joint, (s, reg.reshape(-1)), 1)
            purity = joint.max(1).sum() / s.size
            if best is None or purity > best[1]:
                best = (r, float(purity), np.where(joint.sum(1) > 0, joint.argmax(1), n), joint.sum(1))
        out.append(best)
    return out


def purity_matrix(states: np.ndarray, regs: list, n: int, size: int) -> np.ndarray:
    """[K, R] purity of every head (states [B, T, K]) for every register."""
    out = np.zeros((states.shape[-1], len(regs)))
    for k in range(states.shape[-1]):
        s = states[..., k].reshape(-1)
        for r, reg in enumerate(regs):
            joint = np.zeros((size, n + 1), np.int64)
            np.add.at(joint, (s, reg.reshape(-1)), 1)
            out[k, r] = joint.max(1).sum() / s.size
    return out


TOKEN_KINDS = ("preamble name", "preamble has", "preamble the", "preamble item", "preamble .",
               "swap 1st name", "swaps", "with", "swap 2nd name", "swap .",
               "who", "question has", "question the", "question item", "?")


def token_kinds(x: np.ndarray, words: Words) -> np.ndarray:
    """[B, T] kind of each token (TOKEN_KINDS), from its sentence and its place in it."""
    B, T = x.shape
    start = np.zeros_like(x)
    for t in range(1, T):
        start[:, t] = np.where(x[:, t - 1] == words.ids["."], t, start[:, t - 1])
    pos = np.arange(T)[None] - start
    lead = np.take_along_axis(x, start, 1)
    has = np.take_along_axis(x, np.minimum(start + 1, T - 1), 1) == words.ids["has"]
    kinds = np.full(x.shape, "", dtype=object)
    sent = np.where(lead == words.ids["who"], 2, np.where(has, 0, 1))      # preamble, swap, question
    names = [TOKEN_KINDS[0:5], TOKEN_KINDS[5:10], TOKEN_KINDS[10:15]]
    for si in range(3):
        for p in range(5):
            kinds[(sent == si) & (pos == p)] = names[si][p]
    kinds[(x == words.ids["."]) & (sent == 0)] = "preamble ."
    kinds[(x == words.ids["."]) & (sent == 1)] = "swap ."
    return kinds


def pct(p: float) -> str:
    """Purity as a percentage, truncated so that 100% means exactly all."""
    v = np.floor(p * 1000) / 10
    return f"{v:.0f}%" if v == int(v) else f"{v:.1f}%"


def swaps_moving(x: np.ndarray, y_regs: list, item: int, words: Words) -> int:
    """Swap sentences of one stream whose two names include the asked item's holder."""
    count = 0
    for t in np.nonzero(x == words.ids["swaps"])[0]:
        holder = y_regs[1 + item][t]
        if holder in (x[t - 1], x[t + 2]):
            count += 1
    return count


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="head states of a two-layer TSO NFSM along one stream")
    p.add_argument("--token", required=True)
    p.add_argument("--length", type=int, default=50, help="words in the drawn stream")
    p.add_argument("--streams", type=int, default=256, help="streams the head mapping is read from")
    p.add_argument("--stream-len", type=int, default=TT.TRAIN_LEN)
    p.add_argument("--min-swaps", type=int, default=2, help="swaps that move the asked item")
    p.add_argument("--track-threshold", type=float, default=None,
                   help="draw a head whose best purity is below this as untracked (raw state, grays)")
    p.add_argument("--order-by-register", action="store_true",
                   help="order the heads of each layer by register: parse, then items as the preamble "
                        "gives them; untracked heads last")
    p.add_argument("--no-truth", action="store_true", help="omit the ground-truth register rows")
    p.add_argument("--usetex", action="store_true", help="typeset every text with the local LaTeX")
    p.add_argument("--latex-preamble", default="",
                   help=r"with --usetex, e.g. '\usepackage{times}' to match the paper's font")
    p.add_argument("--require-correct", action="store_true",
                   help="draw only a stream the model answers correctly")
    args = p.parse_args(argv)
    from nfsm import device
    device.configure()
    import jax
    import jax.numpy as jnp

    job = TT.parse_token(args.token)
    if job["layers"] != 2 or not job["task"].startswith("question"):
        raise SystemExit("this figure is for two-layer models on a question task")
    model = runner.load_checkpoint(TT, args.token)
    mod, n = TT._task(job["task"])
    n_st = mod.n_states(n)
    words = Words(job["task"])
    out_dir = os.path.join(runner.RESULTS_ROOT, "tso", "figures", runner.job_name(args.token))
    os.makedirs(out_dir, exist_ok=True)

    # the mapping: head state -> register value, over many streams
    xb, yb, ab = TT.data_fn(job)(jax.random.PRNGKey(job["seed"] + 7), args.streams, args.stream_len)
    xb, yb = np.asarray(xb), np.asarray(yb)
    hb, (s0b, s1b), logb = run(model, xb)
    answer_acc = float(np.mean(np.argmax(logb[:, -1, n_st:], -1) == np.asarray(ab)))
    ok_state = np.argmax(logb[..., :n_st], -1) == np.argmax(yb[..., :n_st], -1)
    seq_acc = float(np.mean(ok_state.all(-1)))
    print(f"[traj] {args.streams} streams of {args.stream_len}: answer accuracy {answer_acc:.4f}, "
          f"sequence accuracy {seq_acc:.4f}, state accuracy {float(ok_state.mean()):.4f}", flush=True)
    regs_b = register_values(job["task"], yb)
    reg_names = ["first name (parse)"] + [f"holder of {it}" for it in words.items]
    maps = [head_mapping(s, regs_b, n, max(job['head_sizes'])) for s in (s0b, s1b)]
    reg_short = ["parse"] + [f"holder of {it}" for it in words.items]
    table = [{"layer": l + 1, "head": k + 1, "register": reg_names[r], "purity": round(pur, 4),
              "value_of_state": [words.names[v] if v < n else "-" for v in val],
              "state_usage": [int(u) for u in use]}
             for l, m in enumerate(maps) for k, (r, pur, val, use) in enumerate(m)]
    for row in table:
        print(f"[traj] layer {row['layer']} head {row['head']}: {row['register']:<20} purity "
              f"{row['purity']:.4f}  states -> {row['value_of_state']}  usage {row['state_usage']}",
              flush=True)
    for l, (s, m) in enumerate(((s0b, maps[0]), (s1b, maps[1]))):   # where a mapping is impure
        for k, (r, pur, val, _) in enumerate(m):
            off = val[s[..., k]] != regs_b[r]
            if 0 < off.sum() < 0.01 * off.size:
                prev = np.concatenate([np.full((xb.shape[0], 1), -1), xb[:, :-1]], 1)
                pairs = np.stack([prev[off], xb[off]], -1)
                w, c = np.unique(pairs, axis=0, return_counts=True)
                where = {" ".join(words.words[int(v)] if v >= 0 else "^" for v in a): int(b)
                         for a, b in zip(w, c)}
                print(f"[traj] layer {l + 1} head {k + 1}: {int(off.sum())} impure positions, at words "
                      f"{where}", flush=True)
                table[l * len(m) + k]["impure_at_words"] = where
    size = max(job["head_sizes"])
    pmat = [purity_matrix(s, regs_b, n, size) for s in (s0b, s1b)]
    base = [float(np.bincount(r.reshape(-1), minlength=n + 1).max() / r.size) for r in regs_b]
    print("[traj] purity matrix (rows: heads, columns: " + ", ".join(reg_names) + ")", flush=True)
    print("[traj]   baseline (majority value)  " + "  ".join(f"{b:.4f}" for b in base), flush=True)
    for l in range(2):
        for k in range(pmat[l].shape[0]):
            print(f"[traj]   layer {l + 1} head {k + 1}             "
                  + "  ".join(f"{v:.4f}" for v in pmat[l][k]), flush=True)
    kinds_b = token_kinds(xb, words)
    change = {}
    for l, s in enumerate((s0b, s1b)):
        moved = s[:, 1:] != s[:, :-1]                               # [B, T-1, K]
        for kind in TOKEN_KINDS:
            m = kinds_b[:, 1:] == kind
            if m.any():
                change.setdefault(kind, {})[f"layer {l + 1}"] = [round(float(moved[..., k][m].mean()), 3)
                                                                 for k in range(s.shape[-1])]
    print("[traj] share of positions where each head changes state, by token kind:", flush=True)
    for kind, v in change.items():
        print(f"[traj]   {kind:<14} layer 1 {v['layer 1']}  layer 2 {v['layer 2']}", flush=True)
    tracked = {row["register"] for row in table}
    print(f"[traj] registers no head tracks: {[r for r in reg_names if r not in tracked]}", flush=True)
    print(f"[traj] answer accuracy on the {args.streams} mapping streams: {answer_acc:.4f}", flush=True)

    # the drawn stream
    for k in range(1000):
        key = jax.random.PRNGKey(job["seed"] + 101 + k)
        x1, y1, a1 = TT.data_fn(job)(key, 1, args.length)
        x1, y1 = np.asarray(x1), np.asarray(y1)
        regs1 = [r[0] for r in register_values(job["task"], y1)]
        asked, true = int(x1[0, -2]) - n, int(np.asarray(a1)[0])
        first_holder = regs1[1 + asked][5 * n - 1]                   # after the preamble
        if swaps_moving(x1[0], regs1, asked, words) >= args.min_swaps and true != first_holder:
            if not args.require_correct:
                break
            _, _, lg = run(model, x1)
            if int(np.argmax(lg[0, -1, n_st:])) == true:
                break
    else:
        raise SystemExit("no stream matches")
    stream = [words.words[int(v)] for v in x1[0]]
    print(f"[traj] stream key seed+{101 + k}: {' '.join(stream)}", flush=True)

    # the plotted states are those of the full forward pass, noise free
    h1, (s0, s1), logits = run(model, x1)
    full = np.asarray(model(jnp.asarray(x1)))
    checked = check(Step(model, n_st), x1, h1, s0, s1, logits)
    checked["run_vs_model_logits_max_abs_diff"] = float(np.max(np.abs(full - logits)))
    rerun = run(model, x1)
    checked["rerun_identical_states"] = bool(np.array_equal(rerun[1][0], s0)
                                             and np.array_equal(rerun[1][1], s1))
    print(f"[traj] forward-pass check: {checked}", flush=True)
    if checked["run_vs_model_logits_max_abs_diff"] > 1e-4 or not checked["rerun_identical_states"]:
        raise SystemExit("the plotted states are not those of the forward pass")
    pred = int(np.argmax(logits[0, -1, n_st:]))

    pre_items = [int(x1[0, 5 * i + 3]) - n for i in range(n)]      # items in the order the preamble gives them
    reg_order = [0] + [1 + i for i in pre_items] if args.order_by_register else list(range(n + 1))
    rows, mism = [], {}
    for l, (s, m) in enumerate(((s0, maps[0]), (s1, maps[1]))):
        layer_rows = []
        for k, (r, pur, val, _) in enumerate(m):
            untracked = args.track_threshold is not None and pur < args.track_threshold
            if untracked:
                layer_rows.append((len(reg_order), k, (f"head {k + 1}", s[0, :, k], True)))
            else:
                layer_rows.append((reg_order.index(r), k, (f"head {k + 1}: {reg_short[r]}",
                                                           val[s[0, :, k]])))
                mism[f"layer {l + 1} head {k + 1}"] = int(np.sum(val[s[0, :, k]] != regs1[r]))
        if args.order_by_register:
            layer_rows.sort(key=lambda e: (e[0], e[1]))
        rows.append([e[2] for e in layer_rows])
    gt_rows = [(reg_names[r], regs1[r]) for r in reg_order]
    print(f"[traj] drawn stream, cells whose mapped value differs from the tracked register: {mism}",
          flush=True)
    print(f"[traj] asked {words.items[asked]}: model {words.names[pred]}, true {words.names[true]}",
          flush=True)
    out = os.path.join(out_dir, "head_trajectory")
    groups = [("layer 1", rows[0]), ("layer 2", rows[1])] + ([] if args.no_truth else [("truth", gt_rows)])
    plot = {"token": args.token, "tokens": stream, "names": list(words.names),
            "token_person": [int(v) if v < n else None for v in x1[0]],
            "rows": [{"group": g, "label": r[0], "values": [int(v) for v in r[1]],
                      "raw": bool(len(r) > 2 and r[2])} for g, grp in groups for r in grp]}
    with open(out + "_plot.json", "w") as f:              # all the figure needs (plot_trajectory.py)
        json.dump(plot, f)
    draw(plot, out, usetex=args.usetex, preamble=args.latex_preamble)
    with open(out + ".json", "w") as f:
        json.dump({"token": args.token, "stream": stream, "stream_key": job["seed"] + 101 + k,
                   "asked": words.items[asked], "pred": words.names[pred], "true": words.names[true],
                   "answer_acc_mapping_streams": answer_acc, "check": checked, "mapping": table,
                   "drawn_mismatches": mism, "seq_acc_mapping_streams": seq_acc,
                   "registers": reg_names, "purity_matrix": {f"layer {l + 1}": pmat[l].round(4).tolist()
                                                              for l in range(2)},
                   "baseline_purity": [round(b, 4) for b in base], "state_change_by_token": change,
                   "track_threshold": args.track_threshold}, f, indent=1)


if __name__ == "__main__":
    main()
