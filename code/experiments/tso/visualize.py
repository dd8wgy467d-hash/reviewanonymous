"""What a trained two-layer TSO NFSM computes on one stream, drawn as one figure.

    python -m experiments.tso.visualize --token question-4,exact,nfsm,2,anchored,42 [--length 40]

Writes results/tso/figures/<name>/trajectory.pdf|png and summary.json:
  top     every head's state along one short stream, above the task's registers (ground truth)
  bottom  one panel per kind of step (first name, assigned item, second name, final `?`): for every
          input a layer-1 update reads -- the word and each layer-0 head, at t-1 and t, the two
          positions of the width-2 causal conv -- the share of its alternative values that change
          a layer-1 head's next state (at `?`: the answer), the other inputs held fixed, averaged
          over the steps of that kind in `--streams` streams.
The one-step recomputation behind the bottom row is checked against the full forward pass first.
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

PERSON = ("#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#F0E442", "#D55E00")
EMPTY = "#EDEDED"
SHADES = ("#F7F7F7", "#D9D9D9", "#B8B8B8", "#969696", "#737373", "#555555")
KINDS = (("first", "first name of a sentence"), ("item", "item in `X has the item`"),
         ("second", "second name of a swap"), ("answer", "final `?`: the answer"))
CHUNK = 8192
RESAMPLES = 16           # whole layer-0 states drawn per step for the `all heads` rows


class Words:
    """Token ids of a TSO task and their words."""

    def __init__(self, task: str):
        from nfsm.data.tso.parcel import ITEMS
        from nfsm.data.tso.swap import NAMES
        mod, n = TT._task(task)
        self.n, self.ids = n, mod.token_ids(n)
        self.names, self.items = NAMES[:n], ITEMS[:n]
        self.words = list(self.names) + list(self.items) + [None] * (mod.alphabet(n) - 2 * n)
        for w, i in self.ids.items():
            self.words[i] = w


def registers(task: str, y: np.ndarray):
    """(names, values [R][B, T], value labels [R][v]) of the task's registers."""
    mod, n = TT._task(task)
    q = np.argmax(y[..., :mod.n_states(n)], -1)
    w = Words(task)
    ini = [nm[0].upper() for nm in w.names]
    names = ["parse"] + [f"holder of {it}" for it in w.items]
    labels = [["–"] + [f"{c}…" for c in ini]] + [ini + ["·"]] * n
    return names, [c[q] for c in mod.head_codes(n)], labels


def label_color(label: str) -> str:
    if label in ("–", "·"):
        return EMPTY
    return PERSON[ord(label[0]) - 65]


def step_kinds(x: np.ndarray, words: Words) -> np.ndarray:
    """[B, T] kind of each step: first, item, second, answer, or '' (no layer-1 head moves)."""
    B, T = x.shape
    start = np.zeros_like(x)
    for t in range(1, T):
        start[:, t] = np.where(x[:, t - 1] == words.ids["."], t, start[:, t - 1])
    pos = np.arange(T)[None] - start
    lead = np.take_along_axis(x, start, 1)
    name, item = x < words.n, (x >= words.n) & (x < 2 * words.n)
    kinds = np.full(x.shape, "", dtype=object)
    kinds[name & (pos == 0)] = "first"
    kinds[item & (pos == 3) & (lead < words.n)] = "item"
    kinds[name & (pos == 3)] = "second"
    kinds[x == words.ids["?"]] = "answer"
    return kinds


# --- the model, whole and one step at a time ------------------------------------------------------

def run(model, x):
    """(layer-0 output stream [B, T, m], head states per layer [B, T, K], logits)."""
    h = model.embed_tokens(x)
    states, streams = [], []
    for layer in model.layers:
        states.append(np.asarray(layer.cell.states(layer.cell_input(h))))
        h = layer(h)[0]
        streams.append(h)
    return streams[0], states, np.asarray(model.head(h))


class Step:
    """One step of a two-layer model recomputed from its inputs (noise free)."""

    def __init__(self, model, n_states: int):
        self.model, (self.l0, self.l1) = model, list(model.layers)
        self.n_states = n_states

    def stream(self, w, s0):
        """Layer-0 output at one position from its word [N] and layer-0 states [N, K0]: [N, m]."""
        import jax.numpy as jnp
        h = self.model.embed_tokens(jnp.asarray(w)) + self.l0.cell.readout(jnp.asarray(s0))
        return h + self.l0.mlp(self.l0.norm_mlp(h))

    def next1(self, h_prev, h_cur, s1_prev):
        """Layer-1 states after the step, from the layer-0 output at t-1 and t: [N, K1]."""
        import jax.numpy as jnp
        u = self.l1.cell_input(jnp.stack([jnp.asarray(h_prev), jnp.asarray(h_cur)], 1))
        maps = self.l1.cell.transition_maps(u)[:, 1]
        return np.asarray(jnp.take_along_axis(maps, jnp.asarray(s1_prev)[..., None], -1)[..., 0])

    def answer(self, h_cur, s1_cur):
        """Answer read at the position: [N]."""
        import jax.numpy as jnp
        h = jnp.asarray(h_cur) + self.l1.cell.readout(jnp.asarray(s1_cur))
        h = h + self.l1.mlp(self.l1.norm_mlp(h))
        return np.asarray(jnp.argmax(self.model.head(h)[..., self.n_states:], -1))


def chunked(fn, *arrays):
    """fn over the leading axis in chunks of CHUNK, concatenated."""
    n = len(arrays[0])
    return np.concatenate([np.asarray(fn(*[a[i:i + CHUNK] for a in arrays])) for i in range(0, n, CHUNK)])


def check(step: Step, x, h1, s0, s1, logits) -> dict:
    """The one-step recomputation against the forward pass: stream, layer-1 updates, answer."""
    B, T, m = h1.shape
    h1 = np.asarray(h1)
    hs = chunked(step.stream, x.reshape(-1), s0.reshape(B * T, -1)).reshape(B, T, m)
    nxt = chunked(step.next1, h1[:, :-1].reshape(-1, m), h1[:, 1:].reshape(-1, m),
                  s1[:, :-1].reshape(-1, s1.shape[-1]))
    ans = step.answer(h1[:, -1], s1[:, -1])
    out = {"stream_max_abs_diff": float(np.max(np.abs(hs - h1))),
           "layer1_update_mismatches": int(np.sum(nxt != s1[:, 1:].reshape(-1, s1.shape[-1]))),
           "answer_mismatches": int(np.sum(ans != np.argmax(logits[:, -1, step.n_states:], -1)))}
    ok = (out["stream_max_abs_diff"] < 1e-4 and not out["layer1_update_mismatches"]
          and not out["answer_mismatches"])
    print(f"[viz] one-step recomputation vs forward pass: {out} -> {'OK' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit("the one-step recomputation does not reproduce the model")
    return out


def dependence(step: Step, x, h1, s0, s1, kinds, kind: str, sizes, vocab: int):
    """(values [sources, targets], source keys, number of steps): the share of each source's
    alternative values that change each target, every other input held fixed. A source ("l0all",
    lag) replaces the whole layer-0 state by one seen at a random other position, RESAMPLES times."""
    b, t = np.nonzero(kinds == kind)
    if kind != "answer":
        b, t = b[t > 0], t[t > 0]
    K0, K1 = s0.shape[-1], s1.shape[-1]
    h1 = np.asarray(h1)
    if kind == "answer":
        sources = ([("word", 0), ("l0all", 0)] + [("l0", k, 0) for k in range(K0)]
                   + [("l1", j, 0) for j in range(K1)])
        base = step.answer(h1[b, t], s1[b, t])
        n_out = 1
    else:
        sources = ([("word", 1), ("word", 0), ("l0all", 1), ("l0all", 0)]
                   + [("l0", k, 1) for k in range(K0)] + [("l0", k, 0) for k in range(K0)])
        base = s1[b, t]
        n_out = K1
    tot, cnt = np.zeros((len(sources), n_out)), np.zeros(len(sources))
    rng = np.random.default_rng(0)
    for si, src in enumerate(sources):
        lag = src[-1]
        tb = t - lag
        alts = (range(vocab) if src[0] == "word" else range(RESAMPLES) if src[0] == "l0all"
                else range(sizes[src[1]]))
        for alt in alts:
            w, st0, st1 = x[b, tb].copy(), s0[b, tb].copy(), s1[b, t].copy()
            if src[0] == "l0all":
                st0 = s0[rng.integers(0, s0.shape[0], len(b)), rng.integers(0, s0.shape[1], len(b))]
                keep = (st0 != s0[b, tb]).any(-1)
            else:
                cur = w if src[0] == "word" else (st0 if src[0] == "l0" else st1)[:, src[1]]
                keep = cur != alt
            if not keep.any():
                continue
            if src[0] == "word":
                w[:] = alt
            elif src[0] == "l0":
                st0[:, src[1]] = alt
            elif src[0] == "l1":
                st1[:, src[1]] = alt
            bk, tk = b[keep], t[keep]
            h_new = h1[bk, tk] if src[0] == "l1" else chunked(step.stream, w[keep], st0[keep])
            if kind == "answer":
                res = step.answer(h_new, st1[keep])
                tot[si, 0] += np.sum(res != base[keep])
            else:
                hp = h_new if lag == 1 else h1[bk, tk - 1]
                hc = h_new if lag == 0 else h1[bk, tk]
                res = chunked(step.next1, hp, hc, s1[bk, tk - 1])
                tot[si] += np.sum(res != base[keep], 0)
            cnt[si] += keep.sum()
    return tot / np.maximum(cnt, 1)[:, None], sources, len(b)


# --- the figure -----------------------------------------------------------------------------------

def draw(job, words, x, s0, s1, regs, reg_names, reg_labels, kinds, pred, true_ans, deps, out: str,
         n_streams: int, train_len: int) -> None:
    from nfsm.plotting import plt_setup
    plt = plt_setup({"text.usetex": False, "font.family": "DejaVu Sans"})
    T, K0, K1 = x.shape[0], s0.shape[-1], s1.shape[-1]
    conv, n0 = job["conv_kernel"], max(job["head_sizes"])
    short = [r.replace("holder of ", "") for r in reg_names]

    groups = [
        (f"Layer 0  ·  {K0} heads × {n0} states, trained freely  ·  its update at step t reads the words "
         f"at t−1 and t (causal conv, width {conv})",
         [(f"L0 head {k + 1}", [f"s{v}" for v in s0[:, k]], [SHADES[v] for v in s0[:, k]])
          for k in range(K0)]),
        (f"Layer 1  ·  {K1} heads × {n0} states, read-out pinned to the task registers  ·  its update at "
         f"step t reads the layer-0 output at t−1 and t (causal conv, width {conv})",
         [(f"L1 head {j + 1} = {reg_names[j]}", [reg_labels[j][v] for v in s1[:, j]],
           [label_color(reg_labels[j][v]) for v in s1[:, j]]) for j in range(K1)]),
        ("Ground truth  ·  the task registers",
         [(reg_names[j], [reg_labels[j][v] for v in regs[j]],
           [label_color(reg_labels[j][v]) for v in regs[j]]) for j in range(len(regs))]),
    ]
    n_rows = sum(len(r) for _, r in groups)
    top_h, bottom_h = 0.33 * n_rows + 2.9, 4.4
    fig = plt.figure(figsize=(max(0.44 * T + 5.0, 19), top_h + bottom_h))
    gs = fig.add_gridspec(2, 4, height_ratios=[top_h, bottom_h], width_ratios=[1, 1, 1, 0.62],
                          hspace=0.42, wspace=0.55, bottom=0.07, top=0.97)
    ax = fig.add_subplot(gs[0, :])

    starts = [t for t in range(T) if t == 0 or x[t - 1] == words.ids["."]]
    y, row_y = 0.0, []
    for title, rows in groups:
        ax.text(-0.5, y + 0.15, title, ha="left", va="bottom", fontsize=9.5, fontweight="bold")
        y += 0.35
        for label, vals, cols in rows:
            for t in range(T):
                ax.add_patch(plt.Rectangle((t - 0.5, y), 1, 1, facecolor=cols[t], edgecolor="white", lw=0.8))
                if t == 0 or vals[t] != vals[t - 1]:
                    ax.text(t, y + 0.5, vals[t], ha="center", va="center", fontsize=7.5,
                            color="white" if cols[t] in SHADES[3:] else "black")
            ax.text(-0.8, y + 0.5, label, ha="right", va="center", fontsize=8.5)
            row_y.append(y)
            y += 1.0
        y += 0.9
    l1_y, gt_y = row_y[K0:K0 + K1], row_y[K0 + K1:]
    for j in range(K1):                          # a red frame: a layer-1 state off the ground truth
        for t in range(T):
            if s1[t, j] != regs[j][t]:
                ax.add_patch(plt.Rectangle((t - 0.5, l1_y[j]), 1, 1, fill=False, edgecolor="red", lw=1.6))
    ends = starts[1:] + [T]
    for s, e in zip(starts, ends):               # sentences: a separator and their kind
        if s > 0:
            ax.axvline(s - 0.5, color="#444444", lw=0.6, ls=(0, (2, 2)))
        seg = list(x[s:e])
        kind = ("question" if x[s] == words.ids["who"] else "assignment" if words.ids["has"] in seg
                else "swap" if words.ids["swaps"] in seg else "")
        if kind and e - s > 1:
            ax.text((s + e - 1) / 2, -0.55, kind, ha="center", va="bottom", fontsize=8, color="#444444",
                    style="italic")
    marks = {k: str(i + 1) for i, (k, _) in enumerate(KINDS)}
    for t in range(T):                           # the kind of each step, matching the bottom row
        if kinds[t]:
            ax.text(t, y - 0.35, marks[kinds[t]], ha="center", va="center", fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.18", facecolor="white", edgecolor="#333333", lw=0.8))
    ax.text(-0.8, y - 0.35, "kind of step", ha="right", va="center", fontsize=8.5)
    mid = (gt_y[0] + gt_y[-1] + 1) / 2
    ax.text(T + 0.3, mid, f"answer: {words.words[pred]}\n(true: {true_ans})", ha="left", va="center",
            fontsize=9, fontweight="bold")
    ax.set_xlim(-0.5, T + 2.6)
    ax.set_ylim(y + 0.1, -1.1)
    ax.set_xticks(range(T))
    ax.set_xticklabels([words.words[int(v)] for v in x], rotation=55, ha="right", fontsize=8.5)
    for tick, v in zip(ax.get_xticklabels(), x):
        if v < words.n:
            tick.set_color(PERSON[int(v)])
            tick.set_fontweight("bold")
    ax.set_yticks([])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    names = ", ".join(f"{nm[0].upper()} = {nm}" for nm in words.names)
    ax.set_title(f"{job['task']}, {job['layers']}-layer NFSM (anchored arm, seed {job['seed']}): one stream "
                 f"of {T} words.  {names}.  parse: – idle, A… = the sentence began with alice.  "
                 f"holder: · = not assigned yet.\nLayer-0 states s0–s4 have no task meaning of their own. "
                 f"A label marks where a row changes state; a red frame would mark a layer-1 state that "
                 f"differs from the ground truth.", fontsize=9, loc="left")

    def src_name(s):
        who = ("word" if s[0] == "word" else f"layer 0, all {K0} heads" if s[0] == "l0all"
               else f"L0 head {s[1] + 1}" if s[0] == "l0" else f"L1 head {s[1] + 1} ({short[s[1]]})")
        return who + (" at t−1" if s[-1] else " at t")

    for i, (kind, desc) in enumerate(KINDS):     # bottom row: what each kind of step depends on
        vals, sources, n_ev = deps[kind]
        axk = fig.add_subplot(gs[1, i])
        axk.imshow(vals, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for r in range(vals.shape[0]):
            for c in range(vals.shape[1]):
                if vals[r, c] >= 0.05:
                    axk.text(c, r, f"{vals[r, c]:.2f}", ha="center", va="center", fontsize=7,
                             color="white" if vals[r, c] > 0.6 else "black")
        axk.set_yticks(range(len(sources)), [src_name(s) for s in sources], fontsize=7.5)
        for r, s_ in enumerate(sources):
            if s_[0] == "l0all":
                axk.get_yticklabels()[r].set_fontweight("bold")
        n_top = sum(s_[0] in ("word", "l0all") for s_ in sources)
        axk.axhline(n_top - 0.5, color="black", lw=1.0)
        cols = ["answer"] if kind == "answer" else [f"L1 head {j + 1}\n({short[j]})" for j in range(K1)]
        axk.set_xticks(range(len(cols)), cols, fontsize=7.5)
        axk.xaxis.tick_top()
        ex = next((t for t in range(T) if kinds[t] == kind and (kind == "answer" or t > 0)), None)
        example = ""
        if ex is not None:
            s = max(u for u in starts if u <= ex)
            e = next((u for u in starts if u > ex), T)
            example = " ".join(f"[{words.words[int(x[u])]}]" if u == ex else words.words[int(x[u])]
                               for u in range(s, e))
        axk.set_title(f"({i + 1}) {desc}\ne.g. {example}\n{n_ev} steps", fontsize=8.5, pad=34)
    fig.text(0.5, 0.035, f"Bottom row: a cell is the share of an input's alternative values that change the "
             f"result -- a layer-1 head's next state, or at (4) the answer -- with every other input held "
             f"fixed. Alternatives: every other word; a head's other states; for 'layer 0, all heads', the "
             f"whole layer-0 state seen at a random other position ({RESAMPLES} draws).\nSingle layer-0 "
             f"heads score low because the sentence's first name is spread over all five heads. Averaged "
             f"over the steps of that kind in {n_streams} streams of {train_len} words; there is no conv "
             f"after layer 1, so the answer reads position t only.", ha="center", va="top", fontsize=8.5)
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}.pdf|png", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="draw what a trained two-layer TSO NFSM computes")
    p.add_argument("--token", required=True)
    p.add_argument("--length", type=int, default=40, help="words in the drawn stream")
    p.add_argument("--streams", type=int, default=64, help="streams the bottom row averages over")
    args = p.parse_args(argv)
    from nfsm import device
    device.configure()
    import jax

    job = TT.parse_token(args.token)
    if job["layers"] != 2:
        raise SystemExit("this figure is for two-layer models")
    model = runner.load_checkpoint(TT, args.token)
    mod, n = TT._task(job["task"])
    n_st = mod.n_states(n)
    words = Words(job["task"])
    step = Step(model, n_st)
    out_dir = os.path.join(runner.RESULTS_ROOT, "tso", "figures", runner.job_name(args.token))
    os.makedirs(out_dir, exist_ok=True)

    xb, yb, ab = TT.data_fn(job)(jax.random.PRNGKey(job["seed"] + 7), args.streams, TT.TRAIN_LEN)
    xb, yb = np.asarray(xb), np.asarray(yb)
    h1b, (s0b, s1b), logb = run(model, xb)
    dense = float(np.mean(np.argmax(logb[..., :n_st], -1) == np.argmax(yb, -1)))
    answer = float(np.mean(np.argmax(logb[:, -1, n_st:], -1) == np.asarray(ab)))
    print(f"[viz] {args.token}: dense acc {dense:.4f}, answer acc {answer:.4f} on {args.streams} streams",
          flush=True)
    checked = check(step, xb, h1b, s0b, s1b, logb)
    regs_b = registers(job["task"], yb)[1]
    agree = [float(np.mean(s1b[..., j] == regs_b[j])) for j in range(s1b.shape[-1])]
    print(f"[viz] layer-1 head j == register j: {[round(a, 4) for a in agree]}", flush=True)
    kinds_b = step_kinds(xb, words)
    stray = int(np.sum((s1b[:, 1:] != s1b[:, :-1]).any(-1) & (kinds_b[:, 1:] == "")))
    print(f"[viz] layer-1 updates outside the four kinds of step: {stray}", flush=True)
    deps = {k: dependence(step, xb, h1b, s0b, s1b, kinds_b, k, tuple(job["head_sizes"]), len(words.words))
            for k, _ in KINDS}

    x1, y1, a1 = TT.data_fn(job)(jax.random.PRNGKey(job["seed"] + 101), 1, args.length)
    x1 = np.asarray(x1)
    _, (s0, s1), logits = run(model, x1)
    reg_names, regs, reg_labels = registers(job["task"], np.asarray(y1))
    pred = int(np.argmax(logits[0, -1, n_st:]))
    draw(job, words, x1[0], s0[0], s1[0], [r[0] for r in regs], reg_names, reg_labels,
         step_kinds(x1, words)[0], pred, words.names[int(np.asarray(a1)[0])], deps,
         os.path.join(out_dir, "trajectory"), args.streams, TT.TRAIN_LEN)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"token": args.token, "dense_acc": dense, "answer_acc": answer, "check": checked,
                   "layer1_equals_register": agree, "stray_layer1_updates": stray,
                   "stream": [words.words[int(v)] for v in x1[0]],
                   "dependence": {k: {"steps": d[2], "sources": [list(map(str, s)) for s in d[1]],
                                      "values": d[0].round(4).tolist()} for k, d in deps.items()}},
                  f, indent=1)


if __name__ == "__main__":
    main()
