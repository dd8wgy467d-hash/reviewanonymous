"""Build results/paper_summary.md: FSA extraction (A), the TSO head trajectory (B), sanity (C).

    python scripts/paper_summary.py

Read-only: loads the FSA NFSM checkpoints, reruns extract_machine and a replica of it that also
records the first conflict / misread and the per-head reach, and reads the stored JSONs."""

import glob
import json
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments import runner  # noqa: E402
from experiments.fsa import tasks as FT  # noqa: E402
from experiments.tso import tasks as TT  # noqa: E402

TASK_ORDER = ["Z2", "Z16", "S3", "S4", "A5", "M11", "DFF5", "FF"]
OUT = os.path.join(runner.RESULTS_ROOT, "paper_summary.md")
COMMAND = "sbatch scripts/slurm_gpu.sh $PWD scripts/paper_summary.py   (runs: python scripts/paper_summary.py)"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def load_parts(suite, pattern="*nfsm*"):
    recs = {}
    for p in sorted(glob.glob(os.path.join(runner.RESULTS_ROOT, suite.SUITE, "parts", pattern + ".json"))):
        with open(p) as f:
            r = json.load(f)
        r.pop("history", None)
        recs[r["token"]] = r
    return recs


def load_faillen(suite):
    out = {}
    for p in glob.glob(os.path.join(runner.RESULTS_ROOT, suite.SUITE, "faillen", "*.json")):
        with open(p) as f:
            r = json.load(f)
        out[r["token"]] = r
    return out


def learned(rec, train_len):
    i = rec["lens"].index(train_len)
    return runner.solved_acc(rec, i) >= (1.0 if rec["cell"] == "nfsm" else 0.9)


def faillen_seed(fl, token):
    """(L_max or None, cap, start) of one seed from the bisection results, or None if absent."""
    cfg, seed = token.rsplit(",", 1)
    r = fl.get(cfg)
    if r is None or seed not in r["seeds"]:
        return None
    v = r["seeds"][seed]
    return v["failing_length"], v.get("cap", r["cap"]), r["start"]


def fmt_len(L, cap=None):
    if L is None:
        return f">={cap}" if cap else "none"
    return str(L)


# --- extraction replica ---------------------------------------------------------------------------

def replica(model, job):
    """extract_machine's algorithm, also returning the reached states, the first conflict and the
    first misread (letter, joint state, expected class, found / predicted class)."""
    import jax.numpy as jnp
    layer = list(model.layers)[0]
    cell = layer.cell
    A, delta = job["vocab"], FT.next_class(job["task"])
    letters = jnp.arange(A, dtype=jnp.int32)[:, None]
    maps = np.asarray(cell.transition_maps(layer.cell_input(model.embed_tokens(letters)))[:, 0])
    K = maps.shape[1]
    start = (0,) * K
    cls_of, frontier, conflicts, edges, first_conflict = {start: 0}, [start], 0, set(), None
    while frontier:
        s = frontier.pop()
        q = cls_of[s]
        for a in range(A):
            s2 = tuple(int(maps[a, k, s[k]]) for k in range(K))
            q2 = int(delta[a, q])
            edges.add((a, s2, q2))
            if s2 not in cls_of:
                cls_of[s2] = q2
                frontier.append(s2)
            elif cls_of[s2] != q2:
                conflicts += 1
                if first_conflict is None:
                    first_conflict = {"letter": a, "from_state": s, "from_class": q, "state": s2,
                                      "expected": q2, "found": cls_of[s2]}
    edges = sorted(edges)
    arr = np.asarray([(a, *s, q) for a, s, q in edges], dtype=np.int32)
    errors, first_misread = 0, None
    for i in range(0, len(arr), 4096):
        e = arr[i:i + 4096]
        h = model.embed_tokens(jnp.asarray(e[:, 0]))
        h = h + cell.readout(jnp.asarray(e[:, 1:1 + K]))
        h = h + layer.mlp(layer.norm_mlp(h))
        pred = np.asarray(jnp.argmax(model.head(h), -1))
        bad = np.nonzero(pred != e[:, -1])[0]
        errors += len(bad)
        if len(bad) and first_misread is None:
            j = bad[0]
            first_misread = {"letter": int(e[j, 0]), "state": tuple(int(v) for v in e[j, 1:1 + K]),
                             "expected": int(e[j, -1]), "predicted": int(pred[j])}
    return {"exact": conflicts == 0 and errors == 0, "states": len(cls_of),
            "classes": len(set(cls_of.values())), "conflicts": conflicts, "readout_errors": errors,
            "maps": maps, "cls_of": cls_of, "delta": delta,
            "first_conflict": first_conflict, "first_misread": first_misread}


def generator_matches_delta(task):
    """Does the task's own data generator follow next_class? (y_t = delta(x_t, y_{t-1}), y_-1 = 0)."""
    import jax
    job = FT.parse_token(f"{task},{FT.RUNGS[task][0]},nfsm,1,anchored,42")
    x, y = FT.data_fn(job)(jax.random.PRNGKey(0), 64, 64)
    x, y = np.asarray(x), np.argmax(np.asarray(y), -1)
    delta = FT.next_class(task)
    prev = np.concatenate([np.zeros((x.shape[0], 1), np.int64), y[:, :-1]], 1)
    ok = delta[x, prev] == y
    return int(ok.sum()), ok.size


# --- the report -----------------------------------------------------------------------------------

def main():
    from nfsm import device
    device.configure()
    from experiments.fsa.extract import extract_machine

    md = ["# Paper summary", "", f"- command: `{COMMAND}`", ""]

    fparts, ffl = load_parts(FT), load_faillen(FT)
    anomalies_a = []

    # A1 inventory
    md += ["## Part A. Extraction on the algebraic tasks (FSA)", "", "### A1. Inventory of NFSM part files", ""]
    groups = {}
    for tok, r in fparts.items():
        groups.setdefault(tok.rsplit(",", 1)[0], []).append(r)
    rows = []
    for cfg in sorted(groups, key=lambda c: (TASK_ORDER.index(c.split(",")[0]), c)):
        rs = groups[cfg]
        t, rung, cell, layers, arm = cfg.split(",")
        rows.append([t, rung, rs[0]["head_sizes"], layers, arm, ", ".join(str(r["seed"]) for r in rs),
                     "yes" if (rung == ("factored" if t in FT.FACTORED_HEADS else "regular") and arm == "anchored")
                     else ""])
    md += [table(["task", "rung", "head sizes", "layers", "arm", "seeds", "main table"], rows), ""]
    md += ["Main table configuration per task (anchored arm): " + ", ".join(
        f"{t}: {'factored ' + str(FT.head_sizes(t, 'factored')) if t in FT.FACTORED_HEADS else 'regular ' + str(FT.head_sizes(t, 'regular'))}"
        + (" (regular " + str(FT.head_sizes(t, 'regular')) + " also run)" if t in FT.FACTORED_HEADS and 'regular' in FT.RUNGS[t] else "")
        for t in TASK_ORDER), ""]

    # A2 per part, with the replica run on every checkpoint
    details = {}
    rows = []
    for tok in sorted(fparts, key=lambda k: (TASK_ORDER.index(k.split(",")[0]), k)):
        r = fparts[tok]
        job = FT.parse_token(tok)
        model = runner.load_checkpoint(FT, tok)
        ext_stored = r.get("extraction")
        recomputed = ext_stored is None
        ext_now = extract_machine(model, job)
        rep = replica(model, job)
        details[tok] = rep
        keys = ("exact", "states", "classes", "conflicts", "readout_errors")
        ext = ext_now if recomputed else ext_stored
        if any(ext_now[k] != ext[k] for k in keys) or any(rep[k] != ext_now[k] for k in keys):
            anomalies_a.append(f"- {tok}: stored extraction {ext_stored}, rerun {ext_now}, replica "
                               f"{ {k: rep[k] for k in keys} } disagree")
        fl = faillen_seed(ffl, tok)
        rows.append([r["task"], r["rung"], r["arm"], r["seed"], r["head_sizes"],
                     "yes" if learned(r, FT.TRAIN_LEN) else "no",
                     fmt_len(runner.failing_length(r), f"{FT.SWEEP_LENS[-1]}"),
                     "-" if fl is None else fmt_len(fl[0], fl[1]),
                     *[ext[k] for k in keys], "recomputed" if recomputed else "stored"])
        print(f"[summary] {tok}: {ext}", flush=True)
    md += ["### A2. Every NFSM part file", "",
           "sweep L = smallest swept length (8..32768, batch 512/256/128) below 100% sequence accuracy; "
           "bisection L = failing length from results/fsa/faillen (from L=64, batch 512).", "",
           table(["task", "rung", "arm", "seed", "heads", "learned@64", "sweep L", "bisection L",
                  "exact", "states", "classes", "conflicts", "readout_errors", "source"], rows), ""]

    # A3 per configuration
    md += ["### A3. Summary per configuration", ""]
    for rung in ("regular", "factored"):
        for arm in ("anchored", "free"):
            rows = []
            for t in TASK_ORDER:
                rs = [fparts[k] for k in fparts if k.startswith(f"{t},{rung},nfsm,1,{arm},")]
                if not rs:
                    continue
                lr = [r for r in rs if learned(r, FT.TRAIN_LEN)]
                ex = [r for r in lr if r["extraction"]["exact"]]
                st = [r["extraction"]["states"] for r in rs]
                rows.append([t, rs[0]["head_sizes"], f"{len(ex)}/{len(lr)}",
                             f"{sum(r['extraction']['exact'] for r in rs)}/{len(rs)}",
                             FT.n_classes(t), f"{min(st)}/{max(st)}",
                             sum(r["extraction"]["conflicts"] for r in rs),
                             sum(r["extraction"]["readout_errors"] for r in rs)])
            md += [f"**{rung}, {arm}**", "",
                   table(["task", "heads", "exact / learned", "exact / all seeds", "task classes",
                          "reached joint states min/max", "conflicts (total)", "readout_errors (total)"],
                         rows), ""]

    # A4 exact runs
    rows = []
    gen = {t: generator_matches_delta(t) for t in TASK_ORDER}
    for tok, rep in details.items():
        if not rep["exact"]:
            continue
        job = FT.parse_token(tok)
        K = len(job["head_sizes"])
        reach = [f"{len({s[k] for s in rep['cls_of']})}/{job['head_sizes'][k]}" for k in range(K)]
        per_class = np.bincount(list(rep["cls_of"].values()), minlength=job["n_classes"])
        ok = pairs = 0
        maps, delta = rep["maps"], rep["delta"]
        for s, q in rep["cls_of"].items():
            for a in range(job["vocab"]):
                s2 = tuple(int(maps[a, k, s[k]]) for k in range(K))
                pairs += 1
                ok += int(rep["cls_of"].get(s2, -1) == int(delta[a, q]))
        if ok != pairs:
            anomalies_a.append(f"- {tok}: exact but {pairs - ok} (state, letter) pairs break class(tau_a(s)) = delta(a, q)")
        rows.append([job["task"], job["rung"], job["arm"], job["seed"], " ".join(reach),
                     f"{per_class.min()}/{statistics.median(per_class.tolist()):g}/{per_class.max()}",
                     f"{ok}/{pairs}"])
    md += ["### A4. Exact runs: reach, joint states per class, check against the target", "",
           "heads reached = states of each head reached from the start state / its size; joint states "
           "per class = min/median/max over all task classes; check = pairs (reached joint state s, "
           "letter a) with class(tau_a(s)) = delta(a, class(s)).", "",
           "Target delta vs the task's own data generator (64 x 64 tokens, y_t = delta(x_t, y_t-1)): "
           + ", ".join(f"{t} {a}/{b}" for t, (a, b) in gen.items()), "",
           table(["task", "rung", "arm", "seed", "heads reached", "joint states per class", "check"], rows), ""]
    for t, (a, b) in gen.items():
        if a != b:
            anomalies_a.append(f"- {t}: next_class disagrees with the data generator on {b - a}/{b} tokens")

    # A5 non-exact runs
    rows = []
    for tok, rep in details.items():
        if rep["exact"]:
            continue
        r = fparts[tok]
        fc, fm = rep["first_conflict"], rep["first_misread"]
        sweep_ok = runner.failing_length(r) is None
        fl = faillen_seed(ffl, tok)
        rows.append([r["task"], r["rung"], r["arm"], r["seed"],
                     "-" if fc is None else f"letter {fc['letter']}: {fc['from_state']} (class {fc['from_class']}) -> {fc['state']}, expected {fc['expected']}, state already class {fc['found']}",
                     "-" if fm is None else f"letter {fm['letter']} into {fm['state']}: expected {fm['expected']}, predicted {fm['predicted']}",
                     "yes" if sweep_ok else "no"])
        if sweep_ok or (fl is not None and fl[0] is None):
            anomalies_a.append(f"- {tok}: not exact, yet no failure in the sweep"
                               + ("" if fl is None else f" (bisection {fmt_len(fl[0], fl[1])})"))
    md += ["### A5. Non-exact runs: first conflict and first misread", "",
           table(["task", "rung", "arm", "seed", "first conflict", "first misread", "no failure in sweep"], rows), ""]

    # A6 S3 in full
    s3 = sorted(k for k, v in details.items() if k.startswith("S3,factored,nfsm,1,anchored,") and v["exact"])
    if s3:
        tok = s3[0]
        rep = details[tok]
        from nfsm.data.groups import permutation_alphabet, permutation_cayley, permutation_elements
        elems, letters = permutation_elements(3, "S"), permutation_alphabet(3, "S")
        job = FT.parse_token(tok)
        md += [f"### A6. Extracted machine of {tok}", "",
               "Classes (index: permutation, as a tuple of images):", "",
               table(["class", "permutation"], [[i, p] for i, p in enumerate(elems)]), "",
               "Letters: " + ", ".join(f"{a} = {p}" for a, p in enumerate(letters)), "",
               "Cayley table tbl[a, b] = class of (b after a) (permutation_cayley):", "",
               table(["a \\ b"] + [str(b) for b in range(len(elems))],
                     [[a] + list(row) for a, row in enumerate(permutation_cayley(3, "S"))]), "",
               "Target delta(letter, class):", "",
               table(["letter \\ class"] + [str(c) for c in range(len(elems))],
                     [[a] + list(row) for a, row in enumerate(rep["delta"])]), ""]
        for k, d in enumerate(job["head_sizes"]):
            reached = sorted({s[k] for s in rep["cls_of"]})
            md += [f"Head {k + 1} ({d} states, reached {reached}): tau_a(state)", "",
                   table(["letter"] + [f"from {v}" for v in reached],
                         [[a] + [int(rep["maps"][a, k, v]) for v in reached] for a in range(job["vocab"])]), ""]
        md += ["Class of each reached joint state:", "",
               table(["joint state", "class", "permutation"],
                     [[s, q, elems[q]] for s, q in sorted(rep["cls_of"].items())]), ""]
    md += ["### Anomalies (Part A)", ""] + (anomalies_a or ["none"]) + [""]

    # Part B
    md += ["## Part B. TSO head-state trajectory", ""]
    cands = sorted(glob.glob(os.path.join(runner.RESULTS_ROOT, "tso", "figures", "question-4_exact_nfsm_2_free_*",
                                          "*.json")))
    if not cands:
        md += ["not produced", ""]
    else:
        path = cands[0]
        with open(path) as f:
            tr = json.load(f)
        fig = path.replace(".json", ".pdf")
        md += [f"- figure: `{os.path.relpath(fig, ROOT)}`" + ("" if os.path.exists(fig) else " (missing)"),
               f"- token: `{tr['token']}`; stream key {tr.get('stream_key')}",
               f"- stream ({len(tr['stream'])} words): {' '.join(tr['stream'])}",
               f"- forward-pass check: {tr['check']}",
               f"- question: who has the {tr['asked']}? model {tr['pred']}, true {tr['true']}",
               f"- answer accuracy on the 256 mapping streams: {tr['answer_acc_mapping_streams']:.4f}", "",
               table(["layer", "head", "register", "purity", "state -> value", "state usage"],
                     [[m["layer"], m["head"], m["register"], m["purity"], " ".join(m["value_of_state"]),
                       m["state_usage"]] for m in tr["mapping"]]), ""]
        if tr["answer_acc_mapping_streams"] < 0.5:
            md += [f"- anomaly: this run answers at chance ({tr['answer_acc_mapping_streams']:.3f}); "
                   f"the figure shows a model that did not learn the task.", ""]

    # Part C
    md += ["## Part C. Sanity", ""]
    tparts, tfl = load_parts(TT), load_faillen(TT)
    lines = []
    for suite in (FT, TT):
        p = os.path.join(runner.RESULTS_ROOT, suite.SUITE, "summary.md")
        if not os.path.exists(p):
            lines.append(f"- {os.path.relpath(p, ROOT)} does not exist: compared against "
                         f"results/{suite.SUITE}/faillen instead.")
    rows = []
    for suite, parts, fl in ((FT, fparts, ffl), (TT, tparts, tfl)):
        for tok, r in sorted(parts.items()):
            f = faillen_seed(fl, tok)
            if f is None:
                continue
            L, cap, start = f
            fl_learned = L is None or L > start
            i = r["lens"].index(suite.TRAIN_LEN)
            accs = f"seq {r['seq_acc'][i]:.4f}, answer {r['answer_acc'][i]:.4f}"
            if fl_learned != learned(r, suite.TRAIN_LEN):
                rows.append([suite.SUITE, tok, "learned", "yes" if learned(r, suite.TRAIN_LEN) else "no",
                             "yes" if fl_learned else "no", accs])
            sw = runner.failing_length(r)
            if r["task"] == "FF":                # scored on a sparser draw than its sweep
                continue
            if (L is None) != (sw is None) or (L is not None and sw > start and not sw // 2 < L <= sw):
                rows.append([suite.SUITE, tok, "failing length", fmt_len(sw, suite.SWEEP_LENS[-1]),
                             fmt_len(L, cap), accs])
    lines.append("")
    lines.append(table(["suite", "token", "field", "part (sweep)", "faillen (bisection)",
                        f"part accuracy at the training length"], rows)
                 if rows else "No NFSM part disagrees with the faillen results on learned flag or failing length.")
    lines.append("")
    lines.append("Failing length disagreement = the bisection L_max is not in (sweep L / 2, sweep L]; "
                 "FF is excluded (scored on a sparser draw than its sweep).")
    if any(r[0] == "tso" for r in rows):
        lines.append("- tso rows: the sweep scores min(sequence, answer) accuracy, the bisection the answer "
                     "only; an answer at 1.0 with sequence accuracy below 1.0 is learned by the table's rule.")
    lines.append("")
    missing = []
    for suite in (FT, TT):
        have_parts = {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(runner.RESULTS_ROOT, suite.SUITE, "parts", "*.json"))}
        cfgs = {}
        for tok in suite.iter_jobs(cells=suite.CELLS):
            cfgs.setdefault(tok.rsplit(",", 1)[0], []).append(tok)
        never = []
        for cfg, toks in cfgs.items():
            miss = [t.rsplit(",", 1)[1] for t in toks if runner.job_name(t) not in have_parts]
            if len(miss) == len(toks):
                never.append(cfg)
            elif miss:
                missing.append([suite.SUITE, cfg, ", ".join(miss)])
        if never:
            kinds = sorted({c.split(",")[0] for c in never})
            lines.append(f"- {suite.SUITE}: {len(never)} configurations defined but never run "
                         f"(tasks {', '.join(kinds)}).")
    lines.append("")
    lines.append(table(["suite", "configuration", "missing seeds"], missing) if missing
                 else "No configuration that was run misses any of seeds 42..46.")
    md += lines + [""]

    with open(OUT, "w") as f:
        f.write("\n".join(md))
    print(f"[summary] wrote {OUT} ({os.path.getsize(OUT)} bytes)", flush=True)


if __name__ == "__main__":
    main()
