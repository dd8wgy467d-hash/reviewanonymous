"""Consistency checks of the forward pass, written to results/checks.txt: segmented = single
pass, recurrent decoding = parallel pass, prompt token counts, shared prefix state = full pass,
tau = inf = model as released, GPU = CPU.

    uv run python checks.py
"""
from __future__ import annotations

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mamba2_jax as m
import niah
from eval_niah import N_GEN, SEED, prefix_states

TEXT = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named "
        "after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. "
        "Locally nicknamed La dame de fer, it was constructed as the centrepiece of the 1889 World's Fair.")

lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


def main():
    p = m.load_params()
    tok = m.load_tokenizer()
    inf = jnp.float32(math.inf)
    log(f"backend: {jax.default_backend()}")

    # 1. segmenting and recurrent decoding
    ids = np.array([tok.encode(TEXT).ids])
    T = ids.shape[1]
    lg, st = m.run(p, ids, m.init_state(1), all_logits=True)
    lg2, st2 = m.run(p, ids, m.init_state(1), segment=32, all_logits=True)
    log(f"[1] T={T}: segments of 32 vs single pass: max|dlogit| {np.abs(lg2 - lg).max():.1e}, "
        f"max|dssm| {np.abs(st2[0] - st[0]).max():.1e}")
    _, s = m.run(p, ids[:, :T - 16], m.init_state(1))
    dl = []
    for t in range(T - 16, T):
        l, s = m.decode_step(p, jnp.asarray(ids[:, t]), s, inf)
        dl.append(np.asarray(l)[0])
    log(f"[1] recurrent decoding of the last 16 tokens vs parallel: max|dlogit| "
        f"{np.abs(np.stack(dl) - np.asarray(lg)[0, T - 16:]).max():.1e}  (logit scale {np.abs(lg).max():.0f})")

    # 2. prompt builder
    pb = niah.PromptBuilder(tok, SEED)
    prompts = niah.make_prompts(100, SEED)
    for layout in ("A", "B"):
        for h in (0, 7):
            for L in (0, 2050, 131075):
                for d in (0, 50, 1025, 8200):
                    for pr in prompts[:10]:
                        full = pb.full(layout, pr, h, L, d)
                        needle = tok.encode(niah.NEEDLE.format(**pr)).ids
                        question = tok.encode(niah.QUESTION.format(key=pr["key"])).ids
                        head = len(pb.instruction) if layout == "A" else 0
                        pre = 0 if layout == "A" else len(pb.instruction)
                        end_needle = head + L + pre + len(needle)
                        assert full[end_needle - len(needle):end_needle] == needle
                        assert full[len(full) - len(question):] == question
                        assert len(full) - len(question) - end_needle == d
                        assert tok.decode(full).count(pr["value"]) == 1
    for h in (0, 7):
        assert pb.before(h, 131075)[-2050:] == pb.before(h, 2050)
    assert pb.before(0, 2050) != pb.before(1, 2050) and pb.after(0, 500) != pb.after(1, 500)
    log("[2] needle ends exactly d tokens before the question: OK, layouts A and B, haystacks {0, 7}, "
        "L in {0, 2050, 131075}, d in {0, 50, 1025, 8200}, 10 prompts; prefixes of one haystack share their "
        "last lines; haystacks differ")

    # 3. shared prefix state vs full pass
    for layout, tau, L, d, i in (("A", math.inf, 2050, 250, 0), ("B", math.inf, 8200, 1025, 11),
                                 ("A", 512.0, 8200, 250, 22), ("B", 128.0, 2050, 50, 33)):
        pr, h = prompts[i], i // 10
        state = prefix_states(p, pb, layout, h, [L], tau)[L]
        lg_a, st_a = m.run(p, np.array([pb.suffix(layout, pr, h, d)]), state, tau)
        lg_b, st_b = m.run(p, np.array([pb.full(layout, pr, h, L, d)]), m.init_state(1), tau)
        ga = tok.decode(m.generate(p, lg_a, st_a, N_GEN, tau)[0].tolist())
        gb = tok.decode(m.generate(p, lg_b, st_b, N_GEN, tau)[0].tolist())
        log(f"[3] layout {layout} tau={tau} L={L} d={d} prompt {i}: shared-state vs full pass max|dlogit| "
            f"{np.abs(lg_a - lg_b).max():.1e}, same greedy output: {ga == gb} ({ga!r})")

    # 4. tau = inf is the model as is
    ids = np.array([pb.full("A", prompts[0], 0, 2050, 250)])
    lg_inf, _ = m.run(p, ids, m.init_state(1), math.inf)
    lg_big, _ = m.run(p, ids, m.init_state(1), 1e30)
    lg_512, _ = m.run(p, ids, m.init_state(1), 512.0)
    lg_asis, _ = m.run(p, ids, m.init_state(1), -0.0)  # -1/-0 = +inf: min(a, +inf) = a, no clamp
    log(f"[4] tau=inf vs no clamp: max|dlogit| {np.abs(lg_inf - lg_asis).max():.1e}; "
        f"tau=1e30 vs inf {np.abs(lg_big - lg_inf).max():.1e}; tau=512 vs inf {np.abs(lg_512 - lg_inf).max():.1e}")

    # 5. backend vs CPU
    if jax.default_backend() != "cpu":
        ids = np.array([pb.full("B", prompts[5], 0, 8200, 1025)])
        cpu = jax.devices("cpu")[0]
        for tau in (math.inf, 512.0):
            lg_g, st_g = m.run(p, ids, m.init_state(1), tau)
            with jax.default_device(cpu):
                pc = jax.device_put(p, cpu)
                lg_c, st_c = m.run(pc, ids, m.init_state(1), tau)
            log(f"[5] {jax.default_backend()} vs cpu, tau={tau}, T={ids.shape[1]}: max|dlogit| "
                f"{np.abs(np.asarray(lg_g) - np.asarray(lg_c)).max():.1e}, max|dssm| "
                f"{np.abs(np.asarray(st_g[0]) - np.asarray(st_c[0])).max():.1e}")
        # the grid's batch-5 path (one needle from the 5 L states, last-token logits, generation)
        from eval_niah import L_DEFAULT, stack_states
        for layout, tau, i, d in (("A", 2048.0, 3, 1025), ("B", 512.0, 42, 250)):
            h, res = i // 10, {}
            for dev in (jax.devices()[0], cpu):
                with jax.default_device(dev):
                    pd = jax.device_put(p, dev)
                    st = prefix_states(pd, pb, layout, h, L_DEFAULT, tau)
                    batch = stack_states([s for _, s in sorted(st.items())])
                    ids = np.tile(np.array([pb.suffix(layout, prompts[i], h, d)]), (len(L_DEFAULT), 1))
                    lg, s2 = m.run(pd, ids, batch, tau)
                    res[dev.platform] = (np.asarray(lg), [tok.decode(g.tolist())
                                                          for g in m.generate(pd, lg, s2, N_GEN, tau)])
            g, c = res[jax.default_backend()], res["cpu"]
            log(f"[5] grid batch of {len(L_DEFAULT)}, layout {layout} tau={tau} needle {i} d={d}: last-token "
                f"max|dlogit| {np.abs(g[0] - c[0]).max():.1e}, identical outputs {g[1] == c[1]}")
        # known backend limitation: all-position logits at batch >= 4 (used by nothing but perplexity,
        # which runs at batch 1)
        base = np.array([pb.full("A", prompts[0], 0, 250, 250)[:1024]] * 4)
        lg4, _ = m.run(p, base, m.init_state(4), all_logits=True)
        log(f"[5] note: all-position logits at batch >= 4 were seen to differ between identical rows on "
            f"{jax.default_backend()} for some inputs (up to 1.6e+02); here, max difference "
            f"{np.abs(np.asarray(lg4)[0] - np.asarray(lg4)[3]).max():.1e}. Only perplexity.py uses "
            f"all-position logits, and it runs one segment at a time")

    out = m.RESULTS / "checks.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
