"""`question`: a parcel stream followed by `who has the <item> ?`, answered at the last token.

The dense target is the parcel machine state at every token (the question is inert, so the state
is constant over it). The answer, the holder of the asked item, is a separate [B] target read from
N logits after the state logits. When T - 5 is not a multiple of 5, the (T - 5) mod 5 remaining
tokens are extra `.` inserted after distinct, randomly chosen sentences (at most one per
sentence), so every length is made of the same local patterns as the training length.
"""

import numpy as np

from nfsm.data.tso import parcel as PC
from nfsm.data.tso.swap import SENTENCE_LEN, check_n, host_rng, to_device

FRAME = PC.FRAME + ("who", "?")


def alphabet(n: int) -> int:
    return 2 * check_n(n) + len(FRAME)


def token_ids(n: int) -> dict:
    return {w: 2 * check_n(n) + k for k, w in enumerate(FRAME)}


n_states = PC.n_states
head_sizes = PC.head_sizes
head_codes = PC.head_codes


def n_answers(n: int) -> int:
    return check_n(n)


def generate(key, batch_size: int, n: int, seq_len: int):
    """(token ids [B, T], one-hot states [B, T, n_states], answer [B] in 0..N-1)."""
    import jax.numpy as jnp

    n, T = check_n(n), int(seq_len)
    pad = (T - SENTENCE_LEN) % SENTENCE_LEN
    body = T - SENTENCE_LEN - pad
    if body < SENTENCE_LEN * (n + 1):
        raise ValueError(f"seq_len {T} leaves no swap sentence before the question")
    ids = token_ids(n)
    rng = host_rng(key)
    x_body, st = PC.parcel_ints(rng, batch_size, n, body)
    final = st[:, -1]
    item = rng.integers(0, n, batch_size)
    holder = np.stack([PC.holder_of(n, q) for q in range(n)])          # [N, n_states]
    answer = holder[item, final].astype(np.int32)

    # One extra `.` after `pad` distinct sentences per sequence; its state is that sentence's last.
    S = body // SENTENCE_LEN
    chosen = np.argsort(rng.random((batch_size, S)), axis=1) < pad        # [B, S], pad per row
    keep = np.concatenate([np.ones((batch_size, S, SENTENCE_LEN), bool), chosen[..., None]], -1)
    xs = np.concatenate([x_body.reshape(batch_size, S, SENTENCE_LEN),
                         np.full((batch_size, S, 1), ids["."], np.int32)], -1)
    ys = st.reshape(batch_size, S, SENTENCE_LEN)
    ys = np.concatenate([ys, ys[..., -1:]], -1)
    x_body, st = xs[keep].reshape(batch_size, -1), ys[keep].reshape(batch_size, -1)

    fill = lambda w: np.full((batch_size,), ids[w], np.int32)          # noqa: E731
    q_tok = np.stack([fill("who"), fill("has"), fill("the"), (n + item).astype(np.int32),
                      fill("?")], axis=-1)
    q_st = np.broadcast_to(final[:, None], (batch_size, SENTENCE_LEN))
    x = np.concatenate([x_body, q_tok], axis=1)
    y = np.concatenate([st, q_st], axis=1)
    xd, yd = to_device(x, y, n_states(n))
    return xd, yd, jnp.asarray(answer)
