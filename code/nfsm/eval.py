"""Deterministic evaluation: accuracies of a batch and the length sweep."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def make_acc_fn(model):
    """acc(model, x, y, ans=None) -> (token_acc, seq_acc, answer_acc).

    The prediction is the argmax over the first y.shape[-1] logits. answer_acc scores the logits
    after those at the last position against `ans` [B], and is nan when `ans` is None."""
    graphdef, _ = nnx.split(model)

    @jax.jit
    def _acc(state, x, y, ans):
        logits = nnx.merge(graphdef, state)(x)
        C = y.shape[-1]
        ok = jnp.argmax(logits[..., :C], -1) == jnp.argmax(y, -1)
        a = (jnp.float32(jnp.nan) if ans is None
             else jnp.mean(jnp.argmax(logits[:, -1, C:], -1) == ans))
        return jnp.mean(ok), jnp.mean(jnp.all(ok, axis=-1)), a

    def acc(model_ref, x, y, ans=None):
        _, state = nnx.split(model_ref)
        return tuple(float(v) for v in _acc(state, x, y, ans))

    return acc


def sweep_batch(length: int) -> int:
    """Sequences scored at one sweep length."""
    return 512 if length <= 1024 else 256 if length <= 8192 else 128


def sweep_accuracy(model, data_fn, seed: int, lens, token_budget: int) -> dict:
    """Accuracy at every length in `lens` on fresh batches, chunked to <= token_budget tokens.

    data_fn(key, B, T) -> (x, y) or (x, y, ans). Returns {token_acc, seq_acc, answer_acc}: lists
    aligned with `lens`."""
    acc_fn = make_acc_fn(model)
    out = {"token_acc": [], "seq_acc": [], "answer_acc": []}
    for L in lens:
        batch = sweep_batch(L)
        chunk = max(1, min(batch, token_budget // L))
        sums, done = np.zeros(3), 0
        while done < batch:
            bs = min(chunk, batch - done)
            b = data_fn(jax.random.PRNGKey(seed * 10_000_019 + L * 1_009 + done), bs, L)
            sums += np.asarray(acc_fn(model, *b)) * bs
            done += bs
        for k, v in zip(out, sums / batch):
            out[k].append(float(v))
    return out
