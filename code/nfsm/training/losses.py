"""Per-position weights and reductions of the training objective. All weights are stop-gradiented."""

import jax
import jax.numpy as jnp

sg = jax.lax.stop_gradient


def first_error_masks(correct, lam):
    """correct [B, T] -> (weights, frontier), both [B, T].

    weights = exp(-lam * max(0, t - t*)), t* the first wrong position; frontier is 1 at t* only.
    """
    B, T = correct.shape
    clean = jnp.cumprod(jnp.concatenate([jnp.ones((B, 1)), correct[:, :-1]], axis=1), axis=1)
    t_star = jnp.sum(clean, axis=1, keepdims=True) - 1.0
    t = jnp.arange(T, dtype=jnp.float32)[None, :]
    return sg(jnp.exp(-lam * jnp.maximum(0.0, t - t_star))), sg(clean * (1.0 - correct))


def class_balance(y, power=1.0):
    """y [B, T, C] one-hot -> [B, T] weight proportional to count_c^-power over the classes in the
    batch, mean 1. power 0 turns it off.
    """
    counts = jnp.sum(y, axis=(0, 1))
    n_pos = y.shape[0] * y.shape[1]
    raw = jnp.where(counts > 0, jnp.maximum(counts, 1.0) ** (-power), 0.0)
    w = raw * n_pos / jnp.maximum(jnp.sum(counts * raw), 1e-12)
    return sg(jnp.sum(y * w, axis=-1))


def weighted_sequence_mean(values, weights):
    """Mean over sequences of each sequence's weighted mean of values ([B, T] both)."""
    totals = jnp.maximum(weights.sum(axis=1), 1e-6)
    return jnp.mean((values * weights).sum(axis=1) / totals)
