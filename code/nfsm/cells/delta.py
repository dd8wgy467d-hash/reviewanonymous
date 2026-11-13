"""DeltaProduct (Siems et al. 2025), x [B, T, m] -> [B, T, m].

Per head, the state H (d_k x d_v) takes n_h delta-rule steps per token:

    k_j = SiLU(W_j x) / |SiLU(W_j x)|,   v_j = V_j x,   beta_j = 2 sigmoid(U_j x)
    H_j = (I - beta_j k_j k_j^T) H_{j-1} + beta_j k_j v_j^T

beta in (0, 2) gives eigenvalues in (-1, 1), the [-1, 1] range used for state tracking; n_h = 1 is
DeltaNet[-1, 1]. Read-out H^T q, per-head RMSNorm, out_proj. The backbone conv replaces the layer's
own short conv. `parallel_mode` picks the associative scan or the sequential loop, `chunk` steps at
a time.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from nfsm.cells.nfsm import _from_chunks, _to_chunks

STATE_BUDGET = 1 << 25     # transition + write elements [B, C, heads, d_k, d_k + d_v] held at once


def _l2norm(x: Array, eps: float = 1e-6) -> Array:
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def _compose(left, right):
    """(A_r, B_r) o (A_l, B_l) = (A_r A_l, A_r B_l + B_r), the earlier step on the left."""
    A_l, B_l = left
    A_r, B_r = right
    return A_r @ A_l, A_r @ B_l + B_r


class DeltaProduct(nnx.Module):
    """DeltaProduct_{n_h}[-1, 1] with `heads` heads of d_k = d_v = d // heads."""

    def __init__(self, d: int, m: int, rngs: nnx.Rngs, *, n_householder: int = 2, heads: int = 8,
                 budget: int = STATE_BUDGET) -> None:
        if d % heads:
            raise ValueError(f"width {d} is not a multiple of {heads} heads")
        self.d, self.m = d, m
        self.n_h, self.heads, self.dk = n_householder, heads, d // heads
        self.budget = budget
        self.q_proj = nnx.Linear(m, d, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(m, n_householder * d, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(m, n_householder * d, use_bias=False, rngs=rngs)
        self.b_proj = nnx.Linear(m, n_householder * heads, use_bias=False, rngs=rngs)
        self.o_norm = nnx.RMSNorm(self.dk, rngs=rngs)
        self.out_proj = nnx.Linear(d, m, use_bias=False, rngs=rngs)

    def transitions(self, x: Array):
        """x [..., m] -> (A [..., heads, d_k, d_k], B [..., heads, d_k, d_v], q [..., heads, d_k])."""
        lead, H, dk, nh = x.shape[:-1], self.heads, self.dk, self.n_h
        k = _l2norm(jax.nn.silu(self.k_proj(x)).reshape(*lead, nh, H, dk))
        v = self.v_proj(x).reshape(*lead, nh, H, dk)
        beta = 2.0 * jax.nn.sigmoid(self.b_proj(x)).reshape(*lead, nh, H)
        q = _l2norm(jax.nn.silu(self.q_proj(x)).reshape(*lead, H, dk))
        A = jnp.broadcast_to(jnp.eye(dk, dtype=x.dtype), (*lead, H, dk, dk))
        B = jnp.zeros((*lead, H, dk, dk), x.dtype)
        for j in range(nh):                       # (I - b k k^T) M = M - b k (k^T M)
            kj, bj = k[..., j, :, :], beta[..., j, :, None, None]
            A = A - bj * kj[..., :, None] * jnp.einsum("...i,...ij->...j", kj, A)[..., None, :]
            B = (B - bj * kj[..., :, None] * jnp.einsum("...i,...ij->...j", kj, B)[..., None, :]
                 + bj * kj[..., :, None] * v[..., j, :, None, :])
        return A, B, q

    def _chunk_len(self, Bsz: int, T: int) -> int:
        per_step = self.heads * self.dk * 2 * self.dk
        limit = max(1, self.budget // max(1, Bsz * per_step))
        if limit >= T:
            return T
        C = 1
        while C * 2 <= limit:
            C *= 2
        return C

    def _states(self, x: Array, parallel_mode: bool) -> Array:
        """x [B, T, m] -> the state read-out H_t^T q_t [B, T, heads, d_v]."""
        Bsz, T = x.shape[:2]
        C = self._chunk_len(Bsz, T)
        pad = (-T) % C
        if pad:
            x = jnp.concatenate([x, jnp.zeros((Bsz, pad, x.shape[2]), x.dtype)], axis=1)

        @jax.checkpoint
        def body(h_prev, x_c):                     # h_prev [B, heads, d_k, d_v]
            A, Bw, q = self.transitions(x_c)       # [B, C, heads, ...]
            if parallel_mode:
                A_cum, B_cum = jax.lax.associative_scan(_compose, (A, Bw), axis=1)
                h = A_cum @ h_prev[:, None] + B_cum
            else:
                def step(hh, inp):
                    hh = inp[0] @ hh + inp[1]
                    return hh, hh
                _, h = jax.lax.scan(step, h_prev, (jnp.moveaxis(A, 1, 0), jnp.moveaxis(Bw, 1, 0)))
                h = jnp.moveaxis(h, 0, 1)
            return h[:, -1], jnp.einsum("bthkv,bthk->bthv", h, q)

        h0 = jnp.zeros((Bsz, self.heads, self.dk, self.dk), x.dtype)
        _, o = jax.lax.scan(body, h0, _to_chunks(x, C))
        return _from_chunks(o)[:, :T]

    def __call__(self, x: Array, parallel_mode: bool = True) -> Array:
        o = self.o_norm(self._states(x, parallel_mode))
        return self.out_proj(o.reshape(*x.shape[:2], self.d))
