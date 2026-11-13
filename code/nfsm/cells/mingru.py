"""minGRU with negative eigenvalues (minGRU^-), x [B, T, m] -> [B, T, m].

minGRU (Feng et al. 2024) with its decay stretched to (-1, 1), as Mamba to Mamba^-:

    z_t = sigmoid(W_z x_t + b_z),  h~_t = W_h x_t + b_h,  h_t = (1 - 2 z_t) h_{t-1} + z_t h~_t

The recurrence is affine in the state and runs through the same scans as the other affine
baselines (no log-space scan: a negative decay has no logarithm). Read-out out_proj(h_t).
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from nfsm.cells.ssm import _affine_parallel_scan, _affine_sequential_scan


class MinGRU(nnx.Module):
    """minGRU^- with d hidden units: decay 1 - 2 z_t in (-1, 1)."""

    def __init__(self, d: int, m: int, rngs: nnx.Rngs) -> None:
        self.d, self.m = d, m
        self.in_proj = nnx.Linear(m, 2 * d, rngs=rngs)            # (W_z, W_h) with biases
        self.out_proj = nnx.Linear(d, m, rngs=rngs)

    def coefficients(self, x: Array):
        """x [..., m] -> (a, b) = (1 - 2 z, z h~), each [..., d]."""
        k, h_tilde = jnp.split(self.in_proj(x), 2, axis=-1)
        z = jax.nn.sigmoid(k)
        return 1.0 - 2.0 * z, z * h_tilde

    def __call__(self, x: Array, parallel_mode: bool = True) -> Array:
        a, b = self.coefficients(x)
        scan = _affine_parallel_scan if parallel_mode else _affine_sequential_scan
        return self.out_proj(scan(a, b))
