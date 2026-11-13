"""Causal depthwise short convolution."""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array


class ShortConv(nnx.Module):
    """Causal depthwise 1D convolution followed by SiLU: [B, T, m] -> [B, T, m]."""

    def __init__(self, m: int, kernel_size: int, rngs: nnx.Rngs) -> None:
        self.kernel_size = kernel_size
        init = nnx.initializers.variance_scaling(1.0, "fan_in", "uniform")
        self.kernel = nnx.Param(init(rngs.params(), (kernel_size, m), jnp.float32))
        self.bias = nnx.Param(jnp.zeros((m,), jnp.float32))

    def __call__(self, x: Array) -> Array:
        K, T = self.kernel_size, x.shape[1]
        w = self.kernel.value
        xpad = jnp.pad(x, ((0, 0), (K - 1, 0), (0, 0)))
        y = xpad[:, K - 1:, :] * w[K - 1]
        for k in range(K - 1):
            y = y + xpad[:, k:k + T, :] * w[k]
        return jax.nn.silu(y + self.bias.value)
