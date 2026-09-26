"""JAX batch generators for the group word problems.

Each returns (x, y): x the letters [B, T] (token ids, or one-hot with ids=False) and y [B, T, C]
the one-hot class of the running product after each letter.
"""

import functools

import jax
import jax.numpy as jnp
from jax import Array

from nfsm.data.groups import (MATHIEU_DEGREE, MATHIEU_KEY_POINTS, mathieu_alphabet_array,
                              mathieu_key_table, mathieu_num_classes, permutation_alphabet_index,
                              permutation_cayley, permutation_num_classes)


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def generate_modular_count_batch(key: Array, batch_size: int, n: int, seq_len: int,
                                 ids: bool = True):
    """Z_n: letters uniform in 0..n-1, class = running sum mod n."""
    numbers = jax.random.randint(key, (batch_size, seq_len), 0, n)
    x = numbers if ids else jax.nn.one_hot(numbers, n)
    return x, jax.nn.one_hot(jnp.cumsum(numbers, axis=1) % n, n)


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def generate_permutation_batch(key: Array, batch_size: int, g: int, seq_len: int,
                               group: str = "S", ids: bool = True):
    """S_g / A_g: letters uniform over the adjacent generators, class = running product."""
    tbl = jnp.asarray(permutation_cayley(g, group))
    alpha = jnp.asarray(permutation_alphabet_index(g, group))
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, alpha.shape[0])
    running = jax.lax.associative_scan(lambda a, b: tbl[a, b], alpha[tokens], axis=1)
    x = tokens if ids else jax.nn.one_hot(tokens, alpha.shape[0])
    return x, jax.nn.one_hot(running, permutation_num_classes(g, group))


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def generate_mathieu_batch(key: Array, batch_size: int, seq_len: int, ids: bool = True):
    """M11: letters uniform over the 4 generators and inverses, class = running product.

    The running product is scanned in the 11-point representation and indexed by its 4-point key."""
    alpha = jnp.asarray(mathieu_alphabet_array())
    table = jnp.asarray(mathieu_key_table())
    tokens = jax.random.randint(key, (batch_size, seq_len), 0, alpha.shape[0])
    running = jax.lax.associative_scan(lambda a, b: jnp.take_along_axis(b, a, axis=-1),
                                       alpha[tokens], axis=1)
    key_ = jnp.zeros(running.shape[:-1], jnp.int32)
    for i in MATHIEU_KEY_POINTS:
        key_ = key_ * MATHIEU_DEGREE + running[..., i]
    x = tokens if ids else jax.nn.one_hot(tokens, alpha.shape[0])
    return x, jax.nn.one_hot(table[key_], mathieu_num_classes())
