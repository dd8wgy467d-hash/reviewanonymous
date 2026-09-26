"""`swap`: a stream of swap sentences over N people, starting from the identity assignment.

    alice swaps with bob . charlie swaps with alice . ...

Machine state = (parse, g): parse is IDLE (0) or HOLD_i (1 + i, the first name of the sentence
in progress), g in S_N with g(p) = who holds item p. The second name fires the transposition and
g' = (i j) o g. State index = parse * N! + index(g); state 0 is the start. Head family: the parse
head (N + 1 states) and N - 1 image heads (N states), head p holding g(p).
"""

import functools
import math

import numpy as np

from nfsm.data.groups import permutation_cayley, permutation_elements

NAMES = ("alice", "bob", "charlie", "dana", "eve", "frank")
FRAME = ("swaps", "with", ".")
SENTENCE_LEN = 5


def check_n(n: int) -> int:
    n = int(n)
    if not 3 <= n <= len(NAMES):
        raise ValueError(f"N = {n} is outside 3..{len(NAMES)}")
    return n


def alphabet(n: int) -> int:
    """Names 0..N-1, then `swaps`, `with`, `.`."""
    return check_n(n) + len(FRAME)


def n_parse(n: int) -> int:
    return check_n(n) + 1


def n_states(n: int) -> int:
    return n_parse(n) * math.factorial(check_n(n))


def n_answers(n: int) -> int:
    return 0


def head_sizes(n: int) -> tuple:
    return (n_parse(n),) + (check_n(n),) * (n - 1)


def head_codes(n: int) -> tuple:
    """(parse register, g(0), ..., g(N-2)) of every state."""
    ng = math.factorial(check_n(n))
    elems = np.asarray(permutation_elements(n, "S"), dtype=np.int32)
    state = np.arange(n_states(n), dtype=np.int32)
    return (state // ng,) + tuple(elems[state % ng, p] for p in range(n - 1))


@functools.lru_cache(maxsize=None)
def transposition_index(n: int) -> np.ndarray:
    """[N, N] int32: class in S_N of the transposition (i j); the identity on the diagonal."""
    index = {p: k for k, p in enumerate(permutation_elements(n, "S"))}
    tau = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            p = list(range(n))
            p[i], p[j] = p[j], p[i]
            tau[i, j] = index[tuple(p)]
    return tau


# --- host-side generation (numpy, seeded from the JAX key) ---------------------------------------

def host_rng(key) -> np.random.Generator:
    """numpy Generator seeded by a JAX PRNG key's raw bits."""
    import jax
    import jax.numpy as jnp
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)
    return np.random.default_rng([int(v) for v in np.asarray(key).ravel()])


def scan_cayley(cay: np.ndarray, letters: np.ndarray) -> np.ndarray:
    """Inclusive product along axis 1: out[:, t] = cay[...cay[l_0, l_1]..., l_t]."""
    out = np.array(letters, copy=True)
    off = 1
    while off < out.shape[1]:
        out[:, off:] = cay[out[:, :-off], out[:, off:]]
        off *= 2
    return out


@functools.lru_cache(maxsize=None)
def _onehot_fn(C: int):
    import jax
    return jax.jit(lambda y: jax.nn.one_hot(y, C))


def to_device(x: np.ndarray, y: np.ndarray, n_classes: int):
    """Host int tokens and states -> (int32 token ids, one-hot states) on the device."""
    import jax.numpy as jnp
    return jnp.asarray(x.astype(np.int32)), _onehot_fn(int(n_classes))(jnp.asarray(y.astype(np.int32)))


def swap_ints(rng: np.random.Generator, batch_size: int, n: int, seq_len: int):
    """(tokens, states), both [B, T] int32. Name pairs are uniform over ordered pairs i != j."""
    T, B = int(seq_len), int(batch_size)
    n_sent = -(-T // SENTENCE_LEN)
    ng = math.factorial(n)
    i = rng.integers(0, n, (B, n_sent))
    j = (i + rng.integers(1, n, (B, n_sent))) % n
    g = scan_cayley(permutation_cayley(n, "S"), transposition_index(n)[i, j])
    g_prev = np.concatenate([np.zeros_like(g[:, :1]), g[:, :-1]], axis=1)
    full = lambda v: np.full(i.shape, v)                                    # noqa: E731
    tokens = np.stack([i, full(n), full(n + 1), j, full(n + 2)], axis=-1)
    hold, zero = 1 + i, np.zeros_like(i)
    parse = np.stack([hold, hold, hold, zero, zero], axis=-1)
    perm = np.stack([g_prev, g_prev, g_prev, g, g], axis=-1)
    state = parse * ng + perm
    return tokens.reshape(B, -1)[:, :T].astype(np.int32), state.reshape(B, -1)[:, :T].astype(np.int32)


def generate(key, batch_size: int, n: int, seq_len: int):
    """(token ids [B, T], one-hot states [B, T, n_states])."""
    x, y = swap_ints(host_rng(key), batch_size, check_n(n), seq_len)
    return to_device(x, y, n_states(n))
