"""The flip-flop monoid word problems DFF5 and FF.

The flip-flop acts on {0, 1}; its alphabet is the identity and the two constant maps reset (-> 0)
and set (-> 1), and the class of a prefix is its running composite: the identity until the first
non-identity letter, then the last non-identity letter seen. The two tasks differ only in the draw:
    dff5  uniform letters, at most four identities in a row (a definite language)
    ff    non-identity letters with probability 0.1 (identity runs of mean length 9)
    ff01  the same monoid drawn with probability 0.01 (runs of mean length 99), for evaluation
Element 0 is the identity; the class count is 3.
"""

import functools
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def compose(a: tuple, b: tuple) -> tuple:
    """a after b: (a o b)(i) = a(b(i))."""
    return tuple(a[j] for j in b)


def rank(m: tuple) -> int:
    return len(set(m))


def _closure(gens, degree: int) -> set:
    ident = tuple(range(degree))
    seen, frontier = {ident}, [ident]
    while frontier:
        w = frontier.pop()
        for a in gens:
            c = compose(a, w)
            if c not in seen:
                seen.add(c)
                frontier.append(c)
    return seen


@dataclass(frozen=True)
class MonoidSpec:
    name: str
    degree: int                         # size of the set acted on
    perm_gens: Tuple[tuple, ...]        # invertible generators
    collapse_gens: Tuple[tuple, ...]    # non-invertible generators
    draw_collapse_p: float = None       # None: uniform letters; else P(non-invertible letter)
    identity_run_cap: int = 0           # 0: unbounded; k: at most k identities in a row


MONOIDS = {s.name: s for s in (
    MonoidSpec("dff5", 2, (), ((0, 0), (1, 1)), identity_run_cap=4),
    MonoidSpec("ff", 2, (), ((0, 0), (1, 1)), draw_collapse_p=0.10),
    MonoidSpec("ff01", 2, (), ((0, 0), (1, 1)), draw_collapse_p=0.01),
)}


def monoid_spec(name: str) -> MonoidSpec:
    if name not in MONOIDS:
        raise KeyError(f"unknown monoid {name!r}; one of {', '.join(MONOIDS)}")
    return MONOIDS[name]


@functools.lru_cache(maxsize=None)
def monoid_elements(name: str) -> tuple:
    """All elements, identity first, then by descending rank."""
    spec = monoid_spec(name)
    ident = tuple(range(spec.degree))
    elems = _closure(spec.perm_gens + spec.collapse_gens, spec.degree)
    return (ident,) + tuple(sorted((e for e in elems if e != ident), key=lambda m: (-rank(m), m)))


@functools.lru_cache(maxsize=None)
def monoid_alphabet(name: str) -> tuple:
    """The group of units (identity first), then the collapsing generators."""
    spec = monoid_spec(name)
    ident = tuple(range(spec.degree))
    units = sorted(_closure(spec.perm_gens, spec.degree), key=lambda m: (-rank(m), m))
    return (ident,) + tuple(u for u in units if u != ident) + tuple(spec.collapse_gens)


def monoid_order(name: str) -> int:
    return len(monoid_elements(name))


def monoid_alphabet_size(name: str) -> int:
    return len(monoid_alphabet(name))


@functools.lru_cache(maxsize=None)
def monoid_cayley(name: str) -> np.ndarray:
    """[C, C] int32: tbl[a, b] = class of (b after a)."""
    elems = monoid_elements(name)
    index = {e: i for i, e in enumerate(elems)}
    tbl = np.empty((len(elems), len(elems)), dtype=np.int32)
    for a, ea in enumerate(elems):
        for b, eb in enumerate(elems):
            tbl[a, b] = index[compose(eb, ea)]
    return tbl


@functools.lru_cache(maxsize=None)
def monoid_alphabet_index(name: str) -> np.ndarray:
    """[A] int32: class of each letter."""
    index = {e: i for i, e in enumerate(monoid_elements(name))}
    return np.asarray([index[a] for a in monoid_alphabet(name)], dtype=np.int32)


def monoid_next_class(name: str) -> np.ndarray:
    """[A, C] int32: the class letter a sends class c to."""
    return monoid_cayley(name)[:, monoid_alphabet_index(name)].T.copy()


@functools.lru_cache(maxsize=None)
def _batch_fn(name: str):
    import jax
    import jax.numpy as jnp

    spec = monoid_spec(name)
    alphabet = monoid_alphabet(name)
    tbl_np, alpha_np = monoid_cayley(name), monoid_alphabet_index(name)
    n_letters, n_classes = len(alphabet), tbl_np.shape[0]
    unit_np = np.asarray([i for i, m in enumerate(alphabet) if rank(m) == spec.degree], np.int32)
    coll_np = np.asarray([i for i, m in enumerate(alphabet) if rank(m) < spec.degree], np.int32)
    p, cap = spec.draw_collapse_p, spec.identity_run_cap
    ident_id = alphabet.index(tuple(range(spec.degree)))

    def draw(key, shape):
        """Letters of the given shape from the spec's base distribution."""
        if p is None:
            return jax.random.randint(key, shape, 0, n_letters)
        k_w, k_u, k_c = jax.random.split(key, 3)
        u = jnp.asarray(unit_np)[jax.random.randint(k_u, shape, 0, len(unit_np))]
        c = jnp.asarray(coll_np)[jax.random.randint(k_c, shape, 0, len(coll_np))]
        return jnp.where(jax.random.bernoulli(k_w, p, shape), c, u)

    def capped(key, batch_size, seq_len):
        """The base draw with a collapsing letter forced after `cap` identities in a row.

        Inside a run of base identities the counter resets at every forced letter, so the forced
        positions are exactly the offsets o with o % (cap + 1) == cap from the run's start: computed
        in parallel from each position's offset in its run, not by a scan over the length."""
        k_b, k_f = jax.random.split(key)
        tok = draw(k_b, (batch_size, seq_len))
        is_id = tok == ident_id
        pos = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
        last_break = jax.lax.cummax(jnp.where(is_id, -1, pos), axis=1)   # last non-identity <= t
        offset = pos - last_break - 1                                   # identities before t in the run
        forced = is_id & (offset % (cap + 1) == cap)
        repl = jnp.asarray(coll_np)[jax.random.randint(k_f, (batch_size, seq_len), 0, len(coll_np))]
        return jnp.where(forced, repl, tok)

    @functools.partial(jax.jit, static_argnums=(1, 2))
    def fn(key, batch_size: int, seq_len: int):
        tokens = capped(key, batch_size, seq_len) if cap else draw(key, (batch_size, seq_len))
        tbl, alpha = jnp.asarray(tbl_np), jnp.asarray(alpha_np)
        running = jax.lax.associative_scan(lambda a, b: tbl[a, b], alpha[tokens], axis=1)
        return tokens, jax.nn.one_hot(running, n_classes)

    return fn


def generate_monoid_batch(key, batch_size: int, name: str, seq_len: int):
    """(letters [B, T] int32, one-hot running composite [B, T, C])."""
    return _batch_fn(name)(key, batch_size, seq_len)
