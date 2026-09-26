"""`parcel`: the initial assignment is declared in the text, then swap sentences follow.

    alice has the key . bob has the ring . charlie has the coin . alice swaps with bob . ...

Machine state = (parse, h) with h[q] = holder of item q, or UNSET (= N) before the preamble names
it. States are those reachable under the grammar (44 / 202 / 1132 for N = 3 / 4 / 5), in BFS
order from (IDLE, all UNSET). Head family: the parse head and one head of N + 1 states per item.
"""

import functools

import numpy as np

from nfsm.data.groups import permutation_cayley, permutation_elements
from nfsm.data.tso.swap import (NAMES, SENTENCE_LEN, check_n, host_rng, n_parse, scan_cayley,
                                to_device, transposition_index)

ITEMS = ("ring", "key", "coin", "stamp", "card", "token")
FRAME = ("swaps", "with", "has", "the", ".")


def alphabet(n: int) -> int:
    """Names, items, then the frame words."""
    return 2 * check_n(n) + len(FRAME)


def token_ids(n: int) -> dict:
    return {w: 2 * check_n(n) + k for k, w in enumerate(FRAME)}


def _step(n: int, st: tuple, x: int) -> tuple:
    """One transition of st = (parse, h) under token x."""
    p, h = st
    if x < n:                                           # a name
        if p == 0:
            return (1 + x, h)
        i, j = p - 1, x
        return (0, tuple(i if v == j else (j if v == i else v) for v in h))
    if n <= x < 2 * n and p != 0:                       # an item closes an assignment
        return (0, tuple(p - 1 if q == x - n else v for q, v in enumerate(h)))
    return (p, h)


@functools.lru_cache(maxsize=None)
def machine(n: int):
    """(states, state -> index, dfa [n_states, alphabet]) over the grammatical streams."""
    check_n(n)
    ids, unset = token_ids(n), n
    start = (0, (unset,) * n)

    def sentences(st):
        free = [q for q in range(n) if st[1][q] == unset]
        if free:
            k = n - len(free)
            return [[k, ids["has"], ids["the"], n + q, ids["."]] for q in free]
        return [[i, ids["swaps"], ids["with"], j, ids["."]]
                for i in range(n) for j in range(n) if i != j]

    seen, order, boundary, frontier = {start: 0}, [start], {start}, [start]
    while frontier:
        st = frontier.pop()
        for sent in sentences(st):
            cur = st
            for x in sent:
                cur = _step(n, cur, x)
                if cur not in seen:
                    seen[cur] = len(order)
                    order.append(cur)
            if cur not in boundary:
                boundary.add(cur)
                frontier.append(cur)
    dfa = np.zeros((len(order), alphabet(n)), np.int32)
    for q, st in enumerate(order):
        for x in range(alphabet(n)):
            dfa[q, x] = seen.get(_step(n, st, x), q)
    return tuple(order), seen, dfa


def n_states(n: int) -> int:
    return len(machine(n)[0])


def n_answers(n: int) -> int:
    return 0


def holder_of(n: int, q: int) -> np.ndarray:
    """[n_states] holder of item q (N = unset)."""
    return np.array([s[1][q] for s in machine(n)[0]], np.int32)


def head_sizes(n: int) -> tuple:
    return (n_parse(n),) + (check_n(n) + 1,) * n


def head_codes(n: int) -> tuple:
    """(parse register, holder of item 0, ..., holder of item N-1) of every state."""
    parse = np.array([s[0] for s in machine(n)[0]], np.int32)
    return (parse,) + tuple(holder_of(n, q) for q in range(check_n(n)))


@functools.lru_cache(maxsize=None)
def _full_index(n: int) -> np.ndarray:
    """[N + 1, N!] -> state index of the fully assigned states (parse, g)."""
    states, index, _ = machine(n)
    pos = {p: k for k, p in enumerate(permutation_elements(n, "S"))}
    out = -np.ones((n_parse(n), len(pos)), np.int32)
    for (p, h), q in index.items():
        if n not in h:
            out[p, pos[tuple(h)]] = q
    return out


@functools.lru_cache(maxsize=None)
def _perm_code_table(n: int) -> np.ndarray:
    """[n^n] int32: mixed-radix code sum_k p[k] n^k of a permutation -> its class, -1 otherwise."""
    table = np.full(n ** n, -1, np.int32)
    for idx, p in enumerate(permutation_elements(n, "S")):
        table[sum(int(p[k]) * n ** k for k in range(n))] = idx
    return table


def parcel_ints(rng: np.random.Generator, batch_size: int, n: int, seq_len: int):
    """(tokens, states), both [B, T] int32: a 5N-token preamble with a uniform assignment, then
    swap sentences scanned in S_N from that assignment."""
    T, B = int(seq_len), int(batch_size)
    pre_len = SENTENCE_LEN * n
    ids, dfa = token_ids(n), machine(n)[2]
    gift = rng.permuted(np.tile(np.arange(n), (B, 1)), axis=1)          # person k gets gift[k]

    pre_tok = np.empty((B, pre_len), np.int32)
    for k in range(n):
        pre_tok[:, k * SENTENCE_LEN:(k + 1) * SENTENCE_LEN] = np.stack(
            [np.full(B, k), np.full(B, ids["has"]), np.full(B, ids["the"]), n + gift[:, k],
             np.full(B, ids["."])], axis=-1)
    pre_st = np.empty((B, pre_len), np.int32)
    q = np.zeros(B, np.int64)
    for t in range(pre_len):
        q = dfa[q, pre_tok[:, t]]
        pre_st[:, t] = q

    h0 = np.empty((B, n), np.int64)                                     # holder of each item
    np.put_along_axis(h0, gift, np.tile(np.arange(n), (B, 1)), axis=1)
    g0 = _perm_code_table(n)[np.sum(h0 * n ** np.arange(n), axis=-1)]

    n_sent = -(-(T - pre_len) // SENTENCE_LEN)
    i = rng.integers(0, n, (B, n_sent))
    j = (i + rng.integers(1, n, (B, n_sent))) % n
    g = scan_cayley(permutation_cayley(n, "S"),
                    np.concatenate([g0[:, None], transposition_index(n)[i, j]], axis=1))
    g_prev, g_now = g[:, :-1], g[:, 1:]
    hold, zero = 1 + i, np.zeros_like(i)
    parse = np.stack([hold, hold, hold, zero, zero], axis=-1)
    perm = np.stack([g_prev, g_prev, g_prev, g_now, g_now], axis=-1)
    swap_st = _full_index(n)[parse, perm].reshape(B, -1)
    full = lambda w: np.full(i.shape, ids[w])                          # noqa: E731
    swap_tok = np.stack([i, full("swaps"), full("with"), j, full(".")], axis=-1).reshape(B, -1)

    x = np.concatenate([pre_tok, swap_tok], axis=1)[:, :T]
    y = np.concatenate([pre_st, swap_st], axis=1)[:, :T]
    return x.astype(np.int32), y.astype(np.int32)


def generate(key, batch_size: int, n: int, seq_len: int):
    """(token ids [B, T], one-hot states [B, T, n_states]); T must exceed the preamble."""
    n = check_n(n)
    if int(seq_len) < SENTENCE_LEN * (n + 1):
        raise ValueError(f"seq_len {seq_len} leaves no swap sentence after the preamble")
    x, y = parcel_ints(host_rng(key), batch_size, n, seq_len)
    return to_device(x, y, n_states(n))


__all__ = ["NAMES", "ITEMS", "alphabet", "n_states", "n_answers", "head_sizes", "head_codes",
           "generate"]
