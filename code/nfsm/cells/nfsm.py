"""The NFSM cell: K finite-state heads with an argmax transition table.

Head k has n_k states. From the cell input x_t, `trans_proj` produces logits theta_t[k, l, j] for
moving head k from state j to state l. The realised transition is tau_t(j) = argmax_l theta_t[k, l, j],
an index map, so the state sequence is a prefix scan of index maps. Each head reads out the value
vector W[h_k]; the K vectors are concatenated and mapped back to m by `out_proj`.

Training draws one standard Gumbel per (destination, source) pair on the sampling logits
u = theta / nu. The backward pass is the straight-through surrogate on the realised column
j = h_{t-1}, with weights sigma = softmax(u[:, j] + G[:, j]). The adjoint lam_t = gbar_t +
lam_{t+1}[tau_{t+1}] is itself an associative scan. Evaluation (`deterministic=True`) is the
noise-free argmax.
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

IDX = jnp.int32
SCAN_IMPLS = ("sequential", "flat")
GUMBEL_STD = float(math.pi / math.sqrt(6.0))   # std of a standard Gumbel
THETA_BUDGET = 1 << 26                         # theta elements held at once (~256 MB float32)


class _Spec(NamedTuple):
    """Static layout of the heads, grouped by size (ascending)."""
    sizes: tuple        # distinct head sizes n
    counts: tuple       # heads per size
    state_off: tuple    # start of each group in the flat state space
    theta_off: tuple    # start of each group in trans_proj's output
    base_pos: tuple     # flat index of every head's state 0, internal order
    theta_dim: int      # sum_k n_k^2
    budget: int         # theta elements held at once


# --- helpers -------------------------------------------------------------------------------------

def _chunk_len(T, per_step, budget):
    """Largest power-of-two divisor of T with C * per_step <= budget; T if the whole table fits."""
    limit = max(1, min(T, budget // max(per_step, 1)))
    C = 1
    while C * 2 <= limit and T % (C * 2) == 0:
        C *= 2
    if C > 1:
        return C
    return T if limit >= T else 1


def _to_chunks(a, C):
    """[B, T, ...] -> [T // C, B, C, ...]."""
    return a.reshape(a.shape[0], a.shape[1] // C, C, *a.shape[2:]).swapaxes(0, 1)


def _from_chunks(a):
    """[nc, B, C, ...] -> [B, nc * C, ...]."""
    nc, B, C = a.shape[:3]
    return a.swapaxes(0, 1).reshape(B, nc * C, *a.shape[3:])


def _split_feats(feats, spec, inv_nu):
    """trans_proj output [..., theta_dim] -> per group, sampling logits u [..., k, n(l), n(j)]."""
    lead = feats.shape[:-1]
    return [feats[..., o:o + k * n * n].reshape(*lead, k, n, n) * inv_nu
            for n, k, o in zip(spec.sizes, spec.counts, spec.theta_off)]


def _argmax_next(u, g):
    """[..., n(l), n(j)] logits and Gumbels -> next state of every source j: [..., n(j)]."""
    return jnp.argmax(u + g, axis=-2).astype(IDX)


def _realised_column(a, j):
    """a [..., n(l), n(j)] read at source columns j [...] -> [..., n(l)]."""
    idx = jnp.broadcast_to(j[..., None, None], a.shape[:-1] + (1,))
    return jnp.take_along_axis(a, idx, axis=-1)[..., 0]


def _column_spread(v):
    """Standard deviation over the last axis, differentiable at a flat column."""
    c = v - jnp.mean(v, axis=-1, keepdims=True)
    return jnp.sqrt(jnp.mean(c * c, axis=-1) + 1e-12)


def _unpack(sl, spec):
    """One scan slice -> (leading arrays, per-group Gumbels)."""
    ng = len(spec.sizes)
    return sl[:-ng], sl[-ng:]


def _scan_xs(spec, gumbels, C, *leading):
    """The chunked xs of a lax.scan over time: leading [B, T, ...] arrays, then the Gumbels."""
    return tuple(_to_chunks(a, C) for a in leading) + tuple(_to_chunks(g, C) for g in gumbels)


def _tau_chunk(us, gs, spec):
    """Per-group logits and Gumbels -> the flat index map [B, C, S]."""
    cols = []
    for gi, (n, k, so) in enumerate(zip(spec.sizes, spec.counts, spec.state_off)):
        tau = _argmax_next(us[gi], gs[gi])
        head_base = so + jnp.arange(k, dtype=IDX) * n
        cols.append((head_base[None, None, :, None] + tau).reshape(*tau.shape[:2], k * n))
    return jnp.concatenate(cols, axis=-1)


def _forward_map(x, kernel, bias, gumbels, inv_nu, spec, C):
    """x [B, T, m] -> the index maps TAU [B, T, S], C timesteps of theta at a time."""
    def body(_, sl):
        (x_c,), gs_c = _unpack(sl, spec)
        return None, _tau_chunk(_split_feats(x_c @ kernel + bias, spec, inv_nu), gs_c, spec)

    _, TAU = jax.lax.scan(body, None, _scan_xs(spec, gumbels, C, x))
    return _from_chunks(TAU)


def _states(TAU, base_pos, kind):
    """Compose TAU [B, T, S] from every head's state 0: h_seq [B, T, K] flat state indices."""
    B, T, _ = TAU.shape
    K = len(base_pos)
    h0 = jnp.broadcast_to(jnp.asarray(base_pos, IDX)[None, :], (B, K))
    if kind == "flat":
        H = jax.lax.associative_scan(lambda a, b: jnp.take_along_axis(b, a, axis=-1), TAU, axis=1)
        return jnp.take_along_axis(H, jnp.broadcast_to(h0[:, None], (B, T, K)), axis=-1)

    def step(h, tau_t):
        h = jnp.take_along_axis(tau_t, h, axis=-1)
        return h, h

    _, h_seq = jax.lax.scan(step, h0, TAU.swapaxes(0, 1))
    return h_seq.swapaxes(0, 1)


def _path(x, kernel, bias, gumbels, inv_nu, spec, kind, C):
    """(TAU [B, T, S], h_seq [B, T, K], h_prev [B, T, K]): maps, states after and before each step."""
    TAU = _forward_map(x, kernel, bias, gumbels, inv_nu, spec, C)
    h_seq = _states(TAU, spec.base_pos, kind)
    h0 = jnp.broadcast_to(jnp.asarray(spec.base_pos, IDX)[None, None, :], (x.shape[0], 1, len(spec.base_pos)))
    return TAU, h_seq, jnp.concatenate([h0, h_seq[:, :-1]], axis=1)


# --- backward ------------------------------------------------------------------------------------

def _readout_bwd(h_seq, g_head, W, spec):
    """Read-out cotangents: (gbar [B, T, S] = <g_head[k(s)], W[s]>, grad_W [S, d_head])."""
    dh = W.shape[-1]
    gbars, grad_Ws, ho = [], [], 0
    for n, k, so in zip(spec.sizes, spec.counts, spec.state_off):
        Wg = W[so:so + k * n].reshape(k, n, dh)
        gg = g_head[:, :, ho:ho + k, :]
        gbars.append(jnp.einsum('btkd,knd->btkn', gg, Wg).reshape(*gg.shape[:2], k * n))
        head_base = so + jnp.arange(k, dtype=IDX) * n
        occ = jax.nn.one_hot(h_seq[:, :, ho:ho + k] - head_base[None, None, :], n, dtype=gg.dtype)
        grad_Ws.append(jnp.einsum('btkn,btkd->knd', occ, gg).reshape(k * n, dh))
        ho += k
    return jnp.concatenate(gbars, axis=-1), jnp.concatenate(grad_Ws, axis=0)


def _lam_seq_idx(tau, gbar):
    """Adjoint lam_t = gbar_t + lam_{t+1}[tau_{t+1}] by a reverse sequential scan. [B, T, S]."""
    n = tau.shape[-1]
    ident = jnp.broadcast_to(jnp.arange(n, dtype=tau.dtype), tau[:, :1].shape)
    tau_next = jnp.concatenate([tau[:, 1:], ident], axis=1)

    def body(lam_next, inp):
        tn, gb = inp
        lam_t = gb + jnp.take_along_axis(lam_next, tn, axis=-1)
        return lam_t, lam_t

    _, lam = jax.lax.scan(body, jnp.zeros_like(gbar[:, 0]),
                          (tau_next.swapaxes(0, 1), gbar.swapaxes(0, 1)), reverse=True)
    return lam.swapaxes(0, 1)


def _lam_assoc_idx(tau, gbar):
    """The same adjoint by a reverse associative scan of the affine maps v -> v[tau] + gbar."""
    n = tau.shape[-1]
    tau_rev = tau[:, ::-1]
    ident = jnp.broadcast_to(jnp.arange(n, dtype=tau.dtype), tau[:, :1].shape)
    A_rev = jnp.concatenate([ident, tau_rev[:, :-1]], axis=1)
    b_rev = gbar[:, ::-1]

    def compose(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return jnp.take_along_axis(a1, a2, axis=-1), jnp.take_along_axis(b1, a2, axis=-1) + b2

    _, lam_rev = jax.lax.associative_scan(compose, (A_rev, b_rev), axis=1)
    return lam_rev[:, ::-1]


def _grad_chunk(x_c, kernel, bias, gs_c, lam_c, hp_c, inv_nu, spec):
    """Surrogate cotangent of trans_proj's output for C steps: [B, C, theta_dim].

    Per head, only the realised column j = h_{t-1} gets sigma * (lam - <sigma, lam>), with
    sigma = softmax(u[:, j] + G[:, j]); the factor inv_nu is the chain rule of u = theta / nu."""
    us = _split_feats(x_c @ kernel + bias, spec, inv_nu)
    grads, ho = [], 0
    for gi, (n, k, so) in enumerate(zip(spec.sizes, spec.counts, spec.state_off)):
        head_base = so + jnp.arange(k, dtype=IDX) * n
        lam = lam_c[..., so:so + k * n].reshape(*lam_c.shape[:2], k, n)
        hp = hp_c[..., ho:ho + k] - head_base[None, None, :]
        sigma = jax.nn.softmax(_realised_column(us[gi], hp) + _realised_column(gs_c[gi], hp), axis=-1)
        a = sigma * (lam - jnp.sum(sigma * lam, axis=-1, keepdims=True))
        g = a[..., :, None] * jax.nn.one_hot(hp, n, dtype=lam.dtype)[..., None, :]
        grads.append((g * inv_nu).reshape(*g.shape[:2], k * n * n))
        ho += k
    return jnp.concatenate(grads, axis=-1)


def _backward_proj(x, kernel, bias, gumbels, lam, h_prev, inv_nu, spec, C):
    """Chunked VJP of trans_proj: (grad_x, grad_kernel, grad_bias)."""
    def body(carry, sl):
        gk, gb = carry
        (x_c, lam_c, hp_c), gs_c = _unpack(sl, spec)
        gf = _grad_chunk(x_c, kernel, bias, gs_c, lam_c, hp_c, inv_nu, spec)
        return (gk + jnp.einsum('bcm,bcf->mf', x_c, gf), gb + jnp.sum(gf, axis=(0, 1))), gf @ kernel.T

    (gk, gb), gx = jax.lax.scan(body, (jnp.zeros_like(kernel), jnp.zeros_like(bias)),
                                _scan_xs(spec, gumbels, C, x, lam, h_prev))
    return _from_chunks(gx), gk, gb


def _run_states(x, kernel, bias, gumbels, inv_nu, W, spec, kind):
    """Forward: (o [B, T, K * d_head] = concat_k W[h_k], residuals for the backward)."""
    B, T = x.shape[0], x.shape[1]
    C = _chunk_len(T, spec.theta_dim * B, spec.budget)
    TAU, h_seq, h_prev = _path(x, kernel, bias, gumbels, inv_nu, spec, kind, C)
    o = W[h_seq].reshape(B, T, -1)
    return o, (x, kernel, bias, TAU, h_seq, h_prev, W, gumbels, inv_nu, C)


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7))
def _run(x, kernel, bias, gumbels, inv_nu, W, spec, kind):
    return _run_states(x, kernel, bias, gumbels, inv_nu, W, spec, kind)[0]


def _run_fwd(x, kernel, bias, gumbels, inv_nu, W, spec, kind):
    return _run_states(x, kernel, bias, gumbels, inv_nu, W, spec, kind)


def _run_bwd(spec, kind, res, g_o):
    x, kernel, bias, TAU, h_seq, h_prev, W, gumbels, inv_nu, C = res
    g_head = g_o.reshape(*g_o.shape[:2], h_seq.shape[-1], -1)
    gbar, grad_W = _readout_bwd(h_seq, g_head, W, spec)
    lam = (_lam_assoc_idx if kind == "flat" else _lam_seq_idx)(TAU, gbar)
    grad_x, grad_kernel, grad_bias = _backward_proj(x, kernel, bias, gumbels, lam, h_prev,
                                                    inv_nu, spec, C)
    return (grad_x, grad_kernel, grad_bias, tuple(jnp.zeros_like(g) for g in gumbels),
            jnp.zeros_like(inv_nu), grad_W)


_run.defvjp(_run_fwd, _run_bwd)


def _excess_chunk(x_c, kernel, bias, hp_c, inv_nu, spec):
    """relu(spread(u[:, j]) / GUMBEL_STD - 1)^2 on the realised columns, mean over heads: [B, C]."""
    us = _split_feats(x_c @ kernel + bias, spec, inv_nu)
    total, ho = 0.0, 0
    for gi, (n, k, so) in enumerate(zip(spec.sizes, spec.counts, spec.state_off)):
        hp = hp_c[..., ho:ho + k] - (so + jnp.arange(k, dtype=IDX) * n)[None, None, :]
        ratio = _column_spread(_realised_column(us[gi], hp)) / GUMBEL_STD
        total = total + jnp.sum(jax.nn.relu(ratio - 1.0) ** 2, axis=-1)
        ho += k
    return total / ho


# --- the cell ------------------------------------------------------------------------------------

class NFSM(nnx.Module):
    """x [B, T, m] -> out [B, T, m] through K heads of `head_sizes` states."""

    def __init__(self, head_sizes, m: int, rngs: nnx.Rngs, d_head: int = None,
                 noise_scale: float = 0.5, scan_impl: str = "flat",
                 theta_budget: int = THETA_BUDGET) -> None:
        """head_sizes: states per head. d_head: value-vector width (default m // K).
        noise_scale: Gumbel sampling temperature nu used in training. scan_impl: flat | sequential."""
        assert scan_impl in SCAN_IMPLS, f"scan_impl must be one of {SCAN_IMPLS}"
        head_sizes = tuple(int(s) for s in head_sizes)
        assert head_sizes and all(s >= 1 for s in head_sizes)
        self.head_sizes = head_sizes
        self.S = sum(head_sizes)
        self.K = len(head_sizes)
        self.m = m
        self.d_head = int(d_head) if d_head is not None else max(1, m // self.K)
        self.noise_scale = float(noise_scale)
        self.scan_impl = scan_impl
        self.theta_budget = theta_budget

        self._sizes = tuple(sorted(set(head_sizes)))
        self._counts = tuple(head_sizes.count(s) for s in self._sizes)
        theta_off, state_off, base_pos, to, so = [], [], [], 0, 0
        for n, k in zip(self._sizes, self._counts):
            theta_off.append(to)
            state_off.append(so)
            base_pos.extend(so + i * n for i in range(k))
            to += k * n * n
            so += k * n
        self._spec = _Spec(sizes=self._sizes, counts=self._counts, state_off=tuple(state_off),
                           theta_off=tuple(theta_off), base_pos=tuple(base_pos), theta_dim=to,
                           budget=theta_budget)

        self.trans_proj = nnx.Linear(m, to, rngs=rngs)
        self.W = nnx.Param(jax.random.normal(rngs.params(), (self.S, self.d_head)))
        self.out_proj = nnx.Linear(self.K * self.d_head, m, rngs=rngs)
        self.rngs = rngs

    def _gumbel(self, B, T, k, n, deterministic, dtype):
        """Standard Gumbels [B, T, k, n, n], one per (destination, source); zeros if deterministic."""
        if deterministic:
            return jnp.zeros((B, T, k, n, n), dtype)
        return jax.random.gumbel(self.rngs.noise(), (B, T, k, n, n), dtype=dtype)

    def __call__(self, x: Array, deterministic: bool = False, scan_impl: str = None,
                 return_explore: bool = False):
        """x [B, T, m] -> out [B, T, m], or (out, excess [B, T]) with return_explore."""
        impl = scan_impl or self.scan_impl
        assert impl in SCAN_IMPLS, f"scan_impl must be one of {SCAN_IMPLS}, got {impl!r}"
        B, T = x.shape[0], x.shape[1]
        dtype = jnp.result_type(x.dtype, self.trans_proj.kernel.value.dtype)
        gumbels = tuple(self._gumbel(B, T, k, n, deterministic, dtype)
                        for n, k in zip(self._sizes, self._counts))
        inv_nu = jnp.asarray(1.0 if deterministic else 1.0 / self.noise_scale, dtype)
        out = self.out_proj(_run(x, self.trans_proj.kernel.value, self.trans_proj.bias.value,
                                 gumbels, inv_nu, self.W.value, self._spec, impl))
        if return_explore:
            return out, self.exploration_excess(x, gumbels, inv_nu, impl)
        return out

    def exploration_excess(self, x, gumbels, inv_nu, kind):
        """relu(spread / GUMBEL_STD - 1)^2 of each realised column, mean over heads: [B, T].

        Zero while a column is no sharper than the sampling noise; carries gradient to trans_proj."""
        T = x.shape[1]
        C = _chunk_len(T, self._spec.theta_dim * max(1, x.shape[0]), self.theta_budget)
        kernel, bias = self.trans_proj.kernel.value, self.trans_proj.bias.value
        sg = jax.lax.stop_gradient
        _, _, h_prev = _path(sg(x), sg(kernel), sg(bias), gumbels, inv_nu, self._spec, kind, C)

        @jax.checkpoint
        def body(_, sl):
            x_c, hp_c = sl
            return None, _excess_chunk(x_c, kernel, bias, hp_c, inv_nu, self._spec)

        _, per_chunk = jax.lax.scan(body, None, (_to_chunks(x, C), _to_chunks(h_prev, C)))
        return _from_chunks(per_chunk)

    # --- head bookkeeping ---

    def head_layout(self):
        """(sizes, offsets, order) per internal head t: its size, first flat state, and index in
        head_sizes. Internal order is head_sizes stably sorted by size."""
        order = tuple(sorted(range(self.K), key=lambda i: self.head_sizes[i]))
        return tuple(self.head_sizes[i] for i in order), tuple(self._spec.base_pos), order

    def head_channel_slices(self) -> tuple:
        """(start, n) per head in head_sizes order: its output channels under `pin_readout`."""
        sizes, _, order = self.head_layout()
        starts = np.cumsum((0,) + sizes[:-1])
        out = [None] * self.K
        for t, i in enumerate(order):
            out[i] = (int(starts[t]), int(sizes[t]))
        return tuple(out)

    def pin_readout(self, codes, scale: float, require_injective: bool = True) -> tuple:
        """Write W and out_proj so that out[start_k + j] = scale * 1[h_k == j]; return the slices.

        codes: one [C] array per head (head_sizes order), code_k[c] = head k's state for class c;
        validated (range, and joint injectivity unless require_injective=False) before writing."""
        sizes, offsets, _ = self.head_layout()
        if len(codes) != self.K:
            raise ValueError(f"got {len(codes)} codes for {self.K} heads")
        codes = [np.asarray(c, np.int64).reshape(-1) for c in codes]
        if sum(sizes) > self.m or max(sizes) > self.d_head:
            raise ValueError(f"pin needs m >= sum(head_sizes) and d_head >= max(head_sizes): "
                             f"m={self.m}, d_head={self.d_head}, head_sizes={self.head_sizes}")
        for i, c in enumerate(codes):
            if c.shape != codes[0].shape or c.min() < 0 or c.max() >= self.head_sizes[i]:
                raise ValueError(f"code of head {i} is not a map into its {self.head_sizes[i]} states")
        if require_injective and np.unique(np.stack(codes, 1), axis=0).shape[0] != codes[0].shape[0]:
            raise ValueError("codes are not jointly injective")
        W = np.zeros((self.S, self.d_head), np.float32)
        kernel = np.zeros((self.K * self.d_head, self.m), np.float32)
        start = 0
        for t, (n, off) in enumerate(zip(sizes, offsets)):
            W[off:off + n, :n] = np.eye(n, dtype=np.float32)
            kernel[t * self.d_head:t * self.d_head + n, start:start + n] = scale * np.eye(n, dtype=np.float32)
            start += n
        self.W.value = jnp.asarray(W)
        self.out_proj.kernel.value = jnp.asarray(kernel)
        self.out_proj.bias.value = jnp.zeros((self.m,), jnp.float32)
        return self.head_channel_slices()

    def transition_maps(self, x: Array) -> Array:
        """Noise-free maps j -> argmax_l theta[l, j] of every head at every position:
        [B, T, K, max(head_sizes)] in head_sizes order, padded with -1."""
        us = _split_feats(self.trans_proj(x), self._spec, 1.0)
        n_max = max(self._sizes)
        parts = [jnp.pad(jnp.argmax(u, axis=-2), ((0, 0), (0, 0), (0, 0), (0, n_max - n)),
                         constant_values=-1) for u, n in zip(us, self._sizes)]
        maps = jnp.concatenate(parts, axis=2)
        _, _, order = self.head_layout()
        slot = {i: t for t, i in enumerate(order)}
        return maps[:, :, [slot[i] for i in range(self.K)], :]

    def states(self, x: Array, scan_impl: str = None) -> Array:
        """Noise-free head states after each step: [B, T, K] head-local, in head_sizes order."""
        impl = scan_impl or self.scan_impl
        B, T = x.shape[0], x.shape[1]
        dtype = jnp.result_type(x.dtype, self.trans_proj.kernel.value.dtype)
        gumbels = tuple(self._gumbel(B, T, k, n, True, dtype)
                        for n, k in zip(self._sizes, self._counts))
        C = _chunk_len(T, self._spec.theta_dim * max(1, B), self.theta_budget)
        _, h_seq, _ = _path(x, self.trans_proj.kernel.value, self.trans_proj.bias.value, gumbels,
                            jnp.asarray(1.0, dtype), self._spec, impl, C)
        _, offsets, order = self.head_layout()
        slot = {i: t for t, i in enumerate(order)}
        return jnp.stack([h_seq[..., slot[i]] - offsets[slot[i]] for i in range(self.K)], axis=-1)

    def readout(self, states: Array) -> Array:
        """Cell output for head-local states [..., K] (head_sizes order): [..., m]."""
        _, offsets, order = self.head_layout()
        flat = jnp.stack([offsets[t] + states[..., i] for t, i in enumerate(order)], axis=-1)
        return self.out_proj(self.W.value[flat].reshape(*flat.shape[:-1], self.K * self.d_head))
