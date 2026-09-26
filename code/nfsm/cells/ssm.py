"""Recurrent baselines, each mapping x [B, T, m] -> [B, T, m].

  Mamba  the Mamba block: in_proj -> SiLU -> selective diagonal SSM with a_t = 2 exp(dt_t A) - 1 in
         (-1, 1] (Grazzi et al. 2024) -> + D u -> x SiLU(z) -> out_proj. The reference block's own
         causal conv (width 4) is left out on purpose: the backbone already runs a causal conv
         (ModelConfig.conv_kernel) before every cell, so Mamba defaults to d_conv = 0.
  AUSSM  the same block around an adaptive unitary SSM (arXiv 2507.05238, eq. 8 and its reference
         code): h_t = exp(i theta_t) h_{t-1} + dt_t B u_t, y_t = Re(C h_t), theta per channel and
         state a linear map of u (bias uniform in (-pi, pi)), B and C fixed complex vectors. As for
         Mamba, the reference block's width-4 conv is left out (d_conv = 0): the backbone's causal
         conv (ModelConfig.conv_kernel) runs before every cell.
  PDSSM  h_t = P_t D_t h_{t-1} + b_t, P_t column one-hot, D_t complex diagonal (arXiv 2509.22284).

`parallel_mode` selects the associative scan (True) or the sequential reference (False). AUSSM and
PD-SSM are complex-valued and have no float16 form."""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from nfsm.cells.conv import ShortConv
from nfsm.cells.nfsm import _chunk_len, _from_chunks, _to_chunks

TWO_PI = 2.0 * math.pi
DICT_BUDGET = 1 << 26    # PD-SSM mixture elements [B, C, N, N] held at once
STATE_BUDGET = 1 << 25   # selective-SSM state elements [B, C, d_inner, d_state] held at once


def _affine_parallel_scan(a: Array, b: Array) -> Array:
    """h_t = a_t h_{t-1} + b_t from h_0 = 0 along axis 1; a, b [B, T, ...] -> [B, T, ...]."""
    def op(left, right):
        return left[0] * right[0], right[0] * left[1] + right[1]
    return jax.lax.associative_scan(op, (a, b), axis=1)[1]


def _affine_sequential_scan(a: Array, b: Array, h0: Array = None) -> Array:
    """Sequential form of `_affine_parallel_scan`, from h0 [B, ...] (zeros by default)."""
    if h0 is None:
        h0 = jnp.zeros(a.shape[:1] + a.shape[2:], jnp.result_type(a, b))

    def step(h_prev, inp):
        a_t, b_t = inp
        h = a_t * h_prev + b_t
        return h, h

    _, h = jax.lax.scan(step, h0, (jnp.moveaxis(a, 1, 0), jnp.moveaxis(b, 1, 0)))
    return jnp.moveaxis(h, 0, 1)


def _scan_with_carry(a: Array, bus: Array, h0: Array) -> Array:
    """h_t = a_t h_{t-1} + bus_t from the carry h0 [B, ...]: the h0 = 0 solution plus (prod a) h0."""
    def op(left, right):
        return left[0] * right[0], right[0] * left[1] + right[1]
    cum_a, h = jax.lax.associative_scan(op, (a, bus), axis=1)
    return h + cum_a * h0[:, None]


def _scan_chunk_len(B: int, T: int, per_step: int, budget: int) -> int:
    """Largest power of two C with B * C * per_step <= budget, or T if the whole scan fits."""
    limit = max(1, budget // max(1, B * per_step))
    if limit >= T:
        return T
    C = 1
    while C * 2 <= limit:
        C *= 2
    return C


def _inv_softplus(y: Array) -> Array:
    """x with softplus(x) = y, y > 0."""
    return y + jnp.log(-jnp.expm1(-y))


# --- selective SSMs ------------------------------------------------------------------------------

class _SelectiveSSM(nnx.Module):
    """in_proj -> causal conv of width d_conv (none if 0) + SiLU -> diagonal recurrence
    h [B, T, d_inner, d_state] -> read-out + D u -> x SiLU(z) -> out_proj. Subclasses supply
    `_drive`: (a_t, input drive, read-out C)."""

    def __init__(self, d: int, m: int, rngs: nnx.Rngs, *, d_state: int = 16, d_conv: int = 4) -> None:
        """d: d_inner (recurrent width); the state is d_inner x d_state."""
        self.d = self.d_inner = d
        self.m = m
        self.d_state = d_state
        self.in_proj = nnx.Linear(m, 2 * d, use_bias=False, rngs=rngs)
        self.conv = ShortConv(d, d_conv, rngs) if d_conv > 0 else None     # ShortConv ends in SiLU
        self.D_res = nnx.Param(jnp.ones((d,)))
        self.out_proj = nnx.Linear(d, m, use_bias=False, rngs=rngs)

    def _drive(self, u: Array):
        """u [..., d_inner] -> (a [..., d_inner, d_state], drive [..., d_inner, d_state], C)."""
        raise NotImplementedError

    def _state_dtype(self):
        return jnp.float32

    def _project_out(self, y: Array, u: Array, z: Array) -> Array:
        """y [..., d_inner] -> skip, gate, project to m."""
        return self.out_proj((y + self.D_res.value * u) * jax.nn.silu(z))

    def _chunked_state_read(self, u: Array, chunk: int, parallel_mode: bool = True) -> Array:
        """The state pipeline `chunk` steps at a time (the state never exists at full length);
        within a chunk, the associative scan or the sequential loop."""
        B, T = u.shape[:2]
        pad = (-T) % chunk
        if pad:
            u = jnp.concatenate([u, jnp.zeros((B, pad, u.shape[2]), u.dtype)], axis=1)

        @jax.checkpoint
        def body(h_prev, u_c):
            a, bus, C = self._drive(u_c)
            h = (_scan_with_carry(a, bus, h_prev) if parallel_mode
                 else _affine_sequential_scan(a, bus, h_prev))
            return h[:, -1], jnp.real(jnp.sum(h * C[..., None, :], axis=-1))

        h0 = jnp.zeros((B, self.d_inner, self.d_state), self._state_dtype())
        _, y = jax.lax.scan(body, h0, _to_chunks(u, chunk))
        return _from_chunks(y)[:, :T]

    def __call__(self, x: Array, parallel_mode: bool = True) -> Array:
        u, z = jnp.split(self.in_proj(x), 2, axis=-1)
        u = self.conv(u) if self.conv is not None else jax.nn.silu(u)
        B, T = u.shape[:2]
        chunk = _scan_chunk_len(B, T, self.d_inner * self.d_state, STATE_BUDGET)
        if chunk < T:
            return self._project_out(self._chunked_state_read(u, chunk, parallel_mode), u, z)
        a, bus, C = self._drive(u)
        scan = _affine_parallel_scan if parallel_mode else _affine_sequential_scan
        y = jnp.real(jnp.sum(scan(a, bus) * C[..., None, :], axis=-1))
        return self._project_out(y, u, z)


class Mamba(_SelectiveSSM):
    """Selective SSM with a = 2 exp(dt A) - 1 in (-1, 1], A = -exp(A_log), A_log = log(1..d_state);
    dt, B and C from the input as in Mamba. No conv inside the block (d_conv = 0): the backbone's
    causal conv before the cell stands in for the reference block's width-4 conv."""

    def __init__(self, d: int, m: int, rngs: nnx.Rngs, *, d_state: int = 16, d_conv: int = 0,
                 dt_rank: int = None, dt_min: float = 1e-3, dt_max: float = 0.1,
                 dt_init_floor: float = 1e-4) -> None:
        super().__init__(d, m, rngs, d_state=d_state, d_conv=d_conv)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(m / 16)
        self.x_proj = nnx.Linear(d, self.dt_rank + 2 * d_state, use_bias=False, rngs=rngs)
        self.dt_proj = nnx.Linear(self.dt_rank, d, use_bias=False, rngs=rngs)
        dt = jnp.exp(jax.random.uniform(rngs.next(), (d,))
                     * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nnx.Param(_inv_softplus(jnp.maximum(dt, dt_init_floor)))
        self.A_log = nnx.Param(
            jnp.broadcast_to(jnp.log(jnp.arange(1, d_state + 1, dtype=jnp.float32)), (d, d_state)).copy())

    def _drive(self, u: Array):
        dt_r, Bt, Ct = jnp.split(self.x_proj(u), (self.dt_rank, self.dt_rank + self.d_state), axis=-1)
        dt = jax.nn.softplus(self.dt_proj(dt_r) + self.dt_bias.value)
        a = 2.0 * jnp.exp(dt[..., :, None] * -jnp.exp(self.A_log.value)) - 1.0
        return a, dt[..., :, None] * Bt[..., None, :] * u[..., :, None], Ct


class AUSSM(_SelectiveSSM):
    """Unit-modulus diagonal a_ij = exp(i theta_ij(u)), theta = xA_proj(u) per channel i and state j;
    drive dt_i B_j u_i and read-out Re(sum_j C_j h_ij) with B, C fixed complex vectors. No conv
    inside the block (d_conv = 0): the backbone's causal conv before the cell stands in for it."""

    def __init__(self, d: int, m: int, rngs: nnx.Rngs, *, d_state: int = 16, d_conv: int = 0,
                 dt_rank: int = None) -> None:
        super().__init__(d, m, rngs, d_state=d_state, d_conv=d_conv)
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d / 16)
        self.x_proj = nnx.Linear(d, self.dt_rank, use_bias=False, rngs=rngs)
        self.dt_proj = nnx.Linear(self.dt_rank, d, rngs=rngs)
        self.xA_proj = nnx.Linear(d, d * d_state, rngs=rngs)
        self.xA_proj.bias.value = jax.random.uniform(rngs.next(), (d * d_state,),
                                                     minval=-math.pi, maxval=math.pi)
        half = math.sqrt(0.5)                   # a standard complex normal, as torch.randn(complex64)
        self.B_re, self.B_im, self.C_re, self.C_im = (
            nnx.Param(jax.random.normal(rngs.next(), (d_state,)) * half) for _ in range(4))

    def _drive(self, u: Array):
        dt = jax.nn.softplus(self.dt_proj(self.x_proj(u)))
        theta = self.xA_proj(u).reshape(*u.shape[:-1], self.d_inner, self.d_state)
        a = jax.lax.complex(jnp.cos(theta), jnp.sin(theta))
        Bc = jax.lax.complex(self.B_re.value, self.B_im.value)
        bus = (dt * u)[..., :, None].astype(a.dtype) * Bc
        return a, bus, jax.lax.complex(self.C_re.value, self.C_im.value)

    def _state_dtype(self):
        return jnp.complex64


# --- PD-SSM --------------------------------------------------------------------------------------
# A_t = P_t D_t is stored as (idx [..., N] int32, diag [..., N] complex): column j sends state j to
# row idx[j] with weight diag[j]. The class is closed under products, which cost O(N).

def _scatter_last(idx: Array, vals: Array) -> Array:
    """out[..., i] = sum over {j : idx[..., j] == i} of vals[..., j]; idx, vals [..., N]."""
    n = idx.shape[-1]
    rows = math.prod(idx.shape[:-1])
    offsets = (jnp.arange(rows, dtype=idx.dtype) * n)[:, None]
    flat = (idx.reshape(rows, n) + offsets).reshape(-1)
    return jnp.zeros((rows * n,), vals.dtype).at[flat].add(vals.reshape(-1)).reshape(idx.shape)


def _pd_apply(idx: Array, diag: Array, v: Array) -> Array:
    """(P D) v."""
    return _scatter_last(idx, diag * v)


def _pd_compose(left, right):
    """(A_r, b_r) o (A_l, b_l) = (A_r A_l, A_r b_l + b_r), with the earlier step on the left."""
    idx_l, diag_l, b_l = left
    idx_r, diag_r, b_r = right
    return (jnp.take_along_axis(idx_r, idx_l, axis=-1),
            jnp.take_along_axis(diag_r, idx_l, axis=-1) * diag_l,
            _pd_apply(idx_r, diag_r, b_l) + b_r)


def _pd_parallel_scan(idx: Array, diag: Array, b: Array) -> Array:
    """h_t = P_t D_t h_{t-1} + b_t from h_0 = 0; all [B, T, N]."""
    return jax.lax.associative_scan(_pd_compose, (idx, diag, b), axis=1)[2]


def _pd_sequential_scan(idx: Array, diag: Array, b: Array) -> Array:
    """Sequential form of `_pd_parallel_scan`."""
    def step(h, inp):
        i_t, d_t, b_t = inp
        h = _pd_apply(i_t, d_t, h) + b_t
        return h, h

    _, h = jax.lax.scan(step, jnp.zeros_like(b[:, 0]),
                        tuple(jnp.moveaxis(v, 1, 0) for v in (idx, diag, b)))
    return jnp.moveaxis(h, 0, 1)


class PDSSM(nnx.Module):
    """PD-SSM with state size N = d.

    P_t is the column-wise hardmax of the mixture M_t = sum_k softmax(sel_proj x)_k dict_M[k];
    |D_t| in (0, 1) and its phase come from two GELU MLPs. The dictionary gets its gradient from the
    softmax-Jacobian surrogate of the paper."""

    def __init__(self, d: int, m: int, rngs: nnx.Rngs, *, n_dict: int = 8, mlp_hidden: int = None,
                 r_min: float = 0.9, r_max: float = 0.999, dict_budget: int = DICT_BUDGET) -> None:
        self.d = self.N = d
        self.m = m
        self.K = n_dict
        self.dict_budget = dict_budget
        hidden = mlp_hidden if mlp_hidden is not None else m
        self.sel_proj = nnx.Linear(m, n_dict, rngs=rngs)
        self.dict_M = nnx.Param(jax.random.normal(rngs.next(), (n_dict, d, d)) * (d ** -0.5))
        self.mag_in = nnx.Linear(m, hidden, rngs=rngs)
        self.mag_out = nnx.Linear(hidden, d, rngs=rngs)
        self.pha_in = nnx.Linear(m, hidden, rngs=rngs)
        self.pha_out = nnx.Linear(hidden, d, rngs=rngs)
        u = jax.random.uniform(rngs.next(), (d,), minval=r_min, maxval=r_max)
        self.mag_out.bias.value = jnp.log(u / (1.0 - u))           # |D| in [r_min, r_max] at init
        v = jax.random.uniform(rngs.next(), (d,), minval=1e-3, maxval=1.0 - 1e-3)
        self.pha_out.bias.value = jnp.log(v / (1.0 - v))           # phase uniform at init
        init = nnx.initializers.xavier_uniform()
        self.B_in = nnx.Linear(m, 2 * d, rngs=rngs)
        self.C_out = nnx.Linear(2 * d, m, rngs=rngs)
        self.D_skip = nnx.Param(init(rngs.next(), (m, m)))
        self.out_proj = nnx.Linear(m, m, rngs=rngs)

    def _mixture(self, x: Array) -> Array:
        """x [..., m] -> M [..., N, N]."""
        return jnp.einsum("...k,kij->...ij", jax.nn.softmax(self.sel_proj(x), axis=-1),
                          self.dict_M.value)

    def _diag_drive(self, x: Array):
        """x [..., m] -> (diag [..., N] complex, b [..., N] complex)."""
        mag = jax.nn.sigmoid(self.mag_out(jax.nn.gelu(self.mag_in(x))))
        pha = TWO_PI * jax.nn.sigmoid(self.pha_out(jax.nn.gelu(self.pha_in(x))))
        diag = jax.lax.complex(mag * jnp.cos(pha), mag * jnp.sin(pha))
        br, bi = jnp.split(self.B_in(x), 2, axis=-1)
        return diag, jax.lax.complex(br, bi)

    def _over_time(self, fn, T: int, *arrays):
        """Apply a per-position `fn` to [B, T, ...] arrays in rematerialised time chunks."""
        B = arrays[0].shape[0]
        C = _chunk_len(T, B * self.N * self.N, self.dict_budget)
        if C >= T:
            return fn(*arrays)
        body = jax.checkpoint(lambda _, xs: (None, fn(*xs)))
        _, out = jax.lax.scan(body, None, tuple(_to_chunks(a, C) for a in arrays))
        return _from_chunks(out)

    def _idx_over_time(self, x: Array) -> Array:
        """idx [B, T, N]: the row each column of M_t selects (no gradient)."""
        def fn(xc):
            return jnp.argmax(self._mixture(xc), axis=-2).astype(jnp.int32)
        return jax.lax.stop_gradient(self._over_time(fn, x.shape[1], x))

    def _surrogate_over_time(self, x: Array, idx: Array, diag: Array, b: Array, scan) -> Array:
        """A zero-valued drive term (S - sg S)(D h_{t-1}), S = softmax(M_t) over rows, that hands
        dict_M the surrogate gradient dL/dP ~ lam_t (D_t h_{t-1})^T."""
        sg = jax.lax.stop_gradient
        h_prev = sg(scan(idx, sg(diag), sg(b)))
        h_prev = jnp.concatenate([jnp.zeros_like(h_prev[:, :1]), h_prev[:, :-1]], axis=1)
        v = sg(diag) * h_prev

        def fn(xc, vc):
            dS = jax.nn.softmax(self._mixture(xc), axis=-2)
            dS = dS - sg(dS)
            return jax.lax.complex(jnp.einsum("...ij,...j->...i", dS, vc.real),
                                   jnp.einsum("...ij,...j->...i", dS, vc.imag))

        return self._over_time(fn, x.shape[1], x, v)

    def __call__(self, x: Array, deterministic: bool = False, parallel_mode: bool = True) -> Array:
        """`deterministic` drops the (zero-valued) surrogate term."""
        idx = self._idx_over_time(x)
        diag, b = self._diag_drive(x)
        scan = _pd_parallel_scan if parallel_mode else _pd_sequential_scan
        if not deterministic:
            b = b + self._surrogate_over_time(x, idx, diag, b, scan)
        h = scan(idx, diag, b)
        y = self.C_out(jnp.concatenate([h.real, h.imag], axis=-1)) + x @ self.D_skip.value
        return self.out_proj(y)
