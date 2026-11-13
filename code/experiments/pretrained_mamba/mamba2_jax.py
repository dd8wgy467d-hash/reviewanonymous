"""Mamba-2 forward pass in JAX, float32.

Weights: `state-spaces/mamba2-{size}`, size from MAMBA2_SIZE (130m, 370m or 780m). Tokenizer:
GPT-NeoX-20B. Per layer (ngroups = 1, no MLP):

    zxbcdt = in_proj(rmsnorm(residual))            -> z, xBC, dt
    xBC    = silu(causal depthwise conv1d(xBC))    -> x, B, C
    dt     = softplus(dt + dt_bias),  a = dt * A,  A = -exp(A_log)
    h_t    = exp(a_t) h_{t-1} + dt_t x_t B_t^T,   y_t = h_t C_t + D x_t
    out    = out_proj(rmsnorm(y * silu(z)))

The decay bound replaces a_t by min(a_t, -1/tau) in the transition only; tau = inf is the model as
released.
"""
from __future__ import annotations

import math
import os
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet transfers fail in some sandboxes
if sys.platform == "linux":
    # NVIDIA GPUs run float32 matmuls in TF32 by default (logits off by ~1e-1 against the CPU); full float32
    jax.config.update("jax_default_matmul_precision", "highest")

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"

SIZES = {"130m": (768, 24), "370m": (1024, 48), "780m": (1536, 48)}    # (d_model, n_layer), config.json
SIZE = os.environ.get("MAMBA2_SIZE", "130m")
D_MODEL, N_LAYER = SIZES[SIZE]
RESULTS = HERE / "results" if SIZE == "130m" else HERE / "results" / f"mamba2_{SIZE}"
D_STATE, D_CONV, HEADDIM, EXPAND = 128, 4, 64, 2
D_INNER = EXPAND * D_MODEL            # 1536 for 130m
N_HEADS = D_INNER // HEADDIM          # 24 for 130m
CONV_DIM = D_INNER + 2 * D_STATE      # 1792 for 130m
EPS = 1e-5
CHUNK = 256                           # SSD chunk length
SEGMENT = 1024                        # long inputs are processed in segments of this length


# ----------------------------------------------------------------------------- weights / tokenizer

def load_params() -> dict:
    """Return the weights as a dict of float32 arrays, stacked over layers."""
    CACHE.mkdir(exist_ok=True)
    npz = CACHE / f"mamba2-{SIZE}.npz"
    if not npz.exists():
        import torch
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(f"state-spaces/mamba2-{SIZE}", "pytorch_model.bin")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd = {k: v.float().numpy() for k, v in sd.items()}
        lay = lambda name: np.stack([sd[f"backbone.layers.{i}.{name}"] for i in range(N_LAYER)])
        out = {
            "embedding": sd["backbone.embedding.weight"],          # (V, D)  tied with lm_head
            "norm_f": sd["backbone.norm_f.weight"],
            "norm": lay("norm.weight"),                          # (L, D)
            "in_proj": lay("mixer.in_proj.weight"),               # (L, 2*D_INNER + 2N + H, D)
            "conv_w": lay("mixer.conv1d.weight")[:, :, 0, :],     # (L, CONV_DIM, D_CONV)
            "conv_b": lay("mixer.conv1d.bias"),                   # (L, CONV_DIM)
            "dt_bias": lay("mixer.dt_bias"),                      # (L, H)
            "A_log": lay("mixer.A_log"),                          # (L, H)
            "D": lay("mixer.D"),                                  # (L, H)
            "gnorm": lay("mixer.norm.weight"),                    # (L, D_INNER)
            "out_proj": lay("mixer.out_proj.weight"),             # (L, D, D_INNER)
        }
        np.savez(npz, **out)
    with np.load(npz) as f:
        return {k: jnp.asarray(f[k]) for k in f.files}


def load_tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(hf_hub_download("EleutherAI/gpt-neox-20b", "tokenizer.json"))


# ----------------------------------------------------------------------------- building blocks

def rmsnorm(x, w):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + EPS) * w


def log_decay(dt, A_log, tau):
    """a = dt * A per (token, head), with the decay floor a <= -1/tau."""
    a = dt * (-jnp.exp(A_log))
    return jnp.minimum(a, -1.0 / tau)       # tau = inf gives -0.0: no-op since a < 0


def ssd(x, dt, a, Bm, Cm, h0):
    """Chunked scan of h_t = exp(a_t) h_{t-1} + dt_t x_t B_t^T, y_t = h_t C_t; T a multiple of CHUNK.
    Returns y (B,T,H,P) and the final state.
    """
    b, T, H, P = x.shape
    nc, Q = T // CHUNK, CHUNK
    x = x.reshape(b, nc, Q, H, P)
    dt = dt.reshape(b, nc, Q, H)
    a = a.reshape(b, nc, Q, H)
    Bm = Bm.reshape(b, nc, Q, -1)
    Cm = Cm.reshape(b, nc, Q, -1)

    acs = jnp.cumsum(a, axis=2)                                   # (b,nc,Q,H)
    # within-chunk decay  exp(acs_i - acs_j) for i >= j
    diff = acs[:, :, :, None, :] - acs[:, :, None, :, :]          # (b,nc,Q_i,Q_j,H)
    causal = jnp.tril(jnp.ones((Q, Q), bool))[None, None, :, :, None]
    Lmat = jnp.exp(jnp.where(causal, diff, -jnp.inf))
    CB = jnp.einsum("bcin,bcjn->bcij", Cm, Bm)                     # (b,nc,Q,Q)
    xdt = x * dt[..., None]                                       # (b,nc,Q,H,P)
    y_intra = jnp.einsum("bcijh,bcjhp->bcihp", CB[..., None] * Lmat, xdt)

    # state contributed by each chunk, then carried across chunks
    decay_to_end = jnp.exp(acs[:, :, -1:, :] - acs)               # (b,nc,Q,H)
    S = jnp.einsum("bcjn,bcjhp->bchpn", Bm, xdt * decay_to_end[..., None])
    chunk_decay = jnp.exp(acs[:, :, -1, :])                       # (b,nc,H)

    def step(h, inp):
        s, g = inp
        return g[:, :, None, None] * h + s, h                     # emit the state entering the chunk

    hT, h_in = jax.lax.scan(step, h0, (jnp.moveaxis(S, 1, 0), jnp.moveaxis(chunk_decay, 1, 0)))
    h_in = jnp.moveaxis(h_in, 0, 1)                               # (b,nc,H,P,N)
    y_inter = jnp.einsum("bcin,bchpn->bcihp", Cm, h_in) * jnp.exp(acs)[..., None]
    return (y_intra + y_inter).reshape(b, T, H, P), hT


def mixer(p, u, ssm0, conv0, n_valid, tau):
    """One mixer over a padded segment; padding tokens leave the state unchanged."""
    b, T, _ = u.shape
    zxbcdt = u @ p["in_proj"].T
    z, xBC, dt = jnp.split(zxbcdt, [D_INNER, D_INNER + CONV_DIM], axis=-1)

    full = jnp.concatenate([conv0, xBC], axis=1)                  # (B, 3+T, CONV_DIM)
    conv = sum(full[:, k:k + T, :] * p["conv_w"][:, k] for k in range(D_CONV)) + p["conv_b"]
    conv1 = jax.lax.dynamic_slice_in_dim(full, n_valid, D_CONV - 1, axis=1)
    xBC = jax.nn.silu(conv)
    x, Bm, Cm = jnp.split(xBC, [D_INNER, D_INNER + D_STATE], axis=-1)
    x = x.reshape(b, T, N_HEADS, HEADDIM)

    valid = (jnp.arange(T) < n_valid)[None, :, None]
    dt = jax.nn.softplus(dt + p["dt_bias"])
    a_raw = dt * (-jnp.exp(p["A_log"]))                           # log decay before the floor
    a = jnp.where(valid, log_decay(dt, p["A_log"], tau), 0.0)
    dt = jnp.where(valid, dt, 0.0)

    y, ssm1 = ssd(x, dt, a, Bm, Cm, ssm0)
    y = y + p["D"][:, None] * x
    y = y.reshape(b, T, D_INNER)
    y = rmsnorm(y * jax.nn.silu(z), p["gnorm"])
    return y @ p["out_proj"].T, ssm1, conv1, a_raw


LAYER_KEYS = ("norm", "in_proj", "conv_w", "conv_b", "dt_bias", "A_log", "D", "gnorm", "out_proj")


@partial(jax.jit, static_argnames=("all_logits", "decays"))
def forward_segment(params, tokens, state, n_valid, tau, all_logits=False, decays=False):
    """tokens (B,T), T a multiple of CHUNK -> (logits, new state); with decays=True, (state, a_raw)."""
    ssm, conv = state
    h = params["embedding"][tokens]
    layers = {k: params[k] for k in LAYER_KEYS}

    def body(carry, inp):
        hidden, residual = carry
        p, s0, c0 = inp
        residual = hidden + residual
        out, s1, c1, a_raw = mixer(p, rmsnorm(residual, p["norm"]), s0, c0, n_valid, tau)
        return (out, residual), (s1, c1, a_raw if decays else None)

    (hidden, residual), (ssm, conv, a_raw) = jax.lax.scan(body, (h, jnp.zeros_like(h)), (layers, ssm, conv))
    if decays:
        return (ssm, conv), a_raw
    hf = rmsnorm(hidden + residual, params["norm_f"])
    if all_logits:
        return hf @ params["embedding"].T, (ssm, conv)
    last = jax.lax.dynamic_index_in_dim(hf, n_valid - 1, axis=1, keepdims=False)
    return last @ params["embedding"].T, (ssm, conv)


def init_state(batch: int):
    return (jnp.zeros((N_LAYER, batch, N_HEADS, HEADDIM, D_STATE), jnp.float32),
            jnp.zeros((N_LAYER, batch, D_CONV - 1, CONV_DIM), jnp.float32))


def run(params, tokens, state, tau=math.inf, segment=SEGMENT, all_logits=False):
    """Process tokens (B,T) from `state` in segments; returns (logits of the last token or of all, state)."""
    tokens = np.asarray(tokens)
    b, T = tokens.shape
    tau = jnp.float32(tau)
    outs = []
    for s in range(0, T, segment):
        seg = tokens[:, s:s + segment]
        n = seg.shape[1]
        padded = -(-n // CHUNK) * CHUNK
        seg = np.pad(seg, ((0, 0), (0, padded - n)))
        logits, state = forward_segment(params, jnp.asarray(seg), state, jnp.int32(n), tau, all_logits)
        outs.append(logits[:, :n] if all_logits else logits)
    return (jnp.concatenate(outs, axis=1) if all_logits else outs[-1]), state


@jax.jit
def decode_step(params, token, state, tau):
    """One recurrent step for token (B,). Returns logits (B,V) and the new state."""
    ssm, conv = state
    h = params["embedding"][token]
    layers = {k: params[k] for k in LAYER_KEYS}

    def body(carry, inp):
        hidden, residual = carry
        p, s0, c0 = inp
        residual = hidden + residual
        u = rmsnorm(residual, p["norm"])
        z, xBC, dt = jnp.split(u @ p["in_proj"].T, [D_INNER, D_INNER + CONV_DIM], axis=-1)
        window = jnp.concatenate([c0, xBC[:, None, :]], axis=1)            # (B,4,CONV_DIM)
        xBC = jax.nn.silu(jnp.einsum("bkc,ck->bc", window, p["conv_w"]) + p["conv_b"])
        x, Bm, Cm = jnp.split(xBC, [D_INNER, D_INNER + D_STATE], axis=-1)
        x = x.reshape(-1, N_HEADS, HEADDIM)
        dt = jax.nn.softplus(dt + p["dt_bias"])
        a = log_decay(dt, p["A_log"], tau)
        s1 = jnp.exp(a)[:, :, None, None] * s0 + jnp.einsum("bh,bhp,bn->bhpn", dt, x, Bm)
        y = jnp.einsum("bhpn,bn->bhp", s1, Cm) + p["D"][:, None] * x
        y = rmsnorm(y.reshape(-1, D_INNER) * jax.nn.silu(z), p["gnorm"])
        return (y @ p["out_proj"].T, residual), (s1, window[:, 1:])

    (hidden, residual), (ssm, conv) = jax.lax.scan(body, (h, jnp.zeros_like(h)), (layers, ssm, conv))
    return rmsnorm(hidden + residual, params["norm_f"]) @ params["embedding"].T, (ssm, conv)


def generate(params, logits, state, n_tokens, tau=math.inf):
    """Greedy decoding: from the logits of the last prompt token, return (B, n_tokens) token ids."""
    tau = jnp.float32(tau)
    out = []
    tok = jnp.argmax(logits, axis=-1)
    for _ in range(n_tokens):
        out.append(tok)
        logits, state = decode_step(params, tok, state, tau)
        tok = jnp.argmax(logits, axis=-1)
    return np.stack([np.asarray(t) for t in out], axis=1)
