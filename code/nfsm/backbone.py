"""The sequence model shared by every suite: token ids [B, T] -> logits [B, T, n_classes].

    h = LayerNorm(MLP-GLU(sqrt(m) * Embed(x)))
    per layer:  h = h + cell(conv(LayerNorm(h)));   h = h + MLP-GLU(LayerNorm(h))
    logits = Linear(MLP-GLU(LayerNorm(h)))
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from nfsm.cells import CELL_REGISTRY, NFSM, PDSSM
from nfsm.cells.conv import ShortConv


@dataclass
class ModelConfig:
    """Architecture of one model."""
    vocab: int                          # input alphabet size
    n_classes: int                      # output width
    head_sizes: Tuple[int, ...]         # NFSM states per head; a baseline's width is the sum
    layer_head_sizes: Optional[Tuple[Tuple[int, ...], ...]] = None   # NFSM, per layer (None: head_sizes)
    cell: str = "nfsm"                  # nfsm | mamba | aussm | pdssm
    n_layers: int = 1
    m: int = 128                        # model width
    dropout: float = 0.1                # in every MLP-GLU
    conv_kernel: int = 0                # causal conv taps before each cell (0 = none)
    noise_scale: float = 0.5            # NFSM Gumbel temperature in training
    scan_impl: str = "flat"             # NFSM scan: flat | sequential
    d_state: int = 16                   # mamba / aussm states per channel
    n_dict: int = 16                    # pdssm dictionary size


class MLPGLU(nnx.Module):
    """x [..., m] -> proj_out(dropout(silu(b) * a)), with (a, b) = proj_in(x) split in two."""

    def __init__(self, m: int, dropout: float, rngs: nnx.Rngs) -> None:
        self.proj_in = nnx.Linear(m, 2 * m, rngs=rngs)
        self.proj_out = nnx.Linear(m, m, rngs=rngs)
        self.drop = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: Array, training: bool = False) -> Array:
        a, b = jnp.split(self.proj_in(x), 2, axis=-1)
        return self.proj_out(self.drop(b * jax.nn.sigmoid(b) * a, deterministic=not training))


class RecurrentLayer(nnx.Module):
    """x -> x + cell(conv(norm(x))) -> + MLP-GLU(norm(.))."""

    def __init__(self, cell: nnx.Module, m: int, dropout: float, rngs: nnx.Rngs,
                 conv_kernel: int = 0) -> None:
        self.cell = cell
        self.norm_cell = nnx.LayerNorm(m, rngs=rngs)
        self.conv = ShortConv(m, conv_kernel, rngs) if conv_kernel > 0 else None
        self.norm_mlp = nnx.LayerNorm(m, rngs=rngs)
        self.mlp = MLPGLU(m, dropout, rngs)

    def cell_input(self, x: Array) -> Array:
        """The tensor the cell reads: conv(norm_cell(x))."""
        u = self.norm_cell(x)
        return u if self.conv is None else self.conv(u)

    def __call__(self, x: Array, training: bool = False, return_explore: bool = False):
        """x [B, T, m] -> (x_out, cell_out, excess [B, T] or None)."""
        u = self.cell_input(x)
        excess = None
        if isinstance(self.cell, NFSM):
            c = self.cell(u, deterministic=not training, return_explore=return_explore)
            if return_explore:
                c, excess = c
        elif isinstance(self.cell, PDSSM):
            c = self.cell(u, deterministic=not training)
        else:
            c = self.cell(u)
        x = x + c
        return x + self.mlp(self.norm_mlp(x), training=training), c, excess


def _make_cell(cfg: ModelConfig, rngs: nnx.Rngs, layer: int = 0) -> nnx.Module:
    cls = CELL_REGISTRY[cfg.cell]
    if cfg.cell == "nfsm":
        sizes = cfg.head_sizes if cfg.layer_head_sizes is None else cfg.layer_head_sizes[layer]
        return cls(tuple(sizes), cfg.m, rngs, noise_scale=cfg.noise_scale, scan_impl=cfg.scan_impl)
    d = sum(cfg.head_sizes)
    if cfg.cell == "pdssm":
        return cls(d, cfg.m, rngs, n_dict=cfg.n_dict)
    return cls(d, cfg.m, rngs, d_state=cfg.d_state)


class Backbone(nnx.Module):
    """Token ids [B, T] -> logits [B, T, n_classes]."""

    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs) -> None:
        if cfg.cell not in CELL_REGISTRY:
            raise ValueError(f"unknown cell {cfg.cell!r}; one of {sorted(CELL_REGISTRY)}")
        if cfg.layer_head_sizes is not None and len(cfg.layer_head_sizes) != cfg.n_layers:
            raise ValueError(f"layer_head_sizes has {len(cfg.layer_head_sizes)} entries for "
                             f"{cfg.n_layers} layers")
        self.cfg = cfg
        self.embed = nnx.Embed(cfg.vocab, cfg.m, rngs=rngs)
        self.input_mlp = MLPGLU(cfg.m, cfg.dropout, rngs)
        self.input_norm = nnx.LayerNorm(cfg.m, rngs=rngs)
        self.layers = nnx.List([
            RecurrentLayer(_make_cell(cfg, rngs, i), cfg.m, cfg.dropout, rngs, cfg.conv_kernel)
            for i in range(cfg.n_layers)])
        self.final_norm = nnx.LayerNorm(cfg.m, rngs=rngs)
        self.output_mlp = MLPGLU(cfg.m, cfg.dropout, rngs)
        self.output_proj = nnx.Linear(cfg.m, cfg.n_classes, rngs=rngs)

    def embed_tokens(self, x: Array, training: bool = False) -> Array:
        """Token ids [...] -> the stream entering layer 0 [..., m]."""
        w = self.embed.embedding.value
        e = (jax.nn.one_hot(x, w.shape[0], dtype=w.dtype) @ w) * jnp.sqrt(jnp.float32(self.cfg.m))
        return self.input_norm(self.input_mlp(e, training=training))

    def head(self, h: Array, training: bool = False) -> Array:
        """Stream leaving the last layer [..., m] -> logits [..., n_classes]."""
        return self.output_proj(self.output_mlp(self.final_norm(h), training=training))

    def __call__(self, x: Array, training: bool = False, return_explore: bool = False,
                 return_cell_out: bool = False):
        """x [B, T] -> logits; with return_explore also the NFSM exploration excess [B, T] (mean over
        layers, 0.0 for baselines); with return_cell_out also the list of per-layer cell outputs."""
        h = self.embed_tokens(x, training)
        excess, n_exc, cell_outs = jnp.float32(0.0), 0, []
        for layer in self.layers:
            h, c, e = layer(h, training=training, return_explore=return_explore)
            cell_outs.append(c)
            if e is not None:
                excess, n_exc = excess + e, n_exc + 1
        out = (self.head(h, training),)
        if return_explore:
            out += (excess / max(1, n_exc),)
        if return_cell_out:
            out += (cell_outs,)
        return out[0] if len(out) == 1 else out

    def param_count(self) -> int:
        return int(sum(jnp.asarray(p).size for p in jax.tree_util.tree_leaves(nnx.state(self, nnx.Param))))
