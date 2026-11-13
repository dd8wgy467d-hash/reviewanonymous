"""The Backbone on MFCC frames with a sequence-level read-out.

    h = LayerNorm(MLP-GLU(Linear(x)))         x [B, T, F]
    recurrent layers of the Backbone
    logits = head(h[:, -1])  (last)   or   head(mean_t h)  (mean)
"""

import jax
import jax.numpy as jnp
from flax import nnx

from nfsm.backbone import Backbone, ModelConfig


class KWSModel(Backbone):
    """Frames [B, T, F] -> logits [B, n_classes] (and the NFSM exploration excess [B, T])."""

    def __init__(self, cfg: ModelConfig, n_features: int, readout: str, rngs: nnx.Rngs) -> None:
        super().__init__(cfg, rngs)
        assert readout in ("last", "mean"), readout
        self.readout = readout
        self.embed = nnx.Linear(n_features, cfg.m, rngs=rngs)   # replaces the token embedding

    def embed_tokens(self, x, training: bool = False):
        return self.input_norm(self.input_mlp(self.embed(x), training=training))

    def __call__(self, x, training: bool = False, return_explore: bool = False):
        h = self.embed_tokens(x, training)
        excess, n_exc = jnp.zeros(x.shape[:2], jnp.float32), 0
        for layer in self.layers:
            h, _, e = layer(h, training=training, return_explore=return_explore)
            if e is not None:
                excess, n_exc = excess + e, n_exc + 1
        pooled = h[:, -1] if self.readout == "last" else jnp.mean(h, axis=1)
        logits = self.head(pooled, training)
        return (logits, excess / max(1, n_exc)) if return_explore else logits


def build_model(job: dict) -> KWSModel:
    """An untrained model of the shape a job dict describes (seeded by job['seed'])."""
    cfg = ModelConfig(vocab=1, n_classes=job["n_classes"], head_sizes=tuple(job["head_sizes"]),
                      cell=job["cell"], n_layers=job["layers"], m=job["m"],
                      conv_kernel=job["conv_kernel"],
                      **({"d_state": job["d_state"]} if "d_state" in job else {}),
                      **({"pdssm_mag_init": job["pdssm_mag_init"]} if "pdssm_mag_init" in job else {}),
                      **({"pdssm_decay_floor": job["pdssm_decay_floor"]} if "pdssm_decay_floor" in job else {}),
                      **({"noise_scale": job["noise_scale"]} if "noise_scale" in job else {}))
    return KWSModel(cfg, job["n_features"], job["readout"],
                    rngs=nnx.Rngs(jax.random.PRNGKey(job["seed"])))
