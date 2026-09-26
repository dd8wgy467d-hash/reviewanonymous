"""AdamW with global-norm clipping and a cosine decay to lr / 10."""

import jax
import jax.numpy as jnp
import optax

# Parameters never decayed: the transition generators and embeddings.
WD_EXCLUDED = ("trans_proj", "embed", "A_log", "xA_proj", "dict_M", "sel_proj")


def weight_decay_mask(params):
    """True on 2D+ weights whose path names none of WD_EXCLUDED."""
    def decay(path, x):
        s = jax.tree_util.keystr(path)
        return jnp.ndim(x) >= 2 and not any(t in s for t in WD_EXCLUDED)
    return jax.tree_util.tree_map_with_path(decay, params)


def make_optimizer(lr: float, steps: int, weight_decay: float, grad_clip: float = 1.0,
                   eps: float = 1e-8, b1: float = 0.9, b2: float = 0.999) -> optax.GradientTransformation:
    """clip -> Adam(b1, b2, eps) -> masked weight decay -> cosine lr from `lr` to lr / 10 over `steps`."""
    schedule = optax.cosine_decay_schedule(lr, max(1, steps), alpha=0.1)
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
        optax.add_decayed_weights(weight_decay, mask=weight_decay_mask),
        optax.scale_by_learning_rate(schedule),
    )
