"""Training loop shared by every suite.

L = CE[w] + free_explore * c(t) * excess[frontier]

c(t) is a cosine from 1 to explore_final_frac. w = exp(-lambda (t - t*)_+) x class balance.
Early stopping and checkpoint selection use the network output. An answer target ans [B] adds its CE at the last position.
"""

import math
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nfsm.eval import make_acc_fn
from nfsm.training.losses import class_balance, first_error_masks, weighted_sequence_mean
from nfsm.training.optim import make_optimizer


@dataclass
class TrainConfig:
    """The training recipe."""
    steps: int = 100_000                  # step budget (early stopping usually ends sooner)
    batch_size: int = 256
    seq_len: int = 64
    lr: float = 1e-3                      # peak; cosine decay to lr / 10
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    first_error_lambda: float = 3.0
    free_explore: float = 0.03            # beta on the exploration excess, free phase
    early_stop_acc: float = 1.0           # stop once the val sequence accuracy is >= this ...
    early_stop_patience: int = 40         # ... on this many consecutive checks
    val_batch: int = 512
    val_every: int = 50
    print_every: int = 500
    use_wandb: bool = True
    wandb_project: str = "nfsm"
    restore_best: bool = True             # end on the best checkpoint (False: on the final model)
    adam_eps: float = 1e-8                # Adam's epsilon
    adam_b1: float = 0.9                  # Adam's betas
    adam_b2: float = 0.999
    class_balance_power: float = 0.5      # class weight ~ count^-power (1 = inverse frequency)
    explore_final_frac: float = 0.1       # beta decays by a cosine to this fraction of itself over `steps`


def make_train_step(model, tx, loss_fn):
    """step(model, opt_state, batch) -> (opt_state, loss, aux); updates `model` in place."""
    graphdef, _ = nnx.split(model)

    @jax.jit
    def _grads(state, batch):
        m = nnx.merge(graphdef, state)
        (loss, aux), grads = nnx.value_and_grad(lambda mm: loss_fn(mm, batch), has_aux=True)(m)
        _, new_state = nnx.split(m)
        return new_state, nnx.state(grads, nnx.Param), loss, aux

    @jax.jit
    def _apply(params, grads, opt_state):
        updates, opt_state = tx.update(grads, opt_state, params=params)
        return optax.apply_updates(params, updates), opt_state

    def step(model_ref, opt_state, batch):
        _, state = nnx.split(model_ref)
        new_state, grads, loss, aux = _grads(state, batch)
        nnx.update(model_ref, new_state)
        params, opt_state = _apply(nnx.state(model_ref, nnx.Param), grads, opt_state)
        nnx.update(model_ref, params)
        return opt_state, loss, aux

    return step


# --- objectives ----------------------------------------------------------------------------------

def _answer_term(logits, y, ans):
    if ans is None:
        return 0.0
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits[:, -1, y.shape[-1]:], ans))


def _dense_ce(logits, y, lam, power=1.0):
    """(weighted CE of the dense targets, frontier [B, T]); power: see class_balance."""
    ld = logits[..., :y.shape[-1]]
    correct = (jnp.argmax(ld, -1) == jnp.argmax(y, -1)).astype(jnp.float32)
    w, frontier = first_error_masks(correct, lam)
    return (weighted_sequence_mean(optax.softmax_cross_entropy(ld, y), w * class_balance(y, power)),
            frontier)


def explore_scale(cfg: TrainConfig, step: int) -> float:
    """Multiplier of beta at `step`: a cosine from 1 to cfg.explore_final_frac over cfg.steps."""
    frac = cfg.explore_final_frac
    t = min(max(step, 0), cfg.steps) / max(1, cfg.steps)
    return frac + (1.0 - frac) * 0.5 * (1.0 + math.cos(math.pi * t))


def _unpack(batch):
    """(x, y, ans, beta multiplier) of a batch; a three-part batch keeps beta as it is."""
    return batch[0], batch[1], batch[2], (batch[3] if len(batch) > 3 else 1.0)


def make_free_loss(cfg: TrainConfig):
    def loss_fn(model, batch):
        x, y, ans, scale = _unpack(batch)
        logits, excess = model(x, training=True, return_explore=True)
        ce, frontier = _dense_ce(logits, y, cfg.first_error_lambda, cfg.class_balance_power)
        exc = weighted_sequence_mean(jnp.broadcast_to(excess, frontier.shape), frontier)
        return ce + cfg.free_explore * scale * exc + _answer_term(logits, y, ans), (ce, exc)
    return loss_fn


# --- the loop ------------------------------------------------------------------------------------

def _frozen_mask(frozen):
    """A mask builder over the parameters: True where `frozen(path string)` holds."""
    return lambda params: jax.tree_util.tree_map_with_path(
        lambda path, _: bool(frozen(jax.tree_util.keystr(path))), params)


def run_training(model, cfg: TrainConfig, data_fn, seed: int,
                 run_name: str = "", group: str = None, config: dict = None,
                 frozen=None) -> dict:
    """Train `model` in place, restore its best checkpoint and return the history.

    data_fn(key, B, T) -> (x, y) or (x, y, ans). frozen: predicate on parameter paths that get no
    update.
    """
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=run_name, group=group,
                   config={**asdict(cfg), **(config or {})})
    acc_fn = make_acc_fn(model)

    def fresh(loss_fn, freeze=None):
        tx = make_optimizer(cfg.lr, cfg.steps, cfg.weight_decay, cfg.grad_clip, eps=cfg.adam_eps,
                            b1=cfg.adam_b1, b2=cfg.adam_b2)
        if freeze is not None:
            zero = optax.masked(optax.set_to_zero(), freeze)
            tx = optax.chain(zero, tx, zero)
        return make_train_step(model, tx, loss_fn), tx.init(nnx.state(model, nnx.Param))

    freeze = _frozen_mask(frozen) if frozen is not None else None

    step_fn, opt_state = fresh(make_free_loss(cfg), freeze=freeze)

    history = {k: [] for k in ("step", "loss", "ce", "excess", "val_token_acc",
                               "val_seq_acc", "val_answer_acc", "explore_scale")}
    key = jax.random.PRNGKey(seed)
    best = {"score": (-1.0, -1.0), "step": None, "params": None}
    perfect_streak = 0
    step = -1
    for step in range(cfg.steps):
        key, sk = jax.random.split(key)
        b = data_fn(sk, cfg.batch_size, cfg.seq_len)
        scale = explore_scale(cfg, step)
        opt_state, loss, (ce, exc) = step_fn(model, opt_state,
                                                  (b[0], b[1], b[2] if len(b) == 3 else None,
                                                   jnp.asarray(scale, jnp.float32)))
        if step % cfg.val_every:
            continue

        loss, ce, exc = float(loss), float(ce), float(exc)
        if not np.isfinite(loss):
            print(f"[{step}] non-finite loss; stopping", flush=True)
            break
        vb = data_fn(jax.random.PRNGKey(seed * 7_919_003 + 13 + step // cfg.val_every),
                     cfg.val_batch, cfg.seq_len)
        v_tok, v_seq, v_ans = acc_fn(model, *vb)
        for k, v in (("step", step), ("loss", loss), ("ce", ce), ("excess", exc),
                     ("val_token_acc", v_tok), ("val_seq_acc", v_seq),
                     ("val_answer_acc", v_ans), ("explore_scale", scale)):
            history[k].append(v)
        log = {"train/loss": loss, "train/ce": ce, "train/excess": exc,
               "val/token_acc": v_tok, "val/seq_acc": v_seq}
        with_answer = len(vb) == 3
        if with_answer:
            log["val/answer_acc"] = v_ans
        v_stop = min(v_seq, v_ans) if with_answer else v_seq
        score = (v_stop, v_tok)
        perfect_streak = perfect_streak + 1 if v_stop >= cfg.early_stop_acc else 0
        if score >= best["score"]:
            best = {"score": score, "step": step,
                    "params": jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))}
        if cfg.use_wandb:
            wandb.log(log, step=step)
        if step % cfg.print_every == 0:
            print(f"[{step}] loss {loss:.4f} ce {ce:.4f} exc {exc:.3f}  val tok {v_tok:.4f} "
                  f"seq {v_seq:.4f}" + (f" answer {v_ans:.4f}" if with_answer else ""),
                  flush=True)
        if perfect_streak >= cfg.early_stop_patience:
            print(f"[{step}] early stop: val {'seq and answer' if with_answer else 'seq'} acc >= "
                  f"{cfg.early_stop_acc} on {cfg.early_stop_patience} checks", flush=True)
            break

    if cfg.restore_best and best["params"] is not None:
        nnx.update(model, best["params"])
    history.update(steps_run=step + 1, best_step=best["step"], best_val_seq_acc=best["score"][0])
    return history
