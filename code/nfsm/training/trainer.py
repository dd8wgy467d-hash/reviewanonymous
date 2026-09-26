"""Training loop shared by every suite: the free arm and the anchored arm.

Free arm:
    L = CE[w] + free_explore * c(t) * excess[frontier],
    c(t) a cosine from 1 to explore_final_frac over the step budget
Anchored arm: the top layer's cell read-out is pinned so that its output channels show each head's
state (`NFSM.pin_readout`), and the cell read-out (W, out_proj) is frozen:
    L = sum_k CE_k[w] + anchor_decoder_weight * CE[w'] + anchor_explore * c(t) * excess[frontier]
where CE_k scores head k against its code of the target class on the cell's own output. Once the
pinned heads reach a sequence accuracy of anchor_release_acc on anchor_release_patience consecutive
checks, the pin is released and training continues on the free loss with a fresh optimizer.
Early stopping and checkpoint selection use the network's own output in both arms and phases,
never the pinned heads.

Weights: w = exp(-lambda (t - t*)_+) x class balance of the batch; `frontier` is the first wrong
position t* of each sequence; `excess` is the NFSM exploration excess. A batch may carry an answer
target ans [B] (the TSO question task); its CE on the last position's logits after the dense
classes is added to both losses, and early stopping and checkpoint selection then use
min(validation sequence accuracy, answer accuracy).
"""

import math
import re
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
    anchor_explore: float = 0.01          # beta on the exploration excess, anchored phase
    anchor_decoder_weight: float = 0.1
    anchor_scale: float = 4.0             # gain of the pinned read-out
    anchor_release_acc: float = 0.9999
    anchor_release_patience: int = 10
    anchor_injective: bool = True         # require the pinned codes to separate every class
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


def _pinned_heads(cell_out, codes, slices, y):
    """[(head logits [B, T, n_k], target head state [B, T])] of the pinned read-out."""
    cls = jnp.argmax(y, -1)
    return [(cell_out[..., s:s + n], code[cls]) for code, (s, n) in zip(codes, slices)]


def _heads_correct(heads):
    ok = jnp.ones(heads[0][1].shape, bool)
    for logits, tgt in heads:
        ok = ok & (jnp.argmax(logits, -1) == tgt)
    return ok


def make_free_loss(cfg: TrainConfig):
    def loss_fn(model, batch):
        x, y, ans, scale = _unpack(batch)
        logits, excess = model(x, training=True, return_explore=True)
        ce, frontier = _dense_ce(logits, y, cfg.first_error_lambda, cfg.class_balance_power)
        exc = weighted_sequence_mean(jnp.broadcast_to(excess, frontier.shape), frontier)
        return ce + cfg.free_explore * scale * exc + _answer_term(logits, y, ans), (ce, exc)
    return loss_fn


def make_anchor_loss(cfg: TrainConfig, codes, slices):
    def loss_fn(model, batch):
        x, y, ans, scale = _unpack(batch)
        logits, excess, cell_outs = model(x, training=True, return_explore=True,
                                          return_cell_out=True)
        heads = _pinned_heads(cell_outs[-1], codes, slices, y)
        w, frontier = first_error_masks(_heads_correct(heads).astype(jnp.float32),
                                        cfg.first_error_lambda)
        w = w * class_balance(y, cfg.class_balance_power)
        ce = sum(weighted_sequence_mean(
            optax.softmax_cross_entropy_with_integer_labels(hl, tgt), w) for hl, tgt in heads)
        dec, _ = _dense_ce(logits, y, cfg.first_error_lambda, cfg.class_balance_power)
        exc = weighted_sequence_mean(jnp.broadcast_to(excess, frontier.shape), frontier)
        loss = (ce + cfg.anchor_decoder_weight * dec + cfg.anchor_explore * scale * exc
                + _answer_term(logits, y, ans))
        return loss, (ce, exc)
    return loss_fn


def make_anchor_acc_fn(model, codes, slices):
    """acc(model, x, y) -> (token, sequence) accuracy of the pinned heads, deterministic."""
    graphdef, _ = nnx.split(model)

    @jax.jit
    def _acc(state, x, y):
        _, cell_outs = nnx.merge(graphdef, state)(x, return_cell_out=True)
        ok = _heads_correct(_pinned_heads(cell_outs[-1], codes, slices, y))
        return jnp.mean(ok), jnp.mean(jnp.all(ok, axis=-1))

    def acc(model_ref, x, y):
        _, state = nnx.split(model_ref)
        return tuple(float(v) for v in _acc(state, x, y))

    return acc


def _pinned_mask(params):
    """True on every cell's read-out (W, out_proj), the parameters frozen while anchored."""
    def frozen(path, _):
        names = set(re.findall(r"\w+", jax.tree_util.keystr(path)))
        return "cell" in names and bool(names & {"W", "out_proj"})
    return jax.tree_util.tree_map_with_path(frozen, params)


# --- the loop ------------------------------------------------------------------------------------

def _frozen_mask(frozen):
    """A mask builder over the parameters: True where `frozen(path string)` holds."""
    return lambda params: jax.tree_util.tree_map_with_path(
        lambda path, _: bool(frozen(jax.tree_util.keystr(path))), params)


def run_training(model, cfg: TrainConfig, data_fn, seed: int, anchor_codes=None,
                 run_name: str = "", group: str = None, config: dict = None,
                 frozen=None) -> dict:
    """Train `model` in place, restore its best checkpoint and return the history.

    data_fn(key, B, T) -> (x [B, T] ids, y [B, T, C] one-hot) or (x, y, ans [B]).
    anchor_codes: one [C] array per head of the top layer (head_sizes order) for the anchored arm;
    None for the free arm. frozen: a predicate on a parameter's path string; the parameters it holds
    for get no update in the free phase (their gradient is zeroed before clipping, their update after
    weight decay). Validation draws a fresh batch at every check. The W&B run, if any, is left open
    for the caller."""
    if cfg.use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=run_name, group=group,
                   config={**asdict(cfg), **(config or {})})
    acc_fn = make_acc_fn(model)

    def fresh(loss_fn, mask=None, freeze=None):
        tx = make_optimizer(cfg.lr, cfg.steps, cfg.weight_decay, cfg.grad_clip, eps=cfg.adam_eps,
                            b1=cfg.adam_b1, b2=cfg.adam_b2)
        if mask is not None:
            tx = optax.chain(tx, optax.masked(optax.set_to_zero(), mask))
        if freeze is not None:
            zero = optax.masked(optax.set_to_zero(), freeze)
            tx = optax.chain(zero, tx, zero)
        return make_train_step(model, tx, loss_fn), tx.init(nnx.state(model, nnx.Param))

    freeze = _frozen_mask(frozen) if frozen is not None else None

    anchored = anchor_codes is not None
    if anchored:
        slices = list(model.layers)[-1].cell.pin_readout(anchor_codes, cfg.anchor_scale,
                                                         require_injective=cfg.anchor_injective)
        codes = tuple(jnp.asarray(c, jnp.int32) for c in anchor_codes)
        step_fn, opt_state = fresh(make_anchor_loss(cfg, codes, slices), _pinned_mask)
        anchor_acc = make_anchor_acc_fn(model, codes, slices)
    else:
        step_fn, opt_state = fresh(make_free_loss(cfg), freeze=freeze)

    history = {k: [] for k in ("step", "loss", "ce", "excess", "val_token_acc",
                               "val_seq_acc", "val_answer_acc", "anchor_seq_acc", "explore_scale")}
    history["anchor_released_step"] = None
    key = jax.random.PRNGKey(seed)
    best = {"score": (-1.0, -1.0), "step": None, "params": None}
    anchor_streak = perfect_streak = 0
    step = -1
    for step in range(cfg.steps):
        if anchored and anchor_streak >= cfg.anchor_release_patience:
            anchored = False
            step_fn, opt_state = fresh(make_free_loss(cfg), freeze=freeze)
            history["anchor_released_step"] = step
            print(f"[{step}] anchor released", flush=True)

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
        if anchored:
            _, a_seq = anchor_acc(model, vb[0], vb[1])
            history["anchor_seq_acc"].append(a_seq)
            log["anchor/seq_acc"] = a_seq
            anchor_streak = anchor_streak + 1 if a_seq >= cfg.anchor_release_acc else 0
        perfect_streak = perfect_streak + 1 if v_stop >= cfg.early_stop_acc else 0
        if score >= best["score"]:
            best = {"score": score, "step": step,
                    "params": jax.tree.map(jnp.copy, nnx.state(model, nnx.Param))}
        if cfg.use_wandb:
            wandb.log(log, step=step)
        if step % cfg.print_every == 0:
            print(f"[{step}] loss {loss:.4f} ce {ce:.4f} exc {exc:.3f}  val tok {v_tok:.4f} "
                  f"seq {v_seq:.4f}" + (f" answer {v_ans:.4f}" if with_answer else "")
                  + (f"  anchor seq {history['anchor_seq_acc'][-1]:.4f}" if anchored else ""),
                  flush=True)
        if perfect_streak >= cfg.early_stop_patience:
            print(f"[{step}] early stop: val {'seq and answer' if with_answer else 'seq'} acc >= "
                  f"{cfg.early_stop_acc} on {cfg.early_stop_patience} checks", flush=True)
            break

    if cfg.restore_best and best["params"] is not None:
        nnx.update(model, best["params"])
    history.update(steps_run=step + 1, best_step=best["step"], best_val_seq_acc=best["score"][0])
    return history
