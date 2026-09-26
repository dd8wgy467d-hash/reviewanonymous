"""Read a trained one-layer NFSM of the FSA suite as a finite machine and check it exactly.

With one layer and no conv, the cell input at position t depends on the letter x_t only, so each
letter a has a fixed map tau_a per head, and the logits at t depend on (x_t, h_t) only.
"""

import jax.numpy as jnp
import numpy as np

from experiments.fsa.tasks import next_class


def extract_machine(model, job: dict, chunk: int = 4096) -> dict:
    """Pair the model's joint head states with the task's classes, from the start state, over the
    reachable set; then check the read-out on every reached (letter, state) pair.

    Returns {exact, states, classes, conflicts, readout_errors}: `conflicts` counts reached joint
    states met with a second class, `readout_errors` the (letter, state, class) triples whose argmax
    is not the class. exact (no conflict, no read-out error) means the model is correct at every
    length."""
    layer = list(model.layers)[0]
    cell = layer.cell
    A, delta = job["vocab"], next_class(job["task"])
    letters = jnp.arange(A, dtype=jnp.int32)[:, None]
    maps = np.asarray(cell.transition_maps(layer.cell_input(model.embed_tokens(letters)))[:, 0])
    K = maps.shape[1]

    start = (0,) * K
    cls_of, frontier, conflicts, edges = {start: 0}, [start], 0, set()
    while frontier:
        s = frontier.pop()
        q = cls_of[s]
        for a in range(A):
            s2 = tuple(int(maps[a, k, s[k]]) for k in range(K))
            q2 = int(delta[a, q])
            edges.add((a, s2, q2))
            if s2 not in cls_of:
                cls_of[s2] = q2
                frontier.append(s2)
            elif cls_of[s2] != q2:
                conflicts += 1

    edges = np.asarray([(a, *s, q) for a, s, q in edges], dtype=np.int32)
    errors = 0
    for i in range(0, len(edges), chunk):
        e = edges[i:i + chunk]
        h = model.embed_tokens(jnp.asarray(e[:, 0]))
        h = h + cell.readout(jnp.asarray(e[:, 1:1 + K]))
        h = h + layer.mlp(layer.norm_mlp(h))
        errors += int(np.sum(np.asarray(jnp.argmax(model.head(h), -1)) != e[:, -1]))
    return {"exact": conflicts == 0 and errors == 0, "states": len(cls_of),
            "classes": len(set(cls_of.values())), "conflicts": conflicts,
            "readout_errors": errors}
