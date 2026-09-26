# FSA suite

These are algebraic state-tracking tasks. Each input symbol is a generator of a monoid, and the
target after every prefix is the element that prefix composes to. Supervision is dense. See the
Experiments section of the draft.

Models train at length 64 and are evaluated at every power of two from 8 to 32768 on fresh
batches. The score at each length is sequence accuracy: a sequence counts only if every position
is right.

| task | structure | letters | classes | NFSM rungs (head sizes) |
|---|---|---|---:|---|
| Z2   | cyclic group             | 0, 1                               | 2    | regular (2) |
| Z16  | cyclic group             | 0 … 15                             | 16   | regular (16) |
| S3   | symmetric group          | adjacent transpositions            | 6    | regular (6), factored (3, 2) |
| S4   | symmetric group          | adjacent transpositions            | 24   | regular (24), factored (4, 3, 2) |
| A5   | alternating group        | adjacent 3-cycles                  | 60   | regular (60), factored (5, 12) |
| M11  | Mathieu group            | 2 generators and their inverses    | 7920 | factored (11, 11, 11, 11) |
| DFF5 | flip-flop, ≤ 4 identities in a row | identity, reset, set     | 3    | regular (3) |
| FF   | flip-flop, 90 % identities         | identity, reset, set     | 3    | regular (3) |

- **Rungs.** `regular` is one head of |Q| states. `factored` is the smallest family of quotient
  heads that separates the classes: point images, the S4 pairing, parity, and the cosets of a
  5-cycle (`nfsm.data.groups`).
- **NFSM setup.** One layer, no conv, width 128.
- **Baselines.** Mamba, AUSSM and PD-SSM run the free arm at width 256 over 4 layers, under the
  rung name `fixed`.
- **Arms.** `anchored` and `free` are described in the top-level README.

## Run

```bash
python -m experiments.fsa.run --list-jobs [--tasks ...] [--cells ...] [--arms ...] [--seeds ...]
python -m experiments.fsa.run --job A5,factored,nfsm,1,free,42 [--steps N] [--no-wandb]
bash scripts/dispatch.sh fsa [--dry-run] [FILTERS]
python -m experiments.fsa.run --aggregate
```

## Outputs

A job writes two files:

- `results/fsa/parts/<token>.json`: the job fields, the recipe, `lens`, `seq_acc`, `token_acc`,
  the training `history` and `extraction`.
- `results/fsa/checkpoints/<token>.msgpack`: the weights.

`extraction` reads a trained NFSM as a finite machine (`extract.py`). With one layer and no conv,
each letter has a fixed map per head. Starting from the start state, the extraction walks the
model's joint head states alongside the true classes and counts two kinds of failure:

- `conflicts`: a reached joint state that is met with a second class;
- `readout_errors`: reached (letter, state) pairs whose argmax is not the class.

`exact` means both counts are zero, which makes the model correct at every length.

`--aggregate` writes `results/fsa/summary.md` and `results/fsa/seq_acc.pdf`. The summary gives,
per configuration, the solved runs (accuracy 1.0 at T = 32768), the median failing length (first
length below 1.0 for NFSM, 0.9 for a baseline) and the number of exact machines.
