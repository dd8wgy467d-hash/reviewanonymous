# TSO suite

Tracking Shuffled Objects is state tracking over English sentences with N = 3, 4 or 5 people
(`nfsm.data.tso`). See the Experiments section of the draft.

| task | stream | target |
|---|---|---|
| `swap-N` | `alice swaps with bob . charlie swaps with alice . …` | parse register and permutation, (N + 1)·N! states |
| `parcel-N` | `alice has the key . bob has the ring . …` followed by swap sentences | parse register and item holders (44 / 202 / 1132 states) |
| `question-N` | a parcel stream followed by `who has the <item> ?` | the parcel states, plus the holder's name at the last token |

- **Parse register.** It holds the first name of the sentence in progress. The second name fires
  the swap.
- **Why NFSM needs two layers.** An item head's transition depends on the parse head's state, so
  NFSM runs at 2 and 3 layers with a 2-tap conv.
- **Head families.**
  - `minimal` (swap): the parse head and N − 1 image heads.
  - `exact` (parcel, question): the parse head and one head of N + 1 states per item.
- **Baselines.** They run the free arm at width 256 over 4 layers.
- **Schedule and arms.** Models train at length 256 and are evaluated from 64 to 32768. Everything
  else (recipe, arms, outputs) is shared with the FSA suite; see the top-level README.
- **Question padding.** When T − 5 is not a multiple of 5, `question` inserts the remaining
  (T − 5) mod 5 tokens as single extra `.` after distinct random sentences. Every tested length is
  therefore built from the same local patterns as the training length.
- **Question scoring.** Early stopping, checkpoint selection and "solved" require both the dense
  sequence accuracy and the answer accuracy to reach the bar.

## Run

```bash
python -m experiments.tso.run --list-jobs [--tasks swap-4 parcel-4] [--arms free]
python -m experiments.tso.run --job parcel-4,exact,nfsm,2,anchored,42
bash scripts/dispatch.sh tso [--dry-run] [FILTERS]
python -m experiments.tso.run --aggregate      # results/tso/summary.md and seq_acc.pdf
```
