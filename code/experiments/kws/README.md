# KWS Digits

Spoken-digit classification on Google Speech Commands v0.02 from MFCC frames. Only the clip's class is
supervised, so whatever state the recurrent layers carry is learned without state labels.

- **Classes:** 11, the ten digits ("zero" to "nine") and background noise.
- **Input:** 101 frames of 13 MFCCs (30 ms window, 10 ms hop), standardised per coefficient with
  training statistics.
- **Split:** the dataset's own `validation_list.txt` and `testing_list.txt`, so no speaker appears in two
  splits. The noise class is cut from the six background-noise recordings: the first 80% of each
  recording goes to training, the next 10% to validation and the last 10% to testing.
- **Readout:** `last` (default) reads the class from the logits of the last frame. `mean` reads it
  from the time-averaged stream.

| model | layers | recurrent cell | model width m |
|---|---|---|---|
| NFSM | 3 | 4 heads × 16 states | 128 |
| minGRU⁻, Mamba⁻, AUSSM | 3 | width 64 | 128 |

The recipe is the repository's `TrainConfig` defaults: AdamW at 1e-3 with a cosine decay to 1e-4,
weight decay 1e-4 and clipping at 1. The NFSM exploration excess has weight 0.03. Training uses
batch 64 for 30k steps, checks validation every 250 steps, and keeps the best checkpoint, which is
then scored on the test split. Each job writes `results/kws/parts/<token>.json` with its test
accuracy and confusion matrix.

## Data

The data stays out of the repository. `prepare.py` downloads the archive (about 2.4 GB) and extracts
only the digit and noise folders. It then writes `$KWS_DATA_DIR/digits.npz`; the default directory is
`data/kws/`. Run it once, on a machine with internet access:

    python -m experiments.kws.prepare

## Jobs

The job token is `cell,layers,readout,seed`. The NFSM cell can name its heads as `nfsm-KxN` (K heads of N states); plain `nfsm` is `nfsm-4x16`. A baseline can name its width as `cell-W`, e.g. `mingru-256`; plain `mingru` is `mingru-64`. `deltaproduct-W` is DeltaProduct with n_h = 2 and 8 heads.

    python -m experiments.kws.run --list-jobs --cells nfsm mingru mamba aussm
    python -m experiments.kws.run --job nfsm,3,last,42
    python -m experiments.kws.run --aggregate          # results/kws/summary.md

On a SLURM cluster, the shared dispatcher submits one job per token. Set `TIME`, because its default
wall time per cell is sized for the long synthetic sweeps:

    TIME=04:00:00 bash scripts/dispatch.sh kws --cells nfsm mingru mamba aussm
