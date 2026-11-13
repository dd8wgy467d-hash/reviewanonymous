# KWS Stream

This task is spoken-digit detection inside long noise. Each sequence is a stream of MFCC frames (100
per second) built from the recorded background noise of Speech Commands v0.02, cut into crops of 1 to
3 s. Each crop is set to a random level of −45 to −30 dBFS RMS, 15 to 30 dB below the spoken part of a
typical digit; as distributed, the recordings range from −48 to −10 dBFS. One digit clip of 101
frames sits at a random position, except in one stream out of 11, which holds noise only. Every join
between pieces is a linear crossfade over 5 to 10 frames.

The model has 11 classes, the ten digits and `none`. The class is read from the last frame's logits
only, so there is no state supervision.

- **Training:** each model starts from its KWS Digits checkpoint (same token, same architecture; the
  KWS `noise` class becomes `none`). A length curriculum follows: T = 101 only for 5k steps, then one
  more length of 128, 160, 200, 256, 320 and 400 frames every 3k steps. Each step draws T uniformly
  among the unlocked lengths, so the full 1 to 4 s range is in use from step 20k to the end (45k).
  Batches are built on the fly from the training split. The checkpoint is selected on fixed
  validation streams at T = 101 and T = 400, among the checks from step 20k on.
- **Test:** fixed streams at T = 101, 200, …, 102400 frames, then 204800, 307200, 409600 and
  819200 (1 s to 8192 s). Every model sees the same streams: every test digit clip once, plus one
  noise-only stream per ten digits.
- **Reported results:** accuracy against T, and the digit accuracy against the number of frames after
  the word (the hold). The hold measures how long the model has to keep the digit in its state.

The default jobs are NFSM 64×4, minGRU⁻ width 256 and Mamba⁻ width 256, with 4 layers, model width
m = 128, 45k steps at batch 64, and 3 seeds. The data comes from `experiments.kws.prepare`, which
writes `digits.npz` and `noise_streams.npz`.

    python -m experiments.kws_stream.run --list-jobs
    python -m experiments.kws_stream.run --job nfsm-64x4,4,last,42
    python -m experiments.kws_stream.run --aggregate      # summary.md
    python scripts/analyse_kws_stream.py                  # figures/kws_stream/
    python -m experiments.kws_stream.examples --out DIR   # listening examples (wav, png)

On a SLURM cluster: `TIME=06:00:00 bash scripts/dispatch.sh kws_stream`.
