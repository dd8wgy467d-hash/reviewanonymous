"""KWS Stream suite: one spoken digit, or none, inside recorded noise.

A stream of T MFCC frames chains noise crops of 1 to 3 s at random levels in NOISE_DBFS; in 10 of 11
streams a digit clip (101 frames) sits at a random position. Joins are crossfades of FADE_FRAMES
frames. The target (digit or `none`) is read from the last frame only.

Training starts from the KWS checkpoint and unlocks the lengths of TRAIN_LENS one by one (1 to 4 s).
The test sweep uses the same fixed streams for every model, at TEST_LENS (1 s to 8192 s). Job tokens
are those of experiments/kws/tasks.py.
"""

from experiments.kws import tasks as kws

SUITE = "kws_stream"
WANDB_PROJECT = "nfsm-kws-stream"
CLASSES = kws.CLASSES[:10] + ("none",)
NONE = 10
WORD = kws.N_FRAMES                      # frames of one digit clip
NONE_FRAC = 1 / 11                       # share of noise-only streams
CROP_FRAMES = (100, 300)                 # length range of one noise crop
FADE_FRAMES = (5, 10)                    # crossfade length range at every join
NOISE_DBFS = (-45.0, -30.0)              # RMS level range of a noise crop

TRAIN_LENS = (101, 128, 160, 200, 256, 320, 400)
CURRICULUM_START = 5_000                 # steps at T = 101 only
CURRICULUM_EVERY = 3_000                 # steps between two unlocked lengths
INIT_SUITE = "kws"                       # warm start from this suite's checkpoint of the same token
VAL_LENS = (101, 400)
TEST_LENS = (101, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800,
             307200, 409600, 819200)                              # up to 8192 s
TEST_SEED = 2027                         # the test streams are the same for every model
# Optional out-of-distribution filter of the test noise (off by default; --ood-filter): no test noise
# crop covers a frame farther from every training noise frame (MFCC c1-c12, Euclidean) than the
# OOD_QUANTILE percentile of the validation noise frames.
OOD_QUANTILE = 99.0

BATCH = 64
STEPS = 45_000
VAL_EVERY = 250
EVAL_TOKENS = 1024 * WORD                # frames per evaluation batch
SEEDS = (42, 43, 44)
CELLS = ("nfsm-64x4", "mingru-256", "mamba-256")
LAYERS = (4,)


def unlocked(step: int) -> tuple:
    """The training lengths unlocked at `step`."""
    n = 1 if step < CURRICULUM_START else 2 + (step - CURRICULUM_START) // CURRICULUM_EVERY
    return TRAIN_LENS[:min(n, len(TRAIN_LENS))]


def full_range_step() -> int:
    """First step at which every training length is unlocked."""
    return CURRICULUM_START + (len(TRAIN_LENS) - 2) * CURRICULUM_EVERY


def parse_token(token: str) -> dict:
    """cell,layers,readout,seed -> job dict (as in the KWS Digits suite)."""
    job = kws.parse_token(token)
    assert job["n_classes"] == len(CLASSES)
    return job


def iter_jobs(cells=None, layers=None, readouts=None, seeds=None) -> list:
    """Job tokens; by default the three cells of CELLS, 4 layers, readout last."""
    return [f"{c},{n},{r},{s}" for c in cells or CELLS for n in layers or LAYERS
            for r in readouts or ("last",) for s in seeds or SEEDS]
