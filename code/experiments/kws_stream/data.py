"""Streams of noise MFCC frames with at most one digit clip inside (see tasks.py for the task)."""

import os

import numpy as np

from experiments.kws import prepare
from experiments.kws import tasks as kws
from experiments.kws_stream import tasks


class Streams:
    """Digit clips and continuous noise frames per split, from $KWS_DATA_DIR (experiments.kws.prepare)."""

    def __init__(self, data_dir: str = None, ood_filter: bool = False) -> None:
        """ood_filter: test noise crops avoid frames flagged by ood_frames()."""
        d = data_dir or kws.DATA_DIR
        self.words, self.labels, self.noise = {}, {}, {}
        with np.load(os.path.join(d, "digits.npz")) as f:
            # a gain of g dB adds g to every mel band in dB, hence g * sqrt(n_mels) to c0 (orthonormal
            # DCT) and nothing to the other coefficients; in standardised units:
            self.c0_per_db = float(np.sqrt(prepare.N_MELS) / f["std"][0])
            for split in ("train", "val", "test"):
                keep = f[f"y_{split}"] < tasks.NONE
                self.words[split] = f[f"x_{split}"][keep]
                self.labels[split] = f[f"y_{split}"][keep]
        with np.load(os.path.join(d, "noise_streams.npz")) as f:
            for split in ("train", "val", "test"):
                keys = sorted(k for k in f.files if k.startswith(split + "_") and not k.endswith("_pow"))
                self.noise[split] = [(f[k], f[k + "_pow"]) for k in keys]
        self.clean, self.ood = {}, {}
        if ood_filter:
            flags = ood_frames(self.noise)
            lo = tasks.CROP_FRAMES[0]
            cums = [np.concatenate([[0], np.cumsum(f)]) for f in flags]
            keep = [i for i, c in enumerate(cums) if (c[lo:] - c[:-lo] == 0).any()]
            self.ood = {"flagged_share": [round(float(f.mean()), 4) for f in flags],
                        "recordings_left_out": [i for i in range(len(flags)) if i not in keep]}
            self.noise["test"] = [self.noise["test"][i] for i in keep]
            self.clean["test"] = [cums[i] for i in keep]

    def noise_frames(self, rng, split: str, T: int) -> np.ndarray:
        """[T, F]: random noise crops at random levels in NOISE_DBFS, crossfaded."""
        recs = self.noise[split]
        F = recs[0][0].shape[1]
        lo, hi = tasks.CROP_FRAMES
        buf = np.empty((T + hi, F), np.float64)        # filled in place up to L
        L = 0
        clean = self.clean.get(split)
        while L < T:
            i = rng.integers(len(recs))
            rec, pw = recs[i]
            n = min(int(rng.integers(lo, hi + 1)), len(rec))
            if clean is None:
                s = int(rng.integers(0, len(rec) - n + 1))
            else:                                      # --ood-filter: only crops free of flagged frames
                c = clean[i]
                ok = np.flatnonzero(c[n:] - c[:-n] == 0)
                if not len(ok):
                    continue
                s = int(ok[rng.integers(len(ok))])
            level = 10 * np.log10(pw[s:s + n].sum() / n + 1e-12)      # = np.mean, same arithmetic
            shift = (rng.uniform(*tasks.NOISE_DBFS) - level) * self.c0_per_db
            k = 0 if L == 0 else min(_fade(rng), L, n)   # same blend as crossfade(buf[:L], piece, k)
            if k:
                head = rec[s:s + k].copy()
                head[:, 0] += shift
                w = _WEIGHTS[k]
                buf[L - k:L] = (1 - w) * buf[L - k:L] + w * head
            tail = rec[s + k:s + n].copy()             # shifted in the recordings' dtype, as before
            tail[:, 0] += shift
            buf[L:L + n - k] = tail
            L += n - k
        return buf[:T]

    def stream(self, rng, split: str, T: int, word=None, place: str = "random") -> tuple:
        """(frames [T, F], hold): noise with `word` at a random place, at the end or at the start;
        hold = frames after the word (-1 without word).
        """
        x = self.noise_frames(rng, split, T)
        if word is None:
            return x, -1
        p = {"end": T - tasks.WORD, "start": 0}.get(place)
        if p is None:
            p = int(rng.integers(0, T - tasks.WORD + 1))
        seg = word.copy()
        for k, head in ((_fade(rng), True), (_fade(rng), False)):
            w = (np.arange(1, k + 1) / (k + 1))[:, None]
            if head:                      # noise fades into the word
                seg[:k] = (1 - w) * x[p:p + k] + w * word[:k]
            else:                         # the word fades into noise
                e = p + tasks.WORD
                seg[-k:] = (1 - w) * word[-k:] + w * x[e - k:e]
        x[p:p + tasks.WORD] = seg
        return x, T - (p + tasks.WORD)

    def batch(self, rng, split: str, T: int, B: int) -> tuple:
        """Random (x [B, T, F], y [B], hold [B]); none with probability NONE_FRAC."""
        xs, ys, hs = [], [], []
        for _ in range(B):
            if rng.random() < tasks.NONE_FRAC:
                x, h = self.stream(rng, split, T)
                y = tasks.NONE
            else:
                i = int(rng.integers(len(self.labels[split])))
                x, h = self.stream(rng, split, T, self.words[split][i])
                y = int(self.labels[split][i])
            xs.append(x), ys.append(y), hs.append(h)
        return np.stack(xs).astype(np.float32), np.asarray(ys, np.int32), np.asarray(hs, np.int32)

    def fixed_set(self, split: str, T: int, seed: int, chunk: int, place: str = "random"):
        """Chunks (x, y, hold) of a fixed set: every digit clip once plus one noise stream per ten.
        With place "end" or "start", digit streams only, all noise before or after the word.
        """
        sub = {"random": [], "end": [1], "start": [2]}[place]
        rng = np.random.default_rng([seed, T] + sub)
        n_words = len(self.labels[split])
        items = list(range(n_words)) + ([] if sub else [-1] * round(n_words / 10))
        for c in range(0, len(items), chunk):
            xs, ys, hs = [], [], []
            for i in items[c:c + chunk]:
                x, h = self.stream(rng, split, T, None if i < 0 else self.words[split][i], place)
                xs.append(x), ys.append(tasks.NONE if i < 0 else int(self.labels[split][i])), hs.append(h)
            yield np.stack(xs).astype(np.float32), np.asarray(ys, np.int32), np.asarray(hs, np.int32)


def nn_distance(q: np.ndarray, ref: np.ndarray, chunk: int = 256) -> np.ndarray:
    """[len(q)]: Euclidean distance of each row of q to its nearest row of ref."""
    return np.concatenate([np.sqrt(((q[c:c + chunk, None] - ref[None]) ** 2).sum(-1).min(1))
                           for c in range(0, len(q), chunk)])


def ood_frames(noise: dict) -> list:
    """Per test noise recording, bool [frames]: farther from the training noise than the
    OOD_QUANTILE percentile of the validation noise (MFCC c1-c12).
    """
    ref = np.concatenate([x for x, _ in noise["train"]])[:, 1:]
    val = np.concatenate([nn_distance(x[:, 1:], ref) for x, _ in noise["val"]])
    th = np.percentile(val, tasks.OOD_QUANTILE)
    return [nn_distance(x[:, 1:], ref) > th for x, _ in noise["test"]]


_WEIGHTS = {k: (np.arange(1, k + 1) / (k + 1))[:, None] for k in range(1, tasks.FADE_FRAMES[1] + 1)}


def _fade(rng) -> int:
    return int(rng.integers(tasks.FADE_FRAMES[0], tasks.FADE_FRAMES[1] + 1))


def crossfade(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """a then b, the last k frames of a blended linearly into the first k frames of b."""
    k = min(k, len(a), len(b))
    w = (np.arange(1, k + 1) / (k + 1))[:, None]
    return np.concatenate([a[:-k], (1 - w) * a[-k:] + w * b[:k], b[k:]])
