"""Build the KWS arrays from Google Speech Commands v0.02 (once, needs internet).

    python -m experiments.kws.prepare [--data DIR] [--archive PATH] [--noise-streams-only]

Writes DIR/digits.npz: x_{train,val,test} [N, 101, 13] standardised MFCCs, y in 0..10 (10 = noise),
the standardisation statistics and the class names. Splits follow the dataset lists (no speaker in
two splits); noise clips are windows of the six background recordings (80 / 10 / 10 %).
Also writes DIR/noise_streams.npz, the continuous MFCC frames and frame power of each recording,
for experiments/kws_stream.

MFCC: 16 kHz, 30 ms Hann window, 10 ms hop, 40 mel bands (20 Hz to 8 kHz), 13 coefficients.
"""

import argparse
import json
import os
import tarfile
import urllib.request
import wave

import numpy as np
from scipy.fft import dct

from experiments.kws.tasks import CLASSES, DATA_DIR, N_FRAMES, N_MFCC, SR

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DIGITS = CLASSES[:10]
NOISE_DIR = "_background_noise_"
N_FFT, HOP, N_MELS, FMIN, FMAX, TOP_DB = 480, 160, 40, 20.0, 8000.0, 80.0


# --- features ------------------------------------------------------------------------------------

def _mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_inv(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """[N_MELS, N_FFT // 2 + 1] triangular filters, equal-area (each row sums to about 1)."""
    freqs = np.linspace(0.0, SR / 2, N_FFT // 2 + 1)
    edges = _mel_inv(np.linspace(_mel(FMIN), _mel(FMAX), N_MELS + 2))
    fb = np.zeros((N_MELS, freqs.size))
    for i in range(N_MELS):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[i] = np.maximum(0.0, np.minimum(up, down)) * 2.0 / (hi - lo)
    return fb


_FB = mel_filterbank()
_WINDOW = np.hanning(N_FFT + 1)[:-1]


def mfcc(wav: np.ndarray) -> np.ndarray:
    """1 s of 16 kHz audio [SR] -> MFCC [N_FRAMES, N_MFCC]."""
    x = np.pad(wav, N_FFT // 2, mode="reflect")
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(N_FRAMES)[:, None]
    power = np.abs(np.fft.rfft(x[idx] * _WINDOW, axis=-1)) ** 2
    db = 10.0 * np.log10(np.maximum(power @ _FB.T, 1e-10))
    db = np.maximum(db, db.max() - TOP_DB)
    return dct(db, type=2, norm="ortho", axis=-1)[:, :N_MFCC].astype(np.float32)


def fit_length(wav: np.ndarray) -> np.ndarray:
    return np.pad(wav, (0, max(0, SR - wav.size)))[:SR]


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as f:
        assert f.getframerate() == SR and f.getsampwidth() == 2 and f.getnchannels() == 1, path
        return np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768.0


# --- dataset -------------------------------------------------------------------------------------

def _wanted(name: str) -> bool:
    top = name.removeprefix("./").split("/")[0]
    return top in DIGITS or top == NOISE_DIR or top in ("validation_list.txt", "testing_list.txt")


def extract(archive: str, root: str) -> None:
    """Stream the archive and keep the digit folders, the noise folder and the split lists."""
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if _wanted(member.name):
                tar.extract(member, root, filter="data")


SPLIT_REGIONS = (("train", 0.0, 0.8), ("val", 0.8, 0.9), ("test", 0.9, 1.0))


def noise_windows(root: str, targets: dict) -> dict:
    """{split: [1 s windows]} cut from the background-noise recordings, about targets[split] per split."""
    d = os.path.join(root, NOISE_DIR)
    wavs = [read_wav(os.path.join(d, n)) for n in sorted(os.listdir(d)) if n.endswith(".wav")]
    out = {}
    for split, a, b in SPLIT_REGIONS:
        regions = [(int(a * w.size), int(b * w.size)) for w in wavs]
        room = sum(max(0, hi - lo - SR) for lo, hi in regions)
        stride = max(1, room // max(1, targets[split] - len(wavs)))
        print(f"  noise {split}: stride {stride / SR:.3f} s", flush=True)
        out[split] = [w[s:s + SR] for w, (lo, hi) in zip(wavs, regions)
                      for s in range(lo, hi - SR + 1, stride)]
    return out


def frame_power(wav: np.ndarray) -> np.ndarray:
    """1 s of audio [SR] -> mean power of each MFCC frame's window [N_FRAMES] (same centring)."""
    x = np.pad(wav, N_FFT // 2, mode="reflect")
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(N_FRAMES)[:, None]
    return np.mean(x[idx] ** 2, axis=-1).astype(np.float32)


def noise_streams(root: str, mean: np.ndarray, std: np.ndarray) -> dict:
    """{split_i: standardised MFCC frames of recording i in the split, split_i_pow: frame power}."""
    d = os.path.join(root, NOISE_DIR)
    wavs = [read_wav(os.path.join(d, n)) for n in sorted(os.listdir(d)) if n.endswith(".wav")]
    out = {}
    for split, a, b in SPLIT_REGIONS:
        for i, w in enumerate(wavs):
            lo, hi = int(a * w.size), int(b * w.size)
            starts = range(lo, hi - SR + 1, SR)
            frames = np.concatenate([mfcc(w[s:s + SR]) for s in starts])
            out[f"{split}_{i}"] = ((frames - mean) / std).astype(np.float32)
            out[f"{split}_{i}_pow"] = np.concatenate([frame_power(w[s:s + SR]) for s in starts])
    return out


def write_noise_streams(root: str, data_dir: str) -> None:
    with np.load(os.path.join(data_dir, "digits.npz")) as f:
        mean, std = f["mean"], f["std"]
    streams = noise_streams(root, mean, std)
    np.savez(os.path.join(data_dir, "noise_streams.npz"), **streams)
    for split, _, _ in SPLIT_REGIONS:
        n = [v.shape[0] for k, v in streams.items()
             if k.startswith(split + "_") and not k.endswith("_pow")]
        print(f"noise stream {split}: {len(n)} recordings, {sum(n)} frames {n}", flush=True)


def build(root: str) -> dict:
    """{split: (features [N, T, F], labels [N])} before standardisation."""
    lists = {}
    for split, fname in (("val", "validation_list.txt"), ("test", "testing_list.txt")):
        with open(os.path.join(root, fname)) as f:
            lists[split] = {line.strip() for line in f if line.strip()}
    feats = {s: [] for s in ("train", "val", "test")}
    labels = {s: [] for s in ("train", "val", "test")}
    for c, word in enumerate(DIGITS):
        for name in sorted(os.listdir(os.path.join(root, word))):
            rel = f"{word}/{name}"
            split = "val" if rel in lists["val"] else "test" if rel in lists["test"] else "train"
            feats[split].append(mfcc(fit_length(read_wav(os.path.join(root, rel)))))
            labels[split].append(c)
        print(f"  {word}: done", flush=True)
    targets = {s: round(len(labels[s]) / len(DIGITS)) for s in labels}
    for split, wins in noise_windows(root, targets).items():
        feats[split] += [mfcc(w) for w in wins]
        labels[split] += [len(DIGITS)] * len(wins)
    return {s: (np.stack(feats[s]), np.asarray(labels[s], np.int32)) for s in feats}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default=DATA_DIR, help=f"output directory (default {DATA_DIR})")
    p.add_argument("--archive", help="local speech_commands_v0.02.tar.gz (default: download it)")
    p.add_argument("--noise-streams-only", action="store_true",
                   help="write noise_streams.npz only, from an existing digits.npz")
    args = p.parse_args()

    raw = os.path.join(args.data, "speech_commands_v0.02")
    if args.noise_streams_only:
        write_noise_streams(raw, args.data)
        return
    os.makedirs(raw, exist_ok=True)
    if not os.path.isdir(os.path.join(raw, NOISE_DIR)):
        archive = args.archive or os.path.join(args.data, os.path.basename(URL))
        if not os.path.exists(archive):
            print(f"downloading {URL}", flush=True)
            urllib.request.urlretrieve(URL, archive)
        print(f"extracting digits and noise from {archive}", flush=True)
        extract(archive, raw)
    data = build(raw)
    x_tr = data["train"][0]
    mean, std = x_tr.mean(axis=(0, 1)), x_tr.std(axis=(0, 1)) + 1e-6
    arrays = {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
              "classes": np.asarray(CLASSES)}
    counts = {}
    for split, (x, y) in data.items():
        arrays[f"x_{split}"] = ((x - mean) / std).astype(np.float32)
        arrays[f"y_{split}"] = y
        counts[split] = np.bincount(y, minlength=len(CLASSES)).tolist()
    out = os.path.join(args.data, "digits.npz")
    np.savez(out, **arrays)
    with open(os.path.join(args.data, "digits_counts.json"), "w") as f:
        json.dump({"classes": list(CLASSES), "counts": counts}, f, indent=1)
    for split, c in counts.items():
        print(f"{split}: {sum(c)} clips  per class {c}", flush=True)
    print(f"wrote {out}", flush=True)
    write_noise_streams(raw, args.data)


if __name__ == "__main__":
    main()
