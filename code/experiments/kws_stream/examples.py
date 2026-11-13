"""Audio examples of the stream task, built from waveforms like data.Streams.

    python -m experiments.kws_stream.examples --out DIR [--n 4] [--lens 101 400 1600] [--split test]

Writes DIR/<name>.wav, an MFCC picture DIR/<name>.png and DIR/index.md.
"""

import argparse
import os
import wave

import numpy as np

from experiments.kws import prepare
from experiments.kws import tasks as kws
from experiments.kws_stream import tasks

HOP = 160


def crossfade(a, b, k):
    k = min(k, len(a), len(b))
    w = np.arange(1, k + 1) / (k + 1)
    return np.concatenate([a[:-k], (1 - w) * a[-k:] + w * b[:k], b[k:]])


def fade():
    return HOP * int(RNG.integers(tasks.FADE_FRAMES[0], tasks.FADE_FRAMES[1] + 1))


def noise_audio(regions, n):
    out = None
    while out is None or len(out) < n:
        rec = regions[RNG.integers(len(regions))]
        m = min(HOP * int(RNG.integers(tasks.CROP_FRAMES[0], tasks.CROP_FRAMES[1] + 1)), len(rec))
        s = int(RNG.integers(0, len(rec) - m + 1))
        piece = rec[s:s + m] * 10 ** ((RNG.uniform(*tasks.NOISE_DBFS)
                                       - 10 * np.log10(np.mean(rec[s:s + m] ** 2) + 1e-12)) / 20)
        out = piece if out is None else crossfade(out, piece, fade())
    return out[:n]


def word_list(raw, split):
    lists = {}
    for sp, fname in (("val", "validation_list.txt"), ("test", "testing_list.txt")):
        with open(os.path.join(raw, fname)) as f:
            lists[sp] = {line.strip() for line in f if line.strip()}
    out = []
    for c, word in enumerate(tasks.CLASSES[:10]):
        for name in sorted(os.listdir(os.path.join(raw, word))):
            rel = f"{word}/{name}"
            sp = "val" if rel in lists["val"] else "test" if rel in lists["test"] else "train"
            if sp == split:
                out.append((rel, c))
    return out


def write_wav(path, x):
    with wave.open(path, "wb") as f:
        f.setnchannels(1), f.setsampwidth(2), f.setframerate(kws.SR)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def picture(path, x, span, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chunks = [prepare.mfcc(prepare.fit_length(x[s:s + kws.SR])) for s in range(0, len(x), kws.SR)]
    m = np.concatenate(chunks)[: len(x) // HOP + 1]
    fig, ax = plt.subplots(figsize=(min(14, 2 + len(m) / 60), 2.4))
    ax.imshow(m.T, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    if span:
        for e in span:
            ax.axvline(e, color="cyan", lw=1)
    ax.set_xlabel("frame (10 ms)")
    ax.set_ylabel("MFCC")
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    global RNG
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=4, help="examples per length")
    p.add_argument("--lens", type=int, nargs="+", default=[101, 400, 1600])
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    RNG = np.random.default_rng(args.seed)
    raw = os.path.join(kws.DATA_DIR, "speech_commands_v0.02")
    nd = os.path.join(raw, prepare.NOISE_DIR)
    wavs = [prepare.read_wav(os.path.join(nd, n)) for n in sorted(os.listdir(nd)) if n.endswith(".wav")]
    a, b = {s: (lo, hi) for s, lo, hi in prepare.SPLIT_REGIONS}[args.split]
    regions = [w[int(a * w.size):int(b * w.size)] for w in wavs]
    words = word_list(raw, args.split)
    os.makedirs(args.out, exist_ok=True)
    lines = ["| file | frames | seconds | label | word starts at (s) | noise after the word (s) |",
             "|---|---:|---:|---|---:|---:|"]
    for T in args.lens:
        for i in range(args.n):
            n = T * HOP
            x = noise_audio(regions, n)
            none = i == args.n - 1 and args.n > 1           # the last example per length has no word
            name = f"T{T}_{i}"
            if none:
                label, start, hold, span = "none", "-", "-", None
            else:
                rel, c = words[RNG.integers(len(words))]
                w = prepare.fit_length(prepare.read_wav(os.path.join(raw, rel)))
                pos = HOP * int(RNG.integers(0, T - tasks.WORD + 1))
                seg = w.copy()
                k1, k2 = fade(), fade()
                r1, r2 = np.arange(1, k1 + 1) / (k1 + 1), np.arange(1, k2 + 1) / (k2 + 1)
                seg[:k1] = (1 - r1) * x[pos:pos + k1] + r1 * w[:k1]
                e = pos + kws.SR
                seg[-k2:] = (1 - r2) * w[-k2:] + r2 * x[e - k2:e]
                x[pos:e] = seg[: len(x[pos:e])]
                label, start = tasks.CLASSES[c], f"{pos / kws.SR:.2f}"
                hold, span = f"{(n - e) / kws.SR:.2f}", (pos // HOP, e // HOP)
                name += f"_{label}"
            write_wav(os.path.join(args.out, name + ".wav"), x)
            picture(os.path.join(args.out, name + ".png"), x, span,
                    f"{name}: label {label}, {T} frames")
            lines.append(f"| {name}.wav | {T} | {T / 100:.2f} | {label} | {start} | {hold} |")
    with open(os.path.join(args.out, "index.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
