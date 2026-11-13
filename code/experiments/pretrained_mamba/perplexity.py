"""Perplexity of each variant on the first N_SEG x 1024 tokens of the WikiText-2 test split,
in independent segments.

    uv run python perplexity.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

import mamba2_jax as m

N_SEG = 8
VARIANTS = {"asis": math.inf, "tau4096": 4096.0, "tau2048": 2048.0, "tau1024": 1024.0, "tau512": 512.0, "tau128": 128.0}


def main():
    params = m.load_params()
    tok = m.load_tokenizer()
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                           repo_type="dataset")
    text = "".join(pq.read_table(path).column("text").to_pylist())
    ids = np.array(tok.encode(text).ids[:N_SEG * m.SEGMENT]).reshape(N_SEG, m.SEGMENT)

    out = {"text": "WikiText-2 raw test split, first tokens", "n_segments": N_SEG,
           "segment": m.SEGMENT, "n_predicted_tokens": int(N_SEG * (m.SEGMENT - 1)), "ppl": {}}
    for name, tau in VARIANTS.items():
        # one segment at a time: on the Metal backend, all-position logits are wrong at batch >= 4
        # (identical rows differ); batch 1 matches the CPU (checks.py)
        nlls = []
        for row in ids:
            logits, _ = m.run(params, row[None], m.init_state(1), tau, all_logits=True)
            logits = np.asarray(logits[0, :-1], np.float64)
            mx = logits.max(-1, keepdims=True)
            logp = logits - mx - np.log(np.exp(logits - mx).sum(-1, keepdims=True))
            nlls.append(-np.take_along_axis(logp, row[1:, None], axis=-1)[:, 0])
        nll = np.concatenate(nlls).mean()
        out["ppl"][name] = round(float(np.exp(nll)), 3)
        print(f"{name:8s} ppl {np.exp(nll):.3f}", flush=True)
    dst = m.RESULTS / "perplexity.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
