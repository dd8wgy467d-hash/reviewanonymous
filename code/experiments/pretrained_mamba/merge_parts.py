"""Join the per-haystack parts of one variant into one result file.

    uv run python merge_parts.py --out results/mamba2_370m/niah_A_asis.json results/mamba2_370m/parts/niah_A_asis_h*.json

--into FILE adds them to an existing file with other d values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SAME = ("layout", "tau", "d", "L", "n_prompts", "n_haystacks", "seed", "n_generated_tokens", "model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--into", type=Path, default=None, help="existing result file to add the parts to")
    ap.add_argument("parts", type=Path, nargs="+")
    args = ap.parse_args()
    parts = [json.loads(p.read_text()) for p in args.parts]
    meta = dict(parts[0]["meta"])
    for p in parts:
        for k in SAME:
            assert p["meta"][k] == meta[k], f"parts differ in {k}"
    hs = sorted(h for p in parts for h in p["meta"]["haystacks"])
    assert hs == list(range(meta["n_haystacks"])), f"haystacks {hs}: missing or repeated"
    meta["haystacks"] = hs
    meta["seconds"] = sum(p["meta"]["seconds"] for p in parts)
    records = [r for p in parts for r in p["records"]]
    if args.into is not None:
        old = json.loads(args.into.read_text())
        for k in SAME:
            if k != "d":
                assert old["meta"].get(k, meta[k]) == meta[k], f"cannot add: {k} differs"
        assert not set(old["meta"]["d"]) & set(meta["d"]), "d values already in the file"
        records += old["records"]
        meta["d"] = sorted(set(old["meta"]["d"]) | set(meta["d"]))
        meta["seconds"] += old["meta"]["seconds"]
    records.sort(key=lambda r: (r["prompt"], r["d"], r["L"]))
    args.out.write_text(json.dumps({"meta": meta, "records": records}, indent=1))
    print(f"{args.out}: {len(records)} records from {len(parts)} parts")


if __name__ == "__main__":
    main()
