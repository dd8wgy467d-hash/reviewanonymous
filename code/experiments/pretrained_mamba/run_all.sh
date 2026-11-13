#!/usr/bin/env bash
# Full grid: 2 layouts x 6 variants, 100 needles over 10 sampled haystacks, then perplexity, floor
# statistics, tables and figures. About 3 h on an Apple M5 Max GPU (Metal, jax-mps).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
uv run python checks.py
for layout in A B; do
  for tau in inf 4096 2048 1024 512 128; do
    name=$([ "$tau" = inf ] && echo asis || echo "tau$tau")
    out="results/niah_${layout}_${name}.json"
    [ -f "$out" ] && { echo "skip $out (exists)"; continue; }
    uv run python eval_niah.py --layout "$layout" --tau "$tau" --out "$out" > "results/log_niah_${layout}_${name}.txt" 2>&1
  done
done
uv run python perplexity.py > results/log_perplexity.txt 2>&1
uv run python floor_stats.py > results/log_floor_stats.txt 2>&1
uv run python analyze.py > results/log_analyze.txt 2>&1
