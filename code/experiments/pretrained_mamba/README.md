# Pretrained Mamba-2 on RULER

Released Mamba-2 checkpoints (`state-spaces/mamba2-{130m,370m,780m}`) on RULER `niah_single_1`, as
released and with the decay bounded by τ (Section 7.2 and Appendix I). Our own forward pass in JAX,
float32, in this folder's environment. `MAMBA2_SIZE` picks the model (130m by default); results go to
`results/` for 130M and to `results/mamba2_<size>/` for the others.

## Files

| file | content |
|---|---|
| `mamba2_jax.py` | forward pass (chunked SSD scan, greedy decoding), weight loading, decay bound |
| `niah.py` | prompts: needles, haystacks of 25-token lines, layouts A, B and C |
| `eval_niah.py` | accuracy over the (L_pre, L_post) grid for one layout and one τ |
| `merge_parts.py` | joins the per-haystack files of a cluster run |
| `perplexity.py` | WikiText-2 perplexity per τ |
| `floor_stats.py` | how often the bound is active, per layer and head |
| `analyze.py`, `plotting.py` | tables and figures, every size found |
| `checks.py` | consistency checks of the forward pass |
| `run_all.sh` | 130M, layouts A and B, on one GPU |
| `slurm_grid.sh` | one array job per size: layouts A, B, C × (as released and 5 values of τ) × 10 haystacks |

## Run

```bash
cd experiments/pretrained_mamba
uv sync
./run_all.sh                                   # 130M, about 1.5 h on an Apple M5 Max GPU
MAMBA2_SIZE=780m sbatch --array=0-179 slurm_grid.sh
uv run python merge_parts.py --out results/mamba2_780m/niah_A_asis.json results/mamba2_780m/parts/niah_A_asis_h*.json
uv run python analyze.py                       # results/tables.md and the figures
```

The first run downloads the checkpoint, the GPT-NeoX-20B tokenizer and WikiText-2 from the Hugging Face
Hub, and caches the converted weights in `.cache/`. `JAX_PLATFORMS=cpu` runs on the CPU.

## Setup

- 100 needles ("One of the special magic numbers for KEY is: VALUE.") over 10 sampled haystacks, the
  same at every point. Each haystack line is a random permutation of the five RULER noise sentences.
- Layouts: A puts the instruction first, B right before the needle, C has none.
- L_pre ∈ {0, 25, 2050, 8200, 32775, 131075} tokens before the needle, L_post ∈ {0, 25, 50, 100, 250, 500,
  750, 1025, 2050, 3075, 4100, 8200} between the needle and the question.
- Greedy decoding of 16 tokens; a needle is retrieved when the output contains its value.
- Decay bound: Δ_t A is replaced by min(Δ_t A, −1/τ) in the state transition only, in every layer, head
  and token. The weights are unchanged.
- Perplexity: the first 8 × 1,024 tokens of the WikiText-2 test split, in independent segments.

Against Hugging Face `Mamba2ForCausalLM` (130M), the largest logit difference is 4e-4, with the same
argmax at every position.
