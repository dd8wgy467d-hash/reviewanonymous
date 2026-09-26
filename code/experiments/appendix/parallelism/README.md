# Scan drift

The parallel scan against its sequential reference, on the hidden state of the affine baselines.
This produces the figure of the draft's Appendix "Sequential versus parallel evaluation": the
order of the operations alone is enough to change the trajectory, and how the gap then evolves
follows the spectral radius of the recurrence.

`drift.py` compares the two scans on the hidden state, step by step, for Mamba, AU-SSM and
PD-SSM at width 32. Each job runs one baseline, one number format (float32 or float16) and one
seed, over a batch of 64 words of length 2^15, with the seed setting both the init and the batch.
Both scans run on the same drives (a_t, b_t), which are computed in float32 and rounded to the
scan dtype, so the two scans differ only in the order of their operations. The recorded error is
e_t = max_i |h_t^par[i] − h_t^seq[i]|: its batch mean, median, min and max at every t, and the
batch mean of max_i |h_t^seq[i]|.

```bash
bash scripts/dispatch_drift.sh [--dry-run] [--models ...] [--dtypes ...] [--seeds ...]
python -m experiments.appendix.parallelism.drift_plots [--band seeds|words] [--logx]
```

The JSONs are in `results/parallelism/drift/<dtype>/<model>_seed<seed>.json`, and the figures are
in `results/parallelism/drift/figures/drift_<dtype>.pdf`. With both dtypes, `drift_combined.pdf`
puts them on the same axes; that is the figure used in the draft. The plotting script needs only
numpy and matplotlib, and uses LaTeX when `latex` is on the PATH.

- There is no complex float16, so the float16 runs of AU-SSM and PD-SSM hold the real and
  imaginary parts as two float16 arrays (`complex_emulated` in the JSON).
- The dispatcher sets `--xla_gpu_deterministic_ops=true`. Without it, PD-SSM's scatter-add would
  also differ from run to run.
