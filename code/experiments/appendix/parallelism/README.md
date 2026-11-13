# Scan drift

The parallel scan against its sequential reference, on the hidden state of the affine baselines.
This produces the figure of the draft's Appendix "Sequential versus parallel evaluation": the
order of the operations alone is enough to change the trajectory, and how the gap then evolves
follows the spectral radius of the recurrence.

`drift.py` compares the two scans on the hidden state, step by step, for Mamba, AU-SSM,
PD-SSM, minGRU and DeltaProduct (n_h = 2, 8 heads; n_h = 1 and 4 as `deltaproduct-nh1|nh4`) at width 32,
and for the NFSM (one head of 32 states), whose merge composes tables exactly, so e_t = 0 at every step;
the figure draws it on a zero strip under the log axis.
One-head DeltaProduct runs (`deltaproduct[-nh1|-nh4]-h1`, one head of d_k = 32) keep the eigenvalue 1
for every word, since the 5 n_h keys of the 5 symbols span at most 5 n_h < 32 directions; the figure
draws `deltaproduct-h1` as DeltaProduct (`drift_plots.py --deltaproduct` picks another run). In the
trained layout (8 heads of d_k = 4) the keys span each head and the products contract. The one-head
runs were made on CPU (`platform` in the JSON): `--batch-chunk 4` keeps them in memory. Each job runs one baseline, one number format (float32 or float16) and one
seed, over a batch of 64 words of length 2^15, with the seed setting both the init and the batch.
Both scans run on the same drives (a_t, b_t), which are computed in float32 and rounded to the
scan dtype, so the two scans differ only in the order of their operations. The recorded error is
e_t = max_i |h_t^par[i] − h_t^seq[i]|: its batch mean, median, min and max at every t, and the
batch mean of max_i |h_t^seq[i]|.

```bash
bash scripts/dispatch_drift.sh [--dry-run] [--models ...] [--dtypes ...] [--seeds ...]
python -m experiments.appendix.parallelism.drift_plots [--band seeds|words] [--logx]
```

The JSONs are in `results/parallelism/drift/<dtype>/<model>_seed<seed>.json`. With both dtypes, the
figure is `results/parallelism/drift/figures/drift_combined.pdf`, the one used in the draft; with
`--dtypes` naming one dtype, it is `drift_<dtype>.pdf`. The plotting script needs only
numpy and matplotlib, and uses LaTeX when `latex` is on the PATH.

- There is no complex float16, so the float16 runs of AU-SSM and PD-SSM hold the real and
  imaginary parts as two float16 arrays (`complex_emulated` in the JSON).
- The dispatcher sets `--xla_gpu_deterministic_ops=true`. Without it, PD-SSM's scatter-add would
  also differ from run to run.
