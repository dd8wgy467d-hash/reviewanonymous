# SSMs under finite precision

A numpy simulation of the 2×2 rotation SSM counting modulo N, with every operation rounded to
fp64, fp32, bf16, fp16, fp8-e4m3 or fp8-e5m2.

`finite_precision_counting.py` measures the phase drift of the unit rotation R(2π/N) and the step
at which the angular read-out fails, in two variants: only the matrix is quantized, or the whole
recurrence is. The first variant is the one reported in the draft (Appendix "Learning versus
finite precision"): the weights are the exact solution up to the resolution of the format, the
run itself is in fp64, and the model still fails at a finite length in every format.

```bash
bash scripts/dispatch_ssms_fp.sh [--dry-run] [--plot-only]                # one CPU job
python -m experiments.appendix.ssms_fp.finite_precision_counting [--plot-only]
```

Figures go to `results/ssms_fp/figures/` and CSVs to `results/ssms_fp/data/`. `--plot-only`
redraws the figures from the CSVs.
