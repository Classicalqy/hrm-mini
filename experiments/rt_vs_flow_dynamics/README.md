# RT versus Flow dynamics

This package tests two separate claims: whether a prescribed flow-matching
interpolant restricts learned trajectories, and whether a recurrent Transformer
learns rapidly contracting but incorrect inference dynamics.  It deliberately uses
the term **wrong attractor/fixed point**, not parameter-space local minimum.

The controlled track gives RT and conditional Flow Matching the same Transformer,
input embedding, fixed codebook decoder, parameter count, optimizer-update budget,
and backbone-call budget.  The observational track loads an unchanged native RT
checkpoint and exposes each of its recurrent micro-steps without modifying `arch/rt.py`.

## Quick validation

Activate the repository's Python environment and run the complete CPU pipeline:

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate daily
python -m experiments.rt_vs_flow_dynamics.cli smoke
```

The smoke run writes ignored checkpoints and results below
`checkpoints/rt_vs_flow_dynamics/smoke/` and
`results/rt_vs_flow_dynamics/smoke/`.

## Formal runs

```bash
python -m experiments.rt_vs_flow_dynamics.cli toy \
  --config experiments/rt_vs_flow_dynamics/configs/full.yaml

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 \
  -m experiments.rt_vs_flow_dynamics.cli train-sudoku \
  --config experiments/rt_vs_flow_dynamics/configs/full.yaml \
  --condition matched_rt flow_linear rt_smooth
```

Trace a controlled final checkpoint:

```bash
python -m experiments.rt_vs_flow_dynamics.cli trace-sudoku \
  --config experiments/rt_vs_flow_dynamics/configs/full.yaml \
  --checkpoint checkpoints/rt_vs_flow_dynamics/full/matched_rt/seed_1/epoch_19.pt \
  --perturb
```

Trace an existing native `tuned_rt` checkpoint by adding `--native`.  Native
checkpoints must be named under `seed_<n>/` and have the repository's
`model_config.json` beside them; the command refuses to choose a checkpoint on the
user's behalf.  It first checks that seven explicit micro-steps reproduce the native
RT forward result.

After tracing every seed/condition, aggregate them with:

```bash
python -m experiments.rt_vs_flow_dynamics.cli analyze \
  --config experiments/rt_vs_flow_dynamics/configs/full.yaml
```

## Interpretation

Flow NFE 56, 112, and 224 all integrate the same normalized interval `[0, 1]`.
Only RT is extended beyond its trained 112-step horizon.  Cross-model geometry is
computed from a shared random projection of output probabilities rather than raw
hidden coordinates.  A fixed point requires eight stable predictions with normalized
state residual below `1e-3` and mean probability change below `1e-4`; the analysis
also reports state thresholds `1e-2` and `1e-4`.

Different trajectory geometry is correlational evidence.  A causal claim requires
the RT curvature intervention or Flow interpolant intervention to change both the
geometry and wrong-attractor rate consistently across seeds.
