# HRM-Mini

Minimalistic implementation of Hierarchical Recurrent Model (HRM).

## Install requirements

Ensure Python and PyTorch is installed, and your machine have at least 1 GPU and total 40 GiB VRAM. Then install pip dependencies, it should be done in 10 minutes:

```bash
pip install -r requirements.txt
```

## W&B Integration

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Download datasets

The following commands pulls the required datasets from HuggingFace repositories.

```bash
mkdir downloaded-datasets
hf download --repo-type dataset --local-dir ./downloaded-datasets/maze-30x30-hard-1k sapientinc/maze-30x30-hard-1k
hf download --repo-type dataset --local-dir ./downloaded-datasets/sudoku-extreme-1k sapientinc/sudoku-extreme-1k
```

## Download checkpoints (optional)

Run the commands below to load trained Sudoku checkpoint for the dynamics analysis.

```bash
hf download --repo-type model --local-dir ./checkpoints/1000_tuned_hrm_new cl-agi/hrm-mini
```

## Note: Running on a single GPU

The original experiments run on one node with 8 H100 GPUs. Sudoku takes about 30 minutes to run. If you want to run on a single GPU, set `--nproc-per-node 1` in the command line. Also multiply local batch size by 8, e.g. `local_batch_size=768`. Sudoku will take ~4 hours per experiment on a single H100. Besides, the script by default runs 3 seeds, append `seeds=[1]` to run a single seed.

## Launch main experiment

Sudoku-Extreme 1000 examples. It should take about 4 GPU*hours for H100 (~30 min for 8 H100 GPUs, ~4 hr for 1 H100 GPU).

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name tuned_hrm
```

## Ablation studies

HRM Full: See above

Recurrent Transformer

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name tuned_rt
```

No dual timescale

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc_per_node 8 train.py --config-name tuned_hrm arch.name=hrm_ablations@HRM arch.L_cycles=1 arch.H_cycles=7
```

Tied H-L parameters (TRM-style)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc_per_node 8 train.py --config-name tuned_hrm arch.name=hrm_ablations@HRM +arch.dual_module=False
```

No H-H links

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc_per_node 8 train.py --config-name tuned_hrm arch.name=hrm_ablations@HRM +arch.hh_link=False
```

MLP Mixer

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc_per_node 8 train.py --config-name tuned_hrm +arch.is_mlp_mixer=True
```

### H/L schedule × readout sweep

This repository includes a single-seed, step-matched Sudoku-Extreme sweep over
`H1L16`, `H2L8`, `H4L4`, `H8L2`, and `H16L1`, with `h`, `l`, and `hl`
(concatenated H+L) readouts. It also trains the `H2L6` + `h` baseline. Runs are
sequential and use deterministic checkpoint groups under `checkpoints/hl_readout_sweep/`.

```bash
# Preview the 16 Hydra configurations without training.
scripts/run_hl_readout_sweep.sh --dry-run

# Run on 8 GPUs (set NPROC_PER_NODE=1 for a single GPU).
scripts/run_hl_readout_sweep.sh

# Evaluate final checkpoints and print the 6 x 3 exact-match table.
scripts/evaluate_hl_readout_sweep.sh
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh && conda activate daily
python scripts/summarize_hl_readout_sweep.py
```

All conditions retain the tuned HRM hyperparameters and identical optimizer-step/data
budgets. Their wall-clock time and total FLOPs differ because the number of H updates
per forward pass differs.

### Three-seed H=2 L/readout sweep with RT baseline

The reasoning-dynamics sweep trains `H2L{1,2,3,4,6,8,16,32}` with each of `h`,
`l`, and `hl` readouts, plus the original compute-matched `tuned_rt` baseline. Each
of the 25 conditions runs seeds `1,2,3` sequentially, retaining the default 20 epochs,
`cycles_per_data=16`, data configuration, and optimizer hyperparameters. Every epoch
is checkpointed; rank 0 is the sole checkpoint writer, so distributed ranks cannot
race on the same file.

```bash
# Validate all 25 Hydra configurations without training.
scripts/run_h2_l_readout_sweep.sh --dry-run

# Train sequentially on 8 GPUs. Checkpoints use a new root so existing results remain intact.
scripts/run_h2_l_readout_sweep.sh
```

By default this writes `epoch_0.pt` through `epoch_19.pt` and `model_config.json` to
`checkpoints/h2_l_readout_rt_sweep/{H2L*_h|H2L*_l|H2L*_hl|RT}/seed_{1,2,3}/`.
Use `NPROC_PER_NODE`, `GROUP_ROOT`, or comma-separated `SEEDS` as environment
variables to override the launch defaults.

### H/L effective-diffusion analysis

After the H=2 sweep has produced its final seed-1 checkpoints, collect fixed-parameter
latent reasoning trajectories for `H2L6-H`, `H2L1-H`, `H2L6-HL`, and `H2L6-L`:

```bash
python scripts/analyze_hl_diffusion.py --samples 1024
```

The command writes compressed trajectories, CSV summaries, and five PNG/PDF figure
groups under `results/diffusion/`. Use `--analyze-only` to regenerate metrics and
figures from existing trajectory files without rerunning GPU inference.

## Dynamics and Visualization

Install Jupyter and load `visualizations.ipynb`. If you want to evaluate other checkpoint, change the checkpoint path in the first cell. It should take several minutes.

## Other tasks

Maze 30x30

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name tuned_hrm data=maze
```

For 3-SAT, please switch to `SAT` branch to train.

## Docker support (optional)

We use this docker image for experiments. You can use this image for exact reproducing.

You can check the exact software version in this image.

```bash
docker pull sapientai/pytorch-docker:26.02.14.hopper
```
