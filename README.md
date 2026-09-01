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

## Rated train / Extreme-test experiments

Download the official full Sudoku-Extreme train and held-out test CSVs before running this setting:

```bash
hf download --repo-type dataset --local-dir ./downloaded-datasets/sudoku-extreme-full sapientinc/sudoku-extreme train.csv test.csv
```

Easy and Medium each select 1,000 base puzzles from the official train split, then retain the original 200 repeats and online Sudoku symmetry augmentation. Difficulty is defined by the number of blank cells in the input: Easy has `<=53` blanks, Medium has `54-57`, and Hard has `>=58`. All groups evaluate on the disjoint official `sudoku-extreme/test` split and log `eval/easy_exact_match`, `eval/medium_exact_match`, and `eval/hard_exact_match` after every epoch.
Each metric uses a fixed 10,000-puzzle sample from its blank-count band, shared across all seeds and both architectures.

```bash
python summarize_sudoku_blanks.py --dataset ./downloaded-datasets/sudoku-extreme-full --split train
python summarize_sudoku_blanks.py --dataset ./downloaded-datasets/sudoku-extreme-full --split test

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name easy_to_hard_hrm
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name easy_to_hard_rt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name easy_to_hard_trm

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name medium_to_hard_hrm
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name medium_to_hard_rt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name medium_to_hard_trm

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name hard_to_hard_hrm
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name hard_to_hard_rt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name hard_to_hard_trm
```

The Easy configurations use `<=53` blanks (at least 28 givens); Medium uses `54-57` blanks (24-27 givens); Hard uses `>=58` blanks (at most 23 givens). Extreme is the original experiment and should run unchanged with `tuned_hrm` / `tuned_rt`.

`*_trm` uses the core Tiny Recursive Model structure: one shared two-layer Transformer alternately updates `z_L` and `z_H`. It intentionally excludes the official repository's puzzle-id embeddings and ACT/Q-learning halting so that it has the same input, loss, optimizer, training loop, and test protocol as HRM and RT. Its parameter count is therefore lower by design and must be reported alongside accuracy and recursive-call budget.

### Five-seed HRM/TRM comparison

The `*_five_seed` configurations run seeds 1–5 and send all runs to the W&B project `hrm-trm`. They preserve the three-seed configurations above unchanged. Launch all six HRM/TRM conditions with:

```bash
for band in easy medium hard; do
  for model in hrm trm; do
    MLP_TASK_NAME="${band}_${model}_five_seed" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nproc-per-node 8 train.py --config-name "${band}_to_hard_${model}_five_seed"
  done
done
```

Every epoch logs held-out Easy, Medium, and Hard exact-match accuracy for each seed.
Use `python cross_evaluate.py --models hrm trm ...` when aggregating this two-model sweep.

### Cross-evaluate Easy / Medium / Hard

`cross_evaluate.py` evaluates the final checkpoint of every HRM, RT, and TRM
run on all three blank-count bands of the held-out official `test` split. It requires
a complete, balanced model × train-band × seed matrix; it writes per-run CSV/
JSON, seed mean±sample-standard-deviation aggregates, per-model 3×3 matrices,
and per-cell correctness files. Use the same seeds for all nine model/training
conditions. The examples below show seed 1; repeat every line for seeds 2 and
3.

```bash
python cross_evaluate.py \
  --checkpoint hrm:easy:1=checkpoints/<easy-hrm-run>/seed_1/epoch_19.pt \
  --checkpoint hrm:medium:1=checkpoints/<medium-hrm-run>/seed_1/epoch_19.pt \
  --checkpoint hrm:hard:1=checkpoints/<hard-hrm-run>/seed_1/epoch_19.pt \
  --checkpoint rt:easy:1=checkpoints/<easy-rt-run>/seed_1/epoch_19.pt \
  --checkpoint rt:medium:1=checkpoints/<medium-rt-run>/seed_1/epoch_19.pt \
  --checkpoint rt:hard:1=checkpoints/<hard-rt-run>/seed_1/epoch_19.pt \
  --checkpoint trm:easy:1=checkpoints/<easy-trm-run>/seed_1/epoch_19.pt \
  --checkpoint trm:medium:1=checkpoints/<medium-trm-run>/seed_1/epoch_19.pt \
  --checkpoint trm:hard:1=checkpoints/<hard-trm-run>/seed_1/epoch_19.pt
```

Repeat the nine `--checkpoint` entries for seed 2 and seed 3.

The blank-count bands are `easy: <=53`, `medium: 54–57`, and `hard: >=58`, matching
the training configs. Each cell uses the same fixed 10,000 test puzzles
(`eval_seed=42`). Primary generalization results always use `epoch_19.pt`, the
checkpoint after the configured 20th epoch; do not select a checkpoint by its
hard-test curve. Per-epoch Easy/Medium/Hard metrics remain useful for plotting
training dynamics only. Results default to `results/cross_evaluation/`; the
script stops with a clear error if checkpoint architectures, final epochs,
seeds, or evaluation datasets do not form a comparable matrix.

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
