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

Download the official full Sudoku-Extreme training CSV before running this setting:

```bash
hf download --repo-type dataset --local-dir ./downloaded-datasets/sudoku-extreme-full sapientinc/sudoku-extreme train.csv
```

Easy and Medium each select 1,000 base puzzles from the official train split, then retain the original 200 repeats and online Sudoku symmetry augmentation. All three groups use the same original `sudoku-extreme-1k/test_hard` evaluation split.

```bash
python summarize_sudoku_ratings.py --dataset ./downloaded-datasets/sudoku-extreme-full --split train
python summarize_sudoku_ratings.py --dataset ./downloaded-datasets/sudoku-extreme-1k --split test_hard

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name easy_to_hard_hrm
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name easy_to_hard_rt

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name medium_to_hard_hrm
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name medium_to_hard_rt
```

The Easy configurations use `0 <= rating <= 15`; Medium uses `16 <= rating <= 30`. Extreme is the original experiment and should run unchanged with `tuned_hrm` / `tuned_rt`.

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
