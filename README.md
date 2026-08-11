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

## Easy-train / Extreme-test mechanism experiment

This setting trains on 32--40-given, uniquely-solvable Sudoku boards and evaluates on the fixed Sudoku-Extreme `test_hard` split. The HRM and RT architectures are unchanged from the tuned configurations.

Before optimizer step 0, rank 0 deterministically builds a bank of 10,000 independently generated solved boards and uniquely-solvable puzzle masks, then saves it under `downloaded-datasets/`. Other ranks wait for and load the same bank. Training draws a base puzzle from the bank and applies a fresh Sudoku-preserving symmetry each time; this preserves uniqueness while avoiding the narrow single-solution-family distribution and per-sample backtracking cost. The bank filename contains the seed and generation settings, so it is reused safely across restarts and shared by HRM and RT for the same seed.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name easy_to_hard_hrm
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 torchrun --nproc-per-node 8 train.py --config-name easy_to_hard_rt
```

To evaluate a checkpoint at multiple external rollout lengths, use its named hard evaluation:

```bash
python eval.py --ckpt checkpoints/<run>/seed_1/epoch_19.pt --eval-name hard --rollout-cycles 1 2 4 8 16
```

Each call writes an `eval_result_<checkpoint>_hard.npz` file next to the checkpoint. Aggregate the three seeds for each model and create the comparison curve with:

```bash
python plot_easy_to_hard.py \
  --hrm checkpoints/<hrm-run>/seed_{1,2,3}/eval_result_epoch_19_hard.npz \
  --rt checkpoints/<rt-run>/seed_{1,2,3}/eval_result_epoch_19_hard.npz \
  --output figures/easy_to_hard_rollouts.png
```

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
