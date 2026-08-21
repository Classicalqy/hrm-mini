#!/usr/bin/env bash
# Train the H=2, L-cycle x readout sweep plus a compute-matched RT baseline.
#
# The defaults intentionally create a fresh checkpoint tree:
#   checkpoints/h2_l_readout_rt_sweep/H2L{L}_{h|l|hl}/seed_{1|2|3}/epoch_*.pt
#   checkpoints/h2_l_readout_rt_sweep/RT/seed_{1|2|3}/epoch_*.pt
#
# Override NPROC_PER_NODE, GROUP_ROOT, or SEEDS (comma-separated) when needed.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source /home/chenguozhang/miniconda3/etc/profile.d/conda.sh
conda activate hrm
nproc_per_node="${NPROC_PER_NODE:-8}"
group_root="${GROUP_ROOT:-h2_l_readout_rt_sweep}"
seeds_csv="${SEEDS:-1,2,3}"
dry_run=false
l_values=(6 8 16 32)
readouts=(h l hl)

if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

run_condition() {
    local l_cycles="$1"
    local readout="$2"
    local schedule="H2L${l_cycles}"
    local group_name="${group_root}/${schedule}_${readout}"

    echo "==> ${group_name}"
    if "$dry_run"; then
        # Validate the exact Hydra composition without importing train.py runtime dependencies.
        python - "$project_root" "$seeds_csv" "$l_cycles" "$readout" <<'PY'
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir

project_root, seeds_csv, l_cycles, readout = sys.argv[1:]
seeds = [int(seed) for seed in seeds_csv.split(",")]
with initialize_config_dir(version_base=None, config_dir=str(Path(project_root) / "config")):
    config = compose(
        config_name="tuned_hrm",
        overrides=[
            f"seeds=[{seeds_csv}]",
            "arch.H_cycles=2",
            f"arch.L_cycles={l_cycles}",
            f"arch.readout={readout}",
        ],
    )
assert config.arch.H_cycles == 2
assert config.arch.L_cycles == int(l_cycles)
assert config.arch.readout == readout
assert config.seeds == seeds
PY
    else
        export MLP_TASK_NAME="$group_name"
        WANDB_MODE=offline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            torchrun --nproc-per-node "$nproc_per_node" train.py --config-name tuned_hrm \
            "seeds=[$seeds_csv]" \
            "arch.H_cycles=2" \
            "arch.L_cycles=$l_cycles" \
            "arch.readout=$readout"
    fi
}

run_rt_baseline() {
    local group_name="${group_root}/RT"

    echo "==> ${group_name} (original tuned_rt baseline)"
    if "$dry_run"; then
        python - "$project_root" "$seeds_csv" <<'PY'
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir

project_root, seeds_csv = sys.argv[1:]
seeds = [int(seed) for seed in seeds_csv.split(",")]
with initialize_config_dir(version_base=None, config_dir=str(Path(project_root) / "config")):
    config = compose(config_name="tuned_rt", overrides=[f"seeds=[{seeds_csv}]"])

assert config.arch.name == "rt@RecurrentTransformer"
assert config.arch.num_layers == 4
assert config.arch.cycles == 7
assert config.seeds == seeds
PY
    else
        export MLP_TASK_NAME="$group_name"
        WANDB_MODE=offline OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            torchrun --nproc-per-node "$nproc_per_node" train.py --config-name tuned_rt \
            "seeds=[$seeds_csv]"
    fi
}

for l_cycles in "${l_values[@]}"; do
    for readout in "${readouts[@]}"; do
        run_condition "$l_cycles" "$readout"
    done
done

run_rt_baseline
