#!/usr/bin/env bash
# Train the H/L schedule x readout sweep sequentially.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

nproc_per_node="${NPROC_PER_NODE:-8}"
seed="${SEED:-1}"
dry_run=false

if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--dry-run]" >&2
    exit 2
fi

run_condition() {
    local schedule="$1"
    local h_cycles="$2"
    local l_cycles="$3"
    local readout="$4"
    local group_name="hl_readout_sweep/${schedule}_${readout}"
    local command=(
        torchrun --nproc-per-node "$nproc_per_node" train.py --config-name tuned_hrm
        "seeds=[$seed]"
        "arch.H_cycles=$h_cycles"
        "arch.L_cycles=$l_cycles"
        "arch.readout=$readout"
    )

    echo "==> ${group_name}"
    if "$dry_run"; then
        # Compose the same Hydra overrides without importing train.py and its runtime-only dependencies.
        python - "$project_root" "$seed" "$h_cycles" "$l_cycles" "$readout" <<'PY'
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir

project_root, seed, h_cycles, l_cycles, readout = sys.argv[1:]
with initialize_config_dir(version_base=None, config_dir=str(Path(project_root) / "config")):
    config = compose(
        config_name="tuned_hrm",
        overrides=[
            f"seeds=[{seed}]",
            f"arch.H_cycles={h_cycles}",
            f"arch.L_cycles={l_cycles}",
            f"arch.readout={readout}",
        ],
    )
assert config.arch.H_cycles == int(h_cycles)
assert config.arch.L_cycles == int(l_cycles)
assert config.arch.readout == readout
assert config.seeds == [int(seed)]
PY
    else
        MLP_TASK_NAME="$group_name" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${command[@]}"
    fi
}

# Reproducible tuned-HRM reference condition.
run_condition "H2L6" 2 6 h

for schedule_spec in "H1L16 1 16" "H2L8 2 8" "H4L4 4 4" "H8L2 8 2" "H16L1 16 1"; do
    read -r schedule h_cycles l_cycles <<< "$schedule_spec"
    for readout in h l hl; do
        run_condition "$schedule" "$h_cycles" "$l_cycles" "$readout"
    done
done
