#!/usr/bin/env bash
# Evaluate final checkpoints produced by run_hl_readout_sweep.sh.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

seed="${SEED:-1}"
epoch="${EPOCH:-19}"
split="${SPLIT:-test_hard}"

evaluate_condition() {
    local schedule="$1"
    local readout="$2"
    local checkpoint="checkpoints/hl_readout_sweep/${schedule}_${readout}/seed_${seed}/epoch_${epoch}.pt"

    if [[ ! -f "$checkpoint" ]]; then
        echo "Missing checkpoint: $checkpoint" >&2
        return 1
    fi

    echo "==> ${schedule}_${readout}"
    python eval.py --ckpt "$checkpoint" --split "$split"
}

evaluate_condition H2L6 h
for schedule in H1L16 H2L8 H4L4 H8L2 H16L1; do
    for readout in h l hl; do
        evaluate_condition "$schedule" "$readout"
    done
done
