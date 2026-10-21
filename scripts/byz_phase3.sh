#!/bin/bash
# Phase 3 rollouts, priority subset: matched triple x shift {none, mild, severe, mild_v1,
# severe_v1} x seeds 1-3, 50 episodes, plus a clean run per (seed, shift).
# Rules: analysis/phase3.py docstring.
#
#     bash scripts/byz_phase3.sh                 # own-seed Tier 3 member only (60 runs)
#     CROSS=1 bash scripts/byz_phase3.sh         # + cross-seed Tier 3 adversaries (90 runs)
#     DRY_RUN=1 bash scripts/byz_phase3.sh       # list the runs, collect nothing
#
# Needs first: results_v2/matched_triple_selection.json (scripts/byz_matched_triple.sh) and
# results_v2/shift_levels.json (scripts/byz_calibrate_shift.sh). Finished runs are skipped, so
# the script can be re-run after an interruption.
#
# Cost at ~17 episodes/min: 60 runs x 50 = 3000 episodes -> ~3 h; with CROSS=1 ~4.5 h.

set -uo pipefail
[[ -f scripts/byz_wandb_env.sh ]] && source scripts/byz_wandb_env.sh
PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python}
PARALLEL=${PARALLEL:-12}
R=${BYZ_RESULTS_ROOT:-byzantine_experiments/results_v2}
OUT=$R/phase3; LOGS=${BYZ_LOGS_ROOT:-logs_v2}/phase3
mkdir -p "$OUT" "$LOGS"

if [[ "${1:-}" == "--single" ]]; then
    IFS=$'\t' read -r name seed tier cfg idx shift <<< "$2"
    out="$OUT/$name.npz"; [[ -f "$out" ]] && { echo "SKIP  $name"; exit 0; }
    n=$(grep -oP '_n\K[0-9]+$' <<< "$name")
    if [[ "$tier" == "clean" ]]; then t=(--no-byzantine); else t=(--tier "$tier" --tier-config "$cfg" --compromised-indices "$idx"); fi
    echo "START $name"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -m byzantine_experiments.rollout.collect --checkpoint "train_dir/exp1_baseline_seed${seed}" \
        "${t[@]}" --shift "$shift" --n-episodes "$n" --seed "$seed" --device cpu --out "$out" \
        --wandb --wandb-group "phase3_${shift}" --wandb-name "$name" > "$LOGS/$name.log" 2>&1 \
        && echo "DONE  $name" || { echo "FAIL  $name -- $LOGS/$name.log"; exit 1; }
    exit 0
fi

JOBS=$LOGS/jobs.tsv
X=(); [[ "${CROSS:-0}" == "1" ]] && X=(--cross-seed)
"$PY" -m byzantine_experiments.analysis.phase3 jobs "${X[@]}" > "$JOBS" || exit 1
echo "Collecting $(wc -l < "$JOBS") runs, $PARALLEL at a time"
[[ "${DRY_RUN:-0}" == "1" ]] && { cut -f1,2,3,6 "$JOBS" | column -t; exit 0; }
export PY OUT LOGS
xargs -a "$JOBS" -d '\n' -P "$PARALLEL" -n 1 bash "$0" --single
"$PY" -m byzantine_experiments.analysis.phase3 table
