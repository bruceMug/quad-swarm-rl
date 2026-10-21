#!/bin/bash
# The matched-damage triple (D1): select members by the pre-registered rule, collect them on
# identical scenarios, run TOST. Rules: analysis/matched_triple.py docstring (fixed 2026-09-26).
#
#     bash scripts/byz_matched_triple.sh
#
# Needs first: the D3/D4 adversaries trained and their own-seed n=200 runs collected
# (scripts/byz_transfer.sh), because the Tier 3 member and the damage target come from them.
# Selection is written to results_v2/matched_triple_selection.json BEFORE any triple episode
# is collected, and is never recomputed once collection has started (delete the file to redo).
#
# Cost: 4 conditions (clean + 3 members) x 3 baseline seeds x n, where n = 400 / 800 / 1600 per
# the power rule. At ~20 episodes/min on this machine: n=400 -> ~4 h, n=800 -> ~8 h.

set -uo pipefail
[[ -f scripts/byz_wandb_env.sh ]] && source scripts/byz_wandb_env.sh
PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python}
PARALLEL=${PARALLEL:-12}
R=${BYZ_RESULTS_ROOT:-byzantine_experiments/results_v2}
OUT=$R/matched_triple; LOGS=${BYZ_LOGS_ROOT:-logs_v2}/matched_triple
mkdir -p "$OUT" "$LOGS"

if [[ "${1:-}" == "--single" ]]; then
    IFS=$'\t' read -r name seed tier cfg idx <<< "$2"
    out="$OUT/$name.npz"; [[ -f "$out" ]] && { echo "SKIP  $name"; exit 0; }
    n=$(grep -oP '_n\K[0-9]+$' <<< "$name")
    if [[ "$tier" == "clean" ]]; then t=(--no-byzantine); else t=(--tier "$tier" --tier-config "$cfg" --compromised-indices "$idx"); fi
    echo "START $name"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -m byzantine_experiments.rollout.collect --checkpoint "train_dir/exp1_baseline_seed${seed}" \
        "${t[@]}" --n-episodes "$n" --seed "$seed" --device cpu --out "$out" \
        --wandb --wandb-group matched_triple --wandb-name "$name" > "$LOGS/$name.log" 2>&1 \
        && echo "DONE  $name" || { echo "FAIL  $name -- $LOGS/$name.log"; exit 1; }
    exit 0
fi

if [[ ! -f "$R/matched_triple_selection.json" ]]; then
    "$PY" -m byzantine_experiments.analysis.matched_triple select || exit 1
else
    echo "Using existing selection (pre-registered; delete $R/matched_triple_selection.json to redo):"
    cat "$R/matched_triple_selection.json"
fi
JOBS=$LOGS/jobs.tsv
"$PY" -m byzantine_experiments.analysis.matched_triple jobs > "$JOBS" || exit 1
echo; echo "Collecting $(wc -l < "$JOBS") runs, $PARALLEL at a time"
[[ "${DRY_RUN:-0}" == "1" ]] && { cut -f1-3 "$JOBS" | column -t; exit 0; }
export PY OUT LOGS
xargs -a "$JOBS" -d '\n' -P "$PARALLEL" -n 1 bash "$0" --single
"$PY" -m byzantine_experiments.analysis.matched_triple test
