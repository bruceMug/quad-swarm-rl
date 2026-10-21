#!/bin/bash
# D5: calibrate the Phase 3 shift levels from their effect on the CLEAN swarm.
#
#     bash scripts/byz_calibrate_shift.sh          # ~55 min, 21 clean rollouts, n=50 each
#
# Amended grid (2026-10-12, analysis/shift_calibration.py docstring), ~1 h, 24 new rollouts:
#     python -m byzantine_experiments.analysis.shift_calibration --freeze-v1     # first, once
#     DRS="0.01 0.02 0.03 0.04" WINDS="0.1 0.25 0.5 1.0" bash scripts/byz_calibrate_shift.sh
# Stage 1 then chooses over the union of every DR ratio collected so far.
#
# Stage 1: dynamics randomisation only, ratios 0.05 0.1 0.2 0.3, on baseline seeds 1-3.
#          -> picks "mild" (smallest ratio with >= 5 % clean degradation, p < 0.05)
# Stage 2: DR at min(2 x mild, 0.3) plus wind 0.5 1 2 m/s^2 (gust = wind / 2), seeds 1-3.
#          -> picks "severe" (smallest wind with >= 25 % clean degradation, p < 0.05)
#
# Rules are fixed in analysis/shift_calibration.py and were written before any calibration
# data existed. Result: results_v2/shift_levels.json, which `collect.py --shift mild|severe`
# reads. Every run is paired against the v2 clean reference at the same seed (identical
# scenarios), so the effect measured is the shift alone.

set -uo pipefail
[[ -f scripts/byz_wandb_env.sh ]] && source scripts/byz_wandb_env.sh
PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python}
PARALLEL=${PARALLEL:-12}
N_EPISODES=${N_EPISODES:-50}
DRS=${DRS:-"0.05 0.1 0.2 0.3"}      # stage-1 ratios to collect (v1 grid by default)
WINDS=${WINDS:-"0.5 1.0 2.0"}       # stage-2 mean winds to collect (v1 grid by default)
R=${BYZ_RESULTS_ROOT:-byzantine_experiments/results_v2}
OUT=$R/shift_calibration
LOGS=${BYZ_LOGS_ROOT:-logs_v2}/shift_calibration
mkdir -p "$OUT" "$LOGS"
export PY N_EPISODES OUT LOGS

for s in 1 2 3; do
    [[ -f "$R/tier1/t1_clean_s${s}.npz" ]] || { echo "ABORT: $R/tier1/t1_clean_s${s}.npz missing (Stage A)"; exit 1; }
done

run_one() {   # tag seed extra-args...
    local tag=$1 seed=$2; shift 2
    local out="$OUT/shiftcal_s${seed}_${tag}.npz"
    [[ -f "$out" ]] && { echo "SKIP  shiftcal_s${seed}_${tag}"; return 0; }
    echo "START shiftcal_s${seed}_${tag}"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -m byzantine_experiments.rollout.collect \
        --checkpoint "train_dir/exp1_baseline_seed${seed}" --no-byzantine \
        --n-episodes "$N_EPISODES" --seed "$seed" --device cpu --shift custom "$@" \
        --out "$out" --wandb --wandb-group shift_calibration \
        --wandb-name "shiftcal_s${seed}_${tag}" > "$LOGS/shiftcal_s${seed}_${tag}.log" 2>&1 \
        && echo "DONE  shiftcal_s${seed}_${tag}" || echo "FAIL  shiftcal_s${seed}_${tag}"
}
export -f run_one

echo "=== Stage 1: dynamics randomisation only"
for r in $DRS; do for s in 1 2 3; do echo "dr${r} $s --dr-ratio $r"; done; done \
    | xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _
"$PY" -m byzantine_experiments.analysis.shift_calibration --stage 1 | tee "$LOGS/stage1.txt"
SEVERE_DR=$(grep -oP '(?<=^SEVERE_DR=)\S+' "$LOGS/stage1.txt")
[[ -n "$SEVERE_DR" ]] || { echo "ABORT: stage 1 did not choose a mild level"; exit 1; }

echo; echo "=== Stage 2: DR $SEVERE_DR + wind"
for w in $WINDS; do for s in 1 2 3; do
    g=$(awk "BEGIN{print $w/2}")
    echo "dr${SEVERE_DR}_wind${w} $s --dr-ratio $SEVERE_DR --wind-accel $w --gust $g --gust-tau 1.0"
done; done | xargs -P "$PARALLEL" -L 1 bash -c 'run_one "$@"' _
"$PY" -m byzantine_experiments.analysis.shift_calibration --stage 2 | tee "$LOGS/stage2.txt"
