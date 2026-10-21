#!/bin/bash
# Tier 3's central claim, measured head to head: does the TRAINED adversary beat the best
# HAND-DESIGNED attack on the same channel, the same baseline, and the same 200 scenarios?
#
#     bash scripts/byz_confirm_hand.sh
#
# The trained adversary's confirmatory run (adv_seed3_stealth0_lieonly @ 194M, n = 200, seed 3)
# measured +7.4 % [+3.1, +11.7]. The hand-designed comparison so far is screening data only
# (n = 50): the Tier 2 liar family on seed 3 averages +9.6 %. Those are not comparable. This
# re-measures the hand-designed side at the same n, on the same scenarios.
#
# Because every run at --seed 3 draws the same 200 scenarios (collect.py::episode_seed), the
# three hand-designed runs are paired not only with clean but with the trained run too, so the
# trained-minus-hand-designed difference is itself a paired, per-scenario comparison.
#
# WHICH hand-designed attacks: all three liar modes at f = 0.125, i.e. one compromised drone,
# matching the trained adversary. All three, not the best one, because picking the mode with the
# highest screening number would bias the comparison in the hand-designed attack's favour. The
# clean arm (confirm_clean_s3_n200) already exists and is reused, so only these three run.
#
# ~65 min: three independent single-process rollouts run concurrently.

set -uo pipefail
[[ -f scripts/byz_wandb_env.sh ]] && source scripts/byz_wandb_env.sh

PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python}
N_EPISODES=${N_EPISODES:-200}
SEED=${SEED:-3}
RESULTS=${BYZ_RESULTS_ROOT:-byzantine_experiments/results_v2}/tier3_confirm
LOGS=${BYZ_LOGS_ROOT:-logs_v2}/tier3_confirm
mkdir -p "$RESULTS" "$LOGS"

CLEAN="$RESULTS/confirm_clean_s${SEED}_n${N_EPISODES}.npz"
[[ -f "$CLEAN" ]] || { echo "ABORT: $CLEAN missing -- run the Tier 3 confirmatory first" >&2; exit 1; }

start=$(date +%s); pids=()
for mode in mirror offset stale; do
    name="confirm_hand_liar_${mode}_s${SEED}_n${N_EPISODES}"
    out="$RESULTS/$name.npz"
    if [[ -f "$out" ]]; then echo "SKIP  $name"; continue; fi
    echo "START $name -> $LOGS/$name.log"
    ( OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" -m byzantine_experiments.rollout.collect \
        --checkpoint "train_dir/exp1_baseline_seed${SEED}" \
        --tier tier2_liar \
        --tier-config "{\"lie_mode\":\"${mode}\",\"magnitude\":3.0,\"delay_k\":50}" \
        --compromised-indices 0 --n-episodes "$N_EPISODES" --seed "$SEED" --device cpu \
        --out "$out" --wandb --wandb-group tier3_confirm --wandb-name "$name" \
        > "$LOGS/$name.log" 2>&1 ) &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "Collection finished in $(( ($(date +%s) - start) / 60 )) min (rc=$rc)"
[[ $rc -ne 0 ]] && { echo "A run FAILED -- see $LOGS/" >&2; exit 1; }
echo
"$PY" -m byzantine_experiments.analysis.head_to_head --seed "$SEED" --n-episodes "$N_EPISODES"
