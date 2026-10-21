#!/bin/bash
# Tier 3 cross-seed transfer: each adversary's FINAL checkpoint against every baseline seed.
#
#     bash scripts/byz_transfer.sh                               # the current airborne adversaries
#     ADV_RUNS="adv_seed2_stealth0_lieonly" bash scripts/byz_transfer.sh   # add a new one later
#
# The spec's gate requires transfer to be TESTED (either outcome is a result): an adversary
# trained against baseline seed i, evaluated against seeds j != i. Rules, fixed 2026-09-26:
#   * checkpoint = the final one (agents/tier3_learned.py::final_milestone_m -- m194 for a 200M
#     run), the same rule as the confirmatory;
#   * n = 200 per cell, paired against a 200-episode clean run at that baseline seed;
#   * adversaries = those that stay airborne. The motor adversaries (rounds 1-2 and `_lie`)
#     crash in 100 % of episodes, so they fail the survival criterion on their own seed and
#     their transfer is moot.
# The diagonal (own seed) IS the confirmatory run and reuses its file if it exists.
#
# Every cell is one single-process rollout; they run PARALLEL at a time. On this machine the
# throughput is ~20 episodes/min in total, i.e. ~10 min of wall clock per 200-episode cell.

set -uo pipefail
[[ -f scripts/byz_wandb_env.sh ]] && source scripts/byz_wandb_env.sh
PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python}
PARALLEL=${PARALLEL:-12}
N=${N_EPISODES:-200}
ADV_RUNS=${ADV_RUNS:-"adv_seed1_stealth0_lieonly adv_seed3_stealth0_lieonly adv_seed3_stealth0_lieonlyego"}
SEEDS=${BASELINE_SEEDS:-"1 2 3"}
R=${BYZ_RESULTS_ROOT:-byzantine_experiments/results_v2}
OUT=$R/tier3_confirm
LOGS=${BYZ_LOGS_ROOT:-logs_v2}/tier3_transfer
mkdir -p "$OUT" "$LOGS"
JOBS=$LOGS/jobs.tsv

if [[ "${1:-}" == "--single" ]]; then
    IFS=$'\t' read -r name bseed kind adv ms <<< "$2"
    out="$OUT/$name.npz"
    [[ -f "$out" ]] && { echo "SKIP  $name"; exit 0; }
    if [[ "$kind" == "clean" ]]; then
        tier=(--no-byzantine)
    else
        tier=(--tier tier3_learned
              --tier-config "{\"checkpoint\": \"train_dir/${adv}\", \"milestone\": ${ms}000000}"
              --compromised-indices 0)
    fi
    echo "START $name"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -m byzantine_experiments.rollout.collect \
        --checkpoint "train_dir/exp1_baseline_seed${bseed}" "${tier[@]}" \
        --n-episodes "$N" --seed "$bseed" --device cpu --out "$out" \
        --wandb --wandb-group tier3_transfer --wandb-name "$name" > "$LOGS/$name.log" 2>&1 \
        && echo "DONE  $name" || { echo "FAIL  $name -- $LOGS/$name.log"; exit 1; }
    exit 0
fi

# Truncate only in the coordinator -- workers (--single) must never touch the queue they are
# being fed from.
: > "$JOBS"
for b in $SEEDS; do
    printf 'confirm_clean_s%s_n%s\t%s\tclean\t-\t-\n' "$b" "$N" "$b" >> "$JOBS"
done
for adv in $ADV_RUNS; do
    [[ -d "train_dir/$adv" ]] || { echo "WARN: train_dir/$adv missing, skipped" >&2; continue; }
    aseed=$(grep -oP 'seed\K[0-9]+' <<< "$adv" | head -1)
    ms=$("$PY" -c "from byzantine_experiments.agents.tier3_learned import final_milestone_m as f; print(f('train_dir/$adv'))")
    msl=$(printf 'm%03d' "$ms")
    for b in $SEEDS; do
        # Diagonal keeps the confirmatory name, so an existing confirmatory run is reused.
        if [[ "$b" == "$aseed" ]]; then name="confirm_${adv}_${msl}_n${N}"; else name="confirm_${adv}_${msl}_on_s${b}_n${N}"; fi
        printf '%s\t%s\tadv\t%s\t%s\n' "$name" "$b" "$adv" "$ms" >> "$JOBS"
    done
done

echo "Tier 3 cross-seed transfer: $(wc -l < "$JOBS") cells (incl. clean references), n=$N"
echo "  adversaries: $ADV_RUNS"
[[ "${DRY_RUN:-0}" == "1" ]] && { cut -f1,2 "$JOBS" | column -t; exit 0; }

export PY N OUT LOGS
# Clean references first: every other cell pairs against them.
grep -P '^confirm_clean' "$JOBS" | xargs -d '\n' -P "$PARALLEL" -n 1 bash "$0" --single
grep -vP '^confirm_clean' "$JOBS" | xargs -d '\n' -P "$PARALLEL" -n 1 bash "$0" --single
echo
"$PY" -m byzantine_experiments.analysis.transfer_matrix --n-episodes "$N" --runs $ADV_RUNS
