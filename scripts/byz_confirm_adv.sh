#!/bin/bash
# Tier 3 — confirmatory measurement of ONE pre-selected checkpoint.
#
#     bash scripts/byz_confirm_adv.sh adv_seed3_stealth0_lieonly 75
#
# Runs the two rollouts it needs CONCURRENTLY (they are independent single-process jobs, so
# running them back to back would double the wall clock for nothing), then prints the verdict.
#
#   * the attacked arm : the chosen checkpoint, n=N_EPISODES fresh episodes
#   * the clean arm    : a matched clean reference at the same episode count
#
# The clean arm is reused by every later comparison on the same baseline seed -- see
# `confirm_tier3.best_clean_reference` -- so it is collected once and skipped if it exists.
#
# Why a separate stage at all: the horizon sweep (byz_eval_adv.sh) is EXPLORATORY. It tests 7
# checkpoints against a p<0.05 bar and reports the best, so its maximum is inflated by both
# multiplicity and the winner's curse. The number that goes in the paper has to be measured on
# a fresh sample, on a checkpoint named in advance. See design/tier3_design.md 6.14.
#
# Environment knobs:
#     N_EPISODES=200   per arm. n=200 vs n=200 gives a 95% CI of about +/- 7 pp on seed 3;
#                      n=50 (the sweep's setting) gives +/- 18 pp, which cannot resolve a 30%
#                      gate at all.
#     WANDB=1  FORCE=0  DRY_RUN=0
#
# Safe to re-run: completed arms are skipped unless FORCE=1.

set -uo pipefail

[[ -f scripts/byz_wandb_env.sh ]] && source scripts/byz_wandb_env.sh

RUN=${1:?usage: byz_confirm_adv.sh <adv_run_dir_name> <milestone_in_millions>}
MILESTONE=${2:?usage: byz_confirm_adv.sh <adv_run_dir_name> <milestone_in_millions>}

PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python}
N_EPISODES=${N_EPISODES:-200}
WANDB=${WANDB:-1}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

RESULTS=${BYZ_RESULTS_ROOT:-byzantine_experiments/results_v2}/tier3_confirm
LOGS=${BYZ_LOGS_ROOT:-logs_v2}/tier3_confirm
mkdir -p "$RESULTS" "$LOGS"

if [[ ! -d "train_dir/$RUN" ]]; then
    echo "ERROR: train_dir/$RUN not found" >&2
    exit 1
fi

# The module owns the naming and the wandb flags, so the script and the analysis cannot drift
# apart about which files belong to which arm.
mapfile -t CMDS < <("$PY" -m byzantine_experiments.analysis.confirm_tier3 \
    --run "$RUN" --milestone "$MILESTONE" --n-episodes "$N_EPISODES" \
    --print-commands 2>/dev/null | grep -E '^/|^python')

if [[ ${#CMDS[@]} -lt 2 ]]; then
    echo "ERROR: could not generate the collect commands" >&2
    exit 1
fi
ADV_CMD=${CMDS[0]}
CLEAN_CMD=${CMDS[1]}
[[ "$WANDB" != "1" ]] && { ADV_CMD=${ADV_CMD/ --wandb*/}; CLEAN_CMD=${CLEAN_CMD/ --wandb*/}; }

adv_out=$(grep -oP '(?<=--out )\S+' <<< "$ADV_CMD")
clean_out=$(grep -oP '(?<=--out )\S+' <<< "$CLEAN_CMD")

echo "Tier 3 CONFIRMATORY measurement"
echo "  adversary   : $RUN @ ${MILESTONE}M"
echo "  episodes    : $N_EPISODES per arm (sweep used 50 -- see the header for why)"
echo "  attacked -> $adv_out"
echo "  clean    -> $clean_out"
echo "  logs     -> $LOGS/"
echo

if [[ "$DRY_RUN" == "1" ]]; then
    echo "$ADV_CMD"; echo; echo "$CLEAN_CMD"; exit 0
fi

start=$(date +%s)
pids=()
run_arm() {
    local name=$1 out=$2 cmd=$3
    if [[ -f "$out" && "$FORCE" != "1" ]]; then
        echo "SKIP  $name (already collected: $out)"
        return 0
    fi
    echo "START $name  -> $LOGS/$name.log"
    ( OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        eval "$cmd" > "$LOGS/$name.log" 2>&1 ) &
    pids+=($!)
}

run_arm attacked "$adv_out" "$ADV_CMD"
run_arm clean    "$clean_out" "$CLEAN_CMD"

rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
echo
echo "Collection finished in $(( ($(date +%s) - start) / 60 )) min (rc=$rc)"
if [[ $rc -ne 0 ]]; then
    echo "At least one arm FAILED -- see $LOGS/*.log" >&2
    exit 1
fi

echo
"$PY" -m byzantine_experiments.analysis.confirm_tier3 \
    --run "$RUN" --milestone "$MILESTONE" --n-episodes "$N_EPISODES"
