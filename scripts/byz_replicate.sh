#!/bin/bash
# Replicate ONE condition several times, so its error bar includes run-to-run variance.
#
#     bash scripts/byz_replicate.sh                       # default: the strongest attack
#     TIER=tier2_liar CFG='{"lie_mode":"mirror"}' bash scripts/byz_replicate.sh
#
# WHY THIS EXISTS
#
# Every comparison in this project so far is ONE attacked run against ONE clean run, with a
# confidence interval built from episode-to-episode variance inside those two runs. Three
# independent clean rollouts on baseline seed 3 showed that is not the whole error:
#
#     t1_clean_s3        n= 50  mean 0.9938
#     confirm_clean_s3   n=200  mean 1.1783     <- same config, same git SHA as the next
#     confirm_clean_rep2 n=200  mean 1.0568     <- differs only in --seed
#
# Cochran Q = 11.25 on 2 df (p = 0.004), I^2 = 82 %, run-level sd tau = 7.2 % of the mean.
# Using only the two same-SHA runs gives tau = 7.3 %, so it is not a code-drift artifact.
# Episodes within a run are iid (Ljung-Box p = 0.96), so this is a per-run offset: different
# seeds draw different scenario mixes and the offset does not average away inside one run.
#
# Folding that component into the corpus turns 20 significant findings into 0 out of 245.
# That correction is only as good as tau, which rests on three runs -- hence this script.
# REPLICATES ARE WORTH MORE THAN EPISODES HERE: 6 runs x 50 episodes pins tau far better than
# 2 runs x 150, for the same compute.
#
# Environment knobs:
#     REPS=3            replicates per arm
#     N_EPISODES=50     episodes per replicate
#     SEED_BASE=3       baseline checkpoint seed (also the first env seed)
#     TIER / CFG / INDICES   the attacked condition; default is Tier 2's `mirror` liar,
#                            the strongest attack in the corpus
#     PARALLEL=6  WANDB=1  DRY_RUN=1

set -uo pipefail

PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python}
REPS=${REPS:-3}
N_EPISODES=${N_EPISODES:-50}
SEED_BASE=${SEED_BASE:-3}
TIER=${TIER:-tier2_liar}
CFG=${CFG:-'{"lie_mode":"mirror","magnitude":3.0}'}
INDICES=${INDICES:-0}
PARALLEL=${PARALLEL:-6}
WANDB=${WANDB:-1}
DRY_RUN=${DRY_RUN:-0}

RESULTS=byzantine_experiments/results/tier3_replicate
LOGS=logs/tier3_replicate
JOBS=$LOGS/jobs.tsv
mkdir -p "$RESULTS" "$LOGS"

# Keep wandb off the AFS home volume (it sits near quota and fails the artifact upload).
[[ -f scripts/byz_wandb_env.sh ]] && source scripts/byz_wandb_env.sh

if [[ "${1:-}" == "--single" ]]; then
    IFS=$'\t' read -r name arm eseed <<< "$2"
    out="$RESULTS/$name.npz"
    if [[ -f "$out" ]]; then echo "SKIP  $name"; exit 0; fi
    if [[ "$arm" == "clean" ]]; then
        byz=(--no-byzantine)
    else
        byz=(--tier "$TIER" --tier-config "$CFG" --compromised-indices "$INDICES")
    fi
    wb=(); [[ "$WANDB" == "1" ]] && wb=(--wandb --wandb-group tier3_replicate --wandb-name "$name")
    echo "START $name (arm=$arm eseed=$eseed)"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -m byzantine_experiments.rollout.collect \
        --checkpoint "train_dir/exp1_baseline_seed${SEED_BASE}" "${byz[@]}" \
        --n-episodes "$N_EPISODES" --seed "$eseed" --device cpu \
        --out "$out" "${wb[@]}" > "$LOGS/$name.log" 2>&1
    rc=$?
    [[ $rc -eq 0 ]] && echo "DONE  $name" || echo "FAIL  $name (see $LOGS/$name.log)"
    exit $rc
fi

: > "$JOBS"
for r in $(seq 1 "$REPS"); do
    eseed=$(( SEED_BASE + 1000 * r ))
    printf 'rep_clean_s%s_r%d\tclean\t%d\n'  "$SEED_BASE" "$r" "$eseed" >> "$JOBS"
    printf 'rep_attack_s%s_r%d\tattack\t%d\n' "$SEED_BASE" "$r" "$eseed" >> "$JOBS"
done

echo "Replicated condition: $REPS x clean + $REPS x attacked, n=$N_EPISODES each"
echo "  attacked condition : tier=$TIER cfg=$CFG indices=[$INDICES]"
echo "  baseline           : train_dir/exp1_baseline_seed${SEED_BASE}"
echo "  results -> $RESULTS      logs -> $LOGS"
echo
[[ "$DRY_RUN" == "1" ]] && { column -t "$JOBS"; exit 0; }

export PY RESULTS LOGS WANDB TIER CFG INDICES SEED_BASE N_EPISODES
start=$(date +%s)
xargs -a "$JOBS" -P "$PARALLEL" -d '\n' -n 1 bash "$0" --single
echo
echo "Finished in $(( ($(date +%s) - start) / 60 )) min"
echo
"$PY" -m byzantine_experiments.analysis.replicate_test --results-dir "$RESULTS" \
    --seed "$SEED_BASE"
