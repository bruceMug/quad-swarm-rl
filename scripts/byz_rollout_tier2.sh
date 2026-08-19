#!/bin/bash
# Tier 2 rollout matrix — structured (untrained) Byzantine attacks.
#
#   2A signflip   : partial (body roll/pitch) + full (all motors)   x 2 f x 3 seeds = 12
#   2B waypoint   : alpha in {0.3, 0.6, 1.0}                        x 2 f x 3 seeds = 18
#   2C liar       : lie_mode in {offset, stale, mirror}             x 2 f x 3 seeds = 18
#   2D collusion  : mode in {same_waypoint, pincer, mirror_pair}    x f=0.25 x 3 seeds = 9
#                                                                     total          = 57
#
# Clean references are NOT re-run: Tier 1 already collected 3 clean rollouts from the same
# checkpoints with the same collector and the same 50 episodes
# (byzantine_experiments/results/tier1/t1_clean_s{1,2,3}.json). The aggregator reads them
# from there. Re-running them would cost 45 minutes for identical numbers.
#
# Run from the REPO ROOT:
#     PARALLEL=12 bash scripts/byz_rollout_tier2.sh
#
# Environment knobs:
#     PARALLEL=12      concurrent rollouts (each pins itself to 1 BLAS thread)
#     N_EPISODES=50    episodes per run
#     WANDB=1          log every run to wandb (default 1)
#     FORCE=1          re-run configs whose output JSON already exists (default: skip)
#     ONLY=2a|2b|2c|2d run a single variant's block only
#     ALPHA_REFINE=1   run the 2B alpha refinement grid instead (12 runs, ~20 min)
#     DRY_RUN=1        print the job list and exit
#
# Safe to re-run: completed configs are skipped, so an interrupted sweep resumes.

set -uo pipefail

PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_recovered/bin/python}
PARALLEL=${PARALLEL:-10}
N_EPISODES=${N_EPISODES:-50}
WANDB=${WANDB:-1}
FORCE=${FORCE:-0}
ONLY=${ONLY:-all}
DRY_RUN=${DRY_RUN:-0}

RESULTS=byzantine_experiments/results/tier2
LOGS=logs/tier2
JOBS=$LOGS/jobs.tsv

# ----------------------------------------------------------------- single-run worker
if [[ "${1:-}" == "--single" ]]; then
    # The whole TSV line arrives as one argument; splitting here (rather than letting xargs
    # do it) keeps the JSON tier-configs intact -- xargs strips quotes.
    IFS=$'\t' read -r name tier cfg indices seed group <<< "$2"

    json="$RESULTS/$name.json"
    mkdir -p "$LOGS/runs"
    log="$LOGS/runs/$name.log"

    if [[ -f "$json" && "$FORCE" != "1" ]]; then
        echo "SKIP  $name (already done)"
        exit 0
    fi

    wandb_args=()
    [[ "$WANDB" == "1" ]] && wandb_args=(--wandb --wandb-group "$group")

    start=$(date +%s)
    echo "START $name  (tier=$tier cfg=$cfg idx=[$indices] seed=$seed)"

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -m byzantine_experiments.rollout.collect \
        --checkpoint "train_dir/exp1_baseline_seed${seed}" \
        --tier "$tier" --tier-config "$cfg" --compromised-indices "$indices" \
        --n-episodes "$N_EPISODES" --seed "$seed" --device cpu \
        --out "$RESULTS/$name.npz" --wandb-name "$name" "${wandb_args[@]}" \
        > "$log" 2>&1

    rc=$?
    elapsed=$(( $(date +%s) - start ))
    if [[ $rc -eq 0 ]]; then
        echo "DONE  $name  (${elapsed}s)  $(grep -m1 'mean episode reward' "$log" || true)"
    else
        echo "FAIL  $name  (rc=$rc, ${elapsed}s) -- see $log"
    fi
    exit $rc
fi

# ----------------------------------------------------------------- job list
mkdir -p "$RESULTS" "$LOGS"
: > "$JOBS"

emit() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$JOBS"; }

consolidate_logs() {
    [[ -d "$LOGS/runs" ]] || return 0
    for variant in signflip waypoint liar collusion; do
        local merged="$LOGS/${variant}.log" found=0
        : > "$merged.tmp"
        for f in "$LOGS/runs"/t2_${variant}_*.log; do
            [[ -e "$f" ]] || continue
            found=1
            { echo "==== $(basename "$f" .log)"; cat "$f"; echo; } >> "$merged.tmp"
        done
        if [[ "$found" == "1" ]]; then mv "$merged.tmp" "$merged"; else rm -f "$merged.tmp"; fi
    done
    [[ "${KEEP_RUN_LOGS:-0}" == "1" ]] || rm -rf "$LOGS/runs"
    echo "Logs consolidated: $(ls "$LOGS"/*.log 2>/dev/null | wc -l) files in $LOGS"
}

if [[ "${1:-}" == "--consolidate-logs" ]]; then consolidate_logs; exit 0; fi

want() { [[ "$ONLY" == "all" || "$ONLY" == "$1" ]]; }

if [[ "${ALPHA_REFINE:-0}" == "1" ]]; then
    # 2B alpha refinement. The spec's grid starts at alpha=0.3, which already causes 77.8 %
    # goal-RMSE degradation -- far above the 20-40 % band needed for matched-damage
    # comparison against Tier 1. The response saturates upward (0.3 -> 77.8, 0.6 -> 100.2,
    # 1.0 -> 116.9), so the band sits well below 0.3 and the spacing has to be fine.
    # 4 alphas x 3 seeds = 12 runs, f = 0.125 (the primary setting).
    for seed in 1 2 3; do
        for alpha in 0.05 0.10 0.15 0.20; do
            emit "t2_waypoint_s${seed}_a${alpha}_f0125" tier2_waypoint_pull \
                 "{\"alpha\":${alpha}}" "0" "$seed" "tier2_waypoint_a${alpha}_f0125"
        done
    done
    n_jobs=$(wc -l < "$JOBS")
    echo "Tier 2B ALPHA refinement: $n_jobs runs x $N_EPISODES episodes, $PARALLEL at a time"
    echo
    if [[ "$DRY_RUN" == "1" ]]; then column -t "$JOBS"; exit 0; fi
    export PY RESULTS LOGS N_EPISODES WANDB FORCE
    start_all=$(date +%s)
    xargs -a "$JOBS" -P "$PARALLEL" -d '\n' -n 1 bash "$0" --single
    echo
    echo "Alpha refinement finished in $(( ($(date +%s) - start_all) / 60 )) min"
    consolidate_logs
    exit 0
fi

for seed in 1 2 3; do
    for f in 0125 025; do
        if [[ "$f" == "0125" ]]; then indices="0"; else indices="0,1"; fi

        # --- 2A sign flip. `body` + dims [1,2] is the canonical PARTIAL attack: it inverts
        #     roll/pitch response while leaving collective thrust honest, so the attacker
        #     stays airborne. `motor` + all dims is the FULL attack, which inverts thrust too.
        if want 2a; then
            emit "t2_signflip_s${seed}_partial_f${f}" tier2_signflip \
                 '{"scale":1.0,"dims":[1,2],"space":"body"}' "$indices" "$seed" \
                 "tier2_signflip_partial_f${f}"
            emit "t2_signflip_s${seed}_full_f${f}" tier2_signflip \
                 '{"scale":1.0,"dims":"all","space":"motor"}' "$indices" "$seed" \
                 "tier2_signflip_full_f${f}"
        fi

        # --- 2B waypoint pull
        if want 2b; then
            for alpha in 0.3 0.6 1.0; do
                emit "t2_waypoint_s${seed}_a${alpha}_f${f}" tier2_waypoint_pull \
                     "{\"alpha\":${alpha}}" "$indices" "$seed" \
                     "tier2_waypoint_a${alpha}_f${f}"
            done
        fi

        # --- 2C state liar
        if want 2c; then
            for mode in offset stale mirror; do
                emit "t2_liar_s${seed}_${mode}_f${f}" tier2_liar \
                     "{\"lie_mode\":\"${mode}\",\"magnitude\":3.0,\"delay_k\":50}" \
                     "$indices" "$seed" "tier2_liar_${mode}_f${f}"
            done
        fi
    done

    # --- 2D collusion: needs two attackers, so f=0.25 only
    if want 2d; then
        for mode in same_waypoint pincer mirror_pair; do
            emit "t2_collusion_s${seed}_${mode}_f025" tier2_collusion \
                 "{\"mode\":\"${mode}\",\"alpha\":0.6}" "0,1" "$seed" \
                 "tier2_collusion_${mode}_f025"
        done
    fi
done

n_jobs=$(wc -l < "$JOBS")
echo "Tier 2 matrix: $n_jobs runs x $N_EPISODES episodes, $PARALLEL at a time"
echo "  results -> $RESULTS      logs -> $LOGS"
echo "  clean references reused from byzantine_experiments/results/tier1/t1_clean_s{1,2,3}"
echo "  wandb   -> $([[ "$WANDB" == "1" ]] && echo enabled || echo disabled)"
echo

if [[ "$DRY_RUN" == "1" ]]; then column -t "$JOBS"; exit 0; fi

export PY RESULTS LOGS N_EPISODES WANDB FORCE

start_all=$(date +%s)
xargs -a "$JOBS" -P "$PARALLEL" -d '\n' -n 1 bash "$0" --single
echo
echo "Matrix finished in $(( ($(date +%s) - start_all) / 60 )) min"
consolidate_logs

completed=$(ls "$RESULTS"/*.json 2>/dev/null | wc -l)
echo "Completed configs: $completed / $n_jobs"
if [[ "$completed" -lt "$n_jobs" ]]; then
    echo "Missing:"
    while IFS=$'\t' read -r name _; do
        [[ -f "$RESULTS/$name.json" ]] || echo "  $name"
    done < "$JOBS"
fi

echo
echo "Next: python -m byzantine_experiments.analysis.aggregate_tier2"
