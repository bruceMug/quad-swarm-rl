#!/bin/bash
# Tier 1 rollout matrix — the negative-control characterisation sweep.
#
#   3 baseline seeds x 3 variants x 3 severities x 2 compromised fractions = 54 runs
#   + 3 clean reference runs (one per seed), needed for the degradation baseline
#   + 6 optional dropout mode='zero' runs at p=0.15, for the mode comparison
#
# Run from the REPO ROOT (paths inside assume it):
#     bash scripts/byz_rollout_tier1.sh
#
# Environment knobs:
#     PARALLEL=10      concurrent rollouts (default 10; each pins itself to 1 thread)
#     N_EPISODES=50    episodes per run
#     WANDB=1          log every run to wandb (default 1; set 0 to disable)
#     FORCE=1          re-run configs whose output JSON already exists (default: skip)
#     EXTRA_MODES=1    include the dropout mode='zero' side-sweep (default 1)
#     REFINE=1         run the in-band refinement grid instead of the main matrix (33 runs)
#     STALL=1          run the 1C' burst-stall grid instead (27 runs)
#     DRY_RUN=1        print the job list and exit
#
# Safe to re-run: completed configs are skipped, so an interrupted sweep resumes.

set -uo pipefail

PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_recovered/bin/python}
PARALLEL=${PARALLEL:-10}
N_EPISODES=${N_EPISODES:-50}
WANDB=${WANDB:-1}
FORCE=${FORCE:-0}
EXTRA_MODES=${EXTRA_MODES:-1}
DRY_RUN=${DRY_RUN:-0}

RESULTS=byzantine_experiments/results/tier1
LOGS=logs/tier1
JOBS=$LOGS/jobs.tsv

# ----------------------------------------------------------------- single-run worker
# Re-entrant: the script calls itself with --single for each row of the job file, which
# keeps the JSON config quoting intact (no eval, no word splitting).
if [[ "${1:-}" == "--single" ]]; then
    # One whole TSV line arrives as a single argument. Splitting here rather than letting
    # xargs do it is deliberate: xargs strips quotes, which would mangle the JSON configs,
    # and it collapses empty fields, which would shift every column on the clean rows.
    IFS=$'\t' read -r name tier cfg indices seed group <<< "$2"
    [[ "$indices" == "none" ]] && indices=""

    out="$RESULTS/$name.npz"
    json="$RESULTS/$name.json"
    # Per-run logs go in a subdirectory and are merged into one file per variant at the end
    # of the sweep (see consolidate_logs). 60+ loose .log files is not a useful artefact.
    mkdir -p "$LOGS/runs"
    log="$LOGS/runs/$name.log"

    if [[ -f "$json" && "$FORCE" != "1" ]]; then
        echo "SKIP  $name (already done)"
        exit 0
    fi

    wandb_args=()
    [[ "$WANDB" == "1" ]] && wandb_args=(--wandb --wandb-group "$group")

    if [[ "$tier" == "clean" ]]; then
        byz_args=(--no-byzantine)
    else
        byz_args=(--tier "$tier" --tier-config "$cfg" --compromised-indices "$indices")
    fi

    start=$(date +%s)
    echo "START $name  (tier=$tier cfg=$cfg idx=[$indices] seed=$seed)"

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -m byzantine_experiments.rollout.collect \
        --checkpoint "train_dir/exp1_baseline_seed${seed}" \
        "${byz_args[@]}" \
        --n-episodes "$N_EPISODES" \
        --seed "$seed" \
        --device cpu \
        --out "$out" \
        --wandb-name "$name" \
        "${wandb_args[@]}" \
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

# ----------------------------------------------------------------- job list generation
mkdir -p "$RESULTS" "$LOGS"
: > "$JOBS"

emit() {  # name tier cfg indices seed group
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$JOBS"
}

# Merge the per-run logs into one file per variant, so the sweep leaves ~5 readable files
# instead of one per run. KEEP_RUN_LOGS=1 preserves the individual ones for debugging.
consolidate_logs() {
    [[ -d "$LOGS/runs" ]] || return 0
    for variant in clean gaussian ou dropout stall; do
        local merged="$LOGS/${variant}.log" found=0
        : > "$merged.tmp"
        for f in "$LOGS/runs"/t1_${variant}_*.log; do
            [[ -e "$f" ]] || continue
            found=1
            {
                echo "================================================================"
                echo "=== $(basename "$f" .log)"
                echo "================================================================"
                cat "$f"
                echo
            } >> "$merged.tmp"
        done
        if [[ "$found" == "1" ]]; then mv "$merged.tmp" "$merged"; else rm -f "$merged.tmp"; fi
    done
    if [[ "${KEEP_RUN_LOGS:-0}" != "1" ]]; then
        rm -rf "$LOGS/runs"
    fi
    echo "Logs consolidated: $(ls "$LOGS"/*.log 2>/dev/null | wc -l) files in $LOGS"
}

# Allow re-consolidating an existing sweep's logs without re-running anything.
if [[ "${1:-}" == "--consolidate-logs" ]]; then
    consolidate_logs
    exit 0
fi

if [[ "${STALL:-0}" == "1" ]]; then
    # Variant 1C' — dropout re-parameterised by stall DURATION rather than per-step
    # probability. The original 1C never entered the 20-40 % band at any p_drop, because at
    # 100 Hz an i.i.d. stall lasts ~1 control tick and reusing a 10 ms-old command is barely
    # a fault. This grid sweeps how long a stall lasts (mean_stall_s) against how much of the
    # time the drone is stalled (duty_cycle), which are the two physically meaningful knobs.
    # 3 durations x 3 duty cycles x 3 seeds = 27 runs, f = 0.125, mode=hold.
    for seed in 1 2 3; do
        for duty in 0.10 0.25 0.50; do
            for ms in 0.1 0.3 1.0; do
                emit "t1_stall_s${seed}_d${duty}_ms${ms}_hold_f0125" \
                     tier1_stall "{\"duty_cycle\":${duty},\"mean_stall_s\":${ms},\"mode\":\"hold\"}" \
                     "0" "$seed" "tier1_stall_d${duty}_ms${ms}_hold_f0125"
            done
        done
    done
    n_jobs=$(wc -l < "$JOBS")
    echo "Tier 1C' STALL grid: $n_jobs runs x $N_EPISODES episodes, $PARALLEL at a time"
    echo
    if [[ "$DRY_RUN" == "1" ]]; then column -t "$JOBS"; exit 0; fi
    export PY RESULTS LOGS N_EPISODES WANDB FORCE
    start_all=$(date +%s)
    xargs -a "$JOBS" -P "$PARALLEL" -d '\n' -n 1 bash "$0" --single
    echo
    echo "Stall grid finished in $(( ($(date +%s) - start_all) / 60 )) min"
    consolidate_logs
    exit 0
fi

if [[ "${REFINE:-0}" == "1" ]]; then
    # Refinement grid. The spec's severity grids straddle the 20-40 % degradation band
    # rather than landing in it (gaussian jumps 16.5 % -> 47.3 %; OU is already at 48.9 %
    # at its LOWEST severity; dropout never leaves the noise floor at p<=0.3). These
    # in-between severities exist to give each variant a config actually inside the band,
    # which is what the validation gate is for. f=0.125 only -- that is the primary setting.
    for seed in 1 2 3; do
        for sigma in 0.30 0.35 0.40; do
            emit "t1_gaussian_s${seed}_sigma${sigma}_f0125" \
                 tier1_gaussian "{\"sigma\":${sigma}}" "0" "$seed" \
                 "tier1_gaussian_sigma${sigma}_f0125"
        done
        for sou in 0.04 0.06 0.08; do
            emit "t1_ou_s${seed}_sigmaou${sou}_f0125" \
                 tier1_ou "{\"sigma_ou\":${sou}}" "0" "$seed" \
                 "tier1_ou_sigmaou${sou}_f0125"
        done
        for p in 0.6 0.85 0.98; do
            emit "t1_dropout_s${seed}_p${p}_hold_f0125" \
                 tier1_dropout "{\"p_drop\":${p},\"mode\":\"hold\"}" "0" "$seed" \
                 "tier1_dropout_p${p}_hold_f0125"
        done
        for p in 0.5 0.85; do
            emit "t1_dropout_s${seed}_p${p}_zero_f0125" \
                 tier1_dropout "{\"p_drop\":${p},\"mode\":\"zero\"}" "0" "$seed" \
                 "tier1_dropout_p${p}_zero_f0125"
        done
    done
    n_jobs=$(wc -l < "$JOBS")
    echo "Tier 1 REFINEMENT grid: $n_jobs runs x $N_EPISODES episodes, $PARALLEL at a time"
    echo
    if [[ "$DRY_RUN" == "1" ]]; then column -t "$JOBS"; exit 0; fi
    export PY RESULTS LOGS N_EPISODES WANDB FORCE
    start_all=$(date +%s)
    xargs -a "$JOBS" -P "$PARALLEL" -d '\n' -n 1 bash "$0" --single
    echo
    echo "Refinement finished in $(( ($(date +%s) - start_all) / 60 )) min"
    consolidate_logs
    exit 0
fi

for seed in 1 2 3; do
    # Clean reference. Needed per seed: degradation is measured against the same
    # checkpoint under the same collector, not against the older evaluate.py JSONs.
    emit "t1_clean_s${seed}" clean '{}' none "$seed" "clean_baseline"

    for f in 0125 025; do
        if [[ "$f" == "0125" ]]; then indices="0"; else indices="0,1"; fi

        for sigma in 0.1 0.25 0.5; do
            emit "t1_gaussian_s${seed}_sigma${sigma}_f${f}" \
                 tier1_gaussian "{\"sigma\":${sigma}}" "$indices" "$seed" \
                 "tier1_gaussian_sigma${sigma}_f${f}"
        done

        for sou in 0.1 0.2 0.4; do
            emit "t1_ou_s${seed}_sigmaou${sou}_f${f}" \
                 tier1_ou "{\"sigma_ou\":${sou}}" "$indices" "$seed" \
                 "tier1_ou_sigmaou${sou}_f${f}"
        done

        for p in 0.05 0.15 0.3; do
            emit "t1_dropout_s${seed}_p${p}_hold_f${f}" \
                 tier1_dropout "{\"p_drop\":${p},\"mode\":\"hold\"}" "$indices" "$seed" \
                 "tier1_dropout_p${p}_hold_f${f}"
        done

        if [[ "$EXTRA_MODES" == "1" ]]; then
            # Side-sweep: is a dead output worse than a frozen one at equal drop rate?
            emit "t1_dropout_s${seed}_p0.15_zero_f${f}" \
                 tier1_dropout '{"p_drop":0.15,"mode":"zero"}' "$indices" "$seed" \
                 "tier1_dropout_p0.15_zero_f${f}"
        fi
    done
done

n_jobs=$(wc -l < "$JOBS")
echo "Tier 1 matrix: $n_jobs runs x $N_EPISODES episodes, $PARALLEL at a time"
echo "  results -> $RESULTS"
echo "  logs    -> $LOGS"
echo "  wandb   -> $([[ "$WANDB" == "1" ]] && echo enabled || echo disabled)"
echo

if [[ "$DRY_RUN" == "1" ]]; then
    column -t "$JOBS"
    exit 0
fi

export PY RESULTS LOGS N_EPISODES WANDB FORCE

start_all=$(date +%s)
xargs -a "$JOBS" -P "$PARALLEL" -d '\n' -n 1 bash "$0" --single
rc=$?
echo
echo "Matrix finished in $(( ($(date +%s) - start_all) / 60 )) min (xargs rc=$rc)"
consolidate_logs

completed=$(ls "$RESULTS"/*.json 2>/dev/null | wc -l)
echo "Completed configs: $completed / $n_jobs"
if [[ "$completed" -lt "$n_jobs" ]]; then
    echo "Missing:"
    while IFS=$'\t' read -r name _; do
        [[ -f "$RESULTS/$name.json" ]] || echo "  $name  (see $LOGS/$name.log)"
    done < "$JOBS"
fi

echo
echo "Next: python -m byzantine_experiments.analysis.aggregate_tier1"
