#!/bin/bash
# Tier 3 — evaluate trained adversaries against the efficacy gate.
#
# Drives the ordinary Tier 0 collector with `--tier tier3_learned`, so Tier 3 trajectories
# come out byte-compatible with Tiers 1 and 2 and every downstream analysis works unchanged.
#
#   HORIZON=1 bash scripts/byz_eval_adv.sh      horizon sweep: milestones of one adversary
#                                               against its own baseline seed. ~8 checkpoints
#                                               x 20 episodes, ~15 min. Answers "was the
#                                               adversary better earlier in training?"
#
#   bash scripts/byz_eval_adv.sh                the gate: each adversary at best+latest
#                                               against all 3 baseline seeds (cross-seed
#                                               transfer), 50 episodes. ~50 min.
#
# Environment knobs:
#     ADV_RUNS="adv_seed1_stealth0 adv_seed1_stealth5"   which adversaries to evaluate
#         (f=0.25 round-2 runs are named adv_seed1_stealth0_f250; the compromised indices are
#          read back out of each run's config.json, so no extra knob is needed here)
#     BASELINE_SEEDS="1 2 3"    baselines to evaluate against (cross-seed transfer)
#     MILESTONES=auto       horizon sweep points, in millions of env steps. "auto" = the same
#                           fractions of each run's own length as the original 200M set
#                           (3 %, 12.5 %, 25 %, 37.5 %, 50 %, 75 %, 97 %), so a 200M run still
#                           gets 6 25 50 75 100 150 194 and a 100M run gets 3 12 25 38 50 75 97.
#                           The last point is "the final checkpoint" used by the confirmatory rule.
#     N_EPISODES=50             (horizon sweep uses N_EPISODES_HORIZON, default 50 -- see
#                               design/tier3_design.md 4.3: n ~ 31 per arm is the threshold
#                               for 80 % power against a true 30 % effect, so the old n=20
#                               could not resolve the gate it was being read against)
#     PARALLEL=12  WANDB=1  FORCE=1  DRY_RUN=1
#
# Safe to re-run: completed configs are skipped.

set -uo pipefail

[[ -f scripts/byz_wandb_env.sh ]] && source scripts/byz_wandb_env.sh

PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python}
PARALLEL=${PARALLEL:-12}
N_EPISODES=${N_EPISODES:-50}
N_EPISODES_HORIZON=${N_EPISODES_HORIZON:-50}
WANDB=${WANDB:-1}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
HORIZON=${HORIZON:-0}

# Round 3 defaults to the observation-channel runs; override for rounds 1-2.
ADV_RUNS=${ADV_RUNS:-"adv_seed1_stealth0_lie adv_seed1_stealth0_lieonly"}
BASELINE_SEEDS=${BASELINE_SEEDS:-"1 2 3"}
MILESTONES=${MILESTONES:-auto}

# v2 root: data from the fixed collector must never mix with the legacy folders, and the
# skip-if-exists logic below would otherwise silently reuse the invalid legacy files.
RESULTS=${BYZ_RESULTS_ROOT:-byzantine_experiments/results_v2}/tier3
LOGS=${BYZ_LOGS_ROOT:-logs_v2}/tier3
JOBS=$LOGS/eval_jobs.tsv

# ----------------------------------------------------------------- single-run worker
if [[ "${1:-}" == "--single" ]]; then
    IFS=$'\t' read -r name adv_run cfg bseed episodes group indices <<< "$2"

    json="$RESULTS/$name.json"
    mkdir -p "$LOGS/runs"
    log="$LOGS/runs/$name.log"

    if [[ -f "$json" && "$FORCE" != "1" ]]; then
        echo "SKIP  $name (already done)"; exit 0
    fi

    wandb_args=()
    [[ "$WANDB" == "1" ]] && wandb_args=(--wandb --wandb-group "$group")

    start=$(date +%s)
    echo "START $name  (adv=$adv_run cfg=$cfg baseline=s$bseed)"

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" -m byzantine_experiments.rollout.collect \
        --checkpoint "train_dir/exp1_baseline_seed${bseed}" \
        --tier tier3_learned --tier-config "$cfg" --compromised-indices "$indices" \
        --n-episodes "$episodes" --seed "$bseed" --device cpu \
        --out "$RESULTS/$name.npz" --wandb-name "$name" "${wandb_args[@]}" \
        > "$log" 2>&1

    rc=$?
    elapsed=$(( $(date +%s) - start ))
    if [[ $rc -eq 0 ]]; then
        echo "DONE  $name  (${elapsed}s)  $(grep -m1 'goal_rmse_mean' "$log" || true)"
    else
        echo "FAIL  $name  (rc=$rc, ${elapsed}s) -- see $log"
    fi
    exit $rc
fi

# ----------------------------------------------------------------- job list
mkdir -p "$RESULTS" "$LOGS"
: > "$JOBS"
emit() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$JOBS"; }

for adv in $ADV_RUNS; do
    [[ -d "train_dir/$adv" ]] || { echo "WARN: train_dir/$adv not found, skipping" >&2; continue; }
    aseed=$(echo "$adv" | grep -oE 'seed[0-9]+' | grep -oE '[0-9]+')

    # Which drones this adversary was trained to control -- read from the run's own config so
    # evaluation can never disagree with training about f. Falls back to drone 0 (f=0.125).
    indices=$("$PY" - "$adv" <<'PYEOF' 2>/dev/null || echo 0
import json, sys
cfg = json.load(open(f'train_dir/{sys.argv[1]}/config.json'))
print(str(cfg.get('adv_indices', '0')).strip() or '0')
PYEOF
)
    indices=${indices:-0}

    if [[ "$HORIZON" == "1" ]]; then
        # Milestones against the adversary's OWN baseline seed: the horizon question.
        run_ms=$MILESTONES
        if [[ "$run_ms" == "auto" ]]; then
            run_ms=$("$PY" - "$adv" <<'PYEOF' 2>/dev/null
import json, sys
# The CONFIGURED length, not the last saved milestone: basing it on the milestone (e.g. 199.5M)
# turned m100 into m099 and m194 into m193, which would have re-evaluated every existing run
# under new names instead of skipping it.
cfg = json.load(open(f'train_dir/{sys.argv[1]}/config.json'))
top = float(cfg.get('train_for_env_steps', 200_000_000))
fr = (0.03, 0.125, 0.25, 0.375, 0.5, 0.75, 0.97)
print(' '.join(str(max(1, int(round(f * top / 1e6)))) for f in fr))
PYEOF
)
            [[ -n "$run_ms" ]] || run_ms="6 25 50 75 100 150 194"
        fi
        for ms in $run_ms; do
            label=$(printf 'm%03dM' "$ms")
            emit "t3_${adv}_on_s${aseed}_${label}" "$adv" \
                 "{\"checkpoint\":\"train_dir/${adv}\",\"milestone\":${ms}000000}" \
                 "$aseed" "$N_EPISODES_HORIZON" "tier3_horizon_${adv}" "$indices"
        done
    else
        # The gate: best + latest, against every baseline seed (cross-seed transfer).
        for kind in best latest; do
            for bseed in $BASELINE_SEEDS; do
                emit "t3_${adv}_on_s${bseed}_${kind}" "$adv" \
                     "{\"checkpoint\":\"train_dir/${adv}\",\"kind\":\"${kind}\"}" \
                     "$bseed" "$N_EPISODES" "tier3_gate_${adv}_${kind}" "$indices"
            done
        done
    fi
done

n_jobs=$(wc -l < "$JOBS")
mode=$([[ "$HORIZON" == "1" ]] && echo "HORIZON SWEEP" || echo "EFFICACY GATE")
echo "Tier 3 $mode: $n_jobs rollouts, $PARALLEL at a time"
echo "  adversaries : $ADV_RUNS"
echo "  results     -> $RESULTS      logs -> $LOGS"
echo

[[ "$DRY_RUN" == "1" ]] && { column -t "$JOBS"; exit 0; }

export PY RESULTS LOGS WANDB FORCE
start_all=$(date +%s)
xargs -a "$JOBS" -P "$PARALLEL" -d '\n' -n 1 bash "$0" --single
echo
echo "Finished in $(( ($(date +%s) - start_all) / 60 )) min"

completed=$(ls "$RESULTS"/*.json 2>/dev/null | wc -l)
echo "Sidecars present: $completed"
echo
echo "Next: python -m byzantine_experiments.analysis.gate_tier3$([[ "$HORIZON" == "1" ]] && echo ' --horizon')"
