#!/bin/bash
# Stage C of the v2 re-collection -- everything after the clean references and the liar block.
#
#     bash scripts/byz_stage_c.sh 1     # phase 1: hand-designed Tier 1 + rest of Tier 2 (~5 h)
#     bash scripts/byz_stage_c.sh 2     # phase 2: every trained Tier 3 adversary      (~2 h)
#
# Run phase 1, READ ITS FINAL REPORT, then run phase 2. The pause is deliberate: the efficacy
# gate (10 %, gate_tier3.GATE) is checked against the complete hand-designed results at the end
# of phase 1, before any Tier 3 number exists, so it cannot be tuned to the Tier 3 outcome.
#
# WHAT IS INCLUDED, AND WHAT IS SKIPPED ON PURPOSE
#
#   phase 1  Tier 2 signflip / waypoint / collusion (ONLY=2a,2b,2d)       39 runs
#            Tier 1 main matrix: gaussian, OU, dropout (clean refs reused)  60 runs
#            Tier 1 stall grid (1C', the variant that superseded dropout)   27 runs
#   phase 2  Tier 3 horizon sweep, 6 adversaries x 7 milestones             42 runs
#
#   skipped  Tier 1 REFINE grid (33) and Tier 2 ALPHA_REFINE grid (12): both exist to locate
#            severities inside the legacy 30 % band, which is no longer the target. Run them
#            later only if a calibration band needs refining.
#            Tier 3 stealth5 adversaries (2 x 7 = 14): their training was dominated 555:1 by
#            the stealth penalty (design 4.2), so they are not meaningful adversaries.
#
# Throughput on this machine is ~0.4 rollouts/min and does not improve above PARALLEL~6-12
# (measured on Stages A/B and the replicate run), so the estimates above are wall-clock.
# Safe to re-run: every sweep skips configs already collected in results_v2.

set -uo pipefail
PHASE=${1:?usage: byz_stage_c.sh 1|2}
PARALLEL=${PARALLEL:-12}
PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python}
export PARALLEL PY
[[ -f scripts/byz_wandb_env.sh ]] && source scripts/byz_wandb_env.sh

R=${BYZ_RESULTS_ROOT:-byzantine_experiments/results_v2}
mkdir -p logs_v2
LOG=logs_v2/stage_c_phase${PHASE}.log
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=== $(date '+%F %T')  $*"; }

need_clean() {
    for s in 1 2 3; do
        [[ -f "$R/tier1/t1_clean_s${s}.npz" ]] || {
            echo "ABORT: $R/tier1/t1_clean_s${s}.npz missing -- run Stage A first:"
            echo "       ONLY=clean bash scripts/byz_rollout_tier1.sh"; exit 1; }
    done
}

case "$PHASE" in
1)
    need_clean
    for blk in 2a 2b 2d; do
        step "Tier 2, block $blk"
        ONLY=$blk bash scripts/byz_rollout_tier2.sh
    done
    step "Tier 1, main matrix"
    bash scripts/byz_rollout_tier1.sh
    step "Tier 1, stall grid"
    STALL=1 bash scripts/byz_rollout_tier1.sh

    step "Analysis"
    "$PY" -m byzantine_experiments.analysis.aggregate_tier1
    "$PY" -m byzantine_experiments.analysis.aggregate_tier2
    "$PY" -m byzantine_experiments.analysis.channel_ceiling > /dev/null
    echo "  tables: $R/tier1_calibration.{csv,md}, $R/tier2_calibration.{csv,md}, $R/channel_ceiling.md"

    step "GATE CALIBRATION CHECK -- read this before running phase 2"
    "$PY" -m byzantine_experiments.analysis.gate_calibration
    echo
    echo "Phase 1 done. Log: $LOG"
    echo "Next: bash scripts/byz_stage_c.sh 2"
    ;;
2)
    need_clean
    ADV_RUNS="adv_seed1_stealth0 adv_seed1_stealth0_f250 adv_seed1_stealth0_lie \
adv_seed1_stealth0_lieonly adv_seed3_stealth0_lieonly adv_seed3_stealth0_lieonlyego"
    step "Tier 3 horizon sweep (screening, n=50 per checkpoint)"
    HORIZON=1 ADV_RUNS="$ADV_RUNS" bash scripts/byz_eval_adv.sh
    step "Gate table"
    "$PY" -m byzantine_experiments.analysis.gate_tier3 --horizon
    "$PY" -m byzantine_experiments.analysis.channel_ceiling > /dev/null
    echo
    echo "Phase 2 done. Rows marked CANDIDATE cleared the gate on screening data (n=50, 57 %"
    echo "power). The decision is the confirmatory run on the best one -- readme step 15."
    ;;
*)
    echo "usage: byz_stage_c.sh 1|2"; exit 1 ;;
esac
