#!/bin/bash
# Block F: the final Tier 3 tables, with every adversary in them.
#
#   bash run_block_F.sh              # ~1.2 h: cross-seed Phase 3 (~1 h) + tables (~15 min)
#   CROSS=0 bash run_block_F.sh      # tables only, ~15 min
#
# F0 is the spec's remaining Phase 3 factor (adversary: matching-seed AND cross-seed). Finished
# runs are skipped, so the 60 priority runs from block E are not repeated.

set -e

LOG_DIR="logs_v2"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_F_$(date +%Y%m%d_%H%M%S).log"

# Write everything to the log AND display it in the terminal.
exec > >(tee -a "$LOG_FILE") 2>&1

source scripts/byz_wandb_env.sh

ALL="adv_seed1_stealth0_lieonly adv_seed2_stealth0_lieonly adv_seed3_stealth0_lieonly \
adv_seed3_stealth0_lieonlyego adv_seed1_stealth0_lieonlyplaus adv_seed2_stealth0_lieonlyplaus \
adv_seed3_stealth0_lieonlyplaus"

echo "========================================"
echo "Starting F-series Byzantine experiments"
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "========================================"

# F0. Phase 3 cross-seed factor: each baseline seed also faces the Tier 3 adversaries trained on
#     the other two seeds, at all five shift levels. 30 new runs x 50 episodes, ~1 h.
#     byz_phase3.sh ends by rewriting results_v2/phase3_table.md (now with the cm column).
if [ "${CROSS:-1}" == "1" ]; then
    CROSS=1 bash scripts/byz_phase3.sh
else
    python -m byzantine_experiments.analysis.phase3 table
fi

# F1. Horizon/gate table with full run names (all 9 swept adversaries), ~2 min.
python -m byzantine_experiments.analysis.gate_tier3 --horizon

# F2. Efficacy gate: every adversary x five criteria, now including the four block-B ones.
python -m byzantine_experiments.analysis.efficacy_table

# F3. Cross-seed transfer matrix, all seven airborne adversaries.
python -m byzantine_experiments.analysis.transfer_matrix --runs $ALL

# F4. Reward components and compute accounting, re-read from TensorBoard so the four block-B
#     runs are included (~40 s per new adversary the first time).
python -m byzantine_experiments.analysis.tier3_report rewards --refresh
python -m byzantine_experiments.analysis.tier3_report compute --refresh

echo "========================================"
echo "All F-series experiments completed successfully."
echo "Finished: $(date)"
echo "Log saved to: $LOG_FILE"
echo "========================================"
