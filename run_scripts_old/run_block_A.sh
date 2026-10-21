#!/bin/bash

set -e

LOG_DIR="logs_v2"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_block_A_$(date +%Y%m%d_%H%M%S).log"

# Write everything to the log AND display it in the terminal.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Starting Byzantine experiment pipeline"
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "========================================"

# A1. Tier 1/2 tables
python -m byzantine_experiments.analysis.aggregate_tier1
python -m byzantine_experiments.analysis.aggregate_tier2

# A2. Cross-seed transfer
bash scripts/byz_transfer.sh

# A3. Shift calibration
bash scripts/byz_calibrate_shift.sh

# A4. Tier 3 reports
python -m byzantine_experiments.analysis.tier3_report rewards
python -m byzantine_experiments.analysis.tier3_report compute
python -m byzantine_experiments.analysis.tier3_report stealth --run adv_seed3_stealth0_lieonly
python -m byzantine_experiments.analysis.tier3_report behaviour --run adv_seed3_stealth0_lieonly

# A5. Baseline checkpoint hash test
BYZ_SLOW_TESTS=1 python -m pytest \
    byzantine_experiments/tests/test_tier3.py -q -k baseline_checkpoint

# A6. Efficacy table
python -m byzantine_experiments.analysis.efficacy_table

echo "========================================"
echo "All Byzantine experiment analysis completed successfully."
echo "Finished: $(date)"
echo "Log saved to: $LOG_FILE"
echo "========================================"
