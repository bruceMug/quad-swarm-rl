#!/bin/bash
# Block C for the adversaries trained in block B that have not had it yet.
# Block C already ran for adv_seed3_stealth0_lieonlyplaus (log: logs_v2/run_C_20261009_004532.log);
# every step below skips outputs that exist, so listing it again would cost nothing.
#
#   bash run_block_C.sh                      # the three remaining adversaries (default)
#   bash run_block_C.sh adv_seed2_stealth0_lieonly   # or any subset
#
# All adversaries go through each step TOGETHER: the transfer cells then run 9 at a time
# instead of 3, which is ~1.5 h for all three instead of ~4 h one after another.
# Expected total: ~1 h horizon sweep + ~1.5 h transfer + a few minutes of reports/tests.

set -e

LOG_DIR="logs_v2"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_C_$(date +%Y%m%d_%H%M%S).log"

# Write everything to the log AND display it in the terminal.
exec > >(tee -a "$LOG_FILE") 2>&1

source scripts/byz_wandb_env.sh

if [ $# -gt 0 ]; then
    RUNS="$*"
else
    RUNS="adv_seed1_stealth0_lieonlyplaus adv_seed2_stealth0_lieonlyplaus adv_seed2_stealth0_lieonly"
fi

echo "========================================"
echo "Starting C-series Byzantine experiments"
echo "Started: $(date)"
echo "Adversaries: $RUNS"
echo "Log: $LOG_FILE"
echo "========================================"

# C1. Horizon sweep, 7 checkpoints x 50 episodes per adversary (damage over training).
HORIZON=1 ADV_RUNS="$RUNS" bash scripts/byz_eval_adv.sh

python -m byzantine_experiments.analysis.gate_tier3 --horizon

# C2. Own-seed confirmatory + transfer, final checkpoint, n = 200 per cell.
ADV_RUNS="$RUNS" bash scripts/byz_transfer.sh

# C3. Stealth and behaviour reports (these liars' broadcasts are recorded).
for RUN in $RUNS; do
    python -m byzantine_experiments.analysis.tier3_report stealth   --run $RUN
    python -m byzantine_experiments.analysis.tier3_report behaviour --run $RUN
done

# C4. Re-issue the seed-3 plausible stealth report. The version logged on 2026-10-09 counted the
#     terminal (auto-reset) step and showed 99.5 % of episodes failing "consistency"; with that
#     step excluded the figure is 0.0 % (tier3_report.broadcasts docstring). ~10 s.
python -m byzantine_experiments.analysis.tier3_report stealth --run adv_seed3_stealth0_lieonlyplaus

# C5. The full Tier 3 test suite (74 tests, a few minutes), including the two added 2026-10-09.
python -m pytest byzantine_experiments/tests/test_tier3.py -q

echo "========================================"
echo "All C-series experiments completed successfully."
echo "Finished: $(date)"
echo "Log saved to: $LOG_FILE"
echo "Next: bash scripts/byz_matched_triple.sh  (block D)"
echo "========================================"
