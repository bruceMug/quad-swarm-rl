#!/bin/bash

set -e

LOG_DIR="logs_v2"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_B_$(date +%Y%m%d_%H%M%S).log"

# Write everything to the log AND display it in the terminal.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Starting B-series Byzantine experiments"
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "========================================"

# B1. D3: Plausible (stealth) liar on seed 3
LIES=1 LIE_ONLY=1 EGO=0 LIE_MODE=plausible bash scripts/byz_train_adv.sh 3 0

# B2. D4: Free liar on seed 2
LIES=1 LIE_ONLY=1 EGO=0 bash scripts/byz_train_adv.sh 2 0

# B3. Plausible liar on seed 1
LIES=1 LIE_ONLY=1 EGO=0 LIE_MODE=plausible bash scripts/byz_train_adv.sh 1 0

# B4. Plausible liar on seed 2
LIES=1 LIE_ONLY=1 EGO=0 LIE_MODE=plausible bash scripts/byz_train_adv.sh 2 0

echo "========================================"
echo "All B-series experiments completed successfully."
echo "Finished: $(date)"
echo "Log saved to: $LOG_FILE"
echo "========================================"
