#!/bin/bash
# Block E: amended shift calibration, then Phase 3 rollouts at all four shift levels.
#
#   bash run_block_E.sh              # ~4 h: calibration ~1 h, Phase 3 ~3 h (60 runs)
#   CROSS=1 bash run_block_E.sh      # + cross-seed Tier 3 adversaries in Phase 3 (~5.5 h total)
#
# Why four levels (amendment 2026-10-12, analysis/shift_calibration.py docstring): the original
# grid's picks (mild +32.6 %, severe +140 % on the clean swarm) are kept as mild_v1 / severe_v1;
# the extended grid, same rules, gives the primary mild / severe. All four are collected and all
# are reported. Every step skips finished runs, so the script is safe to re-run.

set -e

LOG_DIR="logs_v2"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run_E_$(date +%Y%m%d_%H%M%S).log"

# Write everything to the log AND display it in the terminal.
exec > >(tee -a "$LOG_FILE") 2>&1

source scripts/byz_wandb_env.sh

echo "========================================"
echo "Starting E-series Byzantine experiments"
echo "Started: $(date)"
echo "Log: $LOG_FILE"
echo "========================================"

# E0. Keep the original picks as mild_v1 / severe_v1 before the extended grid replaces
#     mild / severe. Instant; refuses if run out of order.
python -m byzantine_experiments.analysis.shift_calibration --freeze-v1

# E1. Extended calibration grid, clean swarm only: 12 DR runs + 12 DR+wind runs, n = 50, ~1 h.
DRS="0.01 0.02 0.03 0.04" WINDS="0.1 0.25 0.5 1.0" bash scripts/byz_calibrate_shift.sh

python - <<'EOF'
import json
lv = json.load(open('byzantine_experiments/results_v2/shift_levels.json'))['levels']
print('\nShift levels for Phase 3 (clean-swarm effect, pooled seeds 1-3):')
for k in sorted(lv, key=lambda k: lv[k]['effect_pct']):
    v = lv[k]
    print(f"  {k:10s} DR {v['dr_ratio']:<5g} wind {v['wind_accel']:<4g}  "
          f"{v['effect_pct']:+7.1f} % [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]"
          f"{'  WEAK' if v.get('weak') else ''}")
EOF

# E2. Phase 3: matched triple x {none, mild, severe, mild_v1, severe_v1} x seeds 1-3, n = 50,
#     plus a clean run per (seed, level). 60 runs, ~3 h. Ends by writing results_v2/phase3_table.md.
CROSS=${CROSS:-0} bash scripts/byz_phase3.sh

echo "========================================"
echo "All E-series experiments completed successfully."
echo "Finished: $(date)"
echo "Log saved to: $LOG_FILE"
echo "========================================"
