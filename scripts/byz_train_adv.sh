#!/bin/bash
# Tier 3 — train one adversarial policy against a frozen honest team.
#
#     bash scripts/byz_train_adv.sh <seed> <stealth>
#     bash scripts/byz_train_adv.sh 1 0        # unconstrained adversary
#     bash scripts/byz_train_adv.sh 1 5        # stealth-constrained (the headline condition)
#
# Six runs make the full matrix: 3 baseline seeds x {0, 5}.
#
# Round 3 -- the observation channel (design/tier3_design.md #5). The adversary's action
# grows from 4 motors to 4+6, the extra six being the state it broadcasts:
#     LIES=1 bash scripts/byz_train_adv.sh 1 0            # lie + fly
#     LIES=1 LIE_ONLY=1 bash scripts/byz_train_adv.sh 1 0 # lie only (Tier 2C analogue)
#
# f = 0.25 (two colluding adversaries), round 2:
#     ADV_INDICES=0,1 W_ALIVE=0.05 bash scripts/byz_train_adv.sh 1 0
# Sample Factory then trains ONE shared policy driving both drones (num_policies stays 1), so
# the two attackers are identical-policy colluders sharing a team-level reward -- the strongest
# coordination available without an explicit comms channel. See design/tier3_design.md #4.
#
# The honest drones live INSIDE the environment (byzantine_experiments/adversary/frozen_team.py),
# so Sample Factory trains exactly one policy and never touches the frozen weights. There is
# no optimiser attached to them anywhere in the process -- `--num_policies` stays at 1.
#
# Environment knobs:
#     ENV_STEPS=200000000   training horizon (spec suggests 2e8, not the baseline's 1e9)
#     NUM_WORKERS=32
#     DEVICE=gpu            gpu|cpu
#     WANDB=1
#     ADV_INDICES=0         comma-separated compromised drones; "0,1" is f=0.25
#     LIES=0                1 = give the adversary the observation channel (round 3)
#     LIE_ONLY=0            1 = fly the honest policy, attack only via the broadcast
#     LIE_MODE=free         free | plausible. plausible = a bounded, smooth drift that passes
#                           naive bounds/speed/consistency/acceleration checks (D3). Budget via
#                           PLAUS_MAX_OFFSET=2.0 PLAUS_MAX_RATE=1.0 PLAUS_MAX_ACCEL=4.0
#     EGO=<auto>            1 = append the adversary's own absolute pos/vel to its obs
#                           (54 -> 60). Defaults to 1 whenever LIES=1, because without it a
#                           state-dependent lie like `mirror` is not expressible: absolute
#                           position appears nowhere in the 54-D vector. Set EGO=0 for the
#                           round-3/4 ablation.
#     MAX_LIE_POS=10.0 MAX_LIE_VEL=6.0   full-deflection lie, in m and m/s
#     ENTROPY=<auto>        exploration_loss_coeff; defaults to 0.003 for round-3 runs and
#                           0.0 otherwise, so rounds 1-2 reproduce bit-for-bit
#     EPSILON=0.3           attack budget for the stealth condition
#     W_TEAM / W_CRASH / W_ALIVE   reward weights (calibrated defaults, see below)
#     DRY_RUN=1             print the command and exit
#
# BEFORE LAUNCHING: sanity-check the reward balance, it takes ~20 s and has already caught
# one fatal miscalibration:
#     python -m byzantine_experiments.adversary.calibrate_reward

set -uo pipefail

SEED=${1:?usage: byz_train_adv.sh <seed> <stealth>}
STEALTH=${2:?usage: byz_train_adv.sh <seed> <stealth>}

PY=${PY:-/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python}
ENV_STEPS=${ENV_STEPS:-200000000}
NUM_WORKERS=${NUM_WORKERS:-32}
DEVICE=${DEVICE:-gpu}
WANDB=${WANDB:-1}
EPSILON=${EPSILON:-0.3}
ADV_INDICES=${ADV_INDICES:-0}
LIES=${LIES:-0}
LIE_ONLY=${LIE_ONLY:-0}
MAX_LIE_POS=${MAX_LIE_POS:-10.0}
MAX_LIE_VEL=${MAX_LIE_VEL:-6.0}
[[ "$LIE_ONLY" == "1" ]] && LIES=1
LIE_MODE=${LIE_MODE:-free}
PLAUS_MAX_OFFSET=${PLAUS_MAX_OFFSET:-2.0}
PLAUS_MAX_RATE=${PLAUS_MAX_RATE:-1.0}
PLAUS_MAX_ACCEL=${PLAUS_MAX_ACCEL:-4.0}
if [[ "$LIES" == "1" ]]; then EGO=${EGO:-1}; else EGO=${EGO:-0}; fi
# Rounds 1-2 both saturated (train/act_max 30-64 by 50M steps) and a saturated tanh policy
# with adaptive_stddev=False has almost no gradient left. That did not cause the ceiling --
# the channel did -- but round 3 should not inherit an absorbing corner. Only round-3 runs
# get the entropy bonus, so re-running rounds 1-2 still reproduces them exactly.
if [[ "$LIES" == "1" ]]; then ENTROPY=${ENTROPY:-0.003}; else ENTROPY=${ENTROPY:-0.0}; fi
# Calibrated, NOT the spec's literal values -- see byzantine_experiments/adversary/reward.py.
# The spec's w_crash charged every step drowns the objective 2622:1.
W_TEAM=${W_TEAM:-20.0}
W_CRASH=${W_CRASH:-10.0}
# Round 1 (f=0.125) showed w_alive=0.01 does not price survival at all -- both adversaries
# converged to a saturated controller that crashes itself. Raise it for multi-adversary runs.
W_ALIVE=${W_ALIVE:-0.01}
DRY_RUN=${DRY_RUN:-0}

BASELINE=train_dir/exp1_baseline_seed${SEED}
N_ADV=$(awk -F, '{print NF}' <<< "$ADV_INDICES")
# f is encoded in the run name so round-1 and round-2 artifacts never collide, and so the
# analysis can recover the compromised fraction from the filename alone. f=0.125 keeps the
# original naming, to avoid orphaning the checkpoints and sidecars already on disk.
FTAG=""
[[ "$N_ADV" != "1" ]] && FTAG="_f$(awk "BEGIN{printf \"%03d\", 1000*$N_ADV/8}")"
# Round-3 runs get their own suffix so the gate table can tell channels apart at a glance and
# nothing collides with the 28 sidecars already on disk.
LTAG=""
[[ "$LIES" == "1" ]] && LTAG="_lie"
[[ "$LIE_ONLY" == "1" ]] && LTAG="_lieonly"
# Round 5 changes the observation space, so its checkpoints must not collide with round 3/4's.
[[ "$LIES" == "1" && "$EGO" == "1" ]] && LTAG="${LTAG}ego"
# D3: plausible-lie runs get their own suffix; the free-lie names above are unchanged.
[[ "$LIES" == "1" && "$LIE_MODE" == "plausible" ]] && LTAG="${LTAG}plaus"
EXPERIMENT=adv_seed${SEED}_stealth${STEALTH}${FTAG}${LTAG}
LOGDIR=${BYZ_LOGS_ROOT:-logs_v2}/tier3_train   # training logs; adversaries trained before the v2 switch keep theirs in logs/tier3
mkdir -p "$LOGDIR"

if [[ ! -d "$BASELINE" ]]; then
    echo "ERROR: baseline checkpoint not found: $BASELINE" >&2
    exit 1
fi

wandb_args=()
if [[ "$WANDB" == "1" ]]; then
    wandb_args=(--with_wandb=True
                --wandb_user=bmugizi-carnegie-mellon-university
                --wandb_project=byzantine_exp
                --wandb_group=tier3_adversarial
                --wandb_job_type=train
                --wandb_tags tier3 adversarial seed${SEED} stealth${STEALTH} "nadv${N_ADV}" \
                    "$([[ "$LIES" == "1" ]] && echo channel_observation || echo channel_motors)")
fi

cmd=("$PY" -m byzantine_experiments.adversary.train_adv
     --adv_baseline_checkpoint="$BASELINE"
     --adv_indices="$ADV_INDICES"
     --adv_enable_lies="$([[ "$LIES" == "1" ]] && echo True || echo False)"
     --adv_lie_only="$([[ "$LIE_ONLY" == "1" ]] && echo True || echo False)"
     --adv_ego_state_obs="$([[ "$EGO" == "1" ]] && echo True || echo False)"
     --adv_lie_mode="$LIE_MODE"
     --adv_plausible_max_offset="$PLAUS_MAX_OFFSET"
     --adv_plausible_max_rate="$PLAUS_MAX_RATE"
     --adv_plausible_max_accel="$PLAUS_MAX_ACCEL"
     --adv_max_lie_pos="$MAX_LIE_POS"
     --adv_max_lie_vel="$MAX_LIE_VEL"
     --exploration_loss_coeff="$ENTROPY"
     --adv_w_team="$W_TEAM"
     --adv_w_crash="$W_CRASH"
     --adv_w_alive="$W_ALIVE"
     --adv_w_stealth="$STEALTH"
     --adv_epsilon="$EPSILON"
     --adv_team_reward_mode=honest
     --adv_crash_penalty_mode=onset
     --seed="$SEED"
     --experiment="$EXPERIMENT"
     --train_for_env_steps="$ENV_STEPS"
     --num_workers="$NUM_WORKERS"
     --device="$DEVICE"
     "${wandb_args[@]}")

echo "Tier 3 adversarial training"
echo "  baseline (frozen) : $BASELINE"
echo "  experiment        : $EXPERIMENT"
echo "  compromised       : drone(s) $ADV_INDICES  ->  n_adv=$N_ADV, f=$(awk "BEGIN{printf \"%.3f\", $N_ADV/8}")"
if [[ "$LIES" == "1" ]]; then
    echo "  channel           : motors + OBSERVATION (4+6 action)$([[ "$LIE_ONLY" == "1" ]] && echo ', motors driven by the frozen honest policy')"
    echo "                      max lie ${MAX_LIE_POS} m / ${MAX_LIE_VEL} m/s, entropy coeff $ENTROPY"
    echo "  adversary obs     : $([[ "$EGO" == "1" ]] && echo '60-D (54 + own absolute pos/vel)' || echo '54-D (round 3/4 ablation)')"
    echo "  lie               : $LIE_MODE$([[ "$LIE_MODE" == "plausible" ]] && echo " (offset<=${PLAUS_MAX_OFFSET} m, drift<=${PLAUS_MAX_RATE} m/s, accel<=${PLAUS_MAX_ACCEL} m/s^2)")"
else
    echo "  channel           : motors only (4 action) -- rounds 1-2 setting"
fi
echo "  stealth weight    : $STEALTH   (epsilon=$EPSILON)"
echo "  reward weights    : w_team=$W_TEAM w_crash=$W_CRASH w_alive=$W_ALIVE (onset, honest-mean)"
echo "  horizon           : $ENV_STEPS env steps on $DEVICE, $NUM_WORKERS workers"
echo

if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
    exit 0
fi

# The spec asks for the frozen-weights check on every training launch. It is ~30 s and it is
# the one failure that would silently invalidate the whole tier.
echo "Pre-flight: frozen-weights test ..."
if ! "$PY" -m pytest byzantine_experiments/tests/test_tier3.py -q \
        -k "frozen or no_optimiser" > "$LOGDIR/${EXPERIMENT}_preflight.log" 2>&1; then
    echo "ABORT: frozen-weights pre-flight FAILED -- see $LOGDIR/${EXPERIMENT}_preflight.log" >&2
    exit 1
fi
echo "  passed."
echo

"${cmd[@]}" 2>&1 | tee "$LOGDIR/${EXPERIMENT}_train.log"
