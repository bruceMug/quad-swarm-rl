"""
Tier 3 round 3 — the observation channel, as a learnable action.

Rounds 1 and 2 gave the adversary four motors and nothing else, and it failed: across 79
motors-only runs (51 hand-designed Tier 2 attacks plus 28 trained Tier 3 checkpoints) exactly
2 do statistically significant damage to the honest team, and the trained adversary lands
*inside* the hand-designed band rather than below it. The optimiser was never the bottleneck.

The channel was. Tier 2's `liar` — which flies honestly and only misreports where it is —
reaches **+30.4 % honest degradation, p = 0.0007**, clearing the efficacy gate with a
hand-designed, unoptimised lie. See `analysis/channel_ceiling.py` and
`design/tier3_design.md` §5.

So round 3 extends the adversary's action from 4 motor commands to **4 + 6**: the extra six
are what it *claims* its position and velocity are. This module holds the two pieces that
training and evaluation must agree on exactly:

  * `decode_lie` -- action -> reported state. Used by `FrozenTeamEnv` during training and by
    `agents/tier3_learned.py` during evaluation.
  * `apply_lies_to_obs` -- reported state -> corrupted neighbour observations. Used during
    training; at evaluation the equivalent lives in
    `wrappers/byzantine_env.py::_apply_broadcast_lies`, which every Tier 2 `liar` run already
    went through.

**The lie is parameterised as a bounded delta from the truth, not as an absolute position.**
`reported = true + max_lie * a` with `a in [-1, 1]`. Three reasons, and the first is the one
that matters:

1. A zero action is a *truthful* broadcast. An absolute parameterisation would make honesty a
   specific non-zero output the policy has to find and then track as the drone moves; here it
   is the origin. Round 2's policies saturated to the action bounds within 50M steps, so the
   behaviour at the extremes of this range is not hypothetical -- it needs to be meaningful.
2. Sample Factory's continuous policy emits unbounded Gaussian samples that the env clips.
   Scoring or executing anything unbounded is what destroyed the round-1 stealth run.
3. It bounds the attack budget explicitly, which is what makes an `epsilon`-style stealth
   constraint on the *lie* expressible later.
"""

import numpy as np

# [x, y, z, vx, vy, vz] -- one broadcast state.
LIE_DIM = 6

# Defaults chosen to span exactly the expressible range and no further. The env clips
# neighbour observations to `clip_neighbor_space_{min,max}_box`, +/-10 m and +/-6 m/s on this
# task, so a reported offset larger than that is indistinguishable from one sitting at the
# clip -- extra range would just be dead action space the policy has to learn to avoid.
DEFAULT_MAX_LIE_POS = 10.0
DEFAULT_MAX_LIE_VEL = 6.0

# Round 5: absolute [x, y, z, vx, vy, vz] appended to the adversary's observation.
EGO_AUG_DIM = 6


def augment_obs(obs_row, pos, vel, max_pos=DEFAULT_MAX_LIE_POS, max_vel=DEFAULT_MAX_LIE_VEL):
    """
    Append the adversary's own ABSOLUTE position and velocity to its observation.

    Round 4 (seed 3) exposed an information asymmetry that made the whole trained-vs-
    hand-designed comparison unfair. The hand-designed `liar` is handed its true absolute
    state by the wrapper (`perturb_broadcast_state(true_state, t)`), so a lie like `mirror`
    -- report `2*centre - pos` -- is trivial for it to compute. The learned adversary saw
    only the ordinary 54-D observation, which carries `pos - goal`, `vel`, `R`, `omega` and
    *relative* neighbour offsets. **Absolute position appears nowhere in it.**

    Measured on a clean seed-3 rollout, a linear probe recovers absolute position from the
    54-D observation at R^2 = 0.27-0.39 (RMSE ~1 m against a ~1.2 m spread) -- i.e. barely.
    Meanwhile the `mirror` displacement is within the 10 m action budget on 99.2 % of steps.
    The action space could express the attack; the observation could not specify it.

    So this is restoring parity with the baseline it is being measured against, not granting
    the adversary an advantage. It still learns nothing about any other drone -- exactly the
    scope of `agents/base.py::EgoState`, which Tier 2's goal-directed agents already get.

    Clipped to the same ranges the env uses for neighbour observations, so the appended
    values sit on the scale the encoder already expects (`normalize_input=False`).
    """
    obs_row = np.asarray(obs_row, dtype=np.float32).reshape(-1)
    aug = np.concatenate([
        np.clip(np.asarray(pos, dtype=np.float64).reshape(-1)[:3], -max_pos, max_pos),
        np.clip(np.asarray(vel, dtype=np.float64).reshape(-1)[:3], -max_vel, max_vel),
    ]).astype(np.float32)
    return np.concatenate([obs_row, aug])


def decode_lie(lie_action, true_state, max_pos=DEFAULT_MAX_LIE_POS,
               max_vel=DEFAULT_MAX_LIE_VEL):
    """
    Turn a 6-vector of policy outputs into the state this drone broadcasts.

    Args:
        lie_action: (6,) in [-1, 1] (clipped here, so an unbounded policy sample is safe).
        true_state: (6,) = [x, y, z, vx, vy, vz], the drone's actual state.

    Returns:
        (6,) reported state. `lie_action == 0` reproduces `true_state` exactly.
    """
    a = np.clip(np.asarray(lie_action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    if a.size != LIE_DIM:
        raise ValueError(f'lie action must have {LIE_DIM} elements, got {a.size}')
    true_state = np.asarray(true_state, dtype=np.float64).reshape(-1)
    if true_state.size != LIE_DIM:
        raise ValueError(f'true_state must have {LIE_DIM} elements, got {true_state.size}')

    scale = np.concatenate([np.full(3, float(max_pos)), np.full(3, float(max_vel))])
    return true_state + scale * a


def lie_magnitude(reported, true_state):
    """Positional distance between claim and truth, in metres. Logged, not penalised."""
    return float(np.linalg.norm(np.asarray(reported)[:3] - np.asarray(true_state)[:3]))


def apply_lies_to_obs(obs, reported, neighbor_indices, pos, vel, ego_dim,
                      clip_lo, clip_hi, liar_indices):
    """
    Rewrite what every honest drone sees of each liar. Mirrors
    `wrappers/byzantine_env.py::_apply_broadcast_lies` exactly.

    Args:
        obs: (n_agents, obs_dim) observation block, modified on a copy.
        reported: {drone index -> (6,) reported state} for the liars.
        neighbor_indices: the env's OWN `neighborhood_indices()` output.
        pos, vel: (n_agents, 3) true states, for computing relative vectors.
        ego_dim: width of the ego block before the neighbour slots (18 here).
        clip_lo, clip_hi: the env's `clip_neighbor_space_{min,max}_box`.
        liar_indices: drones whose own observation must be left untouched.

    `slot` is a **proximity rank, not an agent id**. `neighborhood_indices()` re-sorts each
    drone's neighbours every step and keeps only the closest few, so a liar's position inside
    a given observer's vector changes step to step and on some steps it is absent entirely.
    Reimplementing that sort is the classic way to make this attack silently do nothing, so
    the ordering is always taken from the env itself.

    Only the *values* are corrupted, never which neighbours an observer attends to --
    selection still runs on true positions, because it happens inside the simulator.
    """
    obs = np.array(obs, copy=True)
    liars = set(int(i) for i in liar_indices)
    rewrites = 0
    for observer in range(obs.shape[0]):
        if observer in liars:
            continue                       # a liar's own observation is untouched
        for slot, j in enumerate(neighbor_indices[observer]):
            j = int(j)
            if j not in reported:
                continue
            base = ego_dim + LIE_DIM * slot
            obs[observer, base:base + 3] = np.clip(
                reported[j][:3] - pos[observer], clip_lo[:3], clip_hi[:3])
            obs[observer, base + 3:base + 6] = np.clip(
                reported[j][3:] - vel[observer], clip_lo[3:6], clip_hi[3:6])
            rewrites += 1
    return obs, rewrites


# --------------------------------------------------------------- D3: plausible lies (2026-09-26)
#
# The free lie above is effective and trivially detectable. Measured on the best adversary's
# 200-episode confirmatory run, its broadcast position was outside the 10 x 10 x 10 m room on
# 100 % of steps and below the floor on 76 %, and its reported speed averaged 9.6 m/s against a
# true 1.1 m/s: it evades every action-level detector (it flies the honest policy bit-for-bit)
# and loses to a bounds check. The spec's stealth condition exists precisely so the Tier 3
# adversary is not dismissible by a naive detector.
#
# A plausible lie is a bounded, SMOOTH drift of the claim away from the truth. The policy steers
# the drift's acceleration; the claim is truth + offset:
#
#     u_t = clip_norm(u_{t-1} + a * max_accel * dt,  max_rate)     drift velocity
#     o_t = o_{t-1} + u_t * dt,  braked radially so |o| <= max_offset without a hard stop,
#           and limited per axis so the claim stays inside the room (shrunk by a margin) and is
#           never pushed further out than the truth
#     reported_pos = true_pos + o_t,   reported_vel = true_vel + (o_t - o_{t-1}) / dt
#
# so on every step the broadcast passes the naive physics checks by construction:
#   1. bounds       -- inside the room, above the floor;
#   2. speed        -- never faster than the truth plus max_rate (near a wall only slower);
#   3. consistency  -- the reported position moves exactly as the reported velocity says;
#   4. acceleration -- the claim's acceleration differs from the truth's by at most max_accel,
#                      except in the rare steps where a wall limit engages (measured and
#                      reported by the stealth analysis, not hidden).
# The offset starts at zero with zero drift, so every episode begins with a truthful broadcast.
#
# Budget, the analogue of the spec's action-space epsilon, set from measured honest flight
# (clean seed 3, 60 episodes x 8 drones, 100 Hz): speed p50 0.47 m/s, p99 4.95; acceleration
# p50 1.2 m/s^2, p90 4.6, p99 11.6.
#   max_offset 2.0 m     -- the scale of ordinary GNSS error;
#   max_rate   1.0 m/s   -- the honest swarm's typical speed, so the drift is never faster than
#                           normal flight;
#   max_accel  4.0 m/s^2 -- inside the honest 90th percentile, so the claim's acceleration is
#                           unremarkable. (A first-order drift steered directly by the policy
#                           jittered at a median ~145 m/s^2 in stress tests -- a detector's gift.)

LIE_MODES = ('free', 'plausible')
DEFAULT_PLAUSIBLE_MAX_OFFSET = 2.0
DEFAULT_PLAUSIBLE_MAX_RATE = 1.0
DEFAULT_PLAUSIBLE_MAX_ACCEL = 4.0
DEFAULT_PLAUSIBLE_MARGIN = 0.25     # metres kept inside every room face, floor included


def lie_action_dim(mode):
    """Width of the broadcast half of the action: 6 for a free lie, 3 for a plausible drift."""
    if mode not in LIE_MODES:
        raise ValueError(f'lie mode must be one of {LIE_MODES}, got {mode!r}')
    return LIE_DIM if mode == 'free' else 3


class PlausibleLie:
    """
    One liar's stateful claim. Shared by `FrozenTeamEnv` (training) and `LearnedAdversaryAgent`
    (evaluation), so both produce the identical broadcast for the identical action sequence.

    Call `reset()` at every episode start and `step(action, true_state)` once per control step,
    after the simulator has stepped (the same point the free lie is applied at).
    """

    def __init__(self, room_box, control_dt, max_offset=DEFAULT_PLAUSIBLE_MAX_OFFSET,
                 max_rate=DEFAULT_PLAUSIBLE_MAX_RATE, max_accel=DEFAULT_PLAUSIBLE_MAX_ACCEL,
                 margin=DEFAULT_PLAUSIBLE_MARGIN):
        box = np.asarray(room_box, dtype=np.float64).reshape(2, 3)
        self.lo, self.hi = box[0] + margin, box[1] - margin
        if np.any(self.lo >= self.hi):
            raise ValueError(f'room {box.tolist()} too small for margin {margin}')
        self.dt = float(control_dt)
        if self.dt <= 0:
            raise ValueError(f'control_dt must be positive, got {control_dt}')
        self.max_offset = float(max_offset)
        self.max_rate = float(max_rate)
        self.max_accel = float(max_accel)
        self.reset()

    def reset(self):
        self.offset = np.zeros(3)
        self.drift = np.zeros(3)
        self.wall_limited_steps = 0
        self.steps = 0

    def step(self, lie_action, true_state):
        a = np.clip(np.asarray(lie_action, dtype=np.float64).reshape(-1)[:3], -1.0, 1.0)
        an = float(np.linalg.norm(a))
        if an > 1.0:                       # unit ball: |drift accel| <= max_accel exactly
            a = a / an
        true_state = np.asarray(true_state, dtype=np.float64).reshape(-1)
        pos, vel = true_state[:3], true_state[3:6]

        u = self.drift + a * self.max_accel * self.dt
        un = float(np.linalg.norm(u))
        if un > self.max_rate:
            u *= self.max_rate / un

        # Radial braking: limit outward drift to what can still be stopped within the remaining
        # offset budget at max_accel, so the cap is approached smoothly instead of hit.
        r = float(np.linalg.norm(self.offset))
        if r > 1e-12:
            rhat = self.offset / r
            ur = float(u @ rhat)
            allowed = np.sqrt(2.0 * self.max_accel * max(self.max_offset - r, 0.0))
            if ur > allowed:
                u = u - (ur - allowed) * rhat

        o = self.offset + u * self.dt
        n = float(np.linalg.norm(o))
        if n > self.max_offset:            # numerical safety; braking should prevent this
            o *= self.max_offset / n

        # Room: an interval per axis that always contains 0 -- never push the claim further out
        # than the truth, and let the allowed outward offset shrink as the truth nears a wall.
        o_room = np.clip(o, np.minimum(0.0, self.lo - pos), np.maximum(0.0, self.hi - pos))
        if np.any(np.abs(o_room - o) > 1e-12):
            self.wall_limited_steps += 1
        self.steps += 1

        u_eff = (o_room - self.offset) / self.dt
        self.offset, self.drift = o_room, u_eff
        return np.concatenate([pos + o_room, vel + u_eff])
