"""
Characterisation metrics computed from a collected rollout `.npz`.

Everything here operates on the padded arrays written by
`byzantine_experiments.rollout.collect` and respects `episode_lengths`, so trailing NaN
padding never leaks into a mean.

Terminology, because two of these are easy to misread:

* **goal RMSE** is what the spec calls "formation RMSE". The env's observation encodes
  `pos - goal` in dims 0:3 (see gym_art/quadrotor_multi/get_state.py), and in `mix` mode the
  per-drone goals *are* the formation, so RMS distance-to-goal is the formation error. It is
  reported twice: over all drones, and over honest drones only. The honest-only number is
  the one that measures damage *to the team*, since the compromised drone's own error is a
  direct consequence of the fault rather than a downstream effect of it.
* **team reward** likewise comes in both flavours. `episode_reward` averages all 8 drones
  (the convention `scripts/evaluate.py` and the baseline JSONs use, so it is the comparable
  one); `episode_reward_honest` averages only the uncompromised drones.
"""

import numpy as np

EGO_GOAL_SLICE = slice(0, 3)  # obs dims 0:3 = pos - goal
SATURATION_EPS = 1e-4
FLOOR_Z = 0.15          # below this a drone counts as grounded
GUARD_Z = 0.60          # Tier 2 survival guard engages below this altitude
CURVE_STRIDE = 10       # downsampling of the cumulative-reward curve (100 Hz -> 10 Hz)


def _step_mask(data):
    """(n_episodes, T) bool: True where the step is real data rather than NaN padding."""
    lengths = np.asarray(data['episode_lengths'])
    t_idx = np.arange(data['rewards'].shape[1])[None, :]
    return t_idx < lengths[:, None]


def _masked_mean(values, mask):
    """Mean of `values` over entries where `mask` (broadcast over trailing dims) is True."""
    values = np.asarray(values, dtype=np.float64)
    m = np.broadcast_to(mask.reshape(mask.shape + (1,) * (values.ndim - mask.ndim)), values.shape)
    sel = values[m]
    return float(np.mean(sel)) if sel.size else float('nan')


def compute_metrics(data, action_low=-1.0, action_high=1.0):
    """
    Full characterisation of one rollout run.

    Args:
        data: mapping with the keys written by collect.py (an open np.load handle is fine).
        action_low/action_high: the env action range, for the saturation calculation.

    Returns a flat dict of scalars plus a `per_episode` dict of (n_episodes,) arrays.
    """
    rewards = np.asarray(data['rewards'], dtype=np.float64)          # (E, T, N)
    obs = np.asarray(data['obs'], dtype=np.float64)                  # (E, T, N, D)
    executed = np.asarray(data['actions_executed'], dtype=np.float64)  # (E, T, N, A)
    honest = np.asarray(data['actions_honest'], dtype=np.float64)
    positions = np.asarray(data['positions'], dtype=np.float64)      # (E, T, N, 3)
    collisions = np.asarray(data['collisions'], dtype=np.float64)    # (E, T)
    byz_mask = np.asarray(data['byzantine_mask'], dtype=bool)        # (N,)
    lengths = np.asarray(data['episode_lengths'], dtype=np.int64)

    n_episodes, _, n_agents = rewards.shape
    honest_mask = ~byz_mask
    step_mask = _step_mask(data)

    # ---- episode reward (sum over steps, mean over drones) --------------------------
    rew_zeroed = np.where(step_mask[..., None], rewards, 0.0)
    ep_reward_per_agent = rew_zeroed.sum(axis=1)                     # (E, N)
    ep_reward = ep_reward_per_agent.mean(axis=1)
    ep_reward_honest = (ep_reward_per_agent[:, honest_mask].mean(axis=1)
                        if honest_mask.any() else np.full(n_episodes, np.nan))
    ep_reward_byz = (ep_reward_per_agent[:, byz_mask].mean(axis=1)
                     if byz_mask.any() else np.full(n_episodes, np.nan))

    # ---- collisions -----------------------------------------------------------------
    coll_zeroed = np.where(step_mask, collisions, 0.0)
    ep_collisions = coll_zeroed.sum(axis=1)

    # ---- goal / formation error -----------------------------------------------------
    goal_err = np.linalg.norm(obs[..., EGO_GOAL_SLICE], axis=-1)     # (E, T, N)
    ep_goal_rmse = np.sqrt(_per_episode_mean(goal_err ** 2, step_mask))
    ep_goal_rmse_honest = np.sqrt(_per_episode_mean(goal_err[..., honest_mask] ** 2, step_mask)) \
        if honest_mask.any() else np.full(n_episodes, np.nan)
    ep_goal_rmse_byz = np.sqrt(_per_episode_mean(goal_err[..., byz_mask] ** 2, step_mask)) \
        if byz_mask.any() else np.full(n_episodes, np.nan)

    # ---- action deviation on the compromised drones ---------------------------------
    dev = np.linalg.norm(executed - honest, axis=-1)                 # (E, T, N)
    if byz_mask.any():
        ep_dev_byz = _per_episode_mean(dev[..., byz_mask], step_mask)
        dev_byz_mean = _masked_mean(dev[..., byz_mask], step_mask)
        dev_byz_max = float(np.nanmax(np.where(step_mask[..., None], dev[..., byz_mask], np.nan)))
        perturbed_frac = _masked_mean((dev[..., byz_mask] > SATURATION_EPS).astype(float), step_mask)
    else:
        ep_dev_byz = np.zeros(n_episodes)
        dev_byz_mean = dev_byz_max = perturbed_frac = 0.0
    dev_honest_mean = _masked_mean(dev[..., honest_mask], step_mask) if honest_mask.any() else 0.0

    # ---- clip saturation -------------------------------------------------------------
    at_bound = ((executed <= action_low + SATURATION_EPS)
                | (executed >= action_high - SATURATION_EPS)).astype(float)   # (E, T, N, A)
    if byz_mask.any():
        ep_sat_byz = _per_episode_mean(at_bound[..., byz_mask, :].reshape(
            n_episodes, at_bound.shape[1], -1), step_mask)
        sat_byz = _masked_mean(at_bound[..., byz_mask, :], step_mask)
    else:
        ep_sat_byz = np.zeros(n_episodes)
        sat_byz = 0.0
    sat_honest = _masked_mean(at_bound[..., honest_mask, :], step_mask) if honest_mask.any() else 0.0

    # ---- swarm geometry ---------------------------------------------------------------
    min_sep = _min_pairwise_distance(positions)                      # (E, T)
    ep_min_sep = _per_episode_mean(min_sep, step_mask)

    # ---- crash proxy: drones on/below the floor ---------------------------------------
    low_altitude = (positions[..., 2] < 0.15).astype(float)          # (E, T, N)
    ep_low_alt = _per_episode_mean(low_altitude, step_mask)

    # ---- Tier 2: attacker survival ----------------------------------------------------
    # "Survived" = airborne at the last real step of the episode. An attack that grounds its
    # own attacker removes it from the episode and yields no behavioural signal to detect.
    last_idx = np.maximum(lengths - 1, 0)
    final_z = positions[np.arange(n_episodes), last_idx][:, :, 2]    # (E, N)
    if byz_mask.any():
        ep_survived = (final_z[:, byz_mask] > FLOOR_Z).all(axis=1).astype(float)
        ep_byz_airborne = 1.0 - _per_episode_mean(low_altitude[..., byz_mask], step_mask)
        # Fraction of steps the attacker was inside the survival guard's ramp. The guard
        # fires exactly when z < guard_altitude, so this reconstructs the guard trigger rate
        # from the trajectory alone -- no need to trust (or re-run) the agent's counters.
        below_guard = (positions[..., byz_mask, 2] < GUARD_Z).astype(float)
        ep_below_guard = _per_episode_mean(below_guard, step_mask)
    else:
        ep_survived = np.full(n_episodes, np.nan)
        ep_byz_airborne = np.full(n_episodes, np.nan)
        ep_below_guard = np.full(n_episodes, np.nan)

    # ---- Tier 2: team-reward curve, for time-to-impact --------------------------------
    # Cumulative team reward (mean over drones) as a function of t, downsampled. The
    # aggregator diffs this against the clean run's curve to find the first step at which
    # the attack becomes visible; keeping it here means the .npz never has to be reopened.
    team_step_reward = np.where(step_mask[..., None], rewards, 0.0).mean(axis=2)   # (E, T)
    cum = np.cumsum(team_step_reward, axis=1)
    curve = cum[:, ::CURVE_STRIDE]
    curve_mean = curve.mean(axis=0)
    curve_std = curve.std(axis=0)

    per_episode = dict(
        episode_reward=ep_reward,
        episode_reward_honest=ep_reward_honest,
        episode_reward_byzantine=ep_reward_byz,
        collisions=ep_collisions,
        goal_rmse=ep_goal_rmse,
        goal_rmse_honest=ep_goal_rmse_honest,
        goal_rmse_byzantine=ep_goal_rmse_byz,
        action_deviation_byzantine=ep_dev_byz,
        saturation_byzantine=ep_sat_byz,
        min_separation=ep_min_sep,
        low_altitude_frac=ep_low_alt,
        attacker_survived=ep_survived,
        attacker_airborne_frac=ep_byz_airborne,
        attacker_below_guard_frac=ep_below_guard,
        episode_length=lengths.astype(float),
    )

    out = dict(
        n_episodes=int(n_episodes),
        n_agents=int(n_agents),
        n_byzantine=int(byz_mask.sum()),
        f_fraction=float(byz_mask.sum() / n_agents),
        total_steps=int(lengths.sum()),

        action_deviation_byzantine_mean=dev_byz_mean,
        action_deviation_byzantine_max=dev_byz_max,
        action_deviation_honest_mean=dev_honest_mean,
        perturbed_step_fraction=perturbed_frac,
        saturation_rate_byzantine=sat_byz,
        saturation_rate_honest=sat_honest,
        curve_stride=CURVE_STRIDE,
        team_reward_curve=curve_mean.tolist(),
        team_reward_curve_std=curve_std.tolist(),
    )
    for key, vals in per_episode.items():
        vals = np.asarray(vals, dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        # All-NaN is expected and legitimate: the *_byzantine series on a clean run with no
        # compromised drones. Report NaN without a "mean of empty slice" warning.
        if finite.size == 0:
            out[f'{key}_mean'] = out[f'{key}_std'] = out[f'{key}_sem'] = float('nan')
            continue
        out[f'{key}_mean'] = float(finite.mean())
        out[f'{key}_std'] = float(finite.std())
        out[f'{key}_sem'] = float(finite.std() / np.sqrt(finite.size))
    out['per_episode'] = per_episode
    return out


def _per_episode_mean(values, step_mask):
    """Mean over (steps, ...) per episode, honouring the padding mask. -> (n_episodes,)"""
    values = np.asarray(values, dtype=np.float64)
    m = np.broadcast_to(
        step_mask.reshape(step_mask.shape + (1,) * (values.ndim - step_mask.ndim)), values.shape)
    num = np.where(m, values, 0.0).sum(axis=tuple(range(1, values.ndim)))
    den = m.sum(axis=tuple(range(1, values.ndim)))
    return np.divide(num, den, out=np.full(len(values), np.nan), where=den > 0)


def _min_pairwise_distance(positions):
    """Closest drone-pair distance at each step. -> (n_episodes, T)"""
    diff = positions[:, :, :, None, :] - positions[:, :, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    n = positions.shape[2]
    dist[:, :, np.arange(n), np.arange(n)] = np.inf
    return dist.min(axis=(2, 3))


def degradation_vs_clean(byz, clean):
    """
    Compare a Byzantine run against the clean reference for the same baseline seed.

    Returns degradation figures plus their uncertainty. The percentage is computed as the
    spec asks -- `100 * (clean - byz) / |clean|` -- but **read `reward_degradation_pct`
    together with `reward_degradation_pct_sem`**: mean episode reward on this task is a
    small number (~1.5) sitting on a heavy left tail, so the denominator is near zero and
    the percentage is correspondingly unstable. `goal_rmse_degradation_pct` has a true zero
    and far lower relative variance, and is the more trustworthy severity axis.
    """
    out = {}

    c_mean, b_mean = clean['episode_reward_mean'], byz['episode_reward_mean']
    c_sem, b_sem = clean['episode_reward_sem'], byz['episode_reward_sem']
    drop = c_mean - b_mean
    drop_sem = float(np.hypot(c_sem, b_sem))

    out['clean_episode_reward'] = c_mean
    out['reward_drop_absolute'] = drop
    out['reward_drop_absolute_sem'] = drop_sem
    denom = abs(c_mean) if abs(c_mean) > 1e-9 else float('nan')
    out['reward_degradation_pct'] = 100.0 * drop / denom
    out['reward_degradation_pct_sem'] = 100.0 * drop_sem / denom
    out['reward_drop_sigma'] = drop / drop_sem if drop_sem > 0 else float('nan')

    # ---- Tier 2: time-to-impact -------------------------------------------------------
    # First timestep at which the cumulative team reward of the attacked run departs from
    # the clean run's by more than 2 clean standard deviations. NaN means the attack never
    # separated from clean noise within the episode -- which is itself a result.
    out['time_to_impact_steps'] = float('nan')
    out['time_to_impact_s'] = float('nan')
    b_curve = np.asarray(byz.get('team_reward_curve', []), dtype=np.float64)
    c_curve = np.asarray(clean.get('team_reward_curve', []), dtype=np.float64)
    c_std = np.asarray(clean.get('team_reward_curve_std', []), dtype=np.float64)
    stride = int(byz.get('curve_stride', 10))
    if b_curve.size and b_curve.size == c_curve.size:
        diverged = np.abs(b_curve - c_curve) > 2.0 * np.maximum(c_std, 1e-9)
        # require the divergence to persist, so a single noisy sample does not count
        run = np.convolve(diverged.astype(int), np.ones(3, dtype=int), mode='same') >= 3
        hits = np.flatnonzero(run)
        if hits.size:
            step = int(hits[0]) * stride
            out['time_to_impact_steps'] = float(step)
            out['time_to_impact_s'] = step / 100.0     # env control rate

    # ---- Tier 2: victim spillover -----------------------------------------------------
    # Damage that propagated to drones that were never compromised. This is the quantity the
    # spec calls "victim spillover", named explicitly rather than left implicit in the
    # honest-agent columns.
    c_h, b_h = clean.get('episode_reward_honest_mean'), byz.get('episode_reward_honest_mean')
    if c_h is not None and b_h is not None:
        out['victim_spillover_reward_drop'] = c_h - b_h
        out['victim_spillover_pct'] = (100.0 * (c_h - b_h) / abs(c_h)
                                       if abs(c_h) > 1e-9 else float('nan'))
    c_g, b_g = clean.get('goal_rmse_honest_mean'), byz.get('goal_rmse_honest_mean')
    if c_g is not None and b_g is not None and abs(c_g) > 1e-9:
        out['victim_spillover_goal_rmse_pct'] = 100.0 * (b_g - c_g) / abs(c_g)

    for key in ('episode_reward_honest', 'goal_rmse', 'goal_rmse_honest',
                'collisions', 'min_separation', 'low_altitude_frac',
                'attacker_survived', 'attacker_airborne_frac'):
        c, b = clean.get(f'{key}_mean'), byz.get(f'{key}_mean')
        if c is None or b is None:
            continue
        out[f'clean_{key}'] = c
        out[f'{key}_delta'] = b - c
        # For error-like metrics, "degradation" means the value going UP.
        sign = -1.0 if key in ('episode_reward_honest', 'min_separation',
                               'attacker_survived', 'attacker_airborne_frac') else 1.0
        out[f'{key}_degradation_pct'] = (100.0 * sign * (b - c) / abs(c)
                                         if abs(c) > 1e-9 else float('nan'))
    return out
