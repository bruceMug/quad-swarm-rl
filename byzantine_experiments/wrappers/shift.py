"""
Distribution shift for Phase 3 rollouts (D5, 2026-09-26): none / mild DR / severe DR + wind.

The spec's Phase 3 table crosses every condition with three shift levels, so the detector can be
tested away from the conditions the honest policy was trained in. Two mechanisms, neither of
which edits upstream code (the project rule since Tier 0):

**Dynamics randomisation (DR).** Upstream already implements it: each `QuadrotorSingle` can
resample its physical parameters (mass, inertia, arm length, motor constants, ...) at every
reset through a `RelativeSampler`, which draws each parameter from a normal around its nominal
value with a relative spread `noise_ratio`. `swarm_rl/env_wrappers/quad_utils.py` hard-codes
that off (`dyn_randomization_ratio = None`). This wrapper switches it on per drone, building the
sampler exactly as upstream's own constructor would, and sets `dynamics_randomize_every = 1` so
every episode gets fresh dynamics. The honest policy was trained on nominal dynamics, so this is
a genuine train/test shift.

**Wind.** Upstream has none. It is added here as a uniform external acceleration applied to
every drone after each control step:

    wind_t = w0 + g_t,     g_t an Ornstein-Uhlenbeck gust per axis (std `gust`, time const `tau`)
    vel   += wind_t * control_dt

with `w0` a horizontal mean wind of magnitude `wind_accel` in a random direction per episode.
It is applied between steps by adding to `dynamics.vel`, which the next integration step reads.

**Pairing is preserved.** Every random draw here comes from the global numpy stream after
`collect.py::seed_every_rng` has seeded it for the episode (DR inside upstream's reset, the wind
seed right after), so a clean run and an attacked run at one `--seed` and one shift level see
identical dynamics and identical wind, episode for episode. Runs at *different* shift levels
share spawns but not conditions, so `analysis/paired.py` refuses to pair across levels.

**Levels are defined by effect, not by parameter.** "Mild" and "severe" are the settings chosen
by `analysis/shift_calibration.py` from their measured damage to the *clean* swarm, before any
attack is evaluated under them, and stored in `results_v2/shift_levels.json`. Explicit
parameters can always be passed instead.
"""

import copy
import json
import os

import gymnasium as gym
import numpy as np

SHIFT_LEVELS_FILE = 'shift_levels.json'     # under the results root


def load_shift_levels(results_root):
    """{'mild': {...}, 'severe': {...}} as chosen by the calibration, or {} if not calibrated."""
    path = os.path.join(results_root, SHIFT_LEVELS_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get('levels', {})


def shift_params(level=None, dr_ratio=0.0, wind_accel=0.0, gust=0.0, tau=1.0, results_root=None):
    """Resolve a named level (from the calibration file) or explicit parameters into one dict."""
    if level and level != 'none':
        levels = load_shift_levels(results_root) if results_root else {}
        if level not in levels:
            raise ValueError(f'shift level {level!r} is not calibrated yet -- run '
                             'scripts/byz_calibrate_shift.sh, or pass explicit parameters')
        p = dict(levels[level])
        p['level'] = level
        return p
    p = dict(dr_ratio=float(dr_ratio), wind_accel=float(wind_accel), gust=float(gust),
             tau=float(tau))
    p['level'] = 'none' if not any((dr_ratio, wind_accel, gust)) else 'custom'
    return p


class ShiftWrapper(gym.Wrapper):
    """Applies DR (at every reset, inside upstream) and wind (after every step)."""

    def __init__(self, env, dr_ratio=0.0, wind_accel=0.0, gust=0.0, tau=1.0, **_):
        super().__init__(env)
        self.dr_ratio = float(dr_ratio)
        self.wind_accel = float(wind_accel)
        self.gust = float(gust)
        self.tau = float(tau)
        envs = self.env.unwrapped.envs
        self.control_dt = 1.0 / float(envs[0].control_freq)

        if self.dr_ratio > 0:
            from gym_art.quadrotor_multi.quadrotor_randomization import RelativeSampler
            for e in envs:
                # Same construction as QuadrotorSingle.__init__ uses for dyn_sampler_1, on the
                # nominal parameters (dynamics_change already applied) -- so this is upstream's
                # own DR, merely switched on.
                e.dyn_sampler_1 = RelativeSampler(params=copy.deepcopy(e.dynamics_params),
                                                  noise_ratio=self.dr_ratio, sampler='normal')
                e.dynamics_randomize_every = 1
        self._w0 = np.zeros(3)
        self._g = np.zeros(3)
        self._rng = None

    def _new_wind(self):
        if self.wind_accel <= 0 and self.gust <= 0:
            self._rng = None
            return
        # Drawn from the per-episode-seeded global stream, so both arms get the same wind.
        self._rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
        theta = self._rng.uniform(0, 2 * np.pi)
        self._w0 = self.wind_accel * np.array([np.cos(theta), np.sin(theta), 0.0])
        self._g = np.zeros(3)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._new_wind()
        return out

    def step(self, action):
        out = self.env.step(action)
        if self._rng is None:
            return out
        terminated, truncated = out[2], out[3]
        done = np.any(np.asarray(terminated)) or np.any(np.asarray(truncated))
        if done:
            return out   # the env has already auto-reset; the next episode's wind starts at reset
        dt = self.control_dt
        if self.gust > 0:
            # Exact OU discretisation: stationary std = gust, correlation time = tau.
            a = np.exp(-dt / self.tau)
            self._g = a * self._g + self.gust * np.sqrt(1 - a * a) * self._rng.normal(size=3)
        wind = self._w0 + self._g
        for e in self.env.unwrapped.envs:
            e.dynamics.vel = np.asarray(e.dynamics.vel, dtype=np.float64) + wind * dt
        return out
