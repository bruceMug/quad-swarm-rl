"""
Tier 1 — Stochastic faults. The negative control.

These are NOT attacks. They model an honest drone degraded by hardware failure, sensor
noise, or a compute stall: no goal, no strategy, no model of the team. Tier 1 is the
"degraded" side of the degraded-vs-malicious comparison the whole project rests on, so the
hard constraints below are load-bearing:

  1. No access to team state. The perturbation may depend on this drone's own action, its
     own observation, and t -- never on neighbours, team reward, or the goal. In fact none
     of the three variants reads `obs` at all, which makes the constraint trivially
     verifiable (test_tier1.py::test_no_team_state_access).
  2. Zero-mean in expectation, before clipping (1A and 1B).
  3. No temporal goal structure. 1B is temporally correlated, which is fine -- correlation
     is not intent -- but it drifts toward nothing.
  4. Honest broadcast: none of these override perturb_broadcast_state(). A degraded drone
     reports its true state, it just flies badly. Lying is Tier 2's job.

Every variant draws from `self.rng` (seeded per agent by the wrapper), never global
np.random, so runs are reproducible and agents are independent of the sim's RNG.
"""

import numpy as np

from byzantine_experiments.agents.base import ByzantineAgent, register_byzantine


class _Tier1Base(ByzantineAgent):
    """Shared bookkeeping: records the pre-clip perturbation for diagnostics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-clipping delta actually applied on the last step. The clipping analysis and
        # the variance-scaling test both need the raw draw, not the post-clip difference.
        self.last_noise = np.zeros(self.action_dim)

    def reset(self):
        super().reset()
        self.last_noise = np.zeros(self.action_dim)

    def _finish(self, honest_action, noise):
        self.last_noise = np.asarray(noise, dtype=np.float64).copy()
        return self.clip(honest_action + noise)


@register_byzantine('tier1_gaussian')
class GaussianNoiseAgent(_Tier1Base):
    """
    Variant 1A — additive white Gaussian noise (the primary variant).

        a' = clip(a + N(0, sigma^2 I), low, high)

    Zero-mean and temporally independent: the memoryless baseline against which 1B's
    temporal correlation is the interesting contrast.

    Args:
        sigma: noise standard deviation per action dimension. Matrix values {0.1, 0.25, 0.5}.
            Note that at sigma=0.5 on a [-1, 1] action space a large fraction of samples
            saturate against the clip boundary, which biases the *effective* distribution
            toward the boundary even though the draw itself is unbiased. Measured and
            reported in design/tier1_design.md.
    """

    def __init__(self, *args, sigma=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = float(sigma)
        if self.sigma < 0:
            raise ValueError(f'sigma must be >= 0, got {self.sigma}')

    def perturb_action(self, honest_action, obs, t):
        noise = self.rng.normal(0.0, self.sigma, size=np.shape(honest_action))
        return self._finish(honest_action, noise)


@register_byzantine('tier1_ou')
class OUNoiseAgent(_Tier1Base):
    """
    Variant 1B — temporally correlated drift (Ornstein-Uhlenbeck).

        x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma_ou * sqrt(dt) * N(0, I)
        a'      = clip(a + x_t, low, high)

    More physically realistic than white noise: real sensor drift and wind gusts are
    correlated over time. This produces *sticky* degradation rather than a uniform random
    walk between abstract policy-graph states, which makes it the harder negative control —
    behaviourally it is closer to Tier 3 than 1A is. That is deliberate.

    The process is mean-reverting to mu=0, so it is zero-mean in the stationary limit and
    drifts toward nothing (constraint 3). Its stationary std is
    sigma_ou * sqrt(dt / (2*theta - theta^2*dt)) — for the defaults, notably larger than
    sigma_ou itself, so 1B severities are not directly comparable to 1A sigmas by number.

    Args:
        theta: mean-reversion rate (default 0.15 — the standard OU default, and the same
            value the simulator's own thrust-noise OU process uses in quad_utils.py).
        sigma_ou: per-step noise scale. Matrix values {0.1, 0.2, 0.4}.
        mu: reversion target, default 0.0. Anything non-zero would be a directional bias
            and would violate constraint 3.
        dt: process timestep, default 1.0 (i.e. parameters are expressed per env step
            rather than per second; the env runs at 100 Hz).
    """

    def __init__(self, *args, theta=0.15, sigma_ou=0.2, mu=0.0, dt=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = float(theta)
        self.sigma_ou = float(sigma_ou)
        self.mu = float(mu)
        self.dt = float(dt)
        self.x = np.zeros(self.action_dim)

    def reset(self):
        super().reset()
        self.x = np.zeros(self.action_dim)

    def stationary_std(self):
        """Analytic stationary std of the discretised process, for the tests and the doc."""
        decay = 1.0 - self.theta * self.dt
        return self.sigma_ou * np.sqrt(self.dt / max(1.0 - decay ** 2, 1e-12))

    def perturb_action(self, honest_action, obs, t):
        self.x = (self.x
                  + self.theta * (self.mu - self.x) * self.dt
                  + self.sigma_ou * np.sqrt(self.dt) * self.rng.normal(0.0, 1.0, size=self.action_dim))
        noise = self.x[:np.shape(honest_action)[0]]
        return self._finish(honest_action, noise)


@register_byzantine('tier1_dropout')
class DropoutAgent(_Tier1Base):
    """
    Variant 1C — intermittent dropout (communication / compute stall).

    With probability p_drop per step the agent emits a stale or dead action instead of the
    freshly computed one:

        mode='hold': a' = previous emitted action  (a frozen controller)
        mode='zero': a' = 0                        (dead output; 0 is hover in this action
                                                    space, since zero_action_middle=True)

    Unlike 1A/1B this is NOT zero-mean — a hold or a zero is a systematic departure, not a
    symmetric jitter — which is why the zero-mean test covers only 1A and 1B. It is still a
    fault and not an attack: the choice of when to stall is i.i.d. and depends on nothing.

    Args:
        p_drop: per-step dropout probability. Matrix values {0.05, 0.15, 0.3}.
        mode: 'hold' or 'zero'.
    """

    def __init__(self, *args, p_drop=0.15, mode='hold', **kwargs):
        super().__init__(*args, **kwargs)
        self.p_drop = float(p_drop)
        if not 0.0 <= self.p_drop <= 1.0:
            raise ValueError(f'p_drop must be in [0, 1], got {self.p_drop}')
        if mode not in ('hold', 'zero'):
            raise ValueError(f"mode must be 'hold' or 'zero', got {mode!r}")
        self.mode = mode

        self.prev_action = np.zeros(self.action_dim)
        self.dropped_steps = 0
        self.total_steps = 0
        self.last_dropped = False

    def reset(self):
        super().reset()
        # Hold-at-hover on a dropout in the first step of an episode, rather than repeating
        # whatever the drone was doing at the end of the previous one.
        self.prev_action = np.zeros(self.action_dim)
        self.dropped_steps = 0
        self.total_steps = 0
        self.last_dropped = False

    @property
    def empirical_drop_rate(self):
        return self.dropped_steps / self.total_steps if self.total_steps else 0.0

    def perturb_action(self, honest_action, obs, t):
        self.total_steps += 1
        honest_action = np.asarray(honest_action)

        if self.rng.random() < self.p_drop:
            self.dropped_steps += 1
            self.last_dropped = True
            if self.mode == 'hold':
                emitted = self.prev_action[:honest_action.shape[0]].copy()
            else:
                emitted = np.zeros_like(honest_action, dtype=np.float64)
        else:
            self.last_dropped = False
            emitted = honest_action.astype(np.float64, copy=True)

        self.last_noise = emitted - honest_action
        out = self.clip(emitted)
        self.prev_action = np.asarray(out, dtype=np.float64).copy()
        return out


@register_byzantine('tier1_stall')
class BurstStallAgent(_Tier1Base):
    """
    Variant 1C' — dropout re-parameterised by stall *duration* (a burst stall).

    Why this exists
    ---------------
    `tier1_dropout` draws a stall independently every step, so at the env's 100 Hz control
    rate the average stall lasts 1/(1-p_drop) steps -- 1.4 steps (14 ms) at p_drop=0.3.
    Consecutive policy actions are nearly identical at that timescale, so reusing a 14 ms-old
    command is barely a fault at all: measured mean action deviation was 0.001-0.008, against
    0.19-0.84 for Gaussian noise. Empirically 1C never entered the 20-40 % damage band at any
    p tested -- it jumped from 16.4 % (p=0.85) straight to 65.4 % (p=0.98).

    The diagnosis is that `p_drop` controls how *often* a stall begins, when what matters
    physically is how *long* it lasts. A dropped radio packet or a compute hang lasts
    hundreds of milliseconds, not one control tick.

    Model
    -----
    A two-state Markov chain over {OK, STALL}, so stalls have a realistic geometric duration
    instead of collapsing to single steps:

        OK    --p_enter-->   STALL
        STALL --p_recover--> OK

    Parameterised by the two things that are physically meaningful, rather than by the
    transition probabilities:

        mean_stall_s : mean stall duration in SECONDS   -> p_recover = 1 / (mean_stall_s * f_ctrl)
        duty_cycle   : long-run fraction of time stalled -> p_enter  = p_recover * d / (1 - d)

    which gives stationary occupancy exactly `duty_cycle` and mean stall length exactly
    `mean_stall_s`. Setting mean_stall_s = 1/((1-p)*f_ctrl) and duty_cycle = p recovers the
    old i.i.d. `tier1_dropout` behaviour, so the two variants are nested rather than rival.

    Still a fault, not an attack: the transitions depend on nothing but the agent's own RNG,
    it reads no observation, and it does not override `perturb_broadcast_state`.

    Args:
        mean_stall_s: mean stall duration, seconds. Suggested grid {0.1, 0.3, 1.0}.
        duty_cycle: long-run fraction of steps spent stalled, in [0, 1). Grid {0.1, 0.25, 0.5}.
        mode: 'hold' (freeze the last command) or 'zero' (dead output = hover).
        control_freq: env control rate in Hz, used to convert seconds to steps. The
            quadrotor env runs at 100 Hz; this is exposed rather than hardcoded so the
            variant survives a change of control rate.
    """

    def __init__(self, *args, mean_stall_s=0.5, duty_cycle=0.25, mode='hold',
                 control_freq=100.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.mean_stall_s = float(mean_stall_s)
        self.duty_cycle = float(duty_cycle)
        self.control_freq = float(control_freq)
        if self.mean_stall_s <= 0:
            raise ValueError(f'mean_stall_s must be > 0, got {self.mean_stall_s}')
        if not 0.0 <= self.duty_cycle < 1.0:
            raise ValueError(f'duty_cycle must be in [0, 1), got {self.duty_cycle}')
        if mode not in ('hold', 'zero'):
            raise ValueError(f"mode must be 'hold' or 'zero', got {mode!r}")
        self.mode = mode

        self.mean_stall_steps = self.mean_stall_s * self.control_freq
        # A stall shorter than one step is not representable; clamp rather than silently
        # producing p_recover > 1.
        self.p_recover = min(1.0, 1.0 / max(self.mean_stall_steps, 1.0))
        self.p_enter = (self.p_recover * self.duty_cycle / (1.0 - self.duty_cycle)
                        if self.duty_cycle > 0 else 0.0)

        self.stalled = False
        self.prev_action = np.zeros(self.action_dim)
        self.stalled_steps = 0
        self.total_steps = 0
        self.stall_lengths = []      # completed stall durations, in steps
        self._current_stall_len = 0

    def reset(self):
        super().reset()
        self.stalled = False
        self.prev_action = np.zeros(self.action_dim)
        self.stalled_steps = 0
        self.total_steps = 0
        self.stall_lengths = []
        self._current_stall_len = 0

    @property
    def empirical_duty_cycle(self):
        return self.stalled_steps / self.total_steps if self.total_steps else 0.0

    @property
    def empirical_mean_stall_steps(self):
        return float(np.mean(self.stall_lengths)) if self.stall_lengths else 0.0

    def perturb_action(self, honest_action, obs, t):
        self.total_steps += 1
        honest_action = np.asarray(honest_action)

        # Advance the chain first, then emit according to the state we are now in.
        if self.stalled:
            if self.rng.random() < self.p_recover:
                self.stalled = False
                self.stall_lengths.append(self._current_stall_len)
                self._current_stall_len = 0
        else:
            if self.rng.random() < self.p_enter:
                self.stalled = True

        if self.stalled:
            self.stalled_steps += 1
            self._current_stall_len += 1
            if self.mode == 'hold':
                emitted = self.prev_action[:honest_action.shape[0]].copy()
            else:
                emitted = np.zeros_like(honest_action, dtype=np.float64)
        else:
            emitted = honest_action.astype(np.float64, copy=True)

        self.last_noise = emitted - honest_action
        out = self.clip(emitted)
        # Only a fresh command refreshes the hold buffer; otherwise a long stall would decay
        # toward whatever it emitted rather than freezing the last GOOD command.
        if not self.stalled:
            self.prev_action = np.asarray(out, dtype=np.float64).copy()
        return out
