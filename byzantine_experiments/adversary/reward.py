"""
Tier 3 — the adversarial reward.

    r_adv = -w_team * r_team_mean
            - w_crash * I[adversary crashed]
            + w_alive * I[alive]
            - w_stealth * max(0, ||a_adv - mu_honest||_2 - epsilon)

Every component is returned separately so wandb can show which one is driving learning,
which the spec asks for explicitly.

Two deliberate departures from the spec's literal form, both measured rather than guessed.

Departure 1: the crash penalty fires on crash **onset**, not every step
----------------------------------------------------------------------
`- w_crash * I[adversary crashed]` reads as a per-step indicator, but "crashed" is a
*persistent state*: once a drone is on the floor it stays flagged. Charging `w_crash` every
step for the rest of the episode makes a single crash at t=100 cost `10 * 1400 = 14000`.

Measured over a 600-step rollout (`scripts` aside, reproducible with
`analysis/calibrate_reward.py`), episode-integrated magnitudes at the spec's weights
(`w_team=1, w_crash=10`, 1501-step episodes):

    |team term|                        4.61
    crash term, per-step (as written)  12083.05     <-- 2622 : 1
    crash term, per-onset               ~200        <-- 43 : 1

At 2622:1 the objective is 0.04% of the reward signal. PPO would learn "hover, never crash"
and nothing else -- precisely the spec's own "adversary does nothing (< 10% degradation)"
failure mode. Charging the penalty once per crash *event* keeps the deterrent (one crash
still costs ~2x an entire episode's achievable team damage for a competent adversary) while
leaving the objective visible. `crash_penalty_mode='per_step'` reproduces the literal
reading.

Departure 2: `team_reward_mode` defaults to `'honest'`
------------------------------------------------------
The spec writes `r_team_mean` without saying whether the mean is over all 8 drones or only
the honest ones. That ambiguity matters enormously here, and the Tier 1/2 calibration
settles it (see design/tier2_design.md section 10):

* Clean mean episode reward across all 8 drones is only **+0.861**.
* Any drone that damages *itself* drags that mean deeply negative. A Tier 1 Gaussian *noise
  fault* -- not an attack at all -- registers 214% "team reward degradation" on the all-8
  mean; Tier 2 collusion reaches 1273%.

So an adversary optimising the all-8 mean gets almost all of its reward from wrecking its
own drone. That is the degenerate suicide solution `w_crash` exists to suppress, and paying
`w_crash` to undo an incentive created by the objective itself is a bad trade -- the
adversary would spend its capacity learning to crash *just slowly enough* not to be
penalised.

Averaging over the **honest** drones removes the incentive at the source: the adversary
scores only by damaging drones it does not control. `'all'` is kept for comparison and to
reproduce the spec's literal reading.
"""

import numpy as np

DEFAULTS = dict(w_team=20.0, w_crash=10.0, w_alive=0.01, w_stealth=0.0, epsilon=0.3)


class AdversarialReward:
    """
    Args:
        w_team: weight on negated team reward (the objective). Default raised from the
            spec's 1.0 to **20.0**. Per-step team reward in this env has magnitude ~0.003,
            so at w_team=1 the objective integrates to ~4.6 per episode against a crash term
            of ~150 even in onset mode (32:1). At w_team=20 the measured balance is 1.5:1
            against a *randomly flailing* adversary that crashes 15x per episode; a competent
            adversary crashes far less, so the objective dominates once it stops suiciding,
            which is the intended gradient. Verify any change with
            `python -m byzantine_experiments.adversary.calibrate_reward`.
        w_crash: penalty when the adversary is in a crashed state. Suppresses the degenerate
            "suicide immediately" solution, which is effective but trivially detectable and
            removes the adversary from the episode.
        w_alive: small per-step survival bonus, favouring persistent disruption over a
            one-shot sacrifice. Default lowered from the spec's 0.1 to **0.01**: at 0.1 the
            bonus integrates to 150 per episode against a team term of ~4.6, so it too
            swamps the objective.
        crash_penalty_mode: 'onset' (default) charges w_crash once per crash event;
            'per_step' charges it every step the adversary is in a crashed state, which is
            the spec's literal wording. See the module docstring for why onset.
        w_stealth: weight on the action-deviation hinge. 0 = unconstrained adversary.
        epsilon: attack budget -- deviation from the honest action mean that is free.
        mu_honest: (act_dim,) mean honest action from the clean baseline rollouts. Defaults
            to zeros, which in this action space is hover.
        team_reward_mode: 'honest' (default, recommended) or 'all'. See module docstring.
    """

    def __init__(self, w_team=20.0, w_crash=10.0, w_alive=0.01, w_stealth=0.0,
                 epsilon=0.3, mu_honest=None, team_reward_mode='honest',
                 crash_penalty_mode='onset'):
        self.w_team = float(w_team)
        self.w_crash = float(w_crash)
        self.w_alive = float(w_alive)
        self.w_stealth = float(w_stealth)
        self.epsilon = float(epsilon)
        if team_reward_mode not in ('honest', 'all'):
            raise ValueError(f"team_reward_mode must be 'honest' or 'all', got {team_reward_mode!r}")
        self.team_reward_mode = team_reward_mode
        if crash_penalty_mode not in ('onset', 'per_step'):
            raise ValueError(f"crash_penalty_mode must be 'onset' or 'per_step', "
                             f"got {crash_penalty_mode!r}")
        self.crash_penalty_mode = crash_penalty_mode
        self.mu_honest = None if mu_honest is None else np.asarray(mu_honest, dtype=np.float64)

        self.episode_components = None
        self.reset()

    def reset(self):
        self.episode_components = dict(team=0.0, crash=0.0, alive=0.0, stealth=0.0, total=0.0)
        self._was_crashed = None
        self.crash_onsets = 0

    def stealth_penalty(self, adv_actions):
        """
        Hinge on L2 distance from the honest action mean, per adversary, averaged.

        `adv_actions` must be the **executed** (clipped) action, not the raw policy sample --
        the caller is responsible for clipping. The deviation is always measured, even when
        `w_stealth == 0`, because it is the diagnostic that tells you whether an
        *unconstrained* adversary is winning by brute action magnitude. Returning 0 there
        (the original behaviour) blinded exactly the run that needed watching.
        """
        mu = self.mu_honest
        if mu is None:
            mu = np.zeros(np.shape(adv_actions)[1], dtype=np.float64)
        dev = np.linalg.norm(np.asarray(adv_actions, dtype=np.float64) - mu[None, :], axis=1)
        if self.w_stealth == 0.0:
            return 0.0, float(np.mean(dev))
        excess = np.maximum(0.0, dev - self.epsilon)
        return float(np.mean(excess)), float(np.mean(dev))

    def __call__(self, team_rewards, adv_indices, honest_indices, adv_actions,
                 crashed, positions=None, t=0):
        team_rewards = np.asarray(team_rewards, dtype=np.float64)
        idx = honest_indices if self.team_reward_mode == 'honest' else list(range(len(team_rewards)))
        r_team = float(np.mean(team_rewards[idx]))

        crashed_now = np.asarray(crashed, dtype=bool)[adv_indices]
        adv_crashed = float(np.mean(crashed_now))

        # Onset = a drone that was airborne last step and is crashed now.
        if self._was_crashed is None:
            onset = crashed_now.astype(float)
        else:
            onset = (crashed_now & ~self._was_crashed).astype(float)
        self._was_crashed = crashed_now.copy()
        onset_frac = float(np.mean(onset))
        self.crash_onsets += int(onset.sum())

        excess, mean_dev = self.stealth_penalty(adv_actions)

        team_term = -self.w_team * r_team
        charge = onset_frac if self.crash_penalty_mode == 'onset' else adv_crashed
        crash_term = -self.w_crash * charge
        alive_term = self.w_alive * (1.0 - adv_crashed)
        stealth_term = -self.w_stealth * excess
        total = team_term + crash_term + alive_term + stealth_term

        for k, v in (('team', team_term), ('crash', crash_term),
                     ('alive', alive_term), ('stealth', stealth_term), ('total', total)):
            self.episode_components[k] += v

        return total, dict(
            team=team_term, crash=crash_term, alive=alive_term, stealth=stealth_term,
            total=total, r_team_mean=r_team, adv_crashed=adv_crashed,
            adv_crash_onset=onset_frac, crash_onsets=float(self.crash_onsets),
            action_deviation=mean_dev, stealth_excess=excess,
        )

    def config(self):
        return dict(w_team=self.w_team, w_crash=self.w_crash, w_alive=self.w_alive,
                    w_stealth=self.w_stealth, epsilon=self.epsilon,
                    team_reward_mode=self.team_reward_mode,
                    crash_penalty_mode=self.crash_penalty_mode,
                    mu_honest=None if self.mu_honest is None else self.mu_honest.tolist())


def honest_action_stats(clean_npz_paths):
    """
    `mu_honest` and the action-distribution statistics from the clean baseline rollouts,
    as the spec requires ("come from your clean baseline rollouts").

    Returns dict with per-dimension mean/std, plus the mean L2 norm of an honest action --
    the last of which is the natural scale for choosing `epsilon`.
    """
    means, stds, norms, n = [], [], [], 0
    for path in clean_npz_paths:
        with np.load(path) as d:
            acts = np.asarray(d['actions_executed'], dtype=np.float64)
            lengths = np.asarray(d['episode_lengths'])
            t = np.arange(acts.shape[1])[None, :]
            mask = t < lengths[:, None]
            sel = acts[mask]                       # (steps*agents, act_dim)
            sel = sel.reshape(-1, acts.shape[-1])
            means.append(sel.mean(axis=0))
            stds.append(sel.std(axis=0))
            norms.append(np.linalg.norm(sel, axis=1).mean())
            n += len(sel)
    return dict(
        mu_honest=np.mean(means, axis=0).tolist(),
        sigma_honest=np.mean(stds, axis=0).tolist(),
        mean_action_norm=float(np.mean(norms)),
        n_samples=int(n),
        sources=[str(p) for p in clean_npz_paths],
    )
