"""
Tier 3 — the frozen-team environment.

The structural requirement of Tier 3 is that the honest agents' weights must not update
during adversarial training. Anything else is self-play, and self-play invalidates the
baseline established in Phase 1.

Why this is a wrapper and not `--num_policies=2`
------------------------------------------------
Sample Factory v2.1.1 cannot express "agent 0 is trainable, agents 1-7 are frozen", for two
independent reasons (both verified against the installed source):

1. **Per-agent mapping is random, not addressable.** The only mechanism is
   `AgentPolicyMapping` (`algo/utils/agent_policy_mapping.py`), and with
   `pbt_mix_policies_in_one_env` it does
   `self.env_policies[env_idx] = [self._sample_policy() for _ in range(self.num_agents)]`
   -- a uniform random draw per agent, resampled every 10 episodes. There is no index->policy
   API; the class docstring carries the TODO "would be nice to have a mechanism to provide
   custom mappings through API". It is instantiated hardcoded at
   `algo/sampling/non_batched_sampling.py:369`, so there is no injection point either.
2. **There is no way to freeze a policy.** Sample Factory has no `--frozen_policies` flag or
   equivalent; with `num_policies=2` both policies get a Learner and both update. That is
   exactly the self-play the spec forbids.

So instead of making the honest team an *opponent policy*, this makes it part of the
*environment*. The adversary's learner never sees the honest agents at all: it is handed an
env with `num_agents = n_adv`, and the honest drones are simulated inside `step()` by a
frozen checkpoint running under `torch.no_grad()`.

That inverts the burden of proof in a useful way. "The honest weights did not update" is not
something to verify after the fact -- **there is no optimiser, no gradient, and no learner
attached to them anywhere in the process.** `test_tier3.py` still asserts bit-identical
parameter hashes across training, because a cheap assertion of an invariant you believe is
free, but the invariant holds by construction.
"""

import hashlib

import numpy as np
import gymnasium as gym
import torch

from byzantine_experiments.adversary.lies import (
    DEFAULT_MAX_LIE_POS,
    DEFAULT_MAX_LIE_VEL,
    EGO_AUG_DIM,
    DEFAULT_PLAUSIBLE_MAX_ACCEL,
    DEFAULT_PLAUSIBLE_MAX_OFFSET,
    DEFAULT_PLAUSIBLE_MAX_RATE,
    LIE_DIM,
    PlausibleLie,
    apply_lies_to_obs,
    augment_obs,
    decode_lie,
    lie_action_dim,
    lie_magnitude,
)
from byzantine_experiments.adversary.reward import AdversarialReward


_SWARM_REGISTERED = False


def ensure_swarm_components_registered():
    """
    Register the quad-swarm encoder factory before building any actor-critic.

    Without this, `create_actor_critic` builds a generic MLP encoder and loading the baseline
    checkpoint fails with a state-dict mismatch -- the baseline uses the attention
    neighbour-encoder (`actor_encoder.neighbor_encoder.*`, `self_encoder.*`, `feed_forward.*`).
    Done here rather than left to the caller so the frozen policy is constructible from a
    bare test as well as from the training entry point. Guarded because Sample Factory's
    registries reject duplicate names.
    """
    global _SWARM_REGISTERED
    if _SWARM_REGISTERED:
        return
    from swarm_rl.train import register_swarm_components
    register_swarm_components()
    _SWARM_REGISTERED = True


def policy_param_hash(actor_critic):
    """Stable hash of every parameter tensor, for the frozen-weights test."""
    h = hashlib.sha256()
    for name, p in sorted(actor_critic.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class FrozenTeamPolicy:
    """
    The honest baseline checkpoint, run in inference mode.

    Deliberately minimal: it loads once, moves to eval(), disables gradients on every
    parameter, and exposes a single `act()`. Nothing here can train.
    """

    def __init__(self, cfg, obs_space, action_space, checkpoint_dir, device='cpu',
                 checkpoint_file=None):
        from sample_factory.algo.learning.learner import Learner
        from sample_factory.model.actor_critic import create_actor_critic

        ensure_swarm_components_registered()
        self.device = torch.device(device)
        # Sample Factory's encoder indexes the observation space by key, so it must be a
        # Dict. In the eval path that happens implicitly because BatchedVecEnv inserts
        # BatchedDictObservationsWrapper; here the env is raw, so wrap it explicitly.
        if not isinstance(obs_space, gym.spaces.Dict):
            obs_space = gym.spaces.Dict(dict(obs=obs_space))
        self.obs_space = obs_space
        self.actor_critic = create_actor_critic(cfg, obs_space, action_space)
        self.actor_critic.eval()
        self.actor_critic.model_to_device(self.device)

        if checkpoint_file is not None:
            # An explicit file, used to evaluate a specific training milestone.
            checkpoints = [checkpoint_file]
        else:
            name_prefix = dict(latest='checkpoint', best='best')[cfg.load_checkpoint_kind]
            checkpoints = Learner.get_checkpoints(checkpoint_dir, f'{name_prefix}_*')
            if not checkpoints:
                raise FileNotFoundError(f'no {name_prefix}_* checkpoint under {checkpoint_dir}')
        state = Learner.load_checkpoint(checkpoints, self.device)
        self.actor_critic.load_state_dict(state['model'])
        self.checkpoint_file = str(checkpoints[-1])

        # Belt and braces on top of no-learner-attached: the tensors themselves refuse grad.
        for p in self.actor_critic.parameters():
            p.requires_grad_(False)

        self.param_hash = policy_param_hash(self.actor_critic)
        self._rnn_size = None

    def rnn_states(self, n):
        from sample_factory.model.model_utils import get_rnn_size
        if self._rnn_size is None:
            self._rnn_size = get_rnn_size(self._cfg) if hasattr(self, '_cfg') else None
        return torch.zeros([n, self._rnn_size], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs, rnn_states, deterministic=True):
        """obs: (n, obs_dim) float32 -> (actions (n, act_dim) float32, new_rnn_states)."""
        from sample_factory.algo.utils.action_distributions import argmax_actions
        from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

        obs_dict = {'obs': torch.as_tensor(obs, dtype=torch.float32, device=self.device)}
        normalized = prepare_and_normalize_obs(self.actor_critic, obs_dict)
        outputs = self.actor_critic(normalized, rnn_states)
        actions = (argmax_actions(self.actor_critic.action_distribution())
                   if deterministic else outputs['actions'])
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        return actions.cpu().numpy().astype(np.float32), outputs['new_rnn_states']


class FrozenTeamEnv(gym.Env):
    """
    Presents the 8-drone swarm to Sample Factory as an `n_adv`-agent environment.

    The learner sees only the adversary's observations, actions and reward. The other
    `8 - n_adv` drones are driven by `FrozenTeamPolicy` inside `step()`, which makes them
    part of the environment's transition function rather than an opponent to be trained.

    Args:
        env: the wrapped multi-drone env (gymnasium API, per-agent rows) -- the same stack
            Tier 0/1/2 use, so trajectories stay format-compatible.
        frozen_policy: a FrozenTeamPolicy.
        adv_indices: which drone indices the adversary controls.
        reward_fn: an AdversarialReward.
        deterministic_honest: take the honest policy's argmax (matches how every previous
            tier's rollouts were collected) rather than sampling.
    """

    def __init__(self, env, frozen_policy, adv_indices=(0,), reward_fn=None,
                 deterministic_honest=True, rnn_size=1,
                 enable_lies=False, lie_only=False, ego_state_obs=False,
                 max_lie_pos=DEFAULT_MAX_LIE_POS, max_lie_vel=DEFAULT_MAX_LIE_VEL,
                 lie_mode='free', plausible_max_offset=DEFAULT_PLAUSIBLE_MAX_OFFSET,
                 plausible_max_rate=DEFAULT_PLAUSIBLE_MAX_RATE,
                 plausible_max_accel=DEFAULT_PLAUSIBLE_MAX_ACCEL):
        self.env = env
        self.frozen = frozen_policy
        self.adv_indices = sorted(int(i) for i in adv_indices)
        self.reward_fn = reward_fn or AdversarialReward()
        self.deterministic_honest = deterministic_honest
        self.rnn_size = rnn_size

        self.total_agents = int(env.unwrapped.num_agents)
        for i in self.adv_indices:
            if not 0 <= i < self.total_agents:
                raise ValueError(f'adversary index {i} out of range for {self.total_agents} drones')
        self.honest_indices = [i for i in range(self.total_agents) if i not in self.adv_indices]
        if not self.honest_indices:
            raise ValueError('at least one drone must remain honest')

        # ---- round 3: the observation channel -------------------------------------------
        # With lies enabled the action grows from 4 motors to 4 + 6, the extra six being the
        # state this drone *claims*. See adversary/lies.py for why it is a bounded delta.
        self.enable_lies = bool(enable_lies)
        self.lie_only = bool(lie_only)
        if self.lie_only and not self.enable_lies:
            raise ValueError('lie_only=True requires enable_lies=True')
        self.max_lie_pos, self.max_lie_vel = float(max_lie_pos), float(max_lie_vel)
        # D3: 'free' is rounds 3-5's unconstrained claim; 'plausible' is a bounded, smooth drift
        # that passes naive bounds/speed/consistency/acceleration checks (lies.py::PlausibleLie).
        self.lie_mode = lie_mode
        self.lie_dim = lie_action_dim(lie_mode) if self.enable_lies else 0
        self._plausible = {}
        if self.enable_lies and lie_mode == 'plausible':
            envs = env.unwrapped.envs
            dt = 1.0 / float(envs[0].control_freq)
            for i in self.adv_indices:
                self._plausible[i] = PlausibleLie(
                    envs[i].room_box, dt, max_offset=plausible_max_offset,
                    max_rate=plausible_max_rate, max_accel=plausible_max_accel)
        # Round 5: append the adversary's own absolute pos/vel to its observation. Without it
        # the policy cannot express a state-dependent lie like `mirror` -- absolute position
        # is nowhere in the 54-D vector (linear probe R^2 0.27-0.39). See lies.py::augment_obs.
        self.ego_state_obs = bool(ego_state_obs)
        if self.ego_state_obs and not self.enable_lies:
            raise ValueError('ego_state_obs=True requires enable_lies=True')
        self.lies_told = 0
        self.broadcast_rewrites = 0
        self._ep_lie_magnitude = []

        # What Sample Factory sees.
        self.num_agents = len(self.adv_indices)
        self.is_multiagent = True
        if self.ego_state_obs:
            base = env.observation_space
            low = np.concatenate([np.asarray(base.low, dtype=np.float32).reshape(-1),
                                  np.full(EGO_AUG_DIM, -np.inf, dtype=np.float32)])
            self.observation_space = gym.spaces.Box(
                low, -low, dtype=np.float32)
        else:
            self.observation_space = env.observation_space

        self.motor_dim = int(np.prod(env.action_space.shape))
        self._motor_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        self._motor_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        if self.enable_lies:
            # The lie half is [-1, 1]: `decode_lie` scales it to metres and m/s itself, so the
            # policy never has to represent world units directly.
            self.action_space = gym.spaces.Box(
                np.concatenate([self._motor_low, np.full(self.lie_dim, -1.0, dtype=np.float32)]),
                np.concatenate([self._motor_high, np.full(self.lie_dim, 1.0, dtype=np.float32)]),
                dtype=np.float32)
        else:
            self.action_space = env.action_space

        self._action_low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        self._action_high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)

        self._last_obs = None
        self._honest_rnn = None
        self._t = 0
        # Crash state from the PREVIOUS step. ExperienceReplayWrapper auto-resets the env
        # inside step() when an episode ends, so the `crashed` flags read straight after
        # `self.env.step()` on a terminal step belong to the NEW episode. Reporting survival
        # from those made `adv/survived` read 1.000 for every episode of both training runs,
        # while evaluation measured 0.00. Same class of bug as the stale `.unwrapped` in
        # collect.py: never read env state across an auto-reset boundary.
        self._prev_crashed = None
        self._ever_crashed = False

    # ------------------------------------------------------------------ internals

    # Width of the ego block before the neighbour slots begin. Same constant the Tier 0
    # wrapper uses; the env's own observation layout is 18 ego + 6 per neighbour.
    ego_dim = 18

    def _adv_obs(self):
        """Observation handed to Sample Factory, augmented when round 5 is enabled."""
        rows = self._last_obs[self.adv_indices]
        if not self.ego_state_obs:
            return rows
        pos, vel = self._drone_states()
        return np.stack([
            augment_obs(rows[k], pos[i], vel[i],
                        max_pos=self.max_lie_pos, max_vel=self.max_lie_vel)
            for k, i in enumerate(self.adv_indices)]).astype(np.float32)

    def _fresh_rnn(self):
        return torch.zeros([self._rnn_rows(), self.rnn_size],
                           dtype=torch.float32, device=self.frozen.device)

    def _honest_actions(self):
        """
        Actions for every drone the frozen policy drives.

        In `lie_only` mode the adversary's own motors are driven by the frozen policy too --
        it flies exactly like an honest drone and attacks purely through what it broadcasts,
        which is the direct analogue of Tier 2C's `liar` and isolates the channel from the
        motor authority rounds 1-2 already showed to be useless.
        """
        rows = (self.honest_indices + self.adv_indices) if self.lie_only else self.honest_indices
        obs = np.asarray(self._last_obs, dtype=np.float32)[rows]
        actions, self._honest_rnn = self.frozen.act(
            obs, self._honest_rnn, deterministic=self.deterministic_honest)
        return dict(zip(rows, actions))

    def _rnn_rows(self):
        return len(self.honest_indices) + (len(self.adv_indices) if self.lie_only else 0)

    def _broadcast_lies(self, lie_actions):
        """
        Corrupt the neighbour blocks of `self._last_obs` in place-ish, post-step.

        Timing matters and is chosen to match evaluation exactly: `ByzantineEnvWrapper.step`
        applies lies to the observation it *returns*, so honest drones consume the lie on the
        next control cycle. Doing it here, right after `self.env.step()`, reproduces that --
        the corrupted `_last_obs` is what `_honest_actions()` reads next step.
        """
        unwrapped = self.env.unwrapped
        pos, vel = self._drone_states()
        reported = {}
        for k, i in enumerate(self.adv_indices):
            true_state = np.concatenate([pos[i], vel[i]])
            if self.lie_mode == 'plausible':
                rep = self._plausible[i].step(lie_actions[k], true_state)
            else:
                rep = decode_lie(lie_actions[k], true_state,
                                 max_pos=self.max_lie_pos, max_vel=self.max_lie_vel)
            reported[i] = rep
            self._ep_lie_magnitude.append(lie_magnitude(rep, true_state))
        self.lies_told += len(reported)

        self._last_obs, rewrites = apply_lies_to_obs(
            self._last_obs, reported, unwrapped.neighborhood_indices(), pos, vel,
            ego_dim=self.ego_dim,
            clip_lo=unwrapped.clip_neighbor_space_min_box,
            clip_hi=unwrapped.clip_neighbor_space_max_box,
            liar_indices=self.adv_indices)
        self._last_obs = np.asarray(self._last_obs, dtype=np.float32)
        self.broadcast_rewrites += rewrites

    def _drone_states(self):
        envs = self.env.unwrapped.envs
        return (np.array([e.dynamics.pos for e in envs], dtype=np.float64),
                np.array([e.dynamics.vel for e in envs], dtype=np.float64))

    def _crashed(self):
        """Per-drone crash flags, straight from the simulator's own collision state."""
        out = np.zeros(self.total_agents, dtype=bool)
        for i, e in enumerate(self.env.unwrapped.envs):
            d = e.dynamics
            out[i] = bool(getattr(d, 'crashed_floor', False) or getattr(d, 'crashed_wall', False)
                          or getattr(d, 'crashed_ceiling', False) or getattr(d, 'on_floor', False))
        return out

    # ------------------------------------------------------------------ gym API

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._honest_rnn = self._fresh_rnn()
        self._t = 0
        self._prev_crashed = None
        self._ever_crashed = False
        self._ep_lie_magnitude = []
        for pl in self._plausible.values():
            pl.reset()
        self.reward_fn.reset()
        return self._adv_obs(), {}

    def step(self, adv_actions):
        adv_actions = np.asarray(adv_actions, dtype=np.float32).reshape(
            len(self.adv_indices), -1)

        # Clip to the legal action range BEFORE executing and before scoring.
        #
        # Sample Factory's continuous policy emits unbounded Gaussian samples; the env's
        # RawControl clips internally, so an action of norm 100 flies identically to one of
        # norm 2. Scoring the *raw* sample therefore punishes something no detector could
        # observe, and the stealth term ran away to -542,034/episode (22,700x the objective)
        # while the executed behaviour was unchanged. Every Tier 1/2 agent clips its own
        # output for the same reason; the adversary must too.
        adv_actions = np.clip(adv_actions, self._action_low, self._action_high)

        # Split the motor half from the broadcast half before anything else touches them.
        # Only the motors are executed, and only the motors are scored for stealth -- the
        # stealth term compares against `mu_honest`, which is a distribution over the honest
        # policy's 4 thrusts and says nothing about what a drone claims.
        motor_actions = adv_actions[:, :self.motor_dim]
        lie_actions = adv_actions[:, self.motor_dim:] if self.enable_lies else None

        driven = self._honest_actions()
        full = np.empty((self.total_agents, self.motor_dim), dtype=np.float32)
        for i in self.honest_indices:
            full[i] = driven[i]
        for k, i in enumerate(self.adv_indices):
            full[i] = driven[i] if self.lie_only else motor_actions[k]
        if self.lie_only:
            # Score the stealth term against what was actually flown, which is the honest
            # policy's own output -- deviation is zero by construction, as it is for Tier 2C.
            motor_actions = np.stack([full[i] for i in self.adv_indices]).astype(np.float32)

        obs, rewards, terminated, truncated, infos = self.env.step(full)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        if self.enable_lies:
            self._broadcast_lies(lie_actions)
        self._t += 1

        rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
        crashed = self._crashed()
        if crashed[self.adv_indices].any():
            self._ever_crashed = True
        pos, _ = self._drone_states()

        adv_reward, components = self.reward_fn(
            team_rewards=rewards, adv_indices=self.adv_indices,
            honest_indices=self.honest_indices, adv_actions=motor_actions,
            crashed=crashed, positions=pos, t=self._t,
        )

        adv_infos = []
        for k, i in enumerate(self.adv_indices):
            info = dict(infos[i]) if isinstance(infos, (list, tuple)) else {}
            info['adv_reward_components'] = {n: float(v) for n, v in components.items()}
            adv_infos.append(info)

        def _slice(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                arr = np.asarray(x).reshape(-1)
                return [bool(arr[i]) for i in self.adv_indices]
            return [bool(x)] * len(self.adv_indices)

        term, trunc = _slice(terminated), _slice(truncated)
        if any(term) or any(trunc):
            # Sample Factory harvests `episode_extra_stats` at episode end and forwards it to
            # wandb/tensorboard automatically. This is what makes each reward component
            # separately visible during training, which the spec asks for -- per-step info
            # dicts are not logged anywhere.
            ep = self.reward_fn.episode_components
            extra = {
                'adv/r_team': ep['team'],
                'adv/r_crash': ep['crash'],
                'adv/r_alive': ep['alive'],
                'adv/r_stealth': ep['stealth'],
                'adv/r_total': ep['total'],
                'adv/crash_onsets': float(self.reward_fn.crash_onsets),
                'adv/team_reward_mean': float(np.mean(rewards[self.honest_indices])),
                'adv/action_deviation': float(components['action_deviation']),
                # Use the PRE-reset value; `crashed` above may describe the new episode.
                'adv/survived': float(
                    not bool(np.asarray(self._prev_crashed)[self.adv_indices].any())
                    if self._prev_crashed is not None
                    else not bool(crashed[self.adv_indices].any())),
                'adv/ever_crashed': float(self._ever_crashed),
                'adv/episode_len': float(self._t),
            }
            if self.enable_lies:
                mags = np.asarray(self._ep_lie_magnitude, dtype=np.float64)
                extra.update({
                    # Metres between claim and truth. Watch this: a policy that drives it to
                    # ~0 has abandoned the channel, which is the round-3 analogue of round 2's
                    # flat `adv/r_team`.
                    'adv/lie_magnitude_m': float(mags.mean()) if mags.size else 0.0,
                    'adv/lie_magnitude_max_m': float(mags.max()) if mags.size else 0.0,
                    'adv/broadcast_rewrites': float(self.broadcast_rewrites),
                })
                if self._plausible:
                    # How often a room limit shaped the claim this episode. The claim's
                    # acceleration budget holds except on these steps, so watch it.
                    wl = sum(pl.wall_limited_steps for pl in self._plausible.values())
                    st = max(sum(pl.steps for pl in self._plausible.values()), 1)
                    extra['adv/lie_wall_limited_frac'] = float(wl) / st
            for info in adv_infos:
                info.setdefault('episode_extra_stats', {}).update(extra)

            self._honest_rnn = self._fresh_rnn()
            self.reward_fn.reset()
            self._t = 0
            self._prev_crashed = None
            self._ever_crashed = False
            self._ep_lie_magnitude = []
            for pl in self._plausible.values():
                pl.reset()
        else:
            self._prev_crashed = crashed.copy()

        return (self._adv_obs(),
                np.asarray([adv_reward] * len(self.adv_indices), dtype=np.float32),
                term, trunc, adv_infos)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
