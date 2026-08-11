"""
ByzantineEnvWrapper -- injects Byzantine agents into an already-trained multi-drone policy
at rollout time, without retraining and without touching gym_art/.

Placement in the stack (see swarm_rl/env_wrappers/quad_utils.py::make_quadrotor_env_multi):

    QuadrotorEnvMulti
      -> ExperienceReplayWrapper        (only when replay_buffer_sample_prob > 0)
      -> QuadsRewardShapingWrapper
      -> QuadEnvCompatibility           (old 4-tuple gym API -> gymnasium 5-tuple)
      -> ByzantineEnvWrapper            <-- here
      -> BatchedVecEnv                  (sample_factory, adds by collect.py)

Sitting above QuadEnvCompatibility means we see exactly the action array that
sample_factory hands to the env, and we speak the gymnasium 5-tuple API.
"""

import inspect

import numpy as np
import gymnasium as gym

from byzantine_experiments.agents.base import BYZANTINE_REGISTRY, ByzantineAgent, get_byzantine_class


class ByzantineEnvWrapper(gym.Wrapper):
    """
    Replaces the actions of a fixed subset of drones with Byzantine ones.

    Args:
        env: the wrapped multi-agent quadrotor env (gymnasium API, per-agent action rows).
        tier: registry key (str) of the Byzantine agent, or a ByzantineAgent subclass
            directly (used by tests so they need not pollute the global registry).
        compromised_indices: agent indices to compromise in `fixed` mode, and -- in
            `random` mode -- only its *length* is used (indices are resampled per episode).
        tier_config: dict of extra kwargs forwarded to the agent constructor.
        seed: base seed for the Byzantine agents and for random index assignment.
        assignment_mode: 'fixed' (default) or 'random'.
        log_perturbations: keep the per-step perturbation log (see get_perturbation_log()).
    """

    def __init__(self, env, tier, compromised_indices=(0,), tier_config=None, seed=0,
                 assignment_mode='fixed', log_perturbations=True):
        super().__init__(env)

        if assignment_mode not in ('fixed', 'random'):
            raise ValueError(f"assignment_mode must be 'fixed' or 'random', got {assignment_mode!r}")

        self.num_agents = int(getattr(env.unwrapped, 'num_agents'))
        # sample_factory sniffs these off the outermost env to decide the data layout.
        self.is_multiagent = bool(getattr(env.unwrapped, 'is_multiagent', True))

        # Action range comes from the env, never hardcoded. RawControl(zero_action_middle=True)
        # gives Box(-1, 1, (4,), float32), but read it rather than assume it.
        act_space = env.action_space
        if not isinstance(act_space, gym.spaces.Box):
            raise TypeError(f'ByzantineEnvWrapper expects a Box action space, got {act_space}')
        self.action_dim = int(np.prod(act_space.shape))
        self.action_low = np.asarray(act_space.low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(act_space.high, dtype=np.float64).reshape(-1)
        self.action_dtype = act_space.dtype

        self.tier = tier
        self.tier_name = tier if isinstance(tier, str) else tier.__name__
        self.tier_config = dict(tier_config or {})
        self.seed_value = int(seed)
        self.assignment_mode = assignment_mode
        self.log_perturbations = log_perturbations

        self._agent_cls = tier if isinstance(tier, type) else get_byzantine_class(tier)
        if not issubclass(self._agent_cls, ByzantineAgent):
            raise TypeError(f'{self._agent_cls.__name__} is not a ByzantineAgent subclass')

        requested = [int(i) for i in compromised_indices]
        for i in requested:
            if not 0 <= i < self.num_agents:
                raise ValueError(f'compromised index {i} out of range for {self.num_agents} agents')
        if len(set(requested)) != len(requested):
            raise ValueError(f'duplicate compromised indices: {requested}')
        self._requested_indices = requested
        self.n_compromised = len(requested)

        # RNG for episode-to-episode index resampling only; agents own their own streams.
        self._assign_rng = np.random.default_rng(self.seed_value + 999983)

        self.compromised_indices = list(self._requested_indices)
        self.byz_agents = {}
        self._build_agents()

        self._t = 0
        self._episode_idx = 0
        self._started = False
        self._last_obs = None
        self._perturbation_log = []
        self._last_honest_actions = None
        self._last_executed_actions = None
        # Set when the underlying env signalled `done`; the *next* step starts a new episode.
        # Deferred (rather than done inside step) so the finished episode's log survives.
        self._pending_new_episode = False

    # ---------------------------------------------------------------- construction

    def _build_agents(self):
        """(Re)instantiate one Byzantine agent per compromised index."""
        # A tier may declare a strict __init__ without **kwargs; only pass the action range
        # to constructors that can actually take it, and set it directly afterwards regardless.
        params = inspect.signature(self._agent_cls.__init__).parameters
        takes_range = ('action_low' in params
                       or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()))

        self.byz_agents = {}
        for i in self.compromised_indices:
            kwargs = dict(agent_idx=i, action_dim=self.action_dim, seed=self.seed_value)
            if takes_range:
                kwargs.update(action_low=self.action_low, action_high=self.action_high)
            agent = self._agent_cls(**kwargs, **self.tier_config)

            # Authoritative: whatever the constructor did, the legal range is the env's.
            agent.action_low = self.action_low
            agent.action_high = self.action_high
            self.byz_agents[i] = agent

    def _begin_episode(self):
        """Episode boundary bookkeeping: resample indices (random mode), reset agents."""
        if self._started:
            self._episode_idx += 1
        self._started = True

        if self.assignment_mode == 'random':
            new_indices = sorted(
                int(i) for i in self._assign_rng.choice(
                    self.num_agents, size=self.n_compromised, replace=False)
            )
            if new_indices != self.compromised_indices:
                self.compromised_indices = new_indices
                self._build_agents()

        for agent in self.byz_agents.values():
            agent.reset()

        self._t = 0
        self._perturbation_log = []
        self._pending_new_episode = False

    # ---------------------------------------------------------------- action plumbing

    def _normalise_actions(self, actions):
        """
        Accept whatever sample_factory hands us -- a stacked (n_agents, act_dim) array, a
        list/tuple of per-agent arrays, or a torch tensor -- and return one float array of
        shape (n_agents, act_dim) plus a flag saying whether the input was a sequence.
        """
        was_sequence = isinstance(actions, (list, tuple))

        if hasattr(actions, 'detach'):  # torch.Tensor
            actions = actions.detach().cpu().numpy()

        arr = np.asarray(actions)
        if arr.ndim == 1 and self.action_dim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape != (self.num_agents, self.action_dim):
            raise ValueError(
                f'expected actions of shape ({self.num_agents}, {self.action_dim}), '
                f'got {arr.shape}')
        return arr, was_sequence

    def _apply_byzantine(self, honest):
        """Return a copy of `honest` with compromised rows replaced by perturbed actions."""
        executed = honest.copy()

        for i in self.compromised_indices:
            honest_row = honest[i]
            obs_row = None
            if self._last_obs is not None:
                obs_row = self._last_obs[i]

            perturbed = self.byz_agents[i].perturb_action(honest_row.copy(), obs_row, self._t)
            perturbed = np.asarray(perturbed)

            if perturbed.shape != honest_row.shape:
                raise ValueError(
                    f'{self.tier_name} agent {i} returned shape {perturbed.shape}, '
                    f'expected {honest_row.shape}')
            if np.any(perturbed < self.action_low - 1e-6) or np.any(perturbed > self.action_high + 1e-6):
                raise ValueError(
                    f'{self.tier_name} agent {i} returned an out-of-range action at t={self._t}: '
                    f'{perturbed} not within [{self.action_low}, {self.action_high}]. '
                    f'Every tier must clip its own output.')

            perturbed = perturbed.astype(honest.dtype, copy=False)
            executed[i] = perturbed

            if self.log_perturbations:
                self._perturbation_log.append(dict(
                    episode=self._episode_idx,
                    t=self._t,
                    agent_idx=i,
                    honest_action=honest_row.copy(),
                    perturbed_action=perturbed.copy(),
                    l2_deviation=float(np.linalg.norm(perturbed - honest_row)),
                ))

        return executed

    # ---------------------------------------------------------------- gym API

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._begin_episode()

        obs = out[0] if isinstance(out, tuple) else out
        self._last_obs = np.asarray(obs)
        self._last_honest_actions = None
        self._last_executed_actions = None
        return out

    def step(self, actions):
        if self._pending_new_episode:
            # The env auto-reset itself inside the previous step (ExperienceReplayWrapper
            # does this for the multi-agent env), so start the new episode now.
            self._begin_episode()

        honest, was_sequence = self._normalise_actions(actions)
        executed = self._apply_byzantine(honest)

        self._last_honest_actions = honest.copy()
        self._last_executed_actions = executed.copy()

        to_env = list(executed) if was_sequence else executed
        obs, rewards, terminated, truncated, infos = self.env.step(to_env)

        self._last_obs = np.asarray(obs)
        self._t += 1

        if self._episode_finished(terminated, truncated):
            self._pending_new_episode = True

        return obs, rewards, terminated, truncated, infos

    @staticmethod
    def _episode_finished(terminated, truncated):
        def _any(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return bool(np.any(np.asarray(x)))
            return bool(x)

        return _any(terminated) or _any(truncated)

    # ---------------------------------------------------------------- introspection

    def get_perturbation_log(self):
        """
        Per-step record for the *current* episode: one entry per compromised agent per step,
        i.e. exactly `T * len(compromised_indices)` entries after T steps.
        """
        return list(self._perturbation_log)

    @property
    def byzantine_mask(self):
        """Ground-truth labels, shape (n_agents,) bool. Consumed by the Phase 4 AUC eval."""
        mask = np.zeros(self.num_agents, dtype=bool)
        mask[self.compromised_indices] = True
        return mask

    @property
    def last_honest_actions(self):
        """Policy output of the most recent step, shape (n_agents, act_dim)."""
        return self._last_honest_actions

    @property
    def last_executed_actions(self):
        """What the env actually received on the most recent step."""
        return self._last_executed_actions

    @property
    def t(self):
        return self._t

    def config_summary(self):
        """Serialisable description of this wrapper's configuration, for the run sidecar."""
        return dict(
            tier=self.tier_name,
            tier_config=self.tier_config,
            compromised_indices=list(self.compromised_indices),
            n_compromised=self.n_compromised,
            num_agents=self.num_agents,
            f_fraction=self.n_compromised / self.num_agents,
            assignment_mode=self.assignment_mode,
            seed=self.seed_value,
            action_dim=self.action_dim,
            action_low=self.action_low.tolist(),
            action_high=self.action_high.tolist(),
            action_dtype=str(self.action_dtype),
            registered_tiers=sorted(BYZANTINE_REGISTRY),
        )
