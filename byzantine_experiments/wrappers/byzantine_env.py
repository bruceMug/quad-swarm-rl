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

from byzantine_experiments.agents.base import (
    BYZANTINE_REGISTRY,
    ByzantineAgent,
    EgoState,
    get_byzantine_class,
)


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

        # Size of the ego block that precedes the neighbour blocks in each observation.
        # Derived, not hardcoded: obs layouts differ between obs_repr settings
        # (xyz_vxyz_R_omega = 18, _floor = 19, _wall = 24).
        obs_dim = int(np.prod(env.observation_space.shape))
        n_neighbors = int(getattr(env.unwrapped, 'num_use_neighbor_obs', 0))
        self.n_neighbors = n_neighbors
        self.ego_dim = obs_dim - 6 * n_neighbors

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

        # Shared mutable channel for colluding agents (Tier 2D). Only Byzantine agents can
        # see it; it carries nothing the env computes. Must exist before _build_agents(),
        # which hands it to any agent declaring needs_coordination.
        self._coordination = {}
        self._jacobian_inv = {}
        self.ego_updates = 0
        self.broadcast_rewrites = 0

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
            if getattr(agent, 'needs_coordination', False):
                agent.coordination = self._coordination
            self.byz_agents[i] = agent

        self.any_needs_ego = any(getattr(a, 'needs_ego_state', False)
                                 for a in self.byz_agents.values())
        self.any_lies = any(getattr(a, 'lies_about_state', False)
                            for a in self.byz_agents.values())

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

    # ---------------------------------------------------------- tier 2 capabilities

    def _drone_states(self):
        """True (pos, vel) for every drone, read from the live env. See collect.py on why
        `.unwrapped` is resolved per call rather than cached."""
        envs = self.env.unwrapped.envs
        pos = np.array([e.dynamics.pos for e in envs], dtype=np.float64)
        vel = np.array([e.dynamics.vel for e in envs], dtype=np.float64)
        return pos, vel

    def _push_ego_state(self):
        """Give each goal-directed agent its OWN true state before it decides."""
        if not self.any_needs_ego:
            return
        envs = self.env.unwrapped.envs
        for i, agent in self.byz_agents.items():
            if not getattr(agent, 'needs_ego_state', False):
                continue
            dyn = envs[i].dynamics
            if i not in self._jacobian_inv:
                from byzantine_experiments.agents.control_utils import jacobians_for
                self._jacobian_inv[i] = jacobians_for(dyn)
            agent.ego = EgoState(
                pos=np.array(dyn.pos, dtype=np.float64),
                vel=np.array(dyn.vel, dtype=np.float64),
                rot=np.array(dyn.rot, dtype=np.float64),
                omega=np.array(dyn.omega, dtype=np.float64),
                jacobian=self._jacobian_inv[i][0],
                jacobian_inv=self._jacobian_inv[i][1],
                room_box=np.array(envs[i].room_box, dtype=np.float64),
            )
            self.ego_updates += 1

    def _apply_broadcast_lies(self, obs):
        """
        Rewrite what every OTHER drone sees of a lying agent (Tier 2C).

        The env publishes each drone's true position/velocity into its neighbours'
        observation vectors. Layout, verified empirically against
        `QuadrotorEnvMulti.extend_obs_space`:

            obs[i, 0:18]                              ego block (carries sensor noise)
            obs[i, 18 + 6*slot     : 18 + 6*slot + 3] pos[j] - pos[i]   for the `slot`-th
            obs[i, 18 + 6*slot + 3 : 18 + 6*slot + 6] vel[j] - vel[i]   closest neighbour j

        `slot` is a *proximity rank*, not an agent id: `neighborhood_indices()` re-sorts the
        neighbours of each drone every step (by distance plus closing speed), and when
        `num_use_neighbor_obs < num_agents - 1` it keeps only the closest few. So the index
        of the liar inside a given observer's vector changes step to step, and on some steps
        the liar is not present in a given observer's block at all. This is exactly the
        "get this wrong and the attack silently does nothing" trap: the ordering is taken
        from the env's own `neighborhood_indices()` rather than reimplemented.

        The lie corrupts the *values* an observer sees, not *which* neighbours it attends
        to -- selection still runs on true positions. See design/tier2_design.md.
        """
        if not self.any_lies:
            return obs

        liars = {i: a for i, a in self.byz_agents.items()
                 if getattr(a, 'lies_about_state', False)}
        if not liars:
            return obs

        unwrapped = self.env.unwrapped
        obs = np.array(obs, dtype=np.asarray(obs).dtype, copy=True)
        pos, vel = self._drone_states()
        indices = unwrapped.neighborhood_indices()
        lo = unwrapped.clip_neighbor_space_min_box
        hi = unwrapped.clip_neighbor_space_max_box

        reported = {}
        for i, agent in liars.items():
            true_state = np.concatenate([pos[i], vel[i]])
            out = np.asarray(agent.perturb_broadcast_state(true_state.copy(), self._t),
                             dtype=np.float64)
            if out.shape != (6,):
                raise ValueError(
                    f'{self.tier_name} agent {i} returned a broadcast state of shape '
                    f'{out.shape}, expected (6,) = [x, y, z, vx, vy, vz]')
            reported[i] = out

        for observer in range(self.num_agents):
            if observer in liars:
                continue      # a liar's own observation is untouched
            for slot, j in enumerate(indices[observer]):
                j = int(j)
                if j not in reported:
                    continue
                base = self.ego_dim + 6 * slot
                rel_pos = reported[j][:3] - pos[observer]
                rel_vel = reported[j][3:] - vel[observer]
                obs[observer, base:base + 3] = np.clip(rel_pos, lo[:3], hi[:3])
                obs[observer, base + 3:base + 6] = np.clip(rel_vel, lo[3:6], hi[3:6])
                self.broadcast_rewrites += 1

        return obs

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
        obs = self._apply_broadcast_lies(obs)
        out = (obs,) + tuple(out[1:]) if isinstance(out, tuple) else obs
        self._last_obs = np.asarray(obs)
        self._last_honest_actions = None
        self._last_executed_actions = None
        return out

    def step(self, actions):
        if self._pending_new_episode:
            # The env auto-reset itself inside the previous step (ExperienceReplayWrapper
            # does this for the multi-agent env), so start the new episode now.
            self._begin_episode()

        self._push_ego_state()
        honest, was_sequence = self._normalise_actions(actions)
        executed = self._apply_byzantine(honest)

        self._last_honest_actions = honest.copy()
        self._last_executed_actions = executed.copy()

        to_env = list(executed) if was_sequence else executed
        obs, rewards, terminated, truncated, infos = self.env.step(to_env)

        obs = self._apply_broadcast_lies(obs)
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

    def agent_diagnostics(self):
        """Per-agent internal counters (guard trigger rate, lies told, ...).

        Tier 2 needs these in the run sidecar: a survival guard that fires on most steps
        means the 'attack' is mostly honest flight, and the calibration has to say so.
        """
        out = {}
        for i, agent in self.byz_agents.items():
            if hasattr(agent, 'diagnostics'):
                out[str(i)] = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating))
                                   else v)
                               for k, v in agent.diagnostics().items()}
        return out

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
            ego_dim=self.ego_dim,
            n_neighbors=self.n_neighbors,
            uses_ego_state=self.any_needs_ego,
            lies_about_state=self.any_lies,
            broadcast_rewrites=self.broadcast_rewrites,
            registered_tiers=sorted(BYZANTINE_REGISTRY),
        )
