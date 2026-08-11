"""
Tier 0 validation suite.

Most of these build a small QuadrotorEnvMulti directly (no sample_factory, no checkpoint) so
they run in seconds. Reproducibility is arranged by `seed_all()` below, which pins every RNG
the sim reaches for -- that is what makes "byte-identical rollout" a meaningful claim here.

The last test is the validation gate: it drives the real trained checkpoint through the full
sample_factory stack and requires bit-identical trajectories with and without the wrapper.
It self-skips when `train_dir/exp1_baseline_seed1` is absent.

Run:  pytest byzantine_experiments/tests/test_tier0.py -v
"""

import os
import random

import numpy as np
import pytest

from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING

from byzantine_experiments.agents.base import (
    BYZANTINE_REGISTRY,
    ByzantineAgent,
    clip_to_action_space,
)
from byzantine_experiments.wrappers.byzantine_env import ByzantineEnvWrapper

NUM_AGENTS = 4
EP_TIME = 0.4  # seconds -> ~40 steps at 100 Hz control freq
ENV_SEED = 12345


# --------------------------------------------------------------------------- helpers

def make_env(seed=ENV_SEED):
    """A minimal replica of make_quadrotor_env_multi: no replay buffer, no reward shaping."""
    np.random.seed(seed)
    env = QuadrotorEnvMulti(
        num_agents=NUM_AGENTS,
        ep_time=EP_TIME,
        rew_coeff=dict(DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']),
        obs_repr='xyz_vxyz_R_omega',
        neighbor_visible_num=-1,
        neighbor_obs_type='pos_vel',
        collision_hitbox_radius=2.0,
        collision_falloff_radius=4.0,
        use_obstacles=False,
        obst_density=0.2,
        obst_size=1.0,
        obst_spawn_area=[6.0, 6.0],
        use_downwash=False,
        use_numba=False,
        quads_mode='static_same_goal',
        room_dims=[10.0, 10.0, 10.0],
        use_replay_buffer=False,
        quads_view_mode=['topdown'],
        quads_render=False,
        dynamics_params='Crazyflie',
        dyn_sampler_1=None,
        raw_control=True,
        raw_control_zero_middle=True,
        dynamics_randomize_every=None,
        dynamics_change=dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0)),
        sense_noise='default',
        init_random_state=False,
        render_mode=None,
    )
    return QuadEnvCompatibility(env)


def seed_all(env, seed=ENV_SEED):
    """
    Pin every RNG the sim draws from.

    Most of the sim (OU thrust noise, sensor noise, scenario goals) uses the *global* numpy
    stream, but there are two more:
      - each QuadrotorSingle owns a `np_random` that upstream seeds from OS entropy at
        construction (quadrotor_single.py:234 calls `_seed()` with no argument), and that
        one picks the spawn positions;
      - the replay buffer samples with Python's `random` (quad_experience_replay.py:43).

    A third, numba's JIT-internal RNG state, is *not* reachable from here at all -- see
    `test_noop_matches_unwrapped_on_the_real_policy_stack`.
    """
    random.seed(seed)
    np.random.seed(seed)
    for i, e in enumerate(env.unwrapped.envs):
        e._seed(seed + i)


def action_sequence(n_steps, seed=7):
    """A fixed pseudo-random action stream, standing in for a frozen policy."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n_steps, NUM_AGENTS, 4)).astype(np.float32)


def rollout(env, actions, seed=ENV_SEED):
    """Drive `env` with a fixed action stream; return everything worth comparing."""
    seed_all(env, seed)
    obs, _ = env.reset()
    obs_trace, rew_trace = [np.asarray(obs).copy()], []
    for a in actions:
        obs, rew, terminated, truncated, _ = env.step(a.copy())
        obs_trace.append(np.asarray(obs).copy())
        rew_trace.append(np.asarray(rew).copy())
        if np.any(terminated) or np.any(truncated):
            break
    return np.array(obs_trace), np.array(rew_trace)


class _SignFlip(ByzantineAgent):
    """Test-local attacker: hard sign flip. Deliberately not registered."""

    def perturb_action(self, honest_action, obs, t):
        return self.clip(-honest_action)


class _Unclipped(ByzantineAgent):
    """Test-local attacker that would escape the action range if it forgot to clip."""

    def perturb_action(self, honest_action, obs, t):
        return self.clip(honest_action + 10.0)


class _NoisyRng(ByzantineAgent):
    """Test-local attacker whose output depends on its RNG stream."""

    def perturb_action(self, honest_action, obs, t):
        return self.clip(honest_action + self.rng.normal(0.0, 0.5, size=honest_action.shape))


@pytest.fixture(scope='module')
def actions():
    return action_sequence(30)


# --------------------------------------------------------------------------- tests

def test_isolation_uncompromised_agents_untouched(actions):
    """
    TEST 4 (written first -- catches the most bugs).

    Rows of the action array belonging to honest drones must arrive at the env bit-identical
    to the unwrapped case, even while a neighbour is being attacked hard.
    """
    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier=_SignFlip, compromised_indices=[1], seed=3)

    seed_all(wrapped)
    wrapped.reset()

    honest_rows = [i for i in range(NUM_AGENTS) if i != 1]
    for a in actions:
        wrapped.step(a.copy())
        executed = wrapped.last_executed_actions
        for i in honest_rows:
            assert np.array_equal(executed[i], a[i]), f'agent {i} was modified'
        # and the attack really did fire on the compromised row
        assert np.allclose(executed[1], -a[1])
    wrapped.close()


def test_identity_noop_reproduces_unwrapped_rollout(actions):
    """TEST 1 -- a no-op Byzantine agent must not perturb the trajectory in any way."""
    plain = make_env()
    obs_a, rew_a = rollout(plain, actions)
    plain.close()

    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier='tier0_noop',
                                  compromised_indices=[0, 2], seed=99)
    obs_b, rew_b = rollout(wrapped, actions)
    wrapped.close()

    assert obs_a.shape == obs_b.shape
    assert np.array_equal(obs_a, obs_b), 'no-op wrapper changed the observation trace'
    assert np.array_equal(rew_a, rew_b), 'no-op wrapper changed the reward trace'


def test_shape_and_dtype_preserved(actions):
    """TEST 2 -- perturbed actions match honest actions in shape and dtype."""
    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier=_NoisyRng, compromised_indices=[0, 3], seed=5)
    seed_all(wrapped)
    wrapped.reset()

    for a in actions:
        wrapped.step(a.copy())
        executed = wrapped.last_executed_actions
        assert executed.shape == a.shape
        assert executed.dtype == a.dtype, f'{executed.dtype} != {a.dtype}'
    wrapped.close()


def test_range_compliance(actions):
    """TEST 3 -- no executed action ever falls outside env.action_space."""
    env = make_env()
    low, high = env.action_space.low, env.action_space.high
    wrapped = ByzantineEnvWrapper(env, tier=_Unclipped, compromised_indices=[2], seed=5)
    seed_all(wrapped)
    wrapped.reset()

    for a in actions:
        wrapped.step(a.copy())
        executed = wrapped.last_executed_actions
        assert np.all(executed >= low) and np.all(executed <= high)
        for row in executed:
            assert env.action_space.contains(row.astype(env.action_space.dtype))
    wrapped.close()


def test_registry_entries_are_valid():
    """TEST 5 -- every registered name instantiates and satisfies the ABC contract."""
    assert 'tier0_noop' in BYZANTINE_REGISTRY, 'the Tier 0 control agent must be registered'

    for name, cls in BYZANTINE_REGISTRY.items():
        assert isinstance(name, str) and name
        assert issubclass(cls, ByzantineAgent), f'{name} -> {cls} is not a ByzantineAgent'

        agent = cls(agent_idx=0, action_dim=4, seed=0)
        assert agent.name == cls.__name__
        assert callable(agent.reset) and agent.reset() is None

        honest = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        original = honest.copy()
        out = np.asarray(agent.perturb_action(honest, np.zeros(18, dtype=np.float32), 0))

        assert out.shape == honest.shape, f'{name} changed the action shape'
        assert np.array_equal(honest, original), f'{name} mutated its input in place'
        assert np.all(out >= -1.0) and np.all(out <= 1.0), f'{name} escaped the action range'
        # the broadcast hook must exist and default to honest
        state = np.arange(6, dtype=np.float32)
        assert np.array_equal(np.asarray(agent.perturb_broadcast_state(state, 0)), state)


def test_perturbation_log_integrity(actions):
    """TEST 6 -- the log holds exactly T * len(compromised) entries with sane fields."""
    compromised = [0, 2]
    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier=_SignFlip, compromised_indices=compromised, seed=1)
    seed_all(wrapped)
    wrapped.reset()
    assert wrapped.get_perturbation_log() == []

    n_steps = 20
    for a in actions[:n_steps]:
        wrapped.step(a.copy())

    entries = wrapped.get_perturbation_log()
    assert len(entries) == n_steps * len(compromised)

    for k, entry in enumerate(entries):
        assert entry['t'] == k // len(compromised)
        assert entry['agent_idx'] == compromised[k % len(compromised)]
        assert entry['honest_action'].shape == (4,)
        assert entry['perturbed_action'].shape == (4,)
        expected = float(np.linalg.norm(entry['perturbed_action'] - entry['honest_action']))
        assert entry['l2_deviation'] == pytest.approx(expected)

    assert np.array_equal(wrapped.byzantine_mask,
                          np.array([True, False, True, False]))
    wrapped.close()


def test_determinism_same_seed_same_perturbations(actions):
    """TEST 7 -- the same seed reproduces the perturbation sequence exactly; a new one does not."""

    def run(seed):
        env = make_env()
        wrapped = ByzantineEnvWrapper(env, tier=_NoisyRng, compromised_indices=[1], seed=seed)
        seed_all(wrapped)
        wrapped.reset()
        for a in actions[:15]:
            wrapped.step(a.copy())
        out = np.array([e['perturbed_action'] for e in wrapped.get_perturbation_log()])
        wrapped.close()
        return out

    a1, a2, other = run(42), run(42), run(43)
    assert np.array_equal(a1, a2), 'same seed produced different perturbations'
    assert not np.array_equal(a1, other), 'seed is being ignored'


# --------------------------------------------------------------------------- extras

def test_action_range_is_read_from_env_not_hardcoded():
    """The wrapper must take low/high from env.action_space (the spec's explicit warning)."""
    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier='tier0_noop', compromised_indices=[0])
    assert np.array_equal(wrapped.action_low, np.asarray(env.action_space.low, dtype=np.float64))
    assert np.array_equal(wrapped.action_high, np.asarray(env.action_space.high, dtype=np.float64))
    assert wrapped.action_dim == env.action_space.shape[0] == 4
    wrapped.close()


def test_accepts_list_of_per_agent_actions(actions):
    """sample_factory may hand over a list of rows rather than a stacked array."""
    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier=_SignFlip, compromised_indices=[0], seed=2)
    seed_all(wrapped)
    wrapped.reset()

    a = actions[0]
    wrapped.step([row.copy() for row in a])
    assert np.allclose(wrapped.last_executed_actions[0], -a[0])
    assert np.array_equal(wrapped.last_executed_actions[1], a[1])
    wrapped.close()


def test_random_assignment_resamples_across_episodes(actions):
    """`random` mode changes the compromised set at episode boundaries; `fixed` does not."""
    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier='tier0_noop', compromised_indices=[0],
                                  seed=11, assignment_mode='random')
    seen = set()
    for _ in range(8):
        seed_all(wrapped)
        wrapped.reset()
        wrapped.step(actions[0].copy())
        assert len(wrapped.compromised_indices) == 1
        seen.add(tuple(wrapped.compromised_indices))
    wrapped.close()
    assert len(seen) > 1, 'random assignment never resampled'

    env = make_env()
    fixed = ByzantineEnvWrapper(env, tier='tier0_noop', compromised_indices=[3], seed=11)
    for _ in range(4):
        seed_all(fixed)
        fixed.reset()
        assert fixed.compromised_indices == [3]
    fixed.close()


def test_tier_with_strict_constructor_still_learns_the_action_range():
    """A tier may declare an __init__ without **kwargs; it must still get low/high."""

    class _Strict(ByzantineAgent):
        def __init__(self, agent_idx, action_dim, seed=0, gain=2.0):
            super().__init__(agent_idx, action_dim, seed)
            self.gain = gain

        def perturb_action(self, honest_action, obs, t):
            return self.clip(self.gain * honest_action)

    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier=_Strict, compromised_indices=[0],
                                  tier_config=dict(gain=3.0))
    agent = wrapped.byz_agents[0]
    assert agent.gain == 3.0
    assert np.array_equal(agent.action_low, wrapped.action_low)
    assert np.array_equal(agent.action_high, wrapped.action_high)

    seed_all(wrapped)
    wrapped.reset()
    a = np.full((NUM_AGENTS, 4), 0.9, dtype=np.float32)
    wrapped.step(a)
    assert np.all(wrapped.last_executed_actions[0] == 1.0)  # 3*0.9 clipped to the env range
    wrapped.close()


def test_clip_helper():
    x = np.array([-5.0, 0.25, 5.0])
    assert np.array_equal(clip_to_action_space(x), np.array([-1.0, 0.25, 1.0]))
    assert np.array_equal(clip_to_action_space(x, -0.5, 0.5), np.array([-0.5, 0.25, 0.5]))


def test_rejects_bad_configuration():
    env = make_env()
    with pytest.raises(ValueError):
        ByzantineEnvWrapper(env, tier='tier0_noop', compromised_indices=[NUM_AGENTS])
    with pytest.raises(ValueError):
        ByzantineEnvWrapper(env, tier='tier0_noop', compromised_indices=[0, 0])
    with pytest.raises(KeyError):
        ByzantineEnvWrapper(env, tier='no_such_tier', compromised_indices=[0])
    with pytest.raises(ValueError):
        ByzantineEnvWrapper(env, tier='tier0_noop', compromised_indices=[0],
                            assignment_mode='sometimes')
    env.close()


# ------------------------------------------------------- validation gate (needs checkpoint)

CHECKPOINT = 'train_dir/exp1_baseline_seed1'


@pytest.mark.skipif(not os.path.isdir(CHECKPOINT),
                    reason=f'{CHECKPOINT} not present')
def test_noop_matches_unwrapped_on_the_real_policy_stack():
    """
    The validation gate, made exact.

    A 100-episode reward comparison cannot resolve this: episode reward here is heavily
    left-tailed (rare crash episodes dominate the mean), so n=100 pins the mean only to
    about +-1. Instead run the *real* checkpoint through both the wrapped and unwrapped
    stacks in one process, with every RNG pinned, and require the trajectories to be
    bit-identical. That settles "the wrapper changes nothing" outright.

    `quads_use_numba` is forced off: the jitted noise paths (sense_noise.add_noise_numba,
    quadrotor_dynamics.step1_numba) draw from numba's own RNG state, which np.random.seed()
    does not reach, so with numba on the stack is not even reproducible against *itself*.
    That is an upstream property of the simulator, unrelated to the wrapper -- verified by
    running this same comparison with numba on and off.
    """
    import torch
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.algo.utils.action_distributions import argmax_actions
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.model.model_utils import get_rnn_size
    from swarm_rl.train import register_swarm_components

    from byzantine_experiments.rollout.collect import build_cfg, load_policy
    from byzantine_experiments.rollout.env_factory import make_byzantine_env

    register_swarm_components()
    cfg = build_cfg(CHECKPOINT, [])
    cfg.device = 'cpu'
    cfg.quads_use_numba = False
    device = torch.device('cpu')
    n_steps = 200

    def run(byzantine_kwargs):
        random.seed(2024)
        np.random.seed(2024)
        torch.manual_seed(2024)
        env, _ = make_byzantine_env(cfg, byzantine_kwargs)
        actor_critic, _ = load_policy(cfg, env, device)

        seed_all(env, 2024)
        obs, _ = env.reset()
        rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)],
                                 dtype=torch.float32, device=device)
        env_info = extract_env_info(env, cfg)

        obs_trace, rew_trace, act_trace = [], [], []
        with torch.no_grad():
            for _ in range(n_steps):
                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                policy_outputs = actor_critic(normalized_obs, rnn_states)
                actions = argmax_actions(actor_critic.action_distribution())
                actions = preprocess_actions(env_info, actions)
                rnn_states = policy_outputs['new_rnn_states']

                obs, rew, _, _, _ = env.step(actions)
                obs_trace.append(obs['obs'].cpu().numpy().copy())
                rew_trace.append(rew.cpu().numpy().copy())
                act_trace.append(np.asarray(actions).copy())
        env.close()
        return np.array(obs_trace), np.array(rew_trace), np.array(act_trace)

    plain = run({})
    wrapped = run(dict(tier='tier0_noop', compromised_indices=[0, 1], seed=7))

    for name, a, b in zip(('obs', 'rewards', 'actions'), plain, wrapped):
        assert np.array_equal(a, b), f'no-op wrapper changed the {name} trace on the real stack'
