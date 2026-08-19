"""
Tier 2 validation suite — the nine checks required by specs/TIER2_SIGNFLIP.md.

Tests 4 and 5 (the liar reports false state / the liar's own actions stay honest) are
together the *definition* of a state-misreporting attack, and are written first as the spec
asks. They are also the tests most likely to catch a silent no-op, because the neighbour
block is ordered by proximity rank rather than by agent id.

Efficacy (test 9) needs a trained checkpoint and real rollouts; it self-skips without one and
is marked slow.

Run:  pytest byzantine_experiments/tests/test_tier2.py -v
"""

import os

import numpy as np
import pytest

from byzantine_experiments.agents.base import BYZANTINE_REGISTRY, get_byzantine_class
from byzantine_experiments.agents.control_utils import (
    action_to_body,
    action_to_thrust,
    body_to_action,
    position_control_action,
    thrust_to_action,
)
from byzantine_experiments.agents.tier2_structured import (
    CollusionAgent,
    SignFlipAgent,
    StateLiarAgent,
    WaypointPullAgent,
)
from byzantine_experiments.tests.test_tier0 import NUM_AGENTS, make_env, seed_all
from byzantine_experiments.wrappers.byzantine_env import ByzantineEnvWrapper

ACTION_DIM = 4
EGO_DIM = 18
WIDE = dict(action_low=-1e9, action_high=1e9)


def make_agent(name, **cfg):
    return get_byzantine_class(name)(agent_idx=0, action_dim=ACTION_DIM, seed=0, **cfg)


def drive(wrapped, n_steps, seed=0, amplitude=0.5):
    """Step the wrapped env with a fixed pseudo-random action stream."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n_steps):
        a = rng.uniform(-amplitude, amplitude, size=(NUM_AGENTS, 4)).astype(np.float32)
        obs, *_ = wrapped.step(a)
        out.append((a, np.asarray(obs).copy(), wrapped.last_executed_actions.copy()))
    return out


# ============================================================ 4 & 5: the liar (write first)

def test_liar_neighbours_see_the_false_state_and_the_liar_does_not():
    """
    TEST 4 — every *other* drone's observation of the liar carries the lie; the liar's own
    observation is untouched.

    Verified against a from-scratch reconstruction of the neighbour block, using the env's
    own `neighborhood_indices()` for the proximity ordering — which is the part that makes
    this attack easy to get silently wrong.
    """
    env = make_env()
    liar_idx = 1
    offset = 3.0
    wrapped = ByzantineEnvWrapper(
        env, tier='tier2_liar', compromised_indices=[liar_idx],
        tier_config=dict(lie_mode='offset', magnitude=offset), seed=5)
    seed_all(wrapped)
    obs, _ = wrapped.reset()

    u = wrapped.env.unwrapped
    lo, hi = u.clip_neighbor_space_min_box, u.clip_neighbor_space_max_box
    liar = wrapped.byz_agents[liar_idx]

    checked = 0
    for _ in range(15):
        rng = np.random.default_rng(0)
        obs, *_ = wrapped.step(rng.uniform(-0.4, 0.4, size=(NUM_AGENTS, 4)).astype(np.float32))
        obs = np.asarray(obs)

        pos = np.array([e.dynamics.pos for e in u.envs])
        vel = np.array([e.dynamics.vel for e in u.envs])
        idx = u.neighborhood_indices()
        expected_pos = pos[liar_idx] + offset * liar.offset_dir

        for observer in range(NUM_AGENTS):
            if observer == liar_idx:
                continue
            for slot, j in enumerate(idx[observer]):
                base = EGO_DIM + 6 * slot
                seen_rel = obs[observer, base:base + 3]
                if int(j) == liar_idx:
                    want = np.clip(expected_pos - pos[observer], lo[:3], hi[:3])
                    assert np.allclose(seen_rel, want, atol=1e-4), (
                        f'observer {observer} slot {slot}: saw {seen_rel}, expected the lie {want}')
                    # and it must genuinely differ from the truth
                    truth = np.clip(pos[liar_idx] - pos[observer], lo[:3], hi[:3])
                    assert not np.allclose(seen_rel, truth, atol=1e-3), 'lie had no effect'
                    checked += 1
                else:
                    want = np.clip(pos[int(j)] - pos[observer], lo[:3], hi[:3])
                    assert np.allclose(seen_rel, want, atol=1e-4), (
                        f'observer {observer} slot {slot} (honest neighbour {j}) was corrupted')

    assert checked > 0, 'the liar never appeared in any neighbour block — test proved nothing'
    assert wrapped.broadcast_rewrites > 0
    wrapped.close()


def test_liar_own_observation_is_unchanged():
    """TEST 4b — the liar's own observation vector is bit-identical to the unwrapped case."""
    liar_idx = 2
    actions = np.random.default_rng(3).uniform(-0.4, 0.4, size=(12, NUM_AGENTS, 4)).astype(np.float32)

    plain = make_env()
    seed_all(plain)
    obs_plain, _ = plain.reset()
    trace_plain = []
    for a in actions:
        o, *_ = plain.step(a.copy())
        trace_plain.append(np.asarray(o)[liar_idx].copy())
    plain.close()

    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier='tier2_liar', compromised_indices=[liar_idx],
                                  tier_config=dict(lie_mode='mirror'), seed=5)
    seed_all(wrapped)
    wrapped.reset()
    trace_lie = []
    for a in actions:
        o, *_ = wrapped.step(a.copy())
        trace_lie.append(np.asarray(o)[liar_idx].copy())
    wrapped.close()

    assert np.array_equal(np.array(trace_plain), np.array(trace_lie)), \
        "the liar's own observation was modified"


def test_liar_actions_are_bit_identical_to_honest():
    """
    TEST 5 — the liar's executed actions equal the honest policy's exactly.

    Together with test 4 this is the definition of a state-misreporting attack: false
    communication, honest control.
    """
    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier='tier2_liar', compromised_indices=[0, 2],
                                  tier_config=dict(lie_mode='stale', delay_k=10), seed=1)
    seed_all(wrapped)
    wrapped.reset()

    for honest, _, executed in drive(wrapped, 20, seed=7):
        assert np.array_equal(executed, honest), 'the liar perturbed its own action'
    wrapped.close()


@pytest.mark.parametrize('lie_mode', ['offset', 'stale', 'mirror'])
def test_liar_modes_produce_the_documented_lie(lie_mode):
    agent = make_agent('tier2_liar', lie_mode=lie_mode, magnitude=2.0, delay_k=5,
                       mirror_centre=(0.0, 0.0, 2.0))
    true = np.array([1.0, 2.0, 3.0, 0.5, -0.5, 0.25])

    if lie_mode == 'offset':
        out = agent.perturb_broadcast_state(true.copy(), 0)
        assert np.allclose(np.linalg.norm(out[:3] - true[:3]), 2.0)
        assert np.allclose(out[3:], true[3:]), 'offset should not touch velocity by default'
    elif lie_mode == 'stale':
        states = [true + i for i in range(8)]
        outs = [agent.perturb_broadcast_state(s.copy(), i) for i, s in enumerate(states)]
        # once the buffer is full the report lags by exactly delay_k steps
        assert np.allclose(outs[-1][:3], states[-1 - 5][:3]), 'stale lag is wrong'
    else:
        out = agent.perturb_broadcast_state(true.copy(), 0)
        assert np.allclose(out[:3], 2 * np.array([0.0, 0.0, 2.0]) - true[:3])

    assert np.asarray(out if lie_mode != 'stale' else outs[-1]).shape == (6,)


# ==================================================================== 1 & 2: sign flip

def test_signflip_motor_space_exact():
    """TEST 1 — a' == -scale * a on the targeted dims."""
    for scale in (0.5, 1.0):
        agent = make_agent('tier2_signflip', scale=scale, dims='all', space='motor',
                           survival_guard=False)
        a = np.array([0.3, -0.4, 0.5, -0.6])
        out = agent.perturb_action(a.copy(), None, 0)
        assert np.allclose(out, np.clip(-scale * a, -1, 1))


def test_signflip_partial_leaves_untargeted_dims_bit_identical():
    """TEST 2 — untargeted dimensions pass through untouched."""
    agent = make_agent('tier2_signflip', scale=1.0, dims=[1, 2], space='motor',
                       survival_guard=False)
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform(-1, 1, ACTION_DIM)
        out = agent.perturb_action(a.copy(), None, 0)
        assert out[0] == a[0] and out[3] == a[3], 'untargeted dims were modified'
        assert np.allclose(out[[1, 2]], -a[[1, 2]])


def test_signflip_body_space_preserves_collective_thrust():
    """
    The canonical partial setting: flipping roll/pitch in body space must leave the
    collective-thrust component essentially untouched, which is what keeps the drone up.
    """
    env = make_env()
    wrapped = ByzantineEnvWrapper(
        env, tier='tier2_signflip', compromised_indices=[0],
        tier_config=dict(scale=1.0, dims=[1, 2], space='body', survival_guard=False), seed=0)
    seed_all(wrapped)
    wrapped.reset()

    agent = wrapped.byz_agents[0]
    ratios, flips = [], []
    for honest, _, executed in drive(wrapped, 25, seed=2, amplitude=0.3):
        J, Jinv = agent.ego.jacobian, agent.ego.jacobian_inv
        b_in = action_to_body(honest[0].astype(np.float64), J)
        b_out = action_to_body(executed[0].astype(np.float64), J)
        if abs(b_in[0]) > 1e-6:
            ratios.append(b_out[0] / b_in[0])
        # roll and pitch should have flipped sign
        flips.append(np.sign(b_out[1]) != np.sign(b_in[1]) or abs(b_in[1]) < 1e-9)
    wrapped.close()

    assert np.mean(ratios) == pytest.approx(1.0, abs=0.05), \
        f'collective thrust not preserved: mean ratio {np.mean(ratios):.3f}'
    assert np.mean(flips) > 0.8, 'roll component was not flipped'


def test_body_motor_roundtrip():
    """The Jacobian decomposition is invertible, so `space='body'` is well posed."""
    env = make_env()
    from byzantine_experiments.agents.control_utils import jacobians_for
    J, Jinv = jacobians_for(env.unwrapped.envs[0].dynamics)
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.uniform(-1, 1, 4)
        assert np.allclose(body_to_action(action_to_body(a, J), Jinv), a, atol=1e-9)
    assert np.allclose(action_to_thrust(thrust_to_action(np.array([0.2, 0.4, 0.6, 0.8]))),
                       [0.2, 0.4, 0.6, 0.8])
    env.close()


# ===================================================================== 3: waypoint pull

def test_waypoint_pull_points_toward_the_waypoint():
    """
    TEST 3 — the attack accelerates the drone toward `w_adv`.

    Checked in acceleration space rather than raw motor space: the meaningful claim is that
    the commanded thrust direction has a positive component along the direction to the
    waypoint, once gravity compensation is removed.
    """
    env = make_env()
    w_adv = np.array([3.0, 3.0, 0.8])
    wrapped = ByzantineEnvWrapper(
        env, tier='tier2_waypoint_pull', compromised_indices=[0],
        tier_config=dict(w_adv=w_adv.tolist(), alpha=1.0, survival_guard=False), seed=0)
    seed_all(wrapped)
    wrapped.reset()

    agent = wrapped.byz_agents[0]
    cosines = []
    for honest, _, executed in drive(wrapped, 25, seed=4, amplitude=0.2):
        ego = agent.ego
        to_wp = w_adv - ego.pos
        n = np.linalg.norm(to_wp)
        if n < 1e-6:
            continue
        # Commanded acceleration in world frame, minus gravity compensation.
        thrusts = action_to_thrust(executed[0].astype(np.float64))
        acc_body_z = float(ego.jacobian[0] @ thrusts)
        acc_world = acc_body_z * ego.rot[:, 2] - np.array([0.0, 0.0, 9.81])
        if np.linalg.norm(acc_world) > 1e-3:
            cosines.append(float(acc_world @ (to_wp / n) / np.linalg.norm(acc_world)))
    wrapped.close()

    assert len(cosines) > 5
    assert np.mean(cosines) > 0, (
        f'commanded acceleration does not point toward the waypoint '
        f'(mean cosine {np.mean(cosines):.3f})')


def test_waypoint_direction_helper_and_default_corner():
    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier='tier2_waypoint_pull', compromised_indices=[0],
                                  tier_config=dict(alpha=0.6), seed=0)
    seed_all(wrapped)
    wrapped.reset()
    drive(wrapped, 2, seed=0)
    agent = wrapped.byz_agents[0]

    room_hi = agent.ego.room_box[1]
    assert agent.w_adv is not None
    assert np.all(agent.w_adv <= room_hi + 1e-9), 'waypoint is outside the arena'
    d = agent.direction_to_waypoint()
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert float(d @ (agent.w_adv - agent.ego.pos)) > 0
    wrapped.close()


def test_waypoint_alpha_scales_the_departure_from_honest():
    """alpha=0 must be a no-op; larger alpha must deviate more."""
    devs = {}
    for alpha in (0.0, 0.3, 0.6, 1.0):
        env = make_env()
        wrapped = ByzantineEnvWrapper(
            env, tier='tier2_waypoint_pull', compromised_indices=[0],
            tier_config=dict(alpha=alpha, w_adv=[3.0, 3.0, 0.8], survival_guard=False), seed=0)
        seed_all(wrapped)
        wrapped.reset()
        d = [np.linalg.norm(ex[0] - h[0]) for h, _, ex in drive(wrapped, 20, seed=6, amplitude=0.3)]
        devs[alpha] = float(np.mean(d))
        wrapped.close()

    assert devs[0.0] == pytest.approx(0.0, abs=1e-9), 'alpha=0 should be honest'
    assert devs[0.3] < devs[0.6] < devs[1.0], f'not monotonic in alpha: {devs}'


# ========================================================================= 6: collusion

@pytest.mark.parametrize('mode', ['same_waypoint', 'pincer', 'mirror_pair'])
def test_collusion_agents_are_correlated_as_specified(mode):
    """TEST 6 — the two attackers' perturbations relate as the mode promises."""
    env = make_env()
    wrapped = ByzantineEnvWrapper(
        env, tier='tier2_collusion', compromised_indices=[0, 1],
        tier_config=dict(mode=mode, alpha=1.0, survival_guard=False), seed=0)
    seed_all(wrapped)
    wrapped.reset()
    steps = drive(wrapped, 20, seed=8, amplitude=0.3)

    a0, a1 = wrapped.byz_agents[0], wrapped.byz_agents[1]
    assert {a0.rank, a1.rank} == {0, 1}, 'ranks were not assigned deterministically'

    if mode == 'same_waypoint':
        assert np.allclose(a0.w_adv, a1.w_adv), 'colluders should share one waypoint'
    elif mode == 'pincer':
        assert not np.allclose(a0.w_adv, a1.w_adv), 'pincer should use opposite corners'
        assert float((a0.w_adv - a1.w_adv) @ np.array([1.0, 1.0, 0.0])) > 0
    else:
        # anti-correlated motor commands: disjoint flipped dim sets
        honest, _, executed = steps[-1]
        d0 = executed[0] - honest[0]
        d1 = executed[1] - honest[1]
        assert np.count_nonzero(np.abs(d0) > 1e-9) > 0
        assert np.count_nonzero(np.abs(d1) > 1e-9) > 0
        assert np.all(np.abs(d0[[2, 3]]) < 1e-9), 'rank 0 should only flip dims 0,1'
        assert np.all(np.abs(d1[[0, 1]]) < 1e-9), 'rank 1 should only flip dims 2,3'
    wrapped.close()


def test_collusion_channel_is_not_readable_from_honest_state():
    """Colluders share a channel, but it only ever holds what they put there."""
    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier='tier2_collusion', compromised_indices=[0, 1],
                                  tier_config=dict(mode='same_waypoint'), seed=0)
    seed_all(wrapped)
    wrapped.reset()
    drive(wrapped, 5, seed=0)
    keys = set(wrapped._coordination)
    assert keys <= {'members', 'target_0', 'target_1'}, f'unexpected channel contents: {keys}'
    wrapped.close()


# ============================================== 7: determinism / range / isolation

TIER2_CONFIGS = [
    ('tier2_signflip', dict(scale=1.0, dims='all', space='motor')),
    ('tier2_signflip', dict(scale=1.0, dims=[1, 2], space='body')),
    ('tier2_waypoint_pull', dict(alpha=0.6)),
    ('tier2_liar', dict(lie_mode='offset', magnitude=3.0)),
    ('tier2_collusion', dict(mode='pincer', alpha=0.6)),
]


@pytest.mark.parametrize('name,cfg', TIER2_CONFIGS)
def test_range_compliance_and_isolation(name, cfg):
    """TEST 7 — outputs stay legal and honest drones are untouched."""
    env = make_env()
    low, high = env.action_space.low, env.action_space.high
    compromised = [0, 1] if name == 'tier2_collusion' else [1]
    wrapped = ByzantineEnvWrapper(env, tier=name, compromised_indices=compromised,
                                  tier_config=cfg, seed=3)
    seed_all(wrapped)
    wrapped.reset()

    for honest, _, executed in drive(wrapped, 30, seed=9, amplitude=0.6):
        assert np.all(executed >= low) and np.all(executed <= high)
        for i in range(NUM_AGENTS):
            if i not in compromised:
                assert np.array_equal(executed[i], honest[i]), f'honest drone {i} modified'
    wrapped.close()


@pytest.mark.parametrize('name,cfg', TIER2_CONFIGS)
def test_determinism(name, cfg):
    """TEST 7b — Tier 2 is rule-based: identical seeds give identical action streams."""
    def run(seed):
        env = make_env()
        compromised = [0, 1] if name == 'tier2_collusion' else [1]
        wrapped = ByzantineEnvWrapper(env, tier=name, compromised_indices=compromised,
                                      tier_config=cfg, seed=seed)
        seed_all(wrapped)
        wrapped.reset()
        out = np.array([ex for _, _, ex in drive(wrapped, 20, seed=11)])
        wrapped.close()
        return out

    assert np.array_equal(run(4), run(4)), f'{name} is not deterministic'


# ================================================================== 8: survival guard

def test_survival_guard_fires_below_threshold_and_blends_toward_honest():
    """TEST 8 — the guard engages as altitude falls and is a no-op when disabled."""
    from byzantine_experiments.agents.base import EgoState

    agent = make_agent('tier2_signflip', scale=1.0, dims='all', space='motor',
                       survival_guard=True, guard_altitude=0.6, guard_full_altitude=0.15)
    honest = np.array([0.4, 0.4, 0.4, 0.4])

    def at(z):
        agent.ego = EgoState(pos=np.array([0.0, 0.0, z]), vel=np.zeros(3),
                             rot=np.eye(3), omega=np.zeros(3))
        return np.asarray(agent.perturb_action(honest.copy(), None, 0)), agent.last_guard_blend

    high_out, blend_high = at(2.0)
    mid_out, blend_mid = at(0.375)          # halfway through the ramp
    low_out, blend_low = at(0.10)           # fully engaged

    assert blend_high == 0.0 and np.allclose(high_out, -honest)
    assert blend_mid == pytest.approx(0.5, abs=0.02)
    assert np.allclose(mid_out, 0.5 * (-honest) + 0.5 * honest, atol=1e-9)
    assert blend_low == 1.0 and np.allclose(low_out, honest), 'guard should be fully honest'
    assert 0 < agent.guard_trigger_rate <= 1

    off = make_agent('tier2_signflip', scale=1.0, dims='all', survival_guard=False)
    off.ego = EgoState(pos=np.array([0.0, 0.0, 0.05]), vel=np.zeros(3),
                       rot=np.eye(3), omega=np.zeros(3))
    assert np.allclose(off.perturb_action(honest.copy(), None, 0), -honest)
    assert off.guard_trigger_rate == 0.0


# ====================================================================== registry / config

def test_all_four_variants_registered():
    for name, cls in (('tier2_signflip', SignFlipAgent), ('tier2_waypoint_pull', WaypointPullAgent),
                      ('tier2_liar', StateLiarAgent), ('tier2_collusion', CollusionAgent)):
        assert BYZANTINE_REGISTRY[name] is cls


def test_capability_flags():
    """Only 2C lies; only the goal-directed variants ask for ego state."""
    assert SignFlipAgent.needs_ego_state and not SignFlipAgent.lies_about_state
    assert WaypointPullAgent.needs_ego_state
    assert StateLiarAgent.lies_about_state and not StateLiarAgent.needs_ego_state
    assert CollusionAgent.needs_coordination
    # Tier 1 must be entirely unaffected by the new capabilities
    for t1 in ('tier1_gaussian', 'tier1_ou', 'tier1_dropout', 'tier1_stall'):
        cls = get_byzantine_class(t1)
        assert not cls.needs_ego_state and not cls.lies_about_state and not cls.needs_coordination


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        make_agent('tier2_signflip', dims=[9])
    with pytest.raises(ValueError):
        make_agent('tier2_signflip', space='quaternion')
    with pytest.raises(ValueError):
        make_agent('tier2_waypoint_pull', alpha=1.5)
    with pytest.raises(ValueError):
        make_agent('tier2_liar', lie_mode='fib')
    with pytest.raises(ValueError):
        make_agent('tier2_liar', delay_k=0)
    with pytest.raises(ValueError):
        make_agent('tier2_collusion', mode='swarm')
    with pytest.raises(ValueError):
        make_agent('tier2_signflip', guard_altitude=0.1, guard_full_altitude=0.5)


# ========================================================== 9: efficacy (opt-in, slow)

CHECKPOINT = 'train_dir/exp1_baseline_seed1'
SLOW = os.environ.get('BYZ_SLOW_TESTS') == '1'

CANONICAL = [
    ('tier2_signflip', dict(scale=1.0, dims=[1, 2], space='body'), [0]),
    ('tier2_waypoint_pull', dict(alpha=1.0), [0]),
    ('tier2_collusion', dict(mode='pincer', alpha=0.6), [0, 1]),
]


@pytest.mark.skipif(not SLOW or not os.path.isdir(CHECKPOINT),
                    reason='set BYZ_SLOW_TESTS=1 (needs the checkpoint; ~1 min per config)')
@pytest.mark.parametrize('tier,cfg,indices', CANONICAL)
def test_efficacy_attack_degrades_team_reward(tier, cfg, indices):
    """
    TEST 9 — every canonical config must actually degrade the team. An attack that does
    nothing is not a Byzantine agent.

    This is a smoke check on 3 episodes; the real evidence is the 57-run matrix and its
    calibration table. 2C (liar) is deliberately excluded: its *actions* are honest by
    construction, so it is expected to move team reward far less than the action-space
    attacks — see design/tier2_design.md on the 2C measurement problem.
    """
    import torch
    from swarm_rl.train import register_swarm_components
    from byzantine_experiments.rollout.collect import build_cfg, run_rollouts

    register_swarm_components()
    cfg_sf = build_cfg(CHECKPOINT, [])
    cfg_sf.device = 'cpu'

    def team_reward(byz_kwargs):
        eps, _, _, summary = run_rollouts(cfg_sf, byz_kwargs, n_episodes=3, seed=1)
        return summary['episode_reward_mean']

    clean = team_reward({})
    attacked = team_reward(dict(tier=tier, compromised_indices=indices,
                                tier_config=cfg, seed=1))
    drop_pct = 100.0 * (clean - attacked) / abs(clean) if abs(clean) > 1e-9 else float('inf')
    print(f'\n  {tier} {cfg}: clean={clean:.3f} attacked={attacked:.3f} drop={drop_pct:.0f}%')
    assert attacked < clean, f'{tier} did not degrade the team at all'
    assert drop_pct >= 20.0, f'{tier} degraded team reward by only {drop_pct:.0f}% (<20%)'
