"""
Tier 1 validation suite — the eight tests required by specs/TIER1_STOCHASTIC.md.

These check the *statistical* contract of the negative control (zero-mean, variance
scaling, dropout rate, no team-state access), which is what makes Tier 1 usable as a
calibrated reference rather than just another perturbation. Env-level properties (range
compliance, isolation) reuse the Tier 0 harness.

Run:  pytest byzantine_experiments/tests/test_tier1.py -v
"""

import numpy as np
import pytest

from byzantine_experiments.agents.base import BYZANTINE_REGISTRY, get_byzantine_class
from byzantine_experiments.agents.tier1_stochastic import (
    DropoutAgent,
    GaussianNoiseAgent,
    OUNoiseAgent,
)
from byzantine_experiments.tests.test_tier0 import NUM_AGENTS, make_env, seed_all
from byzantine_experiments.wrappers.byzantine_env import ByzantineEnvWrapper

ACTION_DIM = 4
N_STEPS = 10000

# Wide bounds so the raw noise is observable without clipping interference. Tests that care
# about the real range build a real env instead.
WIDE = dict(action_low=-1e9, action_high=1e9)


def make_agent(name, **cfg):
    return get_byzantine_class(name)(agent_idx=0, action_dim=ACTION_DIM, seed=0, **WIDE, **cfg)


def collect_noise(agent, n=N_STEPS, honest=None):
    """Run the agent for n steps and return the per-step pre-clip perturbation, (n, 4)."""
    honest = np.zeros(ACTION_DIM) if honest is None else honest
    out = np.empty((n, ACTION_DIM))
    for t in range(n):
        agent.perturb_action(honest.copy(), None, t)
        out[t] = agent.last_noise
    return out


# --------------------------------------------------------------------------- 1. zero mean

@pytest.mark.parametrize('name,cfg', [
    ('tier1_gaussian', dict(sigma=0.25)),
    ('tier1_gaussian', dict(sigma=0.5)),
    ('tier1_ou', dict(sigma_ou=0.2)),
    ('tier1_ou', dict(sigma_ou=0.4)),
])
def test_zero_mean_perturbation(name, cfg):
    """
    TEST 1 -- 1A and 1B are zero-mean in expectation, pre-clipping (constraint 2).

    Tolerance is 2 standard errors of the mean. For OU the samples are autocorrelated, so
    the naive SE understates the true uncertainty; the effective sample size is inflated by
    the integrated autocorrelation time tau = (1+rho)/(1-rho), and that is corrected for
    here rather than papered over with a looser constant.
    """
    agent = make_agent(name, **cfg)
    noise = collect_noise(agent)

    mean = noise.mean(axis=0)
    if name == 'tier1_ou':
        rho = 1.0 - agent.theta * agent.dt          # lag-1 autocorrelation of the AR(1)
        n_eff = len(noise) * (1.0 - rho) / (1.0 + rho)
    else:
        n_eff = len(noise)
    se = noise.std(axis=0) / np.sqrt(n_eff)

    print(f'\n{name} {cfg}: mean/dim={np.round(mean, 5)}  2SE={np.round(2 * se, 5)}')
    assert np.all(np.abs(mean) < 2 * se), (
        f'{name} {cfg} is not zero-mean: mean={mean}, 2SE={2 * se}')


# ----------------------------------------------------------------- 2. variance scaling

def test_variance_scales_with_sigma():
    """TEST 2 -- pre-clipping std scales linearly with the configured severity."""
    print()
    gauss = {}
    for sigma in (0.1, 0.25, 0.5):
        std = collect_noise(make_agent('tier1_gaussian', sigma=sigma)).std()
        gauss[sigma] = std
        print(f'  gaussian sigma={sigma}: empirical std={std:.4f}')
        assert std == pytest.approx(sigma, rel=0.05), f'sigma={sigma} -> std={std}'

    ratios = [gauss[0.25] / gauss[0.1], gauss[0.5] / gauss[0.25]]
    assert ratios[0] == pytest.approx(2.5, rel=0.05)
    assert ratios[1] == pytest.approx(2.0, rel=0.05)

    # OU scales linearly too, but about its own stationary std, not sigma_ou.
    ou = {}
    for sigma_ou in (0.1, 0.2, 0.4):
        agent = make_agent('tier1_ou', sigma_ou=sigma_ou)
        std = collect_noise(agent).std()
        ou[sigma_ou] = std
        print(f'  ou sigma_ou={sigma_ou}: empirical std={std:.4f} '
              f'(analytic stationary {agent.stationary_std():.4f})')
        assert std == pytest.approx(agent.stationary_std(), rel=0.10)

    assert ou[0.2] / ou[0.1] == pytest.approx(2.0, rel=0.05)
    assert ou[0.4] / ou[0.2] == pytest.approx(2.0, rel=0.05)


# --------------------------------------------------------------------- 3. determinism

@pytest.mark.parametrize('name,cfg', [
    ('tier1_gaussian', dict(sigma=0.25)),
    ('tier1_ou', dict(sigma_ou=0.2)),
    ('tier1_dropout', dict(p_drop=0.15, mode='hold')),
])
def test_determinism(name, cfg):
    """TEST 3 -- same seed reproduces the perturbation sequence; a different seed does not."""
    def run(seed):
        agent = get_byzantine_class(name)(agent_idx=0, action_dim=ACTION_DIM, seed=seed, **WIDE, **cfg)
        rng = np.random.default_rng(123)
        honest = rng.uniform(-1, 1, size=(300, ACTION_DIM))
        return np.array([agent.perturb_action(h.copy(), None, t) for t, h in enumerate(honest)])

    assert np.array_equal(run(7), run(7)), f'{name} is not deterministic under a fixed seed'
    assert not np.array_equal(run(7), run(8)), f'{name} ignores its seed'


def test_per_agent_streams_are_independent():
    """Two compromised drones must not receive identical noise (the seed is offset by index)."""
    a0 = GaussianNoiseAgent(agent_idx=0, action_dim=ACTION_DIM, seed=5, sigma=0.25, **WIDE)
    a1 = GaussianNoiseAgent(agent_idx=1, action_dim=ACTION_DIM, seed=5, sigma=0.25, **WIDE)
    n0, n1 = collect_noise(a0, 500), collect_noise(a1, 500)
    assert not np.allclose(n0, n1)
    assert abs(np.corrcoef(n0.ravel(), n1.ravel())[0, 1]) < 0.1


# ----------------------------------------------------------------- 4. range compliance

@pytest.mark.parametrize('name,cfg', [
    ('tier1_gaussian', dict(sigma=0.5)),
    ('tier1_ou', dict(sigma_ou=0.4)),
    ('tier1_dropout', dict(p_drop=0.3, mode='zero')),
    ('tier1_dropout', dict(p_drop=0.3, mode='hold')),
])
def test_range_compliance(name, cfg):
    """TEST 4 -- every emitted action stays inside env.action_space, at the worst severity."""
    env = make_env()
    low, high = env.action_space.low, env.action_space.high
    wrapped = ByzantineEnvWrapper(env, tier=name, compromised_indices=[0],
                                  tier_config=cfg, seed=3)
    seed_all(wrapped)
    wrapped.reset()

    rng = np.random.default_rng(1)
    for t in range(60):
        a = rng.uniform(-1, 1, size=(NUM_AGENTS, 4)).astype(np.float32)
        wrapped.step(a)
        executed = wrapped.last_executed_actions
        assert np.all(executed >= low) and np.all(executed <= high)
        assert env.action_space.contains(executed[0].astype(env.action_space.dtype))
    wrapped.close()


# ----------------------------------------------------------------------- 5. isolation

@pytest.mark.parametrize('name,cfg', [
    ('tier1_gaussian', dict(sigma=0.5)),
    ('tier1_ou', dict(sigma_ou=0.4)),
    ('tier1_dropout', dict(p_drop=0.3, mode='hold')),
])
def test_isolation(name, cfg):
    """TEST 5 -- honest drones' actions reach the env untouched, and the fault does fire."""
    env = make_env()
    compromised = [1, 3]
    wrapped = ByzantineEnvWrapper(env, tier=name, compromised_indices=compromised,
                                  tier_config=cfg, seed=11)
    seed_all(wrapped)
    wrapped.reset()

    rng = np.random.default_rng(2)
    touched = 0
    for t in range(60):
        a = rng.uniform(-0.5, 0.5, size=(NUM_AGENTS, 4)).astype(np.float32)
        wrapped.step(a)
        executed = wrapped.last_executed_actions
        for i in range(NUM_AGENTS):
            if i in compromised:
                touched += not np.array_equal(executed[i], a[i])
            else:
                assert np.array_equal(executed[i], a[i]), f'honest agent {i} was modified'
    assert touched > 0, f'{name} never actually perturbed anything'
    wrapped.close()


# ------------------------------------------------------------------------ 6. OU reset

def test_ou_state_reset():
    """TEST 6 -- OU internal state is zeroed on reset(), so episodes do not inherit drift."""
    agent = make_agent('tier1_ou', sigma_ou=0.4)
    collect_noise(agent, 500)
    assert np.any(agent.x != 0.0), 'OU state never moved; the test is not exercising anything'

    agent.reset()
    assert np.array_equal(agent.x, np.zeros(ACTION_DIM))
    assert np.array_equal(agent.last_noise, np.zeros(ACTION_DIM))

    # First post-reset step must be a single small increment from zero, not a resumption.
    first = np.abs(agent.perturb_action(np.zeros(ACTION_DIM), None, 0))
    assert np.all(first < 6 * agent.sigma_ou)

    # Dropout holds should not survive a reset either.
    drop = make_agent('tier1_dropout', p_drop=1.0, mode='hold')
    drop.perturb_action(np.full(ACTION_DIM, 0.7), None, 0)
    drop.perturb_action(np.full(ACTION_DIM, 0.7), None, 1)
    assert np.allclose(drop.prev_action, 0.0), 'hold repeats the *previous emitted* action'
    drop.reset()
    assert np.array_equal(drop.prev_action, np.zeros(ACTION_DIM))
    assert drop.total_steps == 0


# ------------------------------------------------------------- 7. no team-state access

@pytest.mark.parametrize('name,cfg', [
    ('tier1_gaussian', dict(sigma=0.25)),
    ('tier1_ou', dict(sigma_ou=0.2)),
    ('tier1_dropout', dict(p_drop=0.15, mode='hold')),
])
def test_no_team_state_access(name, cfg):
    """
    TEST 7 (the important one) -- the output must not depend on any neighbour's state.

    The observation is 54-D: dims 0:18 are ego (`pos-goal`, `vel`, `R`, `omega`) and dims
    18:54 are the six neighbour blocks (`rxyz`, `rvxyz`). Two agents with identical seeds
    are fed observations that differ *only* in the neighbour block -- one all-zeros, one
    filled with a loud sentinel. If any variant read team state, the action streams would
    diverge.

    Also feeds a NaN-poisoned neighbour block: reading it would propagate NaN into the
    action, which no amount of coincidence can hide.
    """
    ego_dim, obs_dim = 18, 54
    rng = np.random.default_rng(4)
    honest = rng.uniform(-0.6, 0.6, size=(400, ACTION_DIM))
    ego = rng.normal(size=(400, ego_dim))

    def run(neighbour_fill):
        agent = get_byzantine_class(name)(agent_idx=0, action_dim=ACTION_DIM, seed=99, **WIDE, **cfg)
        acts = []
        for t, (h, e) in enumerate(zip(honest, ego)):
            obs = np.empty(obs_dim)
            obs[:ego_dim] = e
            obs[ego_dim:] = neighbour_fill(t)
            acts.append(agent.perturb_action(h.copy(), obs, t))
        return np.array(acts)

    baseline = run(lambda t: np.zeros(obs_dim - ego_dim))
    loud = run(lambda t: np.full(obs_dim - ego_dim, 1e6 * (t % 7 - 3)))
    poisoned = run(lambda t: np.full(obs_dim - ego_dim, np.nan))

    assert np.array_equal(baseline, loud), (
        f'{name} output changed when only NEIGHBOUR observations changed -- it is reading '
        f'team state, which disqualifies it as a Tier 1 fault')
    assert np.array_equal(baseline, poisoned), f'{name} propagated NaN from the neighbour block'
    assert np.all(np.isfinite(baseline))


def test_variants_do_not_override_broadcast():
    """Constraint 4 -- a degraded drone reports its true state. Lying is Tier 2's job."""
    from byzantine_experiments.agents.base import ByzantineAgent

    for name in ('tier1_gaussian', 'tier1_ou', 'tier1_dropout'):
        cls = get_byzantine_class(name)
        assert cls.perturb_broadcast_state is ByzantineAgent.perturb_broadcast_state, (
            f'{name} overrides perturb_broadcast_state')
        agent = make_agent(name)
        state = np.arange(6, dtype=np.float64)
        assert np.array_equal(agent.perturb_broadcast_state(state, 0), state)


# --------------------------------------------------------------------- 8. dropout rate

@pytest.mark.parametrize('p_drop', [0.05, 0.15, 0.3])
@pytest.mark.parametrize('mode', ['hold', 'zero'])
def test_dropout_rate_matches_config(p_drop, mode):
    """TEST 8 -- empirical dropout frequency matches p_drop within binomial tolerance."""
    agent = make_agent('tier1_dropout', p_drop=p_drop, mode=mode)
    rng = np.random.default_rng(6)
    honest = rng.uniform(-0.8, 0.8, size=(N_STEPS, ACTION_DIM))

    prev = np.zeros(ACTION_DIM)
    observed_drops = 0
    for t, h in enumerate(honest):
        out = agent.perturb_action(h.copy(), None, t)
        if mode == 'zero':
            dropped = np.allclose(out, 0.0)
        else:
            dropped = np.allclose(out, prev)
        observed_drops += dropped
        prev = np.asarray(out, dtype=np.float64)

    se = np.sqrt(p_drop * (1 - p_drop) / N_STEPS)
    rate = agent.empirical_drop_rate
    print(f'\n  dropout p={p_drop} mode={mode}: internal={rate:.4f} '
          f'observed={observed_drops / N_STEPS:.4f} target={p_drop} 3SE={3 * se:.4f}')

    assert abs(rate - p_drop) < 3 * se, f'internal drop rate {rate} != {p_drop}'
    # The externally observable rate must agree too -- that is what actually reaches the env.
    assert abs(observed_drops / N_STEPS - p_drop) < 4 * se


def test_dropout_modes_behave_as_documented():
    """`hold` repeats the last emitted action; `zero` emits hover."""
    hold = make_agent('tier1_dropout', p_drop=1.0, mode='hold')
    a1 = np.array([0.3, -0.4, 0.5, -0.6])
    assert np.allclose(hold.perturb_action(a1, None, 0), 0.0)     # nothing to hold yet
    assert np.allclose(hold.perturb_action(a1, None, 1), 0.0)     # holds the emitted zero

    never = make_agent('tier1_dropout', p_drop=0.0, mode='hold')
    assert np.allclose(never.perturb_action(a1, None, 0), a1)     # passthrough
    assert np.allclose(never.perturb_action(a1 * 0.5, None, 1), a1 * 0.5)

    zero = make_agent('tier1_dropout', p_drop=1.0, mode='zero')
    assert np.allclose(zero.perturb_action(a1, None, 0), 0.0)


# ------------------------------------------------------------------------- registration

def test_all_three_variants_registered():
    for name in ('tier1_gaussian', 'tier1_ou', 'tier1_dropout'):
        assert name in BYZANTINE_REGISTRY
    assert BYZANTINE_REGISTRY['tier1_gaussian'] is GaussianNoiseAgent
    assert BYZANTINE_REGISTRY['tier1_ou'] is OUNoiseAgent
    assert BYZANTINE_REGISTRY['tier1_dropout'] is DropoutAgent


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        make_agent('tier1_gaussian', sigma=-0.1)
    with pytest.raises(ValueError):
        make_agent('tier1_dropout', p_drop=1.5)
    with pytest.raises(ValueError):
        make_agent('tier1_dropout', mode='sometimes')


# ------------------------------------------- 1C' burst stall (tier1_stall, re-parameterised)

def test_stall_registered_and_valid():
    from byzantine_experiments.agents.tier1_stochastic import BurstStallAgent
    assert BYZANTINE_REGISTRY['tier1_stall'] is BurstStallAgent
    agent = make_agent('tier1_stall')
    assert agent.perturb_broadcast_state(np.arange(6.0), 0).tolist() == list(range(6))


@pytest.mark.parametrize('mean_stall_s,duty', [
    (0.1, 0.10), (0.3, 0.25), (1.0, 0.25), (0.3, 0.50),
])
def test_stall_duration_and_duty_cycle_match_config(mean_stall_s, duty):
    """
    The whole point of the re-parameterisation: stalls must actually last as long as asked.

    Checks both the long-run occupancy (duty cycle) and the mean *duration* of a stall,
    which is the quantity the old `p_drop` parameterisation could not control.
    """
    agent = make_agent('tier1_stall', mean_stall_s=mean_stall_s, duty_cycle=duty,
                       control_freq=100.0)
    rng = np.random.default_rng(3)
    honest = rng.uniform(-0.8, 0.8, size=(200000, ACTION_DIM))
    for t, h in enumerate(honest):
        agent.perturb_action(h.copy(), None, t)

    target_steps = mean_stall_s * 100.0
    print(f'\n  mean_stall_s={mean_stall_s} duty={duty}: '
          f'duty={agent.empirical_duty_cycle:.4f} (target {duty}) '
          f'stall_len={agent.empirical_mean_stall_steps:.1f} steps (target {target_steps:.0f})')

    assert agent.empirical_duty_cycle == pytest.approx(duty, rel=0.08)
    assert agent.empirical_mean_stall_steps == pytest.approx(target_steps, rel=0.10)


def test_stall_reduces_to_iid_dropout_in_the_limit():
    """
    The two variants are nested: mean_stall_s = 1/((1-p)*f_ctrl) with duty_cycle = p
    reproduces the old i.i.d. dropout statistics. This is what makes the old Tier 1C results
    still comparable to the new ones rather than a separate universe.
    """
    p = 0.3
    equivalent_mean_stall_s = 1.0 / ((1.0 - p) * 100.0)
    stall = make_agent('tier1_stall', mean_stall_s=equivalent_mean_stall_s, duty_cycle=p,
                       control_freq=100.0)
    rng = np.random.default_rng(11)
    honest = rng.uniform(-0.8, 0.8, size=(100000, ACTION_DIM))
    for t, h in enumerate(honest):
        stall.perturb_action(h.copy(), None, t)

    print(f'\n  nested check: duty={stall.empirical_duty_cycle:.4f} (target {p}), '
          f'stall_len={stall.empirical_mean_stall_steps:.2f} steps '
          f'(target {1/(1-p):.2f})')
    assert stall.empirical_duty_cycle == pytest.approx(p, rel=0.05)
    assert stall.empirical_mean_stall_steps == pytest.approx(1 / (1 - p), rel=0.10)


def test_stall_is_far_more_perturbing_than_iid_dropout_at_equal_duty():
    """
    The motivating result: at the SAME fraction of stalled steps, long stalls perturb the
    command stream far more than single-step stalls, because consecutive policy actions at
    100 Hz are nearly identical.

    Uses a realistic smooth action stream (a slow sinusoid, like a hovering controller)
    rather than white noise, since that is the regime the claim is about.
    """
    duty = 0.25
    t = np.arange(40000)
    honest = np.stack([0.4 * np.sin(2 * np.pi * t / 500.0 + k) for k in range(ACTION_DIM)], axis=1)

    def mean_deviation(agent):
        devs = []
        for i, h in enumerate(honest):
            out = np.asarray(agent.perturb_action(h.copy(), None, i))
            devs.append(np.linalg.norm(out - h))
        return float(np.mean(devs))

    iid = mean_deviation(make_agent('tier1_dropout', p_drop=duty, mode='hold'))
    burst = mean_deviation(make_agent('tier1_stall', mean_stall_s=0.5, duty_cycle=duty,
                                      control_freq=100.0))
    print(f'\n  equal duty {duty}: iid dropout dev={iid:.5f}  '
          f'burst stall (0.5 s) dev={burst:.5f}  ratio={burst / iid:.1f}x')
    assert burst > 10 * iid, f'burst stall ({burst}) should dominate iid dropout ({iid})'


def test_stall_state_resets():
    agent = make_agent('tier1_stall', mean_stall_s=1.0, duty_cycle=0.9)
    for t in range(2000):
        agent.perturb_action(np.full(ACTION_DIM, 0.5), None, t)
    assert agent.total_steps == 2000
    agent.reset()
    assert agent.stalled is False
    assert agent.total_steps == 0 and agent.stalled_steps == 0
    assert agent.stall_lengths == []
    assert np.array_equal(agent.prev_action, np.zeros(ACTION_DIM))


def test_stall_hold_freezes_the_last_good_command():
    """During a stall, `hold` must keep emitting the last FRESH command, not decay."""
    agent = make_agent('tier1_stall', mean_stall_s=10.0, duty_cycle=0.0)
    good = np.array([0.3, -0.4, 0.5, -0.6])
    agent.perturb_action(good, None, 0)                 # not stalled -> passthrough, buffered
    agent.stalled = True                                # force a stall
    for t in range(1, 20):
        out = agent.perturb_action(np.full(ACTION_DIM, -0.9), None, t)
        assert np.allclose(out, good), 'hold drifted away from the last good command'


def test_stall_determinism_and_range_and_no_team_state():
    cfg = dict(mean_stall_s=0.3, duty_cycle=0.25)

    def run(seed, neighbour_fill=0.0):
        agent = get_byzantine_class('tier1_stall')(
            agent_idx=0, action_dim=ACTION_DIM, seed=seed, **WIDE, **cfg)
        rng = np.random.default_rng(5)
        out = []
        for t in range(500):
            obs = np.concatenate([rng.normal(size=18), np.full(36, neighbour_fill)])
            out.append(agent.perturb_action(rng.uniform(-1, 1, ACTION_DIM), obs, t))
        return np.array(out)

    assert np.array_equal(run(4), run(4))
    assert not np.array_equal(run(4), run(5))
    assert np.array_equal(run(4), run(4, neighbour_fill=1e6)), 'reads neighbour state'
    assert np.array_equal(run(4), run(4, neighbour_fill=np.nan)), 'reads neighbour state'

    env = make_env()
    wrapped = ByzantineEnvWrapper(env, tier='tier1_stall', compromised_indices=[0],
                                 tier_config=dict(mean_stall_s=0.3, duty_cycle=0.5), seed=2)
    seed_all(wrapped)
    wrapped.reset()
    rng = np.random.default_rng(9)
    for _ in range(60):
        a = rng.uniform(-1, 1, size=(NUM_AGENTS, 4)).astype(np.float32)
        wrapped.step(a)
        ex = wrapped.last_executed_actions
        assert np.all(ex >= env.action_space.low) and np.all(ex <= env.action_space.high)
    wrapped.close()


def test_stall_rejects_bad_config():
    with pytest.raises(ValueError):
        make_agent('tier1_stall', mean_stall_s=0.0)
    with pytest.raises(ValueError):
        make_agent('tier1_stall', duty_cycle=1.0)
    with pytest.raises(ValueError):
        make_agent('tier1_stall', mode='maybe')
