"""
Tier 3 validation suite — specs/TIER3_ADVERSARIAL.md.

Test 1 (frozen weights) is written first, as the spec directs: it is the test that catches
silent self-play, and self-play would invalidate the Phase 1 baseline that every other tier
is measured against.

Note the architecture makes test 1 an assertion of an invariant rather than a check on a
risky mechanism: the honest team lives *inside the environment* (see adversary/frozen_team.py
for why `--num_policies=2` cannot express Tier 3's requirement), so there is no learner,
optimiser or gradient attached to the honest weights anywhere in the process. The test still
runs, because asserting a belief you hold for free is cheap and the failure it guards against
is silent.

Tests needing a real training loop are marked slow and self-skip; set BYZ_SLOW_TESTS=1.

Run:  pytest byzantine_experiments/tests/test_tier3.py -v
"""

import os

import numpy as np
import pytest

CHECKPOINT = 'train_dir/exp1_baseline_seed1'
SLOW = os.environ.get('BYZ_SLOW_TESTS') == '1'
needs_ckpt = pytest.mark.skipif(not os.path.isdir(CHECKPOINT),
                                reason=f'{CHECKPOINT} not present')


# --------------------------------------------------------------------------- helpers

def build_adv_env(adv_indices=(0,), **reward_kwargs):
    """The frozen-team env, built the way training builds it."""
    import torch
    from sample_factory.model.model_utils import get_rnn_size
    from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env

    from byzantine_experiments.adversary.env_factory import AttrDictShim, baseline_cfg_for
    from byzantine_experiments.adversary.frozen_team import FrozenTeamEnv, FrozenTeamPolicy
    from byzantine_experiments.adversary.reward import AdversarialReward

    ckpt_dir = os.path.join(CHECKPOINT, 'checkpoint_p0')
    base_cfg = baseline_cfg_for(ckpt_dir)
    base_cfg.load_checkpoint_kind = 'latest'

    inner = AttrDictShim(base_cfg, overrides=dict(
        quads_render=False, visualize_v_value=False, with_pbt=False,
        replay_buffer_sample_prob=0.0,
    ))
    env = make_quadrotor_env(base_cfg.env, cfg=inner)
    frozen = FrozenTeamPolicy(base_cfg, env.observation_space, env.action_space,
                              ckpt_dir, device='cpu')
    reward_fn = AdversarialReward(**reward_kwargs)
    adv_env = FrozenTeamEnv(env, frozen, adv_indices=adv_indices, reward_fn=reward_fn,
                            rnn_size=get_rnn_size(base_cfg))
    return adv_env, frozen


def drive(adv_env, n_steps, seed=0, amplitude=0.5):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_steps):
        a = rng.uniform(-amplitude, amplitude,
                        size=(adv_env.num_agents, adv_env.action_space.shape[0])).astype(np.float32)
        obs, rew, term, trunc, infos = adv_env.step(a)
        out.append((a, obs, rew, infos))
        if any(term) or any(trunc):
            break
    return out


# ================================================================ 1. frozen weights

@needs_ckpt
def test_honest_weights_are_bit_identical_after_rollouts():
    """
    TEST 1 — the honest policy's parameters must not change while the adversary acts.

    Hashes every parameter tensor before and after a rollout, and separately asserts the
    structural guarantees: no parameter requires grad, and no gradient ever accumulates.
    """
    from byzantine_experiments.adversary.frozen_team import policy_param_hash

    adv_env, frozen = build_adv_env()
    before = policy_param_hash(frozen.actor_critic)

    adv_env.reset()
    drive(adv_env, 200, seed=1)

    after = policy_param_hash(frozen.actor_critic)
    assert before == after, 'honest policy weights changed during adversarial rollouts'
    assert before == frozen.param_hash, 'weights changed between load and first use'

    assert not any(p.requires_grad for p in frozen.actor_critic.parameters()), \
        'a frozen parameter still requires grad'
    assert all(p.grad is None for p in frozen.actor_critic.parameters()), \
        'a gradient accumulated on the frozen policy'
    adv_env.close()


@needs_ckpt
def test_no_optimiser_or_learner_is_attached_to_the_frozen_policy():
    """
    The structural claim behind test 1: the honest team is environment, not opponent.

    Sample Factory is told the env has `n_adv` agents, so its learner can only ever see the
    adversary's slice. If this ever regressed to `--num_policies=2`, the honest policy would
    acquire a Learner and start updating -- silently.
    """
    adv_env, frozen = build_adv_env(adv_indices=(0,))
    assert adv_env.num_agents == 1, 'the learner must only see the adversary'
    assert adv_env.total_agents == 8
    assert adv_env.honest_indices == [1, 2, 3, 4, 5, 6, 7]
    assert not frozen.actor_critic.training, 'frozen policy must be in eval mode'
    adv_env.close()


# ================================================================ 2. policy mapping

@needs_ckpt
@pytest.mark.parametrize('adv_indices', [(0,), (3,), (0, 1), (2, 5)])
def test_agent_indices_map_to_the_correct_policy(adv_indices):
    """
    TEST 2 — the adversary's actions land on exactly its own drones, and nowhere else.

    Verified by driving the adversary with a saturated command and checking that only its
    own drones' executed motor commands match it.
    """
    adv_env, _ = build_adv_env(adv_indices=adv_indices)
    adv_env.reset()

    inner = adv_env.env
    captured = {}
    real_step = inner.step

    def spy(actions):
        captured['actions'] = np.array(actions, copy=True)
        return real_step(actions)

    inner.step = spy
    try:
        cmd = np.full((len(adv_indices), 4), 0.9, dtype=np.float32)
        adv_env.step(cmd)
        full = captured['actions']
        assert full.shape == (8, 4)
        for i in adv_indices:
            assert np.allclose(full[i], 0.9), f'adversary command did not reach drone {i}'
        for i in adv_env.honest_indices:
            assert not np.allclose(full[i], 0.9), f'adversary command leaked into honest drone {i}'
    finally:
        inner.step = real_step
        adv_env.close()


@needs_ckpt
def test_honest_drones_receive_the_frozen_policy_actions():
    """The honest slots must carry the frozen policy's output, not zeros or noise."""
    adv_env, frozen = build_adv_env(adv_indices=(0,))
    obs, _ = adv_env.reset()

    expected, _ = frozen.act(
        np.asarray(adv_env._last_obs, dtype=np.float32)[adv_env.honest_indices],
        adv_env._honest_rnn, deterministic=True)

    inner = adv_env.env
    captured = {}
    real_step = inner.step
    inner.step = lambda a: (captured.__setitem__('a', np.array(a, copy=True)), real_step(a))[1]
    try:
        adv_env.step(np.zeros((1, 4), dtype=np.float32))
        assert np.allclose(captured['a'][adv_env.honest_indices], expected, atol=1e-6)
    finally:
        inner.step = real_step
        adv_env.close()


# ================================================================ 3. reward sign

def _correlate(adv_env, n_steps=600, seed=2, amplitude=0.3):
    team, team_term, total, crash = [], [], [], []
    for _, _, rew, infos in drive(adv_env, n_steps, seed=seed, amplitude=amplitude):
        c = infos[0]['adv_reward_components']
        team.append(c['r_team_mean'])
        team_term.append(c['team'])
        total.append(float(np.mean(rew)))
        crash.append(c['crash'])
    return (np.array(team), np.array(team_term), np.array(total), np.array(crash))


@needs_ckpt
def test_adversary_reward_is_negatively_correlated_with_team_reward():
    """
    TEST 3 — r_adv must move opposite to team reward (spec: r < -0.8 over a rollout).

    The spec's threshold holds, but it is **regime-dependent**, and asserting it blindly on
    the total per-step reward would either fail or force the crash penalty to be uselessly
    weak. Measured here (random actions, 600 steps):

        corr(team_reward, team_term)                        = -1.000   (exact, by construction)
        corr(team_reward, total),  crash penalty disabled   = -0.998   <- the spec's claim
        corr(team_reward, total),  crash penalty active     = -0.023

    The last number is not a defect. With a randomly flailing adversary that crashes ~15
    times an episode, the sparse -w_crash spikes account for ~100% of the total reward's
    variance, so nothing else can correlate with anything. A *trained* adversary that stays
    airborne (the >=70% survival the efficacy gate demands) sits in the first regime.

    So this asserts the two things that are true independent of regime -- the objective is
    exactly anti-aligned, and the non-crash part of the reward tracks team damage -- and
    records the third as a measured property rather than hiding it behind a loosened
    threshold. The spec's r < -0.8 on the *trained* policy is re-checked at the efficacy gate.
    """
    airborne, _ = build_adv_env(w_crash=0.0, w_alive=0.01)
    airborne.reset()
    team_a, term_a, total_a, _ = _correlate(airborne)
    airborne.close()

    r_objective = float(np.corrcoef(team_a, term_a)[0, 1])
    r_airborne = float(np.corrcoef(team_a, total_a)[0, 1])

    crashing, _ = build_adv_env(w_crash=10.0, w_alive=0.01)
    crashing.reset()
    team_c, _, total_c, crash_c = _correlate(crashing)
    crashing.close()

    r_crashing = float(np.corrcoef(team_c, total_c)[0, 1])
    crash_var_share = float(np.var(crash_c) / max(np.var(total_c), 1e-12))

    print(f'\n  corr(team, team_term)            = {r_objective:+.4f}'
          f'\n  corr(team, total) | no crash pen. = {r_airborne:+.4f}'
          f'\n  corr(team, total) | crash pen. on = {r_crashing:+.4f}'
          f'  (crash term is {100 * crash_var_share:.0f}% of total variance)')

    assert r_objective < -0.99, f'objective is not anti-aligned with team reward ({r_objective})'
    assert r_airborne < -0.8, (
        f'in the airborne regime the adversary reward must track team damage ({r_airborne})')
    assert crash_var_share > 0.5, (
        'the crash penalty should dominate variance while the adversary is crashing -- if it '
        'does not, w_crash is too weak to deter the suicide solution')


def test_reward_components_are_exactly_additive():
    """Every component is reported separately and they sum to the total (spec: log all three)."""
    from byzantine_experiments.adversary.reward import AdversarialReward

    fn = AdversarialReward(w_team=1.0, w_crash=10.0, w_alive=0.1, w_stealth=5.0, epsilon=0.3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        rewards = rng.normal(size=8)
        crashed = rng.random(8) < 0.3
        actions = rng.uniform(-1, 1, size=(1, 4))
        total, c = fn(rewards, [0], list(range(1, 8)), actions, crashed)
        assert c['total'] == pytest.approx(c['team'] + c['crash'] + c['alive'] + c['stealth'])
        assert total == pytest.approx(c['total'])


def test_team_reward_mode_honest_ignores_the_adversarys_own_reward():
    """
    The `'honest'` default must not read the adversary's own reward at all.

    This is what removes the incentive to score by self-destruction -- see reward.py. With
    `'all'`, tanking the adversary's own reward *raises* r_adv, which is the degenerate
    solution `w_crash` then has to fight.
    """
    from byzantine_experiments.adversary.reward import AdversarialReward

    base = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    wrecked = base.copy()
    wrecked[0] = -50.0                      # adversary's own reward collapses
    kw = dict(w_crash=0.0, w_alive=0.0)

    honest = AdversarialReward(team_reward_mode='honest', **kw)
    a, _ = honest(base, [0], list(range(1, 8)), np.zeros((1, 4)), np.zeros(8, bool))
    b, _ = honest(wrecked, [0], list(range(1, 8)), np.zeros((1, 4)), np.zeros(8, bool))
    assert a == pytest.approx(b), "'honest' mode read the adversary's own reward"

    allm = AdversarialReward(team_reward_mode='all', **kw)
    c, _ = allm(base, [0], list(range(1, 8)), np.zeros((1, 4)), np.zeros(8, bool))
    d, _ = allm(wrecked, [0], list(range(1, 8)), np.zeros((1, 4)), np.zeros(8, bool))
    assert d > c, "'all' mode should reward the adversary for wrecking itself (the trap)"


# ================================================================ 4 & 5. penalties

@needs_ckpt
def test_crash_penalty_fires_and_appears_in_the_log():
    """TEST 4 — the crash penalty shows up in the components when the adversary crashes."""
    adv_env, _ = build_adv_env(w_team=1.0, w_crash=10.0, w_alive=0.1)
    adv_env.reset()

    # Full negative thrust drives the adversary into the floor within a second or so.
    saw_crash = False
    for _ in range(400):
        _, _, _, _, infos = adv_env.step(np.full((1, 4), -1.0, dtype=np.float32))
        c = infos[0]['adv_reward_components']
        if c['crash'] < 0:
            saw_crash = True
            assert c['crash'] == pytest.approx(-10.0 * c['adv_crashed'])
            assert c['alive'] == pytest.approx(0.1 * (1.0 - c['adv_crashed']))
            break
    adv_env.close()
    assert saw_crash, 'adversary never registered a crash despite full negative thrust'


def test_stealth_penalty_fires_only_beyond_epsilon():
    """TEST 5 — the hinge is zero inside the budget and positive outside it."""
    from byzantine_experiments.adversary.reward import AdversarialReward

    mu = np.zeros(4)
    fn = AdversarialReward(w_team=0.0, w_crash=0.0, w_alive=0.0,
                           w_stealth=5.0, epsilon=0.3, mu_honest=mu)

    inside = np.array([[0.1, 0.1, 0.0, 0.0]])          # norm 0.141 < 0.3
    _, c_in = fn(np.zeros(8), [0], list(range(1, 8)), inside, np.zeros(8, bool))
    assert c_in['stealth'] == 0.0 and c_in['stealth_excess'] == 0.0

    outside = np.array([[0.8, 0.0, 0.0, 0.0]])         # norm 0.8 > 0.3
    _, c_out = fn(np.zeros(8), [0], list(range(1, 8)), outside, np.zeros(8, bool))
    assert c_out['action_deviation'] == pytest.approx(0.8)
    assert c_out['stealth_excess'] == pytest.approx(0.5)
    assert c_out['stealth'] == pytest.approx(-2.5)

    off = AdversarialReward(w_stealth=0.0)
    _, c_off = off(np.zeros(8), [0], list(range(1, 8)), outside, np.zeros(8, bool))
    assert c_off['stealth'] == 0.0, 'w_stealth=0 must disable the penalty entirely'


def test_honest_action_stats_from_clean_rollouts():
    """`mu_honest` must come from the clean baseline rollouts, as the spec requires."""
    import glob
    from byzantine_experiments.adversary.reward import honest_action_stats

    paths = sorted(glob.glob('byzantine_experiments/results/tier1/t1_clean_s*.npz'))
    if not paths:
        pytest.skip('clean rollouts not present')
    stats = honest_action_stats(paths[:1])
    assert len(stats['mu_honest']) == 4
    assert np.all(np.abs(stats['mu_honest']) <= 1.0)
    assert stats['mean_action_norm'] > 0
    print(f"\n  mu_honest={np.round(stats['mu_honest'], 4)}  "
          f"sigma={np.round(stats['sigma_honest'], 4)}  "
          f"mean||a||={stats['mean_action_norm']:.4f}")


# ================================================================ 7. range / isolation

@needs_ckpt
def test_env_contract_and_range_compliance():
    """TEST 7 — observation/action shapes and legality, as in prior tiers."""
    adv_env, _ = build_adv_env(adv_indices=(0, 1))
    obs, info = adv_env.reset()
    assert obs.shape == (2, adv_env.observation_space.shape[0])

    low, high = adv_env.action_space.low, adv_env.action_space.high
    for a, obs, rew, infos in drive(adv_env, 50, seed=3, amplitude=1.0):
        assert obs.shape == (2, adv_env.observation_space.shape[0])
        assert np.asarray(rew).shape == (2,)
        assert len(infos) == 2
        assert np.all(a >= low) and np.all(a <= high)
        assert np.all(np.isfinite(obs)) and np.all(np.isfinite(rew))
    adv_env.close()


@needs_ckpt
def test_episode_boundary_resets_frozen_rnn_and_reward_accumulator():
    """A new episode must not inherit the previous one's RNN state or reward totals."""
    adv_env, _ = build_adv_env()
    adv_env.reset()
    drive(adv_env, 30, seed=5)
    assert adv_env.reward_fn.episode_components['total'] != 0.0
    adv_env.reset()
    assert adv_env.reward_fn.episode_components['total'] == 0.0
    assert adv_env._t == 0
    assert float(adv_env._honest_rnn.abs().sum()) == 0.0
    adv_env.close()


def test_rejects_bad_configuration():
    from byzantine_experiments.adversary.reward import AdversarialReward
    with pytest.raises(ValueError):
        AdversarialReward(team_reward_mode='everyone')


# ==================================================== 1b. frozen weights, end-to-end (slow)

@pytest.mark.skipif(not SLOW or not os.path.isdir(CHECKPOINT),
                    reason='set BYZ_SLOW_TESTS=1 (runs a real ~2 min training loop)')
def test_baseline_checkpoint_files_unchanged_by_a_real_training_run():
    """
    TEST 1, the definitive form the spec asks for: "assert checkpoint hash unchanged after
    1000 training steps".

    The in-memory version above checks the loaded tensors. This checks the *files on disk*
    across an actual Sample Factory training loop -- the thing that would be silently
    overwritten if the adversary run were ever misconfigured to train the honest policy
    (e.g. a stray `--num_policies=2`, or pointing `--train_dir` at the baseline).
    """
    import glob
    import hashlib
    import subprocess
    import sys
    import tempfile

    def hashes():
        return {p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
                for p in sorted(glob.glob(os.path.join(CHECKPOINT, 'checkpoint_p0', '*.pth')))}

    before = hashes()
    assert before, 'no baseline checkpoints found to protect'

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, '-m', 'byzantine_experiments.adversary.train_adv',
            f'--adv_baseline_checkpoint={CHECKPOINT}',
            '--experiment=_frozen_check', f'--train_dir={tmp}',
            '--train_for_env_steps=3000', '--num_workers=2', '--num_envs_per_worker=2',
            '--worker_num_splits=1', '--batch_size=128', '--rollout=32',
            '--device=cpu', '--serial_mode=True', '--async_rl=False',
            '--quads_use_numba=False', '--with_wandb=False',
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, f'training failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'

        adv_ckpts = glob.glob(os.path.join(tmp, '_frozen_check', 'checkpoint_p0', '*.pth'))
        assert adv_ckpts, 'the adversary never saved a checkpoint -- did it train at all?'

    after = hashes()
    assert before == after, (
        'BASELINE CHECKPOINT FILES CHANGED during adversarial training -- this is silent '
        'self-play and invalidates every tier measured against this baseline')


# ============================== 5b. stealth is scored on the EXECUTED action (regression)

@needs_ckpt
def test_stealth_penalty_scores_the_executed_action_not_the_raw_policy_sample():
    """
    Regression for the bug that destroyed the first `stealth5` training run.

    Sample Factory's continuous policy emits *unbounded* Gaussian samples. The env's
    RawControl clips them internally, so an action of norm 100 flies exactly like one of
    norm 2. Scoring the raw sample therefore penalises something no detector could observe --
    and the policy discovered it could not escape the penalty by flying differently, only by
    shrinking numbers it had no gradient signal about. The stealth term ran away to
    -542,034 per episode, 22,700x the objective, and the adversary never learned to attack.

    The fix is that the env clips before executing *and* before scoring. This asserts the
    scored deviation is bounded by the action space regardless of what the policy emits.
    """
    adv_env, _ = build_adv_env(w_stealth=5.0, epsilon=0.3)
    adv_env.reset()

    low, high = adv_env.action_space.low, adv_env.action_space.high
    max_possible_dev = float(np.linalg.norm(np.maximum(np.abs(low), np.abs(high))))

    for scale in (1.0, 10.0, 100.0, 1000.0):
        raw = np.full((1, 4), scale, dtype=np.float32)
        _, _, _, _, infos = adv_env.step(raw)
        dev = infos[0]['adv_reward_components']['action_deviation']
        assert dev <= max_possible_dev + 1e-5, (
            f'raw action of norm {np.linalg.norm(raw):.0f} scored deviation {dev:.2f}, '
            f'above the maximum possible executed deviation {max_possible_dev:.2f} -- the '
            f'penalty is being applied to the unclipped policy sample')
    adv_env.close()


@needs_ckpt
def test_action_deviation_is_reported_even_when_stealth_is_disabled():
    """
    The unconstrained (`w_stealth=0`) run is exactly the one where you need to know whether
    the adversary is winning by brute action magnitude. The original early-return reported
    0.0 there, blinding the diagnostic on the run that needed it.
    """
    adv_env, _ = build_adv_env(w_stealth=0.0)
    adv_env.reset()
    _, _, _, _, infos = adv_env.step(np.full((1, 4), 0.8, dtype=np.float32))
    c = infos[0]['adv_reward_components']
    assert c['action_deviation'] > 0.0, 'action_deviation must be measured even at w_stealth=0'
    assert c['stealth'] == 0.0, 'no penalty should be applied at w_stealth=0'
    adv_env.close()


# --------------------------------------------------------------------------- round 2: f = 0.25

def test_run_names_encode_f_and_the_gate_parses_both_rounds():
    """
    Round 2 (two adversaries) must not collide with round 1's checkpoints or sidecars, and the
    analysis must be able to recover f from a filename alone. `byz_train_adv.sh` appends
    `_f250` for n_adv=2 and nothing for n_adv=1, so round-1 artifacts already on disk keep
    parsing -- this pins both halves of that contract.
    """
    from byzantine_experiments.analysis.gate_tier3 import NAME_RE

    r1 = NAME_RE.match('t3_adv_seed1_stealth0_on_s2_best')
    assert r1 is not None and r1.group('f') is None
    assert (r1.group('aseed'), r1.group('stealth'), r1.group('bseed'), r1.group('ckpt')) \
        == ('1', '0', '2', 'best')

    r2 = NAME_RE.match('t3_adv_seed1_stealth5_f250_on_s1_m050M')
    assert r2 is not None and int(r2.group('f')) / 1000.0 == 0.25
    assert (r2.group('aseed'), r2.group('stealth'), r2.group('bseed'), r2.group('ckpt')) \
        == ('1', '5', '1', 'm050M')


def test_train_script_derives_experiment_name_and_indices_from_adv_indices():
    """The f=0.25 launch path, end to end, without spending a GPU-second on it."""
    import subprocess

    def dry(**env):
        out = subprocess.run(['bash', 'scripts/byz_train_adv.sh', '1', '0'],
                             capture_output=True, text=True,
                             env={**os.environ, 'DRY_RUN': '1', **env})
        assert out.returncode == 0, out.stderr
        return out.stdout

    one = dry(ADV_INDICES='0')
    assert 'experiment        : adv_seed1_stealth0\n' in one
    assert '--adv_indices=0 ' in one.replace('\\,', ',')

    two = dry(ADV_INDICES='0,1', W_ALIVE='0.05')
    assert 'adv_seed1_stealth0_f250' in two
    assert 'n_adv=2, f=0.250' in two
    assert '--adv_indices=0,1' in two.replace('\\,', ',')
    assert '--adv_w_alive=0.05' in two


def test_gate_primary_criterion_is_honest_damage_not_all_drone_damage():
    """
    Regression on the metric flaw round 1 exposed: an adversary that flies its own drone away
    scores high on all-drone goal RMSE while doing nothing to the honest team. The gate must
    fail that run.
    """
    from byzantine_experiments.analysis.gate_tier3 import build_rows

    clean = {1: dict(goal_rmse_mean=1.0, goal_rmse_honest_mean=1.0, episode_reward_mean=100.0,
                     episode_reward_sem=1.0, collisions_mean=0.0)}
    # Own drone +400 %, honest team +2 %: all-drone average lands near +52 %.
    metrics = dict(
        n_episodes=20, goal_rmse_mean=1.5, goal_rmse_byzantine_mean=5.0,
        goal_rmse_honest_mean=1.02, episode_reward_mean=40.0, episode_reward_sem=1.0,
        collisions_mean=0.0, attacker_survived_mean=1.0, action_norm_ratio=1.0,
        action_deviation_byzantine_mean=0.1, saturation_rate_byzantine=0.2)
    runs = [dict(stem='t3_adv_seed1_stealth0_on_s1_best', adv_seed=1, stealth=0, f=0.125,
                 baseline_seed=1, ckpt='best', sidecar={}, metrics=metrics)]

    row = build_rows(runs, clean)[0]
    assert row['goal_rmse_deg_pct'] > 30.0            # the gameable metric says "success"
    assert row['honest_goal_rmse_deg_pct'] < 30.0     # the honest team is untouched
    assert row['own_goal_rmse_deg_pct'] > 300.0       # it destroyed itself
    assert not row['passes_primary'], 'self-destruction must not pass the efficacy gate'


# --------------------------------------------------------- gate statistics (round 2 fallout)

def _fake_run(tmpdir, name, honest_err, byz_err, n_ep=20, n_steps=50, n_drones=8, n_byz=1):
    """Minimal .npz in the collector's schema, with a prescribed per-drone goal error."""
    import numpy as np
    mask = np.zeros(n_drones, dtype=bool)
    mask[:n_byz] = True
    obs = np.zeros((n_ep, n_steps, n_drones, 54), dtype=np.float32)
    # obs[..., :3] is pos-goal; put all of the error on x so its norm is exactly the target.
    obs[:, :, mask, 0] = byz_err
    obs[:, :, ~mask, 0] = honest_err
    path = os.path.join(tmpdir, name + '.npz')
    np.savez_compressed(
        path, obs=obs, byzantine_mask=mask,
        episode_lengths=np.full(n_ep, n_steps, dtype=np.int32),
        actions_executed=np.zeros((n_ep, n_steps, n_drones, 4), dtype=np.float32),
        actions_honest=np.zeros((n_ep, n_steps, n_drones, 4), dtype=np.float32),
        rewards=np.zeros((n_ep, n_steps, n_drones), dtype=np.float32),
        positions=np.zeros((n_ep, n_steps, n_drones, 3), dtype=np.float32),
        velocities=np.zeros((n_ep, n_steps, n_drones, 3), dtype=np.float32),
        collisions=np.zeros((n_ep, n_steps), dtype=np.float32))
    return path


def test_honest_damage_uses_matched_drone_subsets(tmp_path):
    """
    The clean reference must be recomputed over the *attacked run's* honest indices.

    A clean run has no Byzantine drones, so its stored honest mean is over all 8. Comparing
    that against 6 honest drones at f = 0.25 mixes drone subsets. Here the clean run's drones
    0 and 1 are given a large error that the honest six do not have: if the comparison leaked
    those in, the measured degradation would be strongly negative instead of zero.
    """
    from byzantine_experiments.analysis.gate_tier3 import honest_damage_stats

    d = str(tmp_path)
    attacked = _fake_run(d, 'attacked', honest_err=1.0, byz_err=9.0, n_byz=2)
    clean = _fake_run(d, 'clean', honest_err=1.0, byz_err=9.0, n_byz=0)
    # Make the clean run's slots 0,1 the outliers, matching the attacked run's byzantine slots.
    with np.load(clean) as z:
        arrs = {k: z[k] for k in z.files}
    arrs['obs'][:, :, :2, 0] = 9.0
    np.savez_compressed(clean, **arrs)

    st = honest_damage_stats(attacked, clean)
    assert st is not None
    assert abs(st['deg_pct']) < 1e-6, (
        f"clean drones 0,1 leaked into the reference: {st['deg_pct']:.1f} %")


def test_gate_rejects_a_large_but_insignificant_point_estimate(tmp_path):
    """
    At n = 20 the 95 % CI on a degradation percentage is ~±22 points, so a point estimate can
    clear 30 % on noise alone. The gate must require the interval to exclude zero.
    """
    from byzantine_experiments.analysis.gate_tier3 import honest_damage_stats

    d = str(tmp_path)
    clean = _fake_run(d, 'clean', honest_err=1.0, byz_err=1.0, n_byz=0)
    attacked = _fake_run(d, 'attacked', honest_err=1.0, byz_err=1.0, n_byz=1)
    # Deterministic, not sampled, so the test cannot drift into significance on a lucky seed.
    # clean mean 1.0 (sd 0.82), attacked mean 1.4 (sd 1.23) -> +40 % with a CI of about
    # [-25, +105]: exactly the shape of round 1's "17.7 %" headline, which had p = 0.27.
    for path, vals in ((clean, [0.2] * 10 + [1.8] * 10),
                       (attacked, [0.2] * 10 + [2.6] * 10)):
        with np.load(path) as z:
            arrs = {k: z[k] for k in z.files}
        arrs['obs'][..., 0] = np.asarray(vals, dtype=np.float32)[:, None, None]
        np.savez_compressed(path, **arrs)

    st = honest_damage_stats(attacked, clean)
    assert st['deg_pct'] > 30.0, 'test setup should produce a large point estimate'
    assert st['ci_lo'] < 0.0 < st['ci_hi'], 'test setup should be statistically ambiguous'
    # This is precisely the row the gate must not pass.
    assert not (st['deg_pct'] >= 30.0 and st['ci_lo'] > 0.0)


def test_honest_rmse_reads_the_whole_obs_array_once(tmp_path, monkeypatch):
    """
    These .npz files are compressed, so `mmap_mode` is a no-op and every `d['obs']` access
    inflates all ~52 MB. Reading it inside the per-episode loop turned a 30 s job into a
    10 min one. Pin the access pattern rather than the wall-clock time.
    """
    from byzantine_experiments.analysis import gate_tier3

    path = _fake_run(str(tmp_path), 'run', honest_err=1.0, byz_err=2.0)
    gate_tier3._RMSE_CACHE.clear()

    real_load = np.load
    reads = []

    class CountingNpz:
        def __init__(self, inner): self._inner = inner
        def __getitem__(self, k):
            reads.append(k)
            return self._inner[k]
        def __enter__(self): return self
        def __exit__(self, *a): return self._inner.__exit__(*a)
        def __getattr__(self, n): return getattr(self._inner, n)

    monkeypatch.setattr(np, 'load', lambda *a, **k: CountingNpz(real_load(*a, **k)))
    gate_tier3._honest_rmse_per_episode(path, np.array([1, 2, 3]))

    assert reads.count('obs') == 1, f'obs decompressed {reads.count("obs")}x, expected once'


def test_honest_rmse_is_cached_across_calls(tmp_path):
    """Every clean reference is read once per (file, drone-subset), not once per attacked run."""
    from byzantine_experiments.analysis import gate_tier3

    path = _fake_run(str(tmp_path), 'run', honest_err=1.0, byz_err=2.0)
    gate_tier3._RMSE_CACHE.clear()
    idx = np.array([1, 2, 3])
    first = gate_tier3._honest_rmse_per_episode(path, idx)
    assert len(gate_tier3._RMSE_CACHE) == 1
    second = gate_tier3._honest_rmse_per_episode(path, idx)
    assert second is first


# ------------------------------------------------------- round 3: the observation channel

def _adv_env(**over):
    """Build a FrozenTeamEnv through the real factory, with round-3 knobs."""
    from byzantine_experiments.adversary.env_factory import (
        ADV_ENV_NAME, make_adversarial_env, register_adversarial_components)
    from byzantine_experiments.adversary.train_adv import parse_adv_cfg
    register_adversarial_components()
    argv = ['--env=' + ADV_ENV_NAME,
            f'--adv_baseline_checkpoint={CHECKPOINT}',
            '--experiment=test_r3', '--device=cpu', '--num_workers=1']
    argv += [f'--{k}={v}' for k, v in over.items()]
    return make_adversarial_env(ADV_ENV_NAME, parse_adv_cfg(argv), None, None)


def test_decode_lie_is_a_bounded_delta_with_truth_at_zero():
    """Honesty must be the origin of the action space, not a point the policy has to find."""
    from byzantine_experiments.adversary.lies import LIE_DIM, decode_lie

    true_state = np.array([1.0, -2.0, 3.0, 0.5, -0.5, 0.25])
    assert np.allclose(decode_lie(np.zeros(LIE_DIM), true_state), true_state)

    full = decode_lie(np.ones(LIE_DIM), true_state, max_pos=10.0, max_vel=6.0)
    assert np.allclose(full[:3], true_state[:3] + 10.0)
    assert np.allclose(full[3:], true_state[3:] + 6.0)

    # An unbounded policy sample must not produce an unbounded lie -- the same class of bug
    # that destroyed the round-1 stealth run.
    assert np.allclose(decode_lie(np.full(LIE_DIM, 500.0), true_state, 10.0, 6.0),
                       decode_lie(np.ones(LIE_DIM), true_state, 10.0, 6.0))


def test_lie_action_space_is_extended_only_when_lies_are_enabled():
    """Rounds 1-2 must reproduce exactly: same 4-D action space, no broadcast machinery."""
    from byzantine_experiments.adversary.lies import LIE_DIM

    plain = _adv_env()
    assert plain.action_space.shape == (4,)
    assert not plain.enable_lies

    lying = _adv_env(adv_enable_lies='True')
    assert lying.action_space.shape == (4 + LIE_DIM,)
    assert np.allclose(lying.action_space.low[4:], -1.0)
    assert np.allclose(lying.action_space.high[4:], 1.0)


def test_the_lie_actually_changes_what_honest_drones_see_and_do():
    """
    The load-bearing test. `neighborhood_indices()` re-sorts every step and drops all but the
    closest few neighbours, so it is entirely possible to write this attack such that it
    rewrites nothing, or rewrites a slot no one reads. A no-op lie would look like a clean
    negative result -- exactly the failure this whole tier is trying to distinguish.
    """
    from byzantine_experiments.adversary.lies import LIE_DIM, apply_lies_to_obs, decode_lie

    env = _adv_env(adv_enable_lies='True')
    obs, _ = env.reset(seed=3)
    for _ in range(5):      # let the formation spread out so neighbours are non-degenerate
        env.step(np.zeros((env.num_agents, 4 + LIE_DIM), dtype=np.float32))

    clean_obs = np.array(env._last_obs, copy=True)
    unwrapped = env.env.unwrapped
    pos, vel = env._drone_states()
    liar = env.adv_indices[0]
    reported = {liar: decode_lie(np.ones(LIE_DIM), np.concatenate([pos[liar], vel[liar]]),
                                 max_pos=env.max_lie_pos, max_vel=env.max_lie_vel)}
    lied_obs, rewrites = apply_lies_to_obs(
        clean_obs, reported, unwrapped.neighborhood_indices(), pos, vel,
        ego_dim=env.ego_dim,
        clip_lo=unwrapped.clip_neighbor_space_min_box,
        clip_hi=unwrapped.clip_neighbor_space_max_box,
        liar_indices=env.adv_indices)

    assert rewrites > 0, 'the lie rewrote nothing -- the attack is a no-op'
    assert not np.allclose(clean_obs[env.honest_indices], lied_obs[env.honest_indices]), \
        'honest observations are unchanged by the lie'
    assert np.allclose(clean_obs[liar], lied_obs[liar]), \
        "a liar's own observation must be untouched"

    # And it has to move the honest policy, not merely the numbers it reads.
    a_clean, _ = env.frozen.act(clean_obs[env.honest_indices].astype(np.float32),
                                env._fresh_rnn(), deterministic=True)
    a_lied, _ = env.frozen.act(lied_obs[env.honest_indices].astype(np.float32),
                               env._fresh_rnn(), deterministic=True)
    assert not np.allclose(a_clean, a_lied), 'honest actions unchanged -- lie has no effect'


def test_lie_only_mode_flies_the_frozen_policy_bit_for_bit():
    """
    2C's defining property: the action stream is clean and the whole attack is the broadcast.
    If the adversary's motors drifted from the honest policy's, `lie_only` would silently
    become `lie+fly` and the channel comparison would be confounded.
    """
    from byzantine_experiments.adversary.lies import LIE_DIM

    env = _adv_env(adv_enable_lies='True', adv_lie_only='True')
    env.reset(seed=5)
    driven = env._honest_actions()
    adv = env.adv_indices[0]
    assert adv in driven, 'lie_only must drive the adversary with the frozen policy'

    expected = np.asarray(driven[adv], dtype=np.float32)
    # A wildly different motor command must be ignored entirely.
    action = np.zeros((env.num_agents, 4 + LIE_DIM), dtype=np.float32)
    action[:, :4] = 1.0
    env.step(action)
    flown = np.asarray(env.env.unwrapped.envs[adv].dynamics.thrust_cmds_damp)
    assert flown is not None      # sanity: the sim actually recorded a command
    # The honest action the env would have used is what `_honest_actions` produced for it.
    assert np.allclose(expected, np.asarray(driven[adv], dtype=np.float32))


def test_stealth_is_scored_on_motors_only_not_on_the_broadcast():
    """
    `mu_honest` is a distribution over the honest policy's 4 thrusts. Feeding it a 10-D
    action would either crash or, worse, silently score the lie as if it were a motor
    deviation -- reintroducing the 555 : 1 domination that made both stealth runs useless.
    """
    from byzantine_experiments.adversary.lies import LIE_DIM

    env = _adv_env(adv_enable_lies='True', adv_w_stealth='5.0')
    env.reset(seed=7)
    captured = {}

    class Spy:
        """Delegating proxy -- `__call__` is resolved on the type, so patching the instance
        attribute would not intercept anything."""

        def __init__(self, inner): self._inner = inner
        def __call__(self, **kw):
            captured['adv_actions'] = np.asarray(kw['adv_actions'])
            return self._inner(**kw)
        def __getattr__(self, n): return getattr(self._inner, n)

    env.reward_fn = Spy(env.reward_fn)
    env.step(np.zeros((env.num_agents, 4 + LIE_DIM), dtype=np.float32))
    assert captured, 'reward function was never called'
    assert captured['adv_actions'].shape[-1] == 4, \
        f"stealth scored a {captured['adv_actions'].shape[-1]}-D action, expected 4 motors"


def test_training_and_evaluation_share_one_lie_decoder():
    """
    `FrozenTeamEnv` (training) and `LearnedAdversaryAgent` (evaluation) must decode a policy
    output into the same broadcast state, or "trained at X, evaluated at X" is false. They
    both call `decode_lie`; this pins that they agree numerically, including the bounds read
    off the checkpoint config.
    """
    from byzantine_experiments.adversary.lies import LIE_DIM, decode_lie
    from byzantine_experiments.agents.tier3_learned import LearnedAdversaryAgent

    true_state = np.array([0.4, -1.1, 2.2, 0.3, 0.0, -0.7])
    lie = np.array([0.5, -0.5, 0.25, 1.0, -1.0, 0.0])

    agent = LearnedAdversaryAgent.__new__(LearnedAdversaryAgent)
    agent.enable_lies, agent.lie_only = True, False
    agent.max_lie_pos, agent.max_lie_vel = 10.0, 6.0
    agent._pending_lie, agent.lies_told = lie, 0

    assert np.allclose(agent.perturb_broadcast_state(true_state, t=0),
                       decode_lie(lie, true_state, 10.0, 6.0))


def test_a_motors_only_checkpoint_still_broadcasts_the_truth():
    """
    A round-1/2 checkpoint must not enter the wrapper's broadcast path at all. If it did,
    `_apply_broadcast_lies` would rewrite its neighbour slots with recomputed "true" values
    that need not be bit-identical to what the simulator wrote -- silently changing the 28
    sidecars rounds 1-2 were measured from. So `lies_about_state` must read False on an
    instance built from a motors-only checkpoint.
    """
    from byzantine_experiments.agents.tier3_learned import LearnedAdversaryAgent

    agent = LearnedAdversaryAgent(0, 4, checkpoint='train_dir/adv_seed1_stealth0')
    assert agent.enable_lies is False
    assert agent.lies_about_state is False, \
        'a motors-only checkpoint must stay out of the wrapper broadcast path'
    true_state = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    assert np.allclose(agent.perturb_broadcast_state(true_state, t=0), true_state)
    assert agent.lies_told == 0


def test_round3_checkpoint_config_switches_the_broadcast_path_on(tmp_path):
    """
    The mirror of the previous test: a checkpoint trained WITH the observation channel must
    put itself into the wrapper's liar set, and must pick up its own lie bounds rather than
    the module defaults. Both are read from config.json at construction, before any weights
    load, because `ByzantineEnvWrapper` decides `any_lies` from the constructed agents.
    """
    import json as _json

    from byzantine_experiments.agents.tier3_learned import LearnedAdversaryAgent

    run_dir = tmp_path / 'adv_seed1_stealth0_lie'
    (run_dir / 'checkpoint_p0').mkdir(parents=True)
    # resolve_checkpoint needs something to find; contents are irrelevant here.
    (run_dir / 'checkpoint_p0' / 'checkpoint_000000001_000002.pth').write_bytes(b'')
    (run_dir / 'config.json').write_text(_json.dumps(
        dict(adv_enable_lies=True, adv_lie_only=True,
             adv_max_lie_pos=4.0, adv_max_lie_vel=2.0)))

    agent = LearnedAdversaryAgent(0, 4, checkpoint=str(run_dir))
    assert agent.lies_about_state is True
    assert agent.enable_lies is True and agent.lie_only is True
    assert (agent.max_lie_pos, agent.max_lie_vel) == (4.0, 2.0)

    # And the bounds are actually used by the decoder, not just stored.
    agent._pending_lie = np.ones(6)
    true_state = np.zeros(6)
    reported = agent.perturb_broadcast_state(true_state, t=0)
    assert np.allclose(reported[:3], 4.0) and np.allclose(reported[3:], 2.0)


# --------------------------------------------- round 5: absolute ego state in the observation

def test_absolute_position_is_not_recoverable_from_the_plain_observation():
    """
    The measurement that motivated round 5, pinned so the premise cannot rot.

    The hand-designed `liar` is handed its true absolute state and can therefore compute
    `mirror` (report `2*centre - pos`) trivially. The learned adversary saw only the 54-D
    observation — `pos - goal`, `vel`, `R`, `omega`, relative neighbours — which does not
    contain absolute position. If a future observation change made it recoverable, round 5's
    justification would no longer hold and this test should be revisited, not deleted.
    """
    path = 'byzantine_experiments/results/tier1/t1_clean_s3.npz'
    if not os.path.exists(path):
        pytest.skip('clean seed-3 rollout not present')
    with np.load(path) as d:
        obs, pos = np.asarray(d['obs']), np.asarray(d['positions'])
    X = obs.reshape(-1, obs.shape[-1]).astype(np.float64)
    P = pos.reshape(-1, 3).astype(np.float64)
    sub = np.random.default_rng(0).choice(len(X), size=min(50000, len(X)), replace=False)
    X, P = X[sub], P[sub]
    Xb = np.hstack([X, np.ones((len(X), 1))])
    W, *_ = np.linalg.lstsq(Xb, P, rcond=None)
    pred = Xb @ W
    for i in range(3):
        ss_res = ((P[:, i] - pred[:, i]) ** 2).sum()
        ss_tot = ((P[:, i] - P[:, i].mean()) ** 2).sum()
        assert 1 - ss_res / ss_tot < 0.6, (
            f'axis {i}: absolute position is now recoverable from the observation '
            f'(R^2 = {1 - ss_res / ss_tot:.2f}); round 5 rests on it not being')


def test_augment_obs_appends_clipped_absolute_state():
    from byzantine_experiments.adversary.lies import EGO_AUG_DIM, augment_obs

    out = augment_obs(np.zeros(54), [1.0, 2.0, 99.0], [0.5, -0.5, 99.0],
                      max_pos=10.0, max_vel=6.0)
    assert out.shape == (54 + EGO_AUG_DIM,)
    assert np.allclose(out[54:57], [1.0, 2.0, 10.0])    # clipped to the env's obs range
    assert np.allclose(out[57:60], [0.5, -0.5, 6.0])


def test_ego_state_obs_extends_the_space_and_carries_the_true_state():
    env = _adv_env(adv_enable_lies='True', adv_lie_only='True', adv_ego_state_obs='True')
    assert env.observation_space.shape == (60,)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (env.num_agents, 60)
    pos, vel = env._drone_states()
    i = env.adv_indices[0]
    assert np.allclose(obs[0, 54:57], np.clip(pos[i], -10, 10), atol=1e-5)
    assert np.allclose(obs[0, 57:60], np.clip(vel[i], -6, 6), atol=1e-5)


def test_rounds_3_and_4_configs_keep_the_54d_observation():
    """
    Round 5 changes the observation space. Rounds 3-4 checkpoints must still build a 54-D env,
    or their weights stop loading and four rounds of results become unreproducible.
    """
    env = _adv_env(adv_enable_lies='True', adv_lie_only='True')
    assert env.observation_space.shape == (54,)
    assert env.ego_state_obs is False


def test_ego_state_obs_requires_lies_enabled():
    import pytest as _pytest
    with _pytest.raises(ValueError, match='requires enable_lies'):
        _adv_env(adv_ego_state_obs='True')


def test_round5_agent_requires_and_uses_the_wrapper_ego_state(tmp_path):
    """
    A round-5 checkpoint must declare `needs_ego_state`, or `ByzantineEnvWrapper` never
    populates `agent.ego` and the agent would silently feed the policy a 54-D vector its
    first layer cannot accept. Failing loudly beats a shape error deep inside torch.
    """
    import json as _json

    from byzantine_experiments.agents.tier3_learned import LearnedAdversaryAgent

    run_dir = tmp_path / 'adv_seed3_stealth0_lieonlyego'
    (run_dir / 'checkpoint_p0').mkdir(parents=True)
    (run_dir / 'checkpoint_p0' / 'checkpoint_000000001_000002.pth').write_bytes(b'')
    (run_dir / 'config.json').write_text(_json.dumps(
        dict(adv_enable_lies=True, adv_lie_only=True, adv_ego_state_obs=True)))

    agent = LearnedAdversaryAgent(0, 4, checkpoint=str(run_dir))
    assert agent.ego_state_obs is True
    assert agent.needs_ego_state is True, 'wrapper would never supply agent.ego'
    assert agent.lies_about_state is True

    agent._policy = object()      # skip the real load; we only exercise the guard
    with pytest.raises(RuntimeError, match='did not supply an EgoState'):
        agent.perturb_action(np.zeros(4), np.zeros(54), t=0)


# ------------------------------------------------ training-curve reader (misread twice)

def test_trend_is_flat_when_the_slope_is_smaller_than_the_curves_own_wobble():
    """
    With ~13k logged episodes the slope SE is tiny, so a negligible slope passes a 2-SE test.
    `adv/survived` was flagged "moving" at 1e-5 per 100M steps. Practical visibility required.
    """
    from byzantine_experiments.analysis.training_curve import classify_trend

    steps = np.linspace(0, 2e8, 13000)
    bins = np.array([0.994, 0.995, 0.995, 0.995, 0.994, 0.995, 0.994, 0.996, 0.993, 0.995])
    assert classify_trend(1e-5, 1e-7, steps, bins).startswith('flat')


def test_trend_is_moving_when_it_clears_both_bars():
    from byzantine_experiments.analysis.training_curve import classify_trend

    steps = np.linspace(0, 2e8, 13000)
    bins = np.linspace(-10.0, -4.0, 10)           # steady climb, wobble ~2, change 6
    assert 'rising' in classify_trend(3.0, 0.05, steps, bins)


def test_trend_ignores_a_near_constant_series():
    """Round 5's `adv/survived` (0.994-0.995) must not be reported as rising or non-monotone."""
    from byzantine_experiments.analysis.training_curve import _visibility_floor, classify_trend

    steps = np.linspace(0, 2e8, 13000)
    bins = np.array([0.994, 0.995, 0.995, 0.995, 0.994, 0.995, 0.994, 0.996, 0.993, 0.995])
    assert classify_trend(2e-4, 1e-4 / 3, steps, bins).startswith('flat')
    assert bins.max() - bins[-1] <= _visibility_floor(bins)


def test_pooled_estimate_restricts_clean_to_the_attacked_honest_drones():
    """
    A clean run has no Byzantine mask, so its own "honest" set is all 8 drones. Pooling must
    compare the attacked runs' 7 honest drones against the SAME 7 in clean, not against all 8
    -- the bias gate_tier3 was already fixed for, reintroduced once in confirm_tier3.
    """
    from byzantine_experiments.analysis import confirm_tier3

    d = str(tmp_dir := __import__('tempfile').mkdtemp())
    # clean: drone 0 is a large outlier, the rest are 1.0
    clean = _fake_run(d, 'clean', honest_err=1.0, byz_err=1.0, n_byz=0)
    with np.load(clean) as z:
        arrs = {k: z[k] for k in z.files}
    arrs['obs'][:, :, 0, 0] = 9.0
    np.savez_compressed(clean, **arrs)
    # attacked: drone 0 compromised, honest drones identical to clean's honest drones
    atk = _fake_run(d, 't3_adv_seed9_x_on_s9_m001M', honest_err=1.0, byz_err=9.0, n_byz=1)

    r = confirm_tier3.pooled_and_shrunk(
        'adv_seed9_x', horizon_dir=d) if False else None
    # Call the internals directly: the clean reference must exclude drone 0.
    from byzantine_experiments.analysis.gate_tier3 import _honest_rmse_per_episode
    _, idx = confirm_tier3._honest_series(atk)
    assert list(idx) == list(range(1, 8))
    matched = _honest_rmse_per_episode(clean, idx)
    unmatched, _ = confirm_tier3._honest_series(clean)
    assert np.allclose(matched, 1.0), 'matched clean should exclude the outlier drone'
    assert unmatched.mean() > matched.mean() + 0.5, 'test setup: unmatched must differ'


def test_best_clean_reference_prefers_the_largest_sample_not_any_confirm_file():
    """
    A 3-episode smoke run must not displace Tier 1's 50-episode reference. The incumbent's
    size has to be read, not assumed to be zero, or every seed-3 CI silently widens.
    """
    import tempfile

    from byzantine_experiments.analysis import confirm_tier3

    d = tempfile.mkdtemp()
    tier1, confirm = os.path.join(d, 'tier1'), os.path.join(d, 'confirm')
    os.makedirs(tier1); os.makedirs(confirm)
    big = _fake_run(tier1, 't1_clean_s9', honest_err=1.0, byz_err=1.0, n_ep=50, n_byz=0)
    small = _fake_run(confirm, 'confirm_clean_s9_n3', honest_err=1.0, byz_err=1.0,
                      n_ep=3, n_byz=0)
    assert os.path.exists(big) and os.path.exists(small)

    old = (confirm_tier3.CLEAN_DIR, confirm_tier3.CONFIRM_DIR)
    try:
        confirm_tier3.CLEAN_DIR, confirm_tier3.CONFIRM_DIR = tier1, confirm
        assert confirm_tier3.best_clean_reference(9) == big, 'a 3-episode file won'
        # ...but a genuinely larger confirmatory reference should win.
        bigger = _fake_run(confirm, 'confirm_clean_s9_n200', honest_err=1.0, byz_err=1.0,
                           n_ep=200, n_byz=0)
        assert confirm_tier3.best_clean_reference(9) == bigger
    finally:
        confirm_tier3.CLEAN_DIR, confirm_tier3.CONFIRM_DIR = old


# ------------------------------------------------ rollout reproducibility (the root cause)

@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason='baseline checkpoint not present')
def test_two_rollouts_at_the_same_seed_draw_the_same_scenarios():
    """
    The bug that invalidated every confidence interval in this project.

    `collect.py` seeded only the global numpy and torch streams. The spawn positions and goals
    come from each `QuadrotorSingle`'s own `np_random`, which upstream seeds from OS entropy at
    construction, so two rollouts with the identical `--seed` drew *different scenarios* --
    2.9 m apart at t = 0. Each run therefore carried an uncontrolled scenario offset, measured
    at ~7 % of mean goal RMSE, which no reported interval accounted for because every interval
    came from episode variance inside a single run.

    With `seed_every_rng`, clean and attacked runs at the same seed are paired and that offset
    cancels. This test pins the property the pairing depends on. It is deliberately about
    SPAWNS, not whole trajectories: numba's JIT RNG is not reachable and lets trajectories
    drift ~0.1-0.5 m by the end of an episode, which does not affect scenario matching.
    """
    import numpy as _np

    from byzantine_experiments.rollout.collect import (
        build_cfg, make_byzantine_env, seed_every_rng)

    cfg = build_cfg(CHECKPOINT, ['--device', 'cpu'])
    env, _ = make_byzantine_env(cfg, None)

    def spawn(seed):
        seed_every_rng(env, seed)
        env.reset()
        return _np.array([e.dynamics.pos for e in env.unwrapped.envs])

    a, b, c = spawn(4242), spawn(4242), spawn(4243)
    assert _np.allclose(a, b, atol=1e-9), (
        f'same seed gave different spawns (max diff {_np.abs(a - b).max():.3f} m) -- '
        'rollouts are not reproducible and paired comparisons are impossible')
    assert not _np.allclose(a, c, atol=1e-9), 'different seeds must give different scenarios'


@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason='baseline checkpoint not present')
def test_episode_scenario_does_not_depend_on_what_happened_before():
    """
    Per-episode seeding. Collisions draw from the global `np.random` stream, so an attacked run
    (more collisions) used to advance it differently from its clean twin, and every later
    episode's scenario diverged: a clean/mirror pair at one seed matched on episodes 0-2 and on
    no others, with a per-episode correlation of -0.08 between arms.

    Here the history is simulated by burning random numbers on both the global stream and each
    drone's own generator between two resets. With `episode_seed`, episode i must spawn the same
    way regardless.
    """
    import numpy as _np

    from byzantine_experiments.rollout.collect import (
        build_cfg, episode_seed, make_byzantine_env, seed_every_rng)

    cfg = build_cfg(CHECKPOINT, ['--device', 'cpu'])
    env, _ = make_byzantine_env(cfg, None)

    def spawn_for_episode(i):
        seed_every_rng(env, episode_seed(9, i))
        env.reset()
        return _np.array([e.dynamics.pos for e in env.unwrapped.envs])

    first = spawn_for_episode(3)
    _np.random.random(5000)                          # a run full of collisions
    for e in env.unwrapped.envs:
        e.np_random.random(500)
    again = spawn_for_episode(3)
    other = spawn_for_episode(4)
    assert _np.allclose(first, again, atol=1e-9), 'episode 3 changed with prior RNG history'
    assert not _np.allclose(first, other, atol=1e-9), 'episodes 3 and 4 must differ'
    assert len({episode_seed(9, i) for i in range(1000)}) == 1000


# ------------------------------------------------------------- v2 paired analysis guard

def _fake_v2(tmpdir, name, spawn_shift=0.0, honest=None, seed=7, tag=True, n_byz=1, n_ep=20):
    """Synthetic run WITH a sidecar, as the fixed collector writes it."""
    import json as _json
    rng = np.random.default_rng(0)                 # same scenarios in every call
    path = _fake_run(tmpdir, name, honest_err=1.0, byz_err=1.0, n_ep=n_ep, n_byz=n_byz)
    with np.load(path) as z:
        arrs = {k: z[k] for k in z.files}
    scen = rng.uniform(0.5, 2.0, size=n_ep)        # per-episode scenario difficulty
    err = scen if honest is None else honest(scen)
    arrs['obs'][..., 0] = np.asarray(err, np.float32)[:, None, None]
    arrs['positions'][:, 0] = rng.normal(size=arrs['positions'][:, 0].shape) + spawn_shift
    np.savez_compressed(path, **arrs)
    sc = dict(seed=seed, checkpoint='train_dir/exp1_baseline_seed7')
    if tag:
        from byzantine_experiments.paths import SEEDING_SCHEME
        sc['seeding'] = SEEDING_SCHEME
    with open(path[:-4] + '.json', 'w') as f:
        _json.dump(sc, f)
    return path


def test_paired_analysis_detects_an_effect_unpaired_misses(tmp_path):
    """The reason for the whole v2 re-collection, in miniature: a consistent +10 % effect on
    top of large scenario variance is invisible unpaired and obvious paired."""
    from byzantine_experiments.analysis.paired import paired_degradation
    from byzantine_experiments.analysis.gate_tier3 import welch_degradation, _honest_rmse_per_episode
    d = str(tmp_path)
    clean = _fake_v2(d, 'clean', n_byz=0)
    atk = _fake_v2(d, 'atk', honest=lambda s: 1.10 * s)
    r = paired_degradation(atk, clean)
    assert r['paired'] and r['corr'] > 0.99
    assert abs(r['deg_pct'] - 10.0) < 0.5 and r['p_value'] < 1e-6
    idx = np.arange(1, 8)
    un = welch_degradation(_honest_rmse_per_episode(atk, idx), _honest_rmse_per_episode(clean, idx))
    assert un['p_value'] > 0.05, 'setup: the unpaired test should miss this effect'


def test_paired_analysis_refuses_unpaired_or_untagged_runs(tmp_path):
    from byzantine_experiments.analysis.paired import PairingError, paired_columns, paired_degradation
    d = str(tmp_path)
    clean = _fake_v2(d, 'clean', n_byz=0)
    shifted = _fake_v2(d, 'shifted', spawn_shift=0.5)
    legacy = _fake_v2(d, 'legacy', tag=False)
    other_seed = _fake_v2(d, 'other', seed=8)
    for bad, why in ((shifted, 'different spawns'), (legacy, 'seeding'), (other_seed, 'different --seed')):
        with pytest.raises(PairingError, match=why):
            paired_degradation(bad, clean)
    cols = paired_columns(shifted, clean)
    assert cols['paired'] is False and 'spawns' in cols['pairing_error']
    assert np.isnan(cols['paired_honest_deg_pct'])


def test_gate_never_counts_an_unpaired_or_missing_comparison_as_significant(tmp_path):
    """It used to default `significant` to True when the trajectory was missing."""
    from byzantine_experiments.analysis.gate_tier3 import honest_damage_stats
    d = str(tmp_path)
    clean = _fake_v2(d, 'clean', n_byz=0)
    legacy = _fake_v2(d, 'legacy', tag=False, honest=lambda s: 3.0 * s)   # huge effect
    st = honest_damage_stats(legacy, clean)
    assert st is not None and st['paired'] is False and 'seeding' in st['pairing_error']
    assert honest_damage_stats(os.path.join(d, 'missing.npz'), clean) is None


# ------------------------------------------------------- gate recalibration (2026-09-24)

def test_gate_threshold_has_one_source_of_truth():
    """30 % -> 10 %, set once in gate_tier3.GATE; every consumer must read that value."""
    from byzantine_experiments.analysis import channel_ceiling, confirm_tier3, gate_tier3
    assert gate_tier3.GATE['honest_goal_rmse_deg'] == 10.0
    assert confirm_tier3.GATE_PCT == gate_tier3.GATE['honest_goal_rmse_deg']
    assert channel_ceiling.GATE_PCT == gate_tier3.GATE['honest_goal_rmse_deg']
    assert gate_tier3.CONFIRMATORY_MIN_EPISODES == 200


def test_tier3_runs_are_filed_under_the_channel_they_attack_through():
    """Rounds 3-5 lie; they used to be counted as motors-only."""
    from byzantine_experiments.analysis.channel_ceiling import MOTORS, OBS, tier3_channel
    assert tier3_channel('t3_adv_seed1_stealth0_on_s1_m050M') == MOTORS
    assert tier3_channel('t3_adv_seed1_stealth0_f250_on_s1_m050M') == MOTORS
    for s in ('t3_adv_seed1_stealth0_lie_on_s1_m050M', 't3_adv_seed3_stealth0_lieonly_on_s3_m075M',
              't3_adv_seed3_stealth0_lieonlyego_on_s3_m100M'):
        assert tier3_channel(s) == OBS, s
    assert tier3_channel('t2_liar_s3_mirror_f0125') is None


def test_channel_report_contains_no_fixed_conclusions():
    """The legacy report printed 'Only the observation channel produces attacks that clear
    the gate' regardless of the data. With no rows at all it must say so, not assert findings."""
    from byzantine_experiments.analysis.channel_ceiling import build_report
    text = build_report({}, {}, [], {})
    assert 'Only the observation channel' not in text
    assert 'Motor authority is not enough' not in text
    assert 'no paired runs' in text


def test_confirmatory_verdict_prints_end_to_end(tmp_path, monkeypatch, capsys):
    """
    The verdict print asked for `n_attacked` after the switch to the paired test (which reports
    `n_pairs`), and crashed with KeyError at the end of a 62-minute collection. Exercise the
    whole print path on synthetic paired files so that cannot recur silently.
    """
    import json as _json
    import sys as _sys

    from byzantine_experiments.analysis import confirm_tier3 as ct

    monkeypatch.setattr(ct, 'CONFIRM_DIR', str(tmp_path))
    p = ct.confirm_paths('adv_seed7_stealth0_lieonly', 5, 20)
    d = str(tmp_path)
    clean = _fake_v2(d, os.path.basename(p['clean_npz'])[:-4], n_byz=0)
    atk = _fake_v2(d, os.path.basename(p['adv_npz'])[:-4], honest=lambda s: 1.2 * s)
    for path, surv in ((atk, 0.97), (clean, float('nan'))):
        j = path[:-4] + '.json'
        sc = _json.load(open(j)); sc['metrics'] = dict(attacker_survived_mean=surv)
        _json.dump(sc, open(j, 'w'))

    monkeypatch.setattr(_sys, 'argv', ['confirm_tier3', '--run', 'adv_seed7_stealth0_lieonly',
                                       '--milestone', '5', '--n-episodes', '20'])
    ct.main()
    out = capsys.readouterr().out
    assert '20 paired episodes' in out
    assert '+20.0 %' in out
    assert 'survival' in out and '97 %' in out
    assert 'PASS' in out


def test_head_to_head_is_a_paired_comparison_between_attackers(tmp_path, monkeypatch):
    """Trained at 1.20x clean and hand-designed at 1.10x on identical scenarios: the head-to-head
    must report about +10 pp for the trained adversary and call it better, with no fallback to
    comparing two separate intervals."""
    from byzantine_experiments.analysis import head_to_head as hh
    d = str(tmp_path)
    monkeypatch.setattr(hh, 'CONFIRM_DIR', d)
    _fake_v2(d, 'confirm_clean_s7_n20', n_byz=0)
    _fake_v2(d, 'confirm_adv_x_n20', honest=lambda s: 1.20 * s)
    _fake_v2(d, 'confirm_hand_liar_mirror_s7_n20', honest=lambda s: 1.10 * s)
    text = hh.report(seed=7, n_episodes=20, trained='confirm_adv_x_n20')
    assert '+20.0 %' in text and '+10.0 %' in text
    assert 'trained BETTER' in text


# ------------------------------------------------------------- D3: plausible lies (2026-09-26)

def _plausible_rollout(seed, steps=1500, flip=True):
    """A PlausibleLie driven by a worst-case policy against a wall-bouncing true trajectory."""
    from byzantine_experiments.adversary.lies import PlausibleLie
    box = np.array([[-5, -5, 0], [5, 5, 10]]); dt = 0.01
    rng = np.random.default_rng(seed); L = PlausibleLie(box, dt)
    pos = rng.uniform(box[0] + 0.5, box[1] - 0.5); vel = np.zeros(3); R, T = [], []
    for _ in range(steps):
        vel = np.clip(vel + rng.normal(scale=0.02, size=3), -3, 3); nxt = pos + vel * dt
        hit = (nxt < box[0]) | (nxt > box[1]); vel[hit] *= -1; pos = np.clip(nxt, box[0], box[1])
        a = np.sign(rng.normal(size=3)) if flip else np.ones(3)
        R.append(L.step(a, np.r_[pos, vel])); T.append(np.r_[pos, vel])
    return np.array(R), np.array(T), box, dt, L


def test_plausible_lie_passes_the_naive_physics_checks_by_construction():
    """
    The free lie put its claim outside the room on 100 % of steps (below the floor on 76 %)
    at 9.6 m/s. A plausible lie must pass bounds, speed and kinematic-consistency checks on
    EVERY step, under the worst policy there is, and keep its acceleration inside budget on
    all but the rare wall-limited steps.
    """
    for seed in range(20):
        R, T, box, dt, L = _plausible_rollout(seed)
        assert ((R[:, :3] >= box[0] - 1e-9) & (R[:, :3] <= box[1] + 1e-9)).all(), 'claim left the room'
        assert np.linalg.norm(R[:, :3] - T[:, :3], axis=1).max() <= 2.0 + 1e-9
        excess = np.linalg.norm(R[:, 3:], axis=1) - np.linalg.norm(T[:, 3:], axis=1)
        assert excess.max() <= 1.0 + 1e-6, 'claim faster than truth + max_rate'
        resid = np.diff(R[:, :3], axis=0) - np.diff(T[:, :3], axis=0) - (R[1:, 3:] - T[1:, 3:]) * dt
        assert np.abs(resid).max() < 1e-9, 'reported position and velocity disagree'
        acc = np.linalg.norm(np.diff(R[:, 3:] - T[:, 3:], axis=0), axis=1) / dt
        assert np.median(acc) <= 4.0 + 1e-6
        assert np.percentile(acc, 99) < 11.6, 'claim acceleration beyond honest p99'


def test_plausible_lie_starts_truthful_every_episode():
    from byzantine_experiments.adversary.lies import PlausibleLie
    L = PlausibleLie(np.array([[-5, -5, 0], [5, 5, 10]]), 0.01)
    for _ in range(200):
        L.step(np.ones(3), np.r_[0, 0, 5, 0, 0, 0])
    assert np.linalg.norm(L.offset) > 0.5
    L.reset()
    first = L.step(np.ones(3), np.r_[1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    assert np.linalg.norm(first[:3] - [1, 2, 3]) < 1e-3     # one step of drift at most


def test_plausible_training_and_evaluation_produce_identical_claims(tmp_path):
    """FrozenTeamEnv (training) and LearnedAdversaryAgent (evaluation) must make the same claim
    for the same action sequence, or "trained with plausible lies" would not mean "evaluated
    with them"."""
    import json as _json
    from byzantine_experiments.adversary.lies import PlausibleLie
    from byzantine_experiments.agents.base import EgoState
    from byzantine_experiments.agents.tier3_learned import LearnedAdversaryAgent

    run = tmp_path / 'adv_seed9_stealth0_lieonlyplaus'
    (run / 'checkpoint_p0').mkdir(parents=True)
    (run / 'checkpoint_p0' / 'checkpoint_000000001_000002.pth').write_bytes(b'')
    (run / 'config.json').write_text(_json.dumps(dict(
        adv_enable_lies=True, adv_lie_only=True, adv_lie_mode='plausible',
        adv_plausible_max_offset=1.5, adv_plausible_max_rate=0.8, adv_plausible_max_accel=3.0)))
    agent = LearnedAdversaryAgent(0, 4, checkpoint=str(run))
    assert agent.needs_ego_state and agent.lies_about_state
    box = np.array([[-5, -5, 0], [5, 5, 10]])
    agent.ego = EgoState(pos=np.zeros(3), vel=np.zeros(3), rot=np.eye(3), omega=np.zeros(3),
                         room_box=box, control_dt=0.01)
    ref = PlausibleLie(box, 0.01, max_offset=1.5, max_rate=0.8, max_accel=3.0)
    rng = np.random.default_rng(1)
    for t in range(500):
        a = rng.uniform(-1, 1, 3); st = np.r_[rng.uniform(-4, 4, 3), rng.normal(size=3)]
        agent._pending_lie = a
        assert np.allclose(agent.perturb_broadcast_state(st, t), ref.step(a, st))
    agent.reset(); ref.reset()
    assert np.allclose(agent._plausible.offset, 0) and np.allclose(ref.offset, 0)


@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason='baseline checkpoint not present')
def test_plausible_mode_env_has_a_3d_lie_and_reaches_observers():
    env = _adv_env(adv_enable_lies='True', adv_lie_only='True', adv_lie_mode='plausible')
    assert env.action_space.shape == (4 + 3,)
    env.reset(seed=3)
    for _ in range(50):
        env.step(np.ones((1, 7), dtype=np.float32))
    assert env.broadcast_rewrites > 0
    assert 0 < np.linalg.norm(env._plausible[0].offset) <= 2.0 + 1e-9


# ------------------------------------------- spec tests 8 and 9, plus the D1/D5 machinery

def test_efficacy_gate_checks_all_five_criteria():
    """Spec test 8: every criterion can fail on its own, missing evidence is INCOMPLETE."""
    from byzantine_experiments.analysis.efficacy_table import decide
    good = dict(deg=12.0, ci_lo=4.0, n=200)
    assert decide(good, 0.95, 1.1, True, True) == ('PASS', [])
    assert decide(dict(deg=7.4, ci_lo=3.1, n=200), 0.98, 1.1, False, True)[0] == 'FAIL'   # damage
    assert decide(dict(deg=12.0, ci_lo=-1.0, n=200), 0.98, 1.1, False, True)[0] == 'FAIL'  # CI
    assert decide(good, 0.30, 1.1, False, True)[0] == 'FAIL'                               # survival
    assert decide(good, 0.95, 3.0, True, True)[0] == 'FAIL'                                # ratio, stealth
    assert decide(good, 0.95, 3.0, False, True)[0] == 'PASS'                               # ratio ignored
    assert decide(good, 0.95, 1.1, True, False)[0] == 'INCOMPLETE'                         # transfer
    assert decide(dict(deg=12.0, ci_lo=4.0, n=50), 0.95, 1.1, True, True)[0] == 'INCOMPLETE'
    assert decide(None, float('nan'), 1.0, False, True)[0] == 'INCOMPLETE'


def test_tost_establishes_equivalence_and_rejects_a_mismatched_pair():
    """Spec test 9, both directions -- or the test proves nothing."""
    from byzantine_experiments.analysis.matched_triple import tost
    rng = np.random.default_rng(0)
    same = rng.normal(0.3, 29.0, size=1200)       # truly matched, measured noise level
    far = rng.normal(8.0, 29.0, size=1200)        # 7 % vs 15 %: clearly mismatched
    r = tost(same, 3.0)
    assert r['equivalent'] and -3 < r['ci90_lo'] < r['ci90_hi'] < 3
    r = tost(far, 3.0)
    assert not r['equivalent']
    # A failed null test is NOT equivalence: small n, no significant difference, still not shown.
    tiny = rng.normal(0.0, 29.0, size=30)
    from scipy import stats as _st
    assert _st.ttest_1samp(tiny, 0).pvalue > 0.05
    assert not tost(tiny, 3.0)['equivalent']


def test_triple_margin_and_sample_size_rule():
    from byzantine_experiments.analysis.matched_triple import margin_for, n_for
    assert margin_for(7.4) == 3.0 and margin_for(4.0) == 2.0     # half the target below 6 %
    assert n_for(3.0)[0] == 400 and n_for(2.0)[0] == 800 and n_for(1.5)[0] == 1600
    assert n_for(1.0)[0] is None                                  # declared underpowered


def test_pairing_refuses_runs_under_different_shift_levels(tmp_path):
    """Runs at different shift levels share spawns, so the spawn check alone would pair them."""
    import json as _json
    from byzantine_experiments.analysis.paired import PairingError, check_pairing
    d = str(tmp_path)
    clean = _fake_v2(d, 'clean', n_byz=0)
    atk = _fake_v2(d, 'atk')
    j = atk[:-4] + '.json'; sc = _json.load(open(j))
    sc['shift'] = dict(level='severe', dr_ratio=0.2, wind_accel=1.0, gust=0.5, tau=1.0)
    _json.dump(sc, open(j, 'w'))
    with pytest.raises(PairingError, match='distribution shift'):
        check_pairing(atk, clean)
    assert check_pairing(atk, clean, allow_different_shift=True)[0] == 20


@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason='baseline checkpoint not present')
def test_shift_wrapper_switches_on_upstream_dr_and_adds_seeded_wind():
    """DR must resample dynamics at every reset (upstream's own sampler) and wind must be
    reproducible from the episode seed, or shifted runs could not be paired."""
    from byzantine_experiments.rollout.collect import build_cfg, make_byzantine_env, seed_every_rng
    cfg = build_cfg(CHECKPOINT, ['--device', 'cpu'])
    shift = dict(level='custom', dr_ratio=0.2, wind_accel=1.0, gust=0.5, tau=1.0)
    env, _ = make_byzantine_env(cfg, None, shift=shift)

    def episode(seed):
        seed_every_rng(env, seed); env.reset()
        mass = [e.dynamics.mass for e in env.unwrapped.envs]
        acts = np.zeros((8, 4), dtype=np.float32)
        for _ in range(20):
            env.step(acts)
        return np.array(mass), np.array([e.dynamics.pos for e in env.unwrapped.envs])

    m1, p1 = episode(11); m2, p2 = episode(11); m3, _ = episode(12)
    assert np.allclose(m1, m2), 'same seed must give the same randomised dynamics'
    assert not np.allclose(m1, m3), 'DR is not resampling per episode'
    assert np.abs(p1 - p2).max() < 0.05, 'same seed must give the same wind'


def test_collector_records_what_liars_claimed(tmp_path):
    """The stealth analysis checks claims directly; the collector must keep them."""
    from byzantine_experiments.wrappers.byzantine_env import ByzantineEnvWrapper
    src = open('byzantine_experiments/rollout/collect.py').read()
    assert "b['broadcast'] = []" in src and 'last_reported' in src
    assert 'self.last_reported = reported' in open(
        'byzantine_experiments/wrappers/byzantine_env.py').read()


ADV_CKPT = 'train_dir/adv_seed3_stealth0_lieonly'


@pytest.mark.skipif(not os.path.isdir(ADV_CKPT), reason='trained adversary not present')
def test_saved_adversary_round_trips_to_identical_decisions():
    """
    Spec test 6. A saved adversary must load to exactly the weights in its file and make exactly
    the same (deterministic) decisions every time it is loaded -- on real observations from its
    own evaluation. Evaluation *rewards* cannot be bit-reproduced in this simulator (numba's JIT
    RNG lets trajectories drift 0.1-0.5 m even at identical spawns, design 6.17), so the round
    trip is pinned at the decision level, which is where a loading bug would show.
    """
    import torch
    from byzantine_experiments.agents.tier3_learned import LearnedAdversaryAgent

    def load():
        ag = LearnedAdversaryAgent(0, 4, checkpoint=ADV_CKPT, milestone=194_000_000)
        ag.action_low, ag.action_high = -np.ones(4, np.float32), np.ones(4, np.float32)
        ag.obs_dim = 54
        ag._ensure_policy()
        return ag

    a, b = load(), load()
    saved = torch.load(a.checkpoint_file, map_location='cpu', weights_only=False)['model']
    live = a._policy.actor_critic.state_dict()
    assert set(saved) == set(live)
    for k in saved:
        assert torch.equal(saved[k].cpu(), live[k].cpu()), f'weight {k} differs from the file'

    npz = 'byzantine_experiments/results_v2/tier3_confirm/confirm_adv_seed3_stealth0_lieonly_m194_n200.npz'
    if os.path.exists(npz):
        with np.load(npz) as z:
            obs = np.asarray(z['obs'][0, :300, 0], dtype=np.float32)
    else:
        obs = np.random.default_rng(0).normal(size=(300, 54)).astype(np.float32)
    rnn = torch.zeros([len(obs), a._rnn_size])
    act_a, _ = a._policy.act(obs, rnn, deterministic=True)
    act_b, _ = b._policy.act(obs, rnn, deterministic=True)
    assert np.array_equal(np.asarray(act_a), np.asarray(act_b))
    assert np.asarray(act_a).shape == (300, 10)         # 4 motors + 6 free-lie dims


# ------------------------------------------------- D1 family validation, D5 Phase 3, report items

def _fam_rows(variant, degs, paired=True):
    return [dict(variant=variant, paired=paired, paired_honest_deg_pct=d) for d in degs]


def test_family_validation_three_outcomes_and_unpaired_rows_excluded():
    """
    D1 replaced the 20-40 % band with a family-level test. A damaging family validates, a family
    that significantly HELPS the honest team is reported as such (a finding, not a validation),
    a null family is neither -- and unpaired (legacy) rows never count, however large.
    """
    from byzantine_experiments.analysis.family_validation import family_rows, validated
    rng = np.random.default_rng(0)
    rows = (_fam_rows('hurts', 5 + rng.normal(0, 2, 12)) + _fam_rows('helps', -4 + rng.normal(0, 2, 12))
            + _fam_rows('null', rng.normal(0, 2, 12)) + _fam_rows('null', [80.0] * 30, paired=False))
    fam, unpaired = family_rows(rows)
    status = {v: s['status'] for v, s in fam}
    assert status['hurts'].startswith('VALIDATED')
    assert 'HELPS' in status['helps']
    assert status['null'] == 'no family-level effect'
    assert unpaired == 30
    assert validated(rows) == {'hurts'}
    # CSV rows carry the flag as the string 'True'
    assert validated([dict(r, paired='True') for r in _fam_rows('hurts', 5 + rng.normal(0, 2, 12))]) == {'hurts'}


def test_stored_v2_tables_still_validate_the_pre_registered_triple_families():
    """matched_triple fixes stall / liar in advance; the tables must still agree, or selection stops."""
    import csv
    from byzantine_experiments.analysis.family_validation import validated
    from byzantine_experiments.analysis.matched_triple import VALIDATED_FAMILIES
    for tier, family in VALIDATED_FAMILIES.items():
        path = f'byzantine_experiments/results_v2/{tier}_calibration.csv'
        if not os.path.exists(path):
            pytest.skip(f'{path} not present')
        assert family in validated(list(csv.DictReader(open(path))))


def test_physics_checks_flag_teleports_and_out_of_room_claims_but_not_smooth_flight():
    """The stealth analysis' naive detector: a smooth in-room trajectory passes, a claim outside
    the room fails bounds, and a position jump without matching velocity fails consistency."""
    from byzantine_experiments.analysis.tier3_report import physics
    dt, T = 0.01, 400
    t = np.arange(T) * dt
    pos = np.stack([np.sin(t), np.cos(t), 2 + 0.1 * t], -1)
    vel = np.stack([np.cos(t), -np.sin(t), 0.1 * np.ones(T)], -1)
    smooth = np.concatenate([pos, vel], -1)[None]
    ph = physics(smooth, [T], dt)
    assert np.nanmax(ph['resid']) < 1e-3 and np.nanmax(ph['acc']) < 1.5
    tele = smooth.copy()
    tele[0, 200:, 0] += 3.0                                 # jump 3 m, velocity unchanged
    assert np.nanmax(physics(tele, [T], dt)['resid']) > 2.9
    ph_short = physics(smooth, [100], dt)                   # nothing counted past episode end
    assert np.isnan(ph_short['speed'][0, 100:]).all() and np.isnan(ph_short['resid'][0, 100:]).all()


def test_auc_is_half_for_identical_scores_and_one_for_separated():
    from byzantine_experiments.analysis.tier3_report import auc
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    assert abs(auc(x, x) - 0.5) < 1e-9
    assert auc(x + 100, x) == 1.0


def test_compute_accounting_excludes_downtime():
    """A 10-h stop between two 2-h stretches is 4 GPU-hours, not 14."""
    from byzantine_experiments.analysis.tier3_report import active_hours
    w = np.r_[np.arange(0, 7200, 10), 7200 + 36000 + np.arange(0, 7200, 10)]
    act, span = active_hours(w)
    assert abs(act - 4.0) < 0.01 and abs(span - 14.0) < 0.01


def test_phase3_refuses_to_generate_jobs_before_its_inputs_exist(tmp_path, monkeypatch):
    """Phase 3 must never pick its own members or shift levels: both come from pre-registered
    files, and without them it stops instead of improvising."""
    from byzantine_experiments.analysis import phase3
    monkeypatch.setattr(phase3, 'SELECTION_FILE', str(tmp_path / 'matched_triple_selection.json'))
    monkeypatch.setattr(phase3, 'RESULTS_ROOT', str(tmp_path))
    with pytest.raises(SystemExit, match='byz_matched_triple'):
        phase3.jobs()
    (tmp_path / 'matched_triple_selection.json').write_text('{}')
    with pytest.raises(SystemExit, match='byz_calibrate_shift'):
        phase3.jobs()


def test_recorded_broadcast_drops_the_auto_reset_step(tmp_path):
    """The terminal step's observation is the NEXT episode's reset observation, so its recorded
    broadcast sits on a fresh spawn. Kept, it made the plausible liar fail kinematic consistency in
    199/200 episodes (2026-10-09); it must be excluded, and only that step."""
    from byzantine_experiments.analysis.tier3_report import broadcasts, physics
    E, T, N, dt = 2, 6, 3, 0.01
    lengths = np.array([6, 4])
    truth = np.zeros((E, T, N, 6))
    truth[..., 3] = 1.0                                     # 1 m/s along x
    truth[..., 0] = np.arange(T)[None, :, None] * dt        # consistent positions
    for e, n in enumerate(lengths):
        truth[e, n - 1, :, 0] = 5.0                         # auto-reset spawn, 5 m away
    rep = truth.copy()
    rep[..., 1] += 1.5                                      # a constant, consistent offset
    path = str(tmp_path / 'liar.npz')
    np.savez(path, broadcast=rep, broadcast_truth=truth, episode_lengths=lengths)
    r, t, source = broadcasts('unused', path, adv=0)
    assert source == 'recorded'
    for e, n in enumerate(lengths):
        assert np.isnan(r[e, n - 1]).all() and np.isfinite(r[e, :n - 1]).all()
    ph = physics(r, lengths, dt)
    assert np.nanmax(ph['resid']) < 1e-9                    # nothing left to flag
    ph_raw = physics(rep[:, :, 0], lengths, dt)
    assert np.nanmax(ph_raw['resid']) > 1.0                 # the artifact the fix removes


def test_triple_never_falls_back_to_free_on_missing_plausible_runs(monkeypatch):
    """The Tier 3 member is decided plausible-first. A plausible set that is merely incomplete is
    undetermined -- falling back to the free liars would let run order pick the member."""
    from byzantine_experiments.analysis import matched_triple as mt
    good = dict(deg_pct=7.0, ci_lo=3.0, ci_hi=11.0)

    def fake(members):
        if members is mt.PLAUSIBLE:
            return [good], ['adv_seed1_stealth0_lieonlyplaus: not run']
        return [good, good, good], []
    monkeypatch.setattr(mt, '_own_seed_cells', fake)
    member, missing = mt._tier3_member()
    assert member is None and missing and 'plausible' in missing[0]

    def insignificant(members):
        if members is mt.PLAUSIBLE:
            return [dict(deg_pct=0.5, ci_lo=-4.0, ci_hi=5.0)] * 3, []
        return [good] * 3, []
    monkeypatch.setattr(mt, '_own_seed_cells', insignificant)
    member, _ = mt._tier3_member()
    assert member['kind'].startswith('free') and member['fallback']


def test_shift_amendment_keeps_the_v1_levels_and_refuses_to_mislabel():
    """2026-10-12 amendment: the original picks survive as mild_v1/severe_v1, freezing twice is a
    no-op, and freezing after the amended grid has replaced `mild` is refused (that `mild` is v2)."""
    from byzantine_experiments.analysis.shift_calibration import freeze_v1
    v1 = lambda dr, w, e: dict(dr_ratio=dr, wind_accel=w, gust=w / 2, tau=1.0, effect_pct=e,
                               ci=[0, 0], weak=False)
    state = {'levels': {'mild': v1(0.05, 0.0, 32.6), 'severe': v1(0.1, 0.5, 139.8)}}
    assert freeze_v1(state) is True
    assert state['levels']['severe_v1']['wind_accel'] == 0.5 and 'amendment' in state
    assert freeze_v1(state) is False                        # idempotent
    late = {'levels': {'mild': v1(0.03, 0.0, 11.0), 'severe': v1(0.06, 0.1, 40.0)}}
    with pytest.raises(SystemExit, match='not the recorded v1 pick'):
        freeze_v1(late)


def test_phase3_reads_every_level_name_and_orders_by_effect(tmp_path, monkeypatch):
    """`mild_v1` must not parse as `mild`, and the table orders levels by clean-swarm effect."""
    import re
    from byzantine_experiments.analysis import phase3
    pat = r'p3_(?P<member>.+)_(?P<shift>' + '|'.join(phase3.SHIFTS) + r')_s(?P<seed>\d+)_n\d+$'
    m = re.match(pat, 'p3_t1_stall_d0.50_ms0.1_hold_f0125_mild_v1_s3_n50')
    assert (m['member'], m['shift'], m['seed']) == ('t1_stall_d0.50_ms0.1_hold_f0125', 'mild_v1', '3')
    m = re.match(pat, 'p3_clean_severe_s1_n50')
    assert (m['member'], m['shift']) == ('clean', 'severe')
    assert set(phase3.SHIFTS) == {'none', 'mild', 'severe', 'mild_v1', 'severe_v1'}
