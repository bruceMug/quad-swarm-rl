"""
Paired attacked-vs-clean comparison -- the only inferential comparison the v2 corpus uses.

With the fixed collector, episode i of every run at a given `--seed` is the same scenario
(`collect.py::episode_seed`). An attacked run and a clean run at that seed are therefore paired
episode-for-episode, and the right test is on the per-episode *differences*:

  * the run-level scenario offset that made every legacy CI too narrow (design 6.16) cancels
    exactly instead of having to be estimated;
  * most scenario-to-scenario variance cancels too. On the first valid pair (Tier 2 `mirror`,
    seed 3, n = 50) the correlation between arms was +0.79 and pairing shrank the standard
    error 2.2x -- the difference between p = 0.20 unpaired and p = 0.007 paired on identical
    data.

**Pairing is verified, never assumed.** The first attempt at a paired run passed the same
`--seed` to both arms and still was not paired (episodes 0-2 matched, 46 of 47 later ones did
not, correlation -0.08) -- and nothing flagged it except reading the correlation by eye. So
`paired_degradation` checks, before computing anything:

  1. both sidecars carry `seeding == SEEDING_SCHEME` (legacy files never pair);
  2. both runs used the same `--seed` and the same baseline checkpoint;
  3. every paired episode's spawn positions agree to 1e-6 m.

Any failure raises `PairingError` with the reason. There is deliberately no silent fallback to
an unpaired test: an unpaired number from this simulator is not trustworthy (see 6.16), and a
fallback is exactly how an unpaired number gets reported as a paired one.

Degradation is a ratio, so its interval uses the delta method with the paired covariance:
    R = mean_a / mean_c,  Var(R) ~ (Var(a) - 2R Cov(a,c) + R^2 Var(c)) / (n mean_c^2)
The p-value is the paired t-test on a - c, which tests R = 1 exactly.
"""

import json
import os

import numpy as np
from scipy import stats

from byzantine_experiments.analysis.gate_tier3 import _honest_rmse_per_episode
from byzantine_experiments.paths import SEEDING_SCHEME

SPAWN_TOL_M = 1e-6


class PairingError(ValueError):
    """The two runs are not episode-paired; a paired statistic would be meaningless."""


def _sidecar(npz):
    path = os.path.splitext(npz)[0] + '.json'
    if not os.path.exists(path):
        raise PairingError(f'no sidecar for {npz}')
    with open(path) as f:
        return json.load(f)


def _seed_of(sc):
    return sc.get('seed', None)


def check_pairing(attacked_npz, clean_npz, allow_different_shift=False):
    """
    Return (n_paired, max_spawn_diff_m). Raise PairingError on any mismatch.

    `allow_different_shift=True` is for one caller only: the shift calibration, whose whole
    point is comparing the same scenarios with and without a shift. Everything else must pair
    within one shift level.
    """
    sa, sc = _sidecar(attacked_npz), _sidecar(clean_npz)
    for name, s in (('attacked', sa), ('clean', sc)):
        if s.get('seeding') != SEEDING_SCHEME:
            raise PairingError(
                f'{name} run has seeding={s.get("seeding")!r}, need {SEEDING_SCHEME!r} -- it was '
                'collected before the seeding fix and its scenarios are not controlled')
    if _seed_of(sa) != _seed_of(sc):
        raise PairingError(f'different --seed: attacked {_seed_of(sa)}, clean {_seed_of(sc)}')
    sh_a = (sa.get('shift') or {}).get('level', 'none'), sa.get('shift')
    sh_c = (sc.get('shift') or {}).get('level', 'none'), sc.get('shift')
    norm = lambda x: {k: v for k, v in (x or {'level': 'none'}).items()}
    if not allow_different_shift and (
            sh_a[0] != sh_c[0] or (sh_a[0] != 'none' and norm(sh_a[1]) != norm(sh_c[1]))):
        # Runs under different shift levels draw the same spawns (DR and wind use the global
        # stream, spawns the per-drone one), so the spawn check below would NOT catch this.
        raise PairingError(f'different distribution shift: attacked {sh_a[1]}, clean {sh_c[1]}')
    if sa.get('checkpoint') != sc.get('checkpoint'):
        raise PairingError(f'different baseline: {sa.get("checkpoint")} vs {sc.get("checkpoint")}')

    with np.load(attacked_npz) as d:
        pa = np.asarray(d['positions'][:, 0])
    with np.load(clean_npz) as d:
        pc = np.asarray(d['positions'][:, 0])
    n = min(len(pa), len(pc))
    diff = np.abs(pa[:n] - pc[:n]).max(axis=(1, 2))
    bad = np.flatnonzero(diff > SPAWN_TOL_M)
    if bad.size:
        raise PairingError(
            f'{bad.size}/{n} episodes have different spawns (first at episode {bad[0]}, '
            f'{diff[bad[0]]:.3f} m) -- runs are not paired')
    return n, float(diff.max()) if n else 0.0


def paired_degradation(attacked_npz, clean_npz, idx=None):
    """
    Honest-drone goal-RMSE degradation of `attacked_npz` over its paired clean run.

    `idx` defaults to the attacked run's honest drones; the clean series is restricted to the
    same drones (the matched-subset rule from design 4.3).
    """
    n, spawn_diff = check_pairing(attacked_npz, clean_npz)
    if idx is None:
        with np.load(attacked_npz) as d:
            idx = np.where(~np.asarray(d['byzantine_mask']))[0]
    a = _honest_rmse_per_episode(attacked_npz, idx)[:n]
    c = _honest_rmse_per_episode(clean_npz, idx)[:n]
    if n < 3 or c.mean() <= 1e-9:
        raise PairingError(f'too few paired episodes ({n})')

    ma, mc = a.mean(), c.mean()
    r = ma / mc
    cov = np.cov(a, c, ddof=1)
    var_r = (cov[0, 0] - 2 * r * cov[0, 1] + r ** 2 * cov[1, 1]) / (n * mc ** 2)
    se_r = float(np.sqrt(max(var_r, 0.0)))
    crit = stats.t.ppf(0.975, n - 1)
    _, p = stats.ttest_rel(a, c)
    corr = float(np.corrcoef(a, c)[0, 1])
    se_unpaired = float(np.sqrt((cov[0, 0] + r ** 2 * cov[1, 1]) / (n * mc ** 2)))
    return dict(
        deg_pct=100.0 * (r - 1.0),
        ci_lo=100.0 * (r - 1.0 - crit * se_r),
        ci_hi=100.0 * (r - 1.0 + crit * se_r),
        p_value=float(p), n_pairs=n, corr=corr,
        se_shrink=(se_unpaired / se_r) if se_r > 0 else float('inf'),
        clean_mean=float(mc), attacked_mean=float(ma), max_spawn_diff_m=spawn_diff,
        paired=True,
    )


def paired_columns(attacked_npz, clean_npz):
    """
    Flat columns for a calibration table row. Never raises: an unpairable run gets NaNs and
    `paired=False` plus the reason, so tables show *why* a row has no reportable number.
    """
    nan = float('nan')
    try:
        r = paired_degradation(attacked_npz, clean_npz)
        return dict(paired=True, paired_honest_deg_pct=r['deg_pct'],
                    paired_ci_lo=r['ci_lo'], paired_ci_hi=r['ci_hi'],
                    paired_p=r['p_value'], paired_corr=r['corr'], pairing_error='')
    except (PairingError, FileNotFoundError, KeyError) as e:
        return dict(paired=False, paired_honest_deg_pct=nan, paired_ci_lo=nan,
                    paired_ci_hi=nan, paired_p=nan, paired_corr=nan,
                    pairing_error=str(e)[:160])


def per_episode_degradation(attacked_npz, clean_npz, idx=None, reference_npz=None,
                            allow_different_shift=False):
    """
    Per-episode honest goal-RMSE difference as a % of the clean mean, after verifying pairing.

    With `reference_npz` the difference is attacked minus reference instead of attacked minus
    clean (both still normalised by the clean mean and both verified against clean) -- the
    per-scenario comparison between two conditions that TOST and the head-to-head need.
    """
    n, _ = check_pairing(attacked_npz, clean_npz, allow_different_shift=allow_different_shift)
    if reference_npz is not None:
        n = min(n, check_pairing(reference_npz, clean_npz)[0])
    if idx is None:
        with np.load(attacked_npz) as d:
            idx = np.where(~np.asarray(d['byzantine_mask']))[0]
    a = _honest_rmse_per_episode(attacked_npz, idx)[:n]
    c = _honest_rmse_per_episode(clean_npz, idx)[:n]
    b = c if reference_npz is None else _honest_rmse_per_episode(reference_npz, idx)[:n]
    return 100.0 * (a - b) / c.mean()


def pooled(differences, alpha=0.05):
    """
    Pool per-episode % differences across seeds (the spec's "pooled across the 3 seeds").

    Each seed's episodes are normalised by that seed's own clean mean, so seeds with different
    baseline error contribute on one scale. The interval treats the clean means as fixed; at
    n >= 50 per seed their relative error is ~5 %, i.e. ~0.35 pp on a 7 % effect.
    """
    d = np.concatenate([np.asarray(x, dtype=np.float64) for x in differences])
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    return dict(mean=float(d.mean()), ci_lo=float(d.mean() - crit * se),
                ci_hi=float(d.mean() + crit * se), se=float(se), n=n,
                p_value=float(stats.ttest_1samp(d, 0).pvalue))


def _honest_reward_per_episode(npz, drone_idx):
    """Per-episode honest team reward: summed over the episode, averaged over `drone_idx`."""
    with np.load(npz) as d:
        rew = np.asarray(d['rewards'][:, :, drone_idx], dtype=np.float64)   # (E, T, |idx|)
        lengths = np.asarray(d['episode_lengths'])
    out = np.empty(len(lengths))
    for e, n in enumerate(lengths):
        out[e] = np.nansum(rew[e, :int(n)], axis=0).mean()
    return out


def paired_reward_degradation(attacked_npz, clean_npz, idx=None):
    """
    The spec's SECONDARY gate criterion, paired: honest team-reward degradation with a CI.

    Degradation = (clean - attacked) / |clean mean|, as a per-episode paired difference.
    Reported, never used to gate or to match: mean episode reward in this env is a small number
    on a heavy left tail, so the percentage is unstable (tier2_design 10).
    """
    n, _ = check_pairing(attacked_npz, clean_npz)
    if idx is None:
        with np.load(attacked_npz) as d:
            idx = np.where(~np.asarray(d['byzantine_mask']))[0]
    a = _honest_reward_per_episode(attacked_npz, idx)[:n]
    c = _honest_reward_per_episode(clean_npz, idx)[:n]
    mc = c.mean()
    if abs(mc) < 1e-9:
        raise PairingError('clean mean reward is zero; percentage undefined')
    d = 100.0 * (c - a) / abs(mc)
    se = d.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    return dict(deg_pct=float(d.mean()), ci_lo=float(d.mean() - crit * se),
                ci_hi=float(d.mean() + crit * se), p_value=float(stats.ttest_rel(a, c).pvalue),
                n_pairs=n, clean_mean=float(mc), attacked_mean=float(a.mean()))
