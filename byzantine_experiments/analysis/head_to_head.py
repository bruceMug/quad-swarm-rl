"""
Trained vs hand-designed, on identical scenarios -- the measurement Tier 3's claim rests on.

Every rollout at one `--seed` draws the same scenarios (`collect.py::episode_seed`), so the
trained adversary's confirmatory run and each hand-designed run at that seed are paired with the
clean run *and with each other*. That allows two readings per hand-designed attack:

  1. its own paired degradation versus clean, directly comparable to the trained adversary's;
  2. the trained-minus-hand-designed difference in honest goal RMSE, per scenario, as a
     fraction of clean. Positive means the trained adversary does more damage. This is the
     head-to-head: one paired test, no comparison of two separately-estimated intervals.

Every pair is verified with `paired.check_pairing` first; nothing is computed on unpaired runs.

    python -m byzantine_experiments.analysis.head_to_head --seed 3 --n-episodes 200
"""

import argparse
import glob
import os

import numpy as np
from scipy import stats

from byzantine_experiments.analysis.gate_tier3 import GATE, _honest_rmse_per_episode
from byzantine_experiments.analysis.paired import check_pairing, paired_degradation
from byzantine_experiments.paths import RESULTS_ROOT

CONFIRM_DIR = os.path.join(RESULTS_ROOT, 'tier3_confirm')
TRAINED_DEFAULT = 'confirm_adv_seed3_stealth0_lieonly_m194_n200'


def difference(trained_npz, hand_npz, clean_npz, idx):
    """Trained minus hand-designed honest goal RMSE, per scenario, as % of the clean mean."""
    check_pairing(trained_npz, clean_npz)
    check_pairing(hand_npz, clean_npz)
    n, _ = check_pairing(trained_npz, hand_npz)
    t = _honest_rmse_per_episode(trained_npz, idx)[:n]
    h = _honest_rmse_per_episode(hand_npz, idx)[:n]
    c = _honest_rmse_per_episode(clean_npz, idx)[:n]
    d = 100.0 * (t - h) / c.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    return dict(diff_pp=float(d.mean()), lo=float(d.mean() - crit * se),
                hi=float(d.mean() + crit * se), p=float(stats.ttest_rel(t, h).pvalue), n=n)


def report(seed=3, n_episodes=200, trained=TRAINED_DEFAULT):
    clean = os.path.join(CONFIRM_DIR, f'confirm_clean_s{seed}_n{n_episodes}.npz')
    trained_npz = os.path.join(CONFIRM_DIR, trained + '.npz')
    hands = sorted(glob.glob(os.path.join(CONFIRM_DIR,
                                          f'confirm_hand_*_s{seed}_n{n_episodes}.npz')))
    for path in (clean, trained_npz):
        if not os.path.exists(path):
            return f'missing {path}'
    if not hands:
        return f'no hand-designed confirmatory runs under {CONFIRM_DIR}'

    with np.load(trained_npz) as z:
        idx = np.where(~np.asarray(z['byzantine_mask']))[0]
    gate = GATE['honest_goal_rmse_deg']
    L = [f'Head to head, baseline seed {seed}, {n_episodes} identical scenarios per arm\n',
         f"  {'attack':34s} {'vs clean':>22s} {'p':>8s}  {'gate':>4s}"]

    def row(label, npz):
        r = paired_degradation(npz, clean, idx=idx)
        ok = r['deg_pct'] >= gate and r['ci_lo'] > 0
        L.append(f"  {label:34s} {r['deg_pct']:+6.1f} % [{r['ci_lo']:+5.1f}, {r['ci_hi']:+5.1f}] "
                 f"{r['p_value']:8.4f}  {'PASS' if ok else 'fail'}")

    row('TRAINED ' + trained.replace('confirm_adv_', '').replace(f'_n{n_episodes}', ''),
        trained_npz)
    for h in hands:
        row('hand-designed ' + os.path.basename(h).replace('confirm_hand_', '')
            .replace(f'_s{seed}_n{n_episodes}.npz', ''), h)

    L.append('\n  Trained minus hand-designed (paired per scenario; + = trained does more damage):')
    for h in hands:
        r = difference(trained_npz, h, clean, idx)
        verdict = ('trained BETTER' if r['lo'] > 0 else
                   'hand-designed BETTER' if r['hi'] < 0 else 'no detectable difference')
        # Strip the exact `_s{seed}_n{n}.npz` suffix. Splitting on '_s' cut `liar_stale` to
        # `liar`, mislabelling the third row of the first real head-to-head.
        name = os.path.basename(h).replace('confirm_hand_', '').replace(
            f'_s{seed}_n{n_episodes}.npz', '')
        L.append(f"    vs {name:14s} "
                 f"{r['diff_pp']:+5.1f} pp [{r['lo']:+5.1f}, {r['hi']:+5.1f}]  "
                 f"p={r['p']:.4f}  -> {verdict}")
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--n-episodes', type=int, default=200)
    ap.add_argument('--trained', default=TRAINED_DEFAULT)
    args = ap.parse_args()
    print(report(args.seed, args.n_episodes, args.trained))


if __name__ == '__main__':
    main()
