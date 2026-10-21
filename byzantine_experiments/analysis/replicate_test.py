"""
Compare a condition to clean using REPLICATE RUNS, not one run against one run.

Every comparison in this project up to now has been a single attacked rollout against a single
clean rollout, with a confidence interval built from the episode-to-episode variance inside
those two runs. Three independent clean rollouts on baseline seed 3 showed that is not the
whole error:

    t1_clean_s3.npz              n= 50   mean 0.9938
    confirm_clean_s3_n200.npz    n=200   mean 1.1783    <- same git SHA and config as below
    confirm_clean_s3_rep2_n200   n=200   mean 1.0568    <- differs only in --seed

Cochran Q = 11.25 on 2 df (p = 0.004), I^2 = 82 %, run-level sd tau = 7.2 % of the mean; the
two same-SHA runs alone give 7.3 %, so it is not code drift. Episodes *within* a run are iid
(Ljung-Box p = 0.96, no trend), so the structure is a per-run offset: different seeds draw
different scenario mixes, and a constant offset does not average away inside its own run.

The model is therefore

    y_ij = mu_condition + u_i + eps_ij ,   u_i ~ N(0, tau^2) ,  eps_ij ~ N(0, sigma^2)

and the quantity with a defensible error bar is the difference between condition means, with
the run-level term included. With R replicates per arm the cleanest estimator needs no
variance decomposition at all: take each run's mean as one observation and do a Welch t-test on
the R attacked means against the R clean means. That is what this module reports. It is
low-powered at R = 3, which is the honest cost of having discovered the problem late.

The per-episode ("naive") figure is printed alongside so the inflation is visible, never
instead of it.

    python -m byzantine_experiments.analysis.replicate_test --results-dir DIR --seed 3
"""

import argparse
import glob
import os

import numpy as np
from scipy import stats

from byzantine_experiments.analysis.gate_tier3 import _honest_rmse_per_episode


def load_arm(results_dir, pattern, idx):
    """Per-run episode series for every replicate matching `pattern`."""
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, pattern))):
        out.append((os.path.basename(path)[:-4], _honest_rmse_per_episode(path, idx)))
    return out


def tau_from(runs):
    """DerSimonian-Laird run-level sd, as a fraction of the weighted mean."""
    if len(runs) < 2:
        return 0.0, 0.0, float('nan')
    means = np.array([v.mean() for _, v in runs])
    sems = np.array([v.std(ddof=1) / np.sqrt(len(v)) for _, v in runs])
    w = 1.0 / sems ** 2
    mu = float((w * means).sum() / w.sum())
    Q = float((w * (means - mu) ** 2).sum())
    df = len(means) - 1
    C = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max((Q - df) / C, 0.0) if C > 0 else 0.0
    p = float(1 - stats.chi2.cdf(Q, df)) if df > 0 else float('nan')
    return float(np.sqrt(tau2) / mu) if mu > 0 else 0.0, Q, p


def report(results_dir, seed, idx=None):
    idx = np.arange(1, 8) if idx is None else idx
    clean = load_arm(results_dir, f'rep_clean_s{seed}_r*.npz', idx)
    attack = load_arm(results_dir, f'rep_attack_s{seed}_r*.npz', idx)
    L = []
    if not clean or not attack:
        return (f'need both arms under {results_dir} '
                f'(found {len(clean)} clean, {len(attack)} attacked)')

    L.append(f'Replicated comparison, baseline seed {seed}')
    for label, runs in (('clean', clean), ('attacked', attack)):
        L.append(f'  {label}:')
        for name, v in runs:
            L.append(f'    {name:32s} n={len(v):4d}  mean {v.mean():.4f}  '
                     f'sem {v.std(ddof=1)/np.sqrt(len(v)):.4f}')

    cm = np.array([v.mean() for _, v in clean])
    am = np.array([v.mean() for _, v in attack])

    # Naive: pool every episode and ignore which run it came from.
    call = np.concatenate([v for _, v in clean])
    aall = np.concatenate([v for _, v in attack])
    _, p_naive = stats.ttest_ind(aall, call, equal_var=False)
    d_naive = 100.0 * (aall.mean() - call.mean()) / call.mean()

    L.append('\n  NAIVE (all episodes pooled, run identity ignored -- what every previous '
             'number in this project did):')
    L.append(f'    degradation {d_naive:+.1f} %   p={p_naive:.4f}')

    if len(cm) >= 2 and len(am) >= 2:
        t, p_run = stats.ttest_ind(am, cm, equal_var=False)
        d_run = 100.0 * (am.mean() - cm.mean()) / cm.mean()
        va, vc = am.var(ddof=1) / len(am), cm.var(ddof=1) / len(cm)
        se = np.sqrt(va + vc)
        df = ((va + vc) ** 2 / (va ** 2 / (len(am) - 1) + vc ** 2 / (len(cm) - 1))
              if va + vc > 0 else 1.0)
        crit = stats.t.ppf(0.975, df)
        lo = 100.0 * ((am.mean() - cm.mean()) - crit * se) / cm.mean()
        hi = 100.0 * ((am.mean() - cm.mean()) + crit * se) / cm.mean()
        L.append(f'\n  RUN-LEVEL (each run is one observation; {len(am)} vs {len(cm)} runs):')
        L.append(f'    degradation {d_run:+.1f} % [{lo:+.1f}, {hi:+.1f}]  p={p_run:.4f}  '
                 f'{"SIGNIFICANT" if p_run < 0.05 and d_run > 0 else "not significant"}')

    tc, Qc, pc = tau_from(clean)
    ta, Qa, pa = tau_from(attack)
    L.append(f'\n  run-level tau: clean {100*tc:.1f} % (Q={Qc:.2f} p={pc:.4f}), '
             f'attacked {100*ta:.1f} % (Q={Qa:.2f} p={pa:.4f})')
    if len(am) < 3:
        L.append('  NOTE only %d replicates per arm -- the run-level test is very low powered. '
                 'Treat a null here as "not established", not as "no effect".' % len(am))
    return '\n'.join(L)


def paired_report(clean_npz, attacked_npz, idx=None):
    """
    Paired comparison: episode i of both runs is the SAME scenario.

    Only valid for rollouts collected after the `seed_every_rng` fix, with the same `--seed`.
    Before it, two runs at one seed drew different scenarios and pairing was meaningless.

    Pairing removes the run-level offset exactly rather than estimating it, and cancels most
    of the scenario-to-scenario variance too -- goal RMSE has a ~40 % coefficient of variation
    across scenarios but the attack's effect on a *given* scenario is far more consistent. The
    unpaired figure is printed beside it so the gain is visible.
    """
    idx = np.arange(1, 8) if idx is None else idx
    a = _honest_rmse_per_episode(attacked_npz, idx)
    c = _honest_rmse_per_episode(clean_npz, idx)
    n = min(len(a), len(c))
    a, c = a[:n], c[:n]
    L = [f'Paired comparison over {n} matched scenarios',
         f'  clean    mean {c.mean():.4f}', f'  attacked mean {a.mean():.4f}']

    _, p_un = stats.ttest_ind(a, c, equal_var=False)
    se_un = np.sqrt(a.var(ddof=1) / n + c.var(ddof=1) / n)
    L.append(f'\n  UNPAIRED: {100*(a.mean()-c.mean())/c.mean():+.1f} %  '
             f'+/- {100*1.96*se_un/c.mean():.1f} pp   p={p_un:.4f}')

    d = a - c
    t, p_p = stats.ttest_rel(a, c)
    se_p = d.std(ddof=1) / np.sqrt(n)
    lo, hi = d.mean() - 1.96 * se_p, d.mean() + 1.96 * se_p
    L.append(f'  PAIRED  : {100*d.mean()/c.mean():+.1f} % '
             f'[{100*lo/c.mean():+.1f}, {100*hi/c.mean():+.1f}]  p={p_p:.4f}  '
             f'{"SIGNIFICANT" if p_p < 0.05 and d.mean() > 0 else "not significant"}')
    if se_un > 0:
        L.append(f'\n  standard error shrank {se_un/se_p:.1f}x by pairing '
                 f'(corr between arms: {np.corrcoef(a, c)[0,1]:+.3f})')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default='byzantine_experiments/results/tier3_replicate')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--paired', nargs=2, metavar=('CLEAN_NPZ', 'ATTACKED_NPZ'),
                    help='paired comparison of two same-seed rollouts (post seeding fix)')
    args = ap.parse_args()
    if args.paired:
        print(paired_report(*args.paired))
    else:
        print(report(args.results_dir, args.seed))


if __name__ == '__main__':
    main()
