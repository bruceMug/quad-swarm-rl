"""
Confirmatory evaluation of ONE pre-selected Tier 3 checkpoint.

Round 4's horizon sweep is *exploratory*: it evaluates 7 milestone checkpoints of one
adversary and reports the best. That is the right tool for "is anything happening", but it is
the wrong one for a headline number, for two reasons that both inflate the apparent result:

  1. **Multiplicity.** 7 checkpoints tested against a 30 % / p<0.05 bar is 7 chances to clear
     it. Seed-3 round 4 had 2 of 7 land at p = 0.008 and p = 0.013; a Bonferroni correction
     across 7 looks is alpha = 0.0071, which m075 (p = 0.008) then just misses.
  2. **Winner's curse.** The checkpoint chosen *because* it had the highest point estimate is
     biased high -- its degradation is partly signal and partly the favourable tail of a
     +/- 22 pp confidence interval.

The honest way to report "this adversary degrades the honest team by X %" is to pick the
checkpoint on the exploratory sweep, then measure it once more on a *fresh, larger* sample and
report that number. This script is the second half: given one run + one milestone, it collects
`--n-episodes` (default 200) fresh episodes and a matched clean reference, and reports the
degradation with a CI, using the exact same `welch_degradation` the gate uses.

It does NOT re-run the sweep or choose the checkpoint for you -- that choice must be made and
named on the exploratory data, so the confirmatory sample stays independent of it.

    # 1. exploratory (already done): pick the checkpoint
    python -m byzantine_experiments.analysis.gate_tier3 --horizon

    # 2. this script provides the command to collect the fresh confirmatory sample:
    python -m byzantine_experiments.analysis.confirm_tier3 \
        --run adv_seed3_stealth0_lieonly --milestone 75 --print-commands

    # 3. after running those two collect commands, read the verdict:
    python -m byzantine_experiments.analysis.confirm_tier3 \
        --run adv_seed3_stealth0_lieonly --milestone 75
"""

from byzantine_experiments.paths import RESULTS_ROOT, results_dir
import argparse
import glob
import json
import os
import re

import numpy as np

from byzantine_experiments.analysis.gate_tier3 import (
    _honest_rmse_per_episode,
    welch_degradation,
)

PY = '/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python'
CONFIRM_DIR = os.path.join(RESULTS_ROOT, 'tier3_confirm')
CLEAN_DIR = results_dir('tier1')
from byzantine_experiments.analysis.gate_tier3 import GATE  # noqa: E402
GATE_PCT = GATE['honest_goal_rmse_deg']   # single source of truth: gate_tier3.GATE


def best_clean_reference(seed):
    """
    Largest clean reference available for this baseline seed.

    The clean arm is shared by every comparison on a seed, so enlarging it tightens all of
    them at once. Tier 1's reference is n = 50, which dominates the error of a pooled
    attacked estimate (n = 350): sqrt(1/350 + 1/50) = 0.151 against sqrt(1/350 + 1/200) =
    0.084. Collecting one bigger clean rollout is therefore the cheapest precision available,
    and this picks it up automatically once it exists.
    """
    def episode_count(path):
        """Read the real episode count rather than trusting the filename."""
        try:
            with np.load(path) as d:
                return int(np.asarray(d['episode_lengths']).size)
        except Exception:
            return 0

    fallback = os.path.join(CLEAN_DIR, f't1_clean_s{seed}.npz')
    # Seed the incumbent with the fallback's ACTUAL size. Starting from 0 meant any
    # confirm_clean_* file won, including a 3-episode smoke test, which would silently
    # replace Tier 1's 50-episode reference and widen every CI on this seed.
    best = fallback if os.path.exists(fallback) else None
    best_n = episode_count(fallback) if best else 0
    for path in glob.glob(os.path.join(CONFIRM_DIR, f'confirm_clean_s{seed}_n*.npz')):
        n = episode_count(path)
        if n > best_n:
            best, best_n = path, n
    return best if best else fallback


def clean_audit(seed, idx=None):
    """
    Every clean reference on one baseline seed, compared against each other.

    Added after the seed-3 confirmatory measurement: Tier 1's n = 50 clean reference has mean
    honest goal RMSE 0.9938, a fresh n = 200 run has 1.1783, and they differ at p = 0.004.
    Degradation is measured *against* this number, so an 18 % error in it moves every result on
    the seed by roughly 18 points -- enough to turn 10 of 37 significant seed-3 findings into 0
    of 37.

    Episodes within a run are iid (verified: Ljung-Box p = 0.96, no trend), so each run is an
    honest sample and the disagreement is either a ~0.4 % tail event or something systematic
    we have not identified. One run cannot tell those apart; a third can. This lists what is
    available so the question is settled with data rather than by preferring whichever
    reference was collected most recently.
    """
    paths = [os.path.join(CLEAN_DIR, f't1_clean_s{seed}.npz')]
    paths += sorted(glob.glob(os.path.join(CONFIRM_DIR, f'confirm_clean_s{seed}_*.npz')))
    idx = np.arange(1, 8) if idx is None else idx
    out = []
    for path in paths:
        if not os.path.exists(path):
            continue
        v = _honest_rmse_per_episode(path, idx)
        out.append((os.path.basename(path), v))
    return out


def run_level_tau(seed=3):
    """
    Run-to-run standard deviation of the clean baseline, as a FRACTION of its mean.

    Three independent clean rollouts on baseline seed 3 -- n=50 (Aug, seed 3), n=200 (Sep,
    seed 3) and n=200 (Sep, seed 3003), the last two at the same git SHA and identical config
    -- have means 0.9938, 1.1783 and 1.0568. Cochran Q = 11.25 on 2 df (p = 0.004), I^2 = 82 %:
    they disagree far more than their episode-level standard errors allow.

    Episodes *within* a run are iid (Ljung-Box p = 0.96, no trend), so this is not
    autocorrelation. It is a per-run offset -- different `--seed` values draw different
    scenario mixes, and the mean does not converge away because the offset is constant within
    the run. The model is y_ij = mu + u_i + eps_ij with u_i ~ N(0, tau^2).

    **Consequence: every comparison in this project is one run against one run, so its true
    standard error carries 2*tau^2 that none of the reported CIs include.** At n = 50 per arm
    that widens a degradation CI from about +/- 13 pp to about +/- 24 pp.

    Estimated by DerSimonian-Laird on whatever clean replicates exist. Returns 0.0 when there
    are fewer than two, which reproduces the old (over-confident) behaviour explicitly rather
    than by omission.
    """
    refs = clean_audit(seed)
    if len(refs) < 2:
        return 0.0
    means = np.array([v.mean() for _, v in refs])
    sems = np.array([v.std(ddof=1) / np.sqrt(len(v)) for _, v in refs])
    w = 1.0 / sems ** 2
    mu = float((w * means).sum() / w.sum())
    Q = float((w * (means - mu) ** 2).sum())
    C = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max((Q - (len(means) - 1)) / C, 0.0) if C > 0 else 0.0
    return float(np.sqrt(tau2) / mu) if mu > 0 else 0.0


def degradation_with_run_variance(atk, cln, tau_rel):
    """
    `welch_degradation`, but with the run-level variance component added to both arms.

    Delta method on D = 100*(mean_a/mean_c - 1), with
    Var(mean) = sem^2 + (tau_rel * mean)^2 for each arm.
    """
    base = welch_degradation(atk, cln)
    if base is None:
        return None
    a, c = np.asarray(atk, float), np.asarray(cln, float)
    ma, mc = a.mean(), c.mean()
    va = a.var(ddof=1) / len(a) + (tau_rel * ma) ** 2
    vc = c.var(ddof=1) / len(c) + (tau_rel * mc) ** 2
    var_d = (100.0 / mc) ** 2 * va + (100.0 * ma / mc ** 2) ** 2 * vc
    se = np.sqrt(var_d)
    z = 1.959963985
    out = dict(base)
    out.update(deg_pct=base['deg_pct'], ci_lo=base['deg_pct'] - z * se,
               ci_hi=base['deg_pct'] + z * se, se_pp=float(se),
               p_value=float(2 * (1 - stats_norm_cdf(abs(base['deg_pct']) / se))) if se > 0 else 1.0,
               tau_rel=tau_rel)
    return out


def stats_norm_cdf(x):
    from scipy import stats as _st
    return float(_st.norm.cdf(x))


def _honest_series(npz):
    with np.load(npz) as d:
        idx = np.where(~np.asarray(d['byzantine_mask']))[0]
    return _honest_rmse_per_episode(npz, idx), idx


def pooled_and_shrunk(run, horizon_dir=os.path.join(RESULTS_ROOT, 'tier3')):
    """
    Two unbiased readings of an exploratory horizon sweep, to put its maximum in context.

    **Pooled**: every checkpoint's episodes concatenated and compared to clean. This estimates
    "how much damage this training run's policies do" with no selection at all, so it is the
    honest summary of the run -- but it is not an estimate of the *best* checkpoint.

    **Shrunk**: the best checkpoint's estimate pulled toward the pooled mean by the standard
    empirical-Bayes factor tau^2 / (tau^2 + se^2), where tau^2 is the between-checkpoint
    variance left after subtracting measurement error. This is the winner's-curse correction:
    if checkpoints genuinely differ (tau^2 large) the maximum is mostly real; if they do not
    (tau^2 ~ 0) the maximum is mostly the lucky tail of 7 draws and shrinks to the pooled mean.

    It is a *prediction* of what a confirmatory sample should find, not a substitute for one.
    """
    seed = baseline_seed(run)
    files = sorted(glob.glob(os.path.join(horizon_dir, f't3_{run}_on_s{seed}_m*.npz')))
    if not files:
        return None
    clean_npz = best_clean_reference(seed)
    series, labels, idx = [], [], None
    for f in files:
        a, f_idx = _honest_series(f)
        if idx is None:
            idx = f_idx
        elif not np.array_equal(idx, f_idx):
            raise ValueError(f'{f} has honest indices {f_idx}, expected {idx}; '
                             'pooling across different compromised sets is not meaningful')
        series.append(a)
        labels.append(re.search(r'_(m\d+M)\.npz$', f).group(1))
    # The clean reference must be restricted to the ATTACKED runs' honest drones. A clean run
    # has no Byzantine mask, so reading its own "honest" set gives all 8 -- comparing 7
    # attacked drones against 8 clean ones is the same systematic bias gate_tier3 was fixed
    # for (design/tier3_design.md 4.3).
    cln = _honest_rmse_per_episode(clean_npz, idx)

    degs = np.array([100.0 * (a.mean() - cln.mean()) / cln.mean() for a in series])
    pooled = welch_degradation(np.concatenate(series), cln)
    # Measurement SE of one checkpoint's degradation, in percentage points.
    n_each = float(np.mean([len(a) for a in series]))
    se_pp = 100.0 * np.sqrt(cln.var(ddof=1) / n_each + cln.var(ddof=1) / len(cln)) / cln.mean()
    observed_var = float(np.var(degs, ddof=1))
    tau2 = max(observed_var - se_pp ** 2, 0.0)
    weight = tau2 / (tau2 + se_pp ** 2) if (tau2 + se_pp ** 2) > 0 else 0.0
    best_i = int(np.argmax(degs))
    shrunk = pooled['deg_pct'] + weight * (degs[best_i] - pooled['deg_pct'])
    return dict(run=run, clean_npz=clean_npz, labels=labels, degs=degs, pooled=pooled,
                se_pp=se_pp, tau=np.sqrt(tau2), weight=weight,
                best_label=labels[best_i], best_deg=float(degs[best_i]), shrunk=float(shrunk))


def baseline_seed(run):
    """Seed 3 from `adv_seed3_stealth0_lieonly`; the baseline it attacks shares that seed."""
    m = re.search(r'seed(\d+)', run)
    if not m:
        raise ValueError(f'cannot parse baseline seed from {run!r}')
    return int(m.group(1))


def confirm_paths(run, milestone, n_episodes):
    seed = baseline_seed(run)
    tag = f'{run}_m{milestone:03d}_n{n_episodes}'
    return dict(
        seed=seed,
        adv_npz=os.path.join(CONFIRM_DIR, f'confirm_{tag}.npz'),
        # A clean reference at the SAME episode count, collected fresh so the confirmatory
        # sample is not compared against the 50-episode exploratory clean.
        clean_npz=os.path.join(CONFIRM_DIR, f'confirm_clean_s{seed}_n{n_episodes}.npz'),
        tag=tag,
    )


def collect_commands(run, milestone, n_episodes, wandb=True):
    p = confirm_paths(run, milestone, n_episodes)
    seed = p['seed']
    cfg = json.dumps({'checkpoint': f'train_dir/{run}', 'milestone': milestone * 1_000_000})
    clean_name = f'confirm_clean_s{seed}_n{n_episodes}'

    def wb(name):
        # Every rollout in this project is logged to wandb; the confirmatory stage is the one
        # whose numbers reach the paper, so it is the last place to skip it.
        return (f' --wandb --wandb-group tier3_confirm --wandb-name {name}'
                if wandb else '')

    adv = (f'{PY} -m byzantine_experiments.rollout.collect '
           f'--checkpoint train_dir/exp1_baseline_seed{seed} '
           f"--tier tier3_learned --tier-config '{cfg}' --compromised-indices 0 "
           f'--n-episodes {n_episodes} --seed {seed} --device cpu '
           f"--out {p['adv_npz']}" + wb(f"confirm_{p['tag']}"))
    clean = (f'{PY} -m byzantine_experiments.rollout.collect '
             f'--checkpoint train_dir/exp1_baseline_seed{seed} --no-byzantine '
             f'--n-episodes {n_episodes} --seed {seed} --device cpu '
             f"--out {p['clean_npz']}" + wb(clean_name))
    return adv, clean


def verdict(run, milestone, n_episodes):
    p = confirm_paths(run, milestone, n_episodes)
    if not os.path.exists(p['adv_npz']):
        return None, f"missing {p['adv_npz']} -- run the collect commands (--print-commands)"
    clean_npz = p['clean_npz'] if os.path.exists(p['clean_npz']) \
        else best_clean_reference(p['seed'])
    if not os.path.exists(clean_npz):
        return None, f'no clean reference for seed {p["seed"]}'
    # Paired, verified, no fallback: the confirmatory number is the one that decides the gate,
    # so an unpaired estimate here would be the worst possible place for one to slip through.
    from byzantine_experiments.analysis.paired import PairingError, paired_degradation
    try:
        stats = paired_degradation(p['adv_npz'], clean_npz)
    except PairingError as e:
        return None, (f'NOT PAIRED, no verdict: {e}. Re-collect both arms with the current '
                      'collector at the same --seed (bash scripts/byz_confirm_adv.sh ...).')
    stats['clean_npz'] = clean_npz
    return stats, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True, help='adversary run dir name under train_dir/')
    ap.add_argument('--clean-audit', type=int, metavar='SEED', default=None,
                    help='compare every clean reference available for a baseline seed and exit')
    ap.add_argument('--pooled', action='store_true',
                    help='report the unbiased pooled estimate for an exploratory sweep and the '
                         "winner's-curse-corrected estimate of its best checkpoint, then exit")
    ap.add_argument('--milestone', type=int, default=None,
                    help='env-step milestone in millions, chosen on the exploratory sweep')
    ap.add_argument('--n-episodes', type=int, default=200)
    ap.add_argument('--print-commands', action='store_true',
                    help='print the two collect commands and exit (user runs them)')
    args = ap.parse_args()

    os.makedirs(CONFIRM_DIR, exist_ok=True)

    if args.clean_audit is not None:
        from scipy import stats as _st
        refs = clean_audit(args.clean_audit)
        if len(refs) < 1:
            print(f'no clean references for seed {args.clean_audit}')
            return
        print(f'Clean references for baseline seed {args.clean_audit} '
              '(honest goal RMSE, drones 1-7):')
        for name, v in refs:
            print(f'  {name:36s} n={len(v):4d}  mean {v.mean():.4f}  '
                  f'sd {v.std(ddof=1):.4f}  sem {v.std(ddof=1)/np.sqrt(len(v)):.4f}')
        if len(refs) > 1:
            print('\n  pairwise Welch tests:')
            for i in range(len(refs)):
                for j in range(i + 1, len(refs)):
                    (na, a), (nb, b) = refs[i], refs[j]
                    t, pv = _st.ttest_ind(a, b, equal_var=False)
                    d = 100.0 * (b.mean() - a.mean()) / a.mean()
                    flag = 'DIFFERENT' if pv < 0.05 else 'consistent'
                    print(f'    {na} vs {nb}: {d:+.1f} %  p={pv:.4f}  {flag}')
            allv = np.concatenate([v for _, v in refs])
            print(f'\n  pooled over every reference: n={len(allv)}  mean {allv.mean():.4f}  '
                  f'sem {allv.std(ddof=1)/np.sqrt(len(allv)):.4f}')
            print('  A degradation measured against the wrong reference is wrong by the same '
                  'percentage. Settle this before quoting any number on this seed.')
        return

    if args.pooled:
        r = pooled_and_shrunk(args.run)
        if r is None:
            print(f'no horizon sidecars found for {args.run}')
            return
        pl = r['pooled']
        print(f"{args.run}  (clean reference: {os.path.basename(r['clean_npz'])}, "
              f"n={pl['n_clean']})")
        print('  per-checkpoint: ' + '  '.join(
            f'{lab}={d:+.1f}' for lab, d in zip(r['labels'], r['degs'])))
        print(f"  POOLED (no selection, n={pl['n_attacked']}): {pl['deg_pct']:+.1f} % "
              f"[{pl['ci_lo']:+.1f}, {pl['ci_hi']:+.1f}]  p={pl['p_value']:.4f}")
        print(f"  per-checkpoint measurement SE: {r['se_pp']:.1f} pp;  "
              f"real between-checkpoint sd (tau): {r['tau']:.1f} pp")
        print(f"  best checkpoint {r['best_label']} = {r['best_deg']:+.1f} %  ->  "
              f"shrunk estimate {r['shrunk']:+.1f} % "
              f"(weight {r['weight']:.2f} on the selected value)")
        print('  The shrunk figure is what a confirmatory sample should be expected to find.')
        return

    if args.milestone is None:
        ap.error('--milestone is required unless --pooled is given')

    if args.print_commands:
        adv, clean = collect_commands(args.run, args.milestone, args.n_episodes)
        # Observed: 50 episodes took ~18 min with 7 rollouts sharing the box, i.e. ~21 s per
        # episode. A single collect process is not internally parallel, so scale linearly and
        # run the two commands CONCURRENTLY rather than back to back.
        mins = int(args.n_episodes * 21 / 60)
        print(f'# Confirmatory sample for {args.run} @ {args.milestone}M, n={args.n_episodes}')
        print(f'# ~{mins} min EACH; run them concurrently (separate shells) -- they are '
              f'independent single-process jobs.')
        print('# 1. adversary:\n' + adv)
        print('\n# 2. matched clean reference:\n' + clean)
        print(f'\n# then: python -m byzantine_experiments.analysis.confirm_tier3 '
              f'--run {args.run} --milestone {args.milestone} --n-episodes {args.n_episodes}')
        return

    stats, err = verdict(args.run, args.milestone, args.n_episodes)
    if err:
        print(err)
        return
    # Survival is part of the gate (gate_tier3.GATE['survival']); read it from the attacked run.
    adv_json = confirm_paths(args.run, args.milestone, args.n_episodes)['adv_npz'][:-4] + '.json'
    with open(adv_json) as fh:
        survival = float(json.load(fh)['metrics'].get('attacker_survived_mean', float('nan')))
    passes = (stats['deg_pct'] >= GATE_PCT and stats['ci_lo'] > 0.0
              and survival >= GATE['survival'])
    # `paired_degradation` reports one count, n_pairs; the old unpaired path had separate
    # n_attacked / n_clean, and this print still asked for them (KeyError after a 62-min run).
    print(f'Confirmatory result: {args.run} @ {args.milestone}M, '
          f'{stats["n_pairs"]} paired episodes')
    print(f'  paired over {stats["n_pairs"]} scenarios, correlation between arms '
          f'{stats["corr"]:+.2f}, SE shrink {stats["se_shrink"]:.1f}x')
    print(f'  honest goal-RMSE degradation : {stats["deg_pct"]:+.1f} % '
          f'[{stats["ci_lo"]:+.1f}, {stats["ci_hi"]:+.1f}]  p={stats["p_value"]:.4f}')
    print(f'  adversary survival           : {100 * survival:.0f} % '
          f'(gate needs >= {100 * GATE["survival"]:.0f} %)')
    print(f'  gate (>= {GATE_PCT:.0f} %, CI excludes 0, survival): '
          f'{"PASS" if passes else "fail"}')


if __name__ == '__main__':
    main()
