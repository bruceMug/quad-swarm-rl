"""
Aggregate the Tier 2 rollout matrix into its severity calibration table.

Same shape as the Tier 1 aggregator, plus the three metrics the Tier 2 spec adds:
attacker survival, victim spillover, and time-to-impact. It also computes **matched-damage
pairs** against the Tier 1 canonical settings, which is what makes the paper's central
comparison clean: same damage, different mechanism, does the detector tell them apart?

Decision D1 (2026-09-26): the spec's 20-40 % band is retired -- it was set on legacy numbers the
seeding bug inflated and no v2 config reaches it. A Tier 2 attack family is validated when its
runs together significantly damage the honest team on the PAIRED metric
(`family_validation.py`). The matched-damage pairs below use legacy Tier 1 numbers and are kept
for the record only; the paper's matched comparison is `matched_triple.py` (paired, TOST).

Clean references come from the Tier 1 results directory by default. Those runs used the same
checkpoints, the same collector and the same 50 episodes, so re-collecting them for Tier 2
would produce identical numbers at the cost of 45 minutes.

Usage:
    python -m byzantine_experiments.analysis.aggregate_tier2 [--no-wandb]
"""

from byzantine_experiments.paths import RESULTS_ROOT, results_dir
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from byzantine_experiments.analysis.family_validation import markdown as family_markdown
from byzantine_experiments.analysis.family_validation import validated
from byzantine_experiments.analysis.metrics import degradation_vs_clean
from byzantine_experiments.analysis.paired import paired_columns
from byzantine_experiments.analysis.wandb_logging import log_summary_table

RESULTS_DIR = results_dir('tier2')
CLEAN_DIR = results_dir('tier1')
OUT_CSV = os.path.join(RESULTS_ROOT, 'tier2_calibration.csv')
OUT_MD = os.path.join(RESULTS_ROOT, 'tier2_calibration.md')

# The 20-40 % band is retired (D1); matched-triple candidates are validated families at this f.
TRIPLE_F = 0.125

NAME_RE = re.compile(
    r'^t2_(?P<variant>signflip|waypoint|liar|collusion)_s(?P<seed>\d+)'
    r'_(?P<severity>.+?)_f(?P<f>\d+)$'
)
CLEAN_RE = re.compile(r'^t1_clean_s(?P<seed>\d+)$')

# Tier 1 canonical settings, for matched-damage pairing. Values from
# design/tier1_design.md section 4 (goal-RMSE degradation of the compromised drone).
TIER1_CANONICAL = {
    '1A gaussian sigma=0.25': dict(goal_rmse_deg=23.6, action_dev=0.467),
    '1B ou sigma_ou=0.06': dict(goal_rmse_deg=35.3, action_dev=0.213),
    "1C' stall d0.25/ms0.3": dict(goal_rmse_deg=25.5, action_dev=0.089),
}


def load_runs(results_dir=RESULTS_DIR, clean_dir=CLEAN_DIR):
    byz, clean = [], {}

    for path in sorted(glob.glob(os.path.join(clean_dir, 't1_clean_s*.json'))):
        m = CLEAN_RE.match(os.path.splitext(os.path.basename(path))[0])
        if not m:
            continue
        with open(path) as f:
            sidecar = json.load(f)
        if 'metrics' in sidecar:
            clean[int(m.group('seed'))] = sidecar['metrics']

    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        m = NAME_RE.match(stem)
        if not m:
            print(f'  ! skipping unrecognised name: {stem}')
            continue
        with open(path) as f:
            sidecar = json.load(f)
        if 'metrics' not in sidecar:
            print(f'  ! skipping {stem}: no metrics block')
            continue
        d = m.groupdict()
        byz.append(dict(stem=stem, variant=d['variant'], seed=int(d['seed']),
                        severity=d['severity'], f_label=d['f'],
                        sidecar=sidecar, metrics=sidecar['metrics']))
    return byz, clean


def guard_rate(sidecar):
    """Mean survival-guard trigger rate across this run's attackers."""
    diag = sidecar.get('agent_diagnostics') or {}
    rates = [v.get('guard_trigger_rate') for v in diag.values()
             if isinstance(v, dict) and v.get('guard_trigger_rate') is not None]
    return float(np.mean(rates)) if rates else float('nan')


def build_rows(byz_runs, clean_by_seed):
    rows = []
    for run in byz_runs:
        seed = run['seed']
        if seed not in clean_by_seed:
            print(f"  ! no clean reference for seed {seed}; skipping {run['stem']}")
            continue
        m = run['metrics']
        deg = degradation_vs_clean(m, clean_by_seed[seed])
        # The only reportable degradation in the v2 corpus: paired against the clean run at
        # the same seed (design 6.18). The unpaired columns below are kept for comparison.
        pc = paired_columns(os.path.join(RESULTS_DIR, run['stem'] + '.npz'),
                            os.path.join(results_dir('tier1'), f't1_clean_s{seed}.npz'))
        rows.append(dict(
            **pc,
            config=f"{run['variant']}_{run['severity']}_f{run['f_label']}",
            variant=run['variant'], severity=run['severity'],
            f_label=run['f_label'], f_fraction=m['f_fraction'],
            n_byzantine=m['n_byzantine'], seed=seed, n_episodes=m['n_episodes'],

            episode_reward=m['episode_reward_mean'],
            clean_episode_reward=deg['clean_episode_reward'],
            reward_degradation_pct=deg['reward_degradation_pct'],
            reward_degradation_pct_sem=deg['reward_degradation_pct_sem'],

            goal_rmse=m['goal_rmse_mean'],
            goal_rmse_degradation_pct=deg['goal_rmse_degradation_pct'],
            goal_rmse_byzantine=m['goal_rmse_byzantine_mean'],
            goal_rmse_honest=m['goal_rmse_honest_mean'],

            # Tier 2 additions
            victim_spillover_pct=deg.get('victim_spillover_pct', float('nan')),
            victim_spillover_goal_rmse_pct=deg.get('victim_spillover_goal_rmse_pct', float('nan')),
            attacker_survival_rate=m.get('attacker_survived_mean', float('nan')),
            attacker_airborne_frac=m.get('attacker_airborne_frac_mean', float('nan')),
            # Derived from the trajectory (fraction of steps below guard_altitude), which is
            # the guard's firing condition. The agent's own counter is also read, but it is
            # zeroed by the final episode reset in runs collected before that was fixed.
            guard_exposure=m.get('attacker_below_guard_frac_mean', float('nan')),
            guard_trigger_rate=guard_rate(run['sidecar']),
            time_to_impact_s=deg.get('time_to_impact_s', float('nan')),

            collisions=m['collisions_mean'],
            collision_delta=deg['collisions_delta'],
            action_deviation=m['action_deviation_byzantine_mean'],
            saturation_rate=m['saturation_rate_byzantine'],
            min_separation=m['min_separation_mean'],
            low_altitude_frac=m['low_altitude_frac_mean'],
        ))
    return rows


AGG_KEYS = ['episode_reward', 'clean_episode_reward', 'reward_degradation_pct',
            'goal_rmse', 'goal_rmse_degradation_pct', 'goal_rmse_byzantine',
            'goal_rmse_honest', 'victim_spillover_pct', 'victim_spillover_goal_rmse_pct',
            'attacker_survival_rate', 'attacker_airborne_frac', 'guard_exposure',
            'guard_trigger_rate',
            'time_to_impact_s', 'collisions', 'collision_delta', 'action_deviation',
            'saturation_rate', 'min_separation', 'low_altitude_frac']


def aggregate_over_seeds(rows):
    by_config = defaultdict(list)
    for r in rows:
        by_config[r['config']].append(r)

    out = []
    for config, group in by_config.items():
        first = group[0]
        agg = dict(config=config, variant=first['variant'], severity=first['severity'],
                   f_label=first['f_label'], f_fraction=first['f_fraction'],
                   n_seeds=len(group),
                   n_episodes_total=sum(r['n_episodes'] for r in group))
        for key in AGG_KEYS:
            vals = np.array([r[key] for r in group], dtype=float)
            vals = vals[np.isfinite(vals)]
            agg[key] = float(vals.mean()) if vals.size else float('nan')
            agg[f'{key}_seed_std'] = float(vals.std()) if vals.size else float('nan')
        sems = np.array([r['reward_degradation_pct_sem'] for r in group], dtype=float)
        sems = sems[np.isfinite(sems)]
        agg['reward_degradation_pct_sem'] = (float(np.sqrt((sems ** 2).sum()) / len(sems))
                                             if sems.size else float('nan'))
        # Paired honest degradation, averaged over the seeds whose pairing verified.
        pd = np.array([r['paired_honest_deg_pct'] for r in group if r.get('paired')], dtype=float)
        agg['paired_honest_deg_pct'] = float(pd.mean()) if pd.size else float('nan')
        agg['paired_seeds'] = int(pd.size)
        out.append(agg)
    fams = validated(rows)
    for agg in out:
        agg['family_validated'] = agg['variant'] in fams
        agg['triple_candidate'] = bool(agg['family_validated']
                                       and abs(agg['f_fraction'] - TRIPLE_F) < 1e-9)
    return sorted(out, key=lambda r: (r['variant'], r['f_fraction'], r['severity']))


def matched_pairs(agg):
    """Closest Tier 2 config to each Tier 1 canonical setting, by goal-RMSE degradation."""
    pairs = []
    for label, t1 in TIER1_CANONICAL.items():
        best, best_gap = None, np.inf
        for r in agg:
            g = r['goal_rmse_degradation_pct']
            if not np.isfinite(g):
                continue
            gap = abs(g - t1['goal_rmse_deg'])
            if gap < best_gap:
                best, best_gap = r, gap
        if best is not None:
            pairs.append(dict(tier1=label, tier1_goal_rmse_deg=t1['goal_rmse_deg'],
                              tier1_action_dev=t1['action_dev'],
                              tier2=best['config'],
                              tier2_goal_rmse_deg=best['goal_rmse_degradation_pct'],
                              tier2_action_dev=best['action_deviation'],
                              gap_pp=best_gap,
                              tier2_spillover_pct=best['victim_spillover_goal_rmse_pct']))
    return pairs


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f'{x:.{nd}f}'


def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {path}  ({len(rows)} rows)')


def write_markdown(agg, per_seed, pairs, path):
    L = []
    L.append('# Tier 2 — severity calibration table\n')
    L.append(f'Generated from `{RESULTS_DIR}/*.json`. {len(per_seed)} rollout runs over '
             f'{len({r["seed"] for r in per_seed})} baseline seeds, 50 episodes each. Clean '
             f'references reused from `{CLEAN_DIR}` (same checkpoints, same collector).\n')
    L.append('Values are means over seeds. Read `reward deg %` together with its SEM — see '
             'the Tier 1 note; goal-RMSE degradation is the axis that can actually rank '
             'severities.\n')

    L.append('| config | f | **paired honest deg %** | goal RMSE deg % | **spillover %** '
             '| reward deg % (±SEM) | **survival** | guard rate | **t-to-impact s** | action dev '
             '| collisions Δ |')
    L.append('|' + '---|' * 11)
    for r in agg:
        star = ' **†**' if r['triple_candidate'] else ''
        L.append(
            f"| `{r['variant']}` {r['severity']}{star} | {r['f_fraction']:.3f} "
            f"| {fmt(r['paired_honest_deg_pct'], 1)} ({r['paired_seeds']} seeds) "
            f"| {fmt(r['goal_rmse_degradation_pct'], 1)} "
            f"| {fmt(r['victim_spillover_goal_rmse_pct'], 1)} "
            f"| {fmt(r['reward_degradation_pct'], 0)} ± {fmt(r['reward_degradation_pct_sem'], 0)} "
            f"| {fmt(100 * r['attacker_survival_rate'], 0)}% "
            f"| {fmt(100 * r['guard_exposure'], 0)}% "
            f"| {fmt(r['time_to_impact_s'], 1)} "
            f"| {fmt(r['action_deviation'], 3)} | {fmt(r['collision_delta'], 2)} |")

    L.append(f'\n† = matched-triple candidate: family validated below, f = {TRIPLE_F}. '
             '(Replaces the retired 20–40 % band ★; decision D1.) **paired honest deg %** is '
             'the only reportable degradation in the v2 corpus (design 6.18).\n')
    L.append('**spillover %** = degradation of the *honest* drones\' goal RMSE. This is the '
             'column that distinguishes Tier 2 from Tier 1: Tier 1 never exceeded ~10 % here.\n')
    L.append('**survival** = fraction of episodes the attacker is still airborne at the end. '
             'A low number means the attack removes itself from the episode.\n')
    L.append('**guard rate** = fraction of steps the attacker spent below the survival guard\'s '
             '0.6 m threshold, i.e. steps on which the guard blended it back toward honest. '
             'Derived from the trajectory, so it is exact regardless of the agent counters. '
             'A high number means the "attack" is substantially honest flight.\n')
    L.append('**t-to-impact** = seconds until cumulative team reward departs from the clean '
             'run by more than 2 clean σ, sustained. `n/a` = never separated.\n')

    L.append(family_markdown(per_seed, 'Family-level validation (decision D1, replaces the '
                             '20–40 % band)'))
    L.append('\n## Matched-damage pairs against Tier 1 (legacy, superseded)\n')
    L.append('Kept for the record: these use the legacy (unpaired, seeding-bug) Tier 1 canonical '
             'numbers. The paper\'s *same damage, different mechanism* comparison is the '
             'pre-registered matched triple (`analysis/matched_triple.py`, paired TOST).\n')
    L.append('| Tier 1 canonical | its goal-RMSE deg | closest Tier 2 config | its deg | gap | '
             'T1 action dev | T2 action dev | T2 spillover |')
    L.append('|' + '---|' * 8)
    for p in pairs:
        L.append(f"| {p['tier1']} | {fmt(p['tier1_goal_rmse_deg'], 1)} % | `{p['tier2']}` "
                 f"| {fmt(p['tier2_goal_rmse_deg'], 1)} % | {fmt(p['gap_pp'], 1)} pp "
                 f"| {fmt(p['tier1_action_dev'], 3)} | {fmt(p['tier2_action_dev'], 3)} "
                 f"| {fmt(p['tier2_spillover_pct'], 1)} % |")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'Wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--results-dir', default=RESULTS_DIR)
    p.add_argument('--clean-dir', default=CLEAN_DIR)
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--wandb-name', default='tier2_calibration')
    args = p.parse_args()

    byz, clean = load_runs(args.results_dir, args.clean_dir)
    print(f'Loaded {len(byz)} Tier 2 runs and {len(clean)} clean references '
          f'(seeds {sorted(clean)})')
    if not byz:
        raise SystemExit('No Tier 2 runs found. Run scripts/byz_rollout_tier2.sh first.')

    per_seed = build_rows(byz, clean)
    agg = aggregate_over_seeds(per_seed)
    pairs = matched_pairs(agg)

    write_csv(per_seed, OUT_CSV)
    write_markdown(agg, per_seed, pairs, OUT_MD)

    fams = sorted(validated(per_seed))
    print(f'\n{len(agg)} configs; validated families (paired, family p < 0.05, damaging): '
          f"{', '.join(fams) or 'none'}")
    for r in agg:
        star = '+' if r['triple_candidate'] else ' '
        print(f"{star} {r['config']:38s} paired={fmt(r['paired_honest_deg_pct'],1):>6s}% goal_deg={fmt(r['goal_rmse_degradation_pct'],1):>7s}% "
              f"spill={fmt(r['victim_spillover_goal_rmse_pct'],1):>7s}% "
              f"surv={fmt(100*r['attacker_survival_rate'],0):>4s}% "
              f"guard={fmt(100*r['guard_exposure'],0):>4s}% "
              f"tti={fmt(r['time_to_impact_s'],1):>5s}s")
    print('+ = matched-triple candidate (validated family, f = 0.125)')
    print(family_markdown(per_seed))

    if not args.no_wandb:
        url = log_summary_table(
            rows=agg, run_name=args.wandb_name, group='tier2_analysis',
            tags=['tier2', 'calibration', 'analysis'],
            extra_summary={'n_configs': len(agg), 'n_runs': len(per_seed),
                           'validated_families': ','.join(fams),
                           'n_validated_families': len(fams)},
            artifacts=[OUT_CSV, OUT_MD],
        )
        if url:
            print(f'\nwandb: {url}')


if __name__ == '__main__':
    main()
