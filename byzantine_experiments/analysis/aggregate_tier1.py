"""
Aggregate the Tier 1 rollout matrix into the severity calibration table.

Reads only the small `.json` sidecars written by collect.py (the metrics are in there, so
the 250 MB `.npz` files stay untouched), pairs each Byzantine config with the clean
reference for the *same baseline seed*, averages over seeds, and emits:

    byzantine_experiments/results/tier1_calibration.csv    per-config x per-seed, raw
    byzantine_experiments/results/tier1_calibration.md     the seed-averaged paper table
    a wandb "analysis" run holding both as a queryable Table + uploaded files

Why degradation is reported two ways
------------------------------------
The spec asks for "team reward degradation %" and for the configs landing in a 20-40 % band.
That number is computed and reported. But mean episode reward on this task is ~1.5 with a
per-episode std of ~3.5 and a heavy left tail, so `(clean - byz) / |clean|` divides by a
near-zero denominator: at 50 episodes its standard error alone is on the order of +-50
percentage points. Every reward degradation figure therefore carries its SEM and a sigma,
and a second severity axis -- **goal RMSE degradation** -- is reported alongside. Goal RMSE
(RMS distance to the assigned goal, i.e. formation error) has a true zero, is bounded below,
and its SEM is a few percent, which makes it the metric that can actually rank severities.

Usage:
    python -m byzantine_experiments.analysis.aggregate_tier1 [--no-wandb]
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from byzantine_experiments.analysis.metrics import degradation_vs_clean
from byzantine_experiments.analysis.wandb_logging import log_summary_table

RESULTS_DIR = 'byzantine_experiments/results/tier1'
OUT_CSV = 'byzantine_experiments/results/tier1_calibration.csv'
OUT_MD = 'byzantine_experiments/results/tier1_calibration.md'

# The band the spec wants flagged: these configs are the ones Tier 3's adversary must be
# tuned to match, so that the detector is separating malicious from degraded rather than
# high-damage from low-damage.
BAND_LO, BAND_HI = 20.0, 40.0

NAME_RE = re.compile(
    r'^t1_(?P<variant>clean|gaussian|ou|dropout|stall)_s(?P<seed>\d+)'
    r'(?:_(?P<severity>sigma[\d.]+|sigmaou[\d.]+|p[\d.]+_(?:hold|zero)'
    r'|d[\d.]+_ms[\d.]+_(?:hold|zero)))?'
    r'(?:_f(?P<f>\d+))?$'
)


def parse_name(stem):
    m = NAME_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()
    d['seed'] = int(d['seed'])
    return d


def load_runs(results_dir=RESULTS_DIR):
    """-> (byz_runs, clean_by_seed). Each run is the sidecar dict plus parsed name fields."""
    byz, clean = [], {}
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        parsed = parse_name(stem)
        if parsed is None:
            print(f'  ! skipping unrecognised file name: {stem}')
            continue
        with open(path) as f:
            sidecar = json.load(f)
        if 'metrics' not in sidecar:
            print(f'  ! skipping {stem}: no metrics block (collected before metrics existed?)')
            continue
        record = dict(parsed, stem=stem, path=path, sidecar=sidecar, metrics=sidecar['metrics'])
        if parsed['variant'] == 'clean':
            clean[parsed['seed']] = record
        else:
            byz.append(record)
    return byz, clean


def severity_value(variant, severity):
    """
    Numeric severity for sorting/plotting: sigma, sigma_ou, p_drop, or (for tier1_stall)
    the duty cycle -- which is why stall names put `d` before `ms`, so the leading number is
    the one severities should sort on.
    """
    if severity is None:
        return float('nan')
    m = re.search(r'([\d.]+)', severity)
    return float(m.group(1)) if m else float('nan')


def build_rows(byz_runs, clean_by_seed):
    """One row per (config, seed), with degradation measured against that seed's clean run."""
    rows = []
    for run in byz_runs:
        seed = run['seed']
        if seed not in clean_by_seed:
            print(f"  ! no clean reference for seed {seed}; skipping {run['stem']}")
            continue
        m = run['metrics']
        deg = degradation_vs_clean(m, clean_by_seed[seed]['metrics'])
        cfg = run['sidecar'].get('tier_config', {})

        rows.append(dict(
            config=f"{run['variant']}_{run['severity']}_f{run['f']}",
            variant=run['variant'],
            severity=run['severity'],
            severity_value=severity_value(run['variant'], run['severity']),
            mode=cfg.get('mode', ''),
            f_label=run['f'],
            f_fraction=m['f_fraction'],
            n_byzantine=m['n_byzantine'],
            seed=seed,
            n_episodes=m['n_episodes'],

            episode_reward=m['episode_reward_mean'],
            episode_reward_sem=m['episode_reward_sem'],
            episode_reward_honest=m['episode_reward_honest_mean'],
            clean_episode_reward=deg['clean_episode_reward'],
            reward_drop_absolute=deg['reward_drop_absolute'],
            reward_degradation_pct=deg['reward_degradation_pct'],
            reward_degradation_pct_sem=deg['reward_degradation_pct_sem'],
            reward_drop_sigma=deg['reward_drop_sigma'],

            goal_rmse=m['goal_rmse_mean'],
            goal_rmse_sem=m['goal_rmse_sem'],
            goal_rmse_honest=m['goal_rmse_honest_mean'],
            goal_rmse_byzantine=m['goal_rmse_byzantine_mean'],
            clean_goal_rmse=deg['clean_goal_rmse'],
            goal_rmse_degradation_pct=deg['goal_rmse_degradation_pct'],
            goal_rmse_honest_degradation_pct=deg['goal_rmse_honest_degradation_pct'],

            collisions=m['collisions_mean'],
            clean_collisions=deg['clean_collisions'],
            collision_delta=deg['collisions_delta'],

            action_deviation=m['action_deviation_byzantine_mean'],
            action_deviation_max=m['action_deviation_byzantine_max'],
            saturation_rate=m['saturation_rate_byzantine'],
            saturation_rate_honest=m['saturation_rate_honest'],
            perturbed_step_fraction=m['perturbed_step_fraction'],
            min_separation=m['min_separation_mean'],
            low_altitude_frac=m['low_altitude_frac_mean'],
        ))
    return rows


AGG_KEYS = [
    'episode_reward', 'clean_episode_reward', 'reward_drop_absolute',
    'reward_degradation_pct', 'reward_drop_sigma',
    'goal_rmse', 'goal_rmse_honest', 'goal_rmse_byzantine',
    'goal_rmse_degradation_pct', 'goal_rmse_honest_degradation_pct',
    'collisions', 'collision_delta', 'action_deviation', 'action_deviation_max',
    'saturation_rate', 'saturation_rate_honest', 'perturbed_step_fraction',
    'min_separation', 'low_altitude_frac',
]


def aggregate_over_seeds(rows):
    """Average each config across the 3 baseline seeds; std across seeds is the error bar."""
    by_config = defaultdict(list)
    for r in rows:
        by_config[r['config']].append(r)

    out = []
    for config, group in by_config.items():
        first = group[0]
        agg = dict(
            config=config, variant=first['variant'], severity=first['severity'],
            severity_value=first['severity_value'], mode=first['mode'],
            f_label=first['f_label'], f_fraction=first['f_fraction'],
            n_seeds=len(group), seeds=','.join(str(r['seed']) for r in sorted(group, key=lambda x: x['seed'])),
            n_episodes_total=sum(r['n_episodes'] for r in group),
        )
        for key in AGG_KEYS:
            vals = np.array([r[key] for r in group], dtype=float)
            vals = vals[np.isfinite(vals)]
            agg[key] = float(vals.mean()) if vals.size else float('nan')
            agg[f'{key}_seed_std'] = float(vals.std()) if vals.size else float('nan')
        # Pooled SEM of the reward-degradation percentage across all seeds' episodes.
        sems = np.array([r['reward_degradation_pct_sem'] for r in group], dtype=float)
        sems = sems[np.isfinite(sems)]
        agg['reward_degradation_pct_sem'] = (float(np.sqrt((sems ** 2).sum()) / len(sems))
                                             if sems.size else float('nan'))
        agg['in_20_40_band'] = bool(BAND_LO <= agg['reward_degradation_pct'] <= BAND_HI)
        agg['in_20_40_band_goal_rmse'] = bool(BAND_LO <= agg['goal_rmse_degradation_pct'] <= BAND_HI)
        out.append(agg)

    return sorted(out, key=lambda r: (r['variant'], r['f_fraction'], r['severity_value'], r['mode']))


def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys = list(rows[0])
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {path}  ({len(rows)} rows)')


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f'{x:.{nd}f}'


def write_markdown(agg, per_seed, path):
    lines = []
    lines.append('# Tier 1 — severity calibration table\n')
    lines.append(f'Generated from `{RESULTS_DIR}/*.json`. '
                 f'{len(per_seed)} rollout runs over {len({r["seed"] for r in per_seed})} baseline seeds, '
                 f'{per_seed[0]["n_episodes"] if per_seed else 0} episodes each.\n')
    lines.append('Each Byzantine config is compared against the clean rollout of the **same** '
                 'baseline checkpoint, collected with the same collector. Values are means over '
                 'seeds; `+-` is the spread across seeds.\n')
    lines.append('> **Read the two degradation columns together.** `reward deg %` is the figure '
                 'the spec asks for, but mean episode reward is ~1.5 with a per-episode std of '
                 '~3.5 and a heavy left tail, so its percentage form divides by a near-zero '
                 'denominator and is extremely noisy (see the `+-SEM` and `sigma` columns). '
                 '`goal RMSE deg %` — RMS distance to assigned goal, i.e. formation error — has '
                 'a true zero and a few-percent SEM, and is the axis that actually ranks '
                 'severities.\n')

    header = ('| config | f | reward | reward deg % | ±SEM | σ | goal RMSE | goal RMSE deg % | '
              'honest goal RMSE deg % | collisions Δ | action dev | saturation | min sep |')
    sep = '|' + '---|' * 13
    lines.append(header)
    lines.append(sep)

    for r in agg:
        flag = ' **★**' if r['in_20_40_band'] else ''
        lines.append(
            f"| `{r['variant']}` {r['severity']}{flag} | {r['f_fraction']:.3f} "
            f"| {fmt(r['episode_reward'])} | {fmt(r['reward_degradation_pct'], 1)} "
            f"| {fmt(r['reward_degradation_pct_sem'], 1)} | {fmt(r['reward_drop_sigma'], 1)} "
            f"| {fmt(r['goal_rmse'], 3)} | {fmt(r['goal_rmse_degradation_pct'], 1)} "
            f"| {fmt(r['goal_rmse_honest_degradation_pct'], 1)} "
            f"| {fmt(r['collision_delta'], 2)} | {fmt(r['action_deviation'], 3)} "
            f"| {fmt(100 * r['saturation_rate'], 1)}% | {fmt(r['min_separation'], 3)} |")

    lines.append('\n★ = mean reward degradation falls in the 20–40 % band.\n')

    for label, key in (('reward', 'in_20_40_band'), ('goal RMSE', 'in_20_40_band_goal_rmse')):
        hits = [r for r in agg if r[key]]
        lines.append(f'\n## Configs in the 20–40 % band by **{label}** degradation\n')
        if not hits:
            lines.append('_None._\n')
            continue
        for r in hits:
            pct = r['reward_degradation_pct'] if key == 'in_20_40_band' else r['goal_rmse_degradation_pct']
            lines.append(f"- `{r['variant']}` {r['severity']} at f={r['f_fraction']:.3f} "
                         f"— {fmt(pct, 1)} %")
        lines.append('')

    lines.append('\n## Saturation check\n')
    lines.append('The spec flags >~30 % clip saturation as disqualifying for a clean '
                 '"unbiased noise" control.\n')
    lines.append('| config | f | saturation rate |')
    lines.append('|---|---|---|')
    for r in agg:
        mark = ' ⚠️' if r['saturation_rate'] > 0.30 else ''
        lines.append(f"| `{r['variant']}` {r['severity']} | {r['f_fraction']:.3f} "
                     f"| {fmt(100 * r['saturation_rate'], 1)}%{mark} |")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--results-dir', default=RESULTS_DIR)
    p.add_argument('--no-wandb', action='store_true')
    p.add_argument('--wandb-name', default='tier1_calibration')
    args = p.parse_args()

    byz, clean = load_runs(args.results_dir)
    print(f'Loaded {len(byz)} Byzantine runs and {len(clean)} clean references '
          f'(seeds {sorted(clean)})')
    if not byz:
        raise SystemExit('No Byzantine runs found. Run scripts/byz_rollout_tier1.sh first.')

    per_seed = build_rows(byz, clean)
    agg = aggregate_over_seeds(per_seed)

    write_csv(per_seed, OUT_CSV)
    write_markdown(agg, per_seed, OUT_MD)

    print(f'\n{len(agg)} configs; '
          f"{sum(r['in_20_40_band'] for r in agg)} in the 20-40% reward band, "
          f"{sum(r['in_20_40_band_goal_rmse'] for r in agg)} in the band by goal RMSE")
    for r in agg:
        star = '*' if r['in_20_40_band'] else ' '
        print(f"{star} {r['config']:34s} rew_deg={fmt(r['reward_degradation_pct'],1):>8s}% "
              f"±{fmt(r['reward_degradation_pct_sem'],1):>6s}  "
              f"goal_deg={fmt(r['goal_rmse_degradation_pct'],1):>7s}%  "
              f"sat={fmt(100*r['saturation_rate'],1):>5s}%")

    if not args.no_wandb:
        url = log_summary_table(
            rows=agg, run_name=args.wandb_name, group='tier1_analysis',
            tags=['tier1', 'calibration', 'analysis'],
            extra_summary={
                'n_configs': len(agg),
                'n_runs': len(per_seed),
                'n_in_reward_band': sum(r['in_20_40_band'] for r in agg),
                'n_in_goal_rmse_band': sum(r['in_20_40_band_goal_rmse'] for r in agg),
            },
            artifacts=[OUT_CSV, OUT_MD],
        )
        if url:
            print(f'\nwandb: {url}')


if __name__ == '__main__':
    main()
