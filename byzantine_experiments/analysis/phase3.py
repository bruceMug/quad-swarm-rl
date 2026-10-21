"""
Phase 3 rollouts -- the spec's priority subset ("Rollout collection for Phase 3"):

    the matched triple  x  distribution shift {none, mild, severe}  x  baseline seeds 1-3,
    50 episodes each, plus one clean run per (seed, shift level).

Since the 2026-10-12 amendment (analysis/shift_calibration.py) there are four shift levels:
`mild` / `severe` from the amended grid (primary) and `mild_v1` / `severe_v1` from the original
grid. All four are collected and all are reported, ordered by their measured effect on the
clean swarm -- a dose-response, not a choice.

At n = 50 per seed the pooled damage of one member at one level has a 95 % half-width of ~5 pp,
so the table describes; it cannot show a ~3 % attack "disappearing" under shift.

Every attacked run pairs with the clean run at the SAME seed and the SAME shift level;
`paired.check_pairing` refuses any other pairing (DR and wind do not change spawns, so the
spawn check alone would not catch a cross-shift pair).

Members are read from `matched_triple_selection.json` and never re-selected here; the mild and
severe levels are read from `shift_levels.json` (analysis/shift_calibration.py). So this runs
after both `byz_matched_triple.sh` and `byz_calibrate_shift.sh`.

`--cross-seed` adds, per baseline seed, the Tier 3 adversaries trained on the OTHER seeds (the
spec's factor "adversary: matching-seed and cross-seed"). The remaining spec factors (stealth
0 vs constrained, f = 0.25) are "if time permits" and are not generated here.

    python -m byzantine_experiments.analysis.phase3 jobs [--cross-seed]   # TSV for the script
    python -m byzantine_experiments.analysis.phase3 table                  # damage per member x shift
"""

import argparse
import glob
import json
import os
import re

import numpy as np

from byzantine_experiments.analysis.matched_triple import SEEDS, SELECTION_FILE, _source_sidecar
from byzantine_experiments.paths import RESULTS_ROOT

PHASE3_DIR = os.path.join(RESULTS_ROOT, 'phase3')
SHIFTS = ('none', 'mild', 'severe', 'mild_v1', 'severe_v1')
N_EPISODES = 50


def _name(member, shift, seed, n=N_EPISODES):
    return f'p3_{member}_{shift}_s{seed}_n{n}'


def jobs(cross_seed=False, shifts=SHIFTS, n=N_EPISODES):
    """(name, seed, tier, tier_config_json, indices, shift) rows. Settings come from the
    members' own sidecars / the selection file, never retyped."""
    from byzantine_experiments.agents.tier3_learned import final_milestone_m
    from byzantine_experiments.wrappers.shift import load_shift_levels
    if not os.path.exists(SELECTION_FILE):
        raise SystemExit(f'{SELECTION_FILE} missing -- run scripts/byz_matched_triple.sh first')
    missing = [s for s in shifts if s != 'none' and s not in load_shift_levels(RESULTS_ROOT)]
    if missing:
        raise SystemExit(f'shift level(s) {missing} not calibrated -- run '
                         'scripts/byz_calibrate_shift.sh first')
    sel = json.load(open(SELECTION_FILE))
    t3 = sel['tier3']['members']
    rows = []
    for shift in shifts:
        for s in SEEDS:
            rows.append((_name('clean', shift, s, n), s, 'clean', '{}', '-', shift))
            for tier, tag in (('tier1', 't1'), ('tier2', 't2')):
                sc = _source_sidecar(tier, sel[tier]['config'])
                rows.append((_name(f'{tag}_{sel[tier]["config"]}', shift, s, n), s, sc['tier'],
                             json.dumps(sc['tier_config']),
                             ','.join(str(i) for i in sc['compromised_indices']), shift))
            advs = t3 if cross_seed else [t3[SEEDS.index(s)]]
            for run in advs:
                ms = final_milestone_m(os.path.join('train_dir', run))
                rows.append((_name(f't3_{run}', shift, s, n), s, 'tier3_learned',
                             json.dumps({'checkpoint': f'train_dir/{run}',
                                         'milestone': ms * 1_000_000}), '0', shift))
    return rows


def _added_cm(attacked, clean, allow_different_shift=False):
    """Per-episode honest goal-RMSE difference in cm, after verifying pairing. The % column divides
    by the clean error AT THAT LEVEL, which grows 2.6x from none to severe_v1, so a constant
    attack shrinks in % under shift; this column shows whether it shrinks in metres too."""
    from byzantine_experiments.analysis.gate_tier3 import _honest_rmse_per_episode
    from byzantine_experiments.analysis.paired import check_pairing
    n, _ = check_pairing(attacked, clean, allow_different_shift=allow_different_shift)
    with np.load(attacked) as z:
        idx = np.where(~np.asarray(z['byzantine_mask']))[0]
    return 100.0 * (_honest_rmse_per_episode(attacked, idx)[:n]
                    - _honest_rmse_per_episode(clean, idx)[:n])


def table(n=N_EPISODES):
    """Damage per member x shift, pooled over seeds, each run paired with clean at its shift."""
    from byzantine_experiments.analysis.paired import PairingError, per_episode_degradation, pooled
    runs = sorted(glob.glob(os.path.join(PHASE3_DIR, 'p3_*.npz')))
    groups = {}
    for p in runs:
        m = re.match(r'p3_(?P<member>.+)_(?P<shift>' + '|'.join(SHIFTS) + r')_s(?P<seed>\d+)_n\d+$',
                     os.path.basename(p)[:-4])
        if not m or m['member'] == 'clean':
            continue
        seed = int(m['seed'])
        kind = m['member']
        if kind.startswith('t3_'):
            aseed = int(re.search(r'seed(\d+)', kind).group(1))
            kind = 't3 (own seed)' if aseed == seed else 't3 (cross-seed)'
        groups.setdefault((kind, m['shift']), []).append((p, seed))
    L = ['# Phase 3 -- damage per member under distribution shift\n',
         'Honest goal-RMSE degradation, each run paired with the clean run at the same seed and '
         'shift level, pooled over seeds. The first rows give the shift\'s own effect on the '
         'clean swarm (clean at that level vs clean with no shift, same scenarios).\n',
         'The cm column is the same comparison in absolute terms (added honest goal error). The '
         '% column is relative to the clean error at that level, which grows with the shift.\n',
         '| condition | shift | pooled damage % [95 % CI] | added error, cm [95 % CI] | episodes '
         '| runs |', '|---|---|---|---|---|---|']
    from byzantine_experiments.wrappers.shift import load_shift_levels
    levels = load_shift_levels(RESULTS_ROOT)
    # Order by the calibrated clean-swarm effect, so the table reads as a dose-response.
    eff = {s: (0.0 if s == 'none' else levels.get(s, {}).get('effect_pct', float('inf')))
           for s in SHIFTS}
    order = sorted(SHIFTS, key=lambda s: eff[s])
    for shift in order[1:]:
        d, cm = [], []
        for s in SEEDS:
            a = os.path.join(PHASE3_DIR, _name('clean', shift, s, n) + '.npz')
            c = os.path.join(PHASE3_DIR, _name('clean', 'none', s, n) + '.npz')
            if os.path.exists(a) and os.path.exists(c):
                d.append(per_episode_degradation(a, c, allow_different_shift=True))
                cm.append(_added_cm(a, c, allow_different_shift=True))
        if d:
            r, q = pooled(d), pooled(cm)
            L.append(f"| clean swarm (shift effect) | {shift} | {r['mean']:+.1f} "
                     f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] | {q['mean']:+.1f} "
                     f"[{q['ci_lo']:+.1f}, {q['ci_hi']:+.1f}] | {r['n']} | {len(d)} |")
    for (kind, shift), members in sorted(groups.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1]))):
        d, cm, bad = [], [], []
        for p, seed in members:
            c = os.path.join(PHASE3_DIR, _name('clean', shift, seed, n) + '.npz')
            try:
                d.append(per_episode_degradation(p, c))
                cm.append(_added_cm(p, c))
            except (PairingError, FileNotFoundError) as e:
                bad.append(f'{os.path.basename(p)}: {e}')
        if d:
            r, q = pooled(d), pooled(cm)
            L.append(f"| {kind} | {shift} | {r['mean']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] "
                     f"| {q['mean']:+.1f} [{q['ci_lo']:+.1f}, {q['ci_hi']:+.1f}] | {r['n']} | {len(d)} |")
        for b in bad:
            L.append(f'| {kind} | {shift} | NOT PAIRED: {b[:120]} | | | |')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['jobs', 'table'])
    ap.add_argument('--cross-seed', action='store_true')
    ap.add_argument('--n-episodes', type=int, default=N_EPISODES)
    a = ap.parse_args()
    if a.cmd == 'jobs':
        for row in jobs(a.cross_seed, n=a.n_episodes):
            print('\t'.join(str(x) for x in row))
    else:
        text = table(a.n_episodes)
        print(text)
        os.makedirs(RESULTS_ROOT, exist_ok=True)
        with open(os.path.join(RESULTS_ROOT, 'phase3_table.md'), 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
