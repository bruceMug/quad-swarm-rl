"""
The spec's efficacy-gate table: every trained adversary against all five criteria (design item 3).

| criterion | threshold | measured on |
|---|---|---|
| honest goal-RMSE degradation (primary) | >= 10 % AND paired 95 % CI above 0 | own seed, final checkpoint, n = 200, paired |
| team-reward degradation (secondary) | reported with its CI, not gated | same pair |
| adversary survival | >= 70 % of episodes | same run |
| action-magnitude ratio vs honest | <= 1.5 (stealth condition only) | same run |
| damage transfers across seeds | TESTED (either outcome is a result) | transfer matrix, >= 2 other seeds |

The gate value (10 %) is `gate_tier3.GATE`, recalibrated from the spec's 30 % on 2026-09-24 and
kept on 2026-09-26 (decision D2). The stealth condition is `w_stealth > 0` (the spec's action
penalty) or a plausible lie (D3); a lie-only adversary flies the honest policy, so its ratio is
~1.0 by construction and the criterion is met trivially -- which is why D3 exists.

Every adversary gets exactly one verdict:
  PASS        all five criteria met on n >= 200 evidence;
  FAIL        at least one criterion decisively failed (reasons listed);
  INCOMPLETE  a criterion lacks the evidence to decide (what is missing is listed) -- never
              silently treated as a pass or a fail.

Adversaries whose training was defective are listed with the defect and not scored.

    python -m byzantine_experiments.analysis.efficacy_table
"""

import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from byzantine_experiments.analysis.gate_tier3 import CONFIRMATORY_MIN_EPISODES, GATE
from byzantine_experiments.analysis.paired import (
    PairingError,
    paired_degradation,
    paired_reward_degradation,
)
from byzantine_experiments.paths import RESULTS_ROOT

CONFIRM_DIR = os.path.join(RESULTS_ROOT, 'tier3_confirm')

# Training defects established in tier3_design 4.2 / 4.4. Listed, not scored.
DEFECTS = {
    'adv_seed1_stealth5': 'stealth penalty scored unclipped actions (r_stealth -534k); design 4.4',
    'adv_seed1_stealth5_f250': 'stealth penalty dominated the objective 555:1; design 4.2',
}


def _cfg(run):
    with open(os.path.join('train_dir', run, 'config.json')) as f:
        return json.load(f)


def _screening(run):
    """Horizon-sweep rows for this adversary from the v2 gate CSV (n = 50 screening)."""
    path = os.path.join(RESULTS_ROOT, 'tier3_gate.csv')
    if not os.path.exists(path):
        return []
    pref = 't3_' + run + '_on_'
    return [r for r in csv.DictReader(open(path)) if r['run'].startswith(pref)
            and re.match(r'.*_m\d+M$', r['run'])]


def _metrics(npz):
    with open(npz[:-4] + '.json') as f:
        return json.load(f).get('metrics', {})


def decide(primary, survival, ratio, stealth, transfer_tested):
    """
    The five-criterion verdict, as a pure function (spec test 8 exercises every branch).

    primary: None, or dict(deg, ci_lo, n) from a paired own-seed run.
    Returns (verdict, reasons). Missing evidence yields INCOMPLETE, never a silent pass/fail.
    """
    reasons, missing = [], []
    if primary is None or primary.get('n', 0) < CONFIRMATORY_MIN_EPISODES:
        missing.append(f'own-seed n>={CONFIRMATORY_MIN_EPISODES} confirmatory run '
                       '(scripts/byz_transfer.sh)')
    elif not (primary['deg'] >= GATE['honest_goal_rmse_deg'] and primary['ci_lo'] > 0):
        reasons.append(f"damage {primary['deg']:+.1f} % (CI low {primary['ci_lo']:+.1f}) "
                       f"below the {GATE['honest_goal_rmse_deg']:.0f} % gate")
    if survival is None or not np.isfinite(survival):
        missing.append('survival')
    elif survival < GATE['survival']:
        reasons.append(f'survival {100 * survival:.0f} % < {100 * GATE["survival"]:.0f} %')
    if stealth:
        if ratio is None or not np.isfinite(ratio):
            missing.append('action ratio')
        elif ratio > GATE['action_ratio']:
            reasons.append(f'action ratio {ratio:.2f} > {GATE["action_ratio"]}')
    if not transfer_tested:
        missing.append('cross-seed transfer (scripts/byz_transfer.sh)')
    if reasons:
        # A decisive failure cannot be reversed by more evidence, so do not ask for any: listing
        # "missing: transfer" here would suggest collecting runs the transfer script excludes on
        # purpose (crashing adversaries fail survival on their own seed).
        return 'FAIL', reasons
    if missing:
        return 'INCOMPLETE', missing
    return 'PASS', []


def evaluate(run):
    from byzantine_experiments.analysis.transfer_matrix import cell_path, transfer_tested
    cfg = _cfg(run)
    aseed = int(re.search(r'seed(\d+)', run).group(1))
    stealth = float(cfg.get('adv_w_stealth', 0) or 0) > 0 or cfg.get('adv_lie_mode') == 'plausible'
    n_adv = len([i for i in str(cfg.get('adv_indices', '0')).split(',') if i.strip()])
    row = dict(run=run, seed=aseed, stealth=stealth, n_adv=n_adv,
               channel=('observation' if cfg.get('adv_enable_lies') else 'motors'),
               lie_mode=cfg.get('adv_lie_mode', 'free') if cfg.get('adv_enable_lies') else '-')
    if run in DEFECTS:
        row.update(verdict='NOT SCORED', reasons=[DEFECTS[run]])
        return row

    reasons, missing = [], []
    confirm = None
    if n_adv == 1:
        path, _ = cell_path(run, aseed, CONFIRMATORY_MIN_EPISODES)
        clean = os.path.join(CONFIRM_DIR, f'confirm_clean_s{aseed}_n{CONFIRMATORY_MIN_EPISODES}.npz')
        if os.path.exists(path) and os.path.exists(clean):
            try:
                confirm = (path, paired_degradation(path, clean),
                           paired_reward_degradation(path, clean), _metrics(path))
            except PairingError as e:
                missing.append(f'own-seed n=200 run not pairable: {e}')
    screen = _screening(run)
    surv_screen = (np.mean([float(r['survival']) for r in screen]) if screen else float('nan'))

    if confirm:
        _, g, rw, m = confirm
        row.update(deg=g['deg_pct'], deg_ci=(g['ci_lo'], g['ci_hi']), deg_n=g['n_pairs'],
                   reward=rw['deg_pct'], reward_ci=(rw['ci_lo'], rw['ci_hi']),
                   survival=float(m.get('attacker_survived_mean', float('nan'))),
                   ratio=float(m.get('action_norm_ratio', float('nan'))), evidence='n=200 paired')
    else:
        d = [float(r['honest_goal_rmse_deg_pct']) for r in screen]
        row.update(deg=(float(np.mean(d)) if d else float('nan')), deg_ci=None, deg_n=None,
                   reward=float('nan'), reward_ci=None, survival=surv_screen,
                   ratio=(float(np.mean([float(r['action_norm_ratio']) for r in screen]))
                          if screen else float('nan')),
                   evidence=('n=50 screening only' if screen else 'none'))

    tested = n_adv == 1 and transfer_tested(run)
    row['transfer_tested'] = tested
    primary = (dict(deg=row['deg'], ci_lo=row['deg_ci'][0], n=row['deg_n'])
               if row.get('deg_ci') else None)
    verdict, why = decide(primary, row['survival'], row['ratio'], stealth, tested)
    row.update(verdict=verdict, reasons=missing + why if missing else why)
    return row


def report():
    runs = sorted(os.path.basename(p) for p in glob.glob('train_dir/adv_seed*')
                  if os.path.exists(os.path.join(p, 'config.json')))
    L = ['# Tier 3 efficacy gate -- all adversaries, all five criteria\n',
         f'Gate: honest goal-RMSE degradation >= {GATE["honest_goal_rmse_deg"]:.0f} % with the '
         f'paired CI above 0 on n >= {CONFIRMATORY_MIN_EPISODES}; survival >= '
         f'{100 * GATE["survival"]:.0f} %; action ratio <= {GATE["action_ratio"]} (stealth only); '
         'transfer tested. Team reward is secondary: reported, not gated.\n',
         '| adversary | channel | stealth | evidence | damage % [95 % CI] | team reward % [CI] '
         '| survival | ratio | transfer tested | verdict |',
         '|' + '---|' * 10]
    for run in runs:
        r = evaluate(run)
        if r['verdict'] == 'NOT SCORED':
            L.append(f"| `{run}` | {r['channel']} | {'yes' if r['stealth'] else 'no'} | — | — | — | — | — | — "
                     f"| NOT SCORED: {r['reasons'][0]} |")
            continue
        ci = lambda c: f' [{c[0]:+.1f}, {c[1]:+.1f}]' if c else ''
        fmt = lambda x, f='{:+.1f}': (f.format(x) if np.isfinite(x) else 'n/a')
        L.append(f"| `{run}` | {r['channel']}{'/' + r['lie_mode'] if r['lie_mode'] != '-' else ''} "
                 f"| {'yes' if r['stealth'] else 'no'} | {r['evidence']} "
                 f"| {fmt(r['deg'])}{ci(r['deg_ci'])} | {fmt(r['reward'])}{ci(r['reward_ci'])} "
                 f"| {fmt(100 * r['survival'], '{:.0f} %')} | {fmt(r['ratio'], '{:.2f}')} "
                 f"| {'yes' if r['transfer_tested'] else 'no'} | **{r['verdict']}**"
                 f"{': ' + '; '.join(r['reasons']) if r['reasons'] else ''} |")
    return '\n'.join(L)


def main():
    text = report()
    print(text)
    with open(os.path.join(RESULTS_ROOT, 'tier3_efficacy.md'), 'w') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
