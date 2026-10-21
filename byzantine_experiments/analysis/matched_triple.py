"""
The matched-damage triple and its equivalence test (spec: "the paper's central experiment").

Same damage, three causal mechanisms -- a stochastic fault (Tier 1), a hand-designed attack
(Tier 2), a trained adversary (Tier 3) -- so that a detector which separates them is shown to
recover *mechanism*, not severity. Matching must be established by an equivalence test (TOST),
never by a failed null test.

PRE-REGISTERED 2026-09-26 (decision D1), before any triple data was collected. The spec's
original band (~25 %, "land in 20-40 %") and margin (+/- 10 pp) were set on legacy numbers that
the seeding bug inflated ~3x; on v2 data every effect is below 15 %, so both are replaced:

  Tier 3 member   The stealth-constrained adversary -- the plausible-lie adversaries of D3, one
                  per baseline seed, each evaluated on its own seed at its final checkpoint --
                  IF their pooled own-seed damage is significant (95 % CI above 0). Otherwise the
                  free lie-only adversaries, with stealth reported as a limitation. This is the
                  spec's own fallback rule ("fall back to the unconstrained adversary ... and
                  document the stealth constraint as a limitation").
  Target          The Tier 3 member's pooled damage over baseline seeds 1-3 (n = 200 per seed,
                  from the transfer matrix diagonal).
  Tier 1 / 2      From families that pass the v2 family-level validation (the family as a whole
  members         significantly damaging, p < 0.05 -- as of Stage C: Tier 1 `stall`, Tier 2
                  `liar`), at f = 0.125 (one compromised drone, like every Tier 3 adversary):
                  the configuration whose pooled screening damage (seeds 1-3, n = 50 each) is
                  closest to the target. Ties go to the smaller perturbation.
  Sample          All conditions at the SAME --seed per baseline seed, so every comparison is
                  paired per scenario. n per condition per seed = the smallest of 400 / 800 / 1600
                  giving >= 80 % power to establish the margin between two truly equal
                  conditions, from the measured per-episode sd of a difference (29.0 pp):
                  +/-3 pp -> 400, +/-2 pp -> 800, +/-1.5 pp -> 1600. Beyond that the triple is
                  declared underpowered rather than run.
  Margin          +/- 3 pp -- or half the target if the target is below 6 %, so that a
                  near-harmless condition cannot pass as "matched" to a small effect.
  Test            TOST, alpha = 0.05 each side: matched iff the 90 % CI of the pooled per-episode
                  difference lies inside +/- margin. Every member must ALSO be significantly
                  damaging on its own (95 % CI above 0).
  Secondary       Team-reward degradation per member, reported with its CI, not used to match.

Measured precision (per-episode sd of a difference between two attacks on identical scenarios:
29.0 pp): pooled N = 1200 gives SE 0.84 pp, a 90 % half-width of 1.38 pp, and 95 % power to
establish a +/- 3 pp match between two genuinely equal conditions.

    python -m byzantine_experiments.analysis.matched_triple select
    python -m byzantine_experiments.analysis.matched_triple test
"""

import argparse
import csv
import datetime
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

from byzantine_experiments.analysis.paired import PairingError, per_episode_degradation, pooled
from byzantine_experiments.paths import RESULTS_ROOT, results_dir

SEEDS = (1, 2, 3)
N_CHOICES = (400, 800, 1600)
SD_DIFF_PP = 29.0              # measured: trained minus hand-designed, seed 3, n = 200
BASE_MARGIN = 3.0
ALPHA = 0.05
SELECTION_FILE = os.path.join(RESULTS_ROOT, 'matched_triple_selection.json')
TRIPLE_DIR = os.path.join(RESULTS_ROOT, 'matched_triple')
VALIDATED_FAMILIES = {'tier1': 'stall', 'tier2': 'liar'}     # from Stage C phase 1 (6.21)
PLAUSIBLE = [f'adv_seed{s}_stealth0_lieonlyplaus' for s in SEEDS]
FREE = [f'adv_seed{s}_stealth0_lieonly' for s in SEEDS]


# ---------------------------------------------------------------- TOST (spec test 9)

def tost(differences, margin, alpha=ALPHA):
    """
    Two one-sided tests on paired per-episode differences (in pp).

    Equivalent iff both H0: mu <= -margin and H0: mu >= +margin are rejected at `alpha`,
    i.e. the (1 - 2 alpha) CI lies strictly inside (-margin, +margin).
    """
    d = np.asarray(differences, dtype=np.float64)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    m = float(d.mean())
    t_lo = (m + margin) / se            # tests mu > -margin
    t_hi = (m - margin) / se            # tests mu < +margin
    p_lo = float(1 - stats.t.cdf(t_lo, n - 1))
    p_hi = float(stats.t.cdf(t_hi, n - 1))
    crit = stats.t.ppf(1 - alpha, n - 1)
    lo, hi = m - crit * se, m + crit * se
    return dict(diff=m, ci90_lo=float(lo), ci90_hi=float(hi), p_tost=max(p_lo, p_hi),
                equivalent=bool(max(p_lo, p_hi) < alpha), margin=float(margin), n=n)


def margin_for(target):
    return float(min(BASE_MARGIN, 0.5 * abs(target)))


def tost_power(margin, n_per_seed, sd=SD_DIFF_PP, n_seeds=3, alpha=ALPHA):
    """Probability TOST establishes +/- margin when the true difference is exactly 0."""
    se = sd / np.sqrt(n_seeds * n_per_seed)
    return float(max(0.0, 2 * stats.norm.cdf(margin / se - stats.norm.ppf(1 - alpha)) - 1))


def n_for(margin):
    for n in N_CHOICES:
        if tost_power(margin, n) >= 0.8:
            return n, tost_power(margin, n)
    return None, tost_power(margin, N_CHOICES[-1])


# ---------------------------------------------------------------- selection

def _screening_by_config(tier, family):
    """Pooled (over seeds) paired screening damage per config at f = 0.125, from v2 tables."""
    path = os.path.join(RESULTS_ROOT, f'{tier}_calibration.csv')
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    by = defaultdict(list)
    for r in rows:
        if r.get('variant') != family or str(r.get('paired', '')).lower() != 'true':
            continue
        if str(r.get('f_label', r.get('f', ''))) not in ('0125', '0.125'):
            continue
        by[r['config']].append(float(r['paired_honest_deg_pct']))
    return {c: (float(np.mean(v)), len(v)) for c, v in by.items() if len(v) == len(SEEDS)}


def _perturbation_size(config):
    """Smaller-perturbation tiebreak: the first number in the config name."""
    m = re.search(r'([\d.]+)', config.split('_', 1)[-1])
    return float(m.group(1)) if m else float('inf')


def _own_seed_cells(members):
    """(cells, missing): each member's own-seed n=200 paired result, and what is not there yet."""
    from byzantine_experiments.analysis.transfer_matrix import cell_path, matrix
    cells, missing = [], []
    for r in members:
        if not os.path.isdir(os.path.join('train_dir', r)):
            missing.append(f'{r}: not trained')
            continue
        aseed, row = matrix([r])[r]
        cell = row.get(aseed)
        if not cell or 'error' in cell:
            why = cell['error'] if cell else 'not run'
            missing.append(f'{os.path.basename(cell_path(r, aseed, 200)[0])}: {why}')
            continue
        cells.append(cell)
    return cells, missing


def _tier3_member():
    """
    (member, missing). The rule is ordered: the plausible adversaries are decided FIRST, and the
    free ones are consulted only once the plausible ones are complete and not significant. An
    incomplete plausible set is "undetermined", never a reason to fall back -- falling back on
    missing data would let the order in which runs finish choose the member.
    """
    for members, label in ((PLAUSIBLE, 'plausible (stealth)'), (FREE, 'free (unconstrained)')):
        diag, missing = _own_seed_cells(members)
        if missing:
            return None, [f'{label}: {m}' for m in missing]
        # Pool the own-seed confirmatory runs over seeds (equal weight per seed).
        est = float(np.mean([c['deg_pct'] for c in diag]))
        se = float(np.sqrt(np.sum([((c['ci_hi'] - c['ci_lo']) / 3.92) ** 2 for c in diag])) / len(diag))
        sig = est - 1.96 * se > 0
        if sig or label.startswith('free'):
            return dict(members=members, kind=label, target=est, target_ci=[est - 1.96 * se, est + 1.96 * se],
                        significant=bool(sig), fallback=(label.startswith('free'))), []
    return None, []


def select():
    t3, missing = _tier3_member()
    if t3 is None:
        raise SystemExit('Tier 3 member undetermined -- own-seed n=200 runs still missing:\n  '
                         + '\n  '.join(missing) + '\nRun block C (horizon sweep + '
                         'scripts/byz_transfer.sh) for those adversaries, then re-run this.')
    target = t3['target']
    chosen = {}
    from byzantine_experiments.analysis.family_validation import validated
    for tier, family in VALIDATED_FAMILIES.items():
        # The families are fixed in advance (Stage C); refuse if the tables no longer agree,
        # rather than quietly matching against a family that does not validate.
        path = os.path.join(RESULTS_ROOT, f'{tier}_calibration.csv')
        rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
        if family not in validated(rows):
            raise SystemExit(f'{tier} family {family!r} does not pass family-level validation '
                             f'in {path} -- the pre-registered selection cannot proceed')
        cands = _screening_by_config(tier, family)
        if not cands:
            raise SystemExit(f'no v2 {tier} {family} configs at f=0.125 with all three seeds')
        best = min(cands, key=lambda c: (abs(cands[c][0] - target), _perturbation_size(c)))
        chosen[tier] = dict(config=best, screening_pooled_pct=cands[best][0],
                            candidates={c: v[0] for c, v in sorted(cands.items())})
    margin = margin_for(target)
    n, power = n_for(margin)
    if n is None:
        raise SystemExit(f'target {target:+.2f} % gives margin +/-{margin:.2f} pp; even '
                         f'{N_CHOICES[-1]} episodes/seed has only {100 * power:.0f} % power. '
                         'The triple is underpowered at this effect size -- report that '
                         'instead of running it.')
    sel = dict(tier3=t3, tier1=chosen['tier1'], tier2=chosen['tier2'],
               margin_pp=margin, n_per_seed=n, tost_power_if_equal=power,
               selected_at=datetime.datetime.now().isoformat(timespec='seconds'),
               rule='pre-registered 2026-09-26, analysis/matched_triple.py docstring')
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(SELECTION_FILE, 'w') as f:
        json.dump(sel, f, indent=2)
    return sel


# ---------------------------------------------------------------- collection jobs

def _source_sidecar(tier, config, seed=1):
    """The v2 sweep sidecar a calibration-table config came from (config = variant_rest)."""
    variant, rest = config.split('_', 1)
    tn = tier[-1]
    path = os.path.join(results_dir(tier), f't{tn}_{variant}_s{seed}_{rest}.json')
    if not os.path.exists(path):
        raise SystemExit(f'cannot find the source run for {tier} {config}: {path}')
    return json.load(open(path))


def jobs():
    """TSV lines: name, seed, tier, tier_config_json, indices -- read by the collection script.
    Tier and settings come from the member's own v2 sweep sidecar, never retyped."""
    from byzantine_experiments.agents.tier3_learned import final_milestone_m
    sel = json.load(open(SELECTION_FILE))
    out = []
    for s in SEEDS:
        p = member_paths(sel, s)
        out.append((os.path.basename(p['clean'])[:-4], s, 'clean', '{}', '-'))
        for tier in ('tier1', 'tier2'):
            sc = _source_sidecar(tier, sel[tier]['config'])
            idx = ','.join(str(i) for i in sc['compromised_indices'])
            out.append((os.path.basename(p[tier])[:-4], s, sc['tier'],
                        json.dumps(sc['tier_config']), idx))
        run = sel['tier3']['members'][SEEDS.index(s)]
        ms = final_milestone_m(os.path.join('train_dir', run))
        out.append((os.path.basename(p['tier3'])[:-4], s, 'tier3_learned',
                    json.dumps({'checkpoint': f'train_dir/{run}', 'milestone': ms * 1_000_000}), '0'))
    for row in out:
        print('\t'.join(str(x) for x in row))


# ---------------------------------------------------------------- the test

def member_paths(sel, seed):
    """{tier: npz path} for one baseline seed, plus the clean reference."""
    t3 = sel['tier3']['members'][SEEDS.index(seed)]
    N_PER_SEED = sel['n_per_seed']
    return dict(
        clean=os.path.join(TRIPLE_DIR, f'triple_clean_s{seed}_n{N_PER_SEED}.npz'),
        tier1=os.path.join(TRIPLE_DIR, f'triple_t1_{sel["tier1"]["config"]}_s{seed}_n{N_PER_SEED}.npz'),
        tier2=os.path.join(TRIPLE_DIR, f'triple_t2_{sel["tier2"]["config"]}_s{seed}_n{N_PER_SEED}.npz'),
        tier3=os.path.join(TRIPLE_DIR, f'triple_t3_{t3}_s{seed}_n{N_PER_SEED}.npz'),
    )


def test():
    sel = json.load(open(SELECTION_FILE))
    margin = sel['margin_pp']
    idx = list(range(1, 8))          # every member compromises drone 0: honest = 1..7
    per = {k: [] for k in ('tier1', 'tier2', 'tier3')}
    pairs = {k: [] for k in (('tier1', 'tier2'), ('tier1', 'tier3'), ('tier2', 'tier3'))}
    for s in SEEDS:
        p = member_paths(sel, s)
        for k in per:
            per[k].append(per_episode_degradation(p[k], p['clean'], idx=idx))
        for a, b in pairs:
            pairs[(a, b)].append(per_episode_degradation(p[a], p['clean'], idx=idx,
                                                         reference_npz=p[b]))
    L = [f'Matched-damage triple, pooled over seeds {SEEDS}, n = {sel["n_per_seed"]} per seed, '
         f'margin +/- {margin:.2f} pp (pre-registered)\n']
    all_sig = True
    for k, lab in (('tier1', sel['tier1']['config']), ('tier2', sel['tier2']['config']),
                   ('tier3', sel['tier3']['kind'])):
        r = pooled(per[k])
        sig = r['ci_lo'] > 0
        all_sig &= sig
        L.append(f'  {k} {lab:32s} {r["mean"]:+6.2f} % [{r["ci_lo"]:+5.2f}, {r["ci_hi"]:+5.2f}] '
                 f'{"damaging" if sig else "NOT significantly damaging"}')
    L.append('')
    all_eq = True
    for (a, b), diffs in pairs.items():
        t = tost(np.concatenate(diffs), margin)
        all_eq &= t['equivalent']
        L.append(f'  {a} - {b}: {t["diff"]:+5.2f} pp, 90 % CI [{t["ci90_lo"]:+5.2f}, '
                 f'{t["ci90_hi"]:+5.2f}], p_TOST = {t["p_tost"]:.4f} -> '
                 f'{"EQUIVALENT" if t["equivalent"] else "not shown equivalent"}')
    verdict = 'MATCHED' if (all_eq and all_sig) else 'NOT MATCHED'
    L.append(f'\n  Triple: {verdict}' + ('' if all_sig else
             ' (a member is not significantly damaging -- matching would be vacuous)'))
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['select', 'jobs', 'test'])
    args = ap.parse_args()
    if args.stage == 'jobs':
        jobs()
        return
    if args.stage == 'select':
        print(json.dumps(select(), indent=2))
        print(f'\nWrote {SELECTION_FILE}. Collect with: bash scripts/byz_matched_triple.sh')
    else:
        text = test()
        print(text)
        with open(os.path.join(RESULTS_ROOT, 'matched_triple.md'), 'w') as f:
            f.write('# Matched-damage triple\n\n```\n' + text + '\n```\n')


if __name__ == '__main__':
    main()
