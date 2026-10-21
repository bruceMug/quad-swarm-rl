"""
Check the efficacy gate against the complete v2 Tier 1/2 results -- read BEFORE Tier 3 phase 2.

The gate was recalibrated from 30 % to 10 % on 2026-09-24 (see `gate_tier3.GATE`), using the
first v2 data available: the Tier 2 `liar` family. It is meant to mean "at least as damaging as
the best hand-designed attack". This report re-checks that meaning once every hand-designed
family has been re-collected, and it is deliberately run before any v2 Tier 3 evaluation is
analysed, so that if the gate has to move it moves on hand-designed evidence alone and cannot
be fitted to the Tier 3 result.

Per family it reports, from the PAIRED columns only (unpaired rows are counted and excluded):
mean degradation, best run, how many runs are significant, and how many are significant *and*
at or above the gate. The last line compares the gate with the hand-designed ceiling.

All numbers here come from n = 50 runs, i.e. 57 % power to detect a true 10 % effect. Family
means are the more reliable signal; individual maxima are biased upward (winner's curse).

    python -m byzantine_experiments.analysis.gate_calibration
"""

import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats

from byzantine_experiments.analysis.gate_tier3 import GATE
from byzantine_experiments.paths import RESULTS_ROOT

TABLES = (('Tier 1', os.path.join(RESULTS_ROOT, 'tier1_calibration.csv')),
          ('Tier 2', os.path.join(RESULTS_ROOT, 'tier2_calibration.csv')))
ALPHA = 0.05


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float('nan')


def load():
    """(tier, variant) -> list of paired rows; plus a count of unpaired rows per tier."""
    fams, unpaired = defaultdict(list), defaultdict(int)
    for tier, path in TABLES:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if str(r.get('paired', '')).lower() != 'true':
                    unpaired[tier] += 1
                    continue
                fams[(tier, r.get('variant') or r.get('config', '?'))].append(r)
    return fams, unpaired


def report():
    gate = GATE['honest_goal_rmse_deg']
    fams, unpaired = load()
    L = [f'Gate calibration check (results root: {RESULTS_ROOT}); current gate {gate:.0f} %\n']
    missing = [t for t, p in TABLES if not os.path.exists(p)]
    if missing:
        L.append(f'  ! missing calibration table(s) for {", ".join(missing)} -- run the '
                 'aggregate_tier1 / aggregate_tier2 step first\n')
    if not fams:
        L.append('  no paired rows yet.')
        return '\n'.join(L)

    L.append(f"  {'family':28s} {'runs':>4s} {'mean':>7s} {'best':>7s} {'signif':>6s} "
             f"{'sig>=gate':>9s} {'positive':>9s} {'family p':>9s}")
    best_sig = None
    for (tier, fam), rows in sorted(fams.items()):
        d = np.array([_f(r, 'paired_honest_deg_pct') for r in rows])
        p = np.array([_f(r, 'paired_p') for r in rows])
        sig = (p < ALPHA) & (d > 0)
        for r, dd, ss in zip(rows, d, sig):
            if ss and (best_sig is None or dd > best_sig[0]):
                best_sig = (dd, f"{tier} {r.get('config', fam)} s{r.get('seed', '?')}")
        # Family-level test: is the family as a whole damaging? One-sample t-test on the run
        # means. At n = 50 per run, individual significance is barely above chance (5 of 144 on
        # the first full v2 pass, against 3.6 expected from noise), so this column, not the
        # per-run count, is the one to report.
        fam_p = stats.ttest_1samp(d[np.isfinite(d)], 0).pvalue if np.isfinite(d).sum() > 2 \
            else float('nan')
        L.append(f'  {tier + " " + fam:28s} {len(rows):4d} {np.nanmean(d):+6.1f}% '
                 f'{np.nanmax(d):+6.1f}% {int(sig.sum()):6d} {int((sig & (d >= gate)).sum()):9d} '
                 f'{int((d > 0).sum()):4d}/{len(d):<4d} {fam_p:9.4f}')
    for tier, n in unpaired.items():
        L.append(f'  ({n} {tier} rows excluded as unpaired -- see their pairing_error column)')

    n_all = sum(len(v) for v in fams.values())
    n_sig = sum(int(((np.array([_f(r, 'paired_p') for r in v]) < ALPHA)
                     & (np.array([_f(r, 'paired_honest_deg_pct') for r in v]) > 0)).sum())
                for v in fams.values())
    L.append(f'\n  {n_sig}/{n_all} runs individually significant and damaging; noise alone would '
             f'give ~{0.025 * n_all:.1f}. Read the family p column, not the per-run counts.')
    L.append('')
    if best_sig is None:
        L.append(f'  No hand-designed run is significantly damaging. A {gate:.0f} % gate is then '
                 'above anything Tier 1/2 achieves; Tier 3 clearing it would be a strong result, '
                 'failing it the expected one.')
    else:
        L.append(f'  Strongest significant hand-designed run: {best_sig[0]:+.1f} % ({best_sig[1]}).')
        if best_sig[0] >= 2 * gate:
            L.append(f'  -> well above the {gate:.0f} % gate. Before analysing Tier 3, decide '
                     'whether the gate should rise to match; record the decision in '
                     'gate_tier3.GATE with its date.')
        elif best_sig[0] < gate:
            L.append(f'  -> below the {gate:.0f} % gate: the gate asks Tier 3 to beat every '
                     'hand-designed attack. Consistent with its intended meaning.')
        else:
            L.append(f'  -> within reach of the {gate:.0f} % gate: consistent with "at least as '
                     'damaging as the best hand-designed attack". Keep it.')
    return '\n'.join(L)


def main():
    print(report())


if __name__ == '__main__':
    main()
