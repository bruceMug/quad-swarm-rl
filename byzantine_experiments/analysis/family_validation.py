"""
Tier 1 / Tier 2 validation at the FAMILY level -- decision D1, 2026-09-26.

The Tier 1 and Tier 2 specs validated a tier by requiring configurations inside a 20-40 %
degradation band. That band was set on legacy numbers the seeding bug inflated ~3x; on v2 data
no configuration reaches 20 %, so as written both tiers would fail their own gates. It also
tested each configuration on its own at n = 50, where individual significance barely beats
chance (5 of 144 v2 runs, against ~3.6 expected from noise).

Replacement criterion: a family (one Tier 1 fault mechanism or one Tier 2 attack) is
**validated** when its runs, taken together, significantly damage the honest team:
one-sample t-test on the per-run paired degradations, mean > 0 and p < 0.05. A family whose
runs significantly *improve* the honest team is reported as such -- that is a finding, not a
failure to validate.

Only PAIRED rows count (v2 collector, verified by analysis/paired.py). Unpaired rows are counted
and excluded.
"""

from collections import defaultdict

import numpy as np
from scipy import stats

ALPHA = 0.05


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float('nan')


def family_rows(rows):
    """[(variant, dict)] sorted by variant, from calibration-table rows."""
    fams, unpaired = defaultdict(list), 0
    for r in rows:
        if str(r.get('paired', '')).lower() != 'true':
            unpaired += 1
            continue
        d = _f(r, 'paired_honest_deg_pct')
        if np.isfinite(d):
            fams[r.get('variant', '?')].append(d)
    out = []
    for v, d in sorted(fams.items()):
        d = np.asarray(d)
        p = float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 2 else float('nan')
        status = ('VALIDATED -- damaging' if p < ALPHA and d.mean() > 0 else
                  'significantly HELPS the honest team' if p < ALPHA and d.mean() < 0 else
                  'no family-level effect')
        out.append((v, dict(runs=len(d), mean=float(d.mean()), positive=int((d > 0).sum()),
                            p=p, status=status)))
    return out, unpaired


def validated(rows):
    """Families that pass: significantly damaging as a whole."""
    return {v for v, s in family_rows(rows)[0] if s['status'].startswith('VALIDATED')}


def markdown(rows, title='Family-level validation (decision D1)'):
    fam, unpaired = family_rows(rows)
    L = [f'\n## {title}\n',
         'A family is validated when its runs together significantly damage the honest team '
         '(one-sample t-test on per-run **paired** honest goal-RMSE degradation, mean > 0, '
         'p < 0.05). This replaces the 20–40 % band, which was set on legacy numbers and which '
         'no v2 configuration reaches.\n',
         '| family | runs | mean | positive | family p | status |', '|---|---|---|---|---|---|']
    for v, s in fam:
        L.append(f"| {v} | {s['runs']} | {s['mean']:+.1f} % | {s['positive']}/{s['runs']} "
                 f"| {s['p']:.4f} | {s['status']} |")
    if unpaired:
        L.append(f'\n{unpaired} unpaired row(s) excluded (legacy collector or seed mismatch).')
    return '\n'.join(L) + '\n'
