"""
What actually limits a Byzantine attack on this swarm: the action channel.

Written after Tier 3 round 2 (f = 0.25) failed the efficacy gate the same way round 1 did.
The question it answers is the one that decides where to spend the next GPU-day:

    Is the trained adversary weak because the optimiser is broken,
    or because nothing acting through that channel can do better?

Tiers 1 and 2 already contain the control. Group every hand-designed attack by the *channel*
it acts through, and compare each group's ceiling against the trained adversary's:

  * **motors only** -- the attacker corrupts its own 4 thrust commands and nothing else. It
    influences its neighbours purely through where its body ends up. Tier 2's `signflip`,
    `waypoint_pull` and `collusion` live here. **So does the Tier 3 learned adversary** -- its
    action space is exactly the same 4 motors.
  * **observation channel** -- the attacker additionally broadcasts false state, so honest
    drones act on a corrupted picture of where it is. Tier 2's `liar` lives here. No Tier 3
    adversary has this capability: `FrozenTeamEnv` gives the policy motor commands only.

Every figure is honest-drone goal RMSE (never the all-drone version, which is gameable -- see
gate_tier3.py), measured against the clean run **restricted to the same drone indices**, with
a Welch t-test over episodes. The significance test is not decoration: per-episode goal RMSE
has a ~42 % coefficient of variation here, so at n = 50 the 95 % CI on a degradation
percentage is about +/- 17 points. Several eye-catching point estimates in this corpus are
indistinguishable from zero, and a few "attacks" significantly *help* the honest team.

    python -m byzantine_experiments.analysis.channel_ceiling
"""

from byzantine_experiments.paths import RESULTS_ROOT, results_dir
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

from byzantine_experiments.analysis.gate_tier3 import _honest_rmse_per_episode

TIER1_DIR = results_dir('tier1')
TIER2_DIR = results_dir('tier2')
TIER3_DIR = results_dir('tier3')
TIER3_CSV = os.path.join(RESULTS_ROOT, 'tier3_gate.csv')
OUT_MD = os.path.join(RESULTS_ROOT, 'channel_ceiling.md')

# Which channel each Tier 2 family acts through. The split is a property of the agent's
# capability flags, not of how hard it hits: `liar` sets `lies_about_state`, the others do not.
CHANNEL = {
    'signflip': 'motors only',
    'waypoint': 'motors only',
    'collusion': 'motors only',
    'liar': 'observation channel',
}
MOTORS, OBS = 'motors only', 'observation channel'

# Runs skipped because they could not be paired with their clean reference. Reported at the
# end so an empty table is never mistaken for "no effect".
_UNPAIRED = []
from byzantine_experiments.analysis.gate_tier3 import GATE  # noqa: E402
GATE_PCT = GATE['honest_goal_rmse_deg']
ALPHA = 0.05


def compare(run_npz, clean_npz):
    """
    Honest-drone degradation % and p-value, PAIRED when both runs come from the fixed collector.

    Returns (deg_pct, p, paired). Unpaired results are still returned so legacy data can be
    inspected, but `paired=False` marks them as not reportable (design 6.16).
    """
    if not (os.path.exists(run_npz) and os.path.exists(clean_npz)):
        return None
    from byzantine_experiments.analysis.paired import PairingError, paired_degradation
    try:
        r = paired_degradation(run_npz, clean_npz)
        return r['deg_pct'], r['p_value'], True
    except PairingError:
        pass
    with np.load(run_npz) as d:
        idx = np.where(~np.asarray(d['byzantine_mask']))[0]
    if idx.size == 0:
        return None
    atk = _honest_rmse_per_episode(run_npz, idx)
    cln = _honest_rmse_per_episode(clean_npz, idx)
    if len(atk) < 2 or len(cln) < 2 or cln.mean() <= 1e-9:
        return None
    _, p = stats.ttest_ind(atk, cln, equal_var=False)
    return 100.0 * (atk.mean() - cln.mean()) / cln.mean(), float(p), False


def collect(results_dir, channel_of, f_of, seed_of, clean_dir=TIER1_DIR):
    """(channel, f) -> list of (degradation %, p, run name)."""
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, '*.npz'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        channel = channel_of(stem)
        seed = seed_of(stem)
        if channel is None or seed is None:
            continue
        res = compare(path, os.path.join(clean_dir, f't1_clean_s{seed}.npz'))
        if res is None:
            continue
        if not res[2]:
            _UNPAIRED.append(stem)
            continue
        groups[(channel, f_of(stem))].append((res[0], res[1], stem))
    return groups


def tier2_groups():
    def channel_of(stem):
        parts = stem.split('_')
        return CHANNEL.get(parts[1]) if len(parts) > 1 else None

    def seed_of(stem):
        m = re.search(r'_s(\d+)_', stem)
        return int(m.group(1)) if m else None

    return collect(TIER2_DIR, channel_of, lambda s: 0.25 if 'f025' in s else 0.125, seed_of)


def tier3_channel(stem):
    """
    Rounds 1-2 are motors-only (4 thrusts). Rounds 3-5 (`_lie`, `_lieonly`, `_lieonlyego`) also
    broadcast a lie, i.e. the observation channel. This used to label every Tier 3 run
    motors-only, which predates round 3 and would have filed the lying adversaries under the
    wrong channel.
    """
    if not stem.startswith('t3_'):
        return None
    return OBS if '_lie' in stem else MOTORS


def tier3_groups():
    """Tier 3 checkpoints grouped by the channel their adversary acts through."""
    def seed_of(stem):
        m = re.search(r'_on_s(\d+)_', stem)
        return int(m.group(1)) if m else None

    def f_of(stem):
        m = re.search(r'_f(\d+)_', stem)
        return int(m.group(1)) / 1000.0 if m else 0.125

    return collect(TIER3_DIR, tier3_channel, f_of, seed_of)


def summarise(entries):
    deg = np.array([e[0] for e in entries])
    worse = [e for e in entries if e[1] < ALPHA and e[0] > 0]
    better = [e for e in entries if e[1] < ALPHA and e[0] < 0]
    return dict(n=len(entries), max=deg.max(), median=float(np.median(deg)),
                n_worse=len(worse), n_better=len(better),
                best=max(entries, key=lambda e: e[0]))


def per_seed_table(t2_rows, t3_rows):
    """
    The same corpus split by *baseline seed*, which pooling hides.

    This was added after Tier 3 round 3: the channel conclusion was drawn on pooled data, and
    the three baselines turn out to differ by more than the channels do. Reporting only the
    pooled table makes a seed effect look like a channel effect.
    """
    L = ['\n## The same data, split by baseline seed\n']
    L.append('| channel | baseline seed | runs | max deg % | signif worse | clean goal RMSE |')
    L.append('|---|---|---|---|---|---|')
    clean = {}
    for seed in (1, 2, 3):
        path = os.path.join(TIER1_DIR, f't1_clean_s{seed}.npz')
        if os.path.exists(path):
            with np.load(path) as d:
                n = int(np.asarray(d['byzantine_mask']).size)
            clean[seed] = _honest_rmse_per_episode(path, np.arange(n)).mean()
    for channel in (MOTORS, OBS):
        for seed in (1, 2, 3):
            r = [x for x in t2_rows if x[0] == channel and x[1] == seed]
            if not r:
                continue
            sig = sum(1 for x in r if x[3] < ALPHA and x[2] > 0)
            L.append(f'| {channel} | {seed} | {len(r)} | **{max(x[2] for x in r):.1f} %** '
                     f'| {sig} | {clean.get(seed, float("nan")):.4f} |')
    for seed, entries in sorted(t3_rows.items()):
        sig = sum(1 for e in entries if e[1] < ALPHA and e[0] > 0)
        L.append(f'| observation channel (**trained**) | {seed} | {len(entries)} '
                 f'| **{max(e[0] for e in entries):.1f} %** | {sig} '
                 f'| {clean.get(seed, float("nan")):.4f} |')
    return L


def build_report(t2, t3, t2_rows=None, t3_lie_rows=None):
    L = ['# What limits the attack: the action channel\n']
    L.append('Honest-drone goal-RMSE degradation vs. the clean run on the **same drone '
             'indices**, with a Welch t-test over episodes. "signif worse/better" counts runs '
             f'with p < {ALPHA}. Efficacy gate is ≥ {GATE_PCT:.0f} %.\n')
    L.append('| tier | channel | f | runs | max deg % | median | **signif worse** | signif better |')
    L.append('|---|---|---|---|---|---|---|---|')

    def row(tier, key, entries):
        s = summarise(entries)
        L.append(f'| {tier} | {key[0]} | {key[1]:.3f} | {s["n"]} | **{s["max"]:.1f} %** '
                 f'| {s["median"]:.1f} % | **{s["n_worse"]}** | {s["n_better"]} |')

    for key in sorted(t2, key=lambda k: (k[0] != MOTORS, k[1])):
        row('2 (hand-designed)', key, t2[key])
    for key in sorted(t3, key=lambda k: k[1]):
        row('3 (**trained**)', key, t3[key])

    all_t2 = [e for v in t2.values() for e in v]
    all_t3 = [e for v in t3.values() for e in v]
    obs = [e for k, v in t2.items() if k[0] == OBS for e in v]
    mot = [e for k, v in t2.items() if k[0] == MOTORS for e in v] + all_t3
    sig_obs = [e for e in obs if e[1] < ALPHA and e[0] > 0]
    sig_mot = [e for e in mot if e[1] < ALPHA and e[0] > 0]

    L.append('\n## What this data shows\n')
    L.append('_Every statement below is computed from the rows above; none is fixed text. '
             'Interpretation belongs in design/tier3_design.md, where it can cite which data '
             'root it was drawn from._\n')

    def describe(label, entries):
        if not entries:
            L.append(f'* **{label}:** no paired runs.')
            return
        d = np.array([e[0] for e in entries])
        worse = [e for e in entries if e[1] < ALPHA and e[0] > 0]
        better = [e for e in entries if e[1] < ALPHA and e[0] < 0]
        best = max(entries, key=lambda e: e[0])
        at_gate = [e for e in worse if e[0] >= GATE_PCT]
        L.append(f'* **{label}** ({len(entries)} runs): mean {d.mean():+.1f} %, '
                 f'positive in {(d > 0).sum()}/{len(d)}; significantly worse for the honest '
                 f'team in {len(worse)}, significantly better in {len(better)}; best '
                 f'{best[0]:+.1f} % (`{best[2]}`, p = {best[1]:.4f}); '
                 f'{len(at_gate)} significant and at or above the {GATE_PCT:.0f} % gate.')

    t2_obs = [e for k, v in t2.items() if k[0] == OBS for e in v]
    t2_mot = [e for k, v in t2.items() if k[0] == MOTORS for e in v]
    t3_obs = [e for k, v in t3.items() if k[0] == OBS for e in v]
    t3_mot = [e for k, v in t3.items() if k[0] == MOTORS for e in v]
    describe('Hand-designed, observation channel (Tier 2 liar)', t2_obs)
    describe('Hand-designed, motors only (Tier 2 signflip / waypoint / collusion)', t2_mot)
    describe('Trained, observation channel (Tier 3 rounds 3-5)', t3_obs)
    describe('Trained, motors only (Tier 3 rounds 1-2)', t3_mot)

    for f in sorted({k[1] for k in list(t2) + list(t3)}):
        d = [e[0] for k, v in list(t2.items()) + list(t3.items()) if k[1] == f for e in v]
        if d:
            L.append(f'* f = {f:.3f}: {len(d)} runs, mean {np.mean(d):+.1f} %.')

    if t2_rows:
        L.extend(per_seed_table(t2_rows, t3_lie_rows or {}))
    return '\n'.join(L)


def tier2_rows_by_seed():
    """(channel, seed, degradation %, p) for every Tier 2 run."""
    out = []
    for path in sorted(glob.glob(os.path.join(TIER2_DIR, '*.npz'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        parts = stem.split('_')
        channel = CHANNEL.get(parts[1]) if len(parts) > 1 else None
        m = re.search(r'_s(\d+)_', stem)
        if channel is None or not m:
            continue
        seed = int(m.group(1))
        res = compare(path, os.path.join(TIER1_DIR, f't1_clean_s{seed}.npz'))
        if res is not None:
            if res[2]:
                out.append((channel, seed, res[0], res[1]))
            else:
                _UNPAIRED.append(stem)
    return out


def tier3_lie_rows_by_seed():
    """Trained observation-channel checkpoints, keyed by the baseline seed attacked."""
    out = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(TIER3_DIR, '*lie*.npz'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r'_on_s(\d+)_', stem)
        if not m:
            continue
        seed = int(m.group(1))
        res = compare(path, os.path.join(TIER1_DIR, f't1_clean_s{seed}.npz'))
        if res is not None:
            if res[2]:
                out[seed].append((res[0], res[1], stem))
            else:
                _UNPAIRED.append(stem)
    return out


def main():
    t2, t3 = tier2_groups(), tier3_groups()
    if not t2:
        n_unp = len(set(_UNPAIRED))
        print('No PAIRED Tier 2 trajectories found' +
              (f' ({n_unp} runs excluded as unpaired -- legacy data or seed mismatch).'
               if n_unp else '.'))
        return
    report = build_report(t2, t3, tier2_rows_by_seed(), tier3_lie_rows_by_seed())
    if _UNPAIRED:
        report += (f'\n\n**{len(set(_UNPAIRED))} runs excluded as UNPAIRED** (collected before '
                   'the seeding fix, or seed/checkpoint mismatch with their clean reference). '
                   'They are not evidence either way.\n')
    with open(OUT_MD, 'w') as f:
        f.write(report)
    print(report)
    print(f'\nWrote {OUT_MD}')


if __name__ == '__main__':
    main()
