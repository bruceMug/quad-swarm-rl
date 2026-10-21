"""
Cross-seed transfer matrix (spec design item 4): adversary-seed x baseline-seed degradation.

Each cell is the adversary's final checkpoint evaluated against one baseline seed's honest team,
n = 200, paired against a 200-episode clean run at that seed. The diagonal is the adversary's
own seed (its confirmatory run); off-diagonal cells are the transfer test.

Reading, declared in advance: an adversary **transfers** to a seed if its damage there is
significant (paired 95 % CI above 0). The spec is explicit that either outcome is a result --
a transferring adversary is a general one, a non-transferring one is a seed-specific exploit
and the paper must say so.

    python -m byzantine_experiments.analysis.transfer_matrix --runs adv_seed3_stealth0_lieonly ...
"""

import argparse
import os
import re

from byzantine_experiments.agents.tier3_learned import final_milestone_m
from byzantine_experiments.analysis.paired import PairingError, paired_degradation
from byzantine_experiments.paths import RESULTS_ROOT

CONFIRM_DIR = os.path.join(RESULTS_ROOT, 'tier3_confirm')
SEEDS = (1, 2, 3)


def cell_path(run, bseed, n):
    aseed = int(re.search(r'seed(\d+)', run).group(1))
    ms = f'm{final_milestone_m(os.path.join("train_dir", run)):03d}'
    name = (f'confirm_{run}_{ms}_n{n}' if bseed == aseed
            else f'confirm_{run}_{ms}_on_s{bseed}_n{n}')
    return os.path.join(CONFIRM_DIR, name + '.npz'), aseed


def matrix(runs, n=200):
    """{run: {bseed: result dict or None}}, plus the adversary's own seed."""
    out = {}
    for run in runs:
        row = {}
        for b in SEEDS:
            path, aseed = cell_path(run, b, n)
            clean = os.path.join(CONFIRM_DIR, f'confirm_clean_s{b}_n{n}.npz')
            if not (os.path.exists(path) and os.path.exists(clean)):
                row[b] = None
                continue
            try:
                row[b] = paired_degradation(path, clean)
            except PairingError as e:
                row[b] = dict(error=str(e))
        out[run] = (aseed, row)
    return out


def transfer_tested(run, n=200):
    """The gate's fifth criterion: evaluated against at least two OTHER baseline seeds."""
    aseed, row = matrix([run], n)[run]
    return sum(1 for b, r in row.items() if b != aseed and r and 'error' not in r) >= 2


def report(runs, n=200):
    L = [f'Cross-seed transfer, final checkpoints, n = {n} per cell, paired\n',
         f"  {'adversary':34s} " + ' '.join(f'{"baseline s" + str(b):>24s}' for b in SEEDS)]
    for run, (aseed, row) in matrix(runs, n).items():
        cells = []
        for b in SEEDS:
            r = row[b]
            if r is None:
                cells.append(f'{"(not run)":>24s}')
            elif 'error' in r:
                # Say which failure it is: a genuine pairing failure and a too-small test
                # sample need very different responses.
                label = 'TOO FEW EPISODES' if 'too few' in r['error'] else 'UNPAIRED'
                cells.append(f'{label:>24s}')
            else:
                tag = '*' if b == aseed else ('T' if r['ci_lo'] > 0 else ' ')
                cells.append(f"{r['deg_pct']:+5.1f} [{r['ci_lo']:+5.1f},{r['ci_hi']:+5.1f}]{tag:>2s}")
        L.append(f'  {run.replace("adv_", ""):34s} ' + ' '.join(cells))
    L.append('\n  * own seed (the confirmatory run).  T = transfers: significant damage on a seed '
             'the adversary was not trained against.')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--n-episodes', type=int, default=200)
    args = ap.parse_args()
    text = report(args.runs, args.n_episodes)
    print(text)
    with open(os.path.join(RESULTS_ROOT, 'tier3_transfer.md'), 'w') as f:
        f.write('# Tier 3 cross-seed transfer\n\n```\n' + text + '\n```\n')


if __name__ == '__main__':
    main()
