"""
Recompute the `metrics` block of existing run sidecars from their `.npz`, in place.

Needed whenever `analysis/metrics.py` gains a field after a sweep has already run. The
trajectories are the ground truth and they are still on disk, so there is never a reason to
re-simulate for a new *derived* quantity — this replays the metric computation instead.

It has been needed twice so far:
  * Tier 2 added `attacker_survived`, `attacker_below_guard_frac` and the cumulative
    team-reward curve, none of which existed when the Tier 1 clean references were collected.
    Without the curve, Tier 2's time-to-impact is `n/a` for every config.
  * `attacker_below_guard_frac` reconstructs the survival-guard trigger rate from the
    trajectory, which is more trustworthy than the agent's own counters.

Usage:
    # just the three clean references (fast)
    python -m byzantine_experiments.analysis.refresh_metrics \
        --results-dir byzantine_experiments/results/tier1 --pattern 't1_clean_s*.json'

    # a whole sweep (~5 s per run)
    python -m byzantine_experiments.analysis.refresh_metrics \
        --results-dir byzantine_experiments/results/tier2
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from byzantine_experiments.analysis.metrics import compute_metrics


def refresh(json_path, dry_run=False):
    npz_path = os.path.splitext(json_path)[0] + '.npz'
    if not os.path.exists(npz_path):
        return False, 'no .npz alongside'

    with open(json_path) as f:
        sidecar = json.load(f)

    cfg = sidecar.get('byzantine_wrapper_config') or {}
    low = (cfg.get('action_low') or [-1.0])[0]
    high = (cfg.get('action_high') or [1.0])[0]

    with np.load(npz_path) as data:
        metrics = compute_metrics({k: data[k] for k in data.files},
                                  action_low=low, action_high=high)

    before = set(sidecar.get('metrics') or {})
    after = {k: v for k, v in metrics.items() if k != 'per_episode'}
    added = sorted(set(after) - before)

    if not dry_run:
        sidecar['metrics'] = after
        with open(json_path, 'w') as f:
            json.dump(sidecar, f, indent=2)
    return True, f'+{len(added)} fields' + (f' ({", ".join(added[:3])}…)' if added else '')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--results-dir', required=True)
    p.add_argument('--pattern', default='*.json')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, args.pattern)))
    if not paths:
        raise SystemExit(f'nothing matched {args.results_dir}/{args.pattern}')
    print(f'{len(paths)} sidecars to refresh'
          f'{" (dry run)" if args.dry_run else ""}')

    ok = skipped = 0
    t0 = time.time()
    for path in paths:
        done, note = refresh(path, dry_run=args.dry_run)
        name = os.path.basename(path)
        if done:
            ok += 1
            print(f'  {name:44s} {note}')
        else:
            skipped += 1
            print(f'  ! {name:42s} skipped: {note}')
    print(f'refreshed={ok} skipped={skipped} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
