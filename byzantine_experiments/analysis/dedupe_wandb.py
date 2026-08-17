"""
Remove superseded duplicate wandb runs, keeping the newest run of each name.

Re-running a sweep (after a bug fix, say) creates a second wandb run with the same display
name rather than overwriting the first, which leaves the project showing two contradictory
versions of the same config. This deletes the older ones.

Destructive, so it is dry-run by default and prints exactly what it would remove.

Usage:
    python -m byzantine_experiments.analysis.dedupe_wandb                # show only
    python -m byzantine_experiments.analysis.dedupe_wandb --delete       # actually delete
    python -m byzantine_experiments.analysis.dedupe_wandb --delete --name-prefix t1_
"""

import argparse
from collections import defaultdict

from byzantine_experiments.analysis.wandb_logging import DEFAULT_ENTITY, DEFAULT_PROJECT


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--project', default=DEFAULT_PROJECT)
    p.add_argument('--entity', default=DEFAULT_ENTITY)
    p.add_argument('--name-prefix', default='',
                   help='Only consider runs whose name starts with this (e.g. t1_)')
    p.add_argument('--delete', action='store_true',
                   help='Actually delete. Without it, this only reports.')
    args = p.parse_args()

    import wandb
    api = wandb.Api()
    runs = list(api.runs(f'{args.entity}/{args.project}'))
    print(f'{len(runs)} runs in {args.entity}/{args.project}')

    by_name = defaultdict(list)
    for r in runs:
        if r.name and r.name.startswith(args.name_prefix):
            by_name[r.name].append(r)

    stale = []
    for name, group in sorted(by_name.items()):
        if len(group) < 2:
            continue
        # created_at is an ISO-8601 string, so lexicographic sort is chronological.
        group.sort(key=lambda r: r.created_at)
        keep, drop = group[-1], group[:-1]
        print(f'{name}: {len(group)} copies — keep {keep.id} ({keep.created_at}), '
              f"drop {', '.join(r.id for r in drop)}")
        stale.extend(drop)

    if not stale:
        print('No duplicates found.')
        return

    print(f'\n{len(stale)} superseded runs')
    if not args.delete:
        print('Dry run — pass --delete to remove them.')
        return

    deleted = failed = 0
    for r in stale:
        try:
            r.delete()
            deleted += 1
        except Exception as exc:
            print(f'  ! failed to delete {r.name}/{r.id}: {type(exc).__name__}: {exc}')
            failed += 1
    print(f'deleted={deleted} failed={failed}')


if __name__ == '__main__':
    main()
