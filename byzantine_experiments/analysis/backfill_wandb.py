"""
Log already-collected rollout runs to wandb, without re-running the simulation.

Needed because a wandb failure during a sweep does not invalidate the rollout -- collect.py
writes the `.npz` and `.json` before it logs, and treats a logging error as non-fatal. This
script replays the logging step from what is on disk.

It skips runs already present in the wandb project (matched by run name), so it is safe to
run repeatedly; `--force` re-logs them anyway.

Usage:
    python -m byzantine_experiments.analysis.backfill_wandb                    # tier1 dir
    python -m byzantine_experiments.analysis.backfill_wandb --results-dir DIR
    python -m byzantine_experiments.analysis.backfill_wandb --dry-run
"""

import argparse
import glob
import json
import os

import numpy as np

from byzantine_experiments.analysis.metrics import compute_metrics
from byzantine_experiments.analysis.wandb_logging import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    log_rollout,
)

DEFAULT_RESULTS = 'byzantine_experiments/results/tier1'


def group_for(sidecar, stem):
    """Reconstruct the wandb group the sweep would have used for this run."""
    tier = sidecar.get('tier')
    if tier is None:
        return 'clean_baseline'

    cfg = sidecar.get('tier_config', {})
    n_byz = len(sidecar.get('compromised_indices', []))
    n_agents = sidecar.get('num_agents', 8)
    f_label = f'{n_byz / n_agents:.4f}'.replace('.', '').rstrip('0') or '0'
    # Match the sweep's f labels exactly: 0.125 -> f0125, 0.25 -> f025
    f_label = {'01250': '0125', '0125': '0125', '025': '025', '02500': '025'}.get(f_label, f_label)

    if tier == 'tier1_gaussian':
        return f"tier1_gaussian_sigma{cfg.get('sigma')}_f{f_label}"
    if tier == 'tier1_ou':
        return f"tier1_ou_sigmaou{cfg.get('sigma_ou')}_f{f_label}"
    if tier == 'tier1_dropout':
        return f"tier1_dropout_p{cfg.get('p_drop')}_{cfg.get('mode')}_f{f_label}"
    if tier == 'tier1_stall':
        return (f"tier1_stall_d{cfg.get('duty_cycle')}_ms{cfg.get('mean_stall_s')}"
                f"_{cfg.get('mode')}_f{f_label}")
    return tier


def tags_for(sidecar):
    tier = sidecar.get('tier')
    n_agents = sidecar.get('num_agents', 8)
    tags = [
        f"seed{sidecar.get('seed')}",
        f'{n_agents}agents',
        sidecar.get('quads_mode', ''),
        sidecar.get('experiment', ''),
        f"n_ep{sidecar.get('n_episodes')}",
        'backfilled',
    ]
    if tier is None:
        tags += ['clean', 'f0']
    else:
        n_byz = len(sidecar.get('compromised_indices', []))
        tags += [tier, tier.split('_')[0], f'nbyz{n_byz}',
                 'f' + f'{n_byz / n_agents:.4f}'.replace('.', '')]
        tags += [f'{k}{v}' for k, v in sorted(sidecar.get('tier_config', {}).items())]
    return [t for t in tags if t]


def existing_run_names(project, entity):
    """Names already in the wandb project, so backfill is idempotent."""
    try:
        import wandb
        api = wandb.Api()
        return {r.name for r in api.runs(f'{entity}/{project}')}
    except Exception as exc:
        print(f'[wandb] could not list existing runs ({type(exc).__name__}: {exc}); '
              f'proceeding without the skip check')
        return set()


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--results-dir', default=DEFAULT_RESULTS)
    p.add_argument('--project', default=DEFAULT_PROJECT)
    p.add_argument('--entity', default=DEFAULT_ENTITY)
    p.add_argument('--force', action='store_true', help='Re-log runs already in wandb')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    jsons = sorted(glob.glob(os.path.join(args.results_dir, '*.json')))
    if not jsons:
        raise SystemExit(f'No result sidecars in {args.results_dir}')

    already = set() if args.force else existing_run_names(args.project, args.entity)
    print(f'{len(jsons)} runs on disk; {len(already)} already in wandb')

    logged = skipped = failed = 0
    for json_path in jsons:
        stem = os.path.splitext(os.path.basename(json_path))[0]
        if stem in already:
            skipped += 1
            continue

        with open(json_path) as f:
            sidecar = json.load(f)

        npz_path = os.path.splitext(json_path)[0] + '.npz'
        if os.path.exists(npz_path):
            # Recompute so the per-episode series (histograms, curves) are available.
            with np.load(npz_path) as data:
                cfg = sidecar.get('byzantine_wrapper_config') or {}
                low = (cfg.get('action_low') or [-1.0])[0]
                high = (cfg.get('action_high') or [1.0])[0]
                metrics = compute_metrics(dict(data), action_low=low, action_high=high)
        elif 'metrics' in sidecar:
            metrics = dict(sidecar['metrics'])   # scalars only; no per-episode series
            print(f'  {stem}: no .npz, logging summary scalars only')
        else:
            print(f'  ! {stem}: neither .npz nor a metrics block; skipping')
            failed += 1
            continue

        group = group_for(sidecar, stem)
        if args.dry_run:
            print(f'  would log {stem:42s} group={group}')
            logged += 1
            continue

        try:
            url = log_rollout(sidecar, metrics, run_name=stem, group=group,
                              job_type='rollout', tags=tags_for(sidecar),
                              project=args.project, entity=args.entity,
                              notes=f'Backfilled from {json_path}')
            print(f'  logged {stem:42s} -> {url}')
            logged += 1
        except Exception as exc:
            print(f'  ! {stem} FAILED: {type(exc).__name__}: {exc}')
            failed += 1

    print(f'\nlogged={logged} skipped={skipped} failed={failed}')


if __name__ == '__main__':
    main()
