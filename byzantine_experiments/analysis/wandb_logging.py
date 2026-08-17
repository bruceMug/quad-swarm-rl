"""
Weights & Biases logging for Byzantine rollout runs.

Conventions follow README.md's "Project / run organization" section, so Byzantine runs land
next to the baseline training runs and are directly comparable:

    project    byzantine_exp
    entity     bmugizi-carnegie-mellon-university   (the `--wandb_user` in sample_factory)
    group      the experimental condition, e.g. tier1_gaussian   -- aggregates seeds
    job_type   rollout
    tags       seed1, f0125, sigma0.25, 8agents, mix, tier1, ...

The design goal is that no one ever has to open a JSON file to write the paper: every scalar
is in the run summary, every per-episode series is in a Table, and the severity knobs are in
the config so wandb can group and sweep on them directly.
"""

import os

import numpy as np

DEFAULT_PROJECT = 'byzantine_exp'
DEFAULT_ENTITY = 'bmugizi-carnegie-mellon-university'


def _flatten(prefix, obj, out):
    """Flatten nested dicts into dotted keys so wandb config stays queryable."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f'{prefix}.{k}' if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        out[prefix] = ','.join(str(x) for x in obj) if obj else ''
    else:
        out[prefix] = obj
    return out


def log_rollout(sidecar, metrics, run_name, group, job_type='rollout', tags=(),
                project=None, entity=None, notes=None, mode=None):
    """
    Create one wandb run for one rollout config and log everything about it.

    Args:
        sidecar: the run-provenance dict collect.py writes next to the .npz.
        metrics: output of analysis.metrics.compute_metrics (including `per_episode`).
        run_name: wandb display name; should be unique and decodable, e.g.
            `t1_gaussian_s1_sigma0.25_f0125`.
        group: the experimental condition (NOT the seed) -- this is what makes wandb
            aggregate seeds into one mean+-band.
    Returns the wandb run URL, or None if wandb is unavailable.
    """
    try:
        import wandb
    except ImportError:
        print('[wandb] not installed; skipping logging')
        return None

    project = project or os.environ.get('BYZ_WANDB_PROJECT', DEFAULT_PROJECT)
    entity = entity or os.environ.get('BYZ_WANDB_ENTITY', DEFAULT_ENTITY)

    config = _flatten('', {k: v for k, v in sidecar.items()
                           if k not in ('episode_rewards', 'arrays', 'metrics')}, {})
    config.update(_flatten('array', sidecar.get('arrays', {}), {}))

    run = wandb.init(
        project=project, entity=entity, name=run_name, group=group,
        job_type=job_type, tags=list(tags), config=config, notes=notes,
        reinit=True, mode=mode,
    )

    per_episode = metrics.get('per_episode', {})
    summary = {k: v for k, v in metrics.items() if k != 'per_episode'}

    # Scalars -> run summary, so they are sortable/filterable in the runs table.
    for key, value in summary.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            run.summary[key] = float(value)

    # Per-episode series -> a Table (queryable) and histograms (distribution at a glance).
    # Episode reward here is heavily left-tailed, so the distribution matters more than the
    # mean; logging both means that is visible without re-deriving it.
    if per_episode:
        columns = ['episode'] + sorted(per_episode)
        n = len(next(iter(per_episode.values())))

        # Step-indexed series FIRST, so wandb draws per-episode curves. Tables and
        # histograms go afterwards without an explicit step -- logging them first would
        # pin the step counter and make these look like they all happened at step 0.
        for i in range(n):
            run.log({f'episode/{k}': float(per_episode[k][i]) for k in per_episode}, step=i)

        table = wandb.Table(columns=columns)
        for i in range(n):
            table.add_data(i, *[float(per_episode[c][i]) for c in columns[1:]])
        run.log({'episodes': table})

        for key, vals in per_episode.items():
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                run.log({f'hist/{key}': wandb.Histogram(vals)})

    url = run.get_url()
    run.finish()
    return url


def log_summary_table(rows, run_name, group, columns=None, project=None, entity=None,
                      tags=(), job_type='analysis', extra_summary=None, artifacts=()):
    """
    Log the cross-run severity calibration table as a single analysis run.

    `rows` is a list of dicts (one per rollout config). Also uploads any files in
    `artifacts` (the markdown/CSV versions of the table) so the paper numbers are one click
    away rather than buried in the repo.
    """
    try:
        import wandb
    except ImportError:
        print('[wandb] not installed; skipping logging')
        return None

    project = project or os.environ.get('BYZ_WANDB_PROJECT', DEFAULT_PROJECT)
    entity = entity or os.environ.get('BYZ_WANDB_ENTITY', DEFAULT_ENTITY)

    columns = columns or sorted({k for r in rows for k in r})
    run = wandb.init(project=project, entity=entity, name=run_name, group=group,
                     job_type=job_type, tags=list(tags), reinit=True)

    table = wandb.Table(columns=list(columns))
    for r in rows:
        table.add_data(*[r.get(c) for c in columns])
    run.log({'calibration_table': table})

    for key, value in (extra_summary or {}).items():
        run.summary[key] = value

    for path in artifacts:
        if os.path.exists(path):
            run.save(path, policy='now')

    url = run.get_url()
    run.finish()
    return url
