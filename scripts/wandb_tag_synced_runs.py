"""Set group/tags/name/config on runs created by `wandb sync --sync-tensorboard`.

The tfevents sync path only sets run_id/project/entity -- it cannot set a group,
tags, or a config (see wandb/sync/sync.py, which builds the RunRecord with just
those three fields). So synced runs arrive with metrics but no grouping and an
empty config panel, meaning hyperparameters are not filterable in the UI.

This script backfills all of it from each experiment's train_dir/*/config.json.
Runs created by training with --with_wandb=True already carry their group, tags,
and config, and do not need this.

Usage:
    python scripts/wandb_tag_synced_runs.py --dry-run       # inspect first
    python scripts/wandb_tag_synced_runs.py                 # baseline seeds 1-3
    python scripts/wandb_tag_synced_runs.py --no-config      # skip config backfill
    python scripts/wandb_tag_synced_runs.py \
        --run-id byz1_seed1 --group byz_1_attacker --tags byzantine seed1
"""

import argparse
import json
from pathlib import Path

import wandb

ENTITY = "bmugizi-carnegie-mellon-university"
PROJECT = "byzantine_exp"
TRAIN_DIR = Path(__file__).resolve().parent.parent / "train_dir"

# argparse/bookkeeping noise that is not a hyperparameter
CONFIG_SKIP_KEYS = {"help", "with_wandb", "wandb_user", "wandb_project", "wandb_group",
                    "wandb_job_type", "wandb_tags", "wandb_unique_id", "train_dir"}

# run_id -> (group, job_type, tags)
DEFAULT_RUNS = {
    f"exp1_baseline_seed{s}": ("baseline", "train", ["baseline", "exp1", f"seed{s}", "8agents"])
    for s in (1, 2, 3)
}


def load_config(run_id):
    """Read the sample_factory config.json for a run, or None if unavailable."""
    path = TRAIN_DIR / run_id / "config.json"
    if not path.exists():
        print(f"  no config.json at {path} -- skipping config backfill")
        return None

    with open(path) as f:
        cfg = json.load(f)

    return {k: v for k, v in cfg.items() if k not in CONFIG_SKIP_KEYS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--run-id", help="single run to fix up (else the baseline seeds)")
    p.add_argument("--group")
    p.add_argument("--job-type", default="train")
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--config-from", help="run dir name holding config.json (default: run id)")
    p.add_argument("--no-config", action="store_true", help="do not backfill config.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.run_id:
        if not args.group:
            p.error("--group is required with --run-id")
        targets = {args.run_id: (args.group, args.job_type, args.tags)}
    else:
        targets = DEFAULT_RUNS

    api = wandb.Api()
    for run_id, (group, job_type, tags) in targets.items():
        path = f"{args.entity}/{args.project}/{run_id}"
        try:
            run = api.run(path)
        except Exception as exc:
            # In dry-run we still want to validate the local config parsing even
            # though the run has not been synced to wandb yet.
            if not args.dry_run:
                print(f"SKIP {run_id}: {exc}")
                continue
            run = None
            print(f"{run_id}: NOT YET IN WANDB (sync it first)")

        print(f"{run_id}: group={group!r} job_type={job_type!r} tags={tags}")

        config = None if args.no_config else load_config(args.config_from or run_id)
        if config:
            print(f"  config: {len(config)} keys (seed={config.get('seed')}, "
                  f"env={config.get('env')}, agents={config.get('quads_num_agents')})")

        if args.dry_run:
            continue

        run.group = group
        run.job_type = job_type
        run.tags = sorted(set(run.tags) | set(tags))
        run.name = run_id
        if config:
            run.config.update(config)
        run.update()

    if args.dry_run:
        print("\n(dry run -- nothing written)")


if __name__ == "__main__":
    main()
