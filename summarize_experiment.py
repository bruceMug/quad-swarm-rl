"""
Usage:
    python summarize_experiment.py <experiment_name> [--train_dir PATH] [--tail 0.1] [--out FILE]

Examples:
    python summarize_experiment.py exp1_baseline_seed1
    python summarize_experiment.py exp1_baseline_seed1 --tail 0.2 --out results.txt
    python summarize_experiment.py exp1_baseline_seed1 --train_dir /path/to/train_dir
"""

import argparse
import os
import sys
import json
from datetime import datetime

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("tensorboard not found. Run: pip install setuptools tensorboard")


def load_scalars(event_file):
    ea = EventAccumulator(event_file)
    ea.Reload()
    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        try:
            vals = [e.value for e in ea.Scalars(tag)]
            scalars[tag] = vals
        except Exception:
            pass
    return scalars


def find_event_file(logdir):
    for root, dirs, files in os.walk(logdir):
        for f in sorted(files):
            if f.startswith("events.out"):
                return os.path.join(root, f)
    return None


def tail_mean(values, tail_frac):
    start = max(0, int(len(values) * (1.0 - tail_frac)))
    return float(np.mean(values[start:]))


def summarize(scalars, tail_frac):
    summary = {}
    for tag, vals in scalars.items():
        if not vals:
            continue
        summary[tag] = {
            "final":    round(vals[-1], 6),
            f"last_{int(tail_frac*100)}pct_mean": round(tail_mean(vals, tail_frac), 6),
            "max":      round(float(np.max(vals)), 6),
            "min":      round(float(np.min(vals)), 6),
            "n_points": len(vals),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize a SampleFactory experiment to a file.")
    parser.add_argument("experiment", help="Experiment name (subfolder inside train_dir)")
    parser.add_argument("--train_dir", default="train_dir", help="Path to train_dir (default: train_dir)")
    parser.add_argument("--tail", type=float, default=0.1,
                        help="Fraction of tail to average for 'final performance' (default: 0.1 = last 10%%)")
    parser.add_argument("--out", default=None,
                        help="Output file path (default: <experiment>_summary.json)")
    args = parser.parse_args()

    logdir = os.path.join(args.train_dir, args.experiment)
    if not os.path.isdir(logdir):
        sys.exit(f"Directory not found: {logdir}")

    event_file = find_event_file(logdir)
    if event_file is None:
        sys.exit(f"No tensorboard event file found in {logdir}")

    print(f"Loading: {event_file}")
    scalars = load_scalars(event_file)
    if not scalars:
        sys.exit("No scalar data found in event file.")

    summary = summarize(scalars, args.tail)

    # Print key metrics to console
    key_metrics = [
        "metric/agent_success_rate",
        "metric/agent_col_rate",
        "metric/agent_neighbor_col_rate",
        "metric/agent_deadlock_rate",
        "metric/agent_obst_col_rate",
        "reward/reward",
        "reward/reward_max",
        "reward/reward_min",
        "policy_stats/avg_true_objective",
        "policy_stats/avg_num_collisions",
        "policy_stats/avg_num_collisions_after_settle",
        "perf/_fps",
        "train/entropy",
        "train/loss",
    ]

    col_w = max(len(m) for m in key_metrics) + 2
    tail_key = f"last_{int(args.tail*100)}pct_mean"

    print(f"\n{'Metric':<{col_w}} {'Final':>10}  {tail_key:>20}  {'Max':>10}  {'Min':>10}")
    print("-" * (col_w + 58))
    for m in key_metrics:
        if m in summary:
            s = summary[m]
            print(f"{m:<{col_w}} {s['final']:>10.4f}  {s[tail_key]:>20.4f}  {s['max']:>10.4f}  {s['min']:>10.4f}")
        else:
            print(f"{m:<{col_w}} {'N/A':>10}")

    # Also print per-scenario success rates
    scenario_metrics = [t for t in summary if t.endswith("/agent_success_rate") and not t.startswith("metric/")]
    if scenario_metrics:
        print(f"\nPer-scenario agent_success_rate (last {int(args.tail*100)}% mean):")
        for t in sorted(scenario_metrics):
            print(f"  {t:<55} {summary[t][tail_key]:.4f}")

    # Save full summary to JSON
    out_path = args.out or f"{args.experiment}_summary.json"
    output = {
        "experiment": args.experiment,
        "train_dir": args.train_dir,
        "event_file": event_file,
        "tail_fraction": args.tail,
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull summary saved to: {out_path}")
    print(f"Total metrics saved: {len(summary)}")


if __name__ == "__main__":
    main()
