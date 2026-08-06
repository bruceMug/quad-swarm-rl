"""
Baseline results reporter for exp1_baseline_seed{1,2,3}.

Two modes:
  1. --from-tb   : Read from summarize_experiment output JSONs (fast, uses training curve)
  2. --from-eval : Read from evaluate.py output JSONs (accurate, held-out episodes)

Usage:
  # After running summarize_experiment.py for each seed:
  python report_baseline.py --from-tb \
      logs/results1.txt logs/results2.txt logs/results3.txt

  # After running evaluate.py for each seed:
  python report_baseline.py --from-eval \
      seed1_eval.json seed2_eval.json seed3_eval.json

  # Run evaluate.py first, then report (all in one go):
  python report_baseline.py --run-eval --train-dir train_dir \
      --seeds 1 2 3 --num-episodes 100

The script prints a notebook-ready table and saves baseline_report.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── metrics that matter for the notebook ──────────────────────────────────────
NOTEBOOK_METRICS = [
    ("episode_reward",                  "Episode reward (mean)"),
    ("metric/agent_success_rate",       "Agent success rate"),
    ("metric/agent_col_rate",           "Agent collision rate"),
    ("metric/agent_neighbor_col_rate",  "Agent neighbor collision rate"),
    ("metric/agent_deadlock_rate",      "Agent deadlock rate"),
    ("metric/agent_obst_col_rate",      "Agent obstacle collision rate"),
]

# tensorboard summarize_experiment uses these key names for the same metrics
TB_METRIC_MAP = {
    "episode_reward":                 "reward/reward",
    "metric/agent_success_rate":      "metric/agent_success_rate",
    "metric/agent_col_rate":          "metric/agent_col_rate",
    "metric/agent_neighbor_col_rate": "metric/agent_neighbor_col_rate",
    "metric/agent_deadlock_rate":     "metric/agent_deadlock_rate",
    "metric/agent_obst_col_rate":     "metric/agent_obst_col_rate",
}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_from_tb(path):
    """Read a summarize_experiment JSON. Returns {metric_key: last_10pct_mean}."""
    d = json.load(open(path))
    summary = d.get("summary", {})
    out = {}
    for eval_key, tb_key in TB_METRIC_MAP.items():
        s = summary.get(tb_key)
        if s:
            out[eval_key] = s["last_10pct_mean"]
    return d.get("experiment", path), out


def load_from_eval(path):
    """Read an evaluate.py JSON. Returns {metric_key: {mean, std, n}}."""
    d = json.load(open(path))
    summary = d.get("summary", {})
    out = {}
    for eval_key, _ in NOTEBOOK_METRICS:
        s = summary.get(eval_key)
        if s:
            out[eval_key] = s
    return d.get("experiment", path), out


# ── evaluation runner ─────────────────────────────────────────────────────────

def run_evaluate(train_dir, seed, num_episodes, out_path):
    """Run evaluate.py for one seed and return the output JSON path."""
    exp = f"exp1_baseline_seed{seed}"
    cmd = [
        sys.executable, "evaluate.py",
        "--algo", "APPO",
        "--env", "quadrotor_multi",
        "--train_dir", train_dir,
        "--experiment", exp,
        "--replay_buffer_sample_prob", "0",
        "--quads_use_numba", "False",
        "--num_episodes", str(num_episodes),
        "--out", str(out_path),
    ]
    print(f"\n[run-eval] seed{seed}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_path


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate_eval(records):
    """
    records: list of {metric_key: {mean, std, n}} dicts (one per seed)
    Returns {metric_key: {seed_means: [...], overall_mean, overall_std, n_seeds}}
    """
    agg = {}
    for rec in records:
        for key, s in rec.items():
            agg.setdefault(key, []).append(s["mean"])
    result = {}
    for key, means in agg.items():
        result[key] = {
            "seed_means": means,
            "overall_mean": float(np.mean(means)),
            "overall_std": float(np.std(means)),
            "n_seeds": len(means),
        }
    return result


def aggregate_tb(records):
    """
    records: list of {metric_key: scalar} dicts (one per seed)
    Returns same shape as aggregate_eval.
    """
    agg = {}
    for rec in records:
        for key, val in rec.items():
            agg.setdefault(key, []).append(val)
    result = {}
    for key, vals in agg.items():
        result[key] = {
            "seed_means": vals,
            "overall_mean": float(np.mean(vals)),
            "overall_std": float(np.std(vals)),
            "n_seeds": len(vals),
        }
    return result


# ── printing ──────────────────────────────────────────────────────────────────

def print_table(agg, seed_names, source_label):
    w_metric = 36
    w_seed   = 14
    w_agg    = 18

    header_seeds = "".join(f"{n:>{w_seed}}" for n in seed_names)
    header = f"\n{'='*(w_metric + w_seed*len(seed_names) + w_agg + 4)}"
    print(header)
    print(f"  BASELINE RESULTS — {source_label}")
    print(f"{'='*(w_metric + w_seed*len(seed_names) + w_agg + 4)}")
    print(f"  {'Metric':<{w_metric}}{header_seeds}  {'Mean±Std(seeds)':>{w_agg}}")
    print(f"  {'-'*(w_metric + w_seed*len(seed_names) + w_agg + 2)}")

    for key, label in NOTEBOOK_METRICS:
        if key not in agg:
            continue
        a = agg[key]
        seed_cells = "".join(f"{v:>{w_seed}.4f}" for v in a["seed_means"])
        agg_cell   = f"{a['overall_mean']:.4f}±{a['overall_std']:.4f}"
        print(f"  {label:<{w_metric}}{seed_cells}  {agg_cell:>{w_agg}}")

    print(f"{'='*(w_metric + w_seed*len(seed_names) + w_agg + 4)}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-tb",   action="store_true", help="Read from summarize_experiment JSONs")
    mode.add_argument("--from-eval", action="store_true", help="Read from evaluate.py JSONs")
    mode.add_argument("--run-eval",  action="store_true", help="Run evaluate.py, then report")

    parser.add_argument("files", nargs="*", help="Input JSON/txt files (for --from-tb or --from-eval)")
    parser.add_argument("--train-dir", default="train_dir", help="train_dir path (for --run-eval)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--out", default="baseline_report.json", help="Output JSON file")
    args = parser.parse_args()

    seed_names = []
    records    = []

    if args.from_tb:
        if not args.files:
            parser.error("--from-tb requires input files")
        for f in args.files:
            name, rec = load_from_tb(f)
            seed_names.append(name)
            records.append(rec)
        agg = aggregate_tb(records)
        source = "tensorboard last-10% means"

    elif args.from_eval:
        if not args.files:
            parser.error("--from-eval requires input files")
        for f in args.files:
            name, rec = load_from_eval(f)
            seed_names.append(name)
            records.append(rec)
        agg = aggregate_eval(records)
        source = "held-out evaluation (100 episodes each)"

    else:  # --run-eval
        eval_jsons = []
        for seed in args.seeds:
            out_path = Path(f"seed{seed}_eval_final.json")
            run_evaluate(args.train_dir, seed, args.num_episodes, out_path)
            name, rec = load_from_eval(out_path)
            seed_names.append(name)
            records.append(rec)
            eval_jsons.append(str(out_path))
        agg = aggregate_eval(records)
        source = f"held-out evaluation ({args.num_episodes} episodes each)"
        print(f"\nEval JSONs written: {eval_jsons}")

    print_table(agg, seed_names, source)

    # Save
    output = {
        "source": source,
        "seeds": seed_names,
        "aggregated": agg,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full report saved to: {args.out}\n")

    # Print notebook-ready copy-paste block
    print("── Notebook copy-paste block ──────────────────────────────────────")
    print("Baseline (3 seeds, 1B env-steps, quads_mode=mix, 8 agents, attention encoder):")
    for key, label in NOTEBOOK_METRICS:
        if key not in agg:
            continue
        a = agg[key]
        seeds_str = " / ".join(f"{v:.3f}" for v in a["seed_means"])
        print(f"  {label}: {a['overall_mean']:.3f} ± {a['overall_std']:.3f}  "
              f"(per seed: {seeds_str})")
    print("───────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
