"""
Evaluate a trained quad-swarm policy for N episodes and save summary statistics.

Usage:
    python evaluate.py --experiment EXP_NAME [options]

Examples:
    # Evaluate one of your seeds (100 episodes)
    python evaluate.py --experiment exp1_baseline_seed1

    # Evaluate all three seeds
    python evaluate.py --experiment exp1_baseline_seed1 --num_episodes 100 --out seed1_eval.json
    python evaluate.py --experiment exp1_baseline_seed2 --num_episodes 100 --out seed2_eval.json
    python evaluate.py --experiment exp1_baseline_seed3 --num_episodes 100 --out seed3_eval.json

    # Evaluate the paper baseline (note the nested train_dir path)
    python evaluate.py \\
        --train_dir train_dir/paper_quads_multi_mix_baseline_8a_attn_v116/single_ \\
        --experiment 00_single_see_0 \\
        --num_episodes 100 --out paper_baseline_eval.json

    # Compare all results after running the above
    python evaluate.py --compare seed1_eval.json seed2_eval.json seed3_eval.json paper_baseline_eval.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import experiment_dir, log

from swarm_rl.train import parse_swarm_cfg, register_swarm_components


# Metrics to highlight in the printed table
KEY_METRICS = [
    "metric/agent_success_rate",
    "metric/agent_col_rate",
    "metric/agent_neighbor_col_rate",
    "metric/agent_deadlock_rate",
    "metric/agent_obst_col_rate",
]


def extract_episode_extra_stats(infos, num_agents):
    """Pull episode_extra_stats out of infos however the env wraps them."""
    if infos is None:
        return {}

    # infos is a list of per-agent dicts
    if isinstance(infos, (list, tuple)) and len(infos) > 0:
        for agent_info in infos:
            if isinstance(agent_info, dict) and "episode_extra_stats" in agent_info:
                return agent_info["episode_extra_stats"]

    # infos is a single dict (batched wrapper collapsed the list)
    if isinstance(infos, dict) and "episode_extra_stats" in infos:
        return infos["episode_extra_stats"]

    return {}


def run_evaluation(cfg, num_episodes, deterministic=True):
    """
    Load a policy from cfg and run num_episodes evaluation episodes.
    Returns a dict mapping metric name -> list of per-episode values.
    """
    cfg = load_from_checkpoint(cfg)
    cfg.num_envs = 1

    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    env_info = extract_env_info(env, cfg)

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(
        Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*"
    )
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])
    log.info("Loaded checkpoint: %s", checkpoints[-1] if checkpoints else "none")

    num_agents = env.num_agents
    rnn_states = torch.zeros(
        [num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device
    )

    # Accumulators
    episode_rewards = []
    episode_lengths = []
    all_extra_stats = defaultdict(list)  # metric_key -> [episode values]

    obs, _ = env.reset()
    episode_reward = None  # initialized from first rew to follow its device
    episode_length = 0
    completed = 0

    log.info("Running %d evaluation episodes...", num_episodes)

    with torch.no_grad():
        while completed < num_episodes:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            actions = policy_outputs["actions"]
            if deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            obs, rew, terminated, truncated, infos = env.step(actions)
            dones = make_dones(terminated, truncated).cpu().numpy()

            if episode_reward is None:
                episode_reward = rew.float().clone()
            else:
                episode_reward += rew.float()
            episode_length += 1

            # The quad env resets all agents simultaneously, so all dones fire at once
            if all(dones):
                ep_reward = episode_reward.mean().item()
                episode_rewards.append(ep_reward)
                episode_lengths.append(episode_length)

                # Grab episode_extra_stats (same for all agents in this env)
                extra = extract_episode_extra_stats(infos, num_agents)
                for key, val in extra.items():
                    if isinstance(val, (int, float, np.floating, np.integer)):
                        all_extra_stats[key].append(float(val))

                completed += 1
                if completed % 10 == 0:
                    log.info(
                        "  Episodes: %d/%d  |  last reward: %.3f",
                        completed, num_episodes, ep_reward,
                    )

                # Reset for next episode
                episode_reward = None
                episode_length = 0
                rnn_states = torch.zeros(
                    [num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device
                )
                obs, _ = env.reset()

    env.close()

    results = {
        "episode_reward": episode_rewards,
        "episode_length": episode_lengths,
    }
    results.update(dict(all_extra_stats))
    return results


def summarize_results(results):
    """Compute mean ± std for each metric across episodes."""
    summary = {}
    for key, vals in results.items():
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        summary[key] = {
            "mean":   round(float(np.mean(arr)), 6),
            "std":    round(float(np.std(arr)), 6),
            "min":    round(float(np.min(arr)), 6),
            "max":    round(float(np.max(arr)), 6),
            "n":      len(arr),
        }
    return summary


def print_summary(summary, experiment, num_episodes):
    col = max((len(k) for k in summary), default=40) + 2
    print(f"\n{'='*72}")
    print(f"  Experiment : {experiment}")
    print(f"  Episodes   : {num_episodes}")
    print(f"{'='*72}")
    print(f"  {'Metric':<{col}} {'Mean':>8}  {'±Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*(col+38)}")

    # Key metrics first
    for m in ["episode_reward", "episode_length"] + KEY_METRICS:
        if m in summary:
            s = summary[m]
            print(f"  {m:<{col}} {s['mean']:>8.4f}  {s['std']:>8.4f}  {s['min']:>8.4f}  {s['max']:>8.4f}")

    # Per-scenario success rates
    scenario_keys = sorted(
        k for k in summary
        if k.endswith("/agent_success_rate") and not k.startswith("metric/")
    )
    if scenario_keys:
        print(f"\n  Per-scenario agent_success_rate:")
        for k in scenario_keys:
            s = summary[k]
            print(f"    {k:<{col-2}} {s['mean']:>8.4f}  ±{s['std']:>7.4f}")

    print(f"{'='*72}\n")


def compare_mode(json_files):
    """Load multiple eval JSON files and print a side-by-side comparison table."""
    records = []
    for f in json_files:
        with open(f) as fh:
            records.append(json.load(fh))

    metrics = ["episode_reward"] + KEY_METRICS
    names = [r["experiment"] for r in records]
    name_w = max(len(n) for n in names) + 2
    metric_w = max(len(m) for m in metrics) + 2

    print(f"\n{'='*( metric_w + name_w * len(records) + 10)}")
    print("COMPARISON TABLE  (mean ± std)")
    print(f"{'Metric':<{metric_w}}", end="")
    for n in names:
        print(f"  {n:^{name_w}}", end="")
    print()
    print("-" * (metric_w + (name_w + 2) * len(records)))

    for m in metrics:
        print(f"{m:<{metric_w}}", end="")
        for r in records:
            s = r.get("summary", {}).get(m)
            if s:
                cell = f"{s['mean']:.3f}±{s['std']:.3f}"
            else:
                cell = "N/A"
            print(f"  {cell:^{name_w}}", end="")
        print()

    print("=" * (metric_w + (name_w + 2) * len(records)))
    print()


def main():
    # Check for --compare mode first (doesn't need SF cfg parsing)
    if "--compare" in sys.argv:
        idx = sys.argv.index("--compare")
        json_files = sys.argv[idx + 1:]
        if not json_files:
            sys.exit("Usage: python evaluate.py --compare file1.json file2.json ...")
        compare_mode(json_files)
        return

    register_swarm_components()

    # Parse extra args before handing off to SampleFactory's parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--num_episodes", type=int, default=100,
                     help="Number of evaluation episodes (default: 100)")
    pre.add_argument("--out", default=None,
                     help="Output JSON file (default: <experiment>_eval.json)")
    pre.add_argument("--stochastic", action="store_true",
                     help="Sample actions stochastically (default: deterministic)")
    known, remaining = pre.parse_known_args()

    # Let SampleFactory parse the rest (--experiment, --train_dir, --env, --algo, etc.)
    sys.argv = [sys.argv[0]] + remaining
    cfg = parse_swarm_cfg(evaluation=True)

    # Force headless
    cfg.no_render = True
    cfg.save_video = False

    out_path = known.out or f"{cfg.experiment}_eval.json"

    results = run_evaluation(cfg, known.num_episodes, deterministic=not known.stochastic)
    summary = summarize_results(results)

    print_summary(summary, cfg.experiment, known.num_episodes)

    output = {
        "experiment":   cfg.experiment,
        "train_dir":    cfg.train_dir,
        "num_episodes": known.num_episodes,
        "deterministic": not known.stochastic,
        "generated_at": datetime.now().isoformat(),
        "summary":      summary,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
