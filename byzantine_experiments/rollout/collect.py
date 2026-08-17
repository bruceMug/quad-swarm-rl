"""
Collect Byzantine rollout trajectories from a trained quad-swarm checkpoint.

Loads a frozen policy, wraps the env with ByzantineEnvWrapper, runs N episodes and writes
the trajectory tensors Phase 3 consumes, plus a sidecar .json with the full run provenance.

Example:
    python -m byzantine_experiments.rollout.collect \
      --checkpoint train_dir/exp1_baseline_seed1 \
      --tier tier0_noop \
      --tier-config '{}' \
      --compromised-indices 0 \
      --n-episodes 50 \
      --seed 1 \
      --out byzantine_experiments/results/t0_noop_s1.npz
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.utils import log

from swarm_rl.train import parse_swarm_cfg, register_swarm_components

import byzantine_experiments.agents  # noqa: F401  (populates BYZANTINE_REGISTRY)
from byzantine_experiments.agents.base import BYZANTINE_REGISTRY
from byzantine_experiments.analysis.metrics import compute_metrics
from byzantine_experiments.analysis.wandb_logging import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    log_rollout,
)
from byzantine_experiments.rollout.env_factory import make_byzantine_env

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_sha():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:  # provenance is nice to have, not worth crashing a 40-minute rollout
        return 'unknown'


def quad_state(env):
    """Current per-drone position and velocity, straight from the dynamics objects.

    Read from `envs[i].dynamics` rather than the env's cached `pos`/`vel` arrays because
    `vel` is only refreshed inside step(), so it is stale on the first step of an episode.

    Takes the OUTER env and resolves `.unwrapped` on every call. Do not hoist that lookup
    out of the loop: ExperienceReplayWrapper replaces its `self.env` with a deepcopy of a
    buffered snapshot whenever it replays an episode (quad_experience_replay.py:167-190),
    which happens on ~75% of resets at the trained config's replay_buffer_sample_prob=0.75.
    A cached `unwrapped` therefore ends up pointing at an env that is no longer being
    stepped, and silently records frozen positions.
    """
    unwrapped = env.unwrapped
    pos = np.array([e.dynamics.pos for e in unwrapped.envs], dtype=np.float32)
    vel = np.array([e.dynamics.vel for e in unwrapped.envs], dtype=np.float32)
    return pos, vel


def step_collisions(env):
    """Drone-drone collisions registered on the step just taken (a pair counts once).

    Resolves `.unwrapped` per call for the same reason as quad_state().
    """
    last = getattr(env.unwrapped, 'last_step_unique_collisions', None)
    if last is None:
        return 0
    return int(np.asarray(last).size // 2)


def build_cfg(checkpoint, extra_argv):
    """Resolve a sample_factory cfg from a `train_dir/experiment` checkpoint path."""
    checkpoint = checkpoint.rstrip('/')
    train_dir, experiment = os.path.split(checkpoint)
    if not train_dir:
        train_dir = 'train_dir'

    argv = [
        '--env=quadrotor_multi',
        '--algo=APPO',
        f'--train_dir={train_dir}',
        f'--experiment={experiment}',
    ] + list(extra_argv)

    cfg = parse_swarm_cfg(argv=argv, evaluation=True)
    cfg = load_from_checkpoint(cfg)
    cfg.num_envs = 1
    cfg.no_render = True
    cfg.save_video = False
    cfg.quads_render = False
    return cfg


def load_policy(cfg, env, device):
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest='checkpoint', best='best')[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f'{name_prefix}_*')
    if not checkpoints:
        raise FileNotFoundError(f'no {name_prefix}_* checkpoints under {Learner.checkpoint_dir(cfg, policy_id)}')
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])
    log.info('Loaded checkpoint: %s', checkpoints[-1])
    return actor_critic, checkpoints[-1]


def run_rollouts(cfg, byzantine_kwargs, n_episodes, seed, deterministic=True):
    """Run n_episodes and return (per-episode trajectory dicts, wrapper, extra summary)."""
    # The quadrotor env draws its spawn/goal randomisation from the global numpy RNG.
    np.random.seed(seed)
    torch.manual_seed(seed)

    env, byz = make_byzantine_env(cfg, byzantine_kwargs)
    env_info = extract_env_info(env, cfg)

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic, checkpoint_path = load_policy(cfg, env, device)

    num_agents = env.num_agents
    rnn_states = torch.zeros([num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    episodes = []
    episode_rewards = []
    cur = None

    def new_episode_buffers():
        return dict(obs=[], actions_honest=[], actions_executed=[], rewards=[],
                    positions=[], velocities=[], collisions=[])

    obs, _ = env.reset()
    cur = new_episode_buffers()
    episode_reward = None
    completed = 0

    log.info('Collecting %d episodes | tier=%s | compromised=%s',
             n_episodes, byz.tier_name if byz else 'none',
             byz.compromised_indices if byz else [])

    with torch.no_grad():
        while completed < n_episodes:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            actions = policy_outputs['actions']
            if deterministic:
                actions = argmax_actions(actor_critic.action_distribution())
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs['new_rnn_states']

            # State the action was taken in -- aligned with `obs`, hence read pre-step.
            pos, vel = quad_state(env)
            obs_now = obs['obs'].cpu().numpy() if isinstance(obs, dict) else np.asarray(obs)

            obs_next, rew, terminated, truncated, infos = env.step(actions)
            dones = make_dones(terminated, truncated).cpu().numpy()

            executed = byz.last_executed_actions if byz else np.asarray(actions)
            honest = byz.last_honest_actions if byz else np.asarray(actions)

            cur['obs'].append(obs_now.astype(np.float32))
            cur['actions_honest'].append(np.asarray(honest, dtype=np.float32))
            cur['actions_executed'].append(np.asarray(executed, dtype=np.float32))
            cur['rewards'].append(rew.float().cpu().numpy().astype(np.float32))
            cur['positions'].append(pos)
            cur['velocities'].append(vel)
            cur['collisions'].append(step_collisions(env))

            if episode_reward is None:
                episode_reward = rew.float().clone()
            else:
                episode_reward += rew.float()

            obs = obs_next

            # The quad env terminates all agents simultaneously.
            if all(dones):
                episodes.append({k: np.asarray(v) for k, v in cur.items()})
                episode_rewards.append(float(episode_reward.mean().item()))
                completed += 1
                if completed % 5 == 0 or completed == n_episodes:
                    log.info('  episode %d/%d | len=%d | reward=%.3f',
                             completed, n_episodes, len(cur['rewards']), episode_rewards[-1])

                cur = new_episode_buffers()
                episode_reward = None
                rnn_states = torch.zeros([num_agents, get_rnn_size(cfg)],
                                         dtype=torch.float32, device=device)
                obs, _ = env.reset()

    summary = dict(
        checkpoint_file=str(checkpoint_path),
        episode_reward_mean=float(np.mean(episode_rewards)),
        episode_reward_std=float(np.std(episode_rewards)),
        episode_reward_min=float(np.min(episode_rewards)),
        episode_reward_max=float(np.max(episode_rewards)),
        episode_rewards=[round(r, 6) for r in episode_rewards],
    )
    byz_mask = byz.byzantine_mask if byz else np.zeros(num_agents, dtype=bool)
    byz_config = byz.config_summary() if byz else None

    env.close()
    return episodes, byz_mask, byz_config, summary


def pad_and_stack(episodes):
    """
    Stack ragged per-episode arrays into (n_episodes, T_max, ...) tensors, NaN-padded.

    Episodes can end at different lengths, and Phase 3's windowing needs to know where the
    real data stops -- so pad rather than truncate and record the true lengths.
    """
    lengths = np.array([len(ep['rewards']) for ep in episodes], dtype=np.int32)
    t_max = int(lengths.max())
    out = {}

    for key in episodes[0]:
        sample = episodes[0][key]
        trailing = sample.shape[1:]
        arr = np.full((len(episodes), t_max) + trailing, np.nan, dtype=np.float32)
        for e, ep in enumerate(episodes):
            arr[e, :len(ep[key])] = ep[key].astype(np.float32)
        out[key] = arr

    out['episode_lengths'] = lengths
    return out


def main():
    register_swarm_components()

    p = argparse.ArgumentParser(
        description='Collect Byzantine rollout trajectories from a trained checkpoint.')
    p.add_argument('--checkpoint', required=True,
                   help='Path of the form <train_dir>/<experiment>, e.g. train_dir/exp1_baseline_seed1')
    p.add_argument('--tier', default='tier0_noop',
                   help=f'Byzantine tier registry key. Registered: {sorted(BYZANTINE_REGISTRY)}')
    p.add_argument('--tier-config', default='{}', help='JSON blob of kwargs for the tier constructor')
    p.add_argument('--compromised-indices', default='0',
                   help="Comma-separated agent indices, e.g. '0' or '0,3'. Empty string = none.")
    p.add_argument('--assignment-mode', default='fixed', choices=['fixed', 'random'],
                   help='fixed: same indices every episode (default). random: resampled per episode.')
    p.add_argument('--n-episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', required=True, help='Output .npz path (a .json sidecar is written alongside)')
    p.add_argument('--stochastic', action='store_true',
                   help='Sample actions instead of taking the distribution argmax')
    p.add_argument('--no-byzantine', action='store_true',
                   help='Skip the wrapper entirely -- a pure unwrapped baseline rollout')
    p.add_argument('--device', default=None, choices=['cpu', 'gpu'], help='Override cfg.device')

    w = p.add_argument_group('wandb')
    w.add_argument('--wandb', action='store_true', help='Log this run to Weights & Biases')
    w.add_argument('--wandb-project', default=None, help=f'default: {DEFAULT_PROJECT}')
    w.add_argument('--wandb-entity', default=None, help=f'default: {DEFAULT_ENTITY}')
    w.add_argument('--wandb-group', default=None,
                   help='Experimental condition; defaults to the tier name. Set per condition, '
                        'NOT per seed -- group is what aggregates seeds into one band.')
    w.add_argument('--wandb-job-type', default='rollout')
    w.add_argument('--wandb-name', default=None, help='Display name; defaults to the out file stem')
    w.add_argument('--wandb-tags', nargs='*', default=[])
    w.add_argument('--no-npz', action='store_true',
                   help='Compute metrics and log, but do not write the ~250 MB trajectory file')
    known, extra_argv = p.parse_known_args()

    cfg = build_cfg(known.checkpoint, extra_argv)
    if known.device:
        cfg.device = known.device

    indices = [int(x) for x in known.compromised_indices.split(',') if x.strip() != '']
    tier_config = json.loads(known.tier_config)

    # {} rather than None: None would fall back to the --byzantine_* flags on cfg.
    byzantine_kwargs = {}
    if not known.no_byzantine:
        byzantine_kwargs = dict(
            tier=known.tier,
            compromised_indices=indices,
            tier_config=tier_config,
            seed=known.seed,
            assignment_mode=known.assignment_mode,
        )

    episodes, byz_mask, byz_config, summary = run_rollouts(
        cfg, byzantine_kwargs, known.n_episodes, known.seed,
        deterministic=not known.stochastic,
    )

    data = pad_and_stack(episodes)
    data['byzantine_mask'] = byz_mask.astype(bool)

    low = byz_config['action_low'][0] if byz_config else -1.0
    high = byz_config['action_high'][0] if byz_config else 1.0
    metrics = compute_metrics(data, action_low=low, action_high=high)

    out_path = known.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if not known.no_npz:
        np.savez_compressed(out_path, **data)

    sidecar = dict(
        checkpoint=known.checkpoint,
        checkpoint_file=summary['checkpoint_file'],
        experiment=cfg.experiment,
        train_dir=cfg.train_dir,
        tier=known.tier if not known.no_byzantine else None,
        tier_config=tier_config,
        compromised_indices=indices,
        f_fraction=(len(indices) / cfg.quads_num_agents) if not known.no_byzantine else 0.0,
        assignment_mode=known.assignment_mode,
        byzantine_wrapper_config=byz_config,
        n_episodes=known.n_episodes,
        seed=known.seed,
        deterministic=not known.stochastic,
        num_agents=int(cfg.quads_num_agents),
        quads_mode=cfg.quads_mode,
        episode_duration=cfg.quads_episode_duration,
        domain_randomization=dict(
            quads_domain_random=bool(cfg.quads_domain_random),
            quads_obst_density_random=bool(cfg.quads_obst_density_random),
            quads_obst_size_random=bool(cfg.quads_obst_size_random),
            replay_buffer_sample_prob=float(cfg.replay_buffer_sample_prob),
            # Hardcoded in swarm_rl/env_wrappers/quad_utils.py, not CLI-configurable:
            thrust_noise_ratio=0.05,
            sense_noise='default',
            dynamics_randomize_every=None,
        ),
        arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in data.items()},
        git_sha=git_sha(),
        command=' '.join(sys.argv),
        generated_at=datetime.now().isoformat(),
        summary={k: v for k, v in summary.items() if k != 'episode_rewards'},
        episode_rewards=summary['episode_rewards'],
        # Characterisation metrics live in the sidecar so the aggregator never has to open
        # the 250 MB .npz files.
        metrics={k: v for k, v in metrics.items() if k != 'per_episode'},
    )
    json_path = os.path.splitext(out_path)[0] + '.json'
    with open(json_path, 'w') as f:
        json.dump(sidecar, f, indent=2)

    if not known.no_npz:
        print(f'\nWrote {out_path}')
        for k, v in data.items():
            print(f'  {k:20s} {str(v.shape):28s} {v.dtype}')
    print(f'Wrote {json_path}')
    print(f"\nmean episode reward   : {summary['episode_reward_mean']:.6f} "
          f"± {summary['episode_reward_std']:.6f}  (n={known.n_episodes})")
    for k in ('goal_rmse_mean', 'goal_rmse_honest_mean', 'collisions_mean',
              'action_deviation_byzantine_mean', 'saturation_rate_byzantine',
              'min_separation_mean'):
        print(f'{k:22s}: {metrics[k]:.6f}')

    if known.wandb:
        name = known.wandb_name or os.path.splitext(os.path.basename(out_path))[0]
        group = known.wandb_group or (known.tier if not known.no_byzantine else 'clean')
        tags = list(known.wandb_tags) or default_tags(known, cfg, tier_config)
        try:
            url = log_rollout(
                sidecar, metrics, run_name=name, group=group,
                job_type=known.wandb_job_type, tags=tags,
                project=known.wandb_project, entity=known.wandb_entity,
                notes=f'Byzantine rollout: {group} on {cfg.experiment}',
            )
            if url:
                print(f'wandb: {url}')
        except Exception as exc:
            # Never fail a completed rollout because of a logging problem -- the .npz and
            # .json are already on disk at this point. Transient wandb auth failures do
            # happen when many workers read ~/.netrc off AFS at once. Backfill later with
            # `python -m byzantine_experiments.analysis.backfill_wandb`.
            print(f'[wandb] logging FAILED ({type(exc).__name__}: {exc}). '
                  f'Data is saved; re-log with backfill_wandb.')


def default_tags(known, cfg, tier_config):
    """Tags that make the runs table filterable without opening any run."""
    tags = [
        f'seed{known.seed}',
        f'{cfg.quads_num_agents}agents',
        cfg.quads_mode,
        cfg.experiment,
        f'n_ep{known.n_episodes}',
    ]
    if known.no_byzantine:
        tags += ['clean', 'f0']
    else:
        n = len(known.compromised_indices.split(',')) if known.compromised_indices.strip() else 0
        tags += [
            known.tier,
            known.tier.split('_')[0],                      # tier0 / tier1 / ...
            'f' + f'{n / cfg.quads_num_agents:.4f}'.replace('.', ''),
            f'nbyz{n}',
        ]
        tags += [f'{k}{v}' for k, v in sorted(tier_config.items())]
    return tags


if __name__ == '__main__':
    main()
