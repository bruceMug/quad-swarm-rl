"""
Render Byzantine rollouts to video, for the qualitative description in the design doc.

The node is headless, so this must run under a virtual X server:

    xvfb-run -a python -m byzantine_experiments.rollout.render \
      --checkpoint train_dir/exp1_baseline_seed1 \
      --tier tier1_ou --tier-config '{"sigma_ou": 0.2}' \
      --compromised-indices 0 --n-episodes 5 \
      --out byzantine_experiments/results/videos/t1_ou_sigmaou0.2.mp4

Frames come from the env's own OpenGL scene (`render_mode='rgb_array'`), so this is the same
visualisation the upstream `swarm_rl.enjoy` produces -- the Byzantine wrapper simply sits in
the action path as usual.

Note the compromised drone is not highlighted in the render; the env colours drones by its
own scheme. Use `--compromised-indices 0` and the printed legend to know which one to watch.
"""

import argparse
import json
import os

import numpy as np
import torch

from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.utils import log

from swarm_rl.train import register_swarm_components

import byzantine_experiments.agents  # noqa: F401
from byzantine_experiments.rollout.collect import build_cfg, load_policy
from byzantine_experiments.rollout.env_factory import make_byzantine_env


def write_video(frames, path, fps=30):
    import cv2
    if not frames:
        raise RuntimeError('no frames captured')
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'cv2 could not open a writer for {path}')
    for frame in frames:
        writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f'Wrote {path}  ({len(frames)} frames, {w}x{h}, {fps} fps)')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tier', default='tier0_noop')
    p.add_argument('--tier-config', default='{}')
    p.add_argument('--compromised-indices', default='0')
    p.add_argument('--n-episodes', type=int, default=5)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out', required=True)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--every-nth', type=int, default=3,
                   help='Keep every Nth frame (the sim runs at 100 Hz; 3 gives ~33 Hz)')
    p.add_argument('--max-steps-per-episode', type=int, default=1501)
    p.add_argument('--no-byzantine', action='store_true')
    known, extra = p.parse_known_args()

    register_swarm_components()
    cfg = build_cfg(known.checkpoint, extra)
    cfg.device = 'cpu'
    # Turn rendering back on -- build_cfg forces it off for headless data collection.
    cfg.quads_render = True
    cfg.no_render = False
    cfg.quads_view_mode = ['global']

    indices = [int(x) for x in known.compromised_indices.split(',') if x.strip()]
    byz_kwargs = {} if known.no_byzantine else dict(
        tier=known.tier, compromised_indices=indices,
        tier_config=json.loads(known.tier_config), seed=known.seed,
    )

    np.random.seed(known.seed)
    torch.manual_seed(known.seed)

    env, byz = make_byzantine_env(cfg, byz_kwargs, render_mode='rgb_array')
    env_info = extract_env_info(env, cfg)
    device = torch.device('cpu')
    actor_critic, _ = load_policy(cfg, env, device)

    num_agents = env.num_agents
    rnn_states = torch.zeros([num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    print(f'Rendering {known.n_episodes} episodes | tier='
          f'{byz.tier_name if byz else "clean"} | compromised drones: '
          f'{byz.compromised_indices if byz else []}')

    frames = []
    obs, _ = env.reset()
    completed = step = 0

    with torch.no_grad():
        while completed < known.n_episodes:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)
            actions = argmax_actions(actor_critic.action_distribution())
            if actions.ndim == 1:
                actions = actions.unsqueeze(-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs['new_rnn_states']

            obs, _, terminated, truncated, _ = env.step(actions)
            dones = make_dones(terminated, truncated).cpu().numpy()

            if step % known.every_nth == 0:
                frame = env.unwrapped.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
            step += 1

            if all(dones) or step >= known.max_steps_per_episode:
                completed += 1
                log.info('  episode %d/%d (%d frames so far)',
                         completed, known.n_episodes, len(frames))
                step = 0
                rnn_states = torch.zeros([num_agents, get_rnn_size(cfg)],
                                         dtype=torch.float32, device=device)
                obs, _ = env.reset()

    env.close()
    write_video(frames, known.out, fps=known.fps)


if __name__ == '__main__':
    main()
