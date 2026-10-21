"""
Env registration for adversarial training.

Registers a Sample Factory env named `quadrotor_multi_adv`: the ordinary quadrotor swarm with
the honest drones driven by a frozen checkpoint, exposed to the learner as an `n_adv`-agent
environment (see frozen_team.py for why this rather than `--num_policies=2`).

The frozen policy is built with a *separate* cfg reconstructed from the baseline checkpoint's
own `config.json`, not from the adversary's training cfg. That matters: the two runs differ in
`quads_num_agents` as far as Sample Factory is concerned, and the honest network must be
instantiated with the architecture it was trained with.
"""

import json
import os

import numpy as np

from byzantine_experiments.adversary.frozen_team import FrozenTeamEnv, FrozenTeamPolicy
from byzantine_experiments.adversary.lies import (
    DEFAULT_MAX_LIE_POS,
    DEFAULT_MAX_LIE_VEL,
    DEFAULT_PLAUSIBLE_MAX_ACCEL,
    DEFAULT_PLAUSIBLE_MAX_OFFSET,
    DEFAULT_PLAUSIBLE_MAX_RATE,
)
from byzantine_experiments.adversary.reward import AdversarialReward

ADV_ENV_NAME = 'quadrotor_multi_adv'


def baseline_cfg_for(checkpoint_dir):
    """
    Reconstruct the baseline's own cfg so the frozen network matches its trained architecture.

    Uses the run's `config.json` rather than the adversary's cfg, because the adversary run
    legitimately differs in things like `train_dir`, `experiment`, learning rate and horizon,
    and a couple of those (encoder settings, rnn size) would silently change the network shape.
    """
    from sample_factory.utils.attr_dict import AttrDict

    cfg_path = os.path.join(os.path.dirname(checkpoint_dir.rstrip('/')), 'config.json')
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(checkpoint_dir, '..', 'config.json')
    with open(os.path.abspath(cfg_path)) as f:
        return AttrDict(json.load(f))


def make_adversarial_env(full_env_name, cfg=None, _env_config=None, render_mode=None, **kwargs):
    """
    Factory registered with Sample Factory under `quadrotor_multi_adv`.

    Signature must match what `sample_factory.envs.env_utils.create_env` calls:
    `env_func(full_env_name, cfg, env_config, render_mode)` -- four positional arguments,
    the first being the env name. Same shape as `swarm_rl`'s own `make_quadrotor_env`.
    """
    from sample_factory.model.model_utils import get_rnn_size
    from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env

    baseline_dir = cfg.adv_baseline_checkpoint.rstrip('/')
    # `policy_index` is an evaluation-only cfg key and is absent during training. The
    # baseline is a single-policy run, so its checkpoints always live under checkpoint_p0.
    policy_index = getattr(cfg, 'policy_index', 0) or 0
    checkpoint_dir = os.path.join(baseline_dir, f'checkpoint_p{policy_index}')
    base_cfg = baseline_cfg_for(checkpoint_dir)
    base_cfg.load_checkpoint_kind = getattr(cfg, 'load_checkpoint_kind', 'latest')

    # The underlying 8-drone env, built exactly as every other tier builds it, so collected
    # trajectories stay format-compatible with Tiers 0-2.
    inner_cfg = AttrDictShim(base_cfg, overrides=dict(
        quads_render=False, visualize_v_value=False, with_pbt=False,
    ))
    env = make_quadrotor_env(base_cfg.env, cfg=inner_cfg, _env_config=_env_config,
                             render_mode=render_mode)

    frozen = FrozenTeamPolicy(base_cfg, env.observation_space, env.action_space,
                              checkpoint_dir, device='cpu')

    adv_indices = [int(x) for x in str(cfg.adv_indices).split(',') if x.strip() != '']
    mu = json.loads(cfg.adv_mu_honest) if getattr(cfg, 'adv_mu_honest', '') else None

    reward_fn = AdversarialReward(
        w_team=cfg.adv_w_team, w_crash=cfg.adv_w_crash, w_alive=cfg.adv_w_alive,
        w_stealth=cfg.adv_w_stealth, epsilon=cfg.adv_epsilon,
        mu_honest=mu, team_reward_mode=cfg.adv_team_reward_mode,
        crash_penalty_mode=cfg.adv_crash_penalty_mode,
    )

    return FrozenTeamEnv(env, frozen, adv_indices=adv_indices, reward_fn=reward_fn,
                         deterministic_honest=bool(cfg.adv_deterministic_honest),
                         rnn_size=get_rnn_size(base_cfg),
                         enable_lies=bool(getattr(cfg, 'adv_enable_lies', False)),
                         lie_only=bool(getattr(cfg, 'adv_lie_only', False)),
                         ego_state_obs=bool(getattr(cfg, 'adv_ego_state_obs', False)),
                         lie_mode=str(getattr(cfg, 'adv_lie_mode', 'free')),
                         plausible_max_offset=float(getattr(cfg, 'adv_plausible_max_offset',
                                                            DEFAULT_PLAUSIBLE_MAX_OFFSET)),
                         plausible_max_rate=float(getattr(cfg, 'adv_plausible_max_rate',
                                                          DEFAULT_PLAUSIBLE_MAX_RATE)),
                         plausible_max_accel=float(getattr(cfg, 'adv_plausible_max_accel',
                                                           DEFAULT_PLAUSIBLE_MAX_ACCEL)),
                         max_lie_pos=float(getattr(cfg, 'adv_max_lie_pos',
                                                   DEFAULT_MAX_LIE_POS)),
                         max_lie_vel=float(getattr(cfg, 'adv_max_lie_vel',
                                                   DEFAULT_MAX_LIE_VEL)))


class AttrDictShim:
    """Read-through view of the baseline cfg with a few keys overridden."""

    def __init__(self, base, overrides):
        self._base = base
        self._over = dict(overrides)

    def __getattr__(self, name):
        if name in ('_base', '_over'):
            raise AttributeError(name)
        if name in self._over:
            return self._over[name]
        return getattr(self._base, name)

    def __getitem__(self, name):
        return getattr(self, name)

    def get(self, name, default=None):
        try:
            return getattr(self, name)
        except AttributeError:
            return default


def register_adversarial_components():
    """Register the adversarial env and the shared quadrotor model with Sample Factory."""
    from sample_factory.envs.env_utils import register_env

    from byzantine_experiments.adversary.frozen_team import ensure_swarm_components_registered

    ensure_swarm_components_registered()   # model/encoder + the `quadrotor_multi` env
    register_env(ADV_ENV_NAME, make_adversarial_env)


def add_adversary_args(parser):
    """`--adv_*` flags. Declared here, not in the upstream quadrotor_params.py."""
    p = parser
    p.add_argument('--adv_baseline_checkpoint', default='train_dir/exp1_baseline_seed1',
                   type=str, help='Frozen honest baseline run directory')
    p.add_argument('--adv_indices', default='0', type=str,
                   help='Comma-separated drone indices the adversary controls')
    p.add_argument('--adv_w_team', default=20.0, type=float,
                   help='Calibrated, not the spec-s 1.0 -- see adversary/reward.py')
    p.add_argument('--adv_w_crash', default=10.0, type=float)
    p.add_argument('--adv_w_alive', default=0.01, type=float)
    p.add_argument('--adv_w_stealth', default=0.0, type=float,
                   help='0 = unconstrained adversary; 5 = the stealth-constrained condition')
    p.add_argument('--adv_epsilon', default=0.3, type=float,
                   help='Attack budget: free deviation from the honest action mean')
    p.add_argument('--adv_mu_honest', default='', type=str,
                   help='JSON list, mean honest action from clean rollouts. Empty = zeros.')
    p.add_argument('--adv_team_reward_mode', default='honest', type=str,
                   choices=['honest', 'all'],
                   help="Average team reward over honest drones only (default) or all 8. "
                        "'all' lets the adversary score by wrecking itself -- see reward.py.")
    p.add_argument('--adv_crash_penalty_mode', default='onset', type=str,
                   choices=['onset', 'per_step'],
                   help="Charge w_crash once per crash event (default) or every step while "
                        "crashed (the spec's literal wording, which drowns the objective "
                        "2622:1 -- see adversary/reward.py)")
    # ---- round 3: the observation channel (design/tier3_design.md 5) --------------------
    p.add_argument('--adv_enable_lies', default=False, type=str2bool_local,
                   help='extend the action from 4 motors to 4+6, the extra six being the '
                        'state this drone broadcasts. Off by default so rounds 1-2 configs '
                        'reproduce exactly.')
    p.add_argument('--adv_lie_only', default=False, type=str2bool_local,
                   help='drive the adversary motors with the frozen honest policy and learn '
                        'only the lie -- the direct analogue of Tier 2C, isolating the '
                        'observation channel from motor authority. Requires --adv_enable_lies.')
    p.add_argument('--adv_ego_state_obs', default=False, type=str2bool_local,
                   help="append the adversary's own absolute pos/vel to its observation "
                        '(54 -> 60). Restores information parity with the hand-designed '
                        '`liar`, which is handed its true state. Requires --adv_enable_lies. '
                        'Default False so rounds 3-4 checkpoints load unchanged.')
    p.add_argument('--adv_lie_mode', default='free', type=str, choices=['free', 'plausible'],
                   help="'free': any claim within +/-max_lie_pos (rounds 3-5). 'plausible': a "
                        'bounded smooth drift that passes naive bounds / speed / consistency / '
                        'acceleration checks (D3, lies.py::PlausibleLie).')
    p.add_argument('--adv_plausible_max_offset', default=DEFAULT_PLAUSIBLE_MAX_OFFSET, type=float)
    p.add_argument('--adv_plausible_max_rate', default=DEFAULT_PLAUSIBLE_MAX_RATE, type=float)
    p.add_argument('--adv_plausible_max_accel', default=DEFAULT_PLAUSIBLE_MAX_ACCEL, type=float)
    p.add_argument('--adv_max_lie_pos', default=DEFAULT_MAX_LIE_POS, type=float,
                   help='metres of positional lie at full deflection (default spans the env '
                        'neighbour-observation clip, beyond which a lie is unexpressible)')
    p.add_argument('--adv_max_lie_vel', default=DEFAULT_MAX_LIE_VEL, type=float,
                   help='m/s of velocity lie at full deflection')
    p.add_argument('--adv_deterministic_honest', default=True, type=str2bool_local,
                   help='Honest drones take the policy argmax, matching how Tiers 0-2 rolled out')


def str2bool_local(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ('yes', 'true', 't', 'y', '1')
