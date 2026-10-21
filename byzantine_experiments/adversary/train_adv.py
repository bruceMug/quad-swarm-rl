"""
Tier 3 — adversarial policy training entry point.

    python -m byzantine_experiments.adversary.train_adv \
      --env=quadrotor_multi_adv \
      --adv_baseline_checkpoint=train_dir/exp1_baseline_seed1 \
      --adv_w_stealth=0 \
      --experiment=adv_seed1_stealth0 \
      --train_for_env_steps=200000000

The honest drones are inside the environment (see frozen_team.py), so Sample Factory trains
exactly one policy and never sees the frozen weights. `--num_policies` stays at 1.
"""

import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.train import run_rl

from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults

from byzantine_experiments.adversary.env_factory import (
    ADV_ENV_NAME,
    add_adversary_args,
    register_adversarial_components,
)


def parse_adv_cfg(argv=None, evaluation=False):
    # Sample Factory marks --env as required at parser construction, so `set_defaults` cannot
    # supply it. Inject it when the caller has not, since for this entry point there is only
    # ever one valid value.
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(a == '--env' or a.startswith('--env=') for a in argv):
        argv = [f'--env={ADV_ENV_NAME}'] + argv

    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_quadrotors_env_args(partial_cfg.env, parser)
    add_adversary_args(parser)
    quadrotors_override_defaults(partial_cfg.env, parser)
    override_adversary_defaults(parser)
    return parse_full_cfg(parser, argv)


def override_adversary_defaults(parser):
    """
    Baseline hyperparameters (scripts/train_exp1.sh) with the deviations the spec asks for.

    Reused verbatim: APPO, no RNN, tanh, separate actor/critic weights, xavier init,
    non-adaptive stddev, no vtrace, gae_lambda 1.0, ppo_clip_value 5.0, rollout 128.

    Changed, and why:
      * train_for_env_steps 2e8 not 1e9 -- adversarial policies converge faster (spec).
      * learning_rate 5e-5 not 1e-4 -- halved up front. The adversarial objective is
        non-stationary in a way the cooperative one is not (the honest team reacts to the
        attack), and the spec's failure-mode table lists "training unstable / diverges ->
        halve LR" as the expected first fix. Cheaper to start there than to discover it after
        12 GPU-hours.
      * replay_buffer_sample_prob 0.0 -- the buffer replays near-collision snapshots to help
        the *cooperative* policy learn recovery. Feeding an adversary a curriculum of
        already-broken states biases what it learns to exploit, and makes its training
        distribution differ from the clean rollouts it is evaluated against.
      * num_policies 1 -- there is only one trainable policy by construction.

    Round 3 note: `exploration_loss_coeff` stays 0.0 here, matching the baseline, and is
    raised from the launch script instead. Rounds 1-2 both saturated (`train/act_max` 30-64
    within 50M steps) and a saturated tanh policy with `adaptive_stddev=False` has almost no
    gradient left; that could not explain the ceiling (design/tier3_design.md 5) but it is
    worth not having. Changing the default here would silently alter what a re-run of
    rounds 1-2 does, so the knob lives in scripts/byz_train_adv.sh.
    """
    parser.set_defaults(
        env=ADV_ENV_NAME,
        train_for_env_steps=200_000_000,
        algo='APPO',
        use_rnn=False,
        num_workers=32,
        num_envs_per_worker=4,
        learning_rate=5e-5,
        ppo_clip_value=5.0,
        recurrence=1,
        nonlinearity='tanh',
        actor_critic_share_weights=False,
        policy_initialization='xavier_uniform',
        adaptive_stddev=False,
        with_vtrace=False,
        max_policy_lag=100000000,
        rnn_size=256,
        with_pbt=False,
        num_policies=1,
        gae_lambda=1.0,
        max_grad_norm=5.0,
        exploration_loss_coeff=0.0,
        rollout=128,
        batch_size=1024,
        normalize_input=False,
        normalize_returns=False,
        reward_clip=10.0,
        save_milestones_sec=3600,
        replay_buffer_sample_prob=0.0,
        # Env settings must match the baseline the adversary is attacking.
        quads_use_numba=True,
        quads_num_agents=8,
        quads_mode='mix',
        quads_episode_duration=15.0,
        quads_neighbor_encoder_type='attention',
        quads_neighbor_hidden_size=256,
        quads_neighbor_obs_type='pos_vel',
        quads_neighbor_visible_num=6,
        quads_collision_reward=5.0,
        quads_collision_hitbox_radius=2.0,
        quads_collision_falloff_radius=4.0,
        quads_collision_smooth_max_penalty=10.0,
    )


def main():
    register_adversarial_components()
    cfg = parse_adv_cfg()
    return run_rl(cfg)


if __name__ == '__main__':
    sys.exit(main())
