#!/bin/bash
# exp1 (baseline) with WandB logging.
# Requires: conda activate swarm_r311_recovered  (swarm_r311 has a broken protobuf -> wandb import fails)
# Requires: wandb login  (one time)

SEED=${1:-1}

python -m swarm_rl.train \
    --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False \
    --num_workers=32 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 \
    --nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform \
    --adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 --with_pbt=False \
    --gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 \
    --quads_use_numba=True --quads_num_agents=8 --quads_mode=mix --quads_episode_duration=15.0 \
    --quads_neighbor_encoder_type=attention --quads_neighbor_hidden_size=256 --quads_neighbor_obs_type=pos_vel \
    --quads_collision_reward=5.0 --quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 \
    --quads_collision_smooth_max_penalty=10.0 --quads_neighbor_visible_num=6 \
    --replay_buffer_sample_prob=0.75 --anneal_collision_steps=300000000 --normalize_input=False \
    --normalize_returns=False --reward_clip=10.0 --save_milestones_sec=3600 \
    --seed=${SEED} --experiment=exp1_wandb_seed${SEED} \
    --with_wandb=True \
    --wandb_user=bmugizi-carnegie-mellon-university \
    --wandb_project=byzantine_exp \
    --wandb_group=baseline \
    --wandb_job_type=train \
    --wandb_tags baseline exp1 seed${SEED}
