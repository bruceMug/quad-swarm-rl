xvfb-run -a python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi \
  --replay_buffer_sample_prob=0 --quads_use_numba=False \
  --quads_render=True --quads_view_mode topdown \
  --save_video --max_num_episodes=100 \
  --train_dir=train_dir --experiment=exp1_baseline_seed2