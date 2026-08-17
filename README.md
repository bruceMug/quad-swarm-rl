# Drone Simulator: QuadSwarm

A codebase for training reinforcement learning policies for quadrotor swarms.
Includes:
* Flight dynamics simulator forked from https://github.com/amolchanov86/gym_art
and extended to support swarms of quadrotor drones
* Scripts and the necessary wrappers to facilitate training of control policies with Sample Factory
https://github.com/alex-petrenko/sample-factory

[//]: # (**Paper:** https://arxiv.org/abs/2109.07735)

[//]: # ()
[//]: # (**Website:** https://sites.google.com/view/swarm-rl)


<p align="middle">

<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Static_Same_Goal.gif?raw=true" width="45%">
&emsp;
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Swarm_vs_Swarm.gif?raw=true" width="45%">
</p> 

<p align="middle">
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Obstacles_Static_Same_Goal.gif?raw=true" width="45%">
&emsp;
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Scale_32_Static_Same_Goal.gif?raw=true" width="45%">
</p> 

## Installation

Initialize a Python environment, i.e. with `conda` (Python versions >=3.11 are supported):

```
conda create -n swarm-rl python=3.11
conda activate swarm-rl
```

Clone and install this repo as an editable Pip package:
```
git clone https://github.com/Zhehui-Huang/quad-swarm-rl.git
cd quad-swarm-rl
pip install -e .
```

This should pull and install all the necessary dependencies including PyTorch.

## Running experiments

### Train


#### Mine
```bash
tmux new -s training

cd /scratch/bmugizi/marl_research/quad-swarm-rl

bash train_exp1.sh 2>&1 | tee seed1_train.log && \
bash train_exp2.sh 2>&1 | tee seed2_train.log && \
bash train_exp3.sh 2>&1 | tee seed3_train.log

# Then Ctrl+B then D
```


#### repo
This will run the baseline experiment.
Change the number of workers appropriately to match the number of logical CPU cores on your machine, but it is advised that
the total number of simulated environments is close to that in the original command:

We provide a training script `train.sh`, so you can simply start training by command `bash train.sh`.

Or, even better, you can use the runner scripts in `swarm_rl/runs/`. These runner scripts (a Sample Factory feature) are Python files that
contain experiment parameters, and support features such as evaluation on multiple seeds and gridsearches.

To execute a runner script run the following command:

```
python -m sample_factory.launcher.run --run=swarm_rl.runs.single_quad.single_quad --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
```

This command will start training four different seeds in parallel on a 4-GPU server. Adjust the parameters accordingly to match
your hardware setup.

To monitor the experiments, go to the experiment folder, and run the following command:

```
tensorboard --logdir=./
```
### WandB support

If you want to monitor training with WandB, follow the steps below: 
- setup WandB locally by running `wandb login` in the terminal (https://docs.wandb.ai/quickstart#1.-set-up-wandb).
* add `--with_wandb=True` in the command.

#### Mine

Use the `swarm_r311_recovered` conda env — `swarm_r311` has a corrupt protobuf install and
`import wandb` fails there with `No module named 'google.protobuf.wrappers_pb2'`.

One-time setup:
```bash
conda activate swarm_r311_recovered
wandb login   # key from https://wandb.ai/authorize
```

Run exp1 with wandb (`exp1_wandb.sh` takes the seed as its first arg, default 1):
```bash
tmux new -s training
conda activate swarm_r311_recovered
cd /scratch/bmugizi/marl_research/quad-swarm-r1

bash exp1_wandb.sh 1 2>&1 | tee logs/exp1_wandb_seed1_train.log && \
bash exp1_wandb.sh 2 2>&1 | tee logs/exp1_wandb_seed2_train.log && \
bash exp1_wandb.sh 3 2>&1 | tee logs/exp1_wandb_seed3_train.log
# Ctrl+B then D
```

Flag mapping. The entity goes in `--wandb_user`; there is **no** `--wandb_entity` flag and
sample_factory hard-errors on unknown args (`error: unrecognized arguments`):
```
--wandb_project=byzantine_exp
--wandb_user=bmugizi-carnegie-mellon-university
--wandb_group=baseline             # otherwise defaults to the env name, quadrotor_multi
```

#### Project / run organization

Everything goes in the single `byzantine_exp` project — wandb can only compare runs within
one project, and baseline-vs-byzantine is the comparison that matters. Structure inside it:

| level | meaning | examples |
| --- | --- | --- |
| `--wandb_group` | experimental condition | `baseline`, `byz_1_attacker`, `byz_2_attackers` |
| `--wandb_job_type` | train vs eval | `train`, `eval` |
| `--wandb_tags` | cross-cutting filters | `seed1`, `8agents`, `mix` |

Group is what makes wandb aggregate the seeds of one condition into a single mean±band
curve, so always set it per condition (not per seed).

#### Backfilling the finished baseline runs (no retraining)

`wandb sync` can set `--project`/`--entity`/`--id` but **not** group, tags, or config, and
the tfevents path sets no display name either — so sync with an explicit `--id`, then
backfill group/tags/name plus the full hyperparameter config (read from each run's
`train_dir/*/config.json`) with the helper script:

```bash
conda activate swarm_r311_recovered
cd /scratch/bmugizi/marl_research/quad-swarm-r1

for s in 1 2 3; do
  wandb sync --sync-tensorboard \
    --entity bmugizi-carnegie-mellon-university --project byzantine_exp \
    --id exp1_baseline_seed${s} \
    train_dir/exp1_baseline_seed${s}
done

python scripts/wandb_tag_synced_runs.py --dry-run   # check first
python scripts/wandb_tag_synced_runs.py
```

Training runs launched with `--with_wandb=True` already carry their group, tags, and config,
and do not need the fixup script.

Here is a total list of wandb settings: 
```
--with_wandb: Enables Weights and Biases integration (default: False)
--wandb_user: WandB username (entity). Must be specified from command line! Also see https://docs.wandb.ai/quickstart#1.-set-up-wandb (default: None)
--wandb_project: WandB "Project" (default: sample_factory)
--wandb_group: WandB "Group" (to group your experiments). By default this is the name of the env. (default: None)
--wandb_job_type: WandB job type (default: SF)
--wandb_tags: [WANDB_TAGS [WANDB_TAGS ...]] Tags can help with finding experiments in WandB web console (default: [])
```

### Test
To test the trained model, run the following command:

```bash
# to save a video file
xvfb-run -a python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi \
  --replay_buffer_sample_prob=0 --quads_use_numba=False \
  --quads_render=True --quads_view_mode topdown \
  --save_video --max_num_episodes=100 \
  --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME

# no video file
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi \
  --replay_buffer_sample_prob=0 --quads_use_numba=False \
  --no_render \
  --max_num_episodes=10 \
  --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME
```

general command
```bash
bash test_seed1.sh 2>&1 | tee logs/seed1_test.log && \
bash test_seed2.sh 2>&1 | tee logs/seed2_test.log && \
bash test_seed3.sh 2>&1 | tee logs/seed3_test.log
```

or

```
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME --quads_view_mode CAMERA_VIEWS
```
EXPERIMENT_NAME and PATH_TO_TRAIN_DIR can be found in the cfg.json file of your trained model

CAMERA_VIEWS can be any number of views from the following: `[topdown, global, chase, side, corner0, corner1, corner2, corner3, topdownfollow]`


## Unit Tests

To run unit tests:

```bash
./run_tests.sh
```

## Video
To get a video from the trained model after running the test command, run the following command:

```bash
ffmpeg -y -i /tmp/sf2_bmugizi/replay.mp4 -vcodec libx264 train_dir/{EXPERIMENT_NAME}/replay.mp4
```

### Tensorboard (ssh tunnel)

1. On the server (leave this running):
```bash
tensorboard --logdir=train_dir/{EXPERIMENT_NAME} --port=6006
```

2. On your local machine (new terminal, separate SSH):
```bash
ssh -L 6006:localhost:6006 bmugizi@ece019.ece.local.cmu.edu
```

3. On your local browser: open
```bash
http://localhost:6006
```

You'll see training curves for reward, FPS, and all the episode_extra_stats metrics like metric/agent_success_rate, metric/agent_col_rate, etc. — one curve per seed if you point logdir at the parent folder:
```bash
tensorboard --logdir=train_dir --port=6006
```
That will show all three seeds overlaid for comparison.