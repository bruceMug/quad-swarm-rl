# Project Structure — quad-swarm-r1

Layout of this working copy as of 2026-08-11, branch `feature/byzantine`.

The repo has two halves: an upstream fork (`gym_art/` + `swarm_rl/`, the simulator and the
RL training code) and your own experiment layer (`scripts/`, `logs/`, `train_dir/`), which
this branch reorganized into dedicated folders.

```
quad-swarm-r1/
├── gym_art/               simulator — flight dynamics, collisions, scenarios
├── swarm_rl/              RL layer — training, models, env wrappers, sim2real
├── scripts/               YOUR experiment drivers (train/test/eval/report)
├── train_dir/             training output — checkpoints, configs, tfevents  [gitignored]
├── logs/                  captured stdout + evaluation result JSONs
├── paper/                 upstream plotting scripts for paper figures
├── py_environment/        conda/pip environment specs
├── byzantine_experiments/ empty — placeholder for this branch's work
├── train*.sh, run_tests.sh legacy upstream launcher scripts (repo root)
├── setup.py               editable-install package definition
└── README.md              install, train, test, wandb, tensorboard instructions
```

---

## `gym_art/` — flight dynamics simulator

Forked from [amolchanov86/gym_art](https://github.com/amolchanov86/gym_art) and extended for
swarms. This is the physics and environment layer; it has no RL code in it.

Everything lives under `gym_art/quadrotor_multi/`:

| Path | Contents |
| --- | --- |
| `quadrotor_multi.py` | The multi-agent env class — the top-level entry point the RL code instantiates |
| `quadrotor_single.py` | Single-drone env, the building block for the multi version |
| `quadrotor_dynamics.py` | Rigid-body flight dynamics integration |
| `quadrotor_control.py` | Low-level controllers (motor thrust / attitude) |
| `quad_models.py`, `inertia.py` | Physical drone parameters and inertia tensors |
| `quadrotor_randomization.py`, `sensor_noise.py` | Domain randomization and sensor noise models |
| `quad_experience_replay.py` | The replay-buffer mechanism behind `--replay_buffer_sample_prob` |
| `numba_utils.py` | Numba-JIT'd hot paths (enabled by `--quads_use_numba=True`) |
| `get_state.py`, `quad_utils.py` | Observation construction and shared helpers |
| `scenarios/` | One file per task: `static_same_goal`, `dynamic_formations`, `swarm_vs_swarm`, `run_away`, `swap_goals`, `mix` (the `--quads_mode=mix` curriculum), plus `obstacles/` variants |
| `collisions/` | Drone-drone, drone-obstacle, and drone-room collision detection, with its own `test/` (unit + speed tests) |
| `obstacles/` | Obstacle field generation and utilities |
| `aerodynamics/` | `downwash.py` — rotor downwash interaction between drones |
| `rendering3d.py`, `quadrotor_visualization.py`, `quadrotor_multi_visualization.py` | OpenGL rendering, used by `--quads_render=True` and the camera views |
| `tests/` | Env unit tests (`test_multi_env.py`, `test_numba_opt.py`) and V-value plotting utilities |
| `plots/` | Plot output directory |


## `swarm_rl/` — RL training layer

Wraps the simulator for [Sample Factory](https://github.com/alex-petrenko/sample-factory)
(v2.1.1, installed as a dependency, not vendored here).

| Path | Contents |
| --- | --- |
| `train.py` | Training entry point — `python -m swarm_rl.train`. Registers the env and model with Sample Factory |
| `enjoy.py` | Evaluation/rollout entry point — `python -m swarm_rl.enjoy`, used for videos and testing |
| `utils.py` | Shared helpers |
| `models/` | The policy networks: `quad_multi_model.py` (the encoder), `attention_layer.py` (backs `--quads_neighbor_encoder_type=attention`), `weight_recycler.py` |
| `env_wrappers/` | The simulator↔Sample-Factory bridge: `quadrotor_params.py` (all the `--quads_*` CLI args are defined here), `reward_shaping.py` (reward term weights), `compatibility.py` (gym/gymnasium shim), `v_value_map.py`, plus `tests/test_quads.py` |
| `runs/` | Sample Factory *launcher* scripts — Python files declaring param grids and multi-seed sweeps, run via `sample_factory.launcher.run`. Contains `quad_multi_mix_baseline_attn_8.py` (closest to your exp1 config), `single_quad/`, `obstacles/`, and `slurm/sbatch_timeout.sh` |
| `sim2real/` | Export of trained PyTorch policies to C for real Crazyflie hardware (`sim2real.py`, `code_blocks.py`, `c_models/`, `torch_models/`) |
| `gifs/` | The four demo GIFs embedded in `README.md` |
| `train_dir/` | Stale upstream output (`debug_main`, `t4_smoke_test`) — not your experiments |

> Note the `--quads_*` flags in your training scripts are defined in
> `env_wrappers/quadrotor_params.py`, *not* in Sample Factory's `cfg.py`. That's the file to
> read when adding new byzantine-related CLI arguments.

## `scripts/` — your experiment layer

Everything you added. Per `scripts/Readme.md`, these were moved here from the repo root on
this branch for tidiness — **which means the relative paths inside them assume the repo
root as CWD.** Either run them as `bash scripts/train_exp1.sh` from the root, or copy them
back up, but don't `cd scripts/` first.

| File | Purpose |
| --- | --- |
| `train_exp1.sh`, `train_exp2.sh`, `train_exp3.sh` | The three baseline training runs (seeds 1/2/3), 1e9 env steps, 8 agents, attention encoder, `mix` scenario. Identical apart from `--seed` and `--experiment` |
| `exp1_wandb.sh` | The wandb-enabled variant of exp1; takes the seed as `$1`. Adds `--with_wandb`, `--wandb_user`, `--wandb_project`, `--wandb_group`, `--wandb_tags` |
| `test_seed1.sh`, `test_seed2.sh`, `test_seed3.sh` | Rollout/render the trained policy per seed via `swarm_rl.enjoy` |
| `evaluate.py` | Runs N held-out episodes on a checkpoint and writes summary stats to JSON — the accurate numbers |
| `summarize_experiment.py` | Reads the tail of the tensorboard curve for a run and summarizes it — fast but training-curve-based, not held-out |
| `report_baseline.py` | Aggregates the three seeds into one report; `--from-tb` consumes `summarize_experiment.py` output, `--from-eval` consumes `evaluate.py` output |
| `wandb_tag_synced_runs.py` | Backfills group / tags / name / config onto runs uploaded by `wandb sync --sync-tensorboard`, which cannot set those itself |
| `Readme.md` | Notes on the root→`scripts/` move |

## `train_dir/` — training output (gitignored)

One folder per experiment, named by `--experiment`. **~940 MB total** for your three
baseline seeds, which is why it's in `.gitignore`.

```
train_dir/exp1_baseline_seed1/
├── checkpoint_p0/
│   ├── best_000596516_610832384_reward_2.930.pth
│   ├── checkpoint_000976565_1000002560.pth    (2 most recent kept)
│   └── milestones/                            (hourly, from --save_milestones_sec=3600)
├── config.json     full resolved config — the source for wandb config backfill
├── .summary/0/     tensorboard tfevents — what `wandb sync` and tensorboard read
├── sf_log.txt      Sample Factory's own log
├── git.diff        working-tree diff at launch time (provenance)
└── replay.mp4      rendered rollout
```

Contents: `exp1_baseline_seed1/`, `exp1_baseline_seed2/`, `exp1_baseline_seed3/` (yours,
~312 MB each) and `paper_quads_multi_mix_baseline_8a_attn_v116/` (a 6.1 MB upstream
pretrained reference policy).

Note `.summary/` is hidden — `ls train_dir/exp1_baseline_seed1` won't show it.

## `logs/` — run output and results

Captured stdout plus derived results, all committed to git.

| Pattern | Contents |
| --- | --- |
| `seed{1,2,3}_train.log` | Full training stdout per seed |
| `seed{1,2,3}_test.log` | Rollout/test stdout per seed |
| `seed{1,2,3}_eval_final.json` | `evaluate.py` output — held-out episode statistics |
| `results{1,2,3}.txt` | `summarize_experiment.py` output per seed |
| `baseline_report.json` | `report_baseline.py` aggregate across all three seeds |

## Remaining folders

**`paper/`** — Upstream matplotlib scripts that produced the original paper's figures
(`mean_std_plots_quad_*.py` for baselines/annealing/scale/obstacle ablations,
`attn_heatmap.py`, `fps_compare.py`) plus `attn_study.pdf`. Useful as templates for
mean±std multi-seed plots, since that's exactly the shape of your baseline-vs-byzantine
comparison.

**`py_environment/`** — Environment specs: `environment.yml`, `environment_full.yml`
(fully pinned), `requirements.txt`, `env_info.txt`. Note these describe the `swarm_r311`
env, whose protobuf install is currently broken; the working env is `swarm_r311_recovered`.

**`byzantine_experiments/`** — Currently empty. Created for this branch's work but unused
so far; your byzantine code isn't written yet.

**Repo root `.sh` files** — Legacy upstream launchers, superseded by `scripts/` for your
work: `train.sh` (the documented baseline), `train_local.sh`, `train_local_obst.sh`,
`train_brain.sh`, `train_brain_obstacles_pbt.sh`, `train_aws.sh` (cluster-specific), and
`run_tests.sh` (pytest across both `gym_art` and `swarm_rl` test dirs).

**`setup.py` / `swarm_rl.egg-info/`** — Editable-install metadata (`pip install -e .`).
`.egg-info/` is generated and gitignored.

---

## Where things are ignored

`.gitignore` excludes `train_dir/`, `wandb/`, `*.mp4`, `__pycache__`, `*.pkl`, `*.ipynb`,
and `*.egg-info/`. So checkpoints, wandb's local run cache, and rendered videos stay local
— only code, `logs/`, and configs are versioned.
