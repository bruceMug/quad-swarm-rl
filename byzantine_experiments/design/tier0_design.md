# Tier 0 — Byzantine Infrastructure: design notes

Implementation record for [specs/TIER0_INFRASTRUCTURE.md](../specs/TIER0_INFRASTRUCTURE.md).
Branch `feature/byzantine`, base commit `41b4f27`.

---

## 1. Action format findings

**Everything below was read out of the source, not assumed.**

### Shape, dtype, range

The multi-agent env takes its action space straight from drone 0:

```python
# gym_art/quadrotor_multi/quadrotor_multi.py:77-78
self.observation_space = self.envs[0].observation_space
self.action_space = self.envs[0].action_space
```

Each `QuadrotorSingle` delegates to its controller. `make_quadrotor_env_multi` passes
`raw_control=True, raw_control_zero_middle=True` and the default `dim_mode='3D'`, so the
controller is `RawControl`:

```python
# gym_art/quadrotor_multi/quadrotor_single.py:259-273
if self.raw_control:
    ...
    elif self.dim_mode == '3D':
        self.controller = RawControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
...
self.action_space = self.controller.action_space(self.dynamics)
```

```python
# gym_art/quadrotor_multi/quadrotor_control.py:37-49
def action_space(self, dynamics):
    if not self.zero_action_middle:
        # Range of actions 0 .. 1
        self.low = np.zeros(4); self.bias = 0.0; self.scale = 1.0
    else:
        # Range of actions -1 .. 1
        self.low = -np.ones(4); self.bias = 1.0; self.scale = 0.5
    self.high = np.ones(4)
    return spaces.Box(self.low, self.high, dtype=np.float32)
```

So: **`Box(-1.0, 1.0, shape=(4,), float32)` per drone** — normalised motor thrusts, zero =
hover. The spec's stated range is correct, but the wrapper still reads `low`/`high` off
`env.action_space` at construction (`byzantine_env.py`, `self.action_low/_high`) and hands
them to every agent, so a config change to `raw_control_zero_middle=False` (range `0..1`)
would be picked up automatically. `test_action_range_is_read_from_env_not_hardcoded` locks
this in.

Note `RawControl.step` clips again internally
(`action = np.clip(action, a_min=self.low, a_max=self.high)`, quadrotor_control.py:53), so
an out-of-range Byzantine action would be *silently* absorbed by the sim rather than
erroring — which is exactly why the wrapper raises on out-of-range output instead of
trusting the env to complain. See §3.

### What the env receives per step

```python
# gym_art/quadrotor_multi/quadrotor_multi.py:416-425
def step(self, actions):
    ...
    assert len(actions) == self.num_agents, \
        f"Expected {self.num_agents} actions, got {len(actions)}"
    for i in range(self.num_agents):
        self.envs[i].rew_coeff = self.rew_coeff
        observation, reward, done, info = self.envs[i].step(actions[i])
```

Only `len()` and `actions[i]` are used, so a stacked array and a list of rows are both
legal at the bottom of the stack.

### What the sampler actually passes

`enjoy.py` is a 12-line shim over `sample_factory.enjoy`; the concrete answer is in
sample_factory and in this repo's own `scripts/evaluate.py`, which is what produced the
baseline numbers:

```python
# scripts/evaluate.py:136-142
if actions.ndim == 1:
    actions = unsqueeze_tensor(actions, dim=-1)
actions = preprocess_actions(env_info, actions)
...
obs, rew, terminated, truncated, infos = env.step(actions)
```

`preprocess_actions` routes a `Box` space to `process_action_space`, which does
`actions = actions.cpu().numpy()` and asserts `actions.ndim == 2`
(`sample_factory/algo/sampling/batched_sampling.py:58-73`). So in practice the wrapper
receives a **stacked `(8, 4)` `np.float32` array**, not a list.

It is nonetheless written to accept a list/tuple of per-agent rows or a torch tensor as
well (`_normalise_actions`), and to pass the same container *kind* it received down to the
env — covered by `test_accepts_list_of_per_agent_actions`.

---

## 2. Where the hook sits, and why

`swarm_rl/env_wrappers/quad_utils.py::make_quadrotor_env_multi` builds this stack:

```
QuadrotorEnvMulti                 old 4-tuple gym API, actions[i] per drone
  └─ ExperienceReplayWrapper      only when replay_buffer_sample_prob > 0 (it is: 0.75)
      └─ QuadsRewardShapingWrapper
          └─ QuadEnvCompatibility gymnasium 5-tuple API
              └─ ByzantineEnvWrapper      <-- injected here
                  └─ BatchedDictObservationsWrapper  ┐
                      └─ BatchedListToDictWrapper    ├ added by BatchedVecEnv
                          └─ BatchedVecEnv           ┘
```

Full call path for one step:

```
actor_critic(obs, rnn_states) -> policy_outputs['actions']   (torch, (8,4))
argmax_actions(...)                                          deterministic eval
preprocess_actions(env_info, ...)                            -> np.float32 (8,4)
BatchedVecEnv.step  ->  ...  ->  ByzantineEnvWrapper.step    <-- perturbation applied
                                 QuadEnvCompatibility.step
                                 QuadsRewardShapingWrapper.step
                                 ExperienceReplayWrapper.step
                                 QuadrotorEnvMulti.step -> envs[i].step(actions[i])
```

Why immediately above `QuadEnvCompatibility` and below `BatchedVecEnv`:

- **Above the compatibility shim** means we see the exact `(8, 4)` array the sampler
  produced, in the gymnasium 5-tuple API, so the wrapper is plain `gymnasium.Wrapper` code
  with no legacy-API branching.
- **Below `BatchedVecEnv`** means we act before observations get dict-ified and
  tensor-ified. Above it we would have to unpack a `TensorDict`.
- Everything below us — reward shaping, the replay buffer, the sim — sees the perturbed
  action as if it were the policy's own output. That is the point: a Byzantine drone must
  be indistinguishable at the interface, so all downstream reward and collision accounting
  must respond to the attack naturally.

The wrapper is applied by `byzantine_experiments/rollout/env_factory.py`, which mirrors
sample_factory's `make_env_func_batched` with one extra line. This is what keeps the
upstream diff down to the CLI flags (§5) — `quad_utils.py` is untouched.

`_last_obs` is cached from `reset()`/`step()` so `perturb_action(honest_action, obs, t)`
can hand each tier the observation its own drone acted on.

**Auto-reset.** `ExperienceReplayWrapper.step` resets the env *inside* `step()` when the
episode ends (`obs = self.new_episode()`, quad_experience_replay.py:167-210) and may swap
`self.env` for a `deepcopy` from the replay buffer. The wrapper therefore tracks episode
boundaries from the `terminated`/`truncated` flags rather than relying on `reset()` being
called, and defers the boundary bookkeeping to the *next* `step()` so the finished
episode's perturbation log survives long enough to be read.

---

## 3. Deviations from the spec

| # | Deviation | Reason |
|---|---|---|
| 1 | `ByzantineAgent.__init__` also takes `action_low` / `action_high`, and gains a `self.clip()` helper alongside the module-level `clip_to_action_space`. | The spec requires clipping to be the agent's job *and* requires `low`/`high` to come from `env.action_space`. Those only reconcile if the wrapper tells the agent the range, so tiers call `self.clip(x)` and never hardcode ±1. |
| 2 | The wrapper **raises** if a tier returns an out-of-range action rather than clipping for it. | `RawControl.step` clips internally, so a tier that forgot to clip would produce plausible-looking data with a silently altered attack. Failing loudly is the only way this stays a real invariant. Shape mismatches raise likewise. |
| 3 | `tier` accepts a `ByzantineAgent` **subclass** as well as a registry key. | Lets the test suite exercise sign-flip / unclipped / RNG-driven attackers without polluting the global `BYZANTINE_REGISTRY`, which test 5 iterates over. |
| 4 | The default `reset()` does **not** reseed the agent RNG. | Reseeding per episode would replay the identical noise sequence in every episode, collapsing the variation Phase 3 needs. The stream runs continuously and is still fully determined by `seed`, so run-to-run determinism (test 7) holds. |
| 5 | The perturbation log is **per-episode** (cleared at each episode start) and each entry also carries an `episode` field. | Test 6 asks for exactly `T × len(compromised)` entries, which only has a fixed meaning within one episode. A 100-episode run would otherwise accumulate 1.2 M entries in memory. |
| 6 | The `--byzantine_*` flags are registered in `quadrotor_params.py` but consumed by `byzantine_experiments/rollout/env_factory.py` (`byzantine_kwargs_from_cfg`), not by `quad_utils.py`. | The spec forbids touching any upstream file beyond the flag block. The flags are live — `make_byzantine_env(cfg, None)` reads them — they just aren't wired into the stock `make_quadrotor_env` path. |
| 7 | `collect.py --checkpoint` takes a `<train_dir>/<experiment>` path and splits it, and adds `--assignment-mode`, `--no-byzantine`, `--stochastic`, `--device`. Unknown args pass through to sample_factory's parser. | Matches the CLI signature in the spec while keeping the sample_factory cfg mechanics that `scripts/evaluate.py` already relies on. |
| 8 | `positions` / `velocities` are recorded **before** the step (the state the action was taken in), read from `envs[i].dynamics` rather than the env's cached `pos`/`vel`. | Aligns them with `obs` and `actions` at the same index, which is what formation-error windowing wants. The cached `vel` array is only refreshed inside `step()` (quadrotor_multi.py:603-605), so it is stale on step 0 of an episode. |
| 9 | `episode_lengths` is `int32` and not NaN-padded. | It is the length vector itself; NaN would be meaningless. All other arrays are NaN-padded `float32` as specified. |
| 10 | Test suite has 14 tests, not 7. | The seven required ones, plus range-source, list-input, random-assignment, strict-constructor, clip-helper, config-validation, and the real-checkpoint validation gate (§6). |

---

## 4. `.npz` schema as implemented

Real shapes and dtypes, from the 100-episode validation run
(`byzantine_experiments/results/t0_noop_seed1_f0125.npz`), 8 agents, `quads_mode=mix`,
15 s episodes at 100 Hz:

| Key | Shape | dtype | Contents |
|---|---|---|---|
| `obs` | `(100, 1501, 8, 54)` | `float32` | Per-agent observation the action was computed from. 54 = 18 self (`xyz`,`vxyz`,`R`,`omega`) + 6 neighbours × 6 (`rxyz`,`rvxyz`) |
| `actions_executed` | `(100, 1501, 8, 4)` | `float32` | What the env actually received |
| `actions_honest` | `(100, 1501, 8, 4)` | `float32` | Policy output before perturbation |
| `rewards` | `(100, 1501, 8)` | `float32` | Per-agent per-step reward |
| `positions` | `(100, 1501, 8, 3)` | `float32` | XYZ, world frame, pre-step |
| `velocities` | `(100, 1501, 8, 3)` | `float32` | Pre-step |
| `collisions` | `(100, 1501)` | `float32` | Drone–drone collisions on that step (a pair counts once) |
| `byzantine_mask` | `(8,)` | `bool` | Ground-truth labels for the Phase 4 AUC |
| `episode_lengths` | `(100,)` | `int32` | True length before NaN padding |

Padding: all float arrays are allocated as `(n_episodes, T_max, …)` filled with `NaN`, then
overwritten up to each episode's true length. Every episode of this policy runs the full
1501 steps, so the validation run contains zero NaNs — the padding path is exercised by
construction (`pad_and_stack`) rather than by this particular data.

Sidecar `.json` (same basename) carries: checkpoint path and resolved checkpoint file,
experiment, tier + tier config, compromised indices, `f` fraction, assignment mode, the
wrapper's own config summary, n_episodes, seed, determinism flag, `quads_mode`, episode
duration, domain-randomisation settings (including the `thrust_noise_ratio=0.05` and
`sense_noise='default'` values hardcoded in `quad_utils.py`, and
`replay_buffer_sample_prob`), the array schema, git SHA, the full command line, and
per-episode rewards.

---

## 5. Upstream diff

Exactly one upstream file touched:

| File | Change | Lines |
|---|---|---|
| `swarm_rl/env_wrappers/quadrotor_params.py` | Added the five `--byzantine_*` flags to `add_quadrotors_env_args`, following the existing `--quads_*` pattern | **+13, −0** |

Nothing in `gym_art/` was modified. Nothing else in `swarm_rl/` was modified.
(`git diff --stat` also lists `README.md`; that edit predates this work and is unrelated.)

Old checkpoints absorb the new flags cleanly — `load_from_checkpoint` reports
`Adding new argument 'byzantine_enabled'=False that is not in the saved config file!` and
carries on, so `scripts/evaluate.py` and the existing baselines are unaffected.

New files, all under `byzantine_experiments/`:

```
agents/base.py            ByzantineAgent ABC, BYZANTINE_REGISTRY, clip_to_action_space
agents/noop.py            tier0_noop — the null-hypothesis control agent
wrappers/byzantine_env.py ByzantineEnvWrapper
rollout/env_factory.py    make_byzantine_env, byzantine_kwargs_from_cfg
rollout/collect.py        the trajectory collector CLI
tests/test_tier0.py       the validation suite
```

---

## 6. Validation results

### Test suite

`pytest byzantine_experiments/tests/test_tier0.py` — **14 passed** (~9 s).

| Spec test | Name | Result |
|---|---|---|
| 1 Identity | `test_identity_noop_reproduces_unwrapped_rollout` | PASS — obs and reward traces bit-identical over 30 steps |
| 2 Shape/dtype | `test_shape_and_dtype_preserved` | PASS |
| 3 Range | `test_range_compliance` | PASS — `env.action_space.contains()` per row, with a tier that would overshoot to +10 unclipped |
| 4 Isolation | `test_isolation_uncompromised_agents_untouched` | PASS — honest rows `array_equal` while a neighbour is sign-flipped |
| 5 Registry | `test_registry_entries_are_valid` | PASS — also asserts no in-place mutation of the input and that `perturb_broadcast_state` defaults to identity |
| 6 Log integrity | `test_perturbation_log_integrity` | PASS — 40 entries for T=20, 2 compromised; `t`, `agent_idx`, and `l2_deviation` all verified |
| 7 Determinism | `test_determinism_same_seed_same_perturbations` | PASS — seed 42 twice identical, seed 43 different |

Extras: `test_action_range_is_read_from_env_not_hardcoded`,
`test_accepts_list_of_per_agent_actions`, `test_random_assignment_resamples_across_episodes`,
`test_tier_with_strict_constructor_still_learns_the_action_range`, `test_clip_helper`,
`test_rejects_bad_configuration` — all PASS.

### Reward reproduction

Three 100-episode runs on `exp1_baseline_seed1`, deterministic (argmax) actions:

| Run | Mean episode reward | 95% bootstrap CI | Median | Std |
|---|---|---|---|---|
| `scripts/evaluate.py`, 2026-08-03 (`logs/seed1_eval_final.json`) | **1.4845** | — | — | 3.453 |
| `collect.py --tier tier0_noop --compromised-indices 0` | **−0.0717** | [−1.10, 0.90] | 2.096 | 5.091 |
| `collect.py --no-byzantine` (control) | **0.5985** | [−0.10, 1.26] | 1.632 | 3.499 |

**These means do not agree to within their nominal error bars, and the reason is the metric,
not the infrastructure.** Episode reward here is severely left-skewed — the no-op run has
median 2.096 but mean −0.07, with 19 % of episodes below −5 and a worst episode of −23.2
(the 2026-08-03 run's worst was −10.7). A handful of crash episodes move the mean by more
than a point, so n=100 pins it only to roughly ±1 and the three runs are not resolvable
from each other:

- **wrapped vs unwrapped** (the only comparison that isolates this code): diff −0.670,
  z = −1.08, permutation test **p = 0.28**. Not significant.
- unwrapped vs 2026-08-03: diff −0.886, z = −1.80.
- wrapped vs 2026-08-03: diff −1.556, z = −2.53.

The wrapped run is the furthest from the old number, but it is *not* the wrapper doing it —
the unwrapped control, which shares every line of `collect.py` except the wrapper itself,
sits between the two and is 1.8σ from the same baseline.

**So the gate was settled exactly rather than statistically.**
`test_noop_matches_unwrapped_on_the_real_policy_stack` drives the real checkpoint through
the full sample_factory stack twice in one process — once wrapped in `tier0_noop` with two
compromised drones, once unwrapped — with every reachable RNG pinned, and asserts the
observation, reward, and action traces are **bit-identical over 200 steps**. They are. The
wrapper provably contributes nothing, so the residual spread across the table above is
run-to-run variance in a heavy-tailed statistic.

**Two upstream determinism findings came out of building that test**, both recorded here
because Tier 3 will trip over them:

1. `QuadrotorSingle.__init__` calls `self._seed()` with no argument
   (quadrotor_single.py:234), so each drone's `np_random` — which picks spawn positions
   (quadrotor_single.py:394) — is seeded from OS entropy and is not controlled by
   `np.random.seed()`. The replay buffer separately samples with Python's `random`
   (quad_experience_replay.py:43). `seed_all()` pins both.
2. **With `quads_use_numba=True` the stack is not reproducible even against itself.** Two
   identically-configured, identically-seeded runs diverged by 0.028 at reset and 0.81
   after five steps. The jitted noise paths (`sense_noise.add_noise_numba`,
   `quadrotor_dynamics.step1_numba`) draw from numba's own RNG state, which
   `np.random.seed()` cannot reach. Setting `quads_use_numba=False` makes the same
   comparison exactly 0.0. The training config has numba on, so the 100-episode runs above
   inherit it; the identity test forces it off.

Neither is a wrapper bug and neither was fixed — `gym_art/` is untouched.

---

## 7. Known limitations

1. **`perturb_broadcast_state` is defined but not yet wired in.** Neighbour observations are
   assembled inside `QuadrotorEnvMulti.add_neighborhood_obs` from true positions
   (quadrotor_multi.py:250-258); nothing routes through the agent. A tier that lies about
   its state to neighbours — as opposed to acting badly — needs a hook inside that method,
   which means either an upstream edit or monkeypatching. The hook is in the ABC so the
   contract is fixed now, but treat it as unimplemented.

2. **Tier 3 needs a training-time integration, not just this rollout wrapper.** A trained
   adversary is a second policy network mapping obs → action. This API gives it `obs` and
   `t`, so *rolling it out* fits the existing interface (it ignores `honest_action`). But
   training `π_adv` against a frozen team requires the Byzantine env inside sample_factory's
   learner, which means wiring `byzantine_kwargs_from_cfg` into `quad_utils.py` (a real
   upstream edit) or registering a `quadrotor_multi_byzantine` env name. Budget for that.

3. **`byzantine_mask` is a single `(n_agents,)` vector.** Correct under the default `fixed`
   assignment, but in `random` mode it records only the last episode's assignment. If random
   assignment is ever used for a Phase 4 run, the mask must become `(n_episodes, n_agents)`.

4. **`collect.py` runs are not bit-reproducible**, for the two reasons in §6: it seeds only
   the global numpy and torch streams, so the per-drone spawn RNG and Python's `random` are
   uncontrolled, and `quads_use_numba=True` puts a whole RNG state out of reach regardless.
   Runs with the same `--seed` are statistically but not exactly reproducible. If Phase 3
   needs exact replay, call `seed_all()`-equivalent logic in `run_rollouts` **and** run with
   `--quads_use_numba=False` (which costs speed). Deliberately left as-is: the 100-episode
   runs above match the baseline's configuration, which is what makes them comparable.

5. **Collision count on the terminal step** may be attributed to the auto-reset episode when
   the replay buffer is active, since the env resets inside `step()`. One step in 1501, and
   only when the reset restores a buffered snapshot. Not worth fixing unless Phase 3 keys on
   end-of-episode collisions.

6. **`replay_buffer_sample_prob=0.75` is inherited from the training config**, so ~75 % of
   episodes start from a replayed near-collision snapshot rather than a fresh spawn. This is
   what the baseline eval numbers were produced under, so it is kept for comparability — but
   it means episode initial conditions are not i.i.d. Pass `--replay_buffer_sample_prob=0.0`
   to `collect.py` (it forwards unknown args to sample_factory) for clean-start episodes,
   and re-baseline if you do.

7. **File size: ~250 MB per 100-episode run**, dominated by `obs` (100 × 1501 × 8 × 54
   float32 = 259 MB on its own). `savez_compressed` barely helps — float32 sensor noise is
   incompressible. `byzantine_experiments/results/*.npz` is therefore in `.gitignore`; the
   `.json` sidecars stay tracked. Scaling to many tiers × seeds × `f` values will need a
   smaller `--n-episodes`, or dropping/subsampling `obs` if Phase 3 turns out to key only on
   positions and actions.
