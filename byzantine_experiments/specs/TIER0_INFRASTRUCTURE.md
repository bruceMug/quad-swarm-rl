# TIER 0 — Byzantine Infrastructure

**Audience:** claude code
**Repo:** `quad-swarm-r1`, branch `feature/byzantine`
**Build this before Tiers 1–3. All three tiers depend on it.**

---

## Objective

Build the shared scaffolding that lets any Byzantine agent implementation be injected into
an already-trained `quadrotor_multi` policy at rollout time, without retraining and without
modifying `gym_art/`.

Three deliverables: an agent abstraction + registry, an environment wrapper that applies it,
and a trajectory collector that writes the data Phase 3 will consume.

---

## Context you need

- The env is `gym_art/quadrotor_multi/quadrotor_multi.py` (`QuadrotorEnvMulti`).
- Training/eval entry points are `swarm_rl/train.py` and `swarm_rl/enjoy.py`, both built on
  Sample Factory v2.1.1.
- CLI arguments prefixed `--quads_*` are declared in
  `swarm_rl/env_wrappers/quadrotor_params.py`. New Byzantine flags go there and nowhere else.
- Trained baselines live at `train_dir/exp1_baseline_seed{1,2,3}/checkpoint_p0/`.
- The env is multi-agent: 8 drones. Per-step actions are shape `(num_agents, 4)` — normalised
  motor thrusts in `[-1, 1]` (verify this range against `quadrotor_single.py` before coding;
  do not assume).

**Read before writing code:**
1. `gym_art/quadrotor_multi/quadrotor_multi.py` — how `step()` receives and dispatches actions
2. `gym_art/quadrotor_multi/get_state.py` — how observations are constructed, including the
   neighbour-observation block
3. `swarm_rl/env_wrappers/quadrotor_params.py` — the flag declaration pattern
4. `swarm_rl/enjoy.py` — how a checkpoint is loaded and rolled out

---

## Deliverable 1 — `byzantine_experiments/agents/base.py`

Define an abstract base class and a name→class registry.

```python
from abc import ABC, abstractmethod
import numpy as np

BYZANTINE_REGISTRY = {}

def register_byzantine(name):
    def deco(cls):
        BYZANTINE_REGISTRY[name] = cls
        return cls
    return deco


class ByzantineAgent(ABC):
    """
    Transforms an honest policy's output into Byzantine behaviour.

    Contract:
      - perturb_action() is called once per env step per compromised agent
      - it MUST return an action in the same dtype/shape/range as the input
      - it MUST NOT mutate its input array in place
      - it MAY maintain internal state across steps (reset via reset())
    """

    def __init__(self, agent_idx: int, action_dim: int, seed: int = 0, **kwargs):
        self.agent_idx = agent_idx
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed + 1000 * agent_idx)

    @abstractmethod
    def perturb_action(self, honest_action: np.ndarray, obs: np.ndarray,
                       t: int) -> np.ndarray:
        ...

    def perturb_broadcast_state(self, true_state: np.ndarray, t: int) -> np.ndarray:
        """
        Optional: corrupt what THIS agent reports to neighbours.
        Default is honest (identity). Override in tiers that lie about state.
        """
        return true_state

    def reset(self):
        """Called at episode boundaries. Clear any internal state."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
```

**Design constraint that matters:** `perturb_action` receives the honest action, not the
observation-to-action mapping. This means every tier operates on top of the *same* frozen
honest policy, which is exactly what makes the tiers comparable. Do not give tiers access to
the policy network weights.

**Range clipping is the agent's responsibility, not the wrapper's.** Every tier must clip
its output to the env's legal action range. Add a shared helper in `base.py`:

```python
def clip_to_action_space(action, low=-1.0, high=1.0):
    return np.clip(action, low, high)
```

Confirm `low`/`high` from the env's `action_space` rather than hardcoding.

---

## Deliverable 2 — `byzantine_experiments/wrappers/byzantine_env.py`

A `gymnasium.Wrapper` around `QuadrotorEnvMulti` that intercepts `step()`.

Required behaviour:

- Constructor takes: the wrapped env, a list of compromised agent indices, a Byzantine agent
  class name (looked up in `BYZANTINE_REGISTRY`), a config dict passed to the agent
  constructor, and a seed.
- In `step(actions)`: for each compromised index `i`, replace `actions[i]` with
  `byz_agents[i].perturb_action(actions[i], last_obs[i], self._t)`. Pass the modified action
  array to the wrapped env.
- In `reset()`: call `reset()` on every Byzantine agent, zero the step counter.
- Maintain a per-episode log: for every step, record `(t, agent_idx, honest_action,
  perturbed_action, l2_deviation)`. Expose via `get_perturbation_log()`. You need this to
  verify attacks are actually firing.
- **Compromised indices must be stable within an episode** and configurable across episodes.
  Support two modes: `fixed` (always agents `[0]` or `[0,1]`) and `random` (resampled each
  episode from the given seed). Default to `fixed` — random assignment adds variance you do
  not want in early experiments.

Handle the Sample Factory multi-agent action format carefully. Inspect what `enjoy.py`
actually passes to `env.step()` — it may be a list of per-agent arrays rather than a single
stacked `(8, 4)` array. Write the wrapper to accept both and normalise internally.

---

## Deliverable 3 — `byzantine_experiments/rollout/collect.py`

A script that loads a checkpoint, wraps the env with `ByzantineEnvWrapper`, runs N episodes,
and writes the trajectory data Phase 3 needs.

Output format — one `.npz` per run, containing:

| Key | Shape | Contents |
|-----|-------|----------|
| `obs` | `(n_episodes, T, n_agents, obs_dim)` | Per-agent observations |
| `actions_executed` | `(n_episodes, T, n_agents, act_dim)` | What the env actually received |
| `actions_honest` | `(n_episodes, T, n_agents, act_dim)` | What the policy output before perturbation |
| `rewards` | `(n_episodes, T, n_agents)` | Per-agent reward |
| `positions` | `(n_episodes, T, n_agents, 3)` | XYZ, for formation error |
| `velocities` | `(n_episodes, T, n_agents, 3)` | |
| `collisions` | `(n_episodes, T)` | Collision count per step |
| `byzantine_mask` | `(n_agents,)` bool | Ground-truth labels — **critical for Phase 4 AUC** |
| `episode_lengths` | `(n_episodes,)` | For handling early termination |

Plus a sidecar `.json` with the full run config: checkpoint path, seed, tier name, tier
config, `f` fraction, DR settings, git SHA.

Episodes terminate at different lengths. Pad with NaN and record true lengths rather than
truncating to the minimum — Phase 3 windowing needs to know where real data ends.

CLI signature:

```
python -m byzantine_experiments.rollout.collect \
  --checkpoint train_dir/exp1_baseline_seed1 \
  --tier tier1_stochastic \
  --tier-config '{"sigma": 0.25}' \
  --compromised-indices 0 \
  --n-episodes 50 \
  --seed 1 \
  --out byzantine_experiments/results/t1_s1_sigma025_f0125.npz
```

---

## Deliverable 4 — CLI flag registration

In `swarm_rl/env_wrappers/quadrotor_params.py`, add a small block of `--byzantine_*` flags
following the existing `--quads_*` pattern. Keep it minimal:

```
--byzantine_enabled           bool, default False
--byzantine_tier              str, default ''      (registry key)
--byzantine_indices           str, default ''      (comma-separated, e.g. "0,3")
--byzantine_config            str, default '{}'    (JSON blob)
--byzantine_seed              int, default 0
```

This diff should be under 30 lines. Do not modify any other upstream file.

---

## Deliverable 5 — `byzantine_experiments/tests/test_tier0.py`

pytest coverage, minimum:

1. **Identity test** — a no-op Byzantine agent produces byte-identical rollouts to an
   unwrapped env given the same seed. If this fails, the wrapper is leaking state.
2. **Shape/dtype preservation** — perturbed actions match honest actions in shape and dtype.
3. **Range compliance** — no perturbed action ever falls outside `env.action_space`.
4. **Isolation** — uncompromised agents' actions are bit-identical to the unwrapped case.
5. **Registry** — every registered name instantiates and satisfies the ABC.
6. **Log integrity** — `get_perturbation_log()` has exactly `T × len(compromised)` entries.
7. **Determinism** — the same seed produces the same perturbation sequence twice.

Test 4 is the one that catches the most bugs. Write it first.

---

## Validation gate

Tier 0 is done when: all seven tests pass, and a no-op rollout on
`exp1_baseline_seed1` reproduces the mean episode reward from
`logs/seed1_eval_final.json` to within numerical noise. Report the two numbers side by side.

---

## Required output: `byzantine_experiments/design/tier0_design.md`

After implementation, write this document. Include:

1. **Action format findings** — what shape/dtype/range does the env actually use? What did
   `enjoy.py` pass to `step()`? Quote the relevant source lines.
2. **Where you hooked in and why** — the exact call path from policy output to `env.step()`,
   and where the wrapper sits in it.
3. **Deviations from this spec** — anything you changed, and the reason.
4. **The `.npz` schema as actually implemented**, with real dtypes and shapes from a test run.
5. **Upstream diff** — the complete list of upstream files touched and the line count of each
   change.
6. **Validation results** — the seven test outcomes and the reward-reproduction comparison.
7. **Known limitations** — anything that will need revisiting for Tier 3.
