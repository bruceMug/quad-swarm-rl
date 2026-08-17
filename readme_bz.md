# Byzantine experiments — what to run

Operational guide for the Byzantine detection work on branch `feature/byzantine`.
For *why* things are built this way, see
[`tier0_design.md`](byzantine_experiments/design/tier0_design.md) and
[`tier1_design.md`](byzantine_experiments/design/tier1_design.md).

---

## Current state — read this first

**Everything is computed. Nothing needs re-running.**

| Thing | State | Where |
| --- | --- | --- |
| Tier 0 infrastructure | done, 15 tests pass | `byzantine_experiments/{agents,wrappers,rollout}/` |
| Tier 1 agents (4 variants: gaussian, ou, dropout, stall) | done, 41 tests pass | `agents/tier1_stochastic.py` |
| Tier 1 main matrix — 63 runs × 50 episodes | done, 0 failures | `results/tier1/*.{npz,json}` |
| Tier 1 refinement grid — 33 runs | done | same |
| Tier 1C′ burst-stall grid — 27 runs | done | same |
| **Total: 123 runs** | done | 12 GB of `.npz` (gitignored) + `.json` sidecars (committed) |
| Severity calibration table (40 configs) | done | `results/tier1_calibration.{md,csv}` |
| Behaviour analysis + 30 figures | done | `results/tier1_behaviour.json`, `results/figures/` |
| Plain-language guide to every output | done | `byzantine_experiments/why/TIER1_explanation.md` |
| wandb | 123 runs uploaded, duplicates removed | project `byzantine_exp` |
| **Validation gate** | ✅ **met for all three usable variants** | see below |

Canonical Tier 1 settings, for Tier 3 to be matched against:

| variant | config | goal-RMSE degradation | action deviation |
| --- | --- | --- | --- |
| 1A gaussian | σ = 0.25, f = 0.125 | 23.6 % | 0.467 |
| 1B OU | σ_ou = 0.06, f = 0.125 | 35.3 % | 0.213 |
| 1C′ stall | duty = 0.25, mean_stall_s = 0.3, hold, f = 0.125 | 25.5 % | 0.089 |

(1C `tier1_dropout` never reaches the band at any `p_drop` — it is superseded by 1C′
`tier1_stall`, of which it is the short-stall limit. See `design/tier1_design.md` §8.)

Each run is one (seed × variant × severity × f) combination and writes a `.npz`, a `.json`,
and a log. Logs are merged into **one file per variant**
(`logs/tier1/{clean,gaussian,ou,dropout,stall}.log`) plus the sweep-level
`logs/tier1_matrix.log`, so a 123-run sweep leaves 5 log files, not 123.

**Next:** Tier 2. Spec at `byzantine_experiments/specs/TIER2_SIGNFLIP.md`.

---

## 1. Setup (once per shell)

```bash
conda activate swarm_r311_recovered     # NOT swarm_r311 — its protobuf is broken
cd /scratch/bmugizi/marl_research/quad-swarm-r1
wandb login                             # key from https://wandb.ai/authorize
```

Everything below runs from the **repo root**.

Verify the environment is intact — takes ~30 s, needs no GPU:

```bash
pytest byzantine_experiments/tests/test_tier1.py -q      # 41 tests, ~17 s
pytest byzantine_experiments/tests/test_tier0.py -q      # 15 tests, ~2 min (2 use the checkpoints)
```

Expect: `41 passed` and `15 passed`. If wandb auth fails, check your AFS token with
`tokens` and renew with `klog` — `~/.netrc` lives on AFS.

---

## 2. How the Tier 1 sweeps were produced (all complete — for reference)

**Why.** Variant 1C (`tier1_dropout`) turned out to be a near-null fault: at the env's 100 Hz
control rate, an i.i.d. per-step dropout lasts ~1 control tick, and reusing a 10 ms-old
command is barely different from using the current one. It never entered the 20–40 % damage
band at any `p_drop` tested — it jumped from 16.4 % at p=0.85 straight to 65.4 % at p=0.98.

`tier1_stall` fixes the parameterisation. Instead of "how often does a stall start", it takes
the two physically meaningful knobs:

- `mean_stall_s` — how **long** a stall lasts, in seconds (a real radio dropout or compute
  hang lasts 100s of ms, not one tick)
- `duty_cycle` — what **fraction** of the time the drone is stalled

implemented as a two-state Markov chain. Setting `mean_stall_s = 1/((1−p)·100)` and
`duty_cycle = p` reproduces the old `tier1_dropout` exactly, so the two are nested and the
old results stay comparable. Unit-tested: measured stall length and duty cycle match the
config to within 10 %, and at an **equal duty cycle of 0.25** the burst version perturbs the
command stream **35× more** than the i.i.d. version.

```bash
# the three sweeps, in the order they were run (all already done)
PARALLEL=12 bash scripts/byz_rollout_tier1.sh              # main matrix, 63 runs, ~85 min
REFINE=1 PARALLEL=12 bash scripts/byz_rollout_tier1.sh     # refinement,  33 runs, ~45 min
STALL=1  PARALLEL=12 bash scripts/byz_rollout_tier1.sh     # burst stall, 27 runs, ~35 min
```

- **Runtime:** ~35 min for the stall grid (27 runs = 3 durations × 3 duty cycles × 3 seeds).
- **Disk:** ~3.5 GB.
- **Grid:** `mean_stall_s` ∈ {0.1, 0.3, 1.0} s × `duty_cycle` ∈ {0.10, 0.25, 0.50}, mode
  `hold`, f = 0.125, seeds 1–3.

Then regenerate the analysis (seconds to ~1 min each):

```bash
python -m byzantine_experiments.analysis.aggregate_tier1
python -m byzantine_experiments.analysis.behaviour --plots --wandb
```

**What to expect.** A 2-episode smoke test at `duty_cycle=0.25, mean_stall_s=0.5` gave action
deviation **0.142** (vs 0.008 for old dropout at p=0.3) and goal RMSE 1.49 against a clean
baseline of ~1.06 — roughly **40 %** degradation. So the grid should bracket the 20–40 % band
comfortably; expect the in-band configs somewhere around `duty_cycle` 0.10–0.25 with
`mean_stall_s` 0.3–1.0. `aggregate_tier1` marks in-band configs with `*`.

Final validation-gate status by goal-RMSE degradation at f = 0.125:

| variant | in-band config(s) | result |
| --- | --- | --- |
| 1A gaussian | σ = 0.25 / 0.30 / 0.40 | ✅ 23.6 / 35.2 / 38.1 % |
| 1B OU | σ_ou = 0.06 | ✅ 35.3 % |
| 1C dropout | none | ❌ 16.4 % (p=0.85) → 65.4 % (p=0.98) |
| 1C′ stall | d0.25/ms0.3, d0.25/ms1.0, d0.50/ms0.1, d0.50/ms0.3 | ✅ 25.5 / 21.3 / 20.6 / 36.1 % |

Re-running any sweep is safe: completed configs are skipped unless you pass `FORCE=1`.

---

## 3. Episode-count sensitivity (optional, your call)

You asked whether 50 is the right number. **50 came from your spec**
(`TIER1_STOCHASTIC.md`, matrix table: "Episodes per run | 50"). It is defensible for the
stable metrics and too small for the reward metric — see §5.

To measure how much n actually buys you, run one config at four episode counts. This uses
the *most* affected config so the effect is visible:

```bash
# ~55 min total, ~7 GB. Each line is independent — run some or all.
for N in 25 50 100 200; do
  python -m byzantine_experiments.rollout.collect \
    --checkpoint train_dir/exp1_baseline_seed1 \
    --tier tier1_gaussian --tier-config '{"sigma": 0.25}' \
    --compromised-indices 0 --n-episodes $N --seed 1 --device cpu \
    --out byzantine_experiments/results/sensitivity/t1_gaussian_sigma0.25_n${N}.npz \
    --wandb --wandb-group tier1_episode_sensitivity \
    --wandb-tags sensitivity tier1_gaussian n${N}
done

# and the matching clean references (degradation needs a like-for-like baseline)
for N in 25 50 100 200; do
  python -m byzantine_experiments.rollout.collect \
    --checkpoint train_dir/exp1_baseline_seed1 --no-byzantine \
    --n-episodes $N --seed 1 --device cpu \
    --out byzantine_experiments/results/sensitivity/t1_clean_n${N}.npz \
    --wandb --wandb-group tier1_episode_sensitivity --wandb-tags sensitivity clean n${N}
done
```

Roughly 20 s/episode/run, so n=25 → ~9 min, n=200 → ~70 min. Run them in parallel by
appending `&` and `wait`, or prefix with `OMP_NUM_THREADS=1`.

**What to expect:** the `*_sem` fields in each run's wandb summary should fall as `1/√n`.
Concretely, `episode_reward_sem` ≈ 0.70 at n=25 → ≈0.25 at n=200, and
`goal_rmse_sem` ≈ 0.09 → ≈0.03. The point of the exercise is to see that even at n=200 the
*reward* error bar stays too wide to rank severities, while goal RMSE tightens usefully.

To run the **whole matrix** at a different n instead:

```bash
N_EPISODES=100 FORCE=1 PARALLEL=12 bash scripts/byz_rollout_tier1.sh   # ~3 h, ~17 GB
```

---

## 4. Running one rollout by hand

```bash
python -m byzantine_experiments.rollout.collect \
  --checkpoint train_dir/exp1_baseline_seed1 \
  --tier tier1_gaussian \
  --tier-config '{"sigma": 0.25}' \
  --compromised-indices 0 \
  --n-episodes 50 \
  --seed 1 \
  --out byzantine_experiments/results/my_run.npz \
  --wandb --wandb-group my_condition
```

| Flag | Effect |
| --- | --- |
| `--no-byzantine` | clean reference, no wrapper at all |
| `--compromised-indices 0,1` | 2 of 8 drones ⇒ f = 0.25 |
| `--no-npz` | metrics + wandb only, skip the ~120 MB trajectory file |
| `--assignment-mode random` | resample which drones are compromised each episode |
| `--stochastic` | sample actions instead of the distribution argmax |
| `--device gpu` | use the T4 (CPU is fine and parallelises better) |

Unrecognised flags pass through to sample_factory, e.g. `--replay_buffer_sample_prob=0.0`.

**Registered tiers:** `tier0_noop`, `tier1_gaussian`, `tier1_ou`, `tier1_dropout`, `tier1_stall`

```bash
python -c "from byzantine_experiments.agents import BYZANTINE_REGISTRY as R; print(sorted(R))"
```

**Sweep knobs** (`scripts/byz_rollout_tier1.sh`):

```bash
DRY_RUN=1 bash scripts/byz_rollout_tier1.sh     # print the job list, run nothing
WANDB=0   bash scripts/byz_rollout_tier1.sh     # local only
FORCE=1   bash scripts/byz_rollout_tier1.sh     # redo configs that already have output
KEEP_RUN_LOGS=1 bash scripts/byz_rollout_tier1.sh   # keep per-run logs instead of merging
bash scripts/byz_rollout_tier1.sh --consolidate-logs # merge per-run logs after the fact
```

The sweep **resumes**: a config whose `.json` exists is skipped, so you can interrupt and
restart with the same command.

**Videos** (needs a virtual X server — the node is headless):

```bash
xvfb-run -a python -m byzantine_experiments.rollout.render \
  --checkpoint train_dir/exp1_baseline_seed1 \
  --tier tier1_ou --tier-config '{"sigma_ou": 0.2}' \
  --compromised-indices 0 --n-episodes 5 --every-nth 1 \
  --out byzantine_experiments/results/videos/t1_ou_sigmaou0.2.mp4
```

---

## 5. Reading the numbers — the one thing that will bite you

**Do not rank severities by reward degradation %.** Mean episode reward here is ≈1.1 with a
per-episode std of ≈3.5 and a heavy left tail, so `(clean − byz)/|clean|` divides by a
near-zero denominator. In the completed matrix its standard error is **±60 to ±77 percentage
points on every single row**. Two dropout configs show up "in the 20–40 % band" on reward
purely by noise — their actual goal-RMSE effect is 6–8 %.

Use **goal RMSE** (RMS distance to assigned goal = formation error). True zero, and a
seed-to-seed spread of 3–12 pp rather than ±70. Both columns are in the calibration table
with their uncertainties.

**The headline Tier 1 result:** the damage is almost entirely confined to the faulty drone.
Honest-neighbour goal error stays within ±10 % of clean in *every* config, including σ=0.5
at f=0.25 where the faulty drone is 91 % worse and on the floor 16 % of the time. That has
consequences for Tier 3 — see `tier1_design.md` §4.

---

## 6. When something goes wrong

**A run failed but the data is there.** Almost certainly a wandb auth blip (AFS `~/.netrc`
read by 12 workers at once). The collector writes `.npz`/`.json` *before* logging and treats
a logging error as non-fatal, so re-log without re-simulating:

```bash
python -m byzantine_experiments.analysis.backfill_wandb --dry-run
python -m byzantine_experiments.analysis.backfill_wandb
```

**Duplicate wandb runs** after re-running a sweep (wandb does not overwrite by name):

```bash
python -m byzantine_experiments.analysis.dedupe_wandb            # show
python -m byzantine_experiments.analysis.dedupe_wandb --delete   # keep newest of each name
```

**What's missing?**

```bash
ls byzantine_experiments/results/tier1/*.json | wc -l   # 123
grep -c '^DONE' logs/tier1_matrix.log
grep '^FAIL'    logs/tier1_matrix.log
```

**Disk:** each 50-episode run is ~135 MB of `.npz`; the matrix is 8.5 GB.
`byzantine_experiments/results/*.npz` and `logs/tier1` are gitignored; the `.json` sidecars
are committed and carry every metric, so the numbers survive without the bulk.

---

## 7. Where things are in wandb

Project **`byzantine_exp`**, entity **`bmugizi-carnegie-mellon-university`** — same project
as the baseline training runs.

| Level | Meaning | Example |
| --- | --- | --- |
| `group` | one experimental condition; **aggregates the 3 seeds into a mean±band** | `tier1_ou_sigmaou0.2_f0125`, `clean_baseline` |
| `job_type` | `rollout` for data, `analysis` for tables/figures | |
| `tags` | filters | `tier1`, `seed1`, `f0125`, `sigma0.25`, `nbyz1`, `8agents`, `mix` |
| `name` | the run | `t1_gaussian_s1_sigma0.25_f0125` |

Every rollout run carries: all scalar metrics in the **summary** (sortable in the runs
table), an **`episodes` Table** with one row per episode, **`hist/*`** distributions,
**`episode/*`** curves, and the full run provenance in **config** (checkpoint, tier config,
f, DR settings, git SHA, command line). The calibration table and behaviour figures are
separate `analysis` runs in group `tier1_analysis`.

---

## 8. Reproducibility caveats

Two upstream simulator properties (details in `tier0_design.md` §6):

1. Each drone's spawn RNG is seeded from OS entropy (`quadrotor_single.py:234`); the replay
   buffer samples with Python's `random`. Neither is controlled by `np.random.seed()`.
2. With `quads_use_numba=True` (the trained config) the sim is not reproducible even against
   itself — the jitted noise paths carry their own RNG state.

So two runs with the same `--seed` are statistically but not bit-identically equal. This is
why the calibration table's numbers shifted when the matrix was re-run, and why every figure
is quoted with a seed spread. For exact replay you need `--quads_use_numba=False` plus
per-drone seeding (`seed_all()` in `tests/test_tier0.py` shows how).

---

## 9. Adding a new Byzantine tier

1. Subclass `ByzantineAgent` in `byzantine_experiments/agents/`, decorate with
   `@register_byzantine('tier2_whatever')`. Clip with `self.clip(...)` — the wrapper
   **raises** on out-of-range output rather than silently fixing it.
2. Import it in `agents/__init__.py` so registration happens.
3. `tests/test_tier0.py::test_registry_entries_are_valid` picks up any new tier automatically
   and enforces the ABC contract. Add variant-specific tests alongside.
4. Copy a sweep block in `scripts/byz_rollout_tier1.sh`.

The wrapper gives each agent its own action, its own observation, and `t` — nothing about the
team. Tier 2/3 agents that need to lie about broadcast state must implement
`perturb_broadcast_state()`, which is **declared but not yet wired into the env**
(`tier0_design.md` §7, limitation 1).
