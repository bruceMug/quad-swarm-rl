# Byzantine experiments — what to run

Operational guide for the Byzantine detection work on branch `feature/byzantine`.
For *why* things are built this way, see
[`tier0_design.md`](byzantine_experiments/design/tier0_design.md) and
[`tier1_design.md`](byzantine_experiments/design/tier1_design.md).

> **Run every command from the repo root (`quad-swarm-r1/`).** The package imports from anywhere
> (it is pip-installed), but data paths such as `byzantine_experiments/results_v2/...` and
> `train_dir/...` are relative — from `marl_research/` a script reports "Missing clean reference".

---

## Current state — read this first

> ### ⚠️ Every number collected before 2026-09-24 is unreliable — root cause found and fixed
>
> *(Fixed in two steps: 2026-09-21 made runs start identically, 2026-09-24 made every episode match. Only sidecars tagged `seeding: per_episode_v2` — everything under `results_v2/` — are valid.)*
>
> *What "unreliable" means, measured (2026-10-21):* large effects on the faulty or attacking drone
> itself — its behaviour, survival, and the swarm-wide degradation it causes — replicate in v2
> (Tier 1: 20 of 29 configs within 5 pp). The small effects on the **honest** drones, and every
> significance claim, do not (sign changed in 17 of 48 Tier 1/2 configs). Quote only v2 numbers;
> `tier1_design.md` and `tier2_design.md` §0 list what holds.
>
> **The bug.** `collect.py` seeded only the global numpy and torch streams. Spawn positions
> and goals come from each `QuadrotorSingle`'s own `np_random`, which upstream seeds from **OS
> entropy at construction**. Two clean rollouts with the identical `--seed 777` differed by
> **2.9 m at t = 0** — different scenarios entirely. `--seed` was cosmetic.
>
> The correct seeding existed all along in `tests/test_tier0.py::seed_all`, whose docstring
> documents this exactly. It was never wired into the collector, and `collect.py`'s own comment
> asserted the opposite ("draws its spawn/goal randomisation from the global numpy RNG").
>
> **The damage.** Each run drew an uncontrolled scenario set, giving a run-to-run sd of
> **τ ≈ 7 %** of mean goal RMSE that no reported CI accounted for. Folding it in turns
> **20 significant findings into 0 across 245 rollouts**. Measured directly: the corpus's
> strongest attack (Tier 2 `mirror` liar, reported **+30.4 %, p = 0.0007**) re-measured with
> 3 replicates per arm gives **−0.9 %, p = 0.85**. The original was the highest attacked run
> against the lowest clean run; all 7 runs span 1.03–1.30 with no separation by arm.
>
> **The fix.** `collect.py::seed_every_rng` now seeds every drone's RNG plus Python's
> `random`. Verified: two rollouts at the same seed draw **identical spawns (0.000 m)**,
> sequentially and concurrently. Numba's JIT RNG stays unreachable, so trajectories drift
> 0.1–0.5 m by episode end — scenarios match, which is what matters.
>
> **What this unlocks.** Clean and attacked runs at the same seed are now **paired**: episode
> *i* is the same scenario in both. That cancels the run-level offset exactly instead of
> estimating it, and cancels most scenario variance too. The existing sweeps already pass the
> same `--seed` to both arms, so re-running them yields paired data at no extra design cost.
>
> **A second fix was needed (2026-09-24).** Seeding once per run pairs only the first few
> episodes: collisions draw from the global `np.random` stream, so an attacked run (more
> collisions) advances it differently and later scenarios diverge — a clean/`mirror` pair
> matched on episodes 0–2 and on none after, per-episode correlation **−0.08**. The collector
> now re-seeds before **every** episode (`episode_seed(seed, i)`), so episode *i* depends only
> on *(seed, i)*. Verified: **10/10 identical spawns** with `mirror` colliding 5× more than
> clean, correlation between arms **+0.81** — pairing cuts the standard error to ~0.43× at no
> extra compute, and removes the run-level offset outright.
>
> Nothing should be quoted until conditions are re-collected with this collector. See §2C
> step 13 and `design/tier3_design.md` §6.18.

**LEGACY — do not cite; superseded by `results_v2/` (see `design/tier3_audit.md`).** Canonical Tier 1 settings as measured by the broken collector:

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

**Next:** run §2C step 4 (~15 min) to put round 2's number on record, then build round 3.

**Round 2 (f = 0.25) failed, and so did round 1.** 0 of 28 checkpoints pass the gate. Round 2
was *worse* than round 1 (honest damage max +5.8 % vs +17.7 %), and `adv/r_team` was flat
across 200M steps — every point of reward improvement came from crashing less, not from
damaging anyone.

**Nothing any trained adversary did was statistically distinguishable from clean.** The best
point estimate in the whole corpus, +17.2 %, has a 95 % CI of [−13.9, +48.3] and p = 0.27.
Four checkpoints *are* significant — all on the **negative** side, the honest team doing
measurably better with the adversary present.

**The result came from re-reading Tiers 1–2 for free.** Group every attack by the *channel*
it acts through and test each against clean
(`python -m byzantine_experiments.analysis.channel_ceiling`):

| tier | channel | f | runs | max deg % | **signif worse** | signif better |
| --- | --- | --- | --- | --- | --- | --- |
| 2 hand-designed | motors only | 0.125 | 27 | 22.2 % | **2** | 2 |
| 2 hand-designed | motors only | 0.250 | 24 | 15.3 % | **0** | 1 |
| 2 hand-designed | **observation channel** (`liar`) | 0.125 | 9 | **30.4 %** | **3** | 0 |
| 2 hand-designed | **observation channel** | 0.250 | 9 | 29.9 % | **3** | 0 |
| 3 **trained** | motors only | 0.125 | 14 | 17.2 % | **0** | 2 |
| 3 **trained** | motors only | 0.250 | 14 | 7.7 % | **0** | 1 |

"Motors only" = the attacker corrupts its own 4 thrusts and nothing else — that is every Tier 3
adversary. "Observation channel" = it also broadcasts false state.

* Across **79 motors-only runs** (51 hand-designed + all 28 trained), exactly **2** do
  significant damage. The trained adversary sits *inside* the hand-designed band, so
  **the optimiser is not the bottleneck — the channel is.**
* Motors-only attacks are **as likely to help as to hurt** (2 worse vs 6 better). A drone that
  flails and crashes removes a crowding constraint from the formation.
* **`t2_liar_s3_mirror_f0125` clears the gate: +30.4 %, p = 0.0007** — with a *hand-designed,
  unoptimised* lie.

**Next:** run the round-3 gate sweep (§2C step 6, ~40 min).

**Round 3 trained and was evaluated. It fails too — 0 / 42 across all four rounds.** But
`_lieonly` fails on *one criterion only*: it meets survival (96–100 %, against a ≥ 70 % gate)
and action ratio (0 % saturation) outright, which no Tier 3 adversary had ever done. It simply
does not damage the honest team — max **+11.5 %, p = 0.21**.

**And the reason is a confound in my own experiment design, which corrects what I told you
earlier.** The "+30.4 % headroom" that motivated round 3 came from *pooled* data across three
baseline seeds. Split it
(`python -m byzantine_experiments.analysis.channel_ceiling`):

| channel | baseline seed | runs | max deg % | signif worse | clean goal RMSE |
| --- | --- | --- | --- | --- | --- |
| motors only | 1 | 17 | **4.9 %** | 0 | 1.1123 |
| motors only | 2 | 17 | 18.6 % | 0 | 1.0463 |
| motors only | 3 | 17 | 22.2 % | **2** | 1.0085 |
| observation | 1 | 6 | **12.6 %** | 0 | 1.1123 |
| observation | 2 | 6 | 17.0 % | 1 | 1.0463 |
| observation | 3 | 6 | **30.4 %** | **5** | 1.0085 |
| observation (**trained**) | 1 | 14 | 17.6 % | 0 | 1.1123 |

**Baseline seed matters more than channel does.** Seed 1 has *zero* significant results across
all 23 hand-designed attacks on either channel; seed 3 has 7. The 30.4 % that justified round 3
is a **seed-3** number — and round 3 attacked **seed 1**, chosen for continuity with rounds 1–2.
Right instinct for comparability, wrong one for testing the hypothesis.

Apples to apples, seed 1 only, observation channel:

| | max honest degradation |
| --- | --- |
| hand-designed `liar` | +12.6 % (p = 0.20) |
| **trained `_lieonly`** | **+11.5 % (p = 0.21)** |

The learned adversary **matches the hand-designed ceiling on the baseline it attacked**. It is
still not underperforming; there was nothing to find on seed 1.

**The finding worth writing up:** the most attackable baseline is the one with the *lowest*
clean goal RMSE. The tightest formation is the most brittle — less slack to absorb a perturbed
member. Checked against two artifacts: it survives in absolute metres (0.307 vs 0.140, 2.2×),
and seed 3's narrower CIs explain some of the significance gap but not an effect size 2.4×
larger.

**Round 4 trained (seed 3) and its objective is flat too** — `adv/r_team` slope
−0.111 ± 0.082 per 100M steps, within 2 SE of zero. But "flat" is not "doing nothing", and the
reason for the gap is now diagnosed.

Put the hand-designed attacks on the adversary's **own objective scale** (seed 3):

| | honest episode reward | `adv/r_team` equivalent |
| --- | --- | --- |
| clean, no attack | +1.2716 | **−25.43** |
| **trained, 200M steps** | ~+0.42 | **−7.6 … −9.9** |
| hand-designed `mirror` lie | −0.8321 | **+16.64** |

The adversary does move honest reward from +1.27 to ~+0.42 (~64 % degradation) — but a
hand-designed lie reaches +16.64 on the same axis. **It captures ~40 % of the available damage
on the quantity it is directly optimising.** The objective is not mis-specified: honest reward
and honest goal RMSE correlate at **Pearson +0.915** across the 18 Tier 2 `liar` runs.

**The gap is an information asymmetry, and it is ours.** `mirror` reports `2 × centre − pos`,
which needs **absolute position**. The hand-designed `liar` is handed exactly that. The learned
adversary never saw it — the 54-D observation is `pos − goal`, `vel`, `R`, `omega` and
*relative* neighbour offsets. Measured: a linear probe recovers absolute x/y/z at
**R² = 0.39 / 0.35 / 0.27**, while `mirror` is within the action budget on **99.2 %** of steps.
The action space could express the attack; the observation could not specify it.

**Next:** §2C step 8 (round 4's gate sweep, ~20 min) and step 9 (round 5 training).

**A methodological result worth its own paragraph in the paper.** Three of the spec's metrics
mislead, each caught by inspecting a result that looked too good:

* The ≥ 30 % *team-reward* gate is trivially passed — a Tier 1 noise fault scores 214 % —
  because the all-8 reward mean is dominated by the compromised drone's own collapse
  (`design/tier2_design.md` §10).
* The *goal-RMSE* gate has the identical flaw, and a learned adversary **exploits it
  deliberately**: round 1's best-looking checkpoint scored +384 % on its own drone and
  +17.7 % on the honest six (`design/tier3_design.md` §3.2).
* Point estimates were reported with **no uncertainty** against a 30 % threshold, when
  per-episode goal RMSE has a **42 % coefficient of variation** — a ±22 pp CI at n = 20
  (`design/tier3_design.md` §4.3). Detecting a true 30 % effect at 80 % power needs n ≈ 31
  per arm.

The general statement: any Byzantine-severity metric that averages over the compromised agent
measures self-destruction, not attack. That is tolerable against hand-set severity knobs and
fatal against an optimiser. All gates now key on honest-agent-only quantities and carry CIs.

---

## 0. wandb on this machine

AFS home (`/afs/andrew.cmu.edu/usr6/bmugizi`) sits at **~90 % of its 2 GB quota**, and AFS
reports "quota exceeded" as `EACCES`. wandb writes its cache, run data and artifact staging
under `~/.cache/wandb` and `~/.local/share/wandb`, so a long rollout finishes, writes its
`.npz`/`.json`, and then fails at the logging step with

```
PermissionError: [Errno 13] ... /.local/share/wandb/artifacts/staging/tmpXXXX
```

**The rollout is unaffected** — `collect.py` writes results before logging and treats a logging
failure as non-fatal — and the run's metrics still reach wandb; only the artifact upload dies.
Fix it for good:

```bash
source scripts/byz_wandb_env.sh     # or add its exports to ~/.bashrc
```

If a run ever does fail to log, nothing is lost — re-log from disk without re-simulating:

```bash
python -m byzantine_experiments.analysis.backfill_wandb --results-dir <dir> --dry-run
```

---

## 1. Setup (once per shell)

```bash
conda activate swarm_r311_v2            # rebuilt env; the older ones were purged
cd /scratch/bmugizi/marl_research/quad-swarm-r1
wandb login                             # key from https://wandb.ai/authorize
```

Everything below runs from the **repo root**.

Verify the environment is intact — takes ~30 s, needs no GPU:

```bash
pytest byzantine_experiments/tests/test_tier1.py -q      # 41 tests, ~17 s
pytest byzantine_experiments/tests/test_tier2.py -q      # 31 tests, ~6 s
pytest byzantine_experiments/tests/test_tier3.py -q      # 16 tests, ~32 s
pytest byzantine_experiments/tests/test_tier0.py -q      # 15 tests, ~2 min (2 use the checkpoints)
```

Expect: `41 passed`, `31 passed, 3 skipped`, `16 passed, 1 skipped`, and `15 passed`. If wandb auth fails, check your AFS token with
`tokens` and renew with `klog` — `~/.netrc` lives on AFS.

---

## 2C. Tier 3 — the adversaries

Spec: [`specs/TIER3_ADVERSARIAL.md`](byzantine_experiments/specs/TIER3_ADVERSARIAL.md).
Design notes and results: [`design/tier3_design.md`](byzantine_experiments/design/tier3_design.md).

**Round 1 (f = 0.125, one adversary) is complete and failed the gate.** Two adversaries
trained to 194M steps; 0 of 14 checkpoints pass. Both converged to a saturated bang-bang
controller that crashes its own drone — honest-drone damage never exceeds +18 %. Details in
tier3_design.md §3. **Round 2 (f = 0.25) is what to run now — jump to Step 3.**

### ⚠️ The gate metric changed — re-read your old numbers

The efficacy gate used to key on `goal_rmse_degradation_pct`, goal RMSE averaged over **all
eight** drones. That is gameable: an adversary maxes it by flying *itself* away, damaging
nobody. `results/tier3_gate.md` now reports three columns and gates on the first:

| column | meaning |
| --- | --- |
| **honest deg %** | goal-RMSE degradation of the **6 honest drones**. **This is the gate.** ≥ 30 % to pass. |
| own deg % | the adversary's own goal-RMSE degradation. Large + small honest = self-destruction, not an attack. |
| all-drone deg % | the old number. Still printed, because the matched-damage triple against Tiers 1/2 is defined on it. Not gated. |

Round 1's headline "109 % degradation" was **+384 % own, +17.7 % honest**. If you quoted the
old all-drone figures anywhere, they are inflated by roughly 4–6×.

Re-running the gate on existing sidecars is free (reads JSON, no simulation):

```bash
python -m byzantine_experiments.analysis.gate_tier3 --horizon
```

### Step 0 — sanity-check the reward balance (~20 s, do this first)

```bash
python -m byzantine_experiments.adversary.calibrate_reward
```

Expect **`crash : team = 0.4 : 1`** and **`VERDICT: balanced enough to train`**, with the
`weights:` line showing `w_team: 20.0`. If it shows `w_team: 1.0` you are on an old copy of the
file — the calibrator now reads `reward.DEFAULTS`, the same source training uses.

This step exists because the spec's literal weights drown the objective **2798 : 1**; an
adversary trained on them learns "hover, never crash" and nothing else. Re-run it after
changing any weight.

### Step 1 — train at f = 0.125 (round 1 — **done for seed 1, do not re-run**)

```bash
# ~12-24 h each on GPU. Run in parallel on the cluster if you can.
bash scripts/byz_train_adv.sh 1 0     # DONE -> train_dir/adv_seed1_stealth0
bash scripts/byz_train_adv.sh 1 5     # DONE -> train_dir/adv_seed1_stealth5
bash scripts/byz_train_adv.sh 2 0     # not run, and not worth running: see below
bash scripts/byz_train_adv.sh 2 5
bash scripts/byz_train_adv.sh 3 0
bash scripts/byz_train_adv.sh 3 5
```

**Do not spend GPU-days on seeds 2 and 3 at f = 0.125.** Seed 1 already produced two
independent adversaries that both converged to the same degenerate self-destroying policy at
every one of 7 checkpoints. More seeds at the same capacity would buy error bars on a
negative result we can already state. Go to Step 3 instead; come back for seeds 2–3 only if
f = 0.25 produces something that passes.

Each launch runs the **frozen-weights pre-flight test first** and aborts if it fails — the
spec asks for test 1 on every training launch, and silent self-play would invalidate every
tier measured against the baseline.

Knobs: `ENV_STEPS` (default 2e8), `NUM_WORKERS` (32), `DEVICE` (gpu), `WANDB` (1),
`EPSILON` (0.3), `W_TEAM`/`W_CRASH`/`W_ALIVE`, `DRY_RUN=1` to print the command.

### What to watch, and when to kill a run

All in wandb, project `byzantine_exp`, group `tier3_adversarial`. The reward components are
logged per episode as `adv/r_team`, `adv/r_crash`, `adv/r_alive`, `adv/r_stealth`,
`adv/r_total`, plus `adv/crash_onsets`, `adv/survived` and `adv/action_deviation`.

**Sign convention:** `adv/r_team = -w_team x (honest-drone mean reward)`, so **higher is worse
for the team and better for the adversary**. It starts around **+230/episode** with a random
policy and should *rise*. `adv/r_crash` is negative and should rise *toward 0* as crashes stop.

**First note the values at ~1M steps** — that is your baseline for "flat".

| checkpoint | healthy | kill the run if |
| --- | --- | --- |
| **~20M steps** (10%) | `adv/crash_onsets` falling below ~5 (from ~10–20); `adv/survived` climbing off 0 | `adv/crash_onsets` still ≥10 **and** `adv/survived` ≈ 0 → it is not learning *anything*. Spec failure mode 1: check reward wiring, run `pytest -k "reward or frozen"`. |
| **~50M steps** (25%) | `adv/survived` ≥ 0.7; `adv/r_team` measurably above its 1M-step value; `adv/r_total` rising | `adv/survived` ≥ 0.7 **but** `adv/r_team` flat or falling → it learned to fly safely and stopped attacking. Raise `W_TEAM` (try 50) and restart. |
| **any time** | `\|adv/r_crash\| / adv/r_team` falling from ~0.4 | that ratio climbing past ~2 and staying → crashes dominating again; recalibrate. |
| **any time, stealth runs** | `\|adv/r_stealth\| / \|adv/r_team\|` under ~10 | that ratio past ~50 → the stealth penalty has drowned the objective. Both stealth runs so far hit **555 : 1** and are uninformative. The static calibrator does *not* catch this: it reports 5.3 : 1 under a random policy, because a trained policy saturates and a random one does not. Watch it online. |

The two failure signatures are different and the fix differs, so distinguish them: **nothing
moves at all** = wiring/plumbing; **survival improves but damage does not** = weighting.

Do not wait out a bad run to 200M steps. 50M is ~25% of the budget and enough to tell.

### Step 2 — the full frozen-weights check (optional, ~2 min)

```bash
BYZ_SLOW_TESTS=1 pytest byzantine_experiments/tests/test_tier3.py -q -k baseline_checkpoint
```

Runs a real 3000-step training loop and asserts the baseline's on-disk `.pth` file hashes are
unchanged — the spec's literal "assert checkpoint hash unchanged after 1000 training steps".

---

### Step 3 — round 2: f = 0.25, two colluding adversaries (**done — failed**)

Round 1's diagnosis is that one drone out of eight has almost no physical authority over the
other seven. So give the attacker more: two compromised drones, f = 0.25, still under the
classical f < 1/3 bound. Sample Factory trains **one shared policy driving both drones**, so
they are identical-policy colluders sharing a team-level reward — the same arrangement as
Tier 2D's `collude`.

```bash
# Same ~12-24 h each. Two runs, not six.  BOTH ARE DONE — do not re-run.
ADV_INDICES=0,1 W_ALIVE=0.05 bash scripts/byz_train_adv.sh 1 0   # -> train_dir/adv_seed1_stealth0_f250
ADV_INDICES=0,1 W_ALIVE=0.05 bash scripts/byz_train_adv.sh 1 5   # -> train_dir/adv_seed1_stealth5_f250
```

**Outcome: failed.** `adv/r_team` flat over 200M steps (−24.0 → −23.1); `crash_onsets` 2.10,
`survived` 0.000. The stealth5 run is additionally invalid — its stealth penalty outweighed
the objective 555 : 1 (`design/tier3_design.md` §4.2). Kept here as the record of what was run.

Both flags matter:

* `ADV_INDICES=0,1` is what makes it f = 0.25. The run name picks up `_f250` automatically so
  round-2 checkpoints and sidecars can never collide with round 1's, and the analysis
  recovers f from the filename.
* `W_ALIVE=0.05` (up from 0.01). Round 1 proved 0.01 does not price survival *at all* — the
  policy learned that one crash buys unlimited pre-crash flailing. At 0.05 a policy that
  survives the full 15 s banks ~75 instead of ~15, while still sitting well under `w_team=20`.

Everything else is held at round 1's values so the comparison is clean. Re-run Step 0's
calibrator first if you change any weight.

**The one thing to watch that is new in round 2:** `adv/crash_onsets` can now reach 2 per
episode instead of 1, because the onset penalty fires per adversary while the team term is
unchanged. If `adv/crash_onsets` sits near 2.0 past ~50M steps with `adv/survived` near 0,
round 2 is reproducing round 1's failure with twice the penalty — kill it and tell me, that
is the signal to move to the objective change (tier3_design.md §5) rather than tune further.
The Step 1 kill-criteria table above otherwise applies unchanged.

### Step 4 — evaluate against the gate (done for rounds 1 and 2)

```bash
HORIZON=1 ADV_RUNS="adv_seed1_stealth0_f250 adv_seed1_stealth5_f250" \
    bash scripts/byz_eval_adv.sh
python -m byzantine_experiments.analysis.gate_tier3 --horizon
```

Result: **0 / 28 pass**. See §"Current state". Re-running the analysis is free (~40 s, reads
the trajectories on disk); re-running the rollouts is not needed.

#### What a pass looks like, for round 3

Read `results/tier3_gate.md`. **One column decides it: `honest%`** — and now it comes with a
95 % CI and a p-value.

| you need | value | rounds 1–2 got |
| --- | --- | --- |
| `honest%` point estimate | **≥ 30.0** | max **+17.2** |
| `95% CI` lower bound | **> 0** | never; 4 runs significant, all negative |
| `surv` | **≥ 70 %** | 0–30 % |
| `ratio` (stealth runs only) | **≤ 1.5** | 6.0–8.4 |

All must hold at the *same* checkpoint; the `gate` column does that conjunction for you.

**A point estimate alone is not evidence.** Per-episode goal RMSE has a 42 % coefficient of
variation, so at n = 20 the CI is ±22 points — most of the distance to the gate. Round 1's
headline "17.7 %" turned out to be p = 0.27. Evaluate round 3 at **n = 50**, not 20: n ≈ 31
per arm is the threshold for 80 % power against a true 30 % effect.

**Do not read `own%` or `all%` as success.** `own%` is the adversary's damage to *itself*.
Round 1's most impressive row was `all% = 109`, which decomposed into `own% = +384`,
`honest% = +17.7`.

### Step 5 — round 3: the observation channel (**done — seed 1**)

Built and tested; nothing trained. The evidence is in `results/channel_ceiling.md`: lying
about your state is the **only** channel where anything clears the gate, and a hand-designed
lie already does (**+30.4 %, p = 0.0007**). Every Tier 3 adversary so far was on the
motors-only channel, where 79 runs produced 2 significant results between them.

The adversary's action grows from 4 motors to **4 + 6**, the extra six being the state it
broadcasts (`reported = true + max_lie * a`, so a zero action is a *truthful* broadcast).

```bash
# ~12-24 h each on GPU. Two runs.
LIES=1 bash scripts/byz_train_adv.sh 1 0              # -> train_dir/adv_seed1_stealth0_lie
LIES=1 LIE_ONLY=1 bash scripts/byz_train_adv.sh 1 0   # -> train_dir/adv_seed1_stealth0_lieonly
```

| mode | motors | broadcast | what it isolates |
| --- | --- | --- | --- |
| `LIES=1` | learned | learned | the strongest attacker available |
| `LIES=1 LIE_ONLY=1` | **frozen honest policy** | learned | the channel alone — the direct analogue of Tier 2C, whose action stream is clean by construction |

Run **both**: `lie_only` is the clean comparison against Tier 2's `liar`, `lie+fly` is the
stronger attacker. `LIE_ONLY=1` implies `LIES=1`.

Round-3 runs automatically get `exploration_loss_coeff=0.003` (rounds 1–2 keep 0.0, so
re-running them still reproduces bit-for-bit). Rounds 1–2 are otherwise untouched: without
`LIES=1` the action space is still 4-D and no broadcast machinery runs at all.

#### What to watch, on top of the Step 1 table

| metric | healthy | kill the run if |
| --- | --- | --- |
| **`adv/lie_magnitude_m`** | stays well above 0 | drifting toward 0 → **the policy has abandoned the channel.** This is the round-3 analogue of round 2's flat `adv/r_team`. Stop and tell me. |
| **`adv/broadcast_rewrites`** | large and roughly steady | ≈ 0 while `lie_magnitude_m` is large → the lie is landing in observation slots nobody reads. That is a bug, not a result — stop immediately. |
| `adv/r_team` | **rising** off its ~1M-step value | flat through 50M → same failure as rounds 1–2, on a channel that should have headroom. |

`adv/r_team` is the one that matters. Rounds 1–2 both failed by leaving it flat for 200M
steps; if it moves this time, the channel hypothesis is confirmed.

### Step 6 — evaluate round 3 (**done**)

```bash
HORIZON=1 bash scripts/byz_eval_adv.sh
python -m byzantine_experiments.analysis.gate_tier3 --horizon
```

**Result: 0 / 42 pass, all four rounds.** `_lieonly` meets survival (96–100 %) and action ratio
(0 % saturation) at every checkpoint — the first Tier 3 adversary to meet any criterion — and
fails only on damage (max +11.5 %, p = 0.21). See §"Current state" for the seed confound this
exposed.

### Step 7 — round 4: `_lieonly` against baseline seed 3 (**done — objective flat**)

One run. Same mechanism as round 3, different baseline:

```bash
# ~12-24 h.  -> train_dir/adv_seed3_stealth0_lieonly
LIES=1 LIE_ONLY=1 bash scripts/byz_train_adv.sh 3 0
```

Then:

```bash
HORIZON=1 ADV_RUNS="adv_seed3_stealth0_lieonly" bash scripts/byz_eval_adv.sh
python -m byzantine_experiments.analysis.gate_tier3 --horizon
```

The `3` selects `train_dir/exp1_baseline_seed3` as the frozen honest team, and
`byz_eval_adv.sh` evaluates against that same seed automatically.

**Why this run and not another.** Seed 3's observation channel is the only place in the whole
corpus with a real, significant, gate-clearing hand-designed attack (+30.4 %, p = 0.0007, 5 of
6 runs significant). Every previous Tier 3 round optimised somewhere with nothing to find —
rounds 1–2 on a channel where hand-designed attacks do nothing, round 3 on a *baseline* where
they do nothing. This is the first setup where "trained beats hand-designed" is a falsifiable
claim rather than a vacuous one.

`_lieonly` has already shown it can survive 100 % of episodes while lying at full deflection
(17.32 m = √3 × 10). It had no purchase on seed 1. Seed 3 is where the purchase is.

**What to watch:** same as Step 5's table. `adv/lie_magnitude_m` must stay well above 0, and
`adv/r_team` is the one that decides it — but note `len/len` too: on seed 1 it fell 31 % while
`r_team` stayed flat, and since `r_team` is an episode *sum*, a shortening episode masks
per-step progress. If `len/len` falls again, the rollout is the arbiter, not the curve.

**Pass criteria** unchanged (Step 4): `honest%` ≥ 30 with its CI excluding zero, `surv ≥ 70 %`.

**Outcome:** trained fine, objective flat (§"Current state"). Survival 0.994 throughout, lie
magnitude 16.7 m against a 17.32 m ceiling, 9.0M broadcast rewrites — the channel is used to
its limit and `adv/r_team` still does not move.

### Step 8 — evaluate round 4 (**done**)

```bash
HORIZON=1 ADV_RUNS="adv_seed3_stealth0_lieonly" bash scripts/byz_eval_adv.sh
python -m byzantine_experiments.analysis.gate_tier3 --horizon
```

**Result (exploratory sweep, n=50):** best checkpoint m075 = **+29.7 %** honest degradation
[8.0, 51.4], p = 0.008, survival 98 %. All but at the gate, and the first Tier 3 checkpoint to
put *significant* damage on the honest team. But this is the max of 7 checkpoints tested at
p<0.05, so it needs a confirmatory re-measurement before it counts (step 11).

### Step 9 — round 5: give the adversary its own absolute position (**done — did not help**)

```bash
# -> train_dir/adv_seed3_stealth0_lieonlyego
LIES=1 LIE_ONLY=1 bash scripts/byz_train_adv.sh 3 0
```

`EGO=1` is the default whenever `LIES=1`, so that command alone gives the 60-D observation.
Set `EGO=0` to reproduce round 4 exactly.

**Result: the objective is statistically identical to round 4.** `adv/r_team` slope
−0.194 ± 0.090 per 100M steps — flat (implied change 0.39 < the curve's own 1.04 wobble), peak
bin −6.7 @ 50M then sags, per-step mean −0.00807 vs round 4's −0.00834. Giving the adversary
the absolute position that `mirror` needs **did not move the objective**. The information gap
(§6.10) was real but closing it was not sufficient — the policy still cannot discover the
coherent state-dependent lie a two-line closed form expresses (`design/tier3_design.md` §6.12).

**Why.** `mirror` — the only hand-designed attack that clears the gate — needs absolute
position, and the adversary's observation does not contain it. We have been asking RL to beat
a baseline holding strictly more information. Round 5 appends the adversary's own absolute
pos/vel (54 → 60), which is exactly what `agents/base.py::EgoState` already gives Tier 2's
goal-directed agents, and carries nothing about any other drone. **Parity, not privilege.**

Rounds 1–4 are unaffected: `--adv_ego_state_obs` defaults False, so their `config.json` files
(which predate the key) still build a 54-D env and load unchanged. Round-5 runs are named
`_lieonlyego` so nothing collides.

**What to watch:** same as Step 5, plus the one number that decides it —

| metric | round 4 got | round 5 needs |
| --- | --- | --- |
| `adv/r_team` (seed 3) | −7.6 … −9.9, slope flat | **toward +16.6**, the `mirror` lie's value |
| `adv/lie_magnitude_m` | 16.7 (at ceiling) | stays high |
| `adv/survived` | 0.994 | stays ≥ 0.70 |

Read the curves with the new tool, not by eye — single samples of `adv/r_team` span +5 to −25
and one of them misled me into reporting a positive objective that was not there:

```bash
python -m byzantine_experiments.analysis.training_curve adv_seed3_stealth0_lieonlyego
```

It reports binned means, a slope with a standard error, and the per-step figure alongside the
episode sum (they diverge whenever episode length moves, which it has in every round-3+ run).

### Step 10 — evaluate round 5 (**done — worse than round 4**)

```bash
HORIZON=1 ADV_RUNS="adv_seed3_stealth0_lieonlyego" bash scripts/byz_eval_adv.sh
python -m byzantine_experiments.analysis.gate_tier3 --horizon
```

**You never need the tmux scrollback for this** — `gate_tier3` writes the full table to
`results/tier3_gate.md` and `results/tier3_gate.csv` every run. Re-running the analysis alone
is free (~40 s, reads the trajectories already on disk).

**Result:** round 5's max is **+15.9 % (p = 0.055), 0 of 7 significant**, against round 4's
+29.7 % and 3 of 7. The +29.7 % you may have spotted in the table is round 4's m075, not a
round-5 row. Full comparison in `design/tier3_design.md` §6.13.

### Step 11 — **confirmatory number for the paper** (done — superseded by the v2 re-measurement, Step 15)

The horizon sweeps are exploratory — 7 checkpoints against a p<0.05 bar, best reported. A
headline figure has to be re-measured on a fresh, larger sample, or it is inflated by
multiplicity and the winner's curse.

**First, for free, see how much of the +29.7 % is likely real:**

```bash
python -m byzantine_experiments.analysis.confirm_tier3 --run adv_seed3_stealth0_lieonly --pooled
python -m byzantine_experiments.analysis.confirm_tier3 --run adv_seed3_stealth0_lieonlyego --pooled
```

| | round 4 (54-D) | round 5 (60-D +ego) |
| --- | --- | --- |
| **pooled, no selection** (n=350 vs clean 50) | **+12.6 % [+0.9, +24.4]** p=0.035 | **+11.0 % [−0.4, +22.5]** p=0.059 |
| real between-checkpoint sd (τ) | 7.7 pp | **0.0 pp** |
| best checkpoint | m075 = +29.7 % | m050 = +15.9 % |
| **shrunk estimate of the best** | **+21.5 %** | **+11.0 %** |

Round 4's checkpoints genuinely differ (τ = 7.7 pp > the 7.4 pp measurement SE), so m075 is
probably really its best — but only ~half its excess over the pooled mean is signal. **Expect a
confirmatory sample near +21.5 %, i.e. below the gate.** Round 5's τ = 0.0, so its max is pure
noise and there is nothing there to confirm.

**The single highest-value rollout is the clean reference at n = 200.** The clean arm is shared
by every seed-3 comparison and at n = 50 it dominates their error: `sqrt(1/350 + 1/50) = 0.151`
vs `sqrt(1/350 + 1/200) = 0.084`. Collecting it once tightens the pooled CIs from ±11.7 pp to
about ±6 pp **with no new attacked episodes**, and `best_clean_reference()` picks it up
automatically from then on.

Then confirm round 4's m075 (round 5 has nothing worth confirming). **One command does
everything** — it collects both arms concurrently, logs each to a file so nothing is lost in
tmux, and prints the verdict when they finish:

```bash
bash scripts/byz_confirm_adv.sh adv_seed3_stealth0_lieonly 75
```

~70 min (the two arms run in parallel, not back to back). Logs land in `logs/tier3_confirm/`;
both rollouts are logged to wandb under group `tier3_confirm`. Safe to re-run — completed arms
are skipped, so if one fails you only repeat that one. `N_EPISODES=200` is the default;
`DRY_RUN=1` prints the underlying commands without running them.

When it finishes it prints the verdict itself. To re-print it later:

```bash
python -m byzantine_experiments.analysis.confirm_tier3 --run adv_seed3_stealth0_lieonly --milestone 75
```

**If the confirmatory number holds ≥ 30 % with the CI clear of zero**, Tier 3 has a genuine
positive result: a trained adversary reaching the gate on the channel and baseline where
hand-designed attacks work. The deferred work (transfer matrix, matched-damage triple, stealth
analysis) then becomes worth running. **If it regresses toward ~+22 %** — the other significant
checkpoints' level — the clean statement is that this swarm resists a single Byzantine member
across both channels, three baselines, and ~800M steps of optimisation, and that a
hand-designed structural lie (`mirror`, +30.4 %) beats every learned adversary we could train.
Either way the confirmatory number is what the paper quotes, not the sweep max.

### Step 12 — **the decisive experiment: replicate one condition** (done)

Everything else is on hold behind this. `τ` rests on three clean rollouts; before discarding
or re-analysing the corpus we need one condition measured properly, with **run identity
respected**.

```bash
source scripts/byz_wandb_env.sh      # keeps wandb off the AFS home volume (see below)
bash scripts/byz_replicate.sh
```

~25 min (6 rollouts, 6-way parallel). It runs **3 clean + 3 attacked replicates at n = 50**,
each with a different env seed, on the corpus's **strongest** attack — Tier 2's `mirror` liar
on baseline seed 3, nominally +30.4 %, p = 0.0007. Then it reports both readings:

* **naive** — all episodes pooled, run identity ignored. This is what every previous number in
  this project did.
* **run-level** — each run's mean is one observation, Welch on 3 vs 3. No variance
  decomposition, no assumption that seed 3's τ transfers anywhere.

Replicates buy more than episodes here: 6 runs × 50 episodes pins τ far better than 2 × 150
for the same compute.

**How to read the outcome:**

| result | meaning |
| --- | --- |
| run-level test **significant** | τ is over-estimated; the corpus needs re-analysing with run-level CIs, not discarding. Re-run the other conditions with replicates. |
| run-level test **null**, naive significant | confirmed: the corpus's findings are run-to-run noise. The tier-wide negative is real and reportable — with the honest caveat that 3 vs 3 is low-powered, so this is "not demonstrated", not "demonstrated absent". |

If it comes back null and you want to push power, `REPS=6` doubles it for ~50 min.

### Step 13 — **re-measure with properly paired seeds** (done)

The first attempt at this step (2026-09-21) used a collector that paired only episodes 0–2;
its output (`+30.4 %`, correlation between arms −0.08, shrink 1.0×) is **not** a paired result
and should not be quoted. Re-run the same two rollouts with the per-episode seeding fix:

```bash
source scripts/byz_wandb_env.sh      # now also redirects wandb credentials off AFS
D=byzantine_experiments/results/tier3_paired; mkdir -p $D
PY=/scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python

$PY -m byzantine_experiments.rollout.collect \
  --checkpoint train_dir/exp1_baseline_seed3 --no-byzantine \
  --n-episodes 50 --seed 4242 --device cpu --out $D/paired_clean_s3.npz \
  --wandb --wandb-group tier3_paired --wandb-name paired_clean_s3_v2 &

$PY -m byzantine_experiments.rollout.collect \
  --checkpoint train_dir/exp1_baseline_seed3 \
  --tier tier2_liar --tier-config '{"lie_mode":"mirror","magnitude":3.0}' \
  --compromised-indices 0 --n-episodes 50 --seed 4242 --device cpu \
  --out $D/paired_mirror_s3.npz \
  --wandb --wandb-group tier3_paired --wandb-name paired_mirror_s3_v2 &
wait

$PY -m byzantine_experiments.analysis.replicate_test \
  --paired $D/paired_clean_s3.npz $D/paired_mirror_s3.npz
```

~14 min (both run concurrently). The `--out` paths overwrite the invalid first attempt.

**Sanity checks before reading the number** — the output must show a clearly *positive*
correlation between arms (the 10-episode stress test gave +0.81) and a standard-error shrink
well above 1×. If the correlation is near zero again, pairing has broken and the estimate
means nothing; stop and tell me.

Expected precision at ρ ≈ 0.8, n = 50: roughly **± 8 pp**, against ± 19 pp unpaired. That is
tight enough to settle the `mirror` question in one run.

### Step 14 — **re-collect the corpus with the fixed collector (v2)** (done)

Step 13 gave the first valid paired measurement: Tier 2 `mirror` liar, seed 3 —
**+9.7 % [+2.4, +17.0], p = 0.007**, correlation between arms +0.79, pairing shrank the SE 2.2×.
The same attack was reported as +30.4 % in the legacy corpus. Everything else needs the same
treatment.

**What changed so this is safe to run** (none of it needs action from you):

* All sweeps and analyses now read and write **`byzantine_experiments/results_v2/`** (logs in
  `logs_v2/`). The legacy folders are untouched and kept as the "before". Without this the
  sweeps' skip-if-exists logic would have silently reused every invalid legacy file.
* Every sidecar the collector writes is tagged `seeding: per_episode_v2`.
* All analyses compare **paired** (`analysis/paired.py`) and *verify* the pairing first —
  same tag, same `--seed`, same baseline, identical spawns on every episode. A run that fails
  any check is shown as **UNPAIRED — not reportable** and can never pass the gate. There is no
  silent fallback.
* The gate used to count a row with a missing trajectory as "significant". Fixed.

Run in this order. Each stage's output is needed by the next.

**Stage A — clean references (~20 min, 3 runs).** Everything pairs against these.

```bash
source scripts/byz_wandb_env.sh
ONLY=clean bash scripts/byz_rollout_tier1.sh
```

**Stage B — the observation-channel block, as an end-to-end check (~45 min, 18 runs).** This is
the `liar` family (offset / stale / mirror × 3 seeds × 2 values of f). It re-measures the
headline attack on every seed, and it is the first real test that pairing survives the actual
sweep scripts.

```bash
PARALLEL=12 ONLY=2c bash scripts/byz_rollout_tier2.sh
python -m byzantine_experiments.analysis.aggregate_tier2
```

**Check before going further:** in `results_v2/tier2_calibration.csv` every `t2_liar_*` row
must have `paired = True` and a positive `paired_corr` (expect roughly +0.6 to +0.9). If any row
says `paired = False`, read its `pairing_error` column and send it to me before running more.

**Stage A and B: done (2026-09-24).** All 18 liar runs paired (correlation +0.73 to +0.88).
Family result: positive in 16/18, mean **+5.7 %**, strongest on seed 3 (mean +9.6 %, best
+13.1 %), 3 of 18 significant.

**Stage C — two commands, with a pause between them.**

```bash
bash scripts/byz_stage_c.sh 1      # ~5 h, 126 runs: hand-designed Tier 1 + rest of Tier 2
# read the GATE CALIBRATION CHECK printed at the end, then:
bash scripts/byz_stage_c.sh 2      # ~2 h, 42 runs: every trained Tier 3 adversary
```

| phase | contents | runs |
| --- | --- | --- |
| 1 | Tier 2 signflip / waypoint / collusion (`ONLY=2a,2b,2d`) | 39 |
| 1 | Tier 1 main matrix — gaussian, OU, dropout (clean refs reused) | 60 |
| 1 | Tier 1 stall grid (1C′, which superseded dropout) | 27 |
| 2 | Tier 3 horizon sweep: 6 adversaries × 7 milestones | 42 |

**Deliberately skipped** — run later only if needed:

* Tier 1 `REFINE` grid (33) and Tier 2 `ALPHA_REFINE` grid (12): both exist to find
  severities inside the old 30 % band, which is no longer the target.
* Tier 3 `stealth5` adversaries (14): their training was dominated 555:1 by the stealth penalty
  (design §4.2); they are not meaningful adversaries.

**Why the pause.** Phase 1 ends by re-checking the 10 % gate against the *complete*
hand-designed results (`analysis/gate_calibration.py`). That has to happen before any v2 Tier 3
number exists, so the gate cannot drift toward the Tier 3 result. If the report says the gate
still fits, just run phase 2. If it says the hand-designed ceiling is far above 10 %, stop and
send it to me first.

Everything is logged to `logs_v2/stage_c_phase{1,2}.log`, so nothing is lost in tmux, and both
phases are safe to re-run (collected configs are skipped).

### Step 15 — the gate decision (**done, 2026-09-25: fail at +7.4 %**)

`adv_seed3_stealth0_lieonly` @ 194M, 200 paired episodes: **+7.4 % [+3.1, +11.7], p = 0.0007**,
survival 98 %. Real damage, below the 10 % gate. Details: `design/tier3_design.md` §6.23.

### Step 16 — trained vs hand-designed, head to head (**done, 2026-09-25: training matches hand design, does not beat it**)

```bash
bash scripts/byz_confirm_hand.sh
```

Runs all three hand-designed liar attacks (mirror, offset, stale) at n = 200 on the **same 200
scenarios** as the trained adversary, reusing the existing clean run, then prints each one's
degradation and a paired trained-minus-hand-designed difference per mode. All three rather than
the best, so the comparison is not tilted toward the hand-designed side.

### Step 15 procedure (for reference)

**Chosen after phase 2 (2026-09-25):** the final checkpoint of the best adversary.

```bash
bash scripts/byz_confirm_adv.sh adv_seed3_stealth0_lieonly 194
```

Why m194 and not m100 (the highest screening number, +14.2 %): the seven checkpoints of this
adversary do not differ beyond noise — they all estimate one effect of about +10.2 %. Picking the
maximum would add selection bias and no information; "the fully trained adversary" is a cleaner
claim. Expect roughly +10 % ± 4.6 pp, i.e. pass or fail is close to a coin flip, and the estimate
with its interval is what goes in the paper. Details: `design/tier3_design.md` §6.22.

General procedure, for any later adversary:

Phase 2's gate table marks any checkpoint that clears 10 % as **CANDIDATE**, not PASS. Those
rows come from n = 50 screening runs, which detect a true 10 % effect only 57 % of the time and
overstate the best of several checkpoints. The decision comes from re-measuring the best
candidate on a fresh paired sample of 200 episodes (99 % power at 10 %):

```bash
bash scripts/byz_confirm_adv.sh <adversary_run> <milestone>   # e.g. adv_seed3_stealth0_lieonly 75
```

~70 min. It prints the paired verdict against the 10 % gate. If phase 2 produces no CANDIDATE,
confirm the checkpoint with the highest point estimate anyway — the paper needs one unbiased
number for Tier 3 either way.

### Efficacy gate: recalibrated from 30 % to 10 % (2026-09-24)

Decided before any v2 Tier 3 evaluation, from Tier 1/2 evidence only:

* the legacy 30 % was set against numbers inflated ~3× by the seeding bug (`mirror`: +30.4 %
  reported, +9.7 % properly paired);
* on v2 data the best hand-designed attack family reaches about 10 % on its strongest seed, so
  10 % means "at least as damaging as the best hand-designed attack" — what 30 % was meant to
  express;
* **a pass requires all of:** paired point estimate ≥ 10 %, paired 95 % CI above 0, measured on
  ≥ 200 episodes, plus the unchanged survival ≥ 70 % and (stealth runs) action ratio ≤ 1.5.

One property of any threshold worth knowing: an adversary whose *true* effect is exactly 10 %
passes about half the time; one at 13 % passes about 90 % of the time at n = 200.

### Step 17 — close the spec gaps (decisions D1–D5, 2026-09-26) ← current step

Everything the audit (`byzantine_experiments/design/tier3_audit.md`) found missing is now built;
what remains is compute. Decisions: **D1** Tier 1/2 validated at the family level, matched-triple
target = Tier 3's confirmed effect, new TOST margin declared before any triple data; **D2** gate
stays at 10 %, both facts reported; **D3** train a plausible-lie (stealth) round; **D4** train a
seed-2 adversary; **D5** distribution shift built. Why and how: `design/tier3_design.md`
§6.25–6.30; the spec changes are dated amendments A1–A6 inside the specs.

Run the blocks in order. **A** needs no training and can start now. **B** is GPU training, one
run at a time (runs so far took 32–57 h each). **C** follows each finished B run. A and C are
CPU rollouts; they can run while B trains, but both slow down — if you prefer, fit them between
training runs. Every script skips runs whose output exists, so all are safe to re-run.

```bash
source scripts/byz_wandb_env.sh     # once per shell, for every command below
```

**A — now, no training needed (~3 h of rollouts in total)**

```bash
# A1. Tier 1/2 tables in the D1 format (family validation replaces the 20-40 % band). ~5 min each.
python -m byzantine_experiments.analysis.aggregate_tier1
python -m byzantine_experiments.analysis.aggregate_tier2

# A2. Cross-seed transfer for the three airborne adversaries (+ clean n=200 refs on seeds 1 and 2).
#     10 runs x 200 episodes, ~100 min. Ends by printing the transfer matrix.
bash scripts/byz_transfer.sh

# A3. Shift calibration (D5): picks "mild" and "severe" from their effect on the clean swarm.
#     21 runs x 50 episodes, ~55 min. Writes results_v2/shift_levels.json.
bash scripts/byz_calibrate_shift.sh

# A4. Spec items 2, 6, 7, 8 for the existing adversaries. rewards/compute read TensorBoard
#     (~40 s per adversary the first time, cached after); stealth/behaviour ~30 s each.
python -m byzantine_experiments.analysis.tier3_report rewards
python -m byzantine_experiments.analysis.tier3_report compute
python -m byzantine_experiments.analysis.tier3_report stealth   --run adv_seed3_stealth0_lieonly
python -m byzantine_experiments.analysis.tier3_report behaviour --run adv_seed3_stealth0_lieonly

# A5. The one test never run: baseline checkpoint hash unchanged by adversarial training (a few min).
BYZ_SLOW_TESTS=1 /scratch/bmugizi/conda_envs/swarm_r311_v2/bin/python -m pytest \
    byzantine_experiments/tests/test_tier3.py -q -k baseline_checkpoint

# A6. The efficacy table (all adversaries x five criteria), now with transfer filled in.
python -m byzantine_experiments.analysis.efficacy_table
```

**B — training, in this order (GPU, one at a time; 200 M steps each, as every earlier adversary)**

```bash
# B1. D3: the plausible (stealth) liar on seed 3, where the free liar works. Decides the framing.
LIES=1 LIE_ONLY=1 EGO=0 LIE_MODE=plausible bash scripts/byz_train_adv.sh 3 0   # -> adv_seed3_stealth0_lieonlyplaus
# B2. D4: the free liar on seed 2, completing the free 3 x 3 transfer matrix.
LIES=1 LIE_ONLY=1 EGO=0 bash scripts/byz_train_adv.sh 2 0                      # -> adv_seed2_stealth0_lieonly
# B3, B4. The plausible liars on seeds 1 and 2 (the matched triple needs all three seeds).
LIES=1 LIE_ONLY=1 EGO=0 LIE_MODE=plausible bash scripts/byz_train_adv.sh 1 0   # -> adv_seed1_stealth0_lieonlyplaus
LIES=1 LIE_ONLY=1 EGO=0 LIE_MODE=plausible bash scripts/byz_train_adv.sh 2 0   # -> adv_seed2_stealth0_lieonlyplaus
```

`EGO=0` must be given explicitly: with `LIES=1` the script otherwise defaults to the round-5
ego-state observation, which did not help. If time forces it, `ENV_STEPS=100000000` halves each
run; the downstream rules read the configured length, so nothing else changes, but seed 2 would
then differ in horizon from seeds 1 and 3 (design §6.26).

While a run trains (trends, not samples — the wall-limited fraction should stay small; it is
the only place the plausible lie's acceleration bound can be exceeded):

```bash
python -m byzantine_experiments.analysis.training_curve adv_seed3_stealth0_lieonlyplaus
python -m byzantine_experiments.analysis.training_curve adv_seed3_stealth0_lieonlyplaus \
    --key adv/lie_magnitude_m --key adv/lie_wall_limited_frac
```

**C — for EVERY B adversary (all four; D cannot start until each has its C2)**

Status 2026-10-09: done for `adv_seed3_stealth0_lieonlyplaus`; still needed for
`adv_seed1_stealth0_lieonlyplaus`, `adv_seed2_stealth0_lieonlyplaus`, `adv_seed2_stealth0_lieonly`.
List all pending adversaries in one `RUNS`: the steps then run them together (transfer cells 9 at
a time instead of 3), ~2.5 h for three instead of ~5 h. `bash run_block_C.sh` does exactly this,
with a log.

```bash
RUNS="adv_seed1_stealth0_lieonlyplaus adv_seed2_stealth0_lieonlyplaus adv_seed2_stealth0_lieonly"
# C1. Horizon sweep, 7 checkpoints x 50 episodes per adversary, ~1 h for three.
HORIZON=1 ADV_RUNS="$RUNS" bash scripts/byz_eval_adv.sh
python -m byzantine_experiments.analysis.gate_tier3 --horizon
# C2. Own-seed confirmatory + transfer, final checkpoint, n = 200 per cell, ~1.5 h for three.
ADV_RUNS="$RUNS" bash scripts/byz_transfer.sh
# C3. Stealth and behaviour reports (these liars' broadcasts are recorded), ~10 s each.
for RUN in $RUNS; do
    python -m byzantine_experiments.analysis.tier3_report stealth   --run $RUN
    python -m byzantine_experiments.analysis.tier3_report behaviour --run $RUN
done
```

**D — the matched triple (after all four B runs and their C2)**

```bash
bash scripts/byz_matched_triple.sh    # select (pre-registered rule) -> collect -> TOST
```

The selection is written before any triple episode is collected; n per condition per seed is
400 / 800 / 1600 by the power rule (~4 h / ~8 h / ~16 h). If even 1600 is underpowered the
script stops and says so — that is then the result to report, not something to work around.

**D result (2026-10-09):** triple **NOT MATCHED** under the pre-registered TOST (n = 800 per seed,
margin ±2.08 pp). Tier 3 is equivalent to Tier 1 and to Tier 2; Tier 1 vs Tier 2 is not shown
equivalent (−1.11 pp, 90 % CI [−2.15, −0.07]). This is the result; do not add episodes to change
it (design §6.32). Phase 3 uses the selected members regardless.

**E — Phase 3 rollouts (after D)** — `bash run_block_E.sh` runs all of it with a log, ~4 h.

Shift levels were amended on 2026-10-12, before any attacked run under shift (design §6.32): the
original picks are kept as `mild_v1` (+32.6 %) / `severe_v1` (+140 %); the same rules on an
extended grid give the primary `mild` / `severe`. All four levels are run and all are reported.

```bash
# E0. keep the original picks as mild_v1 / severe_v1 (instant; refuses if run out of order)
python -m byzantine_experiments.analysis.shift_calibration --freeze-v1
# E1. extended calibration grid, clean swarm only, 24 runs x 50 episodes, ~1 h
DRS="0.01 0.02 0.03 0.04" WINDS="0.1 0.25 0.5 1.0" bash scripts/byz_calibrate_shift.sh
# E2. Phase 3: triple x {none, mild, severe, mild_v1, severe_v1} x seeds 1-3, n=50, 60 runs, ~3 h
DRY_RUN=1 bash scripts/byz_phase3.sh  # optional: list the runs first
bash scripts/byz_phase3.sh            # ends by writing results_v2/phase3_table.md
CROSS=1 bash scripts/byz_phase3.sh    # optional: + cross-seed Tier 3 adversaries (30 more runs)
```

**E result (2026-10-12):** mild = DR 0.03 (+9.6 % on the clean swarm), severe = DR 0.06 + wind 0.1
(+60.1 %), plus mild_v1 (+32.6 %) and severe_v1 (+140 %). Under every shift level no member's
damage is detectable at n = 50 per seed (design §6.33).

**F — final tables, and the cross-seed Phase 3 factor** — `bash run_block_F.sh`, ~1.2 h, log in
`logs_v2/`; `CROSS=0 bash run_block_F.sh` for the tables only (~15 min).

```bash
CROSS=1 bash scripts/byz_phase3.sh    # F0. cross-seed Tier 3 adversaries, 30 runs, ~1 h; rewrites phase3_table.md
python -m byzantine_experiments.analysis.gate_tier3 --horizon
python -m byzantine_experiments.analysis.efficacy_table
python -m byzantine_experiments.analysis.transfer_matrix --runs adv_seed1_stealth0_lieonly \
    adv_seed2_stealth0_lieonly adv_seed3_stealth0_lieonly adv_seed3_stealth0_lieonlyego \
    adv_seed1_stealth0_lieonlyplaus adv_seed2_stealth0_lieonlyplaus adv_seed3_stealth0_lieonlyplaus
python -m byzantine_experiments.analysis.tier3_report rewards --refresh
python -m byzantine_experiments.analysis.tier3_report compute --refresh
```

**What to look for.** An adversary *passes* only in `results_v2/tier3_efficacy.md`, with a
PASS verdict (≥ 10 % on n = 200 paired, CI above 0, survival ≥ 70 %, action ratio ≤ 1.5 for
the plausible runs, transfer tested). For D3, the question is whether the plausible liar still
does significant damage (CI above 0) — if yes it becomes the triple's Tier 3 member; if not, the
free liar does and the stealth constraint is reported as a limitation. Either way it is a result.

## 2A. Tier 2 — how the matrix was produced (complete — for reference)

Spec: [`specs/TIER2_SIGNFLIP.md`](byzantine_experiments/specs/TIER2_SIGNFLIP.md).
All 57 runs are complete. Results in `design/tier2_design.md` and `why/TIER2_explanation.md`.

```bash
# 57 runs x 50 episodes  (~75 min at PARALLEL=12, ~7 GB)
PARALLEL=12 bash scripts/byz_rollout_tier2.sh

# then the analysis (seconds / ~1 min)
python -m byzantine_experiments.analysis.aggregate_tier2
python -m byzantine_experiments.analysis.behaviour \
    --results-dir byzantine_experiments/results/tier2 \
    --clean-dir  byzantine_experiments/results/tier1 \
    --prefix t2 --plots --wandb
```

Optional first (~6 s): `pytest byzantine_experiments/tests/test_tier2.py -q` → `31 passed, 3 skipped`.
The 3 skips are the efficacy test, which needs real rollouts; to run it too:
`BYZ_SLOW_TESTS=1 pytest byzantine_experiments/tests/test_tier2.py -q -k efficacy` (~4 min).

**The matrix:**

| variant | configs | f | runs |
| --- | --- | --- | --- |
| 2A `signflip` | `partial` (body roll/pitch flip, thrust honest) + `full` (all motors) | 0.125, 0.25 | 12 |
| 2B `waypoint_pull` | α ∈ {0.3, 0.6, 1.0} | 0.125, 0.25 | 18 |
| 2C `liar` | lie_mode ∈ {offset, stale, mirror} | 0.125, 0.25 | 18 |
| 2D `collusion` | mode ∈ {same_waypoint, pincer, mirror_pair} | 0.25 only | 9 |

**Clean references are not re-run.** Tier 1 already collected 3 clean rollouts from the same
checkpoints with the same collector and the same 50 episodes; the Tier 2 aggregator reads them
from `results/tier1/`. That saves 45 minutes of identical numbers.

**Knobs** (same style as Tier 1): `DRY_RUN=1`, `WANDB=0`, `FORCE=1`, `ONLY=2a|2b|2c|2d` to run
one variant's block, `ALPHA_REFINE=1` for the 2B α grid, `KEEP_RUN_LOGS=1`. Resumable —
finished configs are skipped.

```bash
ALPHA_REFINE=1 PARALLEL=12 bash scripts/byz_rollout_tier2.sh   # 12 runs, ~20 min (done)
```

**What the matrix found** (details in `why/TIER2_explanation.md`):

| variant | goal-RMSE degradation | in the 20–40 % band? |
| --- | --- | --- |
| 2A signflip | 28.9 – 67.6 % | ✅ `full` at f=0.125 → 28.9 % (the only one) |
| 2B waypoint | 77.8 – 168.1 % | ❌ too strong even at α=0.3 |
| 2C liar | 5.6 – 11.8 % | ❌ too weak by construction |
| 2D collusion | 73.7 – 181.2 % | ❌ too strong |

And the finding that spans both tiers: **honest-drone formation error stays within −6.8 % to
+9.5 % of clean in every Tier 2 config** — the same ±10 % band as Tier 1. Attacking 1–2 of 8
drones does not measurably damage the other six, intent or no intent.

### If a metric is added after a sweep has run

Do **not** re-simulate. The trajectories are on disk; replay the metric computation:

```bash
python -m byzantine_experiments.analysis.refresh_metrics \
    --results-dir byzantine_experiments/results/tier2          # ~84 s for 57 runs
```

This was needed twice already — Tier 2 added `attacker_survived`, `attacker_below_guard_frac`
and the cumulative-reward curve, none of which existed when Tier 1's clean references were
collected (without the curve, every Tier 2 time-to-impact reads `n/a`).

---

## 2B. How the Tier 1 sweeps were produced (all complete — for reference)

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

**Registered tiers:** `tier0_noop`, `tier1_gaussian`, `tier1_ou`, `tier1_dropout`,
`tier1_stall`, `tier2_signflip`, `tier2_waypoint_pull`, `tier2_liar`, `tier2_collusion`

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
