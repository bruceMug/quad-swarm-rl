# Tier 2 — reading the outputs

**Purpose:** pick up *any* file Tier 2 produced — a filename, a JSON key, a figure — and know
what it is and why it exists, without reading code. Companion to
[`TIER1_explanation.md`](TIER1_explanation.md).

- **What was built and why** → [`specs/TIER2_SIGNFLIP.md`](../specs/TIER2_SIGNFLIP.md)
- **What was decided and measured** → [`design/tier2_design.md`](../design/tier2_design.md)
- **How to run it** → [`readme_bz.md`](../../readme_bz.md)
- **What the files mean** → this file

---

## 1. The one-paragraph version

Same 8-drone swarm, same trained policy, same 3 seeds. But where Tier 1 made drones fly
*badly*, **Tier 2 makes drones fly badly on purpose**. Four attack styles, run at two
compromised fractions on three baseline policies = **57 runs**, 50 episodes each. Tier 1 was
the "broken" reference; Tier 2 is the "malicious but obvious" reference. Together they set up
the paper's question: when Tier 3 arrives with a *trained* adversary, can the detector tell
malicious from merely broken — and is it doing that, or just measuring how much damage
happened?

**The single most important thing to know:** three of the four attacks turned out to be far
*stronger* than the 20–40 % damage band we need for a fair comparison, and one (`liar`) is far
weaker. Only one config landed in the band. See §6.

---

## 2. Decoding a filename

```
behaviour_t2_waypoint_s1_a0.6_f025.png
─────────  ── ──────── ── ──── ────
    │       │     │     │   │    └── f025 = 0.25 = 2 of 8 drones compromised
    │       │     │     │   └─────── a0.6 = severity: alpha (blend weight) = 0.6
    │       │     │     └─────────── s1   = baseline seed 1
    │       │     └───────────────── waypoint = which attack variant
    │       └─────────────────────── t2   = Tier 2
    └─────────────────────────────── behaviour = which analysis produced it
```

### Tokens specific to Tier 2

| Token | Meaning | Values |
| --- | --- | --- |
| `t2` | Tier 2 (`t0` infrastructure, `t1` faults) | — |
| `signflip` / `waypoint` / `liar` / `collusion` | which of the four attacks (§3) | — |
| `partial` / `full` | 2A only: which sign flip. `partial` = invert roll/pitch, keep thrust honest. `full` = invert all four motors. | — |
| `a0.6` | 2B/2D: **alpha**, how committed the attacker is. 0 = honest, 1 = ignores the policy entirely. | 0.3, 0.6, 1.0 |
| `offset` / `stale` / `mirror` | 2C: which kind of lie | — |
| `same_waypoint` / `pincer` / `mirror_pair` | 2D: how the two attackers coordinate | — |
| `s1 s2 s3`, `f0125`, `f025` | as Tier 1 — seed, and 1-or-2-of-8 compromised | — |

Note **2D exists only at `f025`** — you cannot have two colluding drones if only one is
compromised.

---

## 3. The four attacks — what each one physically is

All four run the **same trained policy** and corrupt one or two drones on the way out. Unlike
Tier 1, these may read *their own* position — that is what "intent" requires. None of them can
read an honest drone's state.

### 3A. `signflip` — the tumbling drone

Takes the policy's motor command and negates it.

- **`full`** — negate all four motors. Collective thrust inverts, so the drone falls.
- **`partial`** — negate only the *roll and pitch* components (via the control Jacobian),
  leaving collective thrust honest. Meant to keep it airborne while destroying its lateral
  control.

> **A wrinkle worth knowing.** The spec said "`dims=[1,2]` flips roll/pitch". But an action
> here is **4 raw motor thrusts**, not (thrust, roll, pitch, yaw) — so flipping dims 1 and 2
> flips two *rotors*, not two axes. Both readings are implemented (`space='motor'` vs
> `space='body'`); `partial` uses the body-space one, which is what the spec actually meant.

### 3B. `waypoint` — the defector

Ignores the team goal and flies at a fixed adversarial corner of the arena, dragging
neighbours off-formation through collision avoidance. `alpha` blends between the honest
command and "fly at the corner". It uses the simulator's own Mellinger position controller to
work out the motor commands.

### 3C. `liar` — the honest flier with a false transponder

**Its motor commands are bit-for-bit identical to the honest policy's.** The attack is
entirely in what it *tells its neighbours*:

- `offset` — reports a position a fixed 3 m away from where it really is
- `stale` — reports where it was 0.5 s ago
- `mirror` — reports its position reflected through the arena centre

This is the purest Byzantine attack in the proposal's original sense: correct actions, false
communications. It is also the one that creates a measurement problem — see §5.

### 3D. `collusion` — two attackers with a plan

Only at f = 0.25. `same_waypoint` = both defect to the same corner; `pincer` = opposite
corners; `mirror_pair` = both sign-flip, on complementary motor pairs. Tests whether a
set-based detector handles multiple simultaneous attackers.

### The survival guard — read this before interpreting 2A

Every attack has a safety net: below **0.6 m** altitude, the emitted action is blended back
toward the honest one (fully honest by 0.15 m). Without it, a sign-flip drone hits the floor in
two seconds and there is nothing left to detect.

**But the guard turned out to dominate 2A.** The `guard exposure` column says what fraction of
the episode the attacker spent below 0.6 m, i.e. being rescued:

| attack | guard exposure | honest reading |
| --- | --- | --- |
| `waypoint` | 10–22 % | a real attack, mostly flying free |
| `liar` | 4–6 % | flies honestly anyway |
| `signflip` | **66–83 %** | mostly being rescued from itself |
| `collusion mirror_pair` | **93 %** | almost entirely rescue |

So `signflip` is better described as "a drone that immediately dives at the floor and is then
continuously caught" than as "a drone flying adversarially".

---

## 4. What each output file is

```
byzantine_experiments/results/
├── tier2/                          57 runs × 2 files (7.7 GB of .npz, gitignored)
│   ├── t2_waypoint_s1_a0.6_f025.npz    ← raw trajectories
│   └── t2_waypoint_s1_a0.6_f025.json   ← config + all metrics (committed)
├── tier2_calibration.csv           one row per (config × seed) — 57 rows
├── tier2_calibration.md            averaged over seeds — the paper table
├── tier2_behaviour.json            behavioural descriptors + plain-English sentences
└── figures/behaviour_t2_*.png      19 figures (all seed-1 configs)
```

Same `.npz` schema as Tier 1 (see [`TIER1_explanation.md`](TIER1_explanation.md) §4A). The
`.json` carries every metric so you never need to open a 135 MB file.

**Clean references are Tier 1's.** There is no `t2_clean_*`. Tier 1 already collected three
clean rollouts from the same checkpoints with the same collector and 50 episodes, so Tier 2
reads them from `results/tier1/t1_clean_s{1,2,3}.json` — 45 minutes of identical numbers not
spent.

### Metrics Tier 2 adds

| Field | Meaning |
| --- | --- |
| `victim_spillover_goal_rmse_pct` | how much the **honest** drones' formation error got worse. The key Tier 1-vs-Tier 2 comparison. |
| `attacker_survived_mean` | fraction of episodes the attacker is still airborne at the end |
| `attacker_below_guard_frac_mean` | fraction of steps spent inside the guard ramp (the "guard exposure" above) |
| `time_to_impact_s` | seconds until cumulative team reward departs from clean by >2 clean σ, sustained. `n/a` = never separated. |
| `team_reward_curve` | the cumulative team-reward curve itself, downsampled 10× |

---

## 5. The 2C measurement problem — the thing to understand before reading any 2C number

A liar's own actions are honest **by construction** (its action deviation is exactly `0.000`
in all six configs), and its own observation is untouched. Therefore:

> **A policy graph built from the liar's own behaviour is a clean, honest policy graph.**
> There is nothing anomalous in it to find. A detector that scores the liar as honest is
> *correct*, not broken.

The behavioural analysis says exactly this, unprompted, for all six 2C configs: *"The degraded
drone is behaviourally indistinguishable from a clean drone at this severity."* And
[`behaviour_t2_liar_s1_mirror_f0125.png`](../results/figures/behaviour_t2_liar_s1_mirror_f0125.png)
shows it: in the action-trace panel the executed command lies exactly on top of the honest
command, a single line.

**What this means for the experiment:**

1. A per-agent AUC scored against `byzantine_mask` will look terrible for 2C. That is expected
   and must be reported with the caveat, not as a detector failure.
2. Detection has to come from the **victims** — the honest drones acting on false information.
3. **And there is a concrete signal there.** 2C barely moves goal error (5.6–11.8 %) but it has
   among the *highest collision rates in the entire matrix*: **+0.31 to +0.66 extra collisions
   per episode**, versus +0.02 to +0.11 for sign-flip. The reason is neat: neighbours steer
   around where the liar *claims* to be, and therefore into where it actually is. 2C is a
   **collision attack**, not a formation attack.

---

## 6. Reading the results

### The severity problem

The whole point of the calibration is to find attacks in the **20–40 % damage band**, so Tier 3
can be compared at matched damage. Only **one of 19 configs** landed there:

| variant | goal-RMSE degradation range | verdict |
| --- | --- | --- |
| `signflip` | 28.9 – 67.6 % | ✅ `full` at f=0.125 → **28.9 %** — the only in-band config |
| `waypoint` | 77.8 – 168.1 % | ❌ far too strong, even at α = 0.3 |
| `liar` | 5.6 – 11.8 % | ❌ far too weak (by construction, §5) |
| `collusion` | 73.7 – 181.2 % | ❌ far too strong |

Same pattern as Tier 1: the spec's severity grids bracket the band instead of landing in it.
2B would need α ≈ 0.1–0.15, below the spec's lowest value.

### The finding that carries across both tiers

**Attacking one or two drones out of eight does not measurably damage the other six.**
Honest-drone formation error stays within **−6.8 % to +9.5 %** of clean across every Tier 2
config — the same ±10 % band Tier 1 sat in, despite Tier 2 having actual intent.

It is often **negative**: honest drones do slightly *better* when a neighbour is attacking,
because the attacker leaves the formation and stops crowding everyone else.

> ⚠️ I initially claimed the opposite in an earlier draft, on the strength of a single-episode
> smoke test where honest reward fell from 3.01 to 1.13. The 57-run matrix does not support it.
> One episode of a heavy-tailed metric is not evidence.

### Which numbers to trust

Same rule as Tier 1, and it bites harder here: **do not use reward-based figures.** Across
Tier 2 runs the reward-based spillover ranges from **−575 % to +189 %**. The goal-RMSE version
ranges from −25 % to +29 % with a mean of +3 %. Use goal RMSE.

### At a glance

| figure | what it looks like |
| --- | --- |
| `behaviour_t2_signflip_*` | violent tumbling — wobbles at 4.7–6.0× clean, on the floor 26–30 % of the time |
| `behaviour_t2_waypoint_a1.0_f0125` | **the cleanest attack** — smooth purposeful drift to the wrong place, no wobble, 5.9 % floor time, 100 % survival |
| `behaviour_t2_liar_*` | indistinguishable from clean (§5) |
| `behaviour_t2_collusion_pincer_f025` | like waypoint, doubled |
| `behaviour_t2_collusion_mirror_pair_f025` | like signflip, worse — 37 % survival, 93 % guard exposure |

---

## 7. What to carry into Tier 3

1. **The 30 % team-reward-drop target in the Tier 3 spec is probably unreachable at f = 0.125.**
   A drone flying full-tilt at an adversarial corner (2B, α = 1.0) produced +2.9 % spillover.
   Either Tier 3 must be trained specifically to induce *coupling* damage (collisions,
   formation break-up) rather than to maximise its own disruption, or the target should be
   restated per-agent. Worth deciding before spending GPU hours.
2. **`waypoint` is the attack mechanism to build the Tier 2 story on** — high survival
   (91–100 %), low guard dependence (10–22 %), smooth legible behaviour. `signflip` is
   mostly being rescued by its own safety net.
3. **The liar's signal is in victim collisions.** If the detector is going to catch 2C at all,
   that is where to look, and it gives the set-structured formulation a concrete job.
4. **One in-band config is thin.** All three Tier 1 canonical settings currently pair to the
   same single Tier 2 config (`signflip_full_f0125`, 28.9 %). A short α-refinement sweep for
   2B would give a second, mechanistically different in-band pair and make the headline
   comparison much stronger.
