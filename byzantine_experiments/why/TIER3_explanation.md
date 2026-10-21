# Tier 3 — reading the outputs

**Purpose:** pick up *any* file Tier 3 produced — a filename, a table, a figure — and know what
it is, why it exists, and what it shows, without reading code. Companion to
[`TIER1_explanation.md`](TIER1_explanation.md) and [`TIER2_explanation.md`](TIER2_explanation.md).

- **What was built and why** → [`specs/TIER3_ADVERSARIAL.md`](../specs/TIER3_ADVERSARIAL.md) (with dated amendments A1–A7)
- **What was decided and measured, in order** → [`design/tier3_design.md`](../design/tier3_design.md)
- **What is complete** → [`design/tier3_audit.md`](../design/tier3_audit.md)
- **How to run it** → [`readme_bz.md`](../../readme_bz.md), blocks A–F
- **What the files mean** → this file

Every number here is from the v2 corpus (`results_v2/`, paired comparisons). Nothing from the
legacy `results/` folder is quoted; see §6 for why.

---

## 1. The one-paragraph version

Tiers 1 and 2 broke a drone by hand — by accident (Tier 1) or on purpose (Tier 2). **Tier 3 lets
reinforcement learning find the attack.** One drone in the 8-drone swarm is handed to a new
policy, trained for 200 M steps to make the *other seven* fly worse, while the seven keep their
original, frozen policy. Twelve adversaries were trained (~455 GPU-hours). The first ones, which
could only move their own motors, crashed themselves and did no harm. The ones that work never
touch their motors at all: they fly exactly like an honest drone and **lie about where they are**.
The best of them degrade the honest drones' formation error by 7–8 % on the team they were trained
against — significant, but short of the 10 % gate, and roughly what the best hand-designed liar
already does. A version restricted to *physically plausible* lies (within 2 m, smooth, inside the
room) still does significant damage and passes every naive sanity check on its broadcasts.

**The single most important thing to know:** the attack lives in the *communication* channel, not
the *control* channel. Corrupting what honest drones believe works; flying badly does not — for
hand-designed and trained attackers alike.

---

## 2. Decoding a filename

### The adversary's name (a trained policy, in `train_dir/`)

```
adv_seed3_stealth0_lieonlyplaus
    ───── ──────── ───────────
      │      │         └── channel: (none) = motors only
      │      │                      _lie      = motors + lies
      │      │                      _lieonly  = flies the honest policy, controls only the lie
      │      │                      ...ego    = also sees its own true state (round 5)
      │      │                      ...plaus  = plausible lie (D3, the stealth condition)
      │      └──────────── stealth weight on action deviation (0 or 5); _f250 = 2 attackers
      └─────────────────── seed3 = trained against baseline policy seed 3
```

### Rollout files (each is a `.npz` trajectory file plus a `.json` sidecar with its metrics)

| pattern | what it is | n | folder |
|---|---|---|---|
| `t3_<adv>_on_s<B>_m<XXX>M` | horizon sweep: checkpoint at XXX M steps, against baseline seed B | 50 | `tier3/` |
| `confirm_<adv>_m194_n200` | confirmatory run: final checkpoint, own seed | 200 | `tier3_confirm/` |
| `confirm_<adv>_m194_on_s<B>_n200` | transfer cell: final checkpoint, another seed | 200 | `tier3_confirm/` |
| `confirm_clean_s<B>_n200` | the clean reference those pair with | 200 | `tier3_confirm/` |
| `confirm_hand_liar_<mode>_s3_n200` | hand-designed Tier 2 liar, for the head-to-head | 200 | `tier3_confirm/` |
| `triple_<t1/t2/t3>_<config>_s<S>_n800` | matched-damage triple member | 800 | `matched_triple/` |
| `shiftcal_s<S>_dr<r>[_wind<w>]` | clean swarm under a candidate shift | 50 | `shift_calibration/` |
| `p3_<member>_<shift>_s<S>_n50` | Phase 3 rollout: member × shift level × seed | 50 | `phase3/` |

`m194` is the final checkpoint (97 % of 200 M; the last saved milestone). The horizon sweep uses
3, 12.5, 25, 37.5, 50, 75 and 97 % of training: `m006 … m194`.

**Pairing.** Any two runs with the same `--seed` replay identical scenarios, episode for episode
(same spawns, goals, and under shift the same dynamics and wind). Every comparison in Tier 3 is
attacked-minus-clean on the *same* episodes, and `analysis/paired.py` verifies this before
computing anything.

---

## 3. The adversaries — what each one physically is

### 3A. Motors only (rounds 1–2) — the drone that crashes itself
`adv_seed1_stealth0`, `_f250` (two attackers). The policy chooses its own four thrust commands.
It learns to fly erratically and crashes in every evaluation episode (survival 0 %), doing no
measurable harm (−1.1 % and −5.9 %). The two `stealth5` runs were defective — the stealth penalty
was computed wrongly in one and swamped the objective in the other — and are listed but not scored.

### 3B. Motors + lies (round 3) — still crashes
`adv_seed1_stealth0_lie`. Same, plus it can broadcast a false position and velocity. It still
crashes (survival 0 %): flying badly is what the optimiser finds first, and it kills the attacker.

### 3C. Lie-only (rounds 4–5) — the attack that works
`adv_seed{1,2,3}_stealth0_lieonly`, `adv_seed3_stealth0_lieonlyego`. The drone flies the honest
policy; the adversary controls **only what it tells its neighbours** about itself — up to 10 m per
axis of position and 6 m/s per axis of velocity. In practice it claims to be ~17 m away, mostly
below the floor.

### 3D. Plausible lie (D3) — the stealth condition
`adv_seed{1,2,3}_stealth0_lieonlyplaus`. Same channel, but the lie must look physical: within 2 m
of the truth, drifting at ≤ 1 m/s with ≤ 4 m/s² acceleration, inside the room, and consistent
between the claimed position and claimed velocity. This replaced the action-penalty "stealth" of
the original spec (amendment A3), because a lie-only adversary flies the honest policy and its
actions are already indistinguishable.

### The reward
Every adversary maximises the negative of the honest team's reward, minus 10 for crashing itself,
plus a small bonus per step alive. Nothing in the reward mentions lying, phantoms or collisions —
everything in §5 was found by the optimiser.

---

## 4. What each output file is

| file (in `results_v2/`) | answers | read it as |
|---|---|---|
| `tier3_gate.md` / `.csv` | how does damage evolve over training? | n = 50 screening per checkpoint; `CANDIDATE` means "worth confirming", never "passed" |
| `tier3_efficacy.md` | did any adversary pass the spec's gate? | **the verdict table**: one PASS/FAIL per adversary against all five criteria |
| `tier3_transfer.md` | does an adversary trained on one team hurt the others? | 3 × 3 per adversary; `*` own seed, `T` significant on another seed |
| `matched_triple_selection.json` | which Tier 1/2/3 configurations form the triple | written by the pre-registered rule *before* any triple episode was collected |
| `matched_triple.md` | are the three damage-equivalent? | TOST, ±2.08 pp margin, n = 800 per seed |
| `shift_levels.json` | what "mild" and "severe" mean | four levels, each with its measured effect on the clean swarm |
| `phase3_table.md` | does damage survive distribution shift? | % and cm columns; n = 50 per seed, descriptive |
| `channel_ceiling.md` | which *channel* can do damage at all? | all Tier 2 and Tier 3 runs grouped by channel |
| `tier3_report/rewards*` | what did each reward term do over training? | first-vs-last-10 % table + one figure per adversary |
| `tier3_report/compute.md` | GPU-hours | active hours exclude pauses; 455 GPU-h total |
| `tier3_report/stealth_<adv>*` | would a naive detector catch it? | action-level and broadcast-physics checks, thresholds set on the clean run |
| `tier3_report/behaviour_<adv>*` | what does the attack *do*? | paired table + 10 episodes drawn top-down + damage over time |

### The metric everything uses
**Honest goal-RMSE degradation, paired**: how much further the *honest* drones are from their
assigned formation slots than in the clean run of the same episodes, in % of the clean value.
Team reward is reported alongside as a secondary figure but never gated or matched on: mean episode
reward is a small number with a heavy tail, so its percentage swings wildly (e.g. +78 % [+33, +123]).

---

## 5. Reading the results

### The gate: no adversary passes
The spec asks for ≥ 10 % damage with a CI above zero on n ≥ 200, survival ≥ 70 %, an action ratio
≤ 1.5 for stealth runs, and transfer tested. (The original 30 % was recalibrated to 10 % on
2026-09-24, when the corrected data showed no hand-designed attack came near 30 %.)

| adversary | own-seed damage, n = 200 | survival | verdict |
|---|---|---|---|
| ego, seed 3 | **+8.0 %** [+3.7, +12.3] | 97 % | FAIL (below 10 %) |
| free liar, seed 3 | **+7.4 %** [+3.1, +11.7] | 98 % | FAIL |
| plausible liar, seed 3 | **+7.1 %** [+2.4, +11.8] | 98 % | FAIL |
| free liar, seed 1 | +4.6 % [+0.4, +8.9] | 100 % | FAIL |
| free liar, seed 2 | +3.1 % [−0.7, +7.0] | 100 % | FAIL |
| plausible liar, seed 1 | +2.8 % [−1.4, +7.0] | 98 % | FAIL |
| plausible liar, seed 2 | +2.6 % [−1.0, +6.2] | 98 % | FAIL |
| motors-only adversaries | −6 to −1 % (screening) | 0 % | FAIL (survival) |

The best trained adversaries are statistically indistinguishable from the best hand-designed
liar (head-to-head on the same 200 episodes, design §6.24). Learning found the same ceiling the
hand-designed attack sits at, not a higher one.

### Why: the channel, not the optimiser
`channel_ceiling.md` groups every Tier 2 and Tier 3 run by what the attacker can touch (written
2026-09-25, before the plausible and seed-2 adversaries; it uses an unpaired test, which can only
under-count). On the motors-only channel, 0 of 53 runs — hand-designed or trained — does
significant damage. On the observation channel, 10 of 46 do. A drone that flails mostly removes itself (and a crowding
constraint) from the formation; a drone that lies changes what seven others decide.

### What the trained attack does (design §6.30)
**In plain language: it makes itself invisible.** It flies exactly as an honest drone would, but
tells its neighbours it is somewhere else (free liar: ~17 m away, mostly underground; plausible
liar: ~1.7 m away). Their collision avoidance reacts to the phantom instead of the real drone, so
they fly into it. Seed 3, free liar vs clean on the same episodes:

| | attacked | clean |
|---|---|---|
| team collisions per episode | 0.91 | 0.42 |
| collision steps involving the adversary's slot | 0.54 | 0.11 |
| collision steps between two honest drones | 0.36 | 0.32 (no change) |
| closest approach, adversary ↔ nearest honest | 0.13 m | 0.17 m |

Damage falls on the drones nearest the attacker and grows over the episode (4 / 41 / 55 % of it in
the first / middle / last third). The seed-3 plausible liar does the same thing with a 2 m lie and
produces near-identical totals (0.91 collisions, 0.54 involving its slot) — the size of the lie
matters less than the fact that the real drone is not where its neighbours think.

### Stealth: the plausible liar passes naive checks
Naive broadcast checks (out of the room, too fast, too much acceleration, position and velocity
inconsistent), thresholded on the clean run:

| adversary | its episodes flagged | honest episodes flagged |
|---|---|---|
| free liars | **100 %** | 0.1–0.3 % |
| plausible liars (seeds 1, 2, 3) | **0.0 / 0.0 / 0.5 %** | 0.1–0.4 % |

Action-level: every lie-only adversary flies the honest policy, so its action ratio is 1.05–1.13
and a naive action-magnitude detector flags 0.5–2.5 % of its episodes against 0.5–1.7 % of honest
ones. **A naive detector does not catch the plausible liar.** Constraining the lie cost little:
pooled over seeds, the plausible liars do +4.2 % [+1.7, +6.6] against +5.0 % for the free ones
(seed 3: +7.1 vs +7.4 %).

### Transfer: partial, and the teams differ
4 of the 14 off-seed cells are significant. Every adversary does the most damage against the
seed-3 team (mean over adversaries: seed 1 +3.3 %, seed 2 +2.1 %, seed 3 +6.1 %). The seed-3
baseline policy is simply more vulnerable to being lied to; only seed-3-trained adversaries
reach +7 %.

### The matched-damage triple: NOT MATCHED, by a hair
| member | damage, n = 800 × 3 seeds |
|---|---|
| Tier 1 `stall_d0.50_ms0.1_hold` | +2.69 % [+1.53, +3.85] |
| Tier 2 `liar_mirror` | +3.80 % [+2.65, +4.95] |
| Tier 3 plausible liars | +2.80 % [+1.66, +3.94] |

Tier 3 is equivalent to Tier 1 (−0.11 pp) and to Tier 2 (+1.00 pp) within the ±2.08 pp margin.
Tier 1 vs Tier 2 is not shown equivalent: −1.11 pp, 90 % CI [−2.15, −0.07] — the lower bound
misses the margin by 0.07 pp. Reported as it stands; no episodes were added to change it.

### Distribution shift: damage disappears under it (at this n)
| shift level | settings | clean swarm's own degradation |
|---|---|---|
| mild | dynamics randomisation 3 % | +9.6 % |
| mild_v1 | 5 % | +32.6 % |
| severe | 6 % + wind 0.1 m/s² | +60.1 % |
| severe_v1 | 10 % + wind 0.5 m/s² | +140 % |

Without shift each triple member adds ~6–8 cm of honest error on these 50 episodes; under every
shift level every member's added error is within noise of zero (intervals ±5 to ±9 cm). Whether
the attacks weaken or are merely buried cannot be resolved at n = 50 per seed. Either way, under
shift a damage meter sees none of the three attacks.

---

## 6. Which numbers to trust

- **v2 paired only.** Before the seeding fix (design §6.16–6.18, sidecars tagged
  `seeding: per_episode_v2`) the collector did not control scenarios, so attacked and clean runs
  saw different ones; those numbers (in `results/`, and in older sections of the design docs)
  are kept for history only. Large effects on the attacking drone itself replicate; small effects
  on the honest drones — the ones this tier is about — did not (the best legacy liar run, +30.4 %,
  measures +3.9 to +9.7 % paired), so only v2 numbers are quoted.
- **n = 50 is screening, n = 200 is evidence.** A single n = 50 run has ~57 % power at 10 %. The
  best adversary's final checkpoint screened at +13.0 % (n = 50) and confirmed at +7.4 % (n = 200),
  and every triple member's screening estimate shrank at n = 800 (e.g. stall 4.4 → 2.7 %).
  Selecting on n = 50 picks lucky runs. Quote n ≥ 200 numbers.
- **Percent vs centimetres under shift.** The % in `phase3_table.md` is relative to the clean error
  at that level, which grows 2.6× from no shift to severe_v1; the cm column is the absolute view.
- **Broadcast statistics** exclude each episode's last step, which belongs to the next episode's
  reset (design §6.31). Without that, a plausible liar looks 99.5 % detectable; with it, 0.5 %.

### At a glance
| question | answer |
|---|---|
| Did any trained adversary pass the 10 % gate? | No. Best +8.0 %. |
| Is the trained attack better than hand-designed? | No — equal to the best hand-designed liar. |
| What channel matters? | Communication (lies). Motors-only never works. |
| What does it learn? | To hide where it is; neighbours collide with the real drone. |
| Can a stealthy version still hurt? | Yes, +4.2 % pooled, and it passes naive checks. |
| Does it transfer? | Partly (4/14 cells); the seed-3 team is most vulnerable. |
| Is the triple damage-matched? | Tier 3 ≡ Tier 1 and ≡ Tier 2; Tier 1 vs Tier 2 not shown. |
| Does damage survive shift? | Not detectably at n = 50. |

---

## 7. What to carry into Phase 3 (the detector)

- **The data is ready.** `phase3/` holds the triple at five shift levels, three seeds, own-seed and
  cross-seed adversaries, each with its clean twin; the collector format is the same as Tiers 1–2.
- **Test against the plausible liar, not the free one.** A one-line bounds check catches every free
  liar; it catches no plausible one. A detector that only beats the free liar shows nothing.
- **Under shift, damage is invisible.** A detector that still separates the three mechanisms there
  cannot be doing it by measuring damage — the strongest version of the paper's claim.
- **State the triple honestly.** The trained adversary is damage-matched to both hand-designed
  mechanisms; the two hand-designed ones differ by ~1 pp, which any Tier 1 vs Tier 2 comparison
  must mention.
- **Seed 3 is special.** The only significant own-seed attacks (bar seed 1's free liar) are on the
  seed-3 team; results that hold only there should be reported per seed.
