# Tier 3 audit against `specs/TIER3_ADVERSARIAL.md` (2026-09-25)

## Status update — 2026-10-21: Tier 3 complete (blocks A–F run, final tables built)

Blocks A–E completed (logs in `logs_v2/run_*`). Results in `tier3_design.md` §6.31–6.33.

| spec item / gap | status | result |
|---|---|---|
| Checkpoint hash test | **DONE** | passed (block A); full suite 75 passed, 1 skipped (= that slow test) |
| Stealth-constrained adversary (D3) | **DONE** | 3 seeds; pooled own-seed damage +4.16 % [+1.74, +6.58]; naive broadcast checks flag 0–0.5 % of its episodes vs 0.1–0.4 % honest |
| Seed-2 adversary (D4) | **DONE** | own seed +3.1 % [−0.7, +7.0], not significant |
| Item 3 efficacy gate | **DONE** | all 12 adversaries in `tier3_efficacy.md`: 10 FAIL, 2 NOT SCORED (defective); best +8.0 / +7.4 / +7.1 % (seed 3) |
| Item 4 transfer | **DONE** | 7 adversaries × 3 seeds; 4 of 7 significant on their own seed (free s1, free s3, ego s3, plausible s3); 4 of 14 transfer cells significant; baseline seed 3 is the most vulnerable team (mean over adversaries: s1 +3.3, s2 +2.1, s3 +6.1 %) |
| Item 5 matched triple | **DONE** | NOT MATCHED: Tier 3 ≡ Tier 1 and ≡ Tier 2; Tier 1 vs Tier 2 not shown equivalent (−1.11 pp, 90 % CI [−2.15, −0.07], margin ±2.08) |
| Item 6 stealth | **DONE** | all 3 plausible (stealth) liars + free s2, s3 (free s1 and ego s3 not run; not required); free liars flagged 100 %, plausible 0–0.5 % |
| Item 7 behaviour | **DONE** | same 5 adversaries; plain-language description §6.30 |
| Item 2 / 8 rewards, compute | **DONE** | all 12 runs; 455 GPU-h, 2.4 B env steps |
| Phase 3 priority subset (A6, A7) | **DONE** | triple × {none + 4 shift levels} × 3 seeds; damage undetectable under every shift level at n = 50 |
| Phase 3 cross-seed factor | **DONE** | block F, 30 runs; same pattern as own-seed (design §6.34) |
| Plain-language explanation (as `why/TIER1/2_explanation.md`) | **DONE** | `why/TIER3_explanation.md` (2026-10-21) |
| Tier 1/2 docs on v2 numbers | **DONE** | §0 in `tier1_design.md`, `tier2_design.md` and both explanation files; v2 behaviour figures in `results_v2/figures/` (design §6.35) |


## Status update — 2026-09-26: every gap is built; what remains is compute

Decisions taken by the project owner: **D1** family-level validation for Tiers 1/2, triple target
= Tier 3's confirmed effect, new TOST margin declared first; **D2** keep 10 %, report both facts;
**D3** train a plausible-lie round; **D4** train a seed-2 adversary; **D5** build distribution
shift. Recorded as dated amendments A1–A6 in the specs, and in `tier3_design.md` §6.25–6.30.
Status key below adds **BUILT** = code and tests done, needs the listed run.

| gap (from §1–§5 below) | status | where / command (`readme_bz.md` Step 17) |
|---|---|---|
| 30 % → 10 % applied to Tier 3 only | **DONE** | A1 in Tier 1/2 specs; A2 in Tier 3 spec; aggregators use `family_validation.py` (Step 17 A1 regenerates the tables) |
| Tier 1/2 fail their own band gates | **DONE** (D1) | `stall` and `liar` validated at family level; prerequisite met under A1 |
| Tier 3 spec gate text still 30 % | **DONE** | amendment A2 |
| Matched-triple anchors / band are legacy | **DONE** | pre-registered rule, `matched_triple.py`; amendment A5 |
| TOST margin ±10 pp larger than the effects | **DONE** | ±min(3 pp, ½ target), declared before data; A5 |
| Checkpoint hash test never run | BUILT | Step 17 A5 |
| Stealth-constrained adversary missing | BUILT (D3) | plausible lie, `adversary/lies.py`; train Step 17 B1, B3, B4 |
| Seed-2 adversary missing | BUILT (D4) | train Step 17 B2 |
| Team reward CI unpaired | **DONE** | `paired.paired_reward_degradation`; in the efficacy table |
| Transfer never tested | BUILT | `scripts/byz_transfer.sh` (Step 17 A2, then C2 per new adversary) |
| Consolidated efficacy table | **DONE** | `efficacy_table.py` → `results_v2/tier3_efficacy.md` |
| Test 6 checkpoint round-trip | **DONE** | passes |
| Test 8 transfer criterion | **DONE** | five-criterion verdict, every branch tested |
| Test 9 TOST both directions | **DONE** | passes |
| Phase 3 distribution shift unsupported | BUILT (D5) | `wrappers/shift.py`; calibrate Step 17 A3; rollouts Step 17 E |
| Item 2 reward-component plots | BUILT | `tier3_report rewards` (Step 17 A4) |
| Item 6 stealth analysis + histogram | **DONE** for the best adversary | `results_v2/tier3_report/stealth_*`; new adversaries Step 17 C3 |
| Item 7 behavioural observations | **DONE** for the best adversary | `results_v2/tier3_report/behaviour_*`; plain-language description in design §6.30 |
| Item 8 compute accounting | BUILT | `tier3_report compute` (Step 17 A4) |
| Item 9 test count | **DONE** | 71 passed, 1 skipped (the hash test) |

**One finding added by the new analysis, relevant to §3 below:** the best adversary is not
completely invisible at the action level — action-norm ratio 1.10, AUC 0.575, caused by the attack
(the same slot in the clean run: 0.99 / 0.49) — but a naive action threshold still flags only
2.5 % of its episodes against 1.6 % of honest ones. The broadcast bounds check flags 100 %.

The text below is the original audit of 2026-09-25, kept unchanged.

---

Written before any paper drafting, so nothing the spec requires is forgotten. Every row was
checked against files on disk, not recalled. Status key: **DONE**, **PARTIAL**, **MISSING**,
**DECISION** (needs the project owner before work can proceed).

**Short answer: Tier 3 is not complete per spec, and the 30 % → 10 % recalibration was applied
to the Tier 3 gate only.** Everything below is what remains.

---

## 1. The recalibration is partial

The Tier 3 efficacy gate moved to 10 % (`gate_tier3.GATE`, the single definition, read by
`confirm_tier3` and `channel_ceiling`). **These thresholds were set on the same inflated legacy
scale and were not moved:**

| threshold | where | still says | v2 reality |
|---|---|---|---|
| Tier 1 validation band | `specs/TIER1_STOCHASTIC.md` §validation gate; `aggregate_tier1.BAND_LO/HI` | 20–40 % | best Tier 1 family mean +2.6 %, best single run +12.6 % |
| Tier 2 "attack must work" / band | `specs/TIER2_SIGNFLIP.md` item 4, gate; `aggregate_tier2.BAND_LO/HI` | ≥ 20 %, 20–40 % | best Tier 2 family mean +5.7 %, best single run +13.1 % |
| Matched-triple target band | `TIER3` spec, damage matching | ~25 %, "tune to land in 20–40 %" | Tier 3 confirmed at +7.4 % |
| TOST equivalence margin | `TIER3` spec | ± 10 pp | larger than the effects themselves: clean (0 %) would be "equivalent" to Tier 3 |
| Tier 3 gate text | `TIER3` spec efficacy-gate table | ≥ 30 % | code uses 10 % |

Consequences:

* **No v2 Tier 1 or Tier 2 configuration reaches 20 %, so both earlier tiers fail their own
  validation gates as written.** The Tier 3 spec's prerequisite ("Tiers 0, 1, 2 complete with
  validation gates passed") is formally unmet.
* The aggregate scripts test the band on the **all-drone, unpaired** goal-RMSE figure, the
  metric retired in tier3_design §3.2 and §6.16. Their paired columns are correct; the
  `in_20_40_band*` columns are not meaningful.
* The legacy-era docs (`design/tier1_design.md`, `design/tier2_design.md`,
  `why/TIER1_explanation.md`, `why/TIER2_explanation.md`, the readme's canonical table) quote
  legacy numbers. **A "do not cite" banner was added to each on 2026-09-25**; their contents
  were not rewritten.

## 2. Spec requirements, one by one

### Training setup

| requirement | status | evidence / what remains |
|---|---|---|
| Frozen honest team, path documented | DONE | custom wrapper `FrozenTeamEnv`, design §1; test 1 passes |
| Checkpoint hash unchanged after training steps | PARTIAL | test exists but is slow-gated (`BYZ_SLOW_TESTS=1 pytest -k baseline_checkpoint`); no record it was ever run. Run once. |
| Reward components logged separately in wandb | DONE | `adv/r_team`, `r_crash`, `r_alive`, `r_stealth` |
| Unconstrained adversary (`w_stealth=0`) | DONE | several |
| **Stealth-constrained adversary (`w_stealth=5, ε=0.3`) — the paper's headline condition** | **MISSING** | both `w_stealth=5` runs are invalid: round 1 scored unclipped actions (penalty −534 k), round 2 was dominated 555:1 by the penalty (design §4.2). See decision D3. |
| 6 runs: 3 seeds × 2 stealth settings | PARTIAL | seed 1: 6 variants; seed 3: 2 (lie-only, lie-only+ego); **seed 2: none** |

### Efficacy gate (five criteria)

| criterion | status | result for the best adversary |
|---|---|---|
| Goal-RMSE degradation (primary) | DONE (paired, honest drones, n = 200) | **+7.4 % [+3.1, +11.7], p = 0.0007 — fails 10 %** |
| Team-reward degradation, with CI (secondary) | PARTIAL | computed from sidecar means, **unpaired**; needs the paired treatment |
| Survival ≥ 70 % | DONE | 98 % |
| Action-magnitude ratio ≤ 1.5 (stealth only) | DONE in code | no valid stealth run to apply it to; lie-only flies the honest policy bit-for-bit (ratio 1.0) |
| **Damage transfers across seeds, tested** | **MISSING** | never run in v2 |
| "Adversary that fails the gate is not usable — retune and retrain" | DECISION | D2 |

### Cross-seed transfer (design item 4)

**MISSING.** Needs each adversary evaluated on the other baseline seeds. Possible now for the two
seed-3 adversaries and the seed-1 lie-only adversary (about 12 runs, ~30 min); a full 3 × 3
matrix needs a seed-2 adversary (D4).

### Matched-damage triple + TOST (design item 5, spec test 9)

**MISSING, and currently undefined.** The spec's anchors (Tier 1 gaussian σ = 0.25 at "23.6 %",
Tier 2 waypoint α = 0.20 at "23.2 %") are legacy numbers. Waypoint α = 0.20 was not re-collected
in v2 (it lives in the `ALPHA_REFINE` grid, skipped in Stage C). No TOST code exists anywhere in
the repository. Needs decision D1 first.

### Tests (`tests/test_tier3.py`)

| # | spec test | status |
|---|---|---|
| 1 | frozen weights bit-identical | DONE |
| 2 | policy mapping | DONE |
| 3 | reward sign, r < −0.8 | DONE |
| 4 | crash penalty fires | DONE |
| 5 | stealth penalty fires | DONE |
| 6 | **checkpoint round-trip** (saved adversary reproduces its evaluation reward) | **MISSING** |
| 7 | range compliance and isolation | DONE |
| 8 | efficacy gate, all five criteria, in CI | PARTIAL — transfer criterion absent |
| 9 | **TOST on the matched triple, both directions** | **MISSING** |

### Rollout collection for Phase 3

**MISSING, and blocked:** the spec's distribution-shift factor (none / mild DR / severe DR + wind)
has no support in `rollout/collect.py`. It has to be built before those rollouts can run.

### Required contents of `tier3_design.md`

| item | status |
|---|---|
| 1 frozen-team mechanism | DONE (§1) |
| 2 reward design | PARTIAL — the required **wandb plots of each component** are not in the doc |
| 3 efficacy-gate table, all adversaries, all criteria | PARTIAL — spread across §3–6.23; needs one consolidated final table |
| 4 cross-seed 3 × 3 matrix | MISSING |
| 5 matched triple with TOST | MISSING |
| 6 stealth analysis with histogram | MISSING — but see §3 below: the key finding is already in |
| 7 behavioural observations (render 10 episodes, describe the strategy) | MISSING |
| 8 compute accounting | MISSING — data exists (below) |
| 9 test results | DONE (update the count: 54 pass, 1 skipped) |
| 10 surprises | DONE (throughout §3–6) |

Compute, from run-directory timestamps (to be confirmed against wandb runtimes): training
wall-clock per adversary was 32 h (seed1 stealth0), 32 h (seed1 stealth5), 16 h (f250),
16 h (stealth5 f250), 35 h (lie), 37 h (seed1 lie-only), 57 h (seed3 lie-only), 37 h (seed3
lie-only+ego) — about **262 GPU-hours** of training, before evaluation.

## 3. A finding spec item 6 says to raise immediately

The best adversary (seed-3 lie-only) flies the honest policy bit-for-bit, so **no action-level
detector can see it**. Its *broadcasts*, however, are physically impossible. Measured on 6,004
steps of the confirmatory run, room 10 × 10 × 10 m:

| its broadcast | value |
|---|---|
| reported position inside the room | **0 % of steps** |
| reported position below the floor | 76 % of steps |
| reported speed | 9.6 m/s (true 1.1 m/s) |
| lie at full deflection | 83–97 % of steps per axis |

A naive plausibility check on broadcast state catches it on every step. The spec's framing is
that the Tier 3 adversary defeats naive detectors; this one defeats an action-level detector
and loses to a bounds check.

## 4. Decisions needed

**D1 — recalibrate the rest of the scale.** Proposal: replace the Tier 1/2 20–40 % bands with a
family-level criterion (family damaging at p < 0.05, which `liar` and `stall` pass); set the
matched-triple target at Tier 3's confirmed effect (~7 %); and **declare a new TOST margin
before running TOST**. Precision, from the measured paired SE of a difference between two
attacks on identical scenarios (≈ 2.05 pp at n = 200):

| n per condition | 90 % CI half-width of a difference | usable margin |
|---|---|---|
| 200 | ± 3.4 pp | ± 5 pp (loose next to a 7 % effect) |
| 400 | ± 2.4 pp | ± 4 pp |
| 800 | ± 1.7 pp | ± 3 pp |

**D2 — the gate result.** At equal n the best hand-designed attack is +6.3 %, not ~10 %; the
10 % came from n = 50 screening. So the gate's *meaning* ("match the best hand-designed attack",
which the trained adversary does) and its *number* now disagree. Recommendation: keep 10 %,
because it was fixed before the Tier 3 result, and moving it after seeing +7.4 % would be
exactly the tuning it was designed to prevent. Report both facts. Separately decide whether to
accept this adversary as the paper's Tier 3 or retrain, as the spec requires for a gate failure.

**D3 — stealth.** No valid stealth-constrained adversary exists, and the working one is caught by
a bounds check (§3). Options: train a round with a plausibility constraint on the lie (reported
position inside the room, reported speed within the honest range); or accept lie-only as the
"action-stealthy" condition and report the broadcast-plausibility weakness as a limitation.
This decides the paper's framing.

**D4 — seed 2.** Train an adversary on seed 2 (36–57 h) for a full 3 × 3 transfer matrix, or
report transfer from the seed-1 and seed-3 adversaries only.

**D5 — Phase 3 distribution shift.** Build mild-DR and severe-DR-plus-wind support in the
collector, or descope.

## 5. Who does what

Can be done now without compute: TOST implementation with test 9 (both directions); checkpoint
round-trip test 6; transfer criterion in test 8; paired team-reward CI; switching the aggregate
band columns to the paired honest metric (value pending D1); compute table from wandb;
reward-component plots from TensorBoard; stealth-analysis figures.

Needs compute: cross-seed transfer (~30 min); the matched triple at the n chosen in D1; waypoint
α = 0.20 if it stays an anchor; rendering 10 episodes; any new training (D3, D4); Phase 3
rollouts once shift support exists.
