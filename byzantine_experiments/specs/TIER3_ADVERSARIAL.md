# TIER 3 — Trained Adversarial Policy (The Paper's Centrepiece)

**Audience:** cclaude code
**Prerequisite:** Tiers 0, 1, 2 complete with validation gates passed.

> **Amendments.** Changes decided after this spec was written are recorded in dated blocks
> next to the text they change; the original text is kept. Rationale and evidence for each are
> in `design/tier3_design.md` (section given); `design/tier3_audit.md` tracks what is complete.
>
> | id | date | what changed | design § |
> |---|---|---|---|
> | A1 | 2026-09-26 | Tier 1/2 prerequisite: 20–40 % bands → family-level validation (D1) | 6.25 |
> | A2 | 2026-09-24, kept 2026-09-26 | efficacy gate 30 % → 10 %, measured paired on the honest drones at n ≥ 200 (D2) | 6.20, 6.25 |
> | A3 | 2026-09-26 | stealth condition for the observation channel = the plausible lie (D3) | 6.26 |
> | A4 | 2026-09-26 | training matrix: 3 seeds × {free, plausible} lie-only (D3, D4) | 6.26 |
> | A5 | 2026-09-26 | matched triple: anchors, target, TOST margin and n pre-registered (D1) | 6.27 |
> | A6 | 2026-09-26 | Phase 3 shift levels defined by measured effect on the clean swarm (D5) | 6.28 |
> | A7 | 2026-10-12 | shift calibration grid extended (rules unchanged); four levels run and reported | 6.32, 6.33 |
>
> **Prerequisite status (A1).** Tier 1 is validated through `stall`, Tier 2 through `liar`,
> under the family-level criterion that replaced the bands.

---

## What this tier is for

Tier 3 is the tier that makes the paper publishable. Tiers 1 and 2 are hand-designed — a
reviewer will correctly point out that hand-designed attacks are easy to detect because the
designer implicitly leaked structure into them. Tier 3 removes that objection: the attacker
is a **learned policy trained specifically to maximise team damage**, with no hand-designed
behavioural signature at all.

This is the agent that will fool output-level anomaly detectors, and it is the agent your
spectral method must catch for the central claim to hold.

Method: freeze the honest cooperative team, replace one or two drones with policies trained
via PPO to minimise team reward, following Gleave et al., *Adversarial Policies*
(ICLR 2020, arXiv:1905.10615). Read that paper's Section 3 before writing code.

---

## The training setup

### Frozen team, trainable adversary

The critical structural requirement: **the honest agents' weights must not update during
adversarial training.** If they do, you are doing self-play and the honest policy will adapt,
which changes the baseline you spent Phase 1 establishing.

In Sample Factory this means a mixed-policy rollout: agents `0..k-1` are the trainable
adversary, agents `k..7` load from the frozen baseline checkpoint and run in inference mode
with gradients disabled. Investigate Sample Factory v2.1.1's multi-policy support
(`--num_policies`) — it may support this natively via policy-to-agent mapping. If it does,
use it. If it does not, implement a wrapper that loads the frozen checkpoint and injects its
actions for the honest indices, exposing only the adversary's slice to the learner.

Document which path you took and why. This is the highest-risk engineering decision in the
tier — budget two days for it and do not proceed to the reward design until a smoke test
confirms the honest agents' weights are genuinely static (assert checkpoint hash unchanged
after 1000 training steps).

### Reward design

Base adversarial reward:

```
r_adv = -w_team * r_team_mean  -  w_crash * I[adversary_crashed]  +  w_alive * I[alive]
```

Three components, three jobs:

- `-w_team * r_team_mean` — the actual objective, negated team performance
- `-w_crash * I[crashed]` — prevents the degenerate solution where the adversary immediately
  suicides into a neighbour. That is technically an effective attack but it is a *detectable*
  and uninteresting one.
- `+w_alive` — small per-step survival bonus, encourages persistent disruption over one-shot
  sacrifice

Start with `w_team=1.0`, `w_crash=10.0`, `w_alive=0.1` and tune. Log all three components
separately in wandb so you can see which one is driving learning.

### The stealth constraint — this is what makes the paper

A naive adversary will learn to produce wild, high-magnitude actions. Those are trivially
detectable by an action-magnitude threshold, which would let a reviewer dismiss your method
as unnecessary. To make Tier 3 a genuine test, constrain the adversary to look statistically
ordinary at the action level:

```
r_adv -= w_stealth * max(0, ||a_adv - mu_honest||_2 - epsilon)
```

where `mu_honest` and the action-distribution statistics come from your clean baseline
rollouts. Config: `epsilon` ∈ {0.2, 0.5, 1.0} (attack budget), `w_stealth` ∈ {0, 1, 5}.

Train at minimum two settings:

- **Unconstrained** (`w_stealth=0`) — the easy adversary, shows the attack works at all
- **Stealth-constrained** (`w_stealth=5`, `epsilon=0.3`) — the hard adversary, the paper's
  headline condition

The stealth-constrained adversary is the one where a raw-action-statistics baseline should
fail and your spectral method should succeed. **If your method also fails on this condition,
that is the paper's honest negative result and you must report it** — do not tune the
constraint until the method wins.

> **Amendment A3 — 2026-09-26 (decision D3). The stealth condition for the observation
> channel.** The action-space penalty above is written for an adversary that attacks through
> its motors. Rounds 1–2 did, and crashed (survival 0 %); both `w_stealth = 5` runs were
> defective (`design/tier3_design.md` §4.2, §4.4). The adversary that works attacks through the
> **observation channel**: it flies the honest policy and lies in the state it broadcasts
> (lie-only). Its action ratio is 1.10, so the action-space constraint is met without being
> imposed — but its broadcast is physically impossible (outside the room on 100 % of steps,
> below the floor on 76 %, 9.6 m/s against a true 1.1 m/s) and a naive bounds check catches
> it in every episode (`results_v2/tier3_report/stealth_adv_seed3_stealth0_lieonly.md`).
>
> The stealth-constrained condition is therefore redefined on the channel the adversary
> actually uses: a **plausible lie** (`adversary/lies.py::PlausibleLie`). The adversary
> steers only the *acceleration* of a smooth drift between claim and truth, so its broadcast
> passes, by construction on every step, a bounds check (inside the room), a speed check (≤
> truth + 1 m/s), and a kinematic-consistency check (reported position moves as reported
> velocity says); its acceleration stays within 4 m/s² of the truth except on rare
> wall-limited steps, which are counted and reported. Budget, the analogue of `epsilon`: offset
> ≤ 2 m (ordinary GNSS error), drift ≤ 1 m/s, drift acceleration ≤ 4 m/s² (honest p50 1.2,
> p90 4.6). The action-magnitude criterion (≤ 1.5×) is still applied.
>
> The fallback rule below is unchanged: if the plausible adversaries fail, the free lie-only
> adversary is the paper's Tier 3 and the stealth constraint is reported as a limitation.

### Training configuration

Reuse the baseline hyperparameters from `scripts/train_exp1.sh` where possible. Key
differences: shorter horizon (adversarial policies converge faster — try `2e8` env steps
first, not `1e9`), and a lower learning rate if training is unstable.

One adversary per baseline seed:

```
train_dir/adv_seed1_stealth0/
train_dir/adv_seed1_stealth5/
train_dir/adv_seed2_stealth0/
...
```

Six training runs total (3 seeds × 2 stealth settings). At ~12–24h each, budget 3–5 days
wall clock if you can run them in parallel on the cluster.

> **Amendment A4 — 2026-09-26 (D3, D4).** The six runs are 3 seeds × {free lie-only,
> plausible lie-only}: `adv_seed{1,2,3}_stealth0_lieonly` and
> `adv_seed{1,2,3}_stealth0_lieonlyplaus`. Seed 1 and 3 free runs exist; seed 2 free (D4) and
> the three plausible runs (D3) are to be trained. The motor-channel runs of rounds 1–5 stay
> in the efficacy table as the record of how the channel was found. Measured wall clock was
> 16–57 h per 200 M-step run on the one T4 available (`results_v2/tier3_report/compute.md`).

Script: `scripts/byz_train_adv.sh` taking seed and stealth setting as arguments.

---

## Efficacy gate — do not skip this

Before any of these adversaries enter the detection experiments, each must pass:

| Criterion | Threshold |
|-----------|-----------|
| **Goal-RMSE degradation** (primary) | **≥ 30% vs clean baseline** |
| Team reward degradation (secondary, report with CI) | ≥ 30% vs clean baseline |
| Adversary survival rate | ≥ 70% of episodes |
| Action magnitude ratio vs honest | ≤ 1.5× (stealth condition only) |
| Damage transfers across seeds | tested — see below |

> **Amendment A2 — gate recalibrated 2026-09-24, kept 2026-09-26 (decision D2).** The primary
> threshold is **≥ 10 %**, not 30 %: the 30 % was set on legacy numbers the seeding bug
> inflated ~3×; 10 % matched the best hand-designed family at the time (`liar`: mean +5.7 %,
> strongest seed +9.6 %, best run +13.1 %; §6.20). It is
> measured on the **honest drones only**, **paired** per episode against clean at the same
> seed, on the adversary's **final checkpoint**, with **n ≥ 200** and the paired 95 % CI
> above 0 (n = 50 screening rows are candidates, never passes). Team reward is reported with
> its paired CI and **not gated** (the note below explains why it cannot gate).
>
> D2: at equal n (200, same scenarios) the best hand-designed attack measures +6.3 %, so the
> gate's *meaning* ("at least as damaging as the best hand-designed attack") and its *number*
> now disagree. The number was fixed before any v2 Tier 3 result and is kept; both facts are
> reported. The best adversary so far measures +7.4 % [+3.1, +11.7]: it **fails** the gate and
> is statistically indistinguishable from the best hand-designed attack.
>
> "Not usable" is read per purpose. Passing the gate classifies an adversary as *stronger than
> hand design*. The matched triple needs something else — *equal* damage across tiers — and
> takes its Tier 3 member by the pre-registered rule in A5 (significant damage, CI > 0), not by
> the gate: an adversary above the hand-designed ceiling could not be matched to Tier 1/2 at
> all. All five criteria for every adversary: `results_v2/tier3_efficacy.md`.

> **Why goal-RMSE is the primary criterion.** Measured on the Tier 1 and Tier 2 matrices
> (180 runs, see `design/tier2_design.md` §10), the *reward* form of this criterion does not
> discriminate. Clean mean episode reward in this env is only **+0.861**, and any config that
> damages its own drone drives the 8-drone mean negative, so the percentage explodes:
>
> | config | all-8 episode reward | "team reward degradation" |
> |---|---|---|
> | T1 gaussian σ=0.25 (**a fault, not an attack**) | −0.946 | **214%** |
> | T1 OU σ_ou=0.2 | −3.890 | **663%** |
> | T2 waypoint α=1.0 | −3.618 | **602%** |
> | T2 collusion pincer | −8.094 | **1273%** |
>
> A Tier 1 *noise fault* passes a ≥30% reward gate at 214%. The criterion therefore admits
> everything and separates nothing. Goal-RMSE degradation (RMS distance to assigned goal) has a
> true zero and a seed-to-seed spread of 3–12 pp rather than ±94–112 pp, so it can actually
> gate. Keep the reward figure in the table for continuity, but read it with its CI and never
> rank on it.
>
> The other three criteria **are** discriminating and need no change — Tier 2's sign-flip fails
> the ≥70% survival bar (45–73%) while waypoint passes it (91–100%).

**Cross-seed transfer test.** Take the adversary trained against `seed1` and evaluate it
against `seed2` and `seed3`. If the attack transfers, you have a general adversary and the
result is much stronger. If it does not transfer, you have a seed-specific exploit — still
publishable, but the paper must say so explicitly. Either outcome is a result; report it.

An adversary that fails the efficacy gate is not usable. Retune the reward weights and
retrain rather than proceeding with a weak attack.

---

## Damage matching against Tiers 1 and 2

This is the experimental design point that determines whether your central claim is testable.

From Tier 1 and Tier 2 you have severity calibration tables. Select configurations from each
tier that produce **statistically equivalent goal-RMSE degradation** — a matched triple.

Using the measured calibration tables (`results/tier1_calibration.md`,
`results/tier2_calibration.md`), the ~25% band gives:

```
Tier 1 (degraded):    tier1_gaussian, sigma=0.25,  f=0.125  →  23.6% goal-RMSE degradation
Tier 2 (structured):  tier2_waypoint, alpha=0.20,  f=0.125  →  23.2%      (gap: 0.4 pp)
Tier 3 (adversarial): adv_seedN_stealth5,          f=0.125  →  tune to land in 20-40%
```

> **Amendment A5 (part 1) — 2026-09-26 (D1). These anchors are legacy numbers and are
> replaced.** On the v2 corpus (honest drones, paired): gaussian has no family-level effect,
> and waypoint α = 0.20 was not re-collected. The triple is selected by a rule fixed before any
> triple data (`analysis/matched_triple.py` docstring): **Tier 3 member** = the plausible
> adversaries if their pooled own-seed damage is significant, else the free lie-only
> adversaries (the fallback rule); **target** = that member's pooled own-seed damage (n = 200
> per seed); **Tier 1 / Tier 2 members** = from the validated families (`stall`, `liar`) at
> f = 0.125, the configuration whose pooled paired damage is closest to the target; every
> member must also be significantly damaging on its own.

**Use `waypoint α=0.20`, not `signflip`, as the Tier 2 anchor.** Both are in band, but
sign-flip spends 66–83% of the episode inside the survival guard and only survives 69% of
episodes, so it is substantially "honest flight with a safety net". Waypoint at α=0.20 has 96%
survival and 15% guard exposure — a real attack that stays airborne.

Alternative Tier 1 anchors in the same band, if a different fault mechanism is wanted:
`tier1_ou σ_ou=0.06` (35.3%) and `tier1_stall duty=0.25/0.3 s` (25.5%).

> **A warning for tuning the Tier 3 stealth budget.** Damage is *not* proportional to
> perturbation in this env. For `waypoint`, action deviation is linear in α while damage has a
> cliff between α = 0.15 and α = 0.30 (11.2% → 23.2% → 77.8% for deviations of 0.206 → 0.259 →
> 0.294). A 26% increase in perturbation doubles the damage. Expect `epsilon` to behave the
> same way, and concentrate its grid near the threshold rather than spreading it uniformly —
> and expect a damage-maximising adversary to converge toward exactly that cliff.

> **Do not use the reward-based degradation figures for matching**, and note that the
> illustrative numbers in earlier drafts of this spec (`tier1_ou σ_ou=0.2 → 34%`) assumed a
> reward scale this environment does not have. That config actually measures **663% ± 111** on
> reward and **60.9%** on goal RMSE.

### Verifying the match — use an equivalence test, not a null test

Declare the equivalence margin **before** looking at the data:

> **Two conditions are "matched" if their goal-RMSE degradation differs by no more than
> ±10 percentage points.**

Then test it with **TOST** (two one-sided t-tests) on the per-episode goal-RMSE degradations,
pooled across the 3 seeds. Equivalence is established when *both* one-sided tests reject at
α = 0.05, i.e. the 90% CI of the difference lies entirely inside ±10 pp.

> **Amendment A5 (part 2) — 2026-09-26, declared before any triple data (D1).** ±10 pp is
> larger than every effect in the v2 corpus: the clean swarm (0 %) would be "equivalent" to a
> 7 % Tier 3 attack. The margin is **±min(3 pp, ½ × target)** — the half-target cap stops a
> near-harmless condition passing as matched to a small effect. The n per condition per seed
> is the smallest of 400 / 800 / 1600 giving ≥ 80 % TOST power for two truly equal conditions,
> from the measured per-episode SD of a difference between two attacks on identical scenarios
> (29.0 pp): ±3 pp → 400, ±2 pp → 800, ±1.5 pp → 1600; beyond that the triple is reported as
> underpowered instead of run. Pairwise comparisons are per scenario (all conditions at one
> `--seed` per baseline seed). Test 9 uses this margin.

**Do not use "Mann-Whitney U, p > 0.05" to establish indistinguishability.** A non-significant
result from an underpowered test is absence of evidence, not evidence of equivalence — and at
n = 50 this test is severely underpowered. Per-episode reward has σ ≈ 3.31 against a clean mean
of 0.861, so the minimum detectable difference at 80% power is

```
2.8 · σ / √n  =  2.8 · 3.31 / √50  =  1.31 reward units  =  152% of the clean mean
```

Any two conditions differing by less than ~152 percentage points of reward degradation would
return p > 0.05 and be declared "indistinguishable" — including a genuinely mismatched pair
like 34% vs 180%. Raising n helps only slowly (n=200 → 76 pp, n=400 → 54 pp, n=800 → 38 pp).

Goal-RMSE degradation avoids this: its seed-to-seed spread is 3–12 pp, so a ±10 pp equivalence
margin is resolvable at the 50 episodes × 3 seeds already being collected. If the reward metric
must also be shown equivalent, collect **n ≈ 400 episodes for the matched triple only**
(3 conditions × 3 seeds × 400 ≈ 2 h of rollout — negligible beside the 6 training runs).

This matched triple is the paper's central experiment. Same damage, three different causal
mechanisms. If your detector separates them, you have shown that spectral policy-graph
analysis recovers *mechanism*, not just *severity* — which is precisely the claim no existing
method supports.

Without matched damage, a reviewer will say your detector is just a damage meter. With it,
that objection is closed — **provided the matching is established by an equivalence test.**
Matching asserted from a failed null test is exactly the kind of thing a statistically
literate reviewer will catch.

---

## Tests — `byzantine_experiments/tests/test_tier3.py`

1. **Frozen weights**: honest policy parameters bit-identical before/after 1000 adversarial
   training steps
2. **Policy mapping**: agent indices map to the correct policy (adversary vs frozen)
3. **Reward sign**: adversary reward is negatively correlated with team reward (r < -0.8 over
   a rollout)
4. **Crash penalty fires**: penalty appears in the log when the adversary crashes
5. **Stealth penalty fires**: penalty is nonzero when action deviation exceeds `epsilon`
6. **Checkpoint round-trip**: saved adversary loads and reproduces its evaluation reward
7. **Range compliance and isolation** (as prior tiers)
8. **Efficacy gate**: automated check of all five gate criteria, run as part of CI
9. **Equivalence test**: TOST on the matched triple returns equivalence within the
   declared ±10 pp goal-RMSE margin (and fails on a deliberately mismatched pair — test
   both directions, or the test proves nothing)
   *(Amended A5: the declared margin is ±min(3 pp, ½ × target).)*

Test 1 is the one that catches silent self-play. Write it first and run it on every training
launch.

---

## Rollout collection for Phase 3

Once adversaries pass the gate, collect trajectories using the Tier 0 collector, matching the
Tier 1/2 format exactly so Phase 3 can process all tiers with one code path.

| Factor | Values |
|--------|--------|
| Baseline seed | 1, 2, 3 |
| Adversary | matching-seed and cross-seed |
| Stealth | 0, 5 |
| f | 0.125, 0.25 |
| Distribution shift | none, mild DR, severe DR + wind |
| Episodes | 50 |

Prioritise the matched-damage triple × 3 shift levels × 3 seeds first — that subset alone
produces the paper's main table. Run the rest afterward if time permits.

> **Amendment A6 — 2026-09-26 (decision D5). Shift levels, defined by their effect.** Built as
> environment wrappers (`wrappers/shift.py`; the simulator package is not modified): dynamics
> randomisation switches on the simulator's own relative sampler (mass, inertia, thrust, …
> resampled every episode by a ratio), and wind adds a uniform mean acceleration plus an
> Ornstein–Uhlenbeck gust to every drone each control step. Both are seeded per episode, so a
> shifted run still pairs episode-for-episode with a clean run at the same shift level. Levels
> are chosen by a rule fixed before calibration (`analysis/shift_calibration.py`), on the clean
> swarm's paired honest goal-RMSE degradation, pooled over seeds 1–3:
> **mild** = randomisation only, the smallest ratio in {0.05, 0.1, 0.2, 0.3} with an effect
> ≥ 5 % (p < 0.05); **severe** = randomisation at min(2 × mild, 0.3) plus the smallest mean wind
> in {0.5, 1, 2} m/s² with a combined effect ≥ 25 % (p < 0.05), gust SD = half the mean wind,
> correlation time 1 s. Settings beyond the grid are flagged, not extrapolated; any level with
> an effect > 100 % is flagged as destroying the task. Priority subset:
> `scripts/byz_phase3.sh` (every attacked run paired with clean at the same seed and shift).

> **Amendment A7 — 2026-10-12. The grid, not the rules.** A6's grid started too high: its
> smallest ratio (0.05) already degraded the clean swarm by +32.6 % and its severe pick by
> +140 %. Decided before any attacked run under shift existed: the DR grid adds
> {0.01, 0.02, 0.03, 0.04} and stage-2 wind is {0.1, 0.25, 0.5, 1.0}; thresholds unchanged. Its
> picks are the primary **mild** = DR 0.03 (+9.6 %) and **severe** = DR 0.06 + wind 0.1 (+60.1 %);
> A6's picks are kept as **mild_v1** (+32.6 %) and **severe_v1** (+140 %). All four levels are
> collected and reported, ordered by effect.

---

## Failure modes and what to do

| Symptom | Likely cause | Response |
|---------|--------------|----------|
| Adversary reward flat, no learning | Honest agents not actually frozen, or reward not wired | Run test 1 and test 3 |
| Adversary crashes immediately every episode | `w_crash` too low | Raise to 20–50 |
| Adversary does nothing (< 10% degradation) | Stealth constraint too tight, or LR too low | Train unconstrained first, then add stealth |
| Degradation > 80% | Attack too strong to be interesting | Reduce `epsilon`, raise `w_stealth` |
| Attack does not transfer across seeds | Seed-specific exploit | Report it — this is a finding, not a failure |
| Training unstable / diverges | LR too high for adversarial objective | Halve LR, shorten rollout |

If after two days of tuning the stealth-constrained adversary still fails the efficacy gate,
fall back to the unconstrained adversary as the paper's Tier 3 and document the stealth
constraint as a limitation. A working unconstrained adversary beats a broken stealthy one.

---

## Required output: `byzantine_experiments/design/tier3_design.md`

Include:

1. **The frozen-team mechanism** — how you achieved it in Sample Factory v2.1.1, whether you
   used native multi-policy support or a custom wrapper, and the evidence from test 1 that
   the honest weights are genuinely static.
2. **Reward design as implemented** — final weights, the tuning path that got you there, and
   wandb plots of each reward component over training.
3. **Efficacy gate results** — the full table for all six adversaries against all four
   criteria.
   *(Amended A2: all adversaries against all five criteria, one verdict each —
   `analysis/efficacy_table.py`, `results_v2/tier3_efficacy.md`.)*
4. **Cross-seed transfer results** — the 3×3 matrix of adversary-seed vs baseline-seed
   degradation. State plainly whether attacks transfer.
5. **The matched-damage triple** — the exact Tier 1 / Tier 2 / Tier 3 configurations selected,
   their **goal-RMSE** degradation percentages with confidence intervals, and the **TOST**
   result against the pre-declared ±10 pp equivalence margin (90% CI of each pairwise
   difference). Report reward degradation alongside, with its CI, as a secondary figure — and
   state explicitly that it was not used for matching.
6. **Stealth analysis** — action-magnitude and action-distribution statistics for the
   stealthy adversary vs honest agents. Include a histogram overlay. If a naive
   action-magnitude detector *can* still catch the stealthy adversary, say so — that weakens
   the paper's framing and the researcher needs to know immediately.
7. **Behavioural observations** — render 10 episodes. What does the learned attack actually
   *do*? Describe the strategy in plain language. This description is likely to become one of
   the most quoted sentences in your paper.
8. **Compute accounting** — GPU-hours per run, total, and wall clock.
9. **Test results** with empirical numbers.
10. **Anything that surprised you.** For Tier 3 especially, the learned attack strategy will
    probably not be what anyone predicted. Write down what it did.
