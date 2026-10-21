# TIER 1 — Stochastic Faults (The Negative Control)

**Audience:** claude code
**Prerequisite:** `TIER0_INFRASTRUCTURE.md` complete and its validation gate passed.

> **Amendments.** Changes decided after this spec was written are recorded in dated blocks
> next to the text they change; the original text is kept. **A1 (2026-09-26, decision D1):**
> the 20–40 % band is replaced by family-level validation — see *Severity calibration* and
> *Validation gate* below.

---

## What this tier is for — read this before coding

Tier 1 is **not an attack**. It models an honest drone whose behaviour has degraded because
of hardware failure, sensor noise, or environmental interference. It has no goal, no
strategy, and no model of the team.

This is the **negative control** for the entire research project. The paper's central claim
is that spectral analysis of policy graphs can distinguish _degraded_ agents from _malicious_
agents. Tier 1 is the "degraded" side of that comparison. If Tier 1 is implemented sloppily —
if it accidentally produces goal-directed behaviour, or if it is so destructive that it looks
like an attack — the central claim becomes untestable.

**The design target: Tier 1 agents should be detectable as anomalous, but should NOT be
classified as Byzantine.** A good detector flags them as "something is wrong with agent 3"
while correctly declining to flag them as adversarial.

---

## Implementation

Create `byzantine_experiments/agents/tier1_stochastic.py`. Implement three variants, all
registered in `BYZANTINE_REGISTRY`.

### Variant 1A — Additive Gaussian noise (`tier1_gaussian`)

The primary variant. Zero-mean isotropic noise on the honest action.

```
a' = clip(a + N(0, sigma^2 * I), low, high)
```

Config: `sigma` ∈ {0.1, 0.25, 0.5}. Use the wrapper-provided RNG, never global `np.random`.

Note the interaction with clipping: at `sigma=0.5` on a `[-1,1]` action space a large
fraction of samples saturate, which changes the effective noise distribution and _does_
introduce a systematic bias toward the boundary. Measure the saturation rate and report it —
if it exceeds ~30% at your highest sigma, that variant is no longer a clean "unbiased noise"
control and you must say so in the design doc.

### Variant 1B — Temporally correlated drift (`tier1_ou`)

Ornstein-Uhlenbeck noise. More physically realistic than white noise, because real sensor
drift and wind gusts are correlated over time.

```
x_{t+1} = x_t + theta * (mu - x_t) * dt + sigma_ou * sqrt(dt) * N(0, I)
a' = clip(a + x_t, low, high)
```

Config: `theta` (mean reversion, default 0.15), `sigma_ou` (default 0.2), `mu` (default 0),
`dt` (default 1.0). Reset `x` to zero on `reset()`.

This variant matters more than it looks. White noise produces policy graphs with high
transition entropy — nearly uniform random walks between abstract states. OU noise produces
_sticky_ degradation, which is behaviourally closer to what a real failing drone does and is
a harder negative control for your detector. Expect 1B to be harder to distinguish from Tier 3
than 1A. That is the point.

### Variant 1C — Intermittent dropout (`tier1_dropout`)

With probability `p_drop` per step, the agent emits a hold action (repeat previous action) or
a zero action, simulating a communication/compute stall.

Config: `p_drop` ∈ {0.05, 0.15, 0.3}, `mode` ∈ {`hold`, `zero`}.

---

## Hard design constraints

These are what keep Tier 1 an honest negative control. Violating any of them invalidates the
experiment.

1. **No access to team state.** The perturbation may depend on the agent's own action and
   its own observation, and on time. It must never depend on other agents' positions, the
   team reward, or the goal location. If the code reads a neighbour's state, it is no longer
   Tier 1.
2. **Zero-mean in expectation** (1A and 1B). Verify empirically: over 10,000 steps, the mean
   perturbation per action dimension should be within 2 standard errors of zero. Assert this
   in a test.
3. **No temporal goal structure.** 1B has temporal correlation, which is fine — correlation
   is not intent. But there must be no drift _toward_ anything meaningful.
4. **Honest broadcast.** Do not override `perturb_broadcast_state()`. A degraded drone
   reports its true state; it is simply flying badly. Lying about state is Tier 2's job.

---

## Experimental matrix

No training. Evaluate the three existing baseline checkpoints.

| Factor                   | Values                                                                            |
| ------------------------ | --------------------------------------------------------------------------------- |
| Baseline seed            | `exp1_baseline_seed{1,2,3}`                                                       |
| Variant                  | `tier1_gaussian`, `tier1_ou`, `tier1_dropout`                                     |
| Severity                 | 1A: σ ∈ {0.1, 0.25, 0.5} · 1B: σ_ou ∈ {0.1, 0.2, 0.4} · 1C: p ∈ {0.05, 0.15, 0.3} |
| f (compromised fraction) | 0.125 (1 of 8), 0.25 (2 of 8)                                                     |
| Episodes per run         | 50                                                                                |

That is 3 × 3 × 3 × 2 = 54 rollout runs at ~5–10 min each. Batch them in one script:
`scripts/byz_rollout_tier1.sh`.

---

## Characterisation you must produce

Tier 1's value is as a calibrated reference point, so it needs measurement, not just
implementation. For every configuration, record:

- **Mean episode reward** vs the clean baseline (from `logs/seed{1,2,3}_eval_final.json`)
- **Team reward degradation %** — this is the key calibration number
- **Collision rate** vs clean
- **Formation RMSE** vs clean
- **Mean L2 action deviation** of the compromised agent
- **Action saturation rate** (fraction of perturbed actions hitting the clip boundary)

Then produce a **severity calibration table** mapping each config to its team degradation.
You need this because Tier 3's adversary must be tuned to cause _comparable_ team damage —
otherwise your detector might be separating "high damage" from "low damage" rather than
"malicious" from "degraded," which is a confound a reviewer will catch immediately.

Identify and flag the Tier 1 configurations that produce **20–40% team reward degradation**.
Those are the ones that will be matched against Tier 3 in the paper's headline comparison.

> **Amendment A1 — 2026-09-26 (decision D1). The 20–40 % band is retired.** It was set on
> numbers from a collector with two seeding bugs (runs at one `--seed` did not share
> scenarios; `design/tier3_design.md` §6.16–6.18), which inflated degradations roughly 3×. On
> the re-collected corpus (`results_v2/`), measured on the honest drones and paired per episode
> against clean, **no Tier 1 configuration reaches 20 %** (best family mean +2.6 %, best single
> run +12.6 %), and at n = 50 a single configuration's significance barely beats chance.
>
> **Replacement.** A fault family is *validated* when its runs together significantly damage
> the honest team: one-sample t-test over the per-run paired honest goal-RMSE degradations,
> mean > 0, p < 0.05 (`analysis/family_validation.py`; table in
> `results_v2/tier1_calibration.md`). Result: **`stall` validated** (+2.6 %, 21 of 27 runs
> positive, p = 0.0004); gaussian, OU and dropout have no family-level effect on the honest
> drones — the damage of those faults stays on the faulty drone, which is itself a finding
> about the baseline's robustness, not a failure of the tier.
>
> The configuration matched against Tier 3 is not chosen from a band but by the pre-registered
> matched-triple rule (`analysis/matched_triple.py`): from a validated family, at f = 0.125,
> the configuration whose pooled paired damage is closest to Tier 3's confirmed effect.
>
> **Clarification — 2026-10-21.** "Inflated roughly 3×" above is true of the *honest-drone*
> effects, not of every Tier 1 figure. On the band's original axis — goal-RMSE degradation over
> all 8 drones, dominated by the faulty drone — the v2 numbers replicate the legacy ones (20 of 29
> configs within 5 pp), and five v2 configurations still fall in 20–40 %. The retirement stands for
> the reason that holds on v2 data: tiers are matched on the honest drones, where no Tier 1
> configuration comes near 20 % (`design/tier1_design.md` §0).

---

## Tests — `byzantine_experiments/tests/test_tier1.py`

1. Zero-mean: empirical mean perturbation ≈ 0 for 1A and 1B (2-SE tolerance)
2. Variance scaling: empirical std scales linearly with configured `sigma` (pre-clipping)
3. Determinism: same seed → identical perturbation sequence
4. Range compliance: all outputs within `action_space`
5. Isolation: uncompromised agents unaffected
6. OU reset: internal state zeroed on `reset()`
7. No team-state access: monkeypatch/assert the agent never reads other agents' obs slices
8. Dropout rate: empirical dropout frequency matches `p_drop` within tolerance

Test 7 is the important one. Implement it by passing a sentinel observation and asserting the
output depends only on the ego-agent slice.

---

## Validation gate

Tier 1 is done when all eight tests pass, the 54-run matrix has completed, and the severity
calibration table exists with at least one configuration per variant landing in the 20–40%
degradation band.

> **Amended (A1, 2026-09-26):** "at least one configuration per variant in the 20–40 % band"
> is replaced by "at least one fault family validated at the family level, with the table
> reporting every family's status". **Met:** `stall`. Families without a family-level effect
> are reported as such.

---

## Required output: `byzantine_experiments/design/tier1_design.md`

Include:

1. **The three variants as implemented** — exact update equations, parameter defaults, and
   the reasoning behind any default you chose yourself.
2. **Severity calibration table** — full matrix of config → team degradation %, collision
   rate, formation RMSE, mean action deviation, saturation rate.
3. **The 20–40% band** — which configurations land there, and your recommendation for which
   single config per variant should be the paper's canonical Tier 1 setting.
   *(Amended A1: the family-level validation table and the matched-triple candidates, i.e.
   validated-family configurations at f = 0.125, replace the band.)*
4. **Clipping analysis** — saturation rates at each sigma, and whether zero-mean holds
   post-clipping. Be explicit if it does not.
5. **Behavioural observations** — render 5 episodes per variant and describe in plain
   language what the degraded drone looks like. Does it wobble? Drift? Freeze? This
   qualitative description feeds the paper's Section 5.
6. **Test results** — all eight, with the empirical numbers not just pass/fail.
7. **Anything that surprised you.** If a variant behaved unexpectedly, say so. That is data.
8. **Wandb logging** — All results should be logged to wandb if necessary. See file README.md for details like wandb project name, entity, group, job_type, tags etc. The last thing I want is to have to dig through a bunch of JSON logs to find the numbers I need for the paper. Log even those that you think might not be useful now. The use of wandb is to make it easier to find the numbers later.
