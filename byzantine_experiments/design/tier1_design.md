# Tier 1 — Stochastic Faults: design notes

Implementation record for [specs/TIER1_STOCHASTIC.md](../specs/TIER1_STOCHASTIC.md).
Branch `feature/byzantine`. Built on the Tier 0 infrastructure
([tier0_design.md](tier0_design.md)); no upstream files were modified for Tier 1.

---

## 1. The variants as implemented

All four live in [`agents/tier1_stochastic.py`](../agents/tier1_stochastic.py) and are
registered in `BYZANTINE_REGISTRY`. All draw from `self.rng` — the per-agent stream
the wrapper seeds as `seed + 1000 * agent_idx` — never from global `np.random`, so two
compromised drones get independent noise and a run is reproducible from `--seed` alone.

None of them reads `obs`. That is the strongest possible form of hard constraint 1 ("no
access to team state"): there is no code path by which team state could influence the
output, which is what `test_no_team_state_access` verifies by feeding a 1e6-magnitude and
then a NaN-poisoned neighbour block and requiring bit-identical actions.

### 1A — `tier1_gaussian`: additive white Gaussian noise

```
a' = clip(a + N(0, σ² I),  low, high)
```

Severity: `sigma` ∈ {0.1, 0.25, 0.5} (matrix), plus {0.30, 0.35, 0.40} from the refinement
grid (§3, marked ᴿ). Memoryless and zero-mean — the baseline against which 1B's temporal correlation
is the interesting contrast.

### 1B — `tier1_ou`: temporally correlated drift (Ornstein–Uhlenbeck)

```
x_{t+1} = x_t + θ (μ − x_t) Δt + σ_ou √Δt · N(0, I)
a'      = clip(a + x_t,  low, high)
```

| Parameter | Default | Why |
| --- | --- | --- |
| `theta` | 0.15 | The standard OU default, and the same mean-reversion the simulator's own thrust-noise process uses (`quad_utils.py::OUNoise`). Chosen for consistency with the noise already in the env rather than arbitrarily. |
| `sigma_ou` | 0.2 | Matrix values {0.1, 0.2, 0.4}; refinement adds {0.04, 0.06, 0.08}. |
| `mu` | 0.0 | Any non-zero μ is a directional bias and would violate constraint 3 ("no temporal goal structure"). Correlation is not intent; a drift *target* would be. |
| `dt` | 1.0 | Parameters are per **env step**, not per second. The env runs at 100 Hz, so θ=0.15 per step is a ~6.7-step (67 ms) relaxation time. |

**σ_ou is not comparable to 1A's σ by number.** The discretised process has stationary
standard deviation

```
σ_stationary = σ_ou · √( Δt / (1 − (1 − θΔt)²) ) = σ_ou · 1.898   for the defaults
```

so `sigma_ou=0.1` already injects noise of std ≈0.19, about double what the number suggests.
`OUNoiseAgent.stationary_std()` exposes this, and `test_variance_scales_with_sigma` checks
the empirical std against it (measured 0.1918 / 0.3835 / 0.7670 vs analytic 0.1898 / 0.3797
/ 0.7593). This is the single most important thing to know when reading the severity table:
**1B's severities are roughly 1.9× their nominal value**, and that is most of why 1B looks so
much stronger than 1A at equal-looking numbers.

### 1C — `tier1_dropout`: intermittent stall

With probability `p_drop` per step:

```
mode='hold':  a' = previous emitted action     (frozen controller)
mode='zero':  a' = 0                           (dead output; 0 = hover, since the action
                                                space is zero-action-middle)
```

Severity: `p_drop` ∈ {0.05, 0.15, 0.3} (matrix), plus {0.6, 0.85, 0.98} for `hold` and
{0.5, 0.85} for `zero` from the refinement grid. `mode` defaults to `hold`, the more
physically realistic stall; the matrix sweeps `hold` and adds a `zero` comparison at
p=0.15.

Unlike 1A/1B this is **not zero-mean** — a hold or a zero is a systematic departure, not a
symmetric jitter — which is why the zero-mean test covers only 1A and 1B. It is still a
fault rather than an attack: the stall times are i.i.d. and depend on nothing.

`reset()` clears the held action to zero rather than to whatever the drone was doing at the
end of the previous episode, so no state leaks across an episode boundary
(`test_ou_state_reset` covers this for both 1B and 1C).

### 1C′ — `tier1_stall`: dropout re-parameterised by stall *duration*

Added after the matrix showed 1C to be a near-null fault (§8). A two-state Markov chain over
{OK, STALL}:

```
OK    --p_enter-->   STALL          p_recover = 1 / (mean_stall_s · f_ctrl)
STALL --p_recover--> OK             p_enter   = p_recover · d / (1 − d)
```

which makes stationary occupancy exactly `duty_cycle` and mean stall length exactly
`mean_stall_s`. Emission during a stall is the same as 1C (`hold` freezes the last *fresh*
command; `zero` emits hover).

| Parameter | Default | Why |
| --- | --- | --- |
| `mean_stall_s` | 0.5 | How long a stall lasts, in **seconds**. Grid {0.1, 0.3, 1.0}. This is the knob 1C was missing: a real radio dropout or compute hang lasts hundreds of ms, not one control tick. |
| `duty_cycle` | 0.25 | Long-run fraction of steps stalled. Grid {0.10, 0.25, 0.50}. Plays the role `p_drop` was *intended* to play. |
| `mode` | `hold` | As 1C. |
| `control_freq` | 100.0 | Env control rate, used to convert seconds to steps. Exposed rather than hardcoded so the variant survives a change of control rate. |

**The two variants are nested, not rival.** Setting `mean_stall_s = 1/((1−p)·f_ctrl)` and
`duty_cycle = p` reproduces `tier1_dropout` exactly — verified empirically in
`test_stall_reduces_to_iid_dropout_in_the_limit` (measured 1.43-step stalls at p=0.3 against
a theoretical 1.43). So the original 1C results remain valid as the short-stall limit of 1C′
rather than being superseded.

### A deliberate deviation: `reset()` does not reseed

`ByzantineAgent.reset()` is documented as clearing internal state, and 1B/1C do clear theirs.
But none of them reseeds `self.rng`. Reseeding per episode would replay the identical noise
sequence in all 50 episodes of a run, collapsing exactly the variation the characterisation
is trying to measure. The stream runs continuously across episodes and is still fully
determined by `--seed`, so run-to-run determinism holds (`test_determinism`).

---

## 2. Test results

`pytest byzantine_experiments/tests/test_tier1.py` — **41 passed** (~17 s). The eight the
spec requires, with the measured numbers rather than pass/fail:

| # | Test | Measured |
| --- | --- | --- |
| 1 | Zero-mean (1A, 1B) | gaussian σ=0.25: per-dim mean `[+0.0033, −0.0009, −0.0005, +0.0004]` vs 2SE `≈0.0050`. σ=0.5: `[+0.0066, −0.0018, −0.0010, +0.0007]` vs 2SE `≈0.0100`. OU σ_ou=0.2: `[+0.0174, −0.0045, −0.0025, +0.0022]` vs 2SE `≈0.0272`. All inside 2 SE. For OU the SE is corrected for autocorrelation (n_eff = n(1−ρ)/(1+ρ), ρ = 1−θΔt = 0.85) rather than using a loosened constant. |
| 2 | Variance scaling | gaussian σ=0.1/0.25/0.5 → empirical std 0.1002 / 0.2505 / 0.5009 (ratios 2.50, 2.00). OU σ_ou=0.1/0.2/0.4 → 0.1918 / 0.3835 / 0.7670, each within 10 % of its analytic stationary std. |
| 3 | Determinism | seed 7 twice → bit-identical action streams for all three variants; seed 8 differs. Plus `test_per_agent_streams_are_independent`: drones 0 and 1 under the same run seed have \|corr\| < 0.1. |
| 4 | Range compliance | 60 steps at the worst severity of each variant; every emitted action inside `env.action_space`, checked with `Box.contains` per row. |
| 5 | Isolation | honest drones' action rows bit-identical to input while two neighbours are faulting. |
| 6 | OU reset | `x` returns to exactly zero; first post-reset step bounded by 6σ_ou; dropout's held action also cleared. |
| 7 | No team-state access | actions bit-identical under an all-zero, a ±1e6, and a NaN neighbour block. Plus `test_variants_do_not_override_broadcast` (constraint 4). |
| 8 | Dropout rate | p=0.05/0.15/0.3 → internal 0.0503 / 0.1531 / 0.3008 and externally *observed* 0.0503 / 0.1531 / 0.3008, all within 3 SE. Both are checked: the observable rate is what actually reaches the env. |

Plus registration and config-validation tests, and eleven more covering 1C′:

| Test | Measured |
| --- | --- |
| Stall duration matches config | mean_stall_s 0.1 / 0.3 / 1.0 s → 10.0 / 29.7 / 98.1 steps, targets 10 / 30 / 100 |
| Duty cycle matches config | 0.101 / 0.252 / 0.250 / 0.507 against targets 0.10 / 0.25 / 0.25 / 0.50 |
| **Nested with i.i.d. dropout** | at the equivalent parameters: duty 0.300 (target 0.30), stall length 1.43 steps (target 1.43) |
| **Burst ≫ i.i.d. at equal duty** | on a smooth (sinusoidal) command stream at duty 0.25: i.i.d. deviation 0.0023 vs burst 0.081 — **35×** |
| `hold` freezes the last *good* command | does not decay across a 20-step forced stall |
| Determinism / range / no team-state | as for the other variants, incl. a NaN-poisoned neighbour block |

---

## 3. Severity calibration table

**123 rollout runs**: 54 from the spec's matrix, 3 clean references, 6 dropout-mode
comparisons, 33 from the refinement grid (ᴿ), and 27 from the 1C′ burst-stall grid (ˢ).
50 episodes each, 3 baseline seeds, deterministic actions. Each Byzantine config is compared
against the clean rollout of the **same** checkpoint collected by the same collector.

Full table: [`results/tier1_calibration.md`](../results/tier1_calibration.md);
per-seed rows in [`results/tier1_calibration.csv`](../results/tier1_calibration.csv).
Plain-language guide to every artefact: [`why/TIER1_explanation.md`](../why/TIER1_explanation.md).

| variant | severity | f | goal RMSE | goal RMSE deg % (±seed sd) | honest deg % | reward deg % (±SEM) | action dev | saturation | collisions Δ | floor % |
|---|---|---|---|---|---|---|---|---|---|---|
| `gaussian` | sigma0.1 | 0.125 | 1.062 | 0.7 ± 3.9 | 0.5 | -8 ± 98 | 0.188 | 0.0% | +0.17 | 1.4% |
| `gaussian` **★** | sigma0.25 | 0.125 | 1.301 | 23.6 ± 11.8 | 8.6 | 214 ± 112 | 0.467 | 0.8% | +0.37 | 2.5% |
| `gaussian` **★** ᴿ | sigma0.30 | 0.125 | 1.424 | 35.2 ± 7.3 | 12.5 | 336 ± 118 | 0.553 | 2.6% | +0.32 | 3.4% |
| `gaussian` ᴿ | sigma0.35 | 0.125 | 1.487 | 40.7 ± 8.6 | 3.3 | 436 ± 117 | 0.632 | 5.3% | +0.17 | 4.4% |
| `gaussian` **★** ᴿ | sigma0.40 | 0.125 | 1.455 | 38.1 ± 6.5 | -4.6 | 314 ± 97 | 0.707 | 8.1% | +0.08 | 5.6% |
| `gaussian` | sigma0.5 | 0.125 | 1.607 | 52.1 ± 3.1 | 0.0 | 491 ± 106 | 0.839 | 13.9% | -0.02 | 8.3% |
| `gaussian` | sigma0.1 | 0.250 | 1.096 | 4.0 ± 6.7 | 4.1 | -3 ± 108 | 0.188 | 0.0% | +0.17 | 1.2% |
| `gaussian` **★** | sigma0.25 | 0.250 | 1.299 | 23.5 ± 9.4 | 0.5 | 256 ± 116 | 0.467 | 0.7% | +0.37 | 2.8% |
| `gaussian` | sigma0.5 | 0.250 | 2.016 | 91.3 ± 8.3 | 0.4 | 977 ± 106 | 0.840 | 13.8% | +0.08 | 15.8% |
| `ou` ᴿ | sigmaou0.04 | 0.125 | 1.175 | 11.7 ± 10.5 | 1.5 | 59 ± 102 | 0.142 | 0.0% | +0.36 | 1.8% |
| `ou` **★** ᴿ | sigmaou0.06 | 0.125 | 1.429 | 35.3 ± 3.9 | 1.8 | 381 ± 105 | 0.213 | 0.0% | +0.15 | 4.2% |
| `ou` ᴿ | sigmaou0.08 | 0.125 | 1.622 | 53.7 ± 1.8 | 4.0 | 509 ± 101 | 0.284 | 0.1% | +0.09 | 6.9% |
| `ou` | sigmaou0.1 | 0.125 | 1.657 | 57.2 ± 7.8 | -0.0 | 490 ± 100 | 0.355 | 0.4% | +0.03 | 8.4% |
| `ou` | sigmaou0.2 | 0.125 | 1.695 | 60.9 ± 7.1 | 9.7 | 663 ± 111 | 0.678 | 6.7% | +0.07 | 11.9% |
| `ou` | sigmaou0.4 | 0.125 | 1.553 | 47.5 ± 11.1 | 0.8 | 626 ± 101 | 1.160 | 24.1% | +0.01 | 12.6% |
| `ou` | sigmaou0.1 | 0.250 | 2.107 | 99.7 ± 3.6 | -0.5 | 1098 ± 115 | 0.356 | 0.4% | -0.03 | 15.9% |
| `ou` | sigmaou0.2 | 0.250 | 2.005 | 90.2 ± 6.8 | 0.9 | 1172 ± 111 | 0.679 | 6.8% | +0.10 | 22.3% |
| `ou` | sigmaou0.4 | 0.250 | 1.909 | 81.1 ± 6.6 | 0.3 | 1350 ± 114 | 1.161 | 24.2% | +0.01 | 24.3% |
| `dropout` | p0.05_hold | 0.125 | 1.117 | 5.8 ± 3.6 | 6.4 | 35 ± 102 | 0.001 | 0.0% | +0.22 | 1.4% |
| `dropout` | p0.15_hold | 0.125 | 1.120 | 6.2 ± 4.0 | 5.0 | 44 ± 101 | 0.003 | 0.0% | +0.12 | 1.4% |
| `dropout` | p0.15_zero | 0.125 | 1.040 | -1.3 ± 5.5 | -1.2 | -55 ± 100 | 0.045 | 0.0% | +0.08 | 1.2% |
| `dropout` | p0.3_hold | 0.125 | 1.051 | 0.0 ± 12.6 | 0.4 | -94 ± 94 | 0.008 | 0.0% | +0.09 | 1.1% |
| `dropout` ᴿ | p0.5_zero | 0.125 | 1.241 | 17.9 ± 14.7 | 3.1 | 138 ± 109 | 0.288 | 0.0% | +0.15 | 2.8% |
| `dropout` ᴿ | p0.6_hold | 0.125 | 1.061 | 0.6 ± 4.6 | 0.4 | -48 ± 98 | 0.029 | 0.0% | +0.14 | 1.2% |
| `dropout` ᴿ | p0.85_hold | 0.125 | 1.229 | 16.4 ± 1.1 | 0.2 | 182 ± 102 | 0.127 | 0.0% | +0.21 | 2.6% |
| `dropout` ᴿ | p0.85_zero | 0.125 | 1.671 | 58.4 ± 3.6 | 11.9 | 540 ± 109 | 0.821 | 0.0% | +0.09 | 12.2% |
| `dropout` ᴿ | p0.98_hold | 0.125 | 1.741 | 65.4 ± 12.7 | 5.0 | 558 ± 99 | 0.341 | 0.0% | +0.09 | 9.4% |
| `dropout` | p0.05_hold | 0.250 | 1.075 | 2.1 ± 8.9 | 2.9 | -60 ± 99 | 0.001 | 0.0% | +0.25 | 1.3% |
| `dropout` | p0.15_hold | 0.250 | 1.136 | 8.0 ± 9.4 | 8.1 | -2 ± 102 | 0.003 | 0.0% | +0.11 | 1.4% |
| `dropout` | p0.15_zero | 0.250 | 1.135 | 7.9 ± 9.1 | 3.1 | 26 ± 104 | 0.047 | 0.0% | +0.15 | 1.4% |
| `dropout` | p0.3_hold | 0.250 | 1.056 | 0.4 ± 10.0 | 0.1 | -89 ± 96 | 0.008 | 0.0% | +0.11 | 1.2% |
| `stall` ˢ | d0.10_ms0.1_hold | 0.125 | 1.146 | 8.6 ± 1.9 | 7.3 | 71 ± 107 | 0.016 | 0.0% | +0.27 | 1.5% |
| `stall` ˢ | d0.10_ms0.3_hold | 0.125 | 1.222 | 15.9 ± 4.3 | 8.9 | 150 ± 103 | 0.036 | 0.0% | +0.33 | 2.0% |
| `stall` ˢ | d0.10_ms1.0_hold | 0.125 | 1.200 | 14.1 ± 11.8 | 5.5 | 66 ± 106 | 0.058 | 0.0% | +0.19 | 2.0% |
| `stall` ˢ | d0.25_ms0.1_hold | 0.125 | 1.149 | 9.2 ± 11.9 | 1.9 | 28 ± 101 | 0.045 | 0.0% | +0.31 | 1.7% |
| `stall` **★** ˢ | d0.25_ms0.3_hold | 0.125 | 1.325 | 25.5 ± 8.0 | 9.1 | 268 ± 105 | 0.089 | 0.0% | +0.31 | 2.8% |
| `stall` **★** ˢ | d0.25_ms1.0_hold | 0.125 | 1.276 | 21.3 ± 12.7 | 3.4 | 135 ± 106 | 0.127 | 0.0% | +0.19 | 3.0% |
| `stall` **★** ˢ | d0.50_ms0.1_hold | 0.125 | 1.272 | 20.6 ± 8.1 | 7.4 | 214 ± 104 | 0.095 | 0.0% | +0.41 | 2.5% |
| `stall` **★** ˢ | d0.50_ms0.3_hold | 0.125 | 1.432 | 36.1 ± 10.7 | 2.4 | 324 ± 107 | 0.165 | 0.0% | +0.19 | 4.0% |
| `stall` ˢ | d0.50_ms1.0_hold | 0.125 | 1.587 | 50.6 ± 9.0 | 9.2 | 496 ± 109 | 0.236 | 0.0% | +0.20 | 5.3% |

★ = goal-RMSE degradation inside the 20–40 % band.  ᴿ = refinement grid.  ˢ = 1C′ stall grid.

**Read the two degradation columns very differently.**

- `goal RMSE deg %` — RMS distance to assigned goal, i.e. formation error. True zero, bounded
  below, seed-to-seed spread of **2–13 pp**. This is the axis that can actually rank
  severities.
- `reward deg %` — the figure the spec asks for. Its standard error is **±60 to ±116
  percentage points on every row**, because mean episode reward is ≈1.1 with a per-episode
  std of ≈3.5 and a heavy left tail, so `(clean − byz)/|clean|` divides by a near-zero
  denominator. Three rows fall inside 20–40 % on this metric **purely by noise** — e.g.
  `dropout p0.05_hold` at 35 %, whose action deviation is 0.001 and whose goal-RMSE effect is
  5.8 %. Do not use them.

Secondary observations:

- **Collisions barely move.** `collisions Δ` is between −0.03 and +0.41 per episode against a
  clean baseline of ~0.3, and minimum inter-drone separation is flat at 0.35–0.39 m in every
  config. A degraded drone does not cause pile-ups; the trained collision-avoidance handles it.
- **Floor time is the damage channel for 1A/1B, but not for 1C′.** Gaussian σ=0.5 and OU put
  the drone on the floor 8–24 % of the time; the entire stall grid stays at 1.5–5.3 %, even at
  50 % goal-RMSE degradation. 1C′ damages by *displacement* rather than by grounding, which is
  why its goal RMSE keeps rising instead of saturating.
- **1A and 1B saturate; 1C′ does not.** OU at f=0.125 goes 57.2 → 60.9 → 47.5 % as σ_ou
  increases — non-monotonic, because a drone parked on the floor is at a bounded distance from
  its goal. The stall grid is monotonic in `duty_cycle` throughout.

---

## 4. The 20–40 % band

All three variants now have at least one configuration in the band, so **the validation gate
is met**. Status by goal-RMSE degradation at f = 0.125:

| variant | spec grid | added grids | in band? |
| --- | --- | --- | --- |
| 1A gaussian | 0.1 → 0.7 %, **0.25 → 23.6 %**, 0.5 → 52.1 % | ᴿ **0.30 → 35.2 %**, 0.35 → 40.7 %, **0.40 → 38.1 %** | ✅ σ ∈ {0.25, 0.30, 0.40} |
| 1B OU | 0.1 → 57.2 %, 0.2 → 60.9 %, 0.4 → 47.5 % (all above) | ᴿ 0.04 → 11.7 %, **0.06 → 35.3 %**, 0.08 → 53.7 % | ✅ σ_ou = 0.06 |
| 1C dropout | 0.05 → 5.8 %, 0.15 → 6.2 %, 0.3 → 0.0 % | ᴿ 0.85 → 16.4 %, 0.98 → 65.4 % | ❌ never in band at any p |
| **1C′ stall** | — | ˢ **d0.25/ms0.3 → 25.5 %**, **d0.25/ms1.0 → 21.3 %**, **d0.50/ms0.1 → 20.6 %**, **d0.50/ms0.3 → 36.1 %** | ✅ four configs |

1C′ replaces 1C as the usable stall variant. 1C is retained as its short-stall limit (§1),
and its rows stay in the table as evidence for the control-rate argument in §8.

### Recommended canonical settings

| variant | canonical config | goal-RMSE deg | action dev | saturation | rationale |
| --- | --- | --- | --- | --- | --- |
| 1A gaussian | **σ = 0.25, f = 0.125** | 23.6 % ± 11.8 | 0.467 | 0.8 % | Stable across f: 0.25 gives 23.5 %, essentially the same with twice the compromised drones. Saturation under 1 %, so still genuinely unbiased noise (§5). |
| 1B OU | **σ_ou = 0.06, f = 0.125** | 35.3 % ± 7.4 | 0.213 | 0.0 % | The only in-band OU config, and clean of clipping. Remember the stationary std is 1.9× the nominal σ_ou (§1). |
| 1C′ stall | **duty = 0.25, mean_stall_s = 0.3, `hold`, f = 0.125** | 25.5 % ± 8.0 | 0.089 | 0.0 % | Mid-band with the **lowest seed spread of any in-band stall config**, and physically plausible — 300 ms stalls occupying a quarter of the flight. `d0.50/ms0.3` (36.1 %) is the stronger alternative. Avoid `ms1.0`: with 15 s episodes a 1 s mean stall gives only a handful of stall events per episode, and the seed spread jumps to ±12.7. |

### The finding that matters most for Tier 3

**Tier 1 damage is almost entirely confined to the faulty drone.** The `honest deg %` column
never exceeds ~10 % and is frequently negative, across all 40 configurations — including
gaussian σ=0.5 at f=0.25, where the compromised drones are 91 % worse and grounded 16 % of the
time, yet their honest neighbours sit at +0.4 %.

The spec's plan is to tune Tier 3 to cause *comparable team damage*, so the detector is not
merely separating high-damage from low-damage. But an 8-drone swarm with 1–2 stochastically
degraded members suffers essentially **no** team-level damage here, so matching on team
degradation is only satisfiable at a Tier 3 strength of roughly zero.

Match on a **per-agent** axis instead — the compromised drone's own goal-RMSE degradation and
its mean L2 action deviation. Both are stable and both are in the table. The three canonical
settings give Tier 3 its tuning targets, and usefully they span a **5× range of action
deviation at comparable damage**, which is itself a useful control:

| target | 1A σ=0.25 | 1B σ_ou=0.06 | 1C′ d0.25/ms0.3 |
| --- | --- | --- | --- |
| own goal-RMSE degradation | 23.6 % | 35.3 % | 25.5 % |
| mean L2 action deviation | 0.467 | 0.213 | 0.089 |
| direction persistence | 0.66 | 0.80 | 0.72 |
| floor time | 2.5 % | 22.9 % | 2.8 % |

If Tier 3 *does* produce real team damage where Tier 1 produces none, that asymmetry is itself
a headline result rather than a confound — but it must be stated as such, not hidden inside a
"matched severity" claim.

---

## 5. Clipping analysis

Saturation = fraction of emitted action *components* sitting on the clip boundary (|a| ≥ 1 −
1e−4), measured on the compromised drones only. Honest drones sit at 0.0 % throughout.

| variant | severity | saturation | verdict |
| --- | --- | --- | --- |
| gaussian | σ=0.1 | 0.0 % | clean |
| gaussian | σ=0.25 | **0.8 %** | clean — the canonical setting is unaffected by clipping |
| gaussian | σ=0.5 | **13.9 %** | biased but under the spec's ~30 % disqualification threshold |
| OU | σ_ou=0.1 | 0.4 % | clean |
| OU | σ_ou=0.2 | 6.7 % | mild |
| OU | σ_ou=0.4 | **24.1 %** | approaching the threshold — avoid as a canonical setting |
| dropout | all | 0.0 % | n/a: it emits stale or zero actions, which are in range by construction |
| stall (1C′) | all | 0.0 % | same — a held command was already legal when it was issued |

**No configuration exceeds 30 %, so no variant is disqualified.** The spec's specific worry —
that σ=0.5 on a [−1,1] space saturates "a large fraction" of samples — is *not* borne out:
13.9 %, not a majority. The reason is that the honest policy's actions are small (per-drone
action std ≈0.17, centred near hover), so a σ=0.5 draw usually lands well inside the range.

**Does zero-mean survive clipping? No, and the direction is measurable.** Pre-clipping,
zero-mean holds to within 2 SE (§2, test 1). Post-clipping it cannot hold in general: clipping
is a contraction toward the interior, so whenever the honest action is off-centre the surviving
noise is asymmetric. At σ=0.25 with 0.8 % saturation the effect is negligible. At σ=0.5 with
13.9 % it is not, and the σ=0.5 rows should be described as "heavy-tailed bounded noise",
**not** as unbiased Gaussian noise. Same for OU σ_ou=0.4 at 24.1 %.

This is why the canonical 1A recommendation is σ=0.25 rather than σ=0.5, even though σ=0.5
would also be defensible on damage grounds.

---

## 6. Behavioural observations

Derived from the trajectories rather than from watching renders, so the description is
reproducible and quotable — see [`analysis/behaviour.py`](../analysis/behaviour.py) for the
descriptor definitions and [`results/tier1_behaviour.json`](../results/tier1_behaviour.json)
for the full numbers. Figures (XY trajectory, altitude, goal error, action trace) for all ten
f=0.125 configs are in [`results/figures/`](../results/figures/). Video rendering also works
headlessly via `xvfb-run` (see `rollout/render.py`); it was not needed to reach these
conclusions.

Clean reference (seed 1, all 8 drones): jitter 0.024, action jitter 0.022, action std 0.170,
goal error 0.796 m, direction persistence 0.685, floor fraction 0.7 %.

**1A gaussian — a visible high-frequency shake that stays local.**
At σ=0.1 the only detectable change is a twitchier control signal (action jitter 0.19 vs 0.022
clean, ~8×); the flight path is otherwise normal. At σ=0.25 the drone *wobbles* — velocity
changes 2.5× faster than clean — and strays around its goal (1.4 m vs 0.80 m) **without a
consistent direction**: direction persistence 0.66, essentially identical to the clean 0.69.
That is the signature of white noise: it shakes in place rather than going anywhere. At σ=0.5
it stops being a wobble and becomes a fall — 50 % of the episode below 0.15 m.

**1B OU — a slow, purposeful-looking drift, and this is the important one.**
Even at σ_ou=0.1 the drone leaves its goal by 3.5 m (4.4× clean) with direction persistence
**0.92**, rising to 0.97 at higher severities. It picks a direction and keeps going for
seconds at a time. Crucially it does this at *lower* action deviation than 1A: σ_ou=0.1 has
action deviation 0.355 versus 0.467 for gaussian σ=0.25, yet causes 57 % goal-RMSE degradation
against 24 %. **Temporally correlated noise is ~2.4× more damaging per unit of action
perturbation than white noise**, because white-noise perturbations cancel over the timescale
the controller responds on, while OU perturbations integrate into real displacement. It also
ends up on the floor 55–92 % of the time.

This confirms the spec's prediction that 1B is the harder negative control. Its behavioural
signature — sustained directional departure from goal — is precisely what a naive
"goal-directed behaviour" detector would flag as *intent*. A detector that keys on
directedness will confuse a drifting sensor for an adversary, which is the whole reason 1B is
in the design.

**1C′ stall — a distinct third signature: smooth but stale.**
This is the behaviourally most interesting variant, because its command-stream signature runs
*opposite* to 1A and 1B. Gaussian and OU make the command **twitchier** (action jitter 0.19
and 0.12 against 0.022 clean); a stalling drone makes it **smoother** — action jitter falls to
0.013 at `ms=1.0`, because a held command is perfectly constant. Meanwhile the *spread* of the
command rises (action std 0.29–0.30 vs 0.170 clean), because the drone overshoots during each
stall and the policy responds harder once it regains control.

At the canonical `d0.25/ms0.3` setting the drone sits 1.86 m from its goal (2.3× clean) with
direction persistence 0.72 — mildly elevated over the 0.685 clean baseline but nowhere near
OU's 0.92–0.97. It spends 10.7 % of the time near the floor against 0.7 % clean, but that is
far below OU's 55–92 %. In plain language: **it lurches**. It holds a stale command, drifts
off, catches itself, and repeats — visible in the figure as a staircase in the executed
command and as a sawtooth in altitude.

The action-trace panel of
[`behaviour_t1_stall_s1_d0.25_ms0.3_hold_f0125.png`](../results/figures/behaviour_t1_stall_s1_d0.25_ms0.3_hold_f0125.png)
is the clearest single image of any Tier 1 fault: flat plateaus in the executed command where
the honest command keeps moving.

**1C dropout — nearly invisible, at any severity in the spec's grid.**
At p=0.3 with `hold`, every descriptor is within noise of clean: action jitter 0.021 vs 0.022,
jitter 0.024 vs 0.024, goal error 0.70 vs 0.80 m, floor 1.0 % vs 0.7 %. The behavioural
description generator returns "behaviourally indistinguishable from a clean drone" for
p ∈ {0.05, 0.15, 0.3}. `zero` mode is slightly more visible (action deviation 0.045 vs 0.008
for `hold` at comparable p) but still causes no measurable goal-error change. §8 explains why.

---

## 7. Wandb logging

Everything is in project **`byzantine_exp`**, entity
**`bmugizi-carnegie-mellon-university`** — the same project as the baseline training runs, so
Byzantine rollouts sit next to the clean baselines and can be compared directly. The
organisation follows README.md's convention:

| Level | Value | Purpose |
| --- | --- | --- |
| `group` | one per experimental condition, e.g. `tier1_ou_sigmaou0.2_f0125`, `clean_baseline` | **This is what aggregates the 3 baseline seeds into a single mean±band.** Deliberately *not* per-seed. |
| `job_type` | `rollout` for data, `analysis` for the calibration table and behaviour figures | |
| `tags` | `tier1`, `tier1_ou`, `seed1`, `f0125`, `nbyz1`, `sigma_ou0.2`, `8agents`, `mix`, `exp1_baseline_seed1`, `n_ep50` | filter the runs table without opening anything |
| `name` | `t1_ou_s1_sigmaou0.2_f0125` | decodable at a glance |

Each rollout run carries, per the "don't make me dig through JSON" requirement:

- **run summary** — every scalar in `analysis/metrics.py`: episode reward (all-drone and
  honest-only), goal RMSE (all / honest / byzantine), collisions, action deviation (mean and
  max), saturation rate (byzantine and honest), perturbed-step fraction, min separation,
  low-altitude fraction, episode length — each with `_mean`, `_std`, `_sem`. Sortable and
  filterable directly in the runs table.
- **`episodes` Table** — one row per episode with every per-episode metric, for when a mean
  is not enough.
- **`hist/*`** — a histogram per per-episode metric. Episode reward is heavily left-tailed,
  so the distribution carries information the mean actively hides; logging both means nobody
  has to rediscover that.
- **`episode/*`** — per-episode curves (step-indexed), so drift across a run is visible.
- **config** — the entire run sidecar flattened to dotted keys: checkpoint path and resolved
  checkpoint file, tier, tier config, compromised indices, f, assignment mode, episode count,
  seed, determinism, `quads_mode`, episode duration, every domain-randomisation setting
  (including the `thrust_noise_ratio=0.05` and `replay_buffer_sample_prob=0.75` that are
  hardcoded upstream), array shapes/dtypes, git SHA, and the full command line.

Analysis artefacts go to group `tier1_analysis`: `tier1_calibration` (the calibration table
as a wandb Table, plus the `.csv`/`.md` uploaded as files) and `tier1_behaviour_s1` (the
descriptor table and the trajectory figures as images).

Two operational notes:

- **Logging failures never lose data.** `collect.py` writes the `.npz` and `.json` before it
  touches wandb and treats a logging exception as non-fatal. Three runs in the first sweep
  hit `UsageError: No API key configured` — `~/.netrc` lives on AFS and twelve workers were
  reading it concurrently. They were re-logged from disk with
  `python -m byzantine_experiments.analysis.backfill_wandb`, which skips runs already
  present and so is safe to repeat.
- **Re-running a sweep creates duplicate names rather than overwriting.**
  `python -m byzantine_experiments.analysis.dedupe_wandb --delete` keeps the newest run of
  each name and removes the superseded ones. Used after the re-run described in §8.

---

## 8. What surprised me

**1. Dropout is essentially a no-op, and the reason is the control rate — now fixed.**
This was the biggest surprise. `hold` mode repeats the previous emitted action, and at the
env's **100 Hz** control rate consecutive policy actions are nearly identical — the mean L2
action deviation from holding is **0.001–0.008**, against 0.19–0.84 for gaussian. Holding a
10 ms-old action is almost the same as using the current one. So Variant 1C as specified does
not model a meaningful stall; it models a 10 ms lag. It never entered the 20–40 % band at any
`p_drop`, jumping from 16.4 % at p=0.85 to 65.4 % at p=0.98.

The diagnosis was that `p_drop` controls how *often* a stall begins when what matters is how
*long* it lasts: at p=0.3 the average stall is 1.4 steps, i.e. 14 ms. **`tier1_stall` (1C′)
fixes this** by parameterising the fault as a two-state Markov chain over stall duration and
duty cycle (§1). The correction is large and measurable: at an identical duty cycle of 0.25,
switching from i.i.d. single-step stalls to 0.5 s bursts increases the command perturbation
**35×** (0.0023 → 0.081), and four of the nine stall configs land in the 20–40 % band where
1C had none.

Worth carrying forward: **any per-step i.i.d. fault will be weak in this environment**, because
the control rate is high relative to the drone's dynamics. Tier 2's sign-flip will not have
this problem (a flipped command is maximally wrong every step), but any Tier 2/3 variant that
works by *withholding* or *delaying* rather than *corrupting* needs a duration parameter.

**2. OU's stationary std is 1.9× its nominal σ_ou, which is why 1B looks so strong.**
`σ_stationary = σ_ou·√(Δt/(1−(1−θΔt)²)) = 1.898·σ_ou` for θ=0.15, Δt=1. So the "σ_ou=0.1"
row is really injecting std ≈0.19 noise, and the 1A/1B severity columns are not comparable by
number even though the spec presents them as parallel grids. This is a large part of why 1B's
*lowest* severity already overshoots the 20–40 % band. Anyone reading the calibration table
needs to know this; it is exposed as `OUNoiseAgent.stationary_std()` and asserted in the tests.

**3. But that is not the whole story — OU is genuinely more damaging per unit perturbation.**
Even after correcting for the 1.9× factor, OU σ_ou=0.1 (action deviation 0.355) does 57 %
goal-RMSE degradation while gaussian σ=0.25 (action deviation 0.467, *higher*) does 24 %.
Correlated error integrates into displacement; white error cancels. The spec predicted 1B
would be the harder control and it is, for a quantifiable reason.

**4. Goal RMSE saturates and then goes backwards — for 1A/1B, but not 1C′.**
OU at f=0.125: 57.2 % → 60.9 % → 47.5 % as σ_ou rises. Not a bug — the drone is on the floor
82–92 % of the time at the top severities, and a grounded drone is at a *bounded* distance
from its goal. Any severity axis built on goal error will be non-monotonic once the fault is
strong enough to ground the drone. For the 20–40 % band this does not matter (it is well below
saturation), but a Tier 3 comparison at high severity would need a different metric. Notably
1C′ does *not* show this: its floor time stays at 1.5–5.3 % across the whole grid and its goal
RMSE rises monotonically with duty cycle, because a lurching drone stays airborne. That makes
1C′ the better-behaved severity axis of the three.

**5. The team absorbs the fault almost perfectly.** Covered in §4 — honest neighbours never
degrade more than ~10 %, collisions move by <0.4 per episode, and minimum separation is flat.
I expected at least the f=0.25, σ=0.5 corner to disturb the formation. It does not.

**6. A collector bug that only showed up in the geometry metrics.**
`positions`, `velocities` and `collisions` were being read through a cached
`env.unwrapped`. `ExperienceReplayWrapper` swaps its `self.env` for a deepcopy of a buffered
snapshot on replay, which at `replay_buffer_sample_prob=0.75` happens on most resets — so the
cached handle pointed at an env that was no longer being stepped, and 42 of 50 episodes
recorded a perfectly frozen drone while `obs` kept moving. `obs`, actions and rewards were
always correct (they come back from `step()`), so goal RMSE, action deviation and saturation
were never affected; but collision counts and floor-time were garbage in the first sweep.
Fixed in `rollout/collect.py`, with a regression test
(`test_recorded_positions_track_the_live_env_across_replayed_episodes`) that asserts recorded
position must move whenever `obs` does. The whole 63-run matrix was recollected afterwards.

The general lesson for Tiers 2–3: **never hoist `.unwrapped` out of a loop in this codebase.**
