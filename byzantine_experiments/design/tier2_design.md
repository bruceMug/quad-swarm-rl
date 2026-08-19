# Tier 2 — Structured Byzantine Attacks: design notes

Implementation record for [specs/TIER2_SIGNFLIP.md](../specs/TIER2_SIGNFLIP.md).
Branch `feature/byzantine`. Built on Tier 0 infrastructure
([tier0_design.md](tier0_design.md)) and reusing Tier 1's clean references
([tier1_design.md](tier1_design.md)).

**Still no upstream files modified.** 2C intercepts neighbour observations from inside the
wrapper, so `gym_art/` remains untouched (§3).

---

## 1. The four variants as implemented

All four live in [`agents/tier2_structured.py`](../agents/tier2_structured.py). They are the
first tier with *intent*: they may read their **own** state and act on it. None reads an
honest drone's state; 2D members share only what they themselves publish.

Three new capability flags were added to `ByzantineAgent`, all defaulting off so Tier 0/1
behaviour is bit-for-bit unchanged:

| Flag | Effect |
| --- | --- |
| `needs_ego_state` | the wrapper sets `agent.ego` (pos, vel, rot, omega, control Jacobian, room box) before each decision. Own state only. |
| `lies_about_state` | the wrapper routes the agent's reported pos/vel through `perturb_broadcast_state()` and rewrites every *other* drone's neighbour block (§3). |
| `needs_coordination` | all such agents share one mutable dict. |

### 2A — `tier2_signflip`

```
a'[d] = clip(-scale * a[d])   for d in dims;   a'[d] = a[d] otherwise
```

**A spec/environment mismatch worth stating plainly.** The spec describes `dims=[1,2]` as
"flips roll/pitch but leaves thrust honest". That assumes the action is
(thrust, roll, pitch, yaw). **It is not** — this env's action is four raw *motor* thrusts
(`RawControl`, see tier0_design.md §1), so flipping action dims 1 and 2 flips two rotors, not
two body axes. Both readings are therefore implemented, selected by `space`:

| `space` | meaning | canonical use |
| --- | --- | --- |
| `'motor'` | literal: negate raw motor commands. `dims='all'` inverts collective thrust and the drone falls almost immediately. | the **full** attack |
| `'body'` | the spec's *intent*: decompose the action through the control Jacobian into `[collective, roll, pitch, yaw]` accelerations, negate the selected components, map back. `dims=[1,2]` genuinely means "invert roll and pitch, leave thrust honest". | the **partial** attack |

`space='body', dims=[1,2], scale=1.0` is the canonical partial setting, and
`test_signflip_body_space_preserves_collective_thrust` verifies that the collective component
survives the flip (mean ratio 1.00 ± 0.05) while roll flips sign on >80 % of steps.

### 2B — `tier2_waypoint_pull`

```
a' = clip((1 - alpha) * a + alpha * position_control_action(ego -> w_adv))
```

`alpha` ∈ {0.3, 0.6, 1.0}. `w_adv` defaults to a low far corner of the arena derived from the
agent's own `room_box` (`[hi_x - 1, hi_y - 1, lo_z + 1]`) — deterministic given the arena,
inside the walls, and low, so dragged neighbours are pulled toward the floor. The
direction→action mapping is §2.

### 2C — `tier2_liar`

Flies honestly, lies about where it is. `perturb_action` returns the honest action **by
construction** (`np.array(honest_action, copy=True)`); the entire attack is in
`perturb_broadcast_state`. Modes:

| `lie_mode` | reported position | reported velocity |
| --- | --- | --- |
| `offset` | `pos + magnitude * u`, where `u` is a fixed unit vector drawn once from the agent's seeded RNG (deterministic per agent, not axis-aligned) | true |
| `stale` | position from `delay_k` steps ago (default 50 = 0.5 s at 100 Hz) | also delayed |
| `mirror` | reflected through `mirror_centre` (default `[0, 0, 2]`) | negated |

### 2D — `tier2_collusion`

Two attackers, f = 0.25 only. Roles are assigned deterministically: each member appends its
index to a shared `members` list at reset, and its **rank** is its position in the sorted
list — so `rank` does not depend on dict ordering or step timing.

| `mode` | behaviour |
| --- | --- |
| `same_waypoint` | both fly at the same corner: a coordinated hole punched in one place |
| `pincer` | opposite corners, tearing the formation along a diagonal |
| `mirror_pair` | rank 0 flips motors {0,1}, rank 1 flips {2,3} — anti-correlated commands with no waypoint involved |

### Survival guard (design constraint 2)

Every variant supports `survival_guard` (default on). Below `guard_altitude` (0.6 m) the
emitted action blends back toward honest, reaching fully honest at `guard_full_altitude`
(0.15 m):

```
blend = clip((guard_altitude - z) / (guard_altitude - guard_full_altitude), 0, 1)
emitted = (1 - blend) * attack + blend * honest
```

A *ramp* rather than a hard switch, because a discontinuous action stream would itself be a
detection giveaway unrelated to the attack. `guard_trigger_rate` is counted per agent and
written into every run's sidecar, because **a guard that fires constantly means the "attack"
is mostly honest flight** and the calibration must say so. Confirmed by the matrix: 2A spends
66-83 % of steps inside the guard ramp, 2B only 10-22 % (§6).

---

## 2. The direction→action mapping for 2B

This is the fiddliest part of Tier 2, as the spec predicted.

**What I found.** `gym_art/quadrotor_multi/quadrotor_control.py` contains
`NonlinearPositionController` (a Mellinger controller) whose `step()` maps a desired goal
position to motor commands. Its body (lines 282-329) is:

```python
to_goal  = goal - dynamics.pos
e_p      = -clamp_norm(to_goal, 4.0)
e_v      = dynamics.vel
acc_des  = -kp_p * e_p - kd_p * e_v + [0, 0, GRAV]        # kp_p, kd_p = 4.5, 3.5
zb_des, _ = normalize(acc_des)
yb_des, _ = normalize(cross(zb_des, xc_des))               # xc_des = rot_des[:, 0] = [1,0,0]
xb_des   = cross(yb_des, zb_des)
R_des    = column_stack((xb_des, yb_des, zb_des))
e_R      = 0.5 * vee(R_des.T @ R - R.T @ R_des);  e_R[2] *= 0.2
dw_des   = -kp_a * e_R - kd_a * dynamics.omega             # kp_a, kd_a = 200.0, 50.0
thrust_mag = acc_des @ R[:, 2]
thrusts  = clip(Jinv @ append(thrust_mag, dw_des), 0, 1)
```

**How I reused it.** Not by importing it: `NonlinearPositionController.__init__` executes
`import tensorflow as tf` unconditionally (line 256) *even when `tf_control=False`*, and
tensorflow is not installed in `swarm_r311_recovered`. Instantiating it is impossible here.
So [`agents/control_utils.py`](../agents/control_utils.py) reproduces the ~20 lines above
verbatim, importing the helpers it depends on (`quadrotor_jacobian`, `normalize`, `cross`,
`clamp_norm`) directly from the same modules so the gains and geometry cannot drift.

**The output-convention detail that would silently break it.** The Mellinger controller emits
motor commands in **[0, 1]**, but the env's action space is `RawControl` with
`zero_action_middle=True`, i.e. `Box(-1, 1)` mapped to thrust by `0.5 * (a + 1)`. The final
step is therefore the inverse of that affine map, `a = 2 * thrust - 1`
(`thrust_to_action`). Omitting it would halve every command and add a constant offset — the
attacker would fly, badly, in roughly the right direction, and the bug would be very hard to
spot.

**Limitations.**

1. **Yaw is uncontrolled** in the original (`xc_des` is fixed to the world x-axis and yaw
   error is damped by 0.2), so the attacker never actively yaws. Inherited deliberately.
2. **The gains are tuned for the honest task**, not for aggressive pursuit; at `alpha=1.0`
   the attacker is a competent but not optimal flier toward `w_adv`. This is the point of
   Tier 2 being *untrained* — Tier 3 is where optimisation enters.
3. **No obstacle or neighbour awareness.** The controller flies straight at the waypoint; if a
   drone is in the way it collides. That is a legitimate attack behaviour, not a bug, but it
   means 2B's collision numbers are partly incidental rather than targeted.
4. The Jacobian is computed once per drone per episode from its own `dynamics`. With domain
   randomisation of mass/inertia enabled this would need refreshing; it currently is not
   (`dynamics_randomize_every=None`).

---

## 3. The observation layout for 2C

**The single most reusable finding in this document.** Verified empirically, not inferred —
`scripts`-free reconstruction of the neighbour block matched the env's output to `0.000e+00`.

`QuadrotorEnvMulti` builds each drone's observation as `[ego block | neighbour blocks]`:

```python
# quadrotor_multi.py:212-245
def get_rel_pos_vel_item(self, env_id, indices=None):
    cur_pos = self.pos[i];  cur_vel = self.vel[i]
    pos_rel = np.stack([self.pos[j] for j in indices]) - cur_pos
    vel_rel = np.stack([self.vel[j] for j in indices]) - cur_vel
    return pos_rel, vel_rel

def get_obs_neighbor_rel(self, env_id, closest_drones):
    pos_rel, vel_rel = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])
    return np.concatenate((pos_rel, vel_rel), axis=1)       # (n_neighbors, 6)

def extend_obs_space(self, obs, closest_drones):
    obs_neighbors = np.stack([... .reshape(-1) ...])         # row-major flatten
    obs_neighbors = np.clip(obs_neighbors,
                            self.clip_neighbor_space_min_box, self.clip_neighbor_space_max_box)
    return np.concatenate((obs, obs_neighbors), axis=1)
```

So the exact index mapping is:

| Slice of `obs[i]` | Contents |
| --- | --- |
| `0 : ego_dim` | drone `i`'s own state. `ego_dim` = 18 for `xyz_vxyz_R_omega` (19 for `_floor`, 24 for `_wall`) — the wrapper derives it as `obs_dim − 6·n_neighbors` rather than hardcoding. |
| `ego_dim + 6·slot     : +3` | `pos[j] − pos[i]` for the drone at proximity rank `slot` |
| `ego_dim + 6·slot + 3 : +6` | `vel[j] − vel[i]` for the same drone |

with the whole neighbour block clipped to ±10 m and ±6 m/s
(`clip_neighbor_space_{min,max}_box`).

**Two properties that make this attack easy to get silently wrong:**

1. **`slot` is a proximity rank, not an agent id.** `neighborhood_indices()`
   (quadrotor_multi.py:247-274) re-sorts each drone's neighbours *every step* by
   `rel_dist + rel_pos_unit · rel_vel` (distance plus closing speed), and when
   `num_use_neighbor_obs < num_agents − 1` — which is the trained config: 6 of 7 — it keeps
   only the closest few. So the liar's index inside a given observer's vector moves from step
   to step, and on some steps **the liar is not in that observer's block at all**. Writing to
   a fixed offset would corrupt an innocent neighbour and leave the liar honest.
   The wrapper therefore calls the env's own `neighborhood_indices()` rather than
   reimplementing the sort.
2. **The neighbour block carries *true* positions with no sensor noise**, while the ego block
   does (verified: ego `pos−goal` differs from the true value by ~8e-3, the neighbour block by
   exactly 0). So a lie is the *only* corruption present in that block, which is convenient for
   attribution.

**Where the hook lives.** `ByzantineEnvWrapper._apply_broadcast_lies()` post-processes the
observation array returned by `reset()`/`step()`. Because the wrapper sits above
`QuadEnvCompatibility`, this needs no change to `gym_art/` at all — the earlier concern that
2C would require modifying `add_neighborhood_obs` turned out to be unnecessary.

**Documented limitation.** The lie corrupts the *values* an observer sees, not *which*
neighbours it attends to: neighbour selection still runs on true positions. A fully faithful
liar would also distort selection (by claiming to be far away it could drop out of a
neighbour's attention set entirely). Implementing that would mean re-deriving the sort from
reported positions, which changes what "closest 6" means for every drone simultaneously and
couples the lies of multiple liars. The simpler version isolates the effect of false *state
values*, which is the quantity of interest; the limitation is flagged here so nobody reads the
2C numbers as covering attention manipulation.

---

## 4. The 2C measurement problem — raised explicitly

**The spec asks for this to be surfaced before the experiment rather than discovered after, so
here it is up front.**

A state-liar's *own* actions are honest by construction — `test_liar_actions_are_bit_identical_to_honest`
asserts bit-equality with the honest policy's output. Its own observation is also untouched
(`test_liar_own_observation_is_unchanged`). Therefore:

> **A policy graph built from the liar's own (observation, action) stream is, by construction,
> a clean honest policy graph.** There is no behavioural signature to find in it, because
> there is no behavioural anomaly. Any spectral method applied *to the liar itself* must score
> it as honest, and that is correct behaviour, not a detector failure.

What follows for the experiment:

1. **Detection of 2C must come from the victims, not the attacker.** The observable signature
   is in the *honest* drones' policy graphs, which are being driven by corrupted neighbour
   observations and will therefore respond to a phantom. The detection question changes shape
   from "which agent looks anomalous?" to "which agent's *reported* state is inconsistent with
   the behaviour it induces in others?"
2. **The set-structured (siForest-style) formulation is the natural fit**, precisely because
   the anomaly is a property of a *relation* between agents rather than of any single agent's
   trajectory. This is an argument for the set semantics, and worth making in the paper.
3. **A per-agent AUC computed against `byzantine_mask` will look bad for 2C, and that is not a
   bug.** If the mask labels the liar as positive and the detector scores the liar as honest,
   AUC ≈ 0.5 or worse — while the detector may be correctly flagging the *victims* as
   anomalous. Reporting 2C AUC alongside 2A/2B without this caveat would be actively
   misleading. Recommendation: report 2C separately with victim-side labels as well as
   attacker-side, and state both.
4. **Consequently the efficacy bar in the spec ("team reward degradation ≥ 20 % for every
   canonical config") is not the right gate for 2C.** A lie only damages the team to the
   extent that neighbours act on it; it does not directly move any actuator. Expect 2C's
   team-damage numbers to be well below 2A/2B. That makes 2C a *weak attack but a strong test
   case*, and the calibration table should be read that way. The efficacy unit test excludes
   2C for this reason, with the exclusion documented in the test itself.

This is a genuinely interesting result in either direction, but it needed to be understood
before the numbers arrived rather than after.

---

## 5. Severity calibration table

57 rollout runs, 50 episodes each, 3 baseline seeds, deterministic actions. Clean references
reused from Tier 1 (same checkpoints, same collector). Full table:
[`results/tier2_calibration.md`](../results/tier2_calibration.md); per-seed rows in
[`results/tier2_calibration.csv`](../results/tier2_calibration.csv).

| variant | config | f | goal RMSE deg % | spillover % | attacker survival | guard exposure | t-to-impact | action dev | collisions Δ | floor % |
|---|---|---|---|---|---|---|---|---|---|---|
| `signflip` **★** | full | 0.125 | 28.9 | -6.8 | 69% | 83% | n/a | 0.653 | +0.06 | 5.0% |
| `signflip` | partial | 0.125 | 42.9 | +5.7 | 73% | 67% | 0.7 s | 0.215 | +0.11 | 4.9% |
| `signflip` | full | 0.250 | 60.7 | +2.3 | 45% | 83% | 0.6 s | 0.652 | +0.02 | 8.9% |
| `signflip` | partial | 0.250 | 67.6 | -3.6 | 45% | 66% | 4.7 s | 0.215 | +0.05 | 8.0% |
| `waypoint` | a0.3 | 0.125 | 77.8 | +3.5 | 91% | 22% | n/a | 0.294 | +0.03 | 2.6% |
| `waypoint` | a0.6 | 0.125 | 100.2 | -6.3 | 98% | 13% | n/a | 0.412 | -0.03 | 1.8% |
| `waypoint` | a1.0 | 0.125 | 116.9 | +2.9 | 100% | 10% | n/a | 0.515 | +0.03 | 1.5% |
| `waypoint` | a0.3 | 0.250 | 120.3 | -4.5 | 85% | 22% | 7.1 s | 0.313 | +0.23 | 3.4% |
| `waypoint` | a0.6 | 0.250 | 167.7 | -0.4 | 69% | 29% | 5.8 s | 0.403 | +1.28 | 4.9% |
| `waypoint` | a1.0 | 0.250 | 168.1 | -3.8 | 57% | 41% | 6.7 s | 0.503 | +1.83 | 6.5% |
| `liar` | mirror | 0.125 | 9.9 | +8.9 | 98% | 5% | n/a | 0.000 | +0.31 | 1.6% |
| `liar` | offset | 0.125 | 5.6 | +2.9 | 97% | 6% | n/a | 0.000 | +0.31 | 1.4% |
| `liar` | stale | 0.125 | 9.4 | +9.1 | 96% | 4% | n/a | 0.000 | +0.38 | 1.5% |
| `liar` | mirror | 0.250 | 11.8 | +8.8 | 97% | 5% | n/a | 0.000 | +0.66 | 1.7% |
| `liar` | offset | 0.250 | 10.7 | +9.5 | 97% | 5% | n/a | 0.000 | +0.60 | 1.6% |
| `liar` | stale | 0.250 | 6.2 | +6.5 | 100% | 4% | n/a | 0.000 | +0.35 | 1.4% |
| `collusion` | mirror_pair | 0.250 | 73.7 | +7.9 | 37% | 93% | 0.4 s | 0.384 | +0.00 | 10.3% |
| `collusion` | pincer | 0.250 | 181.2 | +5.8 | 96% | 10% | 4.9 s | 0.463 | +0.02 | 2.1% |
| `collusion` | same_waypoint | 0.250 | 177.8 | +7.9 | 67% | 30% | 4.6 s | 0.412 | +1.26 | 5.1% |

★ = goal-RMSE degradation inside the 20–40 % band.

### The gate is met for 2A only

| variant | range of goal-RMSE degradation | in band? |
| --- | --- | --- |
| 2A signflip | 28.9 – 67.6 % | ✅ `full` at f=0.125 → **28.9 %** |
| 2B waypoint | 77.8 – 168.1 % | ❌ far too strong even at α=0.3 |
| 2C liar | 5.6 – 11.8 % | ❌ far too weak — as predicted in §4 |
| 2D collusion | 73.7 – 181.2 % | ❌ far too strong |

This mirrors Tier 1: the spec's severity grids bracket the band rather than landing in it. To
close the gap, 2B needs α ≈ 0.1–0.15 (below the spec's lowest value) and 2D needs a reduced
α; 2C cannot reach the band by construction (§4).

### Matched-damage pairs

Because 28.9 % is the *only* Tier 2 config in the band, all three Tier 1 canonical settings
pair to the same Tier 2 config:

| Tier 1 canonical | its deg | closest Tier 2 | its deg | gap | T1 action dev | T2 action dev |
| --- | --- | --- | --- | --- | --- | --- |
| 1A gaussian σ=0.25 | 23.6 % | `signflip_full_f0125` | 28.9 % | 5.3 pp | 0.467 | 0.653 |
| 1B OU σ_ou=0.06 | 35.3 % | `signflip_full_f0125` | 28.9 % | 6.4 pp | 0.213 | 0.653 |
| 1C′ stall d0.25/ms0.3 | 25.5 % | `signflip_full_f0125` | 28.9 % | 3.4 pp | 0.089 | 0.653 |

**One usable matched pair is thin for the paper's headline comparison.** The gaps themselves
are good (3–6 pp), but a single Tier 2 config against three Tier 1 configs cannot support a
claim about *mechanism* independent of *which particular attack*. A short α-refinement sweep
for 2B (§9) would give a second, mechanistically different in-band config and make the
comparison much stronger.

---

## 6. Attacker survival and guard trigger rates

The `guard exposure` column is the fraction of steps the attacker spent below the guard's
0.6 m threshold — i.e. steps on which its action was being blended back toward honest.
It is **derived from the trajectory** rather than read from the agent's counters, for a
reason worth recording: the per-agent counters are cleared by `reset()`, and `collect.py`
reads them *after* the final episode's reset, so every run in this sweep reported
`guard_trigger_rate = 0` from a counter that had just been zeroed. Both were fixed — the
agents now keep lifetime counters, and `metrics.py` reconstructs the exposure from
`positions`, which needs no re-simulation and cannot drift from the guard's actual firing
condition.

| variant | attacker survival | guard exposure | reading |
| --- | --- | --- | --- |
| 2B waypoint f=0.125 | **91–100 %** | **10–22 %** | The clean attack. Flies where it wants, stays up, barely needs rescuing. |
| 2C liar | 96–100 % | 4–6 % | Flies honestly, so of course it survives. |
| 2A signflip | 45–73 % | **66–83 %** | **A problem — see below.** |
| 2D collusion `pincer` | 96 % | 10 % | Clean. |
| 2D collusion `same_waypoint` | 67 % | 30 % | Borderline. |
| 2D collusion `mirror_pair` | **37 %** | **93 %** | Worst of both: mostly guarded *and* mostly dead. |

**2A and 2D `mirror_pair` are substantially "honest flight with a safety net".** A sign-flip
drone spends two-thirds to four-fifths of the episode below 0.6 m, which means the guard is
blending its action back toward honest for most of the run — and it *still* only survives
45–73 % of episodes. The honest description of `signflip_full_f0125` (the one in-band config)
is "an attack that immediately drives its drone into the floor and is then continuously
rescued", not "an attack that flies adversarially".

That does not invalidate the numbers, but it changes what they mean, and it is a caveat that
belongs next to the matched-damage table. Two options if a cleaner 2A is wanted: reduce
`scale` to 0.5 so the flip is less violent, or lower `guard_altitude` so the guard intervenes
less and the attack is allowed to be what it is. **The `waypoint` variant is by a clear margin
the better-behaved attack mechanism** — high survival, low guard dependence, low floor time
— and is the one I would build the paper's Tier 2 story around, once its α is brought into
the band.

---

## 7. Behavioural observations

Descriptors and plain-language descriptions in
[`results/tier2_behaviour.json`](../results/tier2_behaviour.json); 19 figures for seed 1 in
[`results/figures/`](../results/figures/) (`behaviour_t2_*.png`). Clean reference for seed 1:
goal error 0.796 m, direction persistence 0.685, floor fraction 0.7 %.

**2A signflip — violent tumbling.** Wobbles at 4.7–6.0× the clean per-step velocity change,
drifts with persistence 0.89–0.94, and spends 26–30 % of the episode near the floor. Partial
(body-space) and full (motor-space) look behaviourally similar despite a 3× difference in
action deviation (0.215 vs 0.653) — the flip destabilises attitude either way.

**2B waypoint — the most legible attack.** At α=1.0 the description is *only* "drifts
persistently away from its goal (6.8× clean goal error, direction autocorrelation 0.96)" —
no wobble term at all, and just 5.9 % floor time. It flies smoothly and purposefully to the
wrong place. This is what an intentional attack should look like, and it is visually the
clearest contrast with Tier 1's OU drift (which reaches similar persistence, 0.92–0.97, by
accident rather than design).

**2C liar — invisible, exactly as predicted.** All six configs return *"behaviourally
indistinguishable from a clean drone at this severity"*. In
[`behaviour_t2_liar_s1_mirror_f0125.png`](../results/figures/behaviour_t2_liar_s1_mirror_f0125.png)
the action trace shows a single line — executed and honest commands are identical, so the
orange curve sits exactly on the blue — and the two goal-error curves overlap inside their
bands. Its action deviation is **exactly 0.000** in all six configs. §4 predicted this from
first principles; the data confirms it.

**2D collusion — mode matters more than coordination.** `pincer` and `same_waypoint` inherit
2B's smooth directed drift (persistence 0.93–0.94, 2–5 % floor time). `mirror_pair` inherits
2A's tumbling instead (34 % floor time, 37 % survival) because it *is* a sign flip. So "two
colluding attackers" is not one behaviour; it is whichever base mechanism the mode uses.

---

## 8. Test results

`pytest byzantine_experiments/tests/test_tier2.py` — **31 passed, 3 skipped** (~6 s). The
skips are the efficacy test, which needs the checkpoint (`BYZ_SLOW_TESTS=1` to include).

| # | Spec test | Status |
| --- | --- | --- |
| 1 | Sign-flip correctness | PASS — `a' == -scale·a` on targeted dims, both scales |
| 2 | Partial flip leaves untargeted dims bit-identical | PASS — over 50 random actions |
| 3 | Waypoint direction | PASS — mean cosine of commanded acceleration (gravity removed) with the direction to `w_adv` is > 0; plus a monotonicity check that deviation increases with α and α=0 is exactly honest |
| 4 | Liar: neighbours see the false state | PASS — reconstructed every observer's block from `neighborhood_indices()`; the liar's slot carries the lie, every other slot is bit-exact truth, and the lie provably differs from the truth |
| 4b | Liar's own observation unchanged | PASS — bit-identical to an unwrapped rollout |
| 5 | Liar's actions identical to honest | PASS — `array_equal` over 20 steps, 2 liars |
| 6 | Collusion correlation | PASS — shared waypoint / opposite corners / disjoint flipped motor sets, per mode; ranks assigned deterministically |
| 7 | Determinism, range, isolation | PASS — all 5 canonical configs |
| 8 | Survival guard | PASS — blend 0.0 above threshold, 0.50 ± 0.02 at the ramp midpoint, 1.0 (fully honest) at 0.10 m; no-op when disabled |
| 9 | Efficacy ≥ 20 % | opt-in (`BYZ_SLOW_TESTS=1`); 2C excluded by design (§4) |

Plus Jacobian round-trip, capability-flag isolation (Tier 1 agents must have none of the new
flags), coordination-channel contents, and config validation.

---

## 9. What surprised me

**1. I was wrong about spillover, and the correction matters.** After a single-episode smoke
test showed honest-drone reward dropping from 3.01 to 1.13, I wrote in this document and in
`readme_bz.md` that Tier 2 damages the honest drones where Tier 1 did not. **The 57-run matrix
does not support that.** Honest-drone goal-RMSE degradation across every Tier 2 config is
**−6.8 % to +9.5 %** — the same ±10 % band Tier 1 sat in. The smoke test was one noisy episode
and I over-read it.

So the headline stands for *both* tiers: **in this environment, 1–2 compromised drones out of
8 do not measurably degrade the honest drones' formation-keeping, whether the compromise is a
stochastic fault or a deliberate attack.** The swarm absorbs them.

Two consequences:

- Tier 3 must be trained to induce **coupling** damage (collisions, formation break-up) if
  team-level damage is wanted at all; maximising its own disruption will not produce it. See
  §10 for what this implies for the Tier 3 gate.
- The reward-based spillover figure is unusable and should not be quoted: across runs it
  ranges from **−575 % to +189 %**, the same near-zero-denominator problem as everywhere else.
  Quote the goal-RMSE version.

**2. Honest drones often do slightly *better* when a neighbour is attacking.** Spillover is
negative in 8 of 19 configs, down to −6.8 %, and the behavioural descriptions repeatedly say
honest neighbours hold formation at 0.79–0.98× clean goal error. The mechanism is
straightforward once seen: the attacker leaves the formation, which *reduces* crowding and
collision-avoidance pressure on everyone else. An attack that removes a drone from the
airspace is, locally, a favour to its neighbours.

**3. The liar is the collision attack.** 2C moves goal error barely at all (5.6–11.8 %) and
action deviation not at all (0.000), but it has among the **highest collision deltas in the
whole matrix**: +0.31 to +0.66 per episode, against +0.02 to +0.11 for sign-flip and −0.03 to
+0.03 for waypoint at f = 0.125. The reason is mechanically satisfying: neighbours steer
around where the liar *says* it is, and therefore into where it actually is. So 2C does have
an observable signature — it is just not in anyone's goal error or in the liar's own actions.
It is in the collision statistics of its *victims*. This strengthens the §4 argument
considerably: the detector should be looking at victim-side collision anomalies, and there is
a concrete signal to look for.

**4. The survival guard silently dominated 2A** (§6), and the diagnostic that should have
caught it was itself broken — the per-agent counters were being zeroed by the final episode
reset before `collect.py` read them, so every run reported `guard_trigger_rate = 0.0`. Fixed
by keeping lifetime counters, and made robust by deriving guard exposure from the trajectory
instead (`attacker_below_guard_frac`), which is recoverable from any past run without
re-simulating. Generalisable lesson: **a diagnostic that reads agent state after the episode
loop must not depend on state that `reset()` clears.**

**5. Time-to-impact needed a metrics backfill, not a re-run.** The Tier 1 clean references
predated the cumulative-reward curve, so every Tier 2 time-to-impact came out `n/a`. Rather
than re-collect, `analysis/refresh_metrics.py` recomputes the metrics block in place from the
`.npz` — 4 seconds for the three clean runs, 84 seconds for all 57 Tier 2 runs. Worth having
permanently: derived quantities should never require re-simulation.

With it working, the numbers are informative: **2C never separates from clean noise at all**
(`n/a` in all six configs — the attack is invisible in team reward, consistent with everything
else about it), while 2A and 2D `mirror_pair` separate almost immediately (0.4–0.7 s) because
they crash their own drone in the first second. 2B at f = 0.125 also shows `n/a` despite
100 %+ goal-RMSE degradation — its damage is confined to the attacker's own trajectory, which
is only 1/8 of the team-reward average.

---

## 10. What this implies for Tier 3

**A correction to an earlier draft of this section.** I first wrote that the Tier 3 spec's
efficacy gate ("team reward degradation ≥ 30 %") looked *unreachable*. That was wrong, and the
error is instructive. It came from reading "team damage" as the honest drones' degradation
(±10 %, as above). On the spec's **literal** metric — mean episode reward across all 8 drones —
every Tier 1 and Tier 2 config blows past 30 %:

| config | all-8 episode reward | "team reward degradation" |
| --- | --- | --- |
| clean | +0.861 | — |
| T1 gaussian σ=0.25, f=0.125 (a *fault*) | −0.946 | **214 %** |
| T1 OU σ_ou=0.2, f=0.125 | −3.890 | **663 %** |
| T2 waypoint α=1.0, f=0.125 | −3.618 | **602 %** |
| T2 collusion pincer, f=0.25 | −8.094 | **1273 %** |

Both facts are true at once, and the reconciliation is the important part: the all-8 mean is
dominated by the **compromised drone's own** reward collapse. One drone of eight going from
+1 to −20 moves the eight-drone mean by −2.6, which against a clean mean of 0.861 reads as
−300 %. The honest six barely move.

So the ≥30 % gate is not a barrier — **it is a non-discriminating filter**. A Tier 1 Gaussian
noise fault passes the Tier 3 "adversary" efficacy gate at 214 %. Anything that damages its own
drone passes it. What it does *not* do is distinguish an adversary from a broken drone, which
is the one thing Tier 3 needs it to do.

### The part that genuinely needs changing: the matched-damage verification

The spec asks for the matched triple to be verified with *"a Mann-Whitney U test across the
50 episodes per condition (p > 0.05 for the pairwise degradation comparisons)"*.

**That test is statistically vacuous at n = 50.** Per-episode reward has σ ≈ 3.31 against a
clean mean of 0.861, so the minimum difference detectable at 80 % power is

```
2.8 · σ / √n = 2.8 · 3.31 / √50 = 1.31 reward units = 152 % of the clean mean
```

Any two conditions whose degradation differs by less than **~152 percentage points** will
return p > 0.05 and be declared "indistinguishable". The spec's own illustrative triple
(34 % / 33 % / 35 %) would pass — but so would 34 % versus 180 %. A non-significant result
from an underpowered test is not evidence of equivalence; it is absence of evidence.

Raising n helps only slowly: n = 200 → 76 pp, n = 400 → 54 pp, n = 800 → 38 pp.

Three changes would fix it, in order of value:

1. **Match on goal-RMSE degradation, not reward.** Seed-to-seed spread is 3–12 pp rather than
   ±94–112 pp, so a 20–40 % band is meaningfully resolvable and matched pairs mean something.
   Report reward degradation alongside, with its CI, as a secondary.
2. **Replace the null test with an equivalence test.** Declare a margin first (e.g. "matched"
   = within ±10 pp of goal-RMSE degradation) and use TOST. This tests what the design actually
   claims, instead of relying on failure to reject.
3. **If reward must be the matching metric, raise episodes for the matched triple only** to
   n ≈ 400. That is 3 conditions × 3 seeds × 400 episodes ≈ 2 h of rollout — cheap next to the
   6 training runs, and only needed for the headline table.

### What the spec gets right and should keep

- The other three gate criteria **are** discriminating and should stay. Adversary survival
  ≥ 70 % is a real filter: Tier 2's sign-flip fails it (45–73 %) while waypoint passes
  (91–100 %). The action-magnitude ratio ≤ 1.5× is exactly the right stealth constraint.
- The failure-mode table already anticipates both "adversary does nothing" and "degradation
  > 80 %", and the fallback to an unconstrained adversary is sound.
- Freezing the honest team, and test 1 asserting bit-identical honest weights, is the right
  highest-risk-first ordering.

### One number to carry in

The spec's illustrative matched triple quotes `tier1_ou σ_ou=0.2 f=0.125 → 34 % degradation`.
Measured, that config gives **663 % ± 111** on reward and **60.9 %** on goal RMSE. The
illustrative numbers assumed a reward scale this environment does not have; do not use them as
targets.
