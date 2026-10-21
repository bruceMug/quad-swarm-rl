# TIER 2 — Structured Byzantine Attacks (Untrained)

**Audience:** claude code
**Prerequisite:** `TIER0_INFRASTRUCTURE.md` and `TIER1_STOCHASTIC.md` complete.

> **Amendments.** Changes decided after this spec was written are recorded in dated blocks
> next to the text they change; the original text is kept. **A1 (2026-09-26, decision D1):**
> the 20–40 % band is replaced by family-level validation — see *severity calibration* and
> *Validation gate*.

---

## What this tier is for

Tier 2 is the first tier with **intent**. These agents have a goal — disrupt the team — but
they pursue it with a fixed, hand-designed rule rather than a learned strategy. They are
adversarial without being adaptive.

This tier does three jobs:

1. **Oracle test for the detector.** Tier 2 attacks produce strongly structured behavioural
   signatures. If your spectral detector cannot catch a sign-flip agent, the detector is
   broken and you should stop and debug before spending 72 GPU-hours on Tier 3.
2. **The interesting middle case.** Tier 2 agents are goal-directed but not stealthy. The
   paper's story becomes much stronger if you can show a monotone difficulty gradient:
   Tier 2 easy to catch, Tier 3 hard, with your method catching both and baselines only
   catching Tier 2.
3. **The lying agent.** This is where `perturb_broadcast_state()` finally gets used — Tier 2
   introduces agents that misreport their state to neighbours.

---

## Implementation

Create `byzantine_experiments/agents/tier2_structured.py`. Four variants.

### Variant 2A — Sign flip (`tier2_signflip`)

The canonical Byzantine attack from the federated-learning literature.

```
a' = clip(-scale * a, low, high)
```

Config: `scale` ∈ {0.5, 1.0}, `dims` — which action dimensions to flip (default: all).
Also implement partial flips (e.g. `dims=[1,2]` flips roll/pitch but leaves thrust honest,
which keeps the drone airborne while destroying its lateral coordination).

Full sign-flip on all dimensions including thrust will usually crash the drone quickly. That
is a legitimate attack but a *boring* one — it is trivially detectable and it removes the
agent from the episode. **Partial flip is the more interesting configuration** and should be
your canonical 2A setting. Measure both.

### Variant 2B — Waypoint pull (`tier2_waypoint_pull`)

The agent ignores the team goal and steers toward a fixed adversarial waypoint, dragging
neighbours off-formation through the collision-avoidance coupling.

```
direction = normalise(w_adv - p_self)
a' = clip((1 - alpha) * a + alpha * to_action(direction), low, high)
```

Config: `w_adv` (adversarial waypoint, default a corner of the arena far from the goal),
`alpha` ∈ {0.3, 0.6, 1.0} (blend weight — how committed the attacker is).

You will need a mapping from a desired direction vector to the env's action space. Look at
`gym_art/quadrotor_multi/quadrotor_control.py` for how the controller converts desired
acceleration to motor commands, and reuse that logic rather than inventing one. Document
whatever you find in the design doc — this mapping is the fiddliest part of Tier 2.

This variant reads the agent's **own position** to compute the direction. That is allowed —
unlike Tier 1, Tier 2 agents may be goal-directed. It must still not read neighbours' states.

### Variant 2C — State misreporting (`tier2_liar`)

The agent flies honestly but **lies about where it is**. This is the purest form of the
Byzantine problem in the proposal's original framing: a drone whose actions are fine but
whose communications are false.

Override `perturb_broadcast_state()`:

```
reported_state = true_state + offset      # constant spatial lie
   or: reported_state = true_state + delayed(true_state, k)   # stale-data lie
   or: reported_state = mirror(true_state) # reflected position
```

Config: `lie_mode` ∈ {`offset`, `stale`, `mirror`}, `magnitude`, `delay_k`.

**Critical implementation note.** This requires the wrapper to intercept the neighbour
observation block in `get_state.py`, not just the action. Read `get_state.py` carefully and
find where each agent's observation of its neighbours is assembled. The cleanest hook is a
post-processing step on the observation array: after the env builds observations, overwrite
the slice corresponding to agent `i`'s reported state inside every *other* agent's
observation vector.

You must document the exact observation layout (which indices correspond to which neighbour's
position/velocity) in the design doc. Get this wrong and the attack silently does nothing.

**Flag this for the researcher:** 2C is theoretically important but creates a measurement
problem for the spectral hypothesis. If the liar's *own* actions are honest, its own policy
graph looks normal — the spectral fingerprint of the liar will be clean, and detection must
instead come from the *victims'* distorted policy graphs. This is a genuinely interesting
result either way, but it needs to be understood before the experiment, not discovered
afterward. Include a note on this in the design doc and raise it explicitly.

### Variant 2D — Coordinated collusion (`tier2_collusion`)

Two compromised agents apply correlated perturbations (e.g. both pull toward the same
adversarial waypoint, or execute a synchronised pincer). Only meaningful at `f = 0.25`.

Config: `mode` ∈ {`same_waypoint`, `pincer`, `mirror_pair`}.

This variant tests whether your set-structured detector handles multiple simultaneous
Byzantine agents — which is exactly what siForest's set semantics are designed for, so it is
worth having.

---

## Design constraints

1. **Deterministic given a seed.** Tier 2 is rule-based. Any randomness (e.g. waypoint
   sampling) must be seeded and reproducible.
2. **Attacker stays airborne.** An attack that immediately crashes the attacker removes it
   from the episode and provides no detection signal. Add a `survival_guard` option that
   blends toward the honest action when altitude drops below a threshold. Report how often
   it triggers.
3. **May read own state, may not read team state.** Exception: 2D collusion agents may
   coordinate with each other but must not read honest agents' states.
4. **Attack must actually work.** Verify team reward degradation ≥ 20% for every canonical
   config. An attack that does nothing is not a Byzantine agent.

---

## Experimental matrix

Still no training.

| Factor | Values |
|--------|--------|
| Baseline seed | `exp1_baseline_seed{1,2,3}` |
| Variant | 2A (partial + full), 2B (α ∈ {0.3, 0.6, 1.0}), 2C (3 lie modes), 2D (f=0.25 only) |
| f | 0.125, 0.25 (2D: 0.25 only) |
| Episodes | 50 |

Roughly 60–70 rollout runs. Script: `scripts/byz_rollout_tier2.sh`.

---

## Characterisation

Same metrics as Tier 1 (team reward degradation, collision rate, formation RMSE, action
deviation), plus:

- **Attacker survival rate** — fraction of episodes the attacker stays airborne to the end
- **Victim spillover** — degradation in *honest* agents' individual rewards, which quantifies
  how much damage propagates through coordination coupling
- **Time-to-impact** — timesteps from episode start to when team reward diverges > 2σ from
  the clean baseline

The severity calibration matters here too. Find Tier 2 configs that land in the same **20–40%
team degradation band** as your chosen Tier 1 configs. Matched-damage pairs are what make the
paper's central comparison clean: *same damage, different mechanism, does the detector tell
them apart?*

> **Amendment A1 — 2026-09-26 (decision D1). The 20–40 % band is retired**, for the reasons
> given in the Tier 1 spec's A1 (seeding-bug inflation; no v2 configuration reaches 20 % on the
> honest drones — best Tier 2 family mean +5.7 %, best single run +13.1 %). A Tier 2 attack
> family is *validated* when its runs together significantly damage the honest team (paired
> honest goal-RMSE, one-sample t-test over runs, mean > 0, p < 0.05;
> `analysis/family_validation.py`). Result: **`liar` validated** (+5.7 %, 16 of 18 runs
> positive, p = 2×10⁻⁵). `signflip` (−2.2 %, p = 0.011) and `collusion` (−2.8 %, p = 0.043)
> significantly *reduce* the honest drones' goal error (why is not established; the attacker
> often crashing out early is one candidate), and `waypoint` has no family-level effect. Matched
> pairs are chosen by the pre-registered matched-triple rule (`analysis/matched_triple.py`),
> not by a band; the legacy matched-pair table in `results_v2/tier2_calibration.md` is kept
> for the record only.
>
> **Clarification — 2026-10-21.** As for Tier 1: the attacker-side figures (its own damage,
> survival, guard exposure) replicate in v2; the honest-drone figures did not, and on v2 data
> `collusion` changes sign (legacy +5.8 to +7.9 %, v2 −2.8 %). See `design/tier2_design.md` §0.

---

## Tests — `byzantine_experiments/tests/test_tier2.py`

1. Sign-flip correctness: `a' == -scale * a` on the specified dims, honest elsewhere
2. Partial flip leaves untargeted dims bit-identical
3. Waypoint pull: direction vector points toward `w_adv` (cosine similarity > 0)
4. Liar: neighbour observations contain the false state; the liar's own obs is unchanged
5. Liar: the liar's executed actions are identical to the honest policy's
6. Collusion: the two compromised agents' perturbations are correlated as specified
7. Determinism, range compliance, isolation (as Tier 0/1)
8. Survival guard fires when altitude drops below threshold
9. Efficacy: every canonical config degrades team reward ≥ 20%
   *(Unchanged by A1: this remains a smoke check that each action-space attack does something,
   on the all-drone team reward over 3 episodes. It is not the validation criterion.)*

Tests 4 and 5 together are the definition of a state-misreporting attack. Write them first
for 2C.

---

## Validation gate

All tests pass; matrix complete; at least one config per variant lands in the 20–40%
degradation band; the observation-layout mapping for 2C is documented and verified.

> **Amended (A1, 2026-09-26):** "at least one config per variant in the 20–40 % band" is
> replaced by "at least one attack family validated at the family level, with every family's
> status reported". **Met:** `liar`.

---

## Required output: `byzantine_experiments/design/tier2_design.md`

Include:

1. **The four variants as implemented** — equations, defaults, and reasoning.
2. **The direction→action mapping for 2B** — what you found in `quadrotor_control.py`, how
   you reused it, and its limitations.
3. **The observation layout for 2C** — the exact index mapping of the neighbour block in
   `get_state.py`, quoted from source. This is the single most reusable finding in the doc.
4. **The 2C measurement problem** — your analysis of whether a state-liar's own policy graph
   carries any signal, and what that implies for the detection experiment. Raise this
   explicitly rather than burying it.
5. **Severity calibration table** — full matrix, plus matched-damage pairs against Tier 1.
6. **Attacker survival and guard trigger rates.**
7. **Behavioural observations** — render 5 episodes per variant, describe what the attack
   looks like. Does the swarm visibly break? How?
8. **Test results** with empirical numbers.
9. **Anything that surprised you.**
