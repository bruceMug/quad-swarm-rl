# Tier 3 — Trained Adversarial Policy: design notes

Implementation record for [specs/TIER3_ADVERSARIAL.md](../specs/TIER3_ADVERSARIAL.md).
Branch `feature/byzantine`.

## Current state (2026-09-26) — read this first

Everything before §6.18 was measured with a collector whose runs at one `--seed` did not share
scenarios; §6.18 fixed it and every number from §6.19 on comes from the re-collected, per-episode
**paired** corpus (`results_v2/`). The legacy header below is kept for the record and is
**superseded**.

* **Best adversary:** `adv_seed3_stealth0_lieonly` (observation channel, flies the honest policy,
  lies in its broadcast) — **+7.4 % [+3.1, +11.7]**, p = 0.0007, n = 200 paired, survival 98 %
  (§6.23). **Fails the 10 % gate**; statistically indistinguishable from the best hand-designed
  attack at equal n (+6.3 %, §6.24).
* **How it hurts** (§6.30): it makes itself invisible — broadcasts a position ~17 m away, mostly
  under the floor — and its neighbours fly into it: collisions involving its slot rise
  five-fold while honest–honest collisions do not change.
* **Decisions D1–D5 (2026-09-26)** and what they built: §6.25–6.29. Spec amendments A1–A6 are
  recorded in the spec itself.
* **Remaining work and exact commands:** `design/tier3_audit.md` §6 and `readme_bz.md`.

### Where each required output lives (spec "Required output")

| spec item | where |
|---|---|
| 1 frozen-team mechanism | §1 |
| 2 reward design + component plots | §2; plots `results_v2/tier3_report/rewards_*.png` (`tier3_report rewards`) |
| 3 efficacy table, all adversaries, five criteria | §6.25; `results_v2/tier3_efficacy.md` |
| 4 cross-seed 3 × 3 matrix | §6.29; `results_v2/tier3_transfer.md` |
| 5 matched triple + TOST | §6.27 (pre-registration); `results_v2/matched_triple.md` |
| 6 stealth analysis + histogram | §6.30; `results_v2/tier3_report/stealth_*.{md,png}` |
| 7 behavioural observations | §6.30; `results_v2/tier3_report/behaviour_*.{md,png}` |
| 8 compute accounting | §6.30; `results_v2/tier3_report/compute.md` |
| 9 test results | §6.30 |
| 10 surprises | throughout; §6.30 (the invisibility mechanism) |

---

## Legacy header (as of 2026-09-24; superseded — see above)
 Four adversaries trained (2 × f=0.125, 2 × f=0.25), **none passes
the efficacy gate** — 0 of 28 checkpoints, and not one does statistically significant damage.
Round 3 (observation channel, seed 1) also fails — 0 of 42 checkpoints. Round 4 moved to
baseline seed 3, where hand-designed lies do clear the gate; its objective is flat too.

**§6.16 is the current headline, and it supersedes every earlier one in this document.**
Three independent clean rollouts of an identical configuration disagree at p = 0.006, giving a
**run-level variance component of ~7.2 %** that appears in none of the confidence intervals
this project has reported. Folding it in turns **20 significant findings into 0, across all
245 attacked rollouts in Tiers 1–3** — Tier 1's canonical severities, Tier 2's `mirror` lie,
the channel-ceiling result and every Tier 3 number included. Nothing here should be quoted
until §6.17's replicate experiment runs.

The tier's own result, now a secondary matter: five rounds, three baselines, two channels,
~1.0B env steps, no trained adversary clears the gate. The confirmatory measurement of the
best checkpoint is **+2.6 % [−6.7, +12.0], p = 0.58** (§6.15), against +29.7 % exploratory.

Earlier headlines, superseded: §5 (channel beats compromised fraction), §6.7 (baseline seed
explains more variance than channel), §6.10 (information asymmetry — real, but shown
insufficient in §6.12), §6.14 (winner's curse — correct, but not the main problem).

**Still no upstream files modified.** The `--adv_*` flags live in
[`adversary/env_factory.py`](../adversary/env_factory.py), not in `quadrotor_params.py`.

---

## 1. The frozen-team mechanism

### Sample Factory v2.1.1 cannot express Tier 3's requirement

The spec asks to "investigate Sample Factory v2.1.1's multi-policy support (`--num_policies`)
— it may support this natively via policy-to-agent mapping. If it does, use it." It does not,
for two independent reasons, both verified against the installed source:

**1. Per-agent mapping exists but is random, not addressable.** The only mechanism is
`AgentPolicyMapping` (`algo/utils/agent_policy_mapping.py`). With
`pbt_mix_policies_in_one_env` enabled it does:

```python
self.env_policies[env_idx] = [self._sample_policy() for _ in range(self.num_agents)]
...
def _sample_policy(self):
    return self.rng.randint(0, self.num_policies)
```

— a uniform random draw per agent, resampled every 10 episodes. There is no index→policy
API. The class docstring carries the TODO *"would be nice to have a mechanism to provide
custom mappings through API"*, and it is instantiated hardcoded at
`algo/sampling/non_batched_sampling.py:369`, so there is no injection point either.

In **batched** mode it is worse: `batched_sampling.py:122` sets
`self.policy_id = worker_idx % self.cfg.num_policies` — one policy per *worker*, and Sample
Factory warns explicitly that batched mode "does not allow us to use multiple different
policies in the same env".

**2. There is no way to freeze a policy.** Sample Factory has no `--frozen_policies` flag or
equivalent. With `num_policies=2`, both policies get a Learner and both update — which is
exactly the self-play the spec forbids.

### What was built instead: the honest team is *environment*, not opponent

[`adversary/frozen_team.py`](../adversary/frozen_team.py) defines `FrozenTeamEnv`, which
presents the 8-drone swarm to Sample Factory as an **`n_adv`-agent environment**. The other
`8 − n_adv` drones are driven inside `step()` by `FrozenTeamPolicy` — the baseline checkpoint
loaded once, `eval()`, `requires_grad_(False)`, called under `@torch.no_grad()`.

This is stronger than the native path would have been. "The honest weights did not update" is
not a property to verify after the fact:

- Sample Factory is told `num_agents = n_adv`, so its learner **cannot see** the honest slice.
- `--num_policies` stays at 1. There is exactly one trainable policy in the process.
- There is no optimiser, no gradient, and no Learner attached to the honest weights anywhere.

Evidence from the smoke run: Sample Factory reports
`EnvInfo(..., num_agents=1, ...)` for `quadrotor_multi_adv`, and the adversary writes its
checkpoints to its own `train_dir`, leaving the baseline's three `.pth` files byte-identical.

### Three integration details that cost time and would cost it again

1. **The registered env factory takes four positional args**, `(full_env_name, cfg,
   env_config, render_mode)` — matching `swarm_rl`'s own `make_quadrotor_env`. A
   `(cfg, ...)` signature fails at rollout-worker start with
   `TypeError: takes from 1 to 3 positional arguments but 4 were given`.
2. **`create_actor_critic` needs a Dict observation space.** In the eval path that happens
   implicitly (`BatchedVecEnv` inserts `BatchedDictObservationsWrapper`); here the env is raw,
   so the Box is wrapped as `Dict({'obs': Box})` explicitly. Without it:
   `AttributeError: 'Box' object has no attribute 'keys'`.
3. **The quad encoder must be registered before building the actor-critic**, or
   `create_actor_critic` silently builds a generic MLP and the checkpoint fails to load with a
   state-dict mismatch against `actor_encoder.neighbor_encoder.*`. `frozen_team.py` calls
   `ensure_swarm_components_registered()` itself so this cannot be forgotten by a caller.

Also: `--env` is `required=True` at parser construction, so `parser.set_defaults(env=...)`
cannot supply it; `train_adv.py` injects it into `argv` instead. And `cfg.policy_index` is an
**evaluation-only** key, absent during training — the baseline is single-policy, so
`checkpoint_p0` is used.

---

## 2. Reward design — two corrections made before training

The spec's reward is

```
r_adv = -w_team * r_team_mean - w_crash * I[crashed] + w_alive * I[alive]
        - w_stealth * max(0, ||a_adv - mu_honest|| - epsilon)
```

with suggested `w_team=1.0, w_crash=10.0, w_alive=0.1`. Implemented in
[`adversary/reward.py`](../adversary/reward.py), with two departures — both **measured**, not
guessed, using [`adversary/calibrate_reward.py`](../adversary/calibrate_reward.py).

### Correction 1: the crash penalty as written drowns the objective 2622 : 1

`I[adversary crashed]` reads as a per-step indicator, but "crashed" is a **persistent state**
— once a drone is on the floor it stays flagged. Charging `w_crash` every subsequent step
makes a single crash at t=100 cost `10 × 1400 = 14000`.

Measured, episode-integrated (1501-step episodes, random adversary):

| term | per-step mean | \|episode total\| | share |
| --- | --- | --- | --- |
| team (`w_team=1`) | −0.00090 | **4.29** | 0.0 % |
| crash (`w_crash=10`, per-step) | −8.00000 | **12008.00** | **99.7 %** |
| alive (`w_alive=0.1`) | +0.02000 | 30.02 | 0.2 % |

At **2798 : 1** the objective is 0.04 % of the reward signal. PPO would learn "hover, never
crash" and nothing else — precisely the spec's own failure mode *"Adversary does nothing
(< 10 % degradation)"*, which the failure table attributes to the stealth constraint or the
learning rate. It is neither; it is the crash term.

**Fix:** charge `w_crash` once per crash **onset** (`crash_penalty_mode='onset'`, the
default). One crash still costs ~2× an entire episode's achievable team damage for a
competent adversary. `'per_step'` reproduces the literal reading.

### Correction 2: scale so the objective is visible

Per-step team reward in this env has magnitude ~0.003, so even in onset mode the balance at
`w_team=1` is 32.6 : 1 against the objective. Calibrated results:

| configuration | crash : team | verdict |
| --- | --- | --- |
| spec literal (`w_team=1`, per-step crash, `w_alive=0.1`) | **2798 : 1** | unusable |
| onset crash, `w_alive=0.01`, `w_team=1` | 32.6 : 1 | workable but skewed |
| onset crash, `w_alive=0.01`, **`w_team=20`** | **1.5 : 1** | **balanced** |

Defaults are therefore `w_team=20.0, w_crash=10.0, w_alive=0.01, crash_penalty_mode='onset'`.
Note the 1.5 : 1 is measured against a *randomly flailing* adversary that crashes 15× per
episode; a competent one crashes far less, so the objective dominates once it stops suiciding
— which is the gradient we want.

**Run the calibrator before changing any weight.** It takes ~20 s:

```bash
python -m byzantine_experiments.adversary.calibrate_reward --w-team 20 --w-crash 10
```

### Correction 3 (ambiguity, not error): `team_reward_mode='honest'`

The spec writes `r_team_mean` without saying whether the mean is over all 8 drones or only
the honest ones. Tier 1/2 settles it (tier2_design.md §10): the all-8 mean is dominated by
the compromised drone's *own* collapse — a Tier 1 **noise fault** scores 214 % "team reward
degradation" on it. An adversary optimising the all-8 mean would score almost entirely by
wrecking itself, which is the degenerate solution `w_crash` then has to fight. Averaging over
the **honest** drones removes the incentive at source. `'all'` is retained for comparison.

### Stealth constraint

`mu_honest` comes from the clean baseline rollouts as the spec requires —
`reward.honest_action_stats()` reads `results/tier1/t1_clean_s*.npz`. Pass it with
`--adv_mu_honest='[...]'`, or use `calibrate_reward --use-clean-mu` to print it.

---

## 3. Training results — round 1 (f = 0.125, one adversary)

Two adversaries trained to ~194M env steps against baseline seed 1:
`adv_seed1_stealth0` (`w_stealth=0`) and `adv_seed1_stealth5` (`w_stealth=5`). Both trained
stably — no divergence, no NaNs, the halved LR was not needed as a rescue.

**Both converged to the same degenerate solution: a saturated bang-bang controller that
destroys its own drone.** Saturation rate (fraction of steps at an action bound) is 83–100 %
at every checkpoint from 6M steps onward. The action-magnitude ratio versus honest drones is
6.4–8.4×. Survival is 0–30 %, against a gate requirement of ≥ 70 %.

### 3.1 The horizon sweep

`HORIZON=1 bash scripts/byz_eval_adv.sh` — 14 rollouts, 7 milestones × 2 adversaries,
20 episodes each, evaluated against the adversary's own baseline seed. Full table in
`results/tier3_gate.md`; the shape of it:

| adversary | ckpt | **honest deg %** | own deg % | all-drone deg % | survival | ratio | sat |
|---|---|---|---|---|---|---|---|
| stealth0 | 6M   | −26.3 | +170.6 | 18.8  | 0 %  | 8.39 | 97 % |
| stealth0 | 100M | +7.5  | +200.7 | 47.6  | 0 %  | 6.86 | 100 % |
| stealth0 | 194M | +13.0 | +188.3 | 48.8  | 0 %  | 6.71 | 100 % |
| stealth5 | 25M  | −9.1  | +380.1 | 91.3  | 20 % | 7.46 | 89 % |
| stealth5 | 150M | +17.7 | +384.0 | 109.0 | 5 %  | 6.56 | 100 % |
| stealth5 | 194M | −0.1  | +319.4 | 79.3  | 0 %  | 7.33 | 99 % |

**0 / 14 pass the gate.** There is no earlier checkpoint that was secretly better — the
horizon question is answered negatively. Honest-drone degradation never exceeds +18 % and is
negative in 8 of 14 cases, i.e. the honest six are sometimes *marginally better off* with a
flailing neighbour than in the clean run (well inside the ±10–18 % band Tiers 1 and 2 also
sat in, so this is noise, not a benefit).

### 3.2 The metric flaw this exposed, and the fix

The gate originally keyed on `goal_rmse_degradation_pct` — goal RMSE averaged over **all
eight** drones. Read that column alone and stealth5@150M looks like a 109 % success.

It is not. **An adversary maximises all-drone RMSE by flying its own drone away. No damage to
anyone else is required.** The split above makes it unambiguous: +384 % on its own drone,
+17.7 % on the honest six. One eighth of the average carrying essentially all of the signal.

This metric was *fine* for Tiers 1 and 2, where severity was a hand-set knob (`p`, `alpha`)
that could not chase the metric. It is not fine against an optimiser, which will find exactly
this. The gate now keys on **honest-drone degradation** (`victim_spillover_goal_rmse_pct`),
with `own deg %` and `all-drone deg %` both printed alongside so the split is always visible.
All-drone is retained but ungated, because the matched-damage triple against Tiers 1/2 is
defined on it. See `analysis/gate_tier3.py` docstring.

This is a correction to my own instrument, caught by inspecting a result that looked too
good. It is worth stating in the paper: *any* Byzantine-severity metric that averages over
the compromised agent is gameable by a learned attacker, and several of the obvious ones do.

### 3.3 Why the adversary self-destructs — the reward's fault, not the optimiser's

The reward is `w_team * (team goal error) − w_crash * (crash onset) + w_alive`. The team term
is computed from honest drones only (`team_reward_mode='honest'`, §2 Correction 3), so
self-destruction earns nothing directly. But it earns *indirectly*: a drone flailing at full
thrust does perturb its neighbours slightly, and that small honest-team signal is the only
gradient available. Meanwhile the crash penalty is paid once (onset mode), so the policy
learns that one crash buys unlimited pre-crash flailing. `w_alive=0.01` — set deliberately
small in §2 so it would not dominate — is far too weak to price survival.

The deeper problem is not tuning. It is that **one drone out of eight, acting only through
its own four motors, has almost no authority over where the other seven go.** The honest
policy's neighbour encoder is an attention module trained on 6 visible neighbours; one
anomalous neighbour is exactly the perturbation it was trained through. This is the same
conclusion Tiers 1 and 2 reached from the other direction. Three independent methods —
stochastic faults, hand-designed attacks, and 200M steps of direct optimisation — now agree.

## 4. Round 2 — f = 0.25 (two colluding adversaries)

Two runs to 200M steps, `--adv_indices=0,1`, `w_alive` raised 0.01 → 0.05, everything else
held at round 1. Sample Factory trained one shared policy driving both drones, so the two
attackers are identical-policy colluders sharing a team-level reward.

**Round 2 failed, with the exact kill signature §2C of readme_bz.md predicted for it**, and
the evaluation confirmed it: **0 of 28 checkpoints pass the gate**, across both rounds.

| | round 1 (f = 0.125) | round 2 (f = 0.25) |
|---|---|---|
| honest deg %, max | +17.7 | **+5.8** |
| honest deg %, negative in | 9 / 14 | **12 / 14** |
| survival, max | 30 % | 25 % |
| rows meeting `honest ≥ 30` | **0** | **0** |
| rows meeting `surv ≥ 70` | **0** | **0** |

Doubling f made the attack *weaker*, which is what §5's channel table predicts for
motors-only attacks (hand-designed ceiling 22.2 % → 15.3 % over the same change).

**And not one of the 28 is statistically distinguishable from clean in the damaging
direction.** The best point estimate in the corpus, +17.2 %, has a 95 % CI of
[−13.9, +48.3] and p = 0.27. Four checkpoints *are* significant — all of them on the
negative side, the honest team doing measurably **better** with the adversary present
(best: −26.4 %, CI [−42.3, −10.4], p = 0.002).

### 4.1 `adv_seed1_stealth0_f250` — the clean run, and the one that matters

| quantity | start | 25 % | 50 % | final | reading |
|---|---|---|---|---|---|
| `adv/r_team` | −24.0 | −40.5 | −24.1 | **−23.1** | **flat over 200M steps** |
| `adv/r_crash` | −273.8 | −11.4 | −10.5 | −10.5 | all of the learning happened here |
| `adv/crash_onsets` | 54.8 | 2.28 | 2.10 | **2.10** | both adversaries crash every episode |
| `adv/survived` | 0.049 | 0.000 | 0.000 | **0.000** | never survives |
| `adv/action_deviation` | 1.57 | 1.97 | 1.99 | 1.99 | saturated |
| `train/act_max` | 4.19 | 29.9 | 31.0 | 29.8 | pre-tanh means pinned at ±30 |

`adv/r_total` improved from −286 to −30, and **every point of that came from `r_crash`**
(−274 → −10.5, i.e. from 55 crash-onsets per episode down to the floor of 2). The team term —
the actual objective — did not move. The policy learned only "crash fewer times", then
stopped.

Round 1 was no better on this axis: its `adv/r_team` went −18.9 → **−28.1**, meaning the
honest team ended up *better off* than at initialisation. **In neither round did the team
objective ever produce usable gradient.**

Raising `w_alive` to 0.05 did not buy survival: it went from 1.000-by-bug in round 1 (see
§4.3) to a measured 0.000 in round 2. Pricing survival is not the lever.

### 4.2 `adv_seed1_stealth5_f250` — invalid, and for an instructive reason

`adv/r_stealth = −9871` against `adv/r_team = −17.8`. **The stealth term outweighs the
objective 555 : 1.** The policy is optimising the stealth penalty essentially to the exclusion
of the attack, so this run says nothing about stealth-constrained adversaries.

This is the *same class* of error as §2's crash-penalty miscalibration — and it slipped
through the calibrator that was built to catch exactly it. Running
`calibrate_reward --w-stealth 5` reports `stealth : team = 5.3 : 1` and
`VERDICT: balanced enough to train`.

**Static reward calibration under a random policy does not predict the balance under a
trained one.** Between the two, the team term shrank ~4.5× (79.3 → 17.8 per episode) while
the stealth term grew ~24× (418 → 9871), because a *trained* policy saturates and a random
one does not. A 5 : 1 ratio became 555 : 1 through training alone.

The fix is not a better static calibration; it is to **watch `|r_stealth| / |r_team|` online**
and kill on it. That row is now in readme_bz.md §2C's kill table.

### 4.3 The gate was reporting point estimates it could not support

Two measurement defects, both found while checking round 2 and both now fixed in
`analysis/gate_tier3.py`:

**Mismatched drone subsets.** `degradation_vs_clean` compared the attacked run's honest
drones (6 of 8 at f = 0.25) against the clean run's `goal_rmse_honest_mean` — which is over
**all 8**, because a clean run has no Byzantine drones. Different formation slots have
different baseline error, so this was a systematic bias. The clean reference is now recomputed
over the *same* indices, from the trajectories.

**No uncertainty at all.** Per-episode goal RMSE on this task has a **42 % coefficient of
variation** (clean seed 1: mean 1.112, sd 0.467, n = 50). That gives:

| comparison | 95 % CI half-width on a degradation % |
|---|---|
| horizon sweep, n = 20 vs clean n = 50 | **± 21.8 pp** |
| full gate, n = 50 vs clean n = 50 | **± 16.5 pp** |
| n = 200 vs n = 200 | ± 8.2 pp |

The gate threshold is 30 %. Reporting a bare point estimate against it, when the interval is
±22 points wide, invites reading noise as signal — and did: round 1's headline "17.7 %" is
p = 0.27. Every honest-damage figure now carries a 95 % CI and a Welch p-value, and
**a run passes only if the point estimate clears 30 % *and* the CI excludes zero.**

Power is adequate where it matters, which is worth stating so the negative result stands:
detecting a *true* 30 % effect at 80 % power needs **n ≈ 31 per arm**, and the gate uses
n = 50. The horizon sweep's n = 20 is under-powered for fine distinctions but not for the
conclusion drawn from it, since nothing there is even directionally positive.

### 4.4 Two round-1 telemetry channels were reading bugged values

Worth recording so nobody re-derives conclusions from them:

* Round 1 `adv/survived` reads a constant **1.000** in training while its own evaluation
  measured 0 %. That is the pre-`_prev_crashed` bug (the value was read after the replay
  wrapper's auto-reset). Round 2's 0.000 is post-fix and real.
* Round 1 `adv/action_deviation` reads **0.000** for stealth0 (logging returned nothing at
  `w_stealth=0`) and **74.2** for stealth5 (the unclipped-action bug). Round 1's stealth5 run
  is contaminated by that bug — `r_stealth = −534,191` — and should be discarded outright.

**`adv_seed1_stealth0_f250` is the only Tier 3 run whose telemetry is fully trustworthy end to
end.** It is the one to quote.

## 5. What actually limits the attack: the channel, not f and not the optimiser

Round 2 raised an obvious worry: the policy saturates (`act_max` ≈ 30, `exploration_loss_coeff`
is 0, so there is no entropy bonus to pull it back), and a saturated tanh policy has almost no
gradient. Maybe the adversary is weak because the optimiser is stuck, not because the attack
is impossible.

Tiers 1 and 2 already contain the control for this, at zero compute cost. Group every attack
by the **channel** it acts through, test each against clean on matched drone indices, and
count how many are *significant* rather than merely large
(`analysis/channel_ceiling.py`, output in `results/channel_ceiling.md`):

| tier | channel | f | runs | max deg % | median | **signif worse** | signif better |
|---|---|---|---|---|---|---|---|
| 2 (hand-designed) | motors only | 0.125 | 27 | 22.2 % | 1.7 % | **2** | 2 |
| 2 (hand-designed) | motors only | 0.250 | 24 | 15.3 % | 3.7 % | **0** | 1 |
| 2 (hand-designed) | observation channel | 0.125 | 9 | **30.4 %** | 2.0 % | **3** | 0 |
| 2 (hand-designed) | observation channel | 0.250 | 9 | 29.9 % | 9.4 % | **3** | 0 |
| 3 (**trained**) | motors only | 0.125 | 14 | 17.2 % | −4.5 % | **0** | 2 |
| 3 (**trained**) | motors only | 0.250 | 14 | 7.7 % | −4.2 % | **0** | 1 |

"Motors only" means the attacker corrupts its own four thrust commands and nothing else —
`signflip`, `waypoint_pull`, `collusion`, **and every Tier 3 adversary**, whose action space
is exactly those four motors. "Observation channel" means it additionally broadcasts false
state, so honest drones act on a corrupted picture: Tier 2's `liar`. No Tier 3 adversary has
that capability; `FrozenTeamEnv` hands the policy motor commands only.

Three things follow, and they redirect the research.

1. **Across 79 motors-only runs — 51 hand-designed plus all 28 trained checkpoints — exactly
   2 do statistically significant damage**, and the ceiling is 22.2 %. Nothing on this channel
   clears the gate, by any method.
2. **The trained adversary is not underperforming; it has nothing to work with.** Its best
   checkpoint reaches 17.2 % (p = 0.27), inside the hand-designed motors-only band, not below
   it. Saturation may cost something at the margin, but it cannot explain a gap that is not
   there. **The optimiser is not the bottleneck.**
3. **Motors-only attacks are as likely to help as to hurt**: 2 significantly worse against
   **6 significantly better** for the honest team. A Byzantine drone that flails and crashes
   removes a crowding constraint from an 8-drone formation, and the remaining drones hold
   station better than in a clean run. In *sign*, motor-authority faults are indistinguishable
   from noise. This also explains why doubling f hurts the attacker.
4. **The channel is the bottleneck.** 6 of 18 `liar` runs do significant damage, and the best
   — `t2_liar_s3_mirror_f0125`, **+30.4 %, p = 0.0007** — *clears the 30 % gate with a
   hand-designed, unoptimised lie.* Corrupting what honest drones **believe** is a different
   kind of attack from flying badly, and it is the only one that works.

## 6. Round 3 — the observation channel (built, not yet trained)

Earlier drafts proposed changing the *objective* next (reward induced honest–honest
collisions). §5 supersedes that: it was a guess, where the channel result is measured, and it
comes with a target already on the board — `t2_liar_s3_mirror_f0125` reaches **+30.4 %,
p = 0.0007** with a hand-designed, unoptimised lie.

It also makes Tier 3's central claim testable for the first time. "A trained adversary beats
hand-designed attacks" has not been falsifiable so far, because every adversary was optimising
on a channel where hand-designed attacks do nothing either. On the observation channel there
is a real hand-designed baseline to beat.

### 6.1 What was built

The adversary's action grows from **4 (motors) to 4 + 6**, the extra six being the state it
broadcasts. New module [`adversary/lies.py`](../adversary/lies.py) holds the two pieces
training and evaluation must agree on: `decode_lie` (action → reported state) and
`apply_lies_to_obs` (reported state → corrupted neighbour blocks).

**The lie is a bounded delta from the truth, not an absolute position:**
`reported = true + max_lie * a`, `a ∈ [-1, 1]`. Three reasons, the first being the one that
matters:

1. **A zero action is a truthful broadcast.** Under an absolute parameterisation, honesty
   would be a specific non-zero output the policy must find and then *track* as the drone
   moves. Here it is the origin. Rounds 1–2 both saturated to the action bounds within 50M
   steps, so behaviour at the extremes is not hypothetical — it has to mean something.
2. Sample Factory's continuous policy emits unbounded Gaussian samples. Executing or scoring
   anything unbounded is what destroyed the round-1 stealth run (§4.2, and §2's clipping fix).
3. It makes the attack budget explicit, so an `epsilon`-style stealth constraint on the *lie*
   is expressible later.

Defaults `max_lie_pos = 10 m`, `max_lie_vel = 6 m/s` span exactly the env's
`clip_neighbor_space_{min,max}_box`. A larger lie is clipped to the same observation, so extra
range would be dead action space the policy has to learn to avoid.

Two configurations, both supported:

| mode | motors | broadcast | what it isolates |
|---|---|---|---|
| `LIES=1` (lie + fly) | learned | learned | the strongest attacker available |
| `LIES=1 LIE_ONLY=1` | **frozen honest policy** | learned | the channel alone — the direct analogue of Tier 2C, whose action stream is clean by construction |

`lie_only` is the scientifically cleaner comparison against Tier 2's `liar`; `lie + fly` is
the stronger attacker. Run both.

### 6.2 Five things that would each have made this silently do nothing

Recorded because every one of them is the kind of bug that produces a *clean-looking negative
result*, which is exactly what this tier is trying to distinguish from a real one.

1. **Timing.** `ByzantineEnvWrapper.step` applies lies to the observation it *returns*, so
   honest drones consume the lie on the next control cycle. `FrozenTeamEnv` now does the same
   thing at the same point — right after `self.env.step()`, corrupting the `_last_obs` that
   `_honest_actions()` reads next step. Applying it anywhere else would make training and
   evaluation different experiments.
2. **Proximity ranks, not agent ids.** `neighborhood_indices()` re-sorts every drone's
   neighbours every step and keeps only the closest few, so the liar's slot in a given
   observer's vector moves, and on some steps it is absent entirely. `apply_lies_to_obs` takes
   the ordering from the env, never reimplements it.
3. **The liar's own observation must be untouched**, matching the Tier 2 wrapper.
4. **Stealth must score the motors only.** `mu_honest` is a distribution over 4 thrusts;
   feeding it a 10-D action would either crash or, worse, silently score the lie as a motor
   deviation — reintroducing the 555 : 1 domination of §4.2.
5. **`lies_about_state` must be per-checkpoint, not class-wide.** The wrapper computes
   `any_lies` from constructed agents. Leaving it True for every Tier 3 agent would route
   rounds 1–2 checkpoints through `_apply_broadcast_lies`, rewriting their neighbour slots
   with recomputed "true" values that need not be bit-identical to what the simulator wrote —
   silently changing the 28 sidecars rounds 1–2 were measured from. It is now read from each
   checkpoint's own `config.json` at construction, before any weights load.

`tests/test_tier3.py` pins all five. The load-bearing one is
`test_the_lie_actually_changes_what_honest_drones_see_and_do`, which asserts the rewrite count
is non-zero, that honest observations change, that the liar's own does not, and that the
frozen policy's *actions* differ on the corrupted observation — a lie that moves the numbers
but not the behaviour would still be a no-op.

### 6.3 Two changes folded in, both justified by rounds 1–2

* **`exploration_loss_coeff = 0.003`** for round-3 runs only (0.0 otherwise, so rounds 1–2
  still reproduce bit-for-bit). §5 shows saturation did not cause the ceiling, but an
  absorbing saturated corner is worth not having, and it costs one flag.
* **Evaluation at n = 50, not 20.** §4.3: n ≈ 31 per arm is the threshold for 80 % power
  against a true 30 % effect. `byz_eval_adv.sh`'s horizon default was raised accordingly.

### 6.4 What to watch

`adv/lie_magnitude_m` is the new one: metres between claim and truth. **A policy that drives
it toward 0 has abandoned the channel** — the round-3 analogue of round 2's flat `adv/r_team`,
and the signal to stop. `adv/broadcast_rewrites` confirms the lie is reaching observers at
all; if it is ~0 while `lie_magnitude` is large, the attack is landing in slots nobody reads.

### 6.5 Training results — the two runs diverged sharply

Both to 200M steps, seed 1, `w_stealth=0`, `exploration_loss_coeff=0.003`.

| | `_lie` (lie + fly) | `_lieonly` (channel only) | rounds 1–2 |
|---|---|---|---|
| `adv/survived` | **0.000** | **1.000** | 0.000–0.235 |
| `adv/crash_onsets` | 1.00 | **0.49** | 1.0–64 |
| `adv/action_deviation` | 1.99 | **0.24** | 1.87–1.99 |
| `train/act_max` | 33.1 | **18.1** | 30–64 |
| `adv/lie_magnitude_m` | 16.6 | 15.9 | n/a |
| `adv/lie_magnitude_max_m` | 17.32 | 17.32 | n/a |
| `adv/broadcast_rewrites` | 2.0M | **9.2M** | n/a |
| `len/len` | 1501 | **1030** | 1501 |
| `adv/r_team` start → final | −12.5 → −17.6 | −12.0 → −14.0 | flat or worse |

**`_lie` reproduced rounds 1–2 exactly.** Given the option to fly badly, the optimiser went
straight back to the saturated bang-bang policy: `act_max` 33, deviation 1.99, and
`adv/r_alive = 0.64` at `w_alive = 0.01` means it is alive for only ~64 of 1501 steps. It
crashes almost immediately and then lies from the floor, which is why its rewrite count is a
fifth of `_lieonly`'s. That is a result about the *reward*, not the channel: motor authority
remains a trap the optimiser falls into whenever it is offered.

**`_lieonly` is the first Tier 3 adversary to look like an attacker rather than a casualty.**
It is the first to meet the survival criterion at all — **1.000 against a ≥ 0.70 gate**, where
every previous run scored 0.000–0.235. Saturation is 0.000 in evaluation against 83–100 % in
rounds 1–2, and action deviation is 0.000 by construction, so the action-ratio criterion is
met trivially. It uses the new channel to its limit: `lie_magnitude_max_m` sits at **17.32 m =
√3 × 10**, full deflection on all three position axes, from 25 % of training onward.

Two readings to keep separate until the evaluation lands:

* `adv/r_team` ends at −14.0 against −12.0 at initialisation, i.e. **still not clearly
  positive progress on the objective.** On its face that is rounds 1–2's failure again.
* But `len/len` fell from 1431 to ~1030, a 31 % drop, and `r_team` is an episode *sum* — so a
  flat per-step objective would show up as a falling total. Shorter episodes with a surviving
  adversary is what honest-drone failure would look like. It is also what a replay-wrapper
  artifact would look like. **The training curves cannot separate these; the rollout can**,
  which is why §6.6 is the decision point and not this table.

Evaluation path verified end to end against the real checkpoint before spending sweep time on
it: `lies_told = 4503` over 3 × 1501 steps (it lies on every step),
`action_deviation_byzantine_mean = 0.000` (it flies the honest action bit-for-bit),
`saturation_rate_byzantine = 0.000`.

### 6.6 Gate results — round 3 fails, but not the way rounds 1–2 did

14 rollouts, n = 50, against baseline seed 1. **0 of 42 checkpoints now pass, across all four
rounds.** But `_lieonly` fails on one criterion only:

| checkpoint | honest % | 95 % CI | p | survival | saturation |
|---|---|---|---|---|---|
| m006 | −9.8 | [−29.3, 9.6] | 0.318 | **100 %** | 0 % |
| m025 | −6.2 | [−21.6, 9.3] | 0.431 | **96 %** | 0 % |
| m050 | −10.0 | [−27.1, 7.1] | 0.247 | **98 %** | 0 % |
| **m075** | **+11.5** | [−6.7, 29.7] | 0.213 | **100 %** | 0 % |
| m100 | −3.6 | [−19.2, 12.1] | 0.652 | **96 %** | 0 % |
| m150 | −3.6 | [−20.7, 13.5] | 0.674 | **98 %** | 0 % |
| m194 | +0.2 | [−17.0, 17.3] | 0.983 | **100 %** | 0 % |

Survival 96–100 % at every checkpoint and saturation 0 % — it **meets the survival and
action-ratio criteria outright**, which no Tier 3 adversary had ever done. It simply does not
damage the honest team: max +11.5 %, p = 0.21.

`_lie` (lie + fly) behaved as its training curves predicted: survival 0 % everywhere,
saturation 77–100 %, own-drone degradation 164–215 %. Its best honest figure, +17.6 %
(p = 0.075 at m075), is the largest in the Tier 3 corpus but comes attached to a drone that
destroys itself, so it fails survival regardless.

### 6.7 The confound that reframes all of this: **baseline seed, not channel**

§5 concluded "the channel is the bottleneck" from *pooled* data. That conclusion was drawn
across three baseline seeds, and the seeds differ by more than the channels do. Splitting the
same corpus (`analysis/channel_ceiling.py`, now reports this by default):

| channel | baseline seed | runs | max deg % | signif worse | clean goal RMSE |
|---|---|---|---|---|---|
| motors only | 1 | 17 | **4.9 %** | 0 | 1.1123 |
| motors only | 2 | 17 | 18.6 % | 0 | 1.0463 |
| motors only | 3 | 17 | 22.2 % | **2** | 1.0085 |
| observation | 1 | 6 | **12.6 %** | 0 | 1.1123 |
| observation | 2 | 6 | 17.0 % | 1 | 1.0463 |
| observation | 3 | 6 | **30.4 %** | **5** | 1.0085 |
| observation (**trained**) | 1 | 14 | 17.6 % | 0 | 1.1123 |

**Round 3 was trained against the most robust baseline of the three.** Seed 1 has *zero*
significant results across all 23 hand-designed attacks on either channel; seed 3 has 7. The
+30.4 % headline that motivated round 3 is a **seed-3** number, and round 3 attacked seed 1 —
chosen for continuity with rounds 1–2, which was the right instinct for comparability and the
wrong one for testing the hypothesis.

The apples-to-apples comparison, seed 1 only, observation channel:

| | max honest degradation |
|---|---|
| hand-designed `liar` | +12.6 % (p = 0.20) |
| **trained `_lieonly`** | **+11.5 % (p = 0.21)** |

The learned adversary **matches the hand-designed ceiling on the baseline it attacked**, to
within noise — the same pattern §5 found on the motors channel. It is still not
underperforming; there was simply nothing to find on seed 1.

Two checks before treating the seed effect as real:

* **Not a denominator artifact.** Seed 3's clean RMSE is *lower* (1.0085 vs 1.1123), which
  inflates a percentage. In absolute metres the gap survives: seed 3's best observation attack
  costs 0.307, seed 1's 0.140 — still 2.2×.
* **Partly, but not mostly, a power artifact.** Seed 3 has the lower coefficient of variation
  (36.2 % vs 42.0 %), so its CIs are ~14 % narrower and significance is easier to reach. That
  cannot explain an effect size 2.4× larger.

**The finding worth writing up:** the most attackable baseline is the one with the *lowest*
clean goal RMSE. The tightest formation is the most brittle — a policy that holds tighter has
less slack to absorb a perturbed member. That is a statement about swarm policies, not about
this attack, and it is the sort of thing the tier was built to discover.

### 6.8 The next run, and why it is one run

`LIES=1 LIE_ONLY=1 bash scripts/byz_train_adv.sh 3 0` — the same round-3 mechanism against
**baseline seed 3**, where hand-designed lies reach 30.4 % with 5 of 6 significant.

This is the first fair test of Tier 3's central claim. "A trained adversary beats hand-designed
attacks" has been untestable for four rounds: rounds 1–2 optimised on a channel where
hand-designed attacks do nothing, and round 3 optimised on a *baseline* where they do nothing.
On seed 3's observation channel there is a real, significant, 30.4 % baseline to beat, and
`_lieonly` has already demonstrated it can survive 100 % of episodes while lying at full
deflection — it just had no purchase on seed 1. (Run and analysed: §6.9–6.11.)

### 6.9 Round 4 trained — flat again, but for a diagnosable reason

`adv_seed3_stealth0_lieonly`, 200M steps, baseline seed 3.

`adv/r_team` slope over training: **−0.111 ± 0.082 per 100M steps — within 2 SE of zero.**
Flat, and the per-step figure (−0.00854 → −0.00876, episode length 1065 → 1026) confirms the
episode-length drop is not masking progress. Survival 0.994 throughout; lie magnitude 16.7 m
against a 17.32 m ceiling; 9.0M broadcast rewrites. The channel is being used to its limit and
the objective does not move.

**A correction worth recording.** The first read of these curves sampled the series at
fractions of training (`v[searchsorted(s, s[-1]*0.5)]`) and reported `adv/r_team = +2.166` at
the halfway point — which would have been the first positive value in the project. It was one
noisy episode; the local mean there is −8.7. `adv/r_team` is logged per finished episode and
individual samples span roughly +5 to −25. `analysis/training_curve.py` now exists so these
curves can only be read as binned means with a slope and a standard error.

### 6.10 Why flat is not the same as "doing nothing" — and what it points at

Put the hand-designed attacks on the adversary's *own* objective scale
(`adv/r_team = −w_team × honest episode reward`, `w_team = 20`), on seed 3:

| | honest episode reward | `adv/r_team` equivalent |
|---|---|---|
| clean, no attack | +1.2716 | **−25.43** |
| **trained, 200M steps** | ~+0.42 | **−7.6 … −9.9** |
| hand-designed `stale` lie | −0.0000 | 0.00 |
| hand-designed `offset` lie | −0.0903 | +1.81 |
| **hand-designed `mirror` lie** | **−0.8321** | **+16.64** |

The adversary is *not* doing nothing: it moves honest reward from +1.27 to ~+0.42, a ~64 %
degradation, and it gets there within the first 10M steps and then plateaus. But a
hand-designed `mirror` lie reaches +16.64 on the same axis. **The trained adversary captures
roughly 40 % of the available damage on the quantity it is directly optimising.**

The objective is not mis-specified — honest reward and honest goal RMSE correlate at
**Pearson +0.915, Spearman +0.903** across the 18 Tier 2 `liar` runs, so optimising reward
does drive the gate metric. The gap is elsewhere.

**It is an information asymmetry, and it is ours.** `mirror` reports `2 × centre − pos`, which
requires the drone's **absolute position**. The hand-designed `liar` is handed exactly that:
`perturb_broadcast_state(true_state, t)` receives absolute pos/vel from the wrapper. The
learned adversary never saw it — the 54-D observation is `pos − goal`, `vel`, `R`, `omega` and
*relative* neighbour offsets.

Measured on a clean seed-3 rollout:

| | |
|---|---|
| linear probe R² recovering absolute x / y / z from the 54-D obs | **0.392 / 0.346 / 0.273** |
| probe RMSE vs. positional spread | ~1.0 m vs ~1.2 m |
| fraction of steps where `mirror` is within the 10 m action budget | **99.2 %** |

**The action space could express the attack; the observation could not specify it.** We were
asking RL to beat a baseline holding strictly more information.

### 6.11 Round 5 — restore information parity (built, not trained)

`lies.py::augment_obs` appends the adversary's own absolute pos/vel to its observation,
54 → 60, clipped to the same ranges the env uses for neighbour observations so the values sit
on the scale the encoder already expects (`normalize_input=False`).

This is **parity, not privilege**: it is precisely what `agents/base.py::EgoState` already
gives Tier 2's goal-directed agents, and it carries nothing about any other drone. Evaluation
uses the existing `needs_ego_state` mechanism, so train and eval read the same six numbers.

Guards, all tested:

* `--adv_ego_state_obs` defaults **False**, so every rounds 1–4 `config.json` — which predates
  the key — still builds a 54-D env and loads its weights unchanged.
* Round-5 runs are named `_lieonlyego`, so their checkpoints and sidecars cannot collide with
  round 4's.
* A round-5 agent sets `needs_ego_state=True` and **raises** if the wrapper did not populate
  `agent.ego`, rather than feeding the policy a 54-D vector its first layer cannot accept.
* `test_absolute_position_is_not_recoverable_from_the_plain_observation` pins the premise: if
  a future observation change made absolute position recoverable, round 5's justification
  would no longer hold.

### 6.12 Round 4 gate + round 5 result — the ego-state fix did not move it

**Round 4** (seed 3, 54-D obs), horizon sweep n = 50. The best checkpoint is real but modest:

| milestone | honest deg % | 95 % CI | p | survival |
|---|---|---|---|---|
| m025 | +16.4 | [0.4, 32.4] | 0.045 | 100 % |
| **m075** | **+29.7** | **[8.0, 51.4]** | **0.008** | 98 % |
| m194 | +22.1 | [4.7, 39.4] | 0.013 | 98 % |

m075 all but reaches the 30 % gate and its CI clears zero -- the first Tier 3 checkpoint to
put significant damage on the honest team. But this is an **exploratory** sweep: 7 checkpoints
against a p<0.05 bar, and m075 is the max-point-estimate winner, so both multiplicity and the
winner's curse inflate it (Bonferroni across 7 looks is alpha = 0.0071; m075's 0.008 misses
it). A headline number needs a confirmatory re-measurement (§6.13).

**Round 5** (seed 3, 60-D obs = 54 + own absolute pos/vel). The training objective is
statistically indistinguishable from round 4:

| | round 4 (54-D) | round 5 (60-D, +ego state) |
|---|---|---|
| `adv/r_team` slope / 100M | −0.111 ± 0.082 (flat) | −0.194 ± 0.090 (flat: implied 0.39 < wobble 1.04) |
| `adv/r_team` peak bin | −7.6 @ 50M | −6.7 @ 30–50M |
| `adv/r_team` per-step (mean) | −0.00834 | −0.00807 |
| survival | 0.994 | 0.994 |
| lie magnitude (m) | 16.8 | 16.2 |

**Giving the adversary its own absolute position -- the thing §6.10 identified as missing --
did not help.** Both curves peak near 50M and sag; neither trends. The information asymmetry
was real (the 54-D observation genuinely lacks absolute position), but closing it was not
sufficient: the policy still does not learn to exploit the channel the way the hand-designed
`mirror` lie does. That is a stronger negative than round 4 alone -- the most obvious
explanation for the trained/hand-designed gap has now been tested and rejected.

Why the hand-designed lie still wins, most likely: `mirror` is a *fixed closed-form function*
of absolute position (`2·centre − pos`) that is coherent across the whole episode, whereas the
policy must discover an equally coherent state-dependent map by gradient descent through a
frozen attention encoder that was trained to be robust to exactly this kind of neighbour
anomaly. Having the input available is necessary, not sufficient.

### 6.13 Round 5 gate results — worse than round 4, and 0 of 7 significant

Horizon sweep, seed 3, n = 50, alongside round 4 for comparison:

| milestone | round 4 (`lieonly`, 54-D) | round 5 (`lieonlyego`, 60-D +ego) |
|---|---|---|
| m006 | +4.4 (p=0.65) | +14.7 (p=0.11) |
| m025 | +16.4 (p=0.045) | +12.8 (p=0.078) |
| m050 | −0.9 (p=0.91) | **+15.9 (p=0.055)** ← round 5 max |
| m075 | **+29.7 (p=0.008)** | +0.6 (p=0.93) |
| m100 | +6.9 (p=0.38) | +12.4 (p=0.17) |
| m150 | +10.0 (p=0.23) | +10.8 (p=0.20) |
| m194 | +22.1 (p=0.013) | +9.9 (p=0.23) |

Round 5: **3 of 7 significant → 0 of 7**, max +29.7 % → +15.9 %. Survival 94–100 % and
saturation 0 % in both. The ego-state observation did not help on the gate metric either, which
closes §6.10's hypothesis from the second direction: the information asymmetry was real, and
supplying the missing information changed nothing measurable.

### 6.14 Putting the +29.7 % in context: pooling and the winner's curse

The horizon sweep is *exploratory* — 7 checkpoints, best reported — so m075's +29.7 % is the
maximum of 7 noisy draws and is biased high. `analysis/confirm_tier3.py --pooled` reports two
unbiased readings instead:

| | round 4 (54-D) | round 5 (60-D +ego) |
|---|---|---|
| per-checkpoint measurement SE | 7.4 pp | 7.4 pp |
| **pooled, no selection** (n = 350 vs clean 50) | **+12.6 % [+0.9, +24.4]** p = 0.035 | **+11.0 % [−0.4, +22.5]** p = 0.059 |
| real between-checkpoint sd (τ) | 7.7 pp | **0.0 pp** |
| best checkpoint | m075 = +29.7 % | m050 = +15.9 % |
| **shrunk estimate of the best** | **+21.5 %** (weight 0.52) | **+11.0 %** (weight 0.00) |

Two things this settles:

* **Round 4's checkpoints genuinely differ** (τ = 7.7 pp exceeds the 7.4 pp measurement SE), so
  m075 is probably really the best one — but only about half of its excess over the pooled mean
  is signal. The empirical-Bayes shrunk estimate, **+21.5 %**, is what a confirmatory sample
  should be expected to find. That is below the 30 % gate.
* **Round 5's checkpoints do not differ at all** (τ = 0.0), so its +15.9 % max is entirely the
  lucky tail of 7 draws around a pooled +11.0 %. There is nothing there to confirm.

The pooled figures are the defensible summary of each training run: **~+12 % and ~+11 % honest
degradation, both marginal at p ≈ 0.04–0.06**, against a 30 % gate and against the
hand-designed `mirror` lie's +30.4 % (p = 0.0007).

A bug worth recording, because it is the third appearance of the same class: `pooled_and_shrunk`
initially compared the attacked runs' **7** honest drones against the clean run's **8** (a clean
run has no Byzantine mask, so reading its own "honest" set returns every drone). That shifted
the pooled estimate from +12.6 % to +11.0 % and every per-checkpoint figure by ~1.5 pp. Same
mismatched-subset error as §4.3, now pinned by
`test_pooled_estimate_restricts_clean_to_the_attacked_honest_drones`.

### 6.15 The confirmatory measurement, and what it exposed

**Confirmatory result for round 4 m075 (n = 200 per arm, fresh sample): +2.6 %
[−6.7, +12.0], p = 0.58.** Against the exploratory +29.7 % and the shrinkage prediction of
+21.5 %. Tier 3 fails the gate unambiguously.

But the attack did not regress -- **the clean reference moved**:

| | n = 50 (Tier 1, Aug 11) | n = 200 (Sep 19) | |
|---|---|---|---|
| clean | 0.9938 | 1.1783 | +18.6 %, **p = 0.004** |
| m075 attacked | 1.2887 | 1.2090 | −6.2 %, p = 0.45 (consistent) |

Ruled out in turn: the stale-`.unwrapped` bug (positions travel 2-4 m in every file), config
drift (identical checkpoint, seed, mode, episode length), replay-buffer starts (initial speed
0.000 m/s and identical formation spread in every block of both arms), and within-run
structure (**no autocorrelation**, Ljung-Box p = 0.96; **no trend**, Spearman p = 0.47).

A claim made and withdrawn during this investigation, recorded because the withdrawal is the
point: a block ANOVA over the n = 200 run gave p = 0.029 and was reported as evidence the run
was "non-stationary". It was one high block out of four, chosen for comparison because it
matched Tier 1's n. Episodes are iid; there is no drift.

### 6.16 The finding that supersedes the rest: run-level variance

A third clean rollout was collected to break the tie (seed 3003, same git SHA, same config,
n = 200). It did not break it -- it changed the question:

| reference | n | mean | |
|---|---|---|---|
| `t1_clean_s3` | 50 | 0.9938 | |
| `confirm_clean_s3_n200` | 200 | 1.1783 | |
| `confirm_clean_s3_rep2_n200` | 200 | 1.0568 | **differs from the one above at p = 0.006** |

**Two rollouts of an identical configuration, same code, differing only in `--seed`, disagree
significantly.** Cochran Q = 11.25 on 2 df (p = 0.004), I² = 82 %, run-level sd
**τ = 7.2 % of the mean**. Dropping the older-SHA run and using only the two same-SHA runs
gives τ = 7.3 %, so this is not code drift.

Since episodes within a run are iid, the structure is a **per-run offset**:

    y_ij = mu + u_i + eps_ij ,   u_i ~ N(0, tau^2)

Different seeds draw different scenario mixes, and a constant offset does not average away
inside its own run -- which is why n = 200 did not fix it and n = 2000 would not either.

**Every comparison in this project is one run against one run, and every reported CI is built
from episode variance alone.** Adding the run-level component to both arms:

| | one-run-vs-one-run CI at n = 50 |
|---|---|
| as reported (episodes only) | ± 13.3 pp |
| with run-level τ | **± 23.9 pp** |

Applied across the corpus -- 245 attacked rollouts, Tiers 1, 2 and 3:

| | significant (p < 0.05, damaging) |
|---|---|
| as currently reported | **20** |
| with run-level variance included | **0** |

Nothing in the corpus survives. That includes Tier 1's canonical severities, Tier 2's
`mirror` lie, the channel-ceiling result, the seed-3 effect, and every Tier 3 number.

### 6.17 What is and is not established

**Established:** run-to-run variance of ~7.2 % exists and is not in any error bar we have
reported. Two identical-configuration rollouts differ at p = 0.006.

**Not established:** that the attacks do nothing. A null obtained by inflating error bars is
"not demonstrated", not "demonstrated absent". τ itself rests on three runs and is imprecise.

**Not the fix:** a global inflation factor. The fix is **replicate runs per condition**, with
each run's mean as one observation -- no variance decomposition, no assumption that seed-3's
τ transfers to other seeds or to attacked conditions.

`scripts/byz_replicate.sh` + `analysis/replicate_test.py` do this. Replicates buy more than
episodes here: 6 runs × 50 episodes pins τ far better than 2 × 150 for the same compute. The
first thing to run is the corpus's **strongest** attack (Tier 2 `mirror` liar on seed 3,
nominally +30.4 %) with 3 replicates per arm. If the strongest attack cannot be distinguished
from clean once run identity is respected, the tier-wide negative is real and reportable. If
it can, τ is over-estimated and the corpus needs re-analysing rather than discarding.

### 6.18 The seeding fix, part two: per-episode, not per-run

The first seeding fix (2026-09-21) made two rollouts at one seed start identically. It did not
make a clean run and an attacked run *stay* paired. Measured on a clean/`mirror` pair at seed
4242, n = 50: episodes 0–2 had identical spawns, 46 of the remaining 47 did not, and the
per-episode correlation of honest goal RMSE between arms was **−0.08** — no pairing at all.
That run's "+30.4 %, p = 0.003" is therefore an unpaired one-run-vs-one-run comparison and is
not reportable.

Cause: collisions draw from the global `np.random` stream (`collisions/quadrotors.py`, 9 normal
draws per drone–drone contact; `collisions/room.py`, several uniforms per wall hit). The
attacked arm collided ~2.3× as often, so it advanced that stream differently and every later
scenario diverged. (The experience-replay buffer was checked and ruled out: 0 replayed episode
starts across every run examined, 12 runs including the high-collision ones.)

Fix: `collect.py::episode_seed(seed, i)` re-seeds every RNG immediately before *each* episode's
reset, so episode *i*'s scenario is a function of *(seed, i)* alone. Verified on a 10-episode
clean/`mirror` pair: **10/10 identical spawns** despite `mirror` colliding 5× as often, and
per-episode correlation **+0.81**. At that correlation the paired standard error is ~0.43× the
unpaired one, and the run-level offset of §6.16 cancels exactly rather than needing to be
estimated. Pinned by `test_episode_scenario_does_not_depend_on_what_happened_before`.

A second wandb failure mode, recorded for the same reason: the run finished and then failed on
`Permission denied: /afs/.../.netrc`. Nothing was wrong with the file; the AFS token (~25 h)
had expired under a long-lived tmux session. `scripts/byz_wandb_env.sh` now sets
`NETRC=/scratch/bmugizi/.netrc` (an identical copy that needs no token).

### 6.19 First valid measurement, and the v2 corpus

Step 13 with the per-episode-seeded collector, Tier 2 `mirror` liar vs clean, baseline seed 3,
`--seed 4242`, n = 50 paired episodes (spawns identical on 50/50):

| | value |
|---|---|
| **paired degradation** | **+9.7 % [+2.4, +17.0], p = 0.007** |
| unpaired, same data | +9.7 %, p = 0.20 |
| correlation between arms | +0.79 |
| SE shrink from pairing | 2.2× |

The same attack was reported at +30.4 % (p = 0.0007) in the legacy corpus. It is real, and about
a third of that size. (The interval above uses the delta method with the paired covariance;
`replicate_test --paired` printed a slightly narrower [+2.9, +16.5] because it treated the clean
mean as fixed.)

Infrastructure for re-collecting everything:

* `byzantine_experiments/paths.py` defines one results root, `results_v2/`. Every sweep script
  and analysis module reads it. Legacy data stays at `results/` as the "before".
* The collector tags each sidecar `seeding: per_episode_v2`.
* `analysis/paired.py::paired_degradation` verifies pairing (tag, seed, baseline, spawns on
  every episode) and raises `PairingError` otherwise. Gate, channel ceiling and both Tier 1/2
  aggregators use it; unpairable rows are labelled and cannot pass.
* Fixed while wiring it in: the gate defaulted `significant = True` when a trajectory was
  missing.

Pinned by `test_paired_analysis_detects_an_effect_unpaired_misses`,
`test_paired_analysis_refuses_unpaired_or_untagged_runs` and
`test_gate_never_counts_an_unpaired_or_missing_comparison_as_significant`.

### 6.20 The efficacy gate, recalibrated: 30 % → 10 % (2026-09-24)

Decided by the project owner before any v2 Tier 3 evaluation, on Tier 1/2 evidence only.

**Why move it.** The spec's 30 % was set against legacy Tier 1/2 numbers that the seeding bug
inflated roughly threefold (`mirror`: +30.4 % reported, +9.7 % paired). On v2 data the whole
Tier 2 `liar` family (18 paired runs) averages +5.7 %, +9.6 % on its strongest seed, best run
+13.1 %. A 30 % gate would ask a trained adversary to triple the best hand-designed attack; 10 %
asks it to match it, which is the bar the spec intended.

**How it is applied.** Detectability decides this as much as the number does. Measured paired
standard error at n = 50 is 4.68 pp, giving:

| episodes per arm | minimum effect detected 80 % of the time | power at a true 10 % |
|---|---|---|
| 50 | 13.1 % | 57 % |
| 100 | 9.3 % | 86 % |
| 200 | 6.6 % | 99 % |

So n = 50 horizon rows that clear the gate are labelled CANDIDATE, and a pass needs the n = 200
confirmatory measurement (`gate_tier3.CONFIRMATORY_MIN_EPISODES`). A pass requires a paired
point estimate ≥ 10 %, a paired 95 % CI above 0, survival ≥ 70 %, and for stealth runs an action
ratio ≤ 1.5.

**Guarding against tuning.** The value is fixed in `gate_tier3.GATE`, the only definition;
`confirm_tier3` and `channel_ceiling` import it. `analysis/gate_calibration.py` re-checks it
against the complete hand-designed results at the end of Stage C phase 1, before phase 2 produces
any Tier 3 number. If the gate has to move, it moves on that evidence, dated here.

**Also fixed in this pass:** `channel_ceiling` printed the legacy conclusions as fixed prose
("Only the observation channel produces attacks that clear the gate…") whatever the data said,
and would have crashed with no significant run; its summary is now computed from the rows. And
its Tier 3 classifier filed *every* Tier 3 run as motors-only, including the round 3–5 lying
adversaries. `confirm_tier3`'s verdict now uses the paired test with no unpaired fallback.

### 6.21 v2 Tier 1/2 results: the effect lives at the family level

Stage C phase 1 (2026-09-25): 144 hand-designed conditions, all paired (correlation between arms
min +0.65, median +0.82). Only **5 of 144** runs are individually significant and damaging, against
~3.6 expected from noise, so at n = 50 per run individual significance barely beats chance. The
signal is in the family-level test (one-sample t on the run means):

| family | runs | mean | positive | family p | reading |
|---|---|---|---|---|---|
| Tier 2 liar | 18 | **+5.7 %** | 16/18 | < 0.0001 | damages the honest team |
| Tier 1 stall | 27 | **+2.6 %** | 21/27 | 0.0004 | damages, weakly |
| Tier 1 gaussian | 18 | +1.9 % | 10/18 | 0.13 | no family effect |
| Tier 1 dropout | 24 | +0.5 % | 10/24 | 0.51 | none |
| Tier 1 OU | 18 | −0.8 % | 6/18 | 0.40 | none |
| Tier 2 waypoint | 18 | −0.4 % | 11/18 | 0.59 | none |
| Tier 2 signflip | 12 | **−2.2 %** | 2/12 | 0.011 | honest team does *better* |
| Tier 2 collusion | 9 | **−2.8 %** | 1/9 | 0.043 | honest team does *better* |

Three readings, all to be confirmed rather than assumed:

* The observation channel (`liar`) is the only hand-designed family with a clear damaging effect,
  and it is small: about +6 % on average, about +10 % on its strongest baseline.
* The stochastic stall fault does measurable harm too, at under half that size.
* The two most aggressive motor attacks (`signflip`, `collusion`) make the honest team *better*.
  This matches the legacy observation that a drone which flails and crashes clears a crowding
  constraint from the formation, but it is now a paired, family-level result rather than an
  artefact of unmatched scenarios.

Gate check: the best significant single run is +13.1 % (winner's-curse inflated; the liar family
mean on its best seed is +9.6 %). 10 % sits between them, i.e. "match the best hand-designed
attack". Gate kept at 10 %; phase 2 proceeds.

### 6.22 v2 Tier 3 screening results (Stage C phase 2, 2026-09-25)

42 checkpoints (6 adversaries × 7 milestones), n = 50 each, all paired (correlation between arms
+0.71 to +0.94). Five checkpoints clear 10 % on screening, all from the seed-3 lying adversaries.

The per-checkpoint maxima are not the result. For every adversary the spread between its
checkpoints is no larger than measurement noise (between-checkpoint τ = 0), so each adversary
does one amount of damage at every checkpoint, best estimated by its mean:

| adversary | channel | mean over 7 checkpoints | range | survival |
|---|---|---|---|---|
| `adv_seed3_stealth0_lieonly` | observation | **+10.2 %** | +4.8 … +14.2 | 99 % |
| `adv_seed3_stealth0_lieonlyego` | observation (+ego obs) | +6.8 % | +2.4 … +12.6 | 98 % |
| `adv_seed1_stealth0_lieonly` | observation | +1.5 % | −3.7 … +6.4 | 99 % |
| `adv_seed1_stealth0` | motors | −1.1 % | −6.6 … +2.3 | 0 % |
| `adv_seed1_stealth0_lie` | motors + observation | −3.2 % | −5.4 … −0.5 | 0 % |
| `adv_seed1_stealth0_f250` | motors, f = 0.25 | −5.9 % | −9.4 … −1.8 | 0 % |

Readings, to be confirmed by step 15:

* The best trained adversary (seed-3 lie-only) does about **+10 %**, the same as the best
  hand-designed attack on the same baseline (Tier 2 liar family on seed 3: mean +9.6 %). On
  this evidence training matches hand design; it does not beat it.
* Giving it absolute position (round 5, `lieonlyego`) did not help; if anything it is lower.
* Every adversary that flies its own motors crashes (survival 0 %) and does no damage or
  slightly *helps* the honest team, consistent with the Tier 2 motor-attack families (§6.21).
* Nothing works on seed 1, trained or hand-designed (liar family seed 1: +2.7 %).

**Confirmatory choice, fixed before running it:** the *final* checkpoint of the best adversary,
`adv_seed3_stealth0_lieonly` @ 194M. With τ = 0 every checkpoint estimates the same effect, so
choosing the maximum (m100, +14.2 %) would add selection bias and no information, whereas "the
fully trained adversary" is the claim the paper can make cleanly. Expected: about +10 % with a
95 % CI of roughly ±4.6 pp at n = 200. Since that sits on the 10 % threshold, pass versus fail is
close to a coin flip; the estimate and its interval are the result, not the verdict alone.

### 6.23 The confirmatory result: Tier 3 fails the gate, with a real effect (2026-09-25)

`adv_seed3_stealth0_lieonly` @ 194M (chosen by rule before the run, §6.22), 200 paired episodes at
seed 3 against a fresh 200-episode clean run:

| | value |
|---|---|
| **honest goal-RMSE degradation** | **+7.4 % [+3.1, +11.7], p = 0.0007** |
| correlation between arms / SE shrink | +0.82 / 2.4× |
| adversary survival | 98 % |
| **gate (≥ 10 %, CI > 0, survival ≥ 70 %)** | **fail** — below 10 %; the interval does not exclude 10 % |

The trained adversary does real, highly significant damage, about 7 %, and does not reach the
recalibrated bar. Screening had put this adversary at +10.2 % (mean over checkpoints); the
confirmatory estimate is lower but within noise of it.

A design flaw found and checked here: at the same `--seed`, episodes 0-49 of the confirmatory
run are the same scenarios the screening sweep used, so the sample was not fully fresh. On the
150 fresh episodes alone the estimate is **+7.1 % [+2.4, +11.8], p = 0.002**, so reuse did not
bias it here (the checkpoint was chosen by rule, not by score). Any future confirmatory run that
chooses a checkpoint *by score* should use a different env seed for both arms.

Also fixed: the verdict printer still read `n_attacked`, a field only the old unpaired test
returned, and crashed after a 62-minute collection; it now reports `n_pairs` and checks survival,
which it had been omitting from the pass condition. Both paths now have an end-to-end test.

**Open question the gate does not answer:** does training beat hand design? The trained +7.4 %
(n = 200) is not yet comparable with the hand-designed liar family's +9.6 % on seed 3 (n = 50
screening). `scripts/byz_confirm_hand.sh` + `analysis/head_to_head.py` measure all three liar
modes at n = 200 on the *same* 200 scenarios, so trained vs hand-designed becomes one paired test
per mode.

### 6.24 Head to head, and the spec audit (2026-09-25)

All three hand-designed liar attacks at n = 200 on the same 200 scenarios as the trained
adversary (seed 3, f = 0.125, one clean run shared):

| attack | vs clean | trained minus this, paired |
|---|---|---|
| **trained** lie-only @ 194M | **+7.4 % [+3.1, +11.7]**, p = 0.0007 | — |
| hand-designed `mirror` | +3.9 % [+0.1, +7.7], p = 0.038 | +3.5 pp [−0.6, +7.6], p = 0.096 |
| hand-designed `offset` | +6.3 % [+2.1, +10.6], p = 0.003 | +1.1 pp [−3.0, +5.1], p = 0.60 |
| hand-designed `stale` | +5.9 % [+1.5, +10.4], p = 0.008 | +1.5 pp [−2.8, +5.7], p = 0.50 |

All four do real damage; none reaches 10 %. **Training matches hand design; it does not
measurably beat it.** (The first print of this table labelled the last row `liar`: the script
cut the name at the `_s` in `_stale`. Fixed.)

The trained adversary's broadcasts are physically impossible on every step (reported position
outside the room 100 % of the time, below the floor 76 %): it evades action-level detectors
and is caught by a bounds check. Full audit of what the spec still requires, and the decisions
it needs: **`design/tier3_audit.md`**.

### 6.25 D1 and D2: one scale for all tiers, and what the gate result means (2026-09-26)

**D1 — the Tier 1/2 bands.** The audit found the 30 % → 10 % recalibration had been applied to
the Tier 3 gate only; the Tier 1/2 specs still required configurations in a 20–40 % band, which
no v2 configuration reaches on the honest drones. Decided by the project owner: validate those
tiers at the **family** level instead (`analysis/family_validation.py`, spec amendment A1). A
family passes when its runs together significantly damage the honest team — one-sample t-test
over per-run paired degradations, mean > 0, p < 0.05, paired rows only. This is the test
`gate_calibration.py` already reported, now the single definition used by both aggregators,
the matched-triple selection and the specs.

| family | runs | mean | positive | family p | status |
|---|---|---|---|---|---|
| Tier 1 `stall` | 27 | +2.6 % | 21/27 | 0.0004 | **validated** |
| Tier 1 `gaussian` | 18 | +1.9 % | 10/18 | 0.13 | no family-level effect |
| Tier 1 `dropout` | 24 | +0.5 % | 10/24 | 0.51 | no family-level effect |
| Tier 1 `ou` | 18 | −0.8 % | 6/18 | 0.40 | no family-level effect |
| Tier 2 `liar` | 18 | +5.7 % | 16/18 | 2×10⁻⁵ | **validated** |
| Tier 2 `waypoint` | 18 | −0.4 % | 11/18 | 0.59 | no family-level effect |
| Tier 2 `signflip` | 12 | −2.2 % | 2/12 | 0.011 | significantly *helps* the honest team |
| Tier 2 `collusion` | 9 | −2.8 % | 1/9 | 0.043 | significantly *helps* the honest team |

Code: the aggregators drop `BAND_LO/HI` and the `in_20_40_band*` columns, add the per-config
paired honest degradation (the only reportable one in v2) and mark matched-triple candidates
(validated family, f = 0.125) with †; their markdown ends with the family table.
`matched_triple.select()` now refuses to run if a pre-registered family (`stall`, `liar`) stops
validating in the tables. The legacy matched-pairs section of the Tier 2 table is labelled
superseded. **Rerun both aggregators** to regenerate `results_v2/tier{1,2}_calibration.*` in
the new format (the CSV numbers do not change; it takes a few minutes because pairing reads the
`.npz` files).

**D2 — the gate.** Kept at 10 % (fixed before any v2 Tier 3 result, §6.20), and both facts are
reported: the best adversary fails it (+7.4 %), and at equal n it matches the best hand-designed
attack (+6.3 %, §6.24). Spec amendment A2 also records how "an adversary that fails the gate is
not usable" is read: the gate classifies an adversary as stronger than hand design; the matched
triple needs *equal* damage and takes its Tier 3 member by its own pre-registered rule (§6.27).
An adversary above the hand-designed ceiling could not be matched at all.

**The consolidated efficacy table** (spec item 3) is `analysis/efficacy_table.py` →
`results_v2/tier3_efficacy.md`: every adversary, five criteria, exactly one verdict (PASS / FAIL
with reasons / INCOMPLETE with what is missing / NOT SCORED for the two defective stealth runs).
The verdict is a pure function covered by spec test 8, every branch. Team reward is now
**paired** (`paired.paired_reward_degradation`) with a CI: +69.6 % [+30.7, +108.6] for the best
adversary — reported, not gated, and as noisy as the spec's note predicts.

### 6.26 D3 and D4: a plausible liar, and a seed-2 adversary (2026-09-26)

**Why (D3).** The best adversary is invisible to action statistics by construction but its
broadcasts are physically impossible (§6.24, §6.30). The spec's stealth condition exists so the
Tier 3 adversary cannot be dismissed by a naive detector; for an observation-channel attacker the
naive detector is a physics check on what it says, not a threshold on what it does. Spec amendment
A3 redefines the stealth condition accordingly.

**What was built** (`adversary/lies.py::PlausibleLie`, shared by training and evaluation). The
policy outputs 3 numbers instead of the free lie's 6: the acceleration of a drift between claim and
truth. The claim is truth + offset, where the offset's velocity is capped at `max_rate`, braked
radially so the offset never exceeds `max_offset` without a hard stop, and limited per axis so the
claim stays inside the room (shrunk by 0.25 m) and is never pushed further out than the truth.
The reported velocity is the true velocity plus the offset's rate of change, so the claimed
position always moves as the claimed velocity says. The offset starts at zero every episode.

Budget, from measured honest flight (clean seed 3, 60 episodes × 8 drones at 100 Hz: speed p50
0.47 m/s, p99 4.95; acceleration p50 1.2 m/s², p90 4.6, p99 11.6): offset ≤ 2 m, drift ≤ 1 m/s,
drift acceleration ≤ 4 m/s².

Three versions failed their own verification before this one, each caught by a test:
clipping the reported position at a wall produced a 15 m/s velocity spike (fixed with the per-axis
interval containing 0); per-axis action scaling let the drift reach √3 × `max_rate` (fixed with
unit-ball normalisation); a first-order drift jittered at ~145 m/s² against an honest median of 1.2
(fixed by controlling acceleration, not velocity).

Verified in `tests/test_tier3.py`: under a worst-case bang-bang policy against a wall-bouncing
trajectory, every step is inside the room, within 2 m, no faster than truth + 1 m/s, kinematically
consistent to 1e-9, with median claim acceleration ≤ 4 m/s² and p99 below the honest p99; every
episode starts truthful; training env and evaluation agent produce identical claims from identical
actions; the plausible env exposes a 3-D lie. The episode statistic `adv/lie_wall_limited_frac`
reports how often a wall limit engaged (the only steps where the acceleration bound can be exceeded).

**Training matrix** (A4): 3 seeds × {free, plausible} lie-only, `EGO=0` (round 4's best
configuration; the ego-state variant did not help, §6.12). New runs, in priority order:

1. `adv_seed3_stealth0_lieonlyplaus` — decides D3's framing earliest (seed 3 is where the free
   adversary works);
2. `adv_seed2_stealth0_lieonly` — D4, completes the free 3 × 3 transfer matrix;
3. `adv_seed1_stealth0_lieonlyplaus`, `adv_seed2_stealth0_lieonlyplaus` — the plausible matrix.

Horizon: the spec's 2e8, as every existing adversary used, so seed 2 is comparable with seeds 1
and 3. Runs so far took 32–57 h each, sequentially on the one T4; four runs ≈ 6–8 days. Halving to
1e8 (`ENV_STEPS=100000000`) would roughly halve that — the free adversaries' damage showed no trend
across checkpoints (§6.22) — but would add a horizon difference to every seed-2 comparison; not
recommended unless time forces it. Every downstream rule (final checkpoint = 0.97 × configured
length, horizon fractions) reads the configured length, so either horizon works without edits.

### 6.27 D1: the matched triple, pre-registered (2026-09-26)

Rules fixed in `analysis/matched_triple.py`'s docstring before any triple data (spec amendment
A5); `scripts/byz_matched_triple.sh` runs select → collect → test:

* **Tier 3 member:** the plausible adversaries (seeds 1–3) if their pooled own-seed damage is
  significant (CI > 0); otherwise the free lie-only adversaries, with stealth reported as a
  limitation (the spec's fallback).
* **Target:** that member's pooled own-seed damage, n = 200 per seed (transfer-matrix diagonal).
* **Tier 1 / 2 members:** from the validated families (`stall`, `liar`) at f = 0.125, the
  configuration whose pooled paired screening damage is closest to the target (ties: smaller
  perturbation). Available now: `stall` configurations pool at +1.1 to +4.7 %, `liar` mirror /
  offset / stale at +4.2 / +4.5 / +5.5 %.
* **Margin:** ±min(3 pp, ½ × target). The spec's ±10 pp exceeded every effect measured: clean
  (0 %) would have been "equivalent" to a 7 % attack.
* **n per condition per seed:** smallest of 400 / 800 / 1600 with ≥ 80 % TOST power for two truly
  equal conditions, from the measured per-episode SD of a difference between two attacks on
  identical scenarios (29.0 pp); otherwise the triple is reported as underpowered, not run.
* **Test:** TOST, α = 0.05 per side (the 90 % CI of the pooled per-episode difference inside the
  margin), for every pair; every member must also be significantly damaging on its own; team
  reward per member reported with its CI, not used to match. Spec test 9 exercises TOST in both
  directions (a matched pair passes, a mismatched pair fails).

The selection is written to `results_v2/matched_triple_selection.json` before any triple episode
is collected and is never recomputed once collection has started.

### 6.28 D5: distribution shift, built as wrappers (2026-09-26)

`wrappers/shift.py::ShiftWrapper`, below the Byzantine wrapper; the simulator package is not
modified. Dynamics randomisation turns on the simulator's own `RelativeSampler` (mass, inertia,
thrust and the other dynamics parameters resampled every episode, normal with the given ratio);
wind adds a uniform mean acceleration plus an exact-discretisation Ornstein–Uhlenbeck gust to every
drone's velocity after each control step. Both draw from the per-episode seeded stream, so a
shifted run still pairs episode-for-episode with a clean run **at the same shift level**.
Spawns are drawn from a different stream and do not change under shift, so the spawn check alone
would pass a cross-shift pair; `paired.check_pairing` therefore also compares the recorded shift
and refuses a mismatch (only the calibration, which exists to compare across levels, opts out).

**Levels are defined by effect, not by parameter** (A6; rules fixed before calibration in
`analysis/shift_calibration.py`): mild = randomisation only, the smallest ratio in
{0.05, 0.1, 0.2, 0.3} whose clean-swarm effect is ≥ 5 % (p < 0.05); severe = randomisation at
min(2 × mild, 0.3) plus the smallest mean wind in {0.5, 1, 2} m/s² with a combined effect ≥ 25 %
(p < 0.05); gust SD = half the mean wind, τ = 1 s. Why calibrate rather than guess: a smoke test
of an arbitrary "severe" setting degraded the clean swarm by **+327 %** — the task destroyed, and
any attack's damage unmeasurable underneath it. `scripts/byz_calibrate_shift.sh` (~55 min)
writes `results_v2/shift_levels.json`; the collector then accepts `--shift mild|severe`.

**Phase 3 priority subset** (`analysis/phase3.py`, `scripts/byz_phase3.sh`): the matched triple
× {none, mild, severe} × seeds 1–3 × 50 episodes, plus a clean run per (seed, level) — 36 runs,
~1.5 h; `CROSS=1` adds the cross-seed Tier 3 adversaries (54 runs). Members come from the
selection file and levels from the calibration file; the job builder refuses to run without
either. `phase3 table` reports each member's damage per level, paired against clean at that
level, and the shift's own effect on the clean swarm.

### 6.29 Cross-seed transfer (spec item 4)

`scripts/byz_transfer.sh` evaluates each adversary's final checkpoint against every baseline
seed at n = 200, each cell paired against a 200-episode clean run at that seed; the diagonal is
the confirmatory run (reused for seed 3). Reading, declared in `analysis/transfer_matrix.py`: an
adversary *transfers* to a seed if its damage there has a paired 95 % CI above 0; either outcome
is reported. The gate's fifth criterion ("transfer tested") is met once an adversary has valid
cells on at least two other seeds. Crashing (motor-channel) adversaries are not evaluated: they
fail survival on their own seed, and a transfer result could not change their verdict.

### 6.30 Stealth, behaviour, compute, tests (spec items 6–9) — `analysis/tier3_report.py`

All from the best adversary's own-seed confirmatory run (n = 200) paired with clean at seed 3.
Its broadcasts were not recorded (the collector records them for every liar since 2026-09-26);
they are **reconstructed** exactly: the policy is feed-forward (`use_rnn` false) and evaluated
deterministically, so each step's lie is a function of that step's recorded observation and true
state, passed through the same decoder the rollout used. Over all 200 episodes it agrees with the
audit's earlier figures, made the same way on a 1-in-10 subsample of 40 episodes (below the floor
75.9 % vs 76 %; speed 9.60 m/s vs 9.6) — a consistency check, not an independent one. Liars
collected from now on carry the recorded arrays, and the report uses those when present.

**Stealth (item 6)** — `results_v2/tier3_report/stealth_adv_seed3_stealth0_lieonly.{md,png}`.
Detector thresholds are set on the clean run only.

| detector | adversary flagged | honest flagged |
|---|---|---|
| action magnitude, episode mean > clean p99 | 2.5 % of episodes | 1.6 % |
| broadcast bounds (claim outside the room) | **100 %** of episodes (100 % of steps) | 0 % |
| broadcast kinematic consistency | 100 % of episodes (20 % of steps) | 0.2 % |
| broadcast speed > clean max (14.0 m/s) | 29.5 % | 0.1 % |
| broadcast acceleration > clean max | 39.0 % | 0.0 % |

**Say it plainly, as the spec asks: a naive detector catches this adversary** — a bounds check on
its broadcast flags every episode with no false alarms. An action-magnitude detector does not
(2.5 % vs 1.6 %). One correction to §6.24's "evades action-level detectors": there *is* a weak
action-level signature — action-norm ratio 1.10, AUC 0.575 — and it is caused by the attack, not
by the adversary's slot (the same slot in the clean run: ratio 0.99, AUC 0.49). The adversary runs
the honest policy, but in a situation the attack has disturbed (below), so its actions differ in
distribution. The plausible liar (§6.26) is the answer to the bounds check; its stealth report is
produced the same way once it is trained.

**Behaviour (item 7)** — `results_v2/tier3_report/behaviour_adv_seed3_stealth0_lieonly_*.png`
(10 episodes top-down; damage over time). Numbers the description rests on, attacked vs clean on
the same 200 scenarios:

| | attacked | clean | paired p |
|---|---|---|---|
| collision steps whose closest pair includes the adversary's slot | 0.54 / episode | 0.11 | < 0.0001 |
| collision steps between two honest drones | 0.36 | 0.32 | 0.43 |
| adversary ↔ nearest honest drone, minimum separation | 0.129 m | 0.174 m | < 0.0001 |
| honest–honest minimum separation | 0.135 m | 0.133 m | 0.62 |
| the adversary's own goal error, change vs same drone clean | +0.18 m | — | < 0.0001 |
| honest goal error | 0.777 m | 0.741 m | 0.004 |
| honest speed / jitter | 1.022 / 0.0242 | 0.991 / 0.0226 | 0.002 / 0.0002 |

The broadcast position is a median 17.3 m from the honest drones; the adversary's true position is
0.65 m from them. Damage concentrates near it (Spearman ρ = −0.16 between an honest drone's damage
and its distance to the adversary's slot, measured in the clean run; the two nearest drones +0.067 m,
the two farthest −0.061 m) and accrues late (4 % / 41 % / 55 % of it in the first / middle / last
third of the episode).

**In plain language: the trained attacker makes itself invisible.** It flies exactly as an honest
drone would, but tells its neighbours it is somewhere else entirely — about 17 m away, mostly
underground. Their collision avoidance reacts to the phantom instead of the real drone, so they fly
into it: collisions involving the attacker rise five-fold, collisions between honest drones do not
change, and the damage falls on the drones nearest to it, growing as the episode goes on and the
formation moves through the space the attacker occupies. Nobody designed this: the reward only said
"hurt the team".

(Measurement note: a first version of the "where" statistic used distances in the *attacked* run and
found the opposite sign — drones knocked off course are far from everything, including the
adversary. Distances are now taken from the clean run of the same scenario.)

**Compute (item 8)** — `tier3_report compute` reads TensorBoard wall times (active hours exclude
pauses > 15 min) → `results_v2/tier3_report/compute.md`. Measured so far for one run:
`adv_seed1_stealth0_f250`, 200 M steps in 15.7 active hours. From directory timestamps (audit),
the eight trained runs took 16–57 h each, ~262 GPU-hours in total, sequentially (no two overlapped).
Evaluation is CPU-bound, ~20 episodes/min.

**Reward components (item 2)** — `tier3_report rewards` → one figure per adversary with every
`adv/r_*` component, survival, episode length and lie magnitude over training, plus a first-vs-last
table.

**Tests (item 9).** `tests/test_tier3.py`: **71 passed, 1 skipped** (2026-09-26). The skip is the
slow checkpoint-hash test (`BYZ_SLOW_TESTS=1`), still to be run once. New this round: checkpoint
round-trip (spec test 6), five-criterion efficacy verdict (test 8), TOST both directions (test 9),
triple margin / n rule, plausible-lie physics, shift pairing refusal, shift reproducibility,
broadcast recording, family validation, physics checks, compute downtime, Phase 3 input guards.

### 6.31 Blocks A–C checked; two analysis fixes; the shift calibration flag (2026-10-09)

**Blocks A, B, C all completed** (logs `logs_v2/run_block_A_20260926_225548.log`,
`logs_v2/run_B_20260927_163004.log`, `logs_v2/run_C_20261009_004532.log`). A: family validation
(Tier 1 `stall` +2.6 %, p = 0.0004; Tier 2 `liar` +5.7 %, p < 0.0001; `signflip` and `collusion`
significantly *help* the honest team), transfer for the three airborne free liars, shift
calibration, reports, the checkpoint-hash test (passed), efficacy table. B: all four adversaries
trained to 200 M steps with the intended flags (`EGO=0`, plausible on seeds 1–3, free on seed 2).
C was run for `adv_seed3_stealth0_lieonlyplaus` only; the other three B adversaries still need it,
which is why the triple selection reported "undetermined".

**D3 result, seed 3.** The plausible liar keeps significant own-seed damage: **+7.1 % [+2.4, +11.8]**
(n = 200, paired), against +7.4 % [+3.1, +11.7] for the free liar on the same seed. It does not
transfer (s1 +3.1 [−1.3, +7.5], s2 +1.8 [−1.8, +5.3]). Survival 100 % at n = 50, action ratio 1.07.

**Fix 1 — stealth report, terminal step.** The block C stealth report showed the plausible liar
failing kinematic consistency in 99.5 % of episodes while only 0.1 % of steps failed. Every flagged
step was the last step of an episode (199/200 episodes; the 200th had a residual just under the
threshold). The observation returned on the terminal step is already the auto-reset observation of
the next episode: `broadcast_truth` there is a fresh spawn up to 6 m from the drone's recorded
position, and the lie is applied on top of it. No honest drone acts on that observation within the
episode, so it is not part of the attack. `tier3_report.broadcasts` now drops that step from
recorded arrays (reconstructed broadcasts take truth from pre-reset positions and never had it).
Corrected: **any naive check flags 0.5 % of adversary episodes vs 0.4 % of honest ones** (bounds 0,
speed 0, consistency 0, accel 0.5 %); below the floor 0 %; broadcast speed 1.51 m/s vs true 1.10;
offset mean 1.65 m. The free liar is caught by every check (100 %), the plausible one by none: D3
did what it was built for. Test `test_recorded_broadcast_drops_the_auto_reset_step`.

**Fix 2 — triple selection order.** `matched_triple._tier3_member` skipped to the free liars whenever
a plausible run was *missing*, not only when the plausible damage was non-significant. Had free
seed 2 been evaluated before plausible seeds 1–2, the free liars would have been selected silently,
contrary to the pre-registered order. An incomplete plausible set is now "undetermined" and the
message lists the missing runs. Test `test_triple_never_falls_back_to_free_on_missing_plausible_runs`.

Cosmetic: `gate_tier3` printed run names cut at 40 characters, which hid the milestone and the
lieonly/lieonlyplaus distinction; names are now printed in full and grouped by adversary.

**Open — shift calibration flag.** The pre-registered rules picked mild = DR 0.05, whose effect on
the clean swarm is already **+32.6 % [+24.3, +40.9]** (above the 25 % *severe* bar), and severe =
DR 0.1 + wind 0.5 m/s², **+140 %**, which `shift_calibration` flagged as destroying the task. The
rules say a > 100 % level is flagged but not what follows. The grid, not the rule, is the problem:
its smallest settings are already far past both thresholds. To be decided before Phase 3 (block E):
see the readme.

### 6.32 Block C for all adversaries, the triple result, and the shift amendment (2026-10-12)

**Block C, remaining adversaries** (`logs_v2/run_C_20261009_030341.log`; 73 tests passed, 1 skipped).
Own-seed damage at n = 200, paired: plausible seed 1 **+2.8 % [−1.4, +7.0]**, plausible seed 2
**+2.6 % [−1.0, +6.2]**, free seed 2 **+3.1 % [−0.7, +7.0]** — none significant on its own seed.
Across all seven airborne adversaries, 4 are significant on their own seed (free s1 +4.6, free s3
+7.4, ego s3 +8.0, plausible s3 +7.1) and 4 of 14 off-diagonal cells transfer (free s1→s3, free
s3→s1, ego s3→s2, plausible s2→s3). The baseline-seed-3 team is the most vulnerable (every
adversary's largest effect is on it; mean over adversaries s1 +3.3, s2 +2.1, s3 +6.1 %). Plausible seed 2 transfers to seed 3
(+6.5 [+2.0, +11.0]). Stealth, all plausible liars: naive broadcast checks flag 0.0 / 0.0 / 0.5 % of
their episodes against 0.1–0.4 % of honest ones; the free seed-2 liar is flagged in 100 % (85.7 % of
its broadcasts below the floor, 10 m/s). Action ratios 1.05–1.07.

**Triple selection** (rule of §6.27, `results_v2/matched_triple_selection.json`): plausible liars,
pooled own-seed target **+4.16 % [+1.74, +6.58]** — significant, so no fallback; Tier 1
`stall_d0.50_ms0.1_hold_f0125`, Tier 2 `liar_mirror_f0125`; margin ±2.08 pp; n = 800 per seed
(TOST power 94 % if truly equal).

**Triple result: NOT MATCHED** (`results_v2/matched_triple.md`). All three members significantly
damaging: Tier 1 +2.69 % [+1.53, +3.85], Tier 2 +3.80 % [+2.65, +4.95], Tier 3 +2.80 % [+1.66, +3.94].
Tier 1 ≡ Tier 3 (−0.11 pp, 90 % CI [−1.12, +0.90], p_TOST = 0.0007); Tier 2 ≡ Tier 3 (+1.00 pp,
[−0.01, +2.01], p = 0.039); Tier 1 vs Tier 2 **not shown equivalent** (−1.11 pp, [−2.15, −0.07],
p = 0.062: the lower bound misses the −2.08 margin by 0.07 pp). The pre-registration fixed the test,
not a follow-up, so this is reported as it stands. Not done, deliberately: adding episodes until the
interval fits (optional stopping), re-choosing members, or widening the margin. How to read it: the
trained adversary — the condition the study is about — is damage-equivalent to both hand-designed
mechanisms; the two hand-designed ones differ by ~1 pp, which any T1-vs-T2 detector comparison must
state. Every screening estimate shrank at n = 800 (stall 4.38 → 2.69, mirror 4.19 → 3.80, Tier 3
target 4.16 → 2.80): selection on n = 50 screening means picks members whose screening luck was good.

**Shift amendment (D5).** The v1 calibration (§6.31) produced mild = DR 0.05 (+32.6 %, above the
severe bar) and severe = DR 0.1 + wind 0.5 (+140 %, flagged by the rules themselves as destroying
the task). Decision, written before any attacked run under shift existed (`results_v2/phase3` empty)
and recorded in `shift_calibration.py`: the **grid** is extended (DR 0.01–0.04; stage-2 wind
0.1–1.0), the **rules** are unchanged, and its picks are the primary `mild` / `severe`. The v1
picks are kept unchanged as `mild_v1` / `severe_v1` (`--freeze-v1`, which refuses to label anything
but the recorded v1 values). Phase 3 runs all four levels plus none — 60 runs — and every level is
reported, ordered by clean-swarm effect, as a dose-response. Running both was chosen over picking
one: neither set is selected after Phase 3 results exist, and the amendment itself used only the
calibrated quantity (clean-swarm degradation), never an outcome of the study. Tests:
`test_shift_amendment_keeps_the_v1_levels_and_refuses_to_mislabel`,
`test_phase3_reads_every_level_name_and_orders_by_effect`.

### 6.33 Block E: amended calibration and Phase 3 under shift (2026-10-12)

Log `logs_v2/run_E_20261012_120547.log`: 24 calibration + 60 Phase 3 rollouts, none failed.

**Calibration (amended grid, rules unchanged).** Stage 1, DR alone: 0.01 +2.3 % (p = 0.45),
0.02 +0.3 % (p = 0.92), **0.03 +9.6 % [+3.1, +16.1]** (p = 0.004), 0.04 +17.6 %, 0.05 +32.6 %. Mild =
DR 0.03. Stage 2 at DR 0.06: wind 0.1 +60.1 %, 0.25 +62.4 %, 0.5 +66.8 %, 1.0 +74.9 % — the DR alone
carries almost all of it, so severe = DR 0.06 + wind 0.1 at **+60.1 % [+49.7, +70.5]** rather than
near the 25 % bar (the rule takes the smallest wind, and the smallest already clears it). The four
levels, by clean-swarm effect: mild +9.6, mild_v1 +32.6, severe +60.1, severe_v1 +139.8 %.

**Pairing holds under shift.** The per-episode correlation between an attacked run and its clean
run at the same level is +0.78 to +0.90 (unshifted +0.82 / +0.91), so the randomised dynamics and
wind are drawn identically in both arms; had they not been, the correlation would have collapsed.
Phase 3's `none` episodes are exactly the first 50 of the n = 800 triple runs (identical spawns),
so the n = 800 triple estimates (+2.7 / +3.8 / +2.8 %) are the better unshifted baseline than the
`none` row (+7.2 / +6.4 / +5.4 % on those 50 episodes).

**Damage under shift** (pooled seeds 1–3, n = 150 per cell; % of the clean error at that level, and
added honest goal error in cm):

| member | none | mild | mild_v1 | severe | severe_v1 |
|---|---|---|---|---|---|
| Tier 1 stall | +7.2 % / +7.9 cm | +0.3 / +0.4 | +0.8 / +1.1 | +0.4 / +0.6 | −1.3 / −3.3 |
| Tier 2 liar mirror | +6.4 / +7.0 | +1.7 / +2.1 | +3.8 / +5.6 | −0.7 / −1.4 | −1.1 / −3.2 |
| Tier 3 plausible (own seed) | +5.4 / +5.8 | +2.9 / +3.5 | +1.8 / +2.9 | −1.1 / −2.3 | −1.9 / −5.5 |

Every cell under shift has a CI that includes 0, and the cm intervals widen from ±5 cm (none) to
±9 cm (severe), because shifted dynamics make each episode noisier. Reading, kept to what n = 50
per seed supports (the limit `phase3.py` stated before the data): **under shift no member's damage
is detectable; whether the attacks weaken or are merely buried cannot be resolved at this n**
(point estimates fall toward zero, but the intervals still contain the unshifted ~3 cm). For the
detector work this is the useful fact: under shift a damage meter cannot see any of the three
attacks, so a detector that still separates them is not reading damage. Not done: more episodes
to chase the trend (no hypothesis about it was pre-registered).

**Spec factors not collected.** Cross-seed Tier 3 adversaries: `CROSS=1` (block F, optional, ~1 h).
Stealth: under A3 the stealth condition is the plausible lie, which the triple already uses; the
free liars are not in Phase 3. f = 0.25: no valid adversary (both f = 0.25 adversaries crash,
survival 0 %).

### 6.34 Block F: final tables; Tier 3 closed (2026-10-12, written up 2026-10-21)

Log `logs_v2/run_F_20261012_155433.log`: 30 cross-seed Phase 3 rollouts, none failed; every final
table regenerated with all adversaries. Efficacy (`tier3_efficacy.md`): 10 FAIL, 2 NOT SCORED, no
PASS. Compute: 12 runs, 2.4 B env steps, **455 GPU-hours** on one T4.

Cross-seed Tier 3 in Phase 3 (each baseline seed against the adversaries trained on the other two;
n = 300 per cell): none +5.6 % [+2.5, +8.8], mild +3.5 % [+0.5, +6.5], mild_v1 +2.1 % [−1.1, +5.2],
severe −2.2 % [−5.3, +0.9], severe_v1 −0.7 % [−2.9, +1.4]. The same fall toward zero as the
own-seed cells; the one shifted cell whose interval clears 0 (mild, lower bound +0.5) is one of 20
shifted cells and is not read as a finding.

Behaviour, a coincidence checked: the seed-3 free and plausible liars report identical rounded
totals (0.91 collisions per episode, 0.54 involving the adversary's slot, honest speed 1.022 m/s).
The two rollout files differ in every episode (181 vs 182 collisions, speed 1.0217 vs 1.0216); the
two lies produce near-identical disruption on that team. Plain-language write-up of all of Tier 3:
`why/TIER3_explanation.md`.

### 6.35 Tier 1/2 docs re-checked against v2 (2026-10-21)

With the Tier 1/2 behaviour figures regenerated from v2 (`results_v2/figures/`,
`results_v2/tier{1,2}_behaviour.json`), every quantitative claim in `tier1_design.md`,
`tier2_design.md` and both explanation files was checked against the v2 corpus; each now opens
with a §0 giving the verdicts, and the original text follows unchanged. The general pattern —
worth stating in any write-up as a methods lesson:

- **Large effects on the faulty/attacking drone replicate.** Swarm-wide goal-RMSE degradation
  (dominated by that drone): Tier 1 median change −1.7 pp, 20 of 29 shared configs within 5 pp;
  Tier 2 median −3.2 pp, 12 of 19 within 5 pp. Behaviour descriptors (jitter, persistence, floor
  time), survival and guard exposure match closely; every behaviour description holds.
- **Small effects on the honest drones did not.** Their sign changed in 17 of 48 configs. Two
  Tier 2 conclusions change: the liar *does* damage the honest drones (+5.7 %, p = 2×10⁻⁵), and
  collusion *helps* them (−2.8 %, p = 0.043; legacy +5.8 to +7.9 %).

So the 2026-09-25 banners ("every figure inflated roughly 3×") overstated the problem for these
documents; they were replaced, and spec amendment A1 (Tiers 1 and 2) gained a dated
clarification. The band-retirement decision is unchanged: it rests on the honest-drone metric.
Not re-collected and still legacy-only: the Tier 1 refinement grid (33 runs) and the Tier 2
waypoint α refinement (12 runs), hence the waypoint "control-authority cliff".
