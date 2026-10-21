# PHASE 3 — Evaluation Protocol (Pre-registration)

**Status:** fixed before any detector runs on attacked data. This document is committed to
git with a timestamp before Step 3 starts. Every number in the paper's main results must
come from the procedure written here. Anything else is labelled *exploratory*.

---

## 1. Data and the matched set

Step 0 (`PHASE3_DETECTOR.md` D0) produces the inventory. This protocol assumes it contains:

- **Clean** episodes per baseline seed, at shift level 0 and at shift levels 1–5
- **Tier 1** (action-level, degraded) corpora
- **Tier 2 / Tier 3** corpora, including the message-level attackers (liars)
- **The matched triple**, n = 800 episodes per member
- **Transfer cells** (adversary trained on seed a, evaluated on seed b)
- **Attacked episodes at shift levels 1–5**

**The matched set.** The central comparison uses conditions with statistically
indistinguishable team damage. The triple already exists. Step 0 also reports clean-episode
team damage at each shift level; **if any shift level falls inside the triple's damage band**
(Mann–Whitney p > 0.05 against each member), it joins the set as the `env` member, making a
matched quadruple `{env, act, msg_rule, msg_learned}` plus `clean`. If no level matches, use
the closest level and report its damage gap explicitly.

Matched damage is what lets the paper claim the detector recovers *mechanism* rather than
*severity*. Without it, a reviewer correctly calls the detector a damage meter.

---

## 2. Splits and the lockbox

Generated once, by Step 0, from a fixed seed, and committed as `splits/phase3_splits.json`.
Split at the episode level, stratified by (condition, baseline seed, shift level):

| Split | Share | Contains | Used for |
|---|---|---|---|
| `fit` | 40% | clean only is used | fitting scalers, codebooks, scorers |
| `dev` | 35% | all conditions | thresholds, hyperparameter choices, all development plots |
| `lockbox` | 25% | all conditions | the final numbers, run once |

Rules:

1. Nothing is fitted on attacked episodes, ever. Unsupervised means unsupervised.
2. Every choice not fixed in `PHASE3_METHOD.md` (thresholds τ, K, W, q, scorer, aggregation)
   is made on `dev` and written into `design/phase3_frozen_config.json` before the lockbox run.
3. The lockbox runner refuses to execute unless `--final` is passed, refuses if
   `phase3_frozen_config.json` is missing, and appends a timestamped entry to
   `results/LOCKBOX_LOG.md`. **One run.** If the code changes after it, the new run is
   reported as a second run, not a replacement.
4. Cross-seed generalisation (leave-one-baseline-seed-out) is reported on `dev` as a
   robustness analysis.

---

## 3. Pre-registered hypotheses

"Full method" means: magnetic-Laplacian fingerprint ψ_{1/4} → scorer S3 (siForest) →
swarm statistics of `PHASE3_METHOD.md` §6, unless `dev` selects a different scorer, in which
case the selected scorer is frozen and named.

All hypotheses are evaluated on the **matched set, shift level 0**, unless stated.

**H1 — Detection.** Task T1 (`act ∪ msg` vs `clean ∪ env`).
The full method achieves AUROC ≥ 0.85, and exceeds the raw-state baseline K3 by
ΔAUROC ≥ 0.05 with the 95% CI excluding zero.

**H2 — Mechanism.** Task T2, four-way `{clean, env, act, msg}`.
A classifier using only the swarm statistics (`D_mean`, `D_max`, `C`, `Q`) with thresholds
fixed on `dev` achieves balanced accuracy ≥ 0.70 (chance 0.25). The pairwise `act` vs `msg`
separation at matched damage reaches AUROC ≥ 0.80.

**H3 — Directed spectra earn their place.** On T1 and on the `act`-vs-`msg` task:
(a) q = 1/4 beats q = 0 by ΔAUROC ≥ 0.03, CI excluding zero;
(b) ψ_{1/4} beats the non-spectral Markov-chain feature set (`PHASE3_METHOD.md` §4.5) by
ΔAUROC ≥ 0.03, CI excluding zero.

**H4 — Robustness to distribution shift.**
(a) At the threshold giving 5% FPR on clean level-0 `dev` episodes, the FPR on clean episodes
at the highest shift level is ≤ 10% for relative scoring (`r`), and is reported for absolute
scoring (`s`) and all baselines.
(b) The AUROC drop on T1 from shift level 0 to the highest level is no more than half the
drop of K3.

**H5 — Localisation (secondary).** `act`: top-1 accuracy ≥ 0.70. `msg`: top-1 ≥ 0.40
(chance 0.125). H5-msg is downgraded to exploratory if Step 0 finds mean off-diagonal
exposure > 0.9 (`PHASE3_METHOD.md` §6.4).

The thresholds above are commitments, not targets to tune toward. If a result lands at
0.84, it is reported as 0.84 and H1 as not met.

---

## 4. Baselines

### Kill-test baselines (built first, AM-2 compliant)

| ID | Baseline | Gives |
|---|---|---|
| K1 | Team-performance monitor: episode formation error and team reward vs clean | T1 only |
| K2 | Collision participation: per-agent count of drone–drone contacts, from true positions | T1, T3 |
| K3 | Raw-state anomaly: S1 isolation forest on raw `x_i(t)` features, no abstraction | T1, T2, T3 |
| K4 | Separation statistic: per-agent mean of minimum distance to neighbours | T1, T3 |

K2 and K4 exist because of a specific risk: a victim that believes the liar is elsewhere will
fly into the liar's true position. The liar may therefore be the agent most involved in
close passes and collisions — a trivially observable signal under AM-2. If so, the paper
must report it, and localisation stops being a contribution.

### Ablation baselines (built with the method)

| ID | Baseline | Tests |
|---|---|---|
| A1 | ψ at q = 0 | H3a: direction |
| A2 | Non-spectral Markov-chain features | H3b: spectrum vs chain |
| A3 | State-visitation KL only | state-marginal OOD |
| A4 | Scorers S0, S1, S2 in place of S3 | whether siForest matters |
| A5 | Absolute-only swarm statistics | value of relative scoring under shift |

### Appendix baselines (other auditor models)

| ID | Baseline | Auditor model |
|---|---|---|
| W1 | Attention-weight trust: mean attention each agent receives from neighbours, compared with clean | AM-3 (in-repo analogue of a learned-trust signal such as BayesG's edge posterior) |
| W2 | Policy residual: ‖a_reported − π(o_reported)‖ | AM-1 |
| O1 | Broadcast vs truth difference | AM-0, sanity upper bound |

O1 should score near 1.0. If it doesn't, the data or the labels are broken; stop and check.

---

## 5. Metrics

| Task | Primary | Secondary |
|---|---|---|
| T1 | AUROC | AUPRC; TPR at 5% FPR; median time-to-detection |
| T2 | balanced accuracy | macro-F1; full confusion matrix |
| T3 | top-1 accuracy | top-2 accuracy |
| Shift | FPR at the level-0 threshold, per level | AUROC per level |

**Time-to-detection.** Compute the detector on growing prefixes of the episode (after 1, 2,
3, … windows). Threshold at 5% FPR on clean `dev`. Report the median number of steps to the
first crossing among detected faulty episodes, and the fraction never detected.

---

## 6. The kill test (Step 2, before the spectral pipeline exists)

Run K1–K4 on the matched set at shift level 0 and at the highest shift level, on `dev` only.
Write `results/phase3_dev/KILL_TEST.md` with all numbers and stop for the researcher.

Decision rules, fixed now:

| Kill-test result | Consequence |
|---|---|
| Any K reaches T1 AUROC ≥ 0.95 at level 0 | Detection at level 0 is not a contribution. The paper's detection claim moves to shifted conditions and to T2. |
| K2 or K4 reaches `msg` top-1 ≥ 0.90 | Liar localisation is not a contribution. K2/K4 become the reported localisation method; H5-msg is dropped. |
| All K below these | Proceed as planned. |

Kill-test baselines appear in the final tables whatever happens.

---

## 7. Statistics

- **Unit of analysis:** the episode. Clusters: baseline seed × adversary seed.
- **Confidence intervals:** cluster-stratified bootstrap, 2,000 resamples, episodes resampled
  within each (seed, condition) stratum.
- **Comparisons between methods:** paired bootstrap of the difference on the same episodes
  (DeLong's test is an acceptable alternative for AUROC).
- **Multiple comparisons:** Holm correction across the primary comparisons only —
  H1 vs K3, H2 balanced accuracy vs chance, H3a, H3b, H4b. Everything else is reported with
  uncorrected CIs and called secondary.
- **Seeds:** all results reported per baseline seed as well as pooled. A result that holds
  pooled but not in at least two of three seeds is reported as seed-dependent.

---

## 8. Figures and tables the paper needs

| ID | Content |
|---|---|
| Fig 1 | **The mechanism map.** Each episode as a point in (`D_mean`, `C`), coloured by condition, `Q` as marker size. The headline figure — if the four clouds separate, the reader sees the paper's claim in one look. |
| Fig 2 | T1 AUROC vs shift level, every method. |
| Fig 3 | ΔAUROC vs charge q ∈ {0, 1/16, 1/8, 3/16, 1/4}. |
| Fig 4 | Example policy graphs — clean agent, degraded agent, liar, liar's victim — nodes laid out by the first magnetic eigenvector's phase. |
| Fig 5 | Time-to-detection curves. |
| Tab 1 | Main results: T1 and T2 for full method, K1–K4, A1–A5. |
| Tab 2 | Mechanism confusion matrix. |
| Tab 3 | Ablations: K, W, q, scorer, aggregation, codebook (per-seed vs cross-seed). |
| Tab 4 | Kill-test baselines at all shift levels. |
| Tab A1 | Appendix: W1, W2, O1. |

---

## 9. Outcomes and which paper each produces

Decide this now, so that results cannot pull the framing around afterwards.

| Outcome | Paper |
|---|---|
| H1–H4 hold | As designed: mechanism inference + directed behavioural spectra. Main-track submission. |
| H2 holds, H3 fails | *Swarm-level anomaly patterns reveal fault mechanism.* Spectral features are one scorer among several; H3 reported as a negative ablation. Still main-track viable. |
| H3 holds, H2 fails | *Directed behavioural spectra for robust anomaly detection in swarms.* Mechanism inference reported as a negative result. Workshop or main-track depending on margins. |
| Kill test removes detection at level 0 | The paper is about shifted conditions and mechanism, where cheap baselines are expected to fail; the kill-test result is stated up front as motivation. |
| All fail | Workshop paper: the matched-damage benchmark (clean / env / act / msg at matched damage across five shift levels) and the negative result. The corpus is a genuine asset; release it either way. |

**Stated limitations, regardless of outcome:**

1. Every gate-passing attacker uses the message channel, so in this corpus intent and
   mechanism are confounded. The paper claims mechanism, not intent.
2. Simulation only; AM-2 assumes accurate external tracking.
3. One task family (QuadSwarm `mix`), N = 8.
