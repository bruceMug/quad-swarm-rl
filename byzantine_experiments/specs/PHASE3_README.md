# PHASE 3 — Read First

**Audience:** the researcher first, then Copilot.
**Location in repo:** `byzantine_experiments/specs/`

Files in this set, in reading order:

1. `PHASE3_README.md` — this file: what changed, decisions to sign off, build order
2. `PHASE3_METHOD.md` — the scientific definition of the detector
3. `PHASE3_EVALUATION.md` — pre-registered tasks, baselines, statistics, success criteria
4. `PHASE3_DETECTOR.md` — the implementation spec Copilot builds from
5. `REFERENCES.md` — primary sources, what each is used for, verification status

---

## Answer to Copilot's question

> *Do you have the rest of the plan defining policy-graph construction, the spectral
> features, the Phase 3 evaluation, and what siForest is?*

Yes, it is this set:

- **Policy-graph construction** → `PHASE3_METHOD.md` §3
- **Spectral features** → `PHASE3_METHOD.md` §4. This uses the *magnetic Laplacian*, a
  generalisation of the Hermitian Laplacian. Read §4.3: the earlier plan had a mathematical
  error here, and this set corrects it.
- **siForest** → a published method, not ours: C. Djidjev, *siForest: Detecting Network
  Anomalies with Set-Structured Isolation Forest*, arXiv:2412.06015 (2024). The paper links
  no code, so we reimplement it. `PHASE3_METHOD.md` §5 defines it. **It is not load-bearing**:
  it is one of four per-agent scorers we compare.
- **Phase 3 evaluation** → `PHASE3_EVALUATION.md`, fixed before any detector code runs on
  attacked data.

Copilot: do not draft an alternative method. If anything here conflicts with what you find
in the repo or the data, write it into `design/phase3_step0_audit.md` and stop for the
researcher. Deviations are fine; silent deviations are not.

---

## What changed, and why it matters

The Tier 2 spec warned that a state-misreporting agent's own policy graph would look normal.
Copilot now reports that **every adversary that passes the efficacy gate is a liar flying
the honest policy.** That reshapes the paper's central claim.

The original claim was: *a Byzantine agent's policy graph has a different spectral
fingerprint from a degraded agent's.* For a liar that is close to false by construction —
its actions come from the honest policy acting on its own true observations.

The revised claim is:

> **Where a fault enters the swarm can be read from how behavioural anomaly is distributed
> across the agents.** Three causes, three patterns:
>
> | Cause | The source agent | The rest of the swarm |
> |---|---|---|
> | Environmental shift (weather, sensor drift) | — (no source) | uniformly disturbed |
> | Action-level fault (a degraded drone) | strongly anomalous | mildly disturbed |
> | Message-level attack (a liar) | **calm** | disturbed |
>
> The liar is the calm agent in a disturbed crowd. Spectral features of directed policy
> graphs are the proposed per-agent anomaly score; whether they are the *best* score is a
> pre-registered comparison, not an assumption.

This is a stronger paper than the original, and it ties distribution shift directly into
the central claim — shift is no longer a nuisance condition, it is one of the three causes
the detector must tell apart.

### The honest limit on the claim

In your data, every working attacker uses the message channel and the only action-channel
fault is the degraded drone. So "degraded vs malicious" and "action channel vs message
channel" are **the same split** in this corpus. The data can support a claim about
*mechanism* (where the fault enters). It cannot support a claim about *intent*. The paper
must say this in one plain sentence, and `PHASE3_EVALUATION.md` §9 records it as a stated
limitation. A reviewer will find it if you don't.

The two contributions are now separable: (a) mechanism inference from the swarm-level
pattern of anomaly; (b) directed spectral fingerprints as the per-agent score. Either can
succeed without the other. `PHASE3_EVALUATION.md` §9 says which paper you write in each case.

---

## Decisions that need the researcher's sign-off

Copilot does not start Deliverable D2 of `PHASE3_DETECTOR.md` until these are confirmed in
`design/phase3_step0_audit.md`.

**D1 — The auditor model.** What the detector may see decides whether the method is needed
at all. If it can read a liar's broadcasts *and* knows true positions, one subtraction
catches every liar and there is no paper. Primary, pre-registered: **AM-2, behaviour-only
auditing** — each agent's own kinematics and executed actions, plus the public
neighbour-selection rule; no message contents, no network weights. `PHASE3_METHOD.md` §2
defines all auditor models and argues why AM-2 is the defensible one.

**D2 — BayesG leaves the experiments.** The swarm was trained with the attention neighbour
encoder, not BayesG. Adding BayesG means retraining three baselines and every adversary.
Recommendation: cite BayesG in Related Work as the learned-trust line of work, and use the
attention encoder's weights as the in-repo analogue of a learned-trust signal (white-box
baseline W1). Your proposal named BayesG, so the professor should agree to this explicitly.

**D3 — The lockbox.** 25% of episodes in every condition are sealed now and not touched until
the final run (`PHASE3_EVALUATION.md` §2). It feels wasteful. It is the cheapest thing you
can do to make reviewers believe your numbers.

---

## Build order

```
Step 0  Audit repo + data, write split file  → design/phase3_step0_audit.md
        ↓  researcher confirms D1–D3
Step 1  Data access layer + leakage firewall  DETECTOR D1
Step 2  Cheap baselines + KILL TEST           DETECTOR D2, EVALUATION §6
        ↓  researcher reads the kill-test report before anything else is built
Step 3  Behavioural abstraction               DETECTOR D3
Step 4  Policy graphs + spectra               DETECTOR D4
Step 5  Per-agent scorers incl. siForest      DETECTOR D5
Step 6  Swarm-level statistics                DETECTOR D6
Step 7  Development evaluation                DETECTOR D7
Step 8  Lockbox evaluation, run once          EVALUATION §2
```

The kill test runs before the spectral pipeline exists, on purpose. If a twenty-line
baseline already solves the problem, you need to know before spending three weeks on
eigenvalues, not after.

---

## On your paper summaries

Do not put them in the specs and do not let Copilot implement from them. Every method here
is defined completely enough to implement without reading any paper; `REFERENCES.md` points
to primary sources. Your summaries were a learning tool and errors in them are expected —
but a summary error that Copilot implements becomes a results error.

Keep them in `notes/`, prefixed `UNVERIFIED_`. They matter for Related Work, which is a
separate, later check against the primary sources.

Some of my earlier material needed correcting too, and you should know exactly what:

1. The `SetStructuredIsolationForest` class in `research_execution_plan.md` **is not
   siForest.** It is an ordinary isolation forest on per-agent summary statistics, which
   Djidjev's paper uses as a *baseline* (the "summarization" approach). The real algorithm
   changes the trees' stopping rule. Both are now implemented and compared.
2. The plan assumed the Hermitian Laplacian's smallest eigenvalue is always zero, so that
   λ₂ is "the Fiedler value". **For directed graphs it generally is not**: the smallest
   eigenvalue is positive when the graph carries net circulating flow. That turns out to be
   useful — `PHASE3_METHOD.md` §4.3.
3. The proposal's "Zhang et al. 2025, AIMS Mathematics" citation could not be verified as
   described. Do not cite it. `REFERENCES.md` lists what to cite instead.
4. The QuadSwarm/Batra arXiv ID in the plan's appendix may be wrong. `REFERENCES.md` marks
   it for checking.
