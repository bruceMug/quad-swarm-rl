# PHASE 3 — Method Definition

**Status:** advisor's method, frozen for implementation. Changes go through the researcher
and are logged in `design/phase3_design.md` §Deviations.
**Audience:** researcher and Copilot. Implementation details are in `PHASE3_DETECTOR.md`.

---

## 1. The problem, stated formally

A swarm of N = 8 agents runs a frozen cooperative policy π. In an episode, the swarm is in
one of four conditions:

| Label | Condition | Source of the fault |
|---|---|---|
| `clean` | nothing wrong | — |
| `env` | environmental shift at level L ∈ {1..5} | the environment, affects everyone |
| `act` | one or two agents' executed actions are corrupted (Tier 1, and any Tier 2/3 action-channel attacker) | inside the agent's control loop |
| `msg` | one or two agents broadcast false state; their own actions are honest (Tier 2C, and Tier 3 if it lies) | in what other agents hear |

`env` can be combined with `act` or `msg` (attacked episodes at shift level L > 0).

Three tasks, in decreasing order of importance:

- **T1 Detection.** Given an episode, is it `act` or `msg` (something is wrong *inside* the
  swarm), as opposed to `clean` or `env`?
- **T2 Mechanism.** Given an episode, which of `clean / env / act / msg` is it?
- **T3 Localisation.** Given a faulty episode, which agent is the source?

T1 and T2 are the paper. T3 is secondary and may be exploratory (§6.4).

---

## 2. The auditor model

The auditor model says what the detector is allowed to read. It is the single decision that
determines whether this method is necessary, so it is fixed before any code is written.

| Model | Auditor sees | Status |
|---|---|---|
| **AM-0** Oracle | true states, broadcasts, labels | not a detector; upper bound only |
| **AM-1** Self-report | every agent's broadcasts (honest agents report truth, liars lie), executed actions | appendix |
| **AM-2** Behaviour-only | every agent's **true** kinematics (position, velocity, attitude, angular velocity), executed actions, the public neighbour-selection rule | **primary** |
| **AM-3** White-box | AM-2 plus the policy network (activations, attention weights) | appendix, baselines W1/W2 |

**Why AM-2 is primary.** It models an external monitor — a ground station with motion
capture or radar tracking — that observes how the drones actually fly but cannot read or
authenticate their peer-to-peer messages (encrypted, proprietary, or simply not logged).
Under AM-2 a liar cannot be caught by comparing its claims to the truth, because the auditor
never sees the claims. Detection must come from behaviour. That is exactly the regime where
a behavioural detector is needed, and it is a realistic one.

**Why not AM-1 as primary.** Under AM-1 the auditor can re-run the known policy on each
agent's reported observation and compare with its reported action. A liar's reported state
does not match the action it actually took, so this residual check is likely to catch liars
almost trivially. It is reported in the appendix as baseline W2, honestly, and the paper
says why AM-2 is the setting of interest.

**What AM-2 forbids in code** (enforced by the firewall in `PHASE3_DETECTOR.md` D1):
`actions_honest`, any recorded broadcast, `byzantine_mask`, condition labels, adversary
identity, and anything derived from them. Note that executed actions are allowed: they are
what the drone physically did.

---

## 3. Policy graph construction

A policy graph here is a **data-driven abstract Markov chain over behavioural states**, in
the sense of Topin & Veloso (AAAI 2019). Unlike their predicate-based construction, the
abstract states come from clustering (as in Zahavy et al., ICML 2016), because quadrotor
behaviour has no natural predicates.

### 3.1 Behavioural features

For agent i at step t:

```
x_i(t) = [ ego_i(t) , a_i(t) ]
```

- `ego_i(t)` — the agent's **true** ego kinematic state: position relative to its goal,
  velocity, attitude (rotation matrix or its 6D form, as in the env), angular velocity.
  Take the exact slice from the observation layout documented in `design/tier2_design.md`,
  but source the values from true state, not from a possibly perturbed observation.
- `a_i(t)` — the executed action.

**Neighbour observations are deliberately excluded.** For a victim of a liar, the neighbour
block contains the lie. Clustering on it would detect the *input* corruption directly, which
AM-2 cannot see and which would not be behavioural detection. We want to detect the victim's
*response*, which shows up in its own kinematics and actions.

### 3.2 Abstraction

- Standardise features with statistics from clean fit-split episodes.
- k-means with K = 32 abstract states (primary); K ∈ {16, 64} as sensitivity.
- **Fit on clean episodes only, separately per baseline seed.** The auditor is assumed to
  have clean flight logs of the specific swarm it monitors. Cross-seed codebooks are an
  ablation.
- The same codebook is then applied to every episode of that seed, in every condition, so
  abstract state k means the same thing everywhere.

Quality checks the codebook must pass (reported, not tuned against attacked data):
every cluster holds ≥ 0.5% of clean points; a first-order chain predicts the next abstract
state on held-out clean data better than the marginal distribution (log-likelihood gain > 0).
If the second check fails, the "policy graph" has no sequential structure worth analysing,
and that is a finding to report.

### 3.3 Windowed graphs

- Window length W steps, stride W/2. W is fixed in Step 0 from the episode length: the
  largest power-of-two-ish value giving at least 4 windows per episode. Sensitivity: W/2, 2W.
- For agent i and window w, the count matrix `C[k,l]` = number of transitions k → l.
- **Self-loops are removed** from C. Dwell behaviour is captured separately in the
  non-spectral features (§4.5). A self-loop contributes equally to degree and adjacency and
  cancels in the Laplacian anyway; removing it explicitly avoids confusion.
- Keep only visited states and restrict to the **largest weakly connected component**.
  Record the component size `n_cc` and the fraction of transitions it covers as
  non-spectral features.

---

## 4. Spectral features

### 4.1 The magnetic Laplacian (our operational definition)

For a window's count matrix C (self-loops removed, restricted to the component), define

```
A_s[k,l] = (C[k,l] + C[l,k]) / 2                         symmetrised weight
f[k,l]   = (C[k,l] − C[l,k]) / (C[k,l] + C[l,k])          net-flow fraction ∈ [−1, 1]
θ[k,l]   = 2π q · f[k,l]                                  phase
H[k,l]   = A_s[k,l] · exp(i θ[k,l])                       Hermitian: H = H*
D_s      = diag(Σ_l A_s[k,l])
L_q      = I − D_s^{−1/2} H D_s^{−1/2}                    normalised magnetic Laplacian
```

`L_q` is Hermitian, its eigenvalues are real and lie in [0, 2].

Relationship to the literature:

- **q = 0** gives the normalised Laplacian of the undirected graph. Direction is discarded.
- **q = 1/4, unweighted** gives exactly the Hermitian adjacency of Liu & Li (2015) / Guo &
  Mohar (2017): one-way edges get ±i, two-way edges get 1.
- This is the magnetic Laplacian used in MagNet (Zhang et al., NeurIPS 2021), with one
  deliberate change: MagNet's phase is `2πq·(A − Aᵀ)` on raw weights, which depends on the
  scale of the counts and therefore on window length. We use the net-flow *fraction*, which
  is scale-free. State this in the paper's method section.

Primary charge: **q = 1/4**. Sweep: q ∈ {0, 1/16, 1/8, 3/16, 1/4}.

### 4.2 Why directed and not undirected

A policy is a process with a direction: approach → brake → hover is not the same behaviour as
hover → brake → approach. An undirected graph sees them as identical. The q = 0 versus q > 0
comparison is the paper's test of whether directed structure carries signal. It is the
answer to the reviewer who says *"this is just OOD detection with a fancier feature"* — see
hypothesis H3 in `PHASE3_EVALUATION.md`.

### 4.3 The correction: the smallest eigenvalue is not always zero

For q = 0, λ_min = 0 always and λ₂ is the Fiedler value. **For q > 0 this fails.** λ_min = 0
only when the phases around every cycle multiply to 1. When the graph carries net
circulating flow — the agent repeatedly goes round a behavioural loop in one direction —
λ_min > 0.

Worked example used as a unit test: a directed 3-cycle, q = 1/4. Each node has symmetrised
degree 1 and every edge has phase π/2, so the normalised magnetic Laplacian is circulant
with eigenvalues `1 − cos(π/2 + 2πk/3)` for k = 0, 1, 2, i.e.

```
{ 1 − √3/2 ≈ 0.1340,   1,   1 + √3/2 ≈ 1.8660 }
```

With q = 0 the same graph gives { 0, 1.5, 1.5 }.

So λ_min measures **behavioural circulation**: how strongly the agent's behaviour flows round
loops in a consistent direction. A purposeful agent cycling through approach–correct–hold
has it; an agent whose transitions are jittered back and forth by noise has less. That is a
directly interpretable quantity, and the earlier plan's assumption would have thrown it away.

### 4.4 The spectral fingerprint ψ_q

For each window graph and each q, with eigenvalues λ₁ ≤ … ≤ λ_n of L_q:

| Feature | Definition | Meaning |
|---|---|---|
| `lam_min` | λ₁ | behavioural circulation (0 when q = 0) |
| `lam_2` | λ₂ | connectivity of behaviour |
| `gap` | λ₂ − λ₁ | separation of the first mode |
| `lam_max` | λ_n | bipartite-like oscillation |
| `kirchhoff` | n · Σ_{λ_k > 1e−8} 1/λ_k | Kirchhoff-type resistance index |
| `spec_entropy` | −Σ p_k log p_k, p_k = λ_k / Σλ | spread of the spectrum |
| `hist_1..8` | fraction of eigenvalues in 8 equal bins on [0, 2] | spectral density shape |

`kirchhoff` is our operational definition. It equals the classical Kirchhoff index (scaled
for the normalised Laplacian) when q = 0. **Do not call it "the Hermitian Kirchhoff index"
in the paper** until the definition in Lin et al. (2021) has been checked against the
primary source; until then it is "a Kirchhoff-type index of the magnetic Laplacian".

### 4.5 Non-spectral Markov-chain features (the control that matters most)

Computed from the same windowed chain, with self-loops kept:

- `n_visited`, `n_cc`, component coverage
- entropy rate of the empirical transition matrix
- mean and max dwell time
- KL divergence of the window's state-visitation distribution from the clean reference
- Frobenius distance of the window's transition matrix from the clean reference
- **net-flow magnitude** `Σ |C − Cᵀ| / Σ C` — a non-spectral measure of directionality

If spectral features do not beat this set, the spectrum is not earning its place. The
net-flow feature specifically tests whether *direction* matters but the *spectrum* does not.

---

## 5. Per-agent scorers

For agent i in episode e, its **set** is `X_{e,i} = { ψ(G_{e,i,w}) : w = 1..W_e }`.
Every scorer outputs an anomaly score s_{e,i} where higher means more anomalous. All are fit
on clean fit-split episodes only.

| ID | Scorer | Definition |
|---|---|---|
| S0 | Robust distance | Median over windows of the Mahalanobis distance (Ledoit–Wolf covariance) from the clean window distribution |
| S1 | iForest, pointwise | Isolation forest on individual windows; agent score = mean window score |
| S2 | iForest, summarised | Isolation forest on one vector per agent: [mean, std, min, max] of its windows. This is what the earlier plan wrongly called siForest; Djidjev uses it as a baseline |
| S3 | **siForest** | See below |

### siForest (Djidjev 2024), as we implement it

- Points are windows; each carries a set ID `(episode, agent)`.
- Trees are built as in isolation forest (random feature, random split value within the
  node's range) with one change: **a node stops splitting when all its points belong to the
  same set**, regardless of how many points it holds. Also stop at max depth.
- A point's path length is the depth of the leaf it reaches. The original iForest adds a
  correction c(m) for unsplit leaves of size m; Djidjev does not specify this. Primary: no
  correction. Ablation: with correction. Record the choice.
- Set score: aggregate the points' scores by **minimum path length** (most anomalous window;
  Djidjev's reported choice) and by **mean** (persistent anomaly). Report both; primary is
  minimum.
- Hyperparameters: 100 trees (Djidjev's setting), subsample 256 points per tree, max depth
  ⌈log₂ 256⌉ = 8.

Because the scorers are interchangeable, siForest is evaluated, not assumed. If S0 matches
it, the paper says so.

---

## 6. Swarm-level statistics

These turn per-agent scores into the mechanism signature of §1.

### 6.1 Absolute and relative scores

- **Absolute** `s_{e,i}`: from §5, relative to clean flight.
- **Relative** `r_{e,i} = (s_{e,i} − median_j s_{e,j}) / (MAD_j s_{e,j} + ε)`: relative to
  the rest of the same swarm in the same episode.

Relative scoring is the method's main answer to distribution shift. Under `env`, everyone
flies in the same weather, so absolute scores rise for all agents while relative scores stay
flat. Under `act`, one agent stands out in both. Under `msg`, the absolute scores rise for
the victims and the liar is unusually *low* in relative terms.

A transductive siForest variant — fit on the 8 agents' windows of a single episode, as in
Djidjev's original setting where anomalies are present in the fitted data — gives a second
relative score `r'`. It is an ablation.

### 6.2 Exposure

From true positions and the public neighbour rule (each agent observes its k nearest
neighbours; k from Step 0), compute for each episode

```
E[j,i] = fraction of steps in which agent j had agent i among its observed neighbours
```

and the neighbourhood anomaly of agent i:

```
n_{e,i} = Σ_{j≠i} E[j,i] · s_{e,j}  /  Σ_{j≠i} E[j,i]
```

### 6.3 Episode-level statistics

| Statistic | Definition | Used for |
|---|---|---|
| `D_mean` | mean_i s_{e,i} | T1 detection, diffuse anomaly |
| `D_max` | max_i s_{e,i} | T1 detection, concentrated anomaly |
| `C` | max_i r_{e,i} | concentration: one agent stands out |
| `Q` | −min_i r_{e,i} | calm outlier: one agent unusually normal |

The mechanism map is the plane (`D_mean`, `C`), with `Q` as a third axis. The pre-registered
expectation, which the evaluation tests:

| Condition | D_mean | C | Q |
|---|---|---|---|
| clean | low | low | low |
| env | high | low | low |
| act | moderate | **high** | low |
| msg | high | low | **high** |

### 6.4 Localisation

- `act`: suspect = argmax_i s_{e,i}.
- `msg`: suspect = argmax_i (n_{e,i} − s_{e,i}) — the calm source of a disturbed
  neighbourhood.

**Caveat known in advance.** If k is close to N − 1, every agent observes nearly every other,
E is almost uniform, n_{e,i} is the same for everyone, and liar localisation reduces to
"pick the calmest agent". Step 0 measures the off-diagonal mean of E. If it exceeds 0.9,
liar localisation is moved to exploratory and the paper makes no claim about it.

---

## 7. What is novel, stated narrowly

Prior work uses each ingredient separately: policy graphs for explanation (Topin & Veloso;
Liu et al. 2023), magnetic/Hermitian Laplacians for directed-graph learning (MagNet) and
community detection (Fanuel et al.), isolation forests for anomaly detection, and Byzantine
threat models for MARL (BARDec-POMDP). The claims of this paper are:

1. **Mechanism inference**: in a swarm, environmental, action-level, and message-level faults
   leave distinct swarm-wide patterns of behavioural anomaly, and the pattern identifies the
   mechanism at matched damage.
2. **Directed behavioural spectra**: the magnetic-Laplacian spectrum of an agent's policy
   graph, including its circulation term λ_min, is a better per-agent behavioural anomaly
   score than state-marginal and non-spectral Markov-chain features.

Each is tested separately in `PHASE3_EVALUATION.md`. Neither depends on siForest being the
best scorer.
