# PHASE 3 — Detector Implementation Spec

**Audience:** GitHub Copilot / AI coding assistant
**Prerequisites:** Tiers 0–3 complete. Read `PHASE3_README.md`, `PHASE3_METHOD.md` and
`PHASE3_EVALUATION.md` in full before writing code. This file says *how to build*; those say
*what* and *why*. Where they disagree, `PHASE3_METHOD.md` wins and you flag the conflict.

---

## Package layout

All new code goes in a new subpackage. No upstream file is touched in this phase.

```
byzantine_experiments/
└── phase3/
    ├── __init__.py
    ├── data.py            D1  corpus loading, AuditorView, splits
    ├── labels.py          D1  the ONLY module that reads ground truth
    ├── baselines.py       D2  K1–K4, A3, W1, W2, O1
    ├── abstraction.py     D3  features, codebooks, quality checks
    ├── policy_graph.py    D4  windowing, count matrices, components
    ├── spectral.py        D4  magnetic Laplacian, ψ_q, Markov-chain features
    ├── scorers.py         D5  S0, S1, S2, S3 (siForest)
    ├── swarm.py           D6  relative scores, exposure, D_mean/D_max/C/Q, localisation
    ├── evaluate.py        D7  metrics, bootstrap, tables, figures (dev)
    └── lockbox.py         D7  the one-shot final run
byzantine_experiments/splits/phase3_splits.json
byzantine_experiments/results/phase3_dev/
byzantine_experiments/tests/test_phase3_*.py
scripts/phase3_*.sh
```

---

## D0 — Step 0 audit (no detector code)

Write `byzantine_experiments/design/phase3_step0_audit.md` answering every question below
from the repo and the data. Quote source lines and real array shapes. Where an answer is
"not recorded", say so — do not guess.

1. **Corpus inventory.** Every `.npz` relevant to Phase 3: condition, tier/variant, baseline
   seed, adversary seed, shift level, f, number of episodes, array shapes.
2. **Mechanism of each gate-passing attacker.** Action channel, message channel, or both.
   Specifically: what is Tier 3 in the final data — a learned action policy, a learned lie,
   or something else? This decides the `act`/`msg` labels.
3. **The matched triple.** Members, their team-damage distributions, the Mann–Whitney
   results already computed.
4. **Shift levels.** What each of the five levels changes, and clean-episode team damage per
   level with CIs. Does any level fall inside the triple's damage band?
5. **Observation layout.** Ego slice indices; neighbour block layout;
   `quads_neighbor_visible_num` (k) as used in training; whether neighbours are k-nearest
   recomputed every step.
6. **What was recorded.** For each corpus: true positions, velocities, attitudes, angular
   velocities? Executed actions? Honest actions? Broadcasts? Per-step neighbour indices?
   Collisions per agent pair, or only a per-step count? (K2 needs per-agent attribution. If
   it is not recorded, it can be derived from true positions with the collision radius from
   `gym_art/quadrotor_multi/collisions/` — document the radius used.)
7. **Timing.** Control frequency, episode length, early termination rate per condition,
   attack onset (from t = 0, or later?).
8. **White-box access.** Can attention weights be extracted from the checkpoints at rollout
   time for W1? Which module, which tensor?
9. **Leakage risks.** Every field in the corpora that would reveal the label or violate AM-2.
10. **Exposure preview.** For 20 clean episodes per seed, the mean off-diagonal of the
    exposure matrix E (`PHASE3_METHOD.md` §6.2).
11. **Proposed W** (window length) per `PHASE3_METHOD.md` §3.3.

Then generate `splits/phase3_splits.json` per `PHASE3_EVALUATION.md` §2 (seed 20261021),
and record the counts per split × condition × seed × shift level in the audit.

**Stop here.** The researcher confirms D1–D3 of `PHASE3_README.md` before D1 begins.

---

## D1 — Data access layer and leakage firewall

`data.py` exposes:

```python
class AuditorView:
    """Everything AM-2 permits, nothing else."""
    episode_id: str
    seed: int
    shift_level: int
    ego: np.ndarray          # (T, N, d_ego)  true ego kinematics
    actions: np.ndarray      # (T, N, d_act)  executed actions
    positions: np.ndarray    # (T, N, 3)      true positions (for exposure, K2, K4)
    length: int              # true episode length
    k_neighbours: int        # public neighbour rule

def load_split(split: str, *, allow_lockbox: bool = False) -> Iterator[AuditorView]: ...
```

`labels.py` is the only module that reads `byzantine_mask`, condition labels, adversary IDs,
honest actions or broadcasts. It exposes `get_labels(episode_id)`.

**Firewall test** (`test_phase3_firewall.py`): parse imports of every module under `phase3/`
except `labels.py`, `evaluate.py` and `lockbox.py`, and fail if any imports `labels`, or
references the strings `byzantine_mask`, `actions_honest`, or `broadcast`. The AM-1/AM-3
appendix baselines W1/W2/O1 live behind an explicit `auditor_model=` argument and are
excluded from this test by name — they are labelled, not hidden.

`load_split("lockbox")` raises unless `allow_lockbox=True`, which only `lockbox.py` passes.

---

## D2 — Kill-test baselines

Implement K1–K4 (`PHASE3_EVALUATION.md` §4) in `baselines.py`, each returning per-episode
and per-agent scores with the same interface the method will use later:

```python
def score_episode(view: AuditorView) -> dict:  # {"agent": (N,), "episode": float}
```

Run on `dev`, matched set, shift level 0 and highest level. Write
`results/phase3_dev/KILL_TEST.md` with a table of T1 AUROC (with bootstrap CIs) and T3 top-1
for every baseline, and the decision from `PHASE3_EVALUATION.md` §6.

Also run O1 here. It must reach ≈ 1.0; if it does not, stop and report — the labels or the
recorded broadcasts are wrong.

**Stop.** The researcher reads the kill-test report before D3.

---

## D3 — Behavioural abstraction

`abstraction.py`:

- `behavioural_features(view) -> (T, N, d)` — `[ego, action]` per `PHASE3_METHOD.md` §3.1.
  Neighbour observations excluded; add a test asserting the feature dimension equals
  d_ego + d_act.
- `fit_codebook(seed, K) -> Codebook` — StandardScaler + `sklearn.cluster.KMeans`
  (`n_init=10`, fixed `random_state`) on clean `fit` episodes of that seed. Subsample to at
  most 500k points. Persist with `joblib` to `results/phase3_dev/codebooks/`.
- `assign(view, codebook) -> (T, N)` integer abstract states.
- `codebook_quality(codebook) -> dict` — cluster occupancy, held-out first-order vs
  zeroth-order next-state log-likelihood (`PHASE3_METHOD.md` §3.2).

Fit K ∈ {16, 32, 64} for each seed. Also fit one cross-seed codebook (all seeds' clean `fit`)
at K = 32 for the ablation.

---

## D4 — Policy graphs and spectral features

`policy_graph.py`:

- `windows(length, W) -> list[(start, end)]` — stride W/2; drop a final window shorter than W.
- `count_matrix(states_1d, start, end, K) -> (K, K)` transition counts.
- `largest_component(C) -> (C_sub, node_idx, coverage)` — self-loops removed, visited nodes
  only, largest weakly connected component.

`spectral.py`:

- `magnetic_laplacian(C, q) -> (n, n) complex` — exactly `PHASE3_METHOD.md` §4.1. Assert
  `np.allclose(L, L.conj().T)` before returning.
- `fingerprint(C, q) -> np.ndarray` — features of §4.4 using `np.linalg.eigvalsh`. Graphs
  with fewer than 3 nodes in the component return a fixed sentinel vector plus a flag
  feature; count how often this happens per condition.
- `markov_features(C_with_selfloops, reference) -> np.ndarray` — features of §4.5.
- `episode_features(view, codebook, W, qs) -> dict` — per agent, per window: ψ for each q,
  Markov features, window bounds. Cache to `.npz` keyed by (episode, codebook, W).

Performance: graphs are at most 64 × 64, so dense `eigvalsh` is fine. Parallelise over
episodes with `joblib`. Do not use `scipy.sparse.linalg.eigsh`.

**Required unit tests** (`test_phase3_spectral.py`), all values verified numerically by the
advisor:

| Test | Input | Expected |
|---|---|---|
| directed 3-cycle, q = 1/4 | `C = [[0,1,0],[0,0,1],[1,0,0]]` | eigenvalues ≈ {0.13397, 1.0, 1.86603}; λ_min = 1 − √3/2 |
| directed 3-cycle, q = 0 | same | {0, 1.5, 1.5} |
| scale invariance | 10 × the cycle, q = 1/4 | identical to the cycle |
| balanced graph | `C + Cᵀ` of the cycle, q = 1/4 | {0, 1.5, 1.5}; λ_min = 0 |
| Hermitian | random non-negative 32 × 32 C, any q | `L == L^H`, all eigenvalues in [0, 2] |
| permutation | randomly relabelled nodes | identical fingerprint |
| self-loops | add a diagonal | identical fingerprint |

---

## D5 — Per-agent scorers

`scorers.py` implements S0–S3 of `PHASE3_METHOD.md` §5 behind one interface:

```python
class Scorer(Protocol):
    def fit(self, sets: dict[tuple, np.ndarray]) -> "Scorer": ...   # (episode, agent) -> (W_e, D)
    def score(self, sets: dict[tuple, np.ndarray]) -> dict[tuple, float]: ...  # higher = more anomalous
```

**siForest must be written from scratch** in NumPy (sklearn's tree builder cannot be given
a set-aware stopping rule without patching its Cython). Requirements:

- Recursive isolation tree: pick a feature uniformly, split uniformly in the node's
  [min, max]; stop when the node has one point, **when all points share one set ID**, or at
  max depth 8.
- 100 trees, 256 points subsampled per tree, fixed RNG.
- Path length of a point = depth of its leaf. `c_correction: bool = False` adds the iForest
  c(m) term for leaves of size m (ablation).
- Point score `2^(−E[h(x)] / c(256))`, as in iForest. Set score: max point score
  (≡ Djidjev's "minimum" under sklearn's sign convention) or mean point score.
- `mode="inductive"` (fit on clean, score others) and `mode="transductive"` (fit and score
  the 8 agents of one episode together).

**siForest tests** (`test_phase3_scorers.py`):

1. Synthetic: 20 Gaussian sets, one shifted by 3σ → the shifted set ranks first in ≥ 95% of
   100 random seeds.
2. Degenerate equivalence: when every set has exactly one point, mean path lengths match a
   plain from-scratch isolation forest on the same data and RNG.
3. A set of identical points is never split (stopping rule fires).
4. Determinism under fixed RNG.
5. Interface parity: S0–S3 accept and return the same structures.

---

## D6 — Swarm-level statistics

`swarm.py`:

- `relative_scores(s: (N,)) -> (N,)` — robust z vs the swarm median and MAD.
- `exposure(positions, k) -> (N, N)` — `PHASE3_METHOD.md` §6.2, from true positions.
- `neighbourhood_anomaly(E, s) -> (N,)`.
- `episode_stats(s, r) -> {"D_mean", "D_max", "C", "Q"}`.
- `localise(s, n, mechanism) -> agent index`.
- `MechanismRule` — a depth-2 decision rule on (`D_mean`, `C`, `Q`) whose thresholds are fit
  on `dev` by grid search maximising balanced accuracy. Also fit a multinomial logistic
  regression on the same four statistics as a check that the rule is not leaving accuracy
  on the table. The rule is primary because it is interpretable and has four parameters.

---

## D7 — Evaluation

`evaluate.py` (dev) and `lockbox.py` (final) implement `PHASE3_EVALUATION.md` §§3–8:
metrics, cluster-stratified bootstrap, paired comparisons, Holm correction on the five
primary comparisons, time-to-detection on prefixes, and every figure and table in §8.

Figures: one script per figure under `scripts/phase3_fig*.py`, reading only cached results,
colour-blind-safe palette, vector PDF output.

Before the lockbox run, write `design/phase3_frozen_config.json` with every choice made on
`dev`: K, W, q, scorer, aggregation, c-correction, thresholds τ, mechanism-rule thresholds.

---

## Compute and order of work

All of Phase 3 is CPU work on cached rollouts — no GPU training. Expected cost on the cluster:
feature extraction is the bulk (episodes × agents × windows × 5 values of q, each a ≤ 64 × 64
eigendecomposition), a few hours with 32 cores. Cache aggressively; never recompute spectra
inside an evaluation loop.

| Step | Gate to pass before the next |
|---|---|
| D0 | Audit written; researcher confirms D1–D3 |
| D1 | Firewall test passes; lockbox guarded |
| D2 | O1 ≈ 1.0; kill-test report written; researcher reads it |
| D3 | Codebook quality reported for every seed and K |
| D4 | All seven spectral tests pass |
| D5 | All five scorer tests pass |
| D6 | Mechanism map (Fig 1) produced on `dev` |
| D7 | Frozen config written; lockbox run once |

---

## Required output: `byzantine_experiments/design/phase3_design.md`

Written incrementally as each deliverable completes, not at the end:

1. **Step 0 summary** — the key facts from the audit that shaped implementation.
2. **Firewall** — how AM-2 is enforced, and the test that proves it.
3. **Kill-test results** — the table, the decision taken, and its consequences.
4. **Codebooks** — quality metrics per seed and K; whether the chain has sequential
   structure (first-order gain); occupancy plots.
5. **Spectral implementation** — the exact code path for the magnetic Laplacian, test
   results, the sentinel-graph frequency per condition.
6. **siForest** — your implementation, the stopping and scoring choices, test results, and
   any ambiguity in Djidjev's description you had to resolve.
7. **Swarm statistics** — exposure statistics per seed; whether liar localisation stayed
   primary or moved to exploratory, and why.
8. **Dev results** — every hypothesis H1–H5 with numbers and CIs, every figure.
9. **Frozen config** — the full content of `phase3_frozen_config.json` with a one-line
   justification per choice.
10. **Deviations** — every departure from `PHASE3_METHOD.md` or this spec, with reasons.
11. **Lockbox result** — numbers, the date, the git SHA, and the `LOCKBOX_LOG.md` entry.
12. **Anything that surprised you.** Especially: what does a liar's victim's policy graph
    actually look like?
