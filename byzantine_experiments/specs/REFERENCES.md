# REFERENCES — Phase 3

**Rule:** specs are self-contained; nobody implements from a paper or a summary. This file
says which primary source backs each part of the method, what it is used for in the paper,
and how far it has been verified. Before any citation goes into the paper, the researcher
reads the cited section of the primary source.

Verification status:
- **V** — verified against the primary source on 2026-10-21 (content or bibliographic record)
- **B** — bibliographic record confirmed, content not re-checked for this phase
- **C** — check before citing; known or suspected problem

---

## Core method

| Source | Used for | Status |
|---|---|---|
| C. Djidjev, *siForest: Detecting Network Anomalies with Set-Structured Isolation Forest*, arXiv:2412.06015, 2024 | Scorer S3. Verified: the stopping rule (stop when all points in a node share one set), set scores by minimum or mean, 100 trees, evaluation on synthetic network-scan data, summarisation baseline. No code link in the paper. Note the paper calls the method both "Set-Structured" (title) and "Set-Partitioned" (body). | **V** |
| F. T. Liu, K. M. Ting, Z.-H. Zhou, *Isolation Forest*, ICDM 2008 | Scorers S1–S3; path-length normalisation c(n); score 2^(−E[h]/c) | **B** |
| Z. Zhang, Y. He, N. Brugnone, M. Perlmutter, M. Hirn, *MagNet: A Neural Network for Directed Graphs*, NeurIPS 2021, arXiv:2102.11391 | Magnetic Laplacian with charge q; our phase uses net-flow fraction instead of raw weight difference (`PHASE3_METHOD.md` §4.1) | **V** |
| M. Fanuel, C. M. Alaíz, J. A. K. Suykens, *Magnetic eigenmaps for community detection in directed networks*, Phys. Rev. E 95, 2017, arXiv:1606.07359 | Magnetic Laplacian reveals directed cyclic structure; motivation for reading λ_min as circulation | **B** — read before citing the circulation interpretation |
| J. Liu, X. Li, *Hermitian-adjacency matrices and Hermitian energies of mixed graphs*, Linear Algebra Appl. 466, 2015 | Hermitian adjacency (±i for one-way edges) = our q = 1/4 unweighted case | **B** |
| K. Guo, B. Mohar, *Hermitian adjacency matrix of digraphs and mixed graphs*, J. Graph Theory, 2017 | Same matrix, independently introduced; cite together with Liu & Li | **B** |
| N. Topin, M. Veloso, *Generation of Policy-Level Explanations for Reinforcement Learning*, AAAI 2019 | The policy-graph concept (abstract Markov chain over agent behaviour) | **B** |
| T. Zahavy, N. Ben-Zrihem, S. Mannor, *Graying the black box: Understanding DQNs*, ICML 2016 | Clustering-based behavioural abstraction | **B** |
| T. Liu et al., *A novel policy-graph approach with natural language and counterfactual abstractions for explaining RL agents*, JAAMAS 37(34), 2023, doi:10.1007/s10458-023-09615-8 | Related work: predicate-based policy graphs. This is the journal paper the proposal cited as "Liu et al."; not the AAMAS conference version | **B** |

## Kirchhoff index — handle with care

| Source | Used for | Status |
|---|---|---|
| "H. Zhang et al., AIMS Mathematics 10(10), 2025" (from the proposal) | — | **C — do not cite.** Could not be located as described. |
| Deng & Wu, *Research on the Consensus Convergence Rate of Multi-Agent Systems Based on Hermitian Kirchhoff Index Measurement*, Entropy 27(10):1035, 2025 | Possibly the paper the proposal meant. Context only: links a Hermitian Kirchhoff index to consensus convergence on MAS topologies, not policy graphs | **C** — read and compare its definition with ours |
| Lin et al., Hermitian Kirchhoff index, Math. Probl. Eng. 2021, doi:10.1155/2021/5534472 | Original definition, as reported by the earlier research pass | **C** — confirm the DOI and the definition before using the name "Hermitian Kirchhoff index" |

Until both C items are checked, the paper calls our feature "a Kirchhoff-type index of the
magnetic Laplacian" (`PHASE3_METHOD.md` §4.4).

## Threat model and attacks

| Source | Used for | Status |
|---|---|---|
| S. Li et al., *Byzantine robust cooperative MARL as a Bayesian game* (BARDec-POMDP), ICLR 2024, arXiv:2305.12872 | Formal Byzantine threat model for MARL; related work | **B** |
| A. Gleave et al., *Adversarial Policies: Attacking Deep RL*, ICLR 2020, arXiv:1905.10615 | Tier 3 training protocol | **B** |
| R. Sommer, V. Paxson, *Outside the Closed World: On Using Machine Learning for Network Intrusion Detection*, IEEE S&P 2010 | Why cheap baselines and realistic evaluation matter for anomaly detectors; supports the kill test and the auditor-model framing | **B** |

## Environment and baselines

| Source | Used for | Status |
|---|---|---|
| S. Batra, Z. Huang, A. Petrenko, T. Kumar, A. Molchanov, G. S. Sukhatme, *Decentralized Control of Quadrotor Swarms with End-to-end Deep RL*, CoRL 2021 (PMLR 164), **arXiv:2109.07735** | The swarm policy, attention encoder, `mix` scenario | **V** — the ID in `research_execution_plan.md` (2111.12084) was wrong; use this one |
| Z. Huang et al., *QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep RL with Direct Thrust Control*, arXiv:2306.09537, 2023 | The simulator | **B** |
| W. Duan, J. Lu, J. Xuan, *Bayesian Ego-graph Inference for Networked MARL* (BayesG), NeurIPS 2025, arXiv:2509.16606 | Related work only (README decision D2). Its posterior is a mean-field Bernoulli over edge masks with Gumbel-Softmax and a GCN — not a VGAE, as the proposal states | **B** |

## Statistics

| Source | Used for | Status |
|---|---|---|
| R. Agarwal et al., *Deep RL at the Edge of the Statistical Precipice*, NeurIPS 2021, arXiv:2108.13264 | Reporting with bootstrap CIs | **B** |
| E. R. DeLong, D. M. DeLong, D. L. Clarke-Pearson, *Comparing the areas under two or more correlated ROC curves*, Biometrics 44, 1988 | Paired AUROC comparison | **B** |
| S. Holm, *A simple sequentially rejective multiple test procedure*, Scand. J. Statist. 6, 1979 | Multiple-comparison correction | **B** |
| O. Ledoit, M. Wolf, *A well-conditioned estimator for large-dimensional covariance matrices*, J. Multivariate Anal., 2004 | Covariance for scorer S0 | **B** |

---

## Your own summaries

Store them in `notes/UNVERIFIED_<paper>.md`. Before Related Work is written, check each one
against the table above and the primary source, and rename the file once checked. Nothing
under `notes/` is read by Copilot during implementation.
