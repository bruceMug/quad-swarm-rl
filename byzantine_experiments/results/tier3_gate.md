# Tier 3 — efficacy gate

Thresholds: **honest-drone** goal-RMSE degradation ≥ 30 %, team-reward degradation ≥ 30 %, survival ≥ 70 %, action-magnitude ratio ≤ 1.5 (stealth runs only).

Clean references are Tier 1's, per baseline seed. `reward deg %` is reported with its SEM and **is not used to gate** — see design/tier2_design.md §10 for why that metric cannot rank severities here.

**Every honest-damage figure below is measured against the clean run restricted to the *same drone indices*, from the trajectories rather than the sidecar means, and carries a 95 % CI and a Welch p-value.** Per-episode goal RMSE has a ~42 % coefficient of variation on this task, so at n = 20 the CI is roughly ± 22 points — most of the distance to the gate. A run passes only if the point estimate clears 30 % **and** the CI excludes zero.

**The gate keys on `honest deg %`, not `all-drone deg %`.** The all-drone figure is gameable: an adversary maximises it by flying its own drone away, with no damage to anyone else. `own deg %` is shown alongside so the split is visible — a large `own` with a small `honest` is self-destruction, not an attack.


## Horizon sweep — how the adversary evolved during training

| adversary | channel | f | checkpoint | **honest deg % [95 % CI]** | p | own deg % | survival | action ratio | saturation | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 5M | **-9.8** [-29.3, 9.6] | 0.318 | -10.9 | 100% | 1.05 | 0% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 5M | **-13.4** [-31.0, 4.1] | 0.132 | 199.5 | 0% | 7.84 | 77% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 6M | **-26.7** [-44.9, -8.5] | 0.005 | 170.6 | 0% | 8.39 | 97% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 12M | **-14.8** [-34.6, 4.9] | 0.137 | 165.5 | 0% | 7.75 | 92% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 23M | **-13.3** [-29.9, 3.3] | 0.115 | 164.2 | 0% | 7.76 | 94% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 24M | **-8.2** [-172.0, 155.6] | 0.737 | 134.1 | 0% | 7.31 | 97% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 25M | **-2.5** [-28.6, 23.5] | 0.843 | 225.9 | 0% | 7.53 | 97% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 25M | **-6.2** [-21.6, 9.3] | 0.431 | 11.9 | 96% | 1.13 | 0% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 49M | **-8.3** [-26.7, 10.2] | 0.372 | 189.7 | 0% | 7.33 | 99% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 50M | **-11.0** [-41.1, 19.1] | 0.460 | 209.3 | 0% | 7.89 | 96% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 50M | **-13.3** [-30.4, 3.9] | 0.127 | 181.7 | 0% | 7.64 | 100% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 52M | **-10.0** [-27.1, 7.1] | 0.247 | -20.4 | 98% | 1.01 | 0% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 73M | **17.6** [-1.8, 37.0] | 0.075 | 215.1 | 0% | 6.62 | 100% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 74M | **11.5** [-6.7, 29.7] | 0.213 | 27.9 | 100% | 1.13 | 0% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 74M | **-4.9** [-28.8, 18.9] | 0.676 | 183.6 | 0% | 7.40 | 100% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 75M | **-5.9** [-34.0, 22.1] | 0.668 | 199.0 | 0% | 7.14 | 88% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 100M | **7.1** [-20.1, 34.2] | 0.599 | 200.7 | 0% | 6.86 | 100% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 101M | **-17.7** [-38.3, 3.0] | 0.091 | 169.3 | 0% | 7.75 | 93% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 101M | **-3.6** [-19.2, 12.1] | 0.652 | 21.0 | 96% | 1.19 | 0% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 103M | **-3.1** [-21.3, 15.0] | 0.733 | 190.3 | 0% | 7.57 | 100% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 149M | **-3.6** [-20.7, 13.5] | 0.674 | -2.6 | 98% | 1.03 | 0% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 149M | **-1.9** [-19.2, 15.4] | 0.824 | 182.9 | 0% | 7.27 | 100% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 150M | **0.3** [-21.7, 22.3] | 0.980 | 182.6 | 0% | 6.99 | 100% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 152M | **-11.4** [-35.0, 12.3] | 0.335 | 174.0 | 0% | 7.56 | 98% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 191M | **-26.4** [-42.3, -10.4] | 0.002 | 153.0 | 0% | 8.00 | 97% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 193M | **1.5** [-16.2, 19.3] | 0.863 | 198.0 | 0% | 7.19 | 100% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 193M | **0.2** [-17.0, 17.3] | 0.983 | 14.1 | 100% | 1.09 | 0% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 194M | **12.5** [-22.8, 47.8] | 0.472 | 188.3 | 0% | 6.71 | 100% | ❌ |
| adv_seed1_stealth5 | motors | 0.125 | 6M | **-20.8** [-37.9, -3.8] | 0.018 | 323.4 | 30% | 7.96 | 96% | ❌ |
| adv_seed1_stealth5 | motors | 0.250 | 12M | **-1.2** [-22.9, 20.4] | 0.910 | 405.5 | 0% | 6.95 | 96% | ❌ |
| adv_seed1_stealth5 | motors | 0.125 | 24M | **-9.5** [-27.2, 8.2] | 0.285 | 380.1 | 20% | 7.46 | 89% | ❌ |
| adv_seed1_stealth5 | motors | 0.250 | 24M | **2.6** [-23.6, 28.7] | 0.842 | 391.2 | 10% | 6.90 | 97% | ❌ |
| adv_seed1_stealth5 | motors | 0.125 | 49M | **8.6** [-16.1, 33.2] | 0.485 | 369.1 | 20% | 6.39 | 86% | ❌ |
| adv_seed1_stealth5 | motors | 0.250 | 49M | **-10.4** [-35.1, 14.4] | 0.400 | 404.4 | 25% | 7.75 | 95% | ❌ |
| adv_seed1_stealth5 | motors | 0.125 | 74M | **-10.3** [-29.3, 8.6] | 0.279 | 336.0 | 5% | 7.00 | 83% | ❌ |
| adv_seed1_stealth5 | motors | 0.250 | 74M | **-0.9** [-22.5, 20.6] | 0.932 | 399.2 | 5% | 6.69 | 92% | ❌ |
| adv_seed1_stealth5 | motors | 0.125 | 99M | **-4.0** [-23.8, 15.9] | 0.689 | 356.2 | 15% | 6.90 | 83% | ❌ |
| adv_seed1_stealth5 | motors | 0.250 | 100M | **1.6** [-23.3, 26.6] | 0.895 | 376.4 | 20% | 6.83 | 85% | ❌ |
| adv_seed1_stealth5 | motors | 0.125 | 149M | **17.2** [-13.9, 48.3] | 0.266 | 384.0 | 5% | 6.56 | 100% | ❌ |
| adv_seed1_stealth5 | motors | 0.250 | 150M | **-2.1** [-29.7, 25.5] | 0.876 | 360.2 | 5% | 6.20 | 73% | ❌ |
| adv_seed1_stealth5 | motors | 0.250 | 188M | **7.7** [-20.5, 35.9] | 0.579 | 415.3 | 0% | 6.02 | 82% | ❌ |
| adv_seed1_stealth5 | motors | 0.125 | 193M | **-0.5** [-26.4, 25.3] | 0.967 | 319.4 | 0% | 7.33 | 99% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 5M | **4.4** [-14.5, 23.4] | 0.645 | 2.9 | 100% | 1.07 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 5M | **14.7** [-3.2, 32.6] | 0.106 | 21.1 | 100% | 1.06 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 26M | **16.4** [0.4, 32.4] | 0.045 | 6.2 | 100% | 1.07 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 26M | **12.8** [-1.5, 27.2] | 0.078 | 26.7 | 100% | 1.12 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 50M | **-0.9** [-16.3, 14.6] | 0.912 | 9.1 | 96% | 1.12 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 53M | **15.9** [-0.3, 32.0] | 0.055 | 37.5 | 96% | 1.16 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 74M | **29.7** [8.0, 51.4] | 0.008 | 39.0 | 98% | 1.10 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 75M | **0.6** [-13.0, 14.3] | 0.927 | 14.1 | 98% | 1.11 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 101M | **6.9** [-8.7, 22.5] | 0.383 | 12.8 | 100% | 1.05 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 102M | **12.4** [-5.6, 30.4] | 0.173 | 21.6 | 96% | 1.09 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 151M | **10.0** [-6.4, 26.3] | 0.229 | 38.7 | 94% | 1.17 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 152M | **10.8** [-5.9, 27.5] | 0.202 | 20.9 | 94% | 1.11 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 193M | **22.1** [4.7, 39.4] | 0.013 | 26.5 | 98% | 1.11 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 196M | **9.9** [-6.3, 26.1] | 0.229 | 22.4 | 96% | 1.13 | 0% | ❌ |

## Verdict

**0 of 0 evaluated adversaries pass the gate.**

No adversary is usable for the detection experiments yet. The spec is explicit: *"An adversary that fails the efficacy gate is not usable. Retune the reward weights and retrain rather than proceeding with a weak attack."*

