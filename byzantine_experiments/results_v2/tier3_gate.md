# Tier 3 — efficacy gate

Thresholds: **honest-drone** goal-RMSE degradation ≥ 10 %, team-reward degradation ≥ 10 %, survival ≥ 70 %, action-magnitude ratio ≤ 1.5 (stealth runs only).

Clean references are Tier 1's, per baseline seed. `reward deg %` is reported with its SEM and **is not used to gate** — see design/tier2_design.md §10 for why that metric cannot rank severities here.

**Every honest-damage figure below is measured against the clean run restricted to the *same drone indices*, from the trajectories rather than the sidecar means, and carries a 95 % CI and a Welch p-value.** Per-episode goal RMSE has a ~42 % coefficient of variation on this task, so at n = 20 the CI is roughly ± 22 points — most of the distance to the gate. A run passes only if the point estimate clears 10 % **and** the CI excludes zero.

**The gate keys on `honest deg %`, not `all-drone deg %`.** The all-drone figure is gameable: an adversary maximises it by flying its own drone away, with no damage to anyone else. `own deg %` is shown alongside so the split is visible — a large `own` with a small `honest` is self-destruction, not an attack.


## Horizon sweep — how the adversary evolved during training

| adversary | channel | f | checkpoint | **honest deg % [95 % CI]** | p | own deg % | survival | action ratio | saturation | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 5M | **-3.7** [-11.0, 3.6] | 0.333 | -10.0 | 98% | 1.00 | 0% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 5M | **-4.2** [-13.2, 4.9] | 0.380 | 197.9 | 0% | 6.89 | 76% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 6M | **2.3** [-5.6, 10.2] | 0.559 | 179.4 | 0% | 6.77 | 97% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 8M | **-0.0** [-7.7, 7.7] | 0.995 | -4.9 | 98% | 1.05 | 0% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 12M | **-1.8** [-10.4, 6.9] | 0.689 | 188.2 | 0% | 6.83 | 93% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 23M | **-1.9** [-9.6, 5.7] | 0.619 | 185.0 | 0% | 6.92 | 94% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 24M | **-1.4** [-10.9, 8.0] | 0.769 | -4.7 | 100% | 1.05 | 0% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 24M | **-0.8** [-8.0, 6.5] | 0.834 | 183.1 | 0% | 6.89 | 97% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 25M | **-6.4** [-16.4, 3.6] | 0.227 | 187.4 | 0% | 7.07 | 97% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 25M | **6.4** [-2.2, 15.0] | 0.125 | -4.7 | 100% | 1.03 | 0% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 48M | **-2.4** [-10.5, 5.7] | 0.568 | -5.3 | 98% | 1.04 | 0% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 49M | **1.7** [-5.4, 8.7] | 0.628 | 186.2 | 0% | 6.85 | 99% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 50M | **-3.5** [-12.4, 5.5] | 0.451 | 187.7 | 0% | 6.99 | 97% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 50M | **-3.5** [-10.4, 3.4] | 0.327 | 183.5 | 0% | 7.06 | 100% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 52M | **-3.7** [-13.2, 5.9] | 0.460 | -2.4 | 98% | 1.05 | 0% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 73M | **-2.6** [-9.9, 4.7] | 0.484 | 183.0 | 0% | 6.94 | 100% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 74M | **4.3** [-3.6, 12.3] | 0.257 | -1.3 | 98% | 1.02 | 0% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 74M | **-2.4** [-10.5, 5.6] | 0.555 | 181.9 | 0% | 6.94 | 100% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 75M | **-9.4** [-18.0, -0.9] | 0.051 | 183.2 | 0% | 6.97 | 90% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 76M | **2.9** [-6.5, 12.4] | 0.525 | 4.2 | 98% | 1.04 | 0% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 100M | **-6.6** [-15.7, 2.5] | 0.180 | 177.5 | 0% | 7.19 | 100% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 101M | **-6.9** [-15.9, 2.2] | 0.159 | 182.7 | 0% | 6.99 | 94% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 101M | **-3.2** [-10.7, 4.3] | 0.415 | -9.3 | 100% | 1.04 | 0% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 101M | **2.8** [-5.3, 10.9] | 0.487 | -2.3 | 100% | 1.04 | 0% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 103M | **-0.5** [-7.9, 6.9] | 0.894 | 182.5 | 0% | 7.00 | 100% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 149M | **1.9** [-3.4, 7.2] | 0.459 | -10.3 | 100% | 0.99 | 0% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 149M | **-5.4** [-13.2, 2.4] | 0.195 | 182.6 | 0% | 7.05 | 100% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 150M | **-0.2** [-7.2, 6.9] | 0.960 | 180.5 | 0% | 6.97 | 100% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 150M | **-4.2** [-13.8, 5.5] | 0.409 | -5.7 | 98% | 1.09 | 0% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 152M | **-6.4** [-15.0, 2.2] | 0.167 | 185.5 | 0% | 7.07 | 97% | ❌ |
| adv_seed1_stealth0 | motors | 0.250 | 191M | **-6.9** [-14.9, 1.1] | 0.113 | 184.1 | 0% | 7.01 | 96% | ❌ |
| adv_seed1_stealth0 | observation | 0.125 | 193M | **-4.4** [-13.5, 4.8] | 0.362 | 184.3 | 0% | 7.04 | 100% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 193M | **2.4** [-5.1, 9.8] | 0.522 | -7.1 | 100% | 1.03 | 0% | ❌ |
| adv_seed1_stealth0 | motors | 0.125 | 194M | **-1.5** [-8.5, 5.5] | 0.672 | 183.1 | 0% | 7.04 | 100% | ❌ |
| adv_seed1_stealth0 | observation (lie only) | 0.125 | 196M | **2.6** [-7.0, 12.2] | 0.581 | -9.1 | 100% | 1.00 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 8M | **1.2** [-6.3, 8.8] | 0.743 | 18.3 | 100% | 1.12 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 8M | **3.6** [-4.8, 12.0] | 0.377 | 13.4 | 98% | 1.11 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 23M | **4.5** [-4.2, 13.3] | 0.285 | 11.0 | 96% | 1.10 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 24M | **2.0** [-6.0, 10.0] | 0.613 | 1.5 | 98% | 1.06 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 48M | **9.2** [0.2, 18.1] | 0.037 | 12.8 | 100% | 1.10 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 51M | **-1.3** [-8.4, 5.8] | 0.713 | 4.3 | 98% | 1.08 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 75M | **3.4** [-4.8, 11.7] | 0.399 | 16.1 | 100% | 1.09 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 77M | **1.9** [-3.5, 7.2] | 0.475 | 8.8 | 96% | 1.09 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 99M | **9.3** [-1.9, 20.6] | 0.086 | 16.3 | 98% | 1.09 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 101M | **7.8** [-1.4, 17.1] | 0.080 | 26.4 | 98% | 1.15 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 149M | **2.3** [-5.2, 9.7] | 0.534 | 13.6 | 100% | 1.12 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 151M | **0.5** [-6.5, 7.6] | 0.879 | 0.7 | 100% | 1.05 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 195M | **3.7** [-4.3, 11.8] | 0.338 | 7.4 | 98% | 1.07 | 0% | ❌ |
| adv_seed2_stealth0 | observation (lie only) | 0.125 | 195M | **2.0** [-5.3, 9.3] | 0.569 | 4.8 | 98% | 1.07 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 4M | **6.5** [-1.7, 14.7] | 0.119 | 18.1 | 94% | 1.10 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 5M | **11.5** [3.1, 20.0] | 0.007 | 15.3 | 98% | 1.08 | 0% | ✅ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 5M | **9.2** [0.7, 17.7] | 0.033 | 19.4 | 98% | 1.10 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 23M | **6.1** [-1.2, 13.3] | 0.104 | 6.3 | 100% | 1.05 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 26M | **4.8** [-1.2, 10.8] | 0.115 | 7.6 | 100% | 1.08 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 26M | **8.6** [-0.7, 17.9] | 0.070 | 12.2 | 100% | 1.06 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 50M | **7.6** [0.8, 14.3] | 0.027 | -0.3 | 98% | 1.03 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 51M | **8.3** [-2.1, 18.8] | 0.120 | 10.2 | 100% | 1.06 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 53M | **5.2** [-3.6, 13.9] | 0.233 | 10.0 | 98% | 1.04 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 74M | **11.3** [0.4, 22.2] | 0.043 | 21.5 | 100% | 1.10 | 0% | ✅ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 74M | **3.1** [-4.6, 10.9] | 0.418 | 0.2 | 98% | 1.04 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 75M | **2.4** [-3.8, 8.7] | 0.435 | 12.8 | 96% | 1.06 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 98M | **6.5** [-0.1, 13.1] | 0.053 | 19.0 | 98% | 1.08 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 101M | **14.2** [2.4, 25.9] | 0.019 | 10.7 | 98% | 1.04 | 0% | ✅ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 102M | **4.4** [-4.6, 13.3] | 0.329 | 11.8 | 98% | 1.08 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 151M | **13.4** [3.2, 23.5] | 0.012 | 11.9 | 94% | 1.04 | 0% | ✅ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 151M | **9.4** [-0.5, 19.2] | 0.061 | 21.8 | 98% | 1.09 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 152M | **12.6** [1.8, 23.4] | 0.022 | 13.0 | 96% | 1.03 | 0% | ✅ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 192M | **6.7** [-1.1, 14.5] | 0.079 | 9.0 | 100% | 1.07 | 0% | ❌ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 193M | **13.0** [3.0, 22.9] | 0.011 | 1.4 | 98% | 1.01 | 0% | ✅ |
| adv_seed3_stealth0 | observation (lie only) | 0.125 | 196M | **5.5** [-1.8, 12.9] | 0.134 | 14.6 | 100% | 1.06 | 0% | ❌ |

## Verdict

**0 of 0 evaluated adversaries pass the gate.**

No adversary is usable for the detection experiments yet. The spec is explicit: *"An adversary that fails the efficacy gate is not usable. Retune the reward weights and retrain rather than proceeding with a weak attack."*

