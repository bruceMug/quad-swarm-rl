# Tier 2 — severity calibration table

Generated from `byzantine_experiments/results/tier2/*.json`. 69 rollout runs over 3 baseline seeds, 50 episodes each. Clean references reused from `byzantine_experiments/results/tier1` (same checkpoints, same collector).

Values are means over seeds. Read `reward deg %` together with its SEM — see the Tier 1 note; goal-RMSE degradation is the axis that can actually rank severities.

| config | f | goal RMSE deg % | **spillover %** | reward deg % (±SEM) | **survival** | guard rate | **t-to-impact s** | action dev | collisions Δ |
|---|---|---|---|---|---|---|---|---|---|
| `collusion` mirror_pair | 0.250 | 73.7 | 7.9 | 1083 ± 72 | 37% | 93% | 0.4 | 0.384 | 0.00 |
| `collusion` pincer | 0.250 | 181.2 | 5.8 | 1273 ± 61 | 96% | 10% | 4.9 | 0.463 | 0.02 |
| `collusion` same_waypoint | 0.250 | 177.8 | 7.9 | 1356 ± 66 | 67% | 30% | 4.6 | 0.412 | 1.26 |
| `liar` mirror | 0.125 | 9.9 | 8.9 | 10 ± 67 | 98% | 5% | n/a | 0.000 | 0.31 |
| `liar` offset | 0.125 | 5.6 | 2.9 | -29 ± 68 | 97% | 6% | n/a | 0.000 | 0.31 |
| `liar` stale | 0.125 | 9.4 | 9.1 | 31 ± 64 | 96% | 4% | n/a | 0.000 | 0.38 |
| `liar` mirror | 0.250 | 11.8 | 8.8 | 47 ± 68 | 97% | 5% | n/a | 0.000 | 0.66 |
| `liar` offset | 0.250 | 10.7 | 9.5 | 114 ± 74 | 97% | 5% | n/a | 0.000 | 0.60 |
| `liar` stale | 0.250 | 6.2 | 6.5 | -12 ± 65 | 100% | 4% | n/a | 0.000 | 0.35 |
| `signflip` full **★** | 0.125 | 28.9 | -6.8 | 309 ± 63 | 69% | 83% | n/a | 0.653 | 0.06 |
| `signflip` partial | 0.125 | 42.9 | 5.7 | 484 ± 61 | 73% | 67% | 0.7 | 0.215 | 0.11 |
| `signflip` full | 0.250 | 60.7 | 2.3 | 806 ± 65 | 45% | 83% | 0.6 | 0.652 | 0.02 |
| `signflip` partial | 0.250 | 67.6 | -3.6 | 999 ± 71 | 45% | 66% | 4.7 | 0.215 | 0.05 |
| `waypoint` a0.05 | 0.125 | 6.6 | 5.9 | 46 ± 65 | 100% | 5% | n/a | 0.065 | 0.23 |
| `waypoint` a0.10 | 0.125 | 8.6 | 5.4 | 56 ± 65 | 100% | 5% | n/a | 0.135 | 0.21 |
| `waypoint` a0.15 | 0.125 | 11.2 | 6.1 | 92 ± 63 | 100% | 6% | n/a | 0.206 | 0.27 |
| `waypoint` a0.20 **★** | 0.125 | 23.2 | -0.3 | 230 ± 87 | 96% | 15% | n/a | 0.259 | 0.11 |
| `waypoint` a0.3 | 0.125 | 77.8 | 3.5 | 470 ± 66 | 91% | 22% | n/a | 0.294 | 0.03 |
| `waypoint` a0.6 | 0.125 | 100.2 | -6.3 | 503 ± 61 | 98% | 13% | n/a | 0.412 | -0.03 |
| `waypoint` a1.0 | 0.125 | 116.9 | 2.9 | 602 ± 62 | 100% | 10% | n/a | 0.515 | 0.03 |
| `waypoint` a0.3 | 0.250 | 120.3 | -4.5 | 873 ± 64 | 85% | 22% | 7.1 | 0.313 | 0.23 |
| `waypoint` a0.6 | 0.250 | 167.7 | -0.4 | 1267 ± 65 | 69% | 29% | 5.8 | 0.403 | 1.28 |
| `waypoint` a1.0 | 0.250 | 168.1 | -3.8 | 1189 ± 60 | 57% | 41% | 6.7 | 0.503 | 1.83 |

★ = goal-RMSE degradation inside the 20–40 % band.

**spillover %** = degradation of the *honest* drones' goal RMSE. This is the column that distinguishes Tier 2 from Tier 1: Tier 1 never exceeded ~10 % here.

**survival** = fraction of episodes the attacker is still airborne at the end. A low number means the attack removes itself from the episode.

**guard rate** = fraction of steps the attacker spent below the survival guard's 0.6 m threshold, i.e. steps on which the guard blended it back toward honest. Derived from the trajectory, so it is exact regardless of the agent counters. A high number means the "attack" is substantially honest flight.

**t-to-impact** = seconds until cumulative team reward departs from the clean run by more than 2 clean σ, sustained. `n/a` = never separated.


## Matched-damage pairs against Tier 1

The paper's central comparison: *same damage, different mechanism*.

| Tier 1 canonical | its goal-RMSE deg | closest Tier 2 config | its deg | gap | T1 action dev | T2 action dev | T2 spillover |
|---|---|---|---|---|---|---|---|
| 1A gaussian sigma=0.25 | 23.6 % | `waypoint_a0.20_f0125` | 23.2 % | 0.4 pp | 0.467 | 0.259 | -0.3 % |
| 1B ou sigma_ou=0.06 | 35.3 % | `signflip_full_f0125` | 28.9 % | 6.4 pp | 0.213 | 0.653 | -6.8 % |
| 1C' stall d0.25/ms0.3 | 25.5 % | `waypoint_a0.20_f0125` | 23.2 % | 2.3 pp | 0.089 | 0.259 | -0.3 % |
