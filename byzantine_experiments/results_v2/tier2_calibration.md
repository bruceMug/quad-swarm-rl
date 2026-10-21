# Tier 2 — severity calibration table

Generated from `byzantine_experiments/results_v2/tier2/*.json`. 57 rollout runs over 3 baseline seeds, 50 episodes each. Clean references reused from `byzantine_experiments/results_v2/tier1` (same checkpoints, same collector).

Values are means over seeds. Read `reward deg %` together with its SEM — see the Tier 1 note; goal-RMSE degradation is the axis that can actually rank severities.

| config | f | **paired honest deg %** | goal RMSE deg % | **spillover %** | reward deg % (±SEM) | **survival** | guard rate | **t-to-impact s** | action dev | collisions Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| `collusion` mirror_pair | 0.250 | -2.6 (3 seeds) | 66.3 | -1.9 | 2384 ± 227 | 51% | 94% | 0.4 | 0.391 | -0.07 |
| `collusion` pincer | 0.250 | -2.4 (3 seeds) | 172.9 | -1.7 | 2880 ± 208 | 99% | 8% | 8.5 | 0.489 | -0.15 |
| `collusion` same_waypoint | 0.250 | -3.6 (3 seeds) | 157.9 | -2.9 | 2970 ± 203 | 77% | 30% | 10.2 | 0.405 | 1.21 |
| `liar` mirror **†** | 0.125 | 4.2 (3 seeds) | 5.4 | 5.0 | 90 ± 234 | 99% | 5% | n/a | 0.000 | 0.41 |
| `liar` offset **†** | 0.125 | 4.5 (3 seeds) | 5.9 | 5.3 | 148 ± 225 | 99% | 5% | n/a | 0.000 | 0.47 |
| `liar` stale **†** | 0.125 | 5.5 (3 seeds) | 6.7 | 6.3 | 183 ± 225 | 98% | 4% | n/a | 0.000 | 0.35 |
| `liar` mirror | 0.250 | 8.0 (3 seeds) | 10.9 | 8.7 | 251 ± 222 | 98% | 5% | n/a | 0.000 | 0.70 |
| `liar` offset | 0.250 | 5.0 (3 seeds) | 9.3 | 5.7 | 244 ± 246 | 100% | 6% | n/a | 0.000 | 0.66 |
| `liar` stale | 0.250 | 6.7 (3 seeds) | 9.8 | 7.4 | 173 ± 220 | 97% | 5% | n/a | 0.000 | 0.53 |
| `signflip` full | 0.125 | -0.6 (3 seeds) | 35.6 | 0.1 | 956 ± 216 | 72% | 83% | n/a | 0.666 | 0.01 |
| `signflip` partial | 0.125 | -1.7 (3 seeds) | 39.2 | -1.0 | 1121 ± 216 | 63% | 66% | n/a | 0.212 | -0.05 |
| `signflip` full | 0.250 | -3.5 (3 seeds) | 58.6 | -2.8 | 1861 ± 223 | 51% | 82% | 0.8 | 0.669 | -0.03 |
| `signflip` partial | 0.250 | -3.2 (3 seeds) | 67.5 | -2.5 | 2368 ± 224 | 48% | 67% | 0.7 | 0.213 | -0.01 |
| `waypoint` a0.3 | 0.125 | 2.1 (3 seeds) | 73.6 | 2.8 | 1211 ± 233 | 93% | 22% | n/a | 0.306 | 0.13 |
| `waypoint` a0.6 | 0.125 | 1.1 (3 seeds) | 103.7 | 1.8 | 1493 ± 209 | 97% | 12% | n/a | 0.437 | 0.02 |
| `waypoint` a1.0 | 0.125 | -1.6 (3 seeds) | 106.2 | -0.9 | 1409 ± 208 | 99% | 7% | n/a | 0.526 | -0.13 |
| `waypoint` a0.3 | 0.250 | -1.8 (3 seeds) | 117.1 | -1.1 | 2009 ± 225 | 85% | 22% | 13.3 | 0.311 | 0.11 |
| `waypoint` a0.6 | 0.250 | -0.8 (3 seeds) | 157.0 | -0.1 | 2893 ± 210 | 73% | 30% | 7.6 | 0.411 | 1.30 |
| `waypoint` a1.0 | 0.250 | -1.4 (3 seeds) | 159.5 | -0.7 | 2966 ± 206 | 63% | 40% | 7.2 | 0.523 | 1.99 |

† = matched-triple candidate: family validated below, f = 0.125. (Replaces the retired 20–40 % band ★; decision D1.) **paired honest deg %** is the only reportable degradation in the v2 corpus (design 6.18).

**spillover %** = degradation of the *honest* drones' goal RMSE. This is the column that distinguishes Tier 2 from Tier 1: Tier 1 never exceeded ~10 % here.

**survival** = fraction of episodes the attacker is still airborne at the end. A low number means the attack removes itself from the episode.

**guard rate** = fraction of steps the attacker spent below the survival guard's 0.6 m threshold, i.e. steps on which the guard blended it back toward honest. Derived from the trajectory, so it is exact regardless of the agent counters. A high number means the "attack" is substantially honest flight.

**t-to-impact** = seconds until cumulative team reward departs from the clean run by more than 2 clean σ, sustained. `n/a` = never separated.


## Family-level validation (decision D1, replaces the 20–40 % band)

A family is validated when its runs together significantly damage the honest team (one-sample t-test on per-run **paired** honest goal-RMSE degradation, mean > 0, p < 0.05). This replaces the 20–40 % band, which was set on legacy numbers and which no v2 configuration reaches.

| family | runs | mean | positive | family p | status |
|---|---|---|---|---|---|
| collusion | 9 | -2.8 % | 1/9 | 0.0429 | significantly HELPS the honest team |
| liar | 18 | +5.7 % | 16/18 | 0.0000 | VALIDATED -- damaging |
| signflip | 12 | -2.2 % | 2/12 | 0.0106 | significantly HELPS the honest team |
| waypoint | 18 | -0.4 % | 11/18 | 0.5939 | no family-level effect |


## Matched-damage pairs against Tier 1 (legacy, superseded)

Kept for the record: these use the legacy (unpaired, seeding-bug) Tier 1 canonical numbers. The paper's *same damage, different mechanism* comparison is the pre-registered matched triple (`analysis/matched_triple.py`, paired TOST).

| Tier 1 canonical | its goal-RMSE deg | closest Tier 2 config | its deg | gap | T1 action dev | T2 action dev | T2 spillover |
|---|---|---|---|---|---|---|---|
| 1A gaussian sigma=0.25 | 23.6 % | `signflip_full_f0125` | 35.6 % | 12.0 pp | 0.467 | 0.666 | 0.1 % |
| 1B ou sigma_ou=0.06 | 35.3 % | `signflip_full_f0125` | 35.6 % | 0.3 pp | 0.213 | 0.666 | 0.1 % |
| 1C' stall d0.25/ms0.3 | 25.5 % | `signflip_full_f0125` | 35.6 % | 10.1 pp | 0.089 | 0.666 | 0.1 % |
