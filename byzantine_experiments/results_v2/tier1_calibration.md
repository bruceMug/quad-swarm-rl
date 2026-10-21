# Tier 1 — severity calibration table

Generated from `byzantine_experiments/results_v2/tier1/*.json`. 87 rollout runs over 3 baseline seeds, 50 episodes each.

Each Byzantine config is compared against the clean rollout of the **same** baseline checkpoint, collected with the same collector. Values are means over seeds; `+-` is the spread across seeds.

> **Read the two degradation columns together.** `reward deg %` is the figure the spec asks for, but mean episode reward is ~1.5 with a per-episode std of ~3.5 and a heavy left tail, so its percentage form divides by a near-zero denominator and is extremely noisy (see the `+-SEM` and `sigma` columns). `goal RMSE deg %` — RMS distance to assigned goal, i.e. formation error — has a true zero and a few-percent SEM, and is the axis that actually ranks severities.

| config | f | **paired honest deg %** | reward | reward deg % | ±SEM | σ | goal RMSE | goal RMSE deg % | honest goal RMSE deg % | collisions Δ | action dev | saturation | min sep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `dropout` p0.05_hold | 0.125 | 0.4 (3 seeds) | 0.43 | -3.7 | 215.0 | 0.3 | 1.120 | 1.6 | 1.2 | 0.16 | 0.001 | 0.0% | 0.345 |
| `dropout` p0.15_hold | 0.125 | 1.2 (3 seeds) | 0.48 | 3.8 | 212.6 | 0.2 | 1.111 | 0.8 | 2.0 | 0.17 | 0.003 | 0.0% | 0.344 |
| `dropout` p0.15_zero | 0.125 | 4.0 (3 seeds) | 0.12 | 99.1 | 220.0 | 0.7 | 1.164 | 5.5 | 4.8 | 0.25 | 0.048 | 0.0% | 0.346 |
| `dropout` p0.3_hold | 0.125 | -1.1 (3 seeds) | 0.63 | -65.8 | 215.5 | 0.0 | 1.089 | -1.1 | -0.4 | 0.04 | 0.008 | 0.0% | 0.345 |
| `dropout` p0.05_hold | 0.250 | -0.3 (3 seeds) | 0.57 | -24.3 | 216.1 | 0.1 | 1.108 | 0.5 | 0.4 | 0.07 | 0.001 | 0.0% | 0.344 |
| `dropout` p0.15_hold | 0.250 | -0.3 (3 seeds) | 0.57 | 5.0 | 216.9 | 0.1 | 1.101 | -0.2 | 0.4 | 0.11 | 0.003 | 0.0% | 0.345 |
| `dropout` p0.15_zero | 0.250 | -0.2 (3 seeds) | 0.39 | 31.4 | 217.9 | 0.3 | 1.127 | 2.2 | 0.5 | 0.15 | 0.047 | 0.0% | 0.347 |
| `dropout` p0.3_hold | 0.250 | 0.7 (3 seeds) | 0.46 | 4.2 | 225.7 | 0.3 | 1.122 | 1.8 | 1.4 | 0.11 | 0.008 | 0.0% | 0.345 |
| `gaussian` sigma0.1 | 0.125 | -0.4 (3 seeds) | 0.46 | 21.7 | 218.6 | 0.2 | 1.109 | 0.6 | 0.3 | 0.10 | 0.188 | 0.0% | 0.345 |
| `gaussian` sigma0.25 | 0.125 | 6.6 (3 seeds) | -1.26 | 549.8 | 246.3 | 2.2 | 1.341 | 21.8 | 7.4 | 0.51 | 0.467 | 0.8% | 0.342 |
| `gaussian` sigma0.5 | 0.125 | 1.0 (3 seeds) | -3.09 | 1176.9 | 225.5 | 4.5 | 1.685 | 52.9 | 1.7 | -0.00 | 0.839 | 13.9% | 0.359 |
| `gaussian` sigma0.1 | 0.250 | 0.5 (3 seeds) | 0.26 | 63.1 | 215.1 | 0.5 | 1.131 | 2.6 | 1.2 | 0.11 | 0.188 | 0.0% | 0.342 |
| `gaussian` sigma0.25 | 0.250 | 5.3 (3 seeds) | -2.06 | 796.0 | 239.4 | 3.0 | 1.425 | 29.3 | 6.1 | 0.51 | 0.467 | 0.8% | 0.345 |
| `gaussian` sigma0.5 | 0.250 | -1.8 (3 seeds) | -6.60 | 2446.7 | 247.0 | 8.6 | 2.098 | 90.2 | -1.2 | -0.03 | 0.838 | 14.1% | 0.370 |
| `ou` sigmaou0.1 | 0.125 | -0.3 (3 seeds) | -3.24 | 1248.2 | 221.8 | 5.1 | 1.717 | 55.8 | 0.5 | 0.06 | 0.355 | 0.4% | 0.360 |
| `ou` sigmaou0.2 | 0.125 | -0.3 (3 seeds) | -3.60 | 1376.2 | 223.8 | 5.4 | 1.648 | 49.5 | 0.5 | 0.01 | 0.677 | 6.8% | 0.358 |
| `ou` sigmaou0.4 | 0.125 | -2.0 (3 seeds) | -4.22 | 1552.0 | 227.9 | 6.2 | 1.602 | 45.4 | -1.3 | -0.06 | 1.157 | 24.3% | 0.360 |
| `ou` sigmaou0.1 | 0.250 | -2.2 (3 seeds) | -7.05 | 2560.8 | 236.4 | 9.5 | 2.167 | 96.6 | -1.5 | -0.04 | 0.356 | 0.4% | 0.371 |
| `ou` sigmaou0.2 | 0.250 | -2.4 (3 seeds) | -7.59 | 2655.1 | 224.2 | 10.1 | 2.029 | 84.0 | -1.8 | -0.01 | 0.678 | 6.8% | 0.371 |
| `ou` sigmaou0.4 | 0.250 | 2.4 (3 seeds) | -9.30 | 3157.6 | 230.2 | 12.0 | 1.999 | 81.3 | 3.2 | 0.09 | 1.159 | 24.4% | 0.370 |
| `stall` d0.10_ms0.1_hold **†** | 0.125 | 1.5 (3 seeds) | 0.18 | 145.0 | 226.6 | 0.6 | 1.151 | 4.4 | 2.3 | 0.17 | 0.016 | 0.0% | 0.343 |
| `stall` d0.10_ms0.3_hold **†** | 0.125 | 2.1 (3 seeds) | -0.14 | 224.9 | 219.4 | 1.0 | 1.198 | 8.7 | 2.9 | 0.21 | 0.034 | 0.0% | 0.344 |
| `stall` d0.10_ms1.0_hold **†** | 0.125 | 1.4 (3 seeds) | -0.13 | 193.7 | 228.0 | 1.0 | 1.208 | 9.6 | 2.2 | 0.13 | 0.053 | 0.0% | 0.346 |
| `stall` d0.25_ms0.1_hold **†** | 0.125 | 1.1 (3 seeds) | -0.12 | 233.5 | 226.0 | 1.0 | 1.184 | 7.4 | 1.9 | 0.26 | 0.043 | 0.0% | 0.343 |
| `stall` d0.25_ms0.3_hold **†** | 0.125 | 3.7 (3 seeds) | -0.99 | 481.1 | 210.6 | 2.1 | 1.336 | 21.3 | 4.4 | 0.29 | 0.086 | 0.0% | 0.347 |
| `stall` d0.25_ms1.0_hold **†** | 0.125 | 2.0 (3 seeds) | -1.05 | 434.4 | 218.9 | 2.1 | 1.354 | 23.0 | 2.8 | 0.25 | 0.124 | 0.0% | 0.346 |
| `stall` d0.50_ms0.1_hold **†** | 0.125 | 4.4 (3 seeds) | -1.09 | 513.3 | 224.4 | 2.2 | 1.318 | 19.5 | 5.1 | 0.47 | 0.091 | 0.0% | 0.347 |
| `stall` d0.50_ms0.3_hold **†** | 0.125 | 4.7 (3 seeds) | -2.22 | 935.2 | 227.3 | 3.6 | 1.522 | 38.0 | 5.5 | 0.25 | 0.155 | 0.0% | 0.351 |
| `stall` d0.50_ms1.0_hold **†** | 0.125 | 2.1 (3 seeds) | -2.40 | 957.9 | 242.1 | 3.8 | 1.575 | 42.9 | 2.8 | 0.16 | 0.233 | 0.0% | 0.351 |

**paired honest deg %** is the only reportable degradation in the v2 corpus (per-episode paired against the clean run at the same seed; design 6.18). The unpaired columns are kept for comparison with the legacy tables.

† = matched-triple candidate: family validated below, f = 0.125. (Replaces the retired 20–40 % band ★; decision D1.)


## Family-level validation (decision D1, replaces the 20–40 % band)

A family is validated when its runs together significantly damage the honest team (one-sample t-test on per-run **paired** honest goal-RMSE degradation, mean > 0, p < 0.05). This replaces the 20–40 % band, which was set on legacy numbers and which no v2 configuration reaches.

| family | runs | mean | positive | family p | status |
|---|---|---|---|---|---|
| dropout | 24 | +0.5 % | 10/24 | 0.5064 | no family-level effect |
| gaussian | 18 | +1.9 % | 10/18 | 0.1266 | no family-level effect |
| ou | 18 | -0.8 % | 6/18 | 0.4032 | no family-level effect |
| stall | 27 | +2.6 % | 21/27 | 0.0004 | VALIDATED -- damaging |


## Saturation check

The spec flags >~30 % clip saturation as disqualifying for a clean "unbiased noise" control.

| config | f | saturation rate |
|---|---|---|
| `dropout` p0.05_hold | 0.125 | 0.0% |
| `dropout` p0.15_hold | 0.125 | 0.0% |
| `dropout` p0.15_zero | 0.125 | 0.0% |
| `dropout` p0.3_hold | 0.125 | 0.0% |
| `dropout` p0.05_hold | 0.250 | 0.0% |
| `dropout` p0.15_hold | 0.250 | 0.0% |
| `dropout` p0.15_zero | 0.250 | 0.0% |
| `dropout` p0.3_hold | 0.250 | 0.0% |
| `gaussian` sigma0.1 | 0.125 | 0.0% |
| `gaussian` sigma0.25 | 0.125 | 0.8% |
| `gaussian` sigma0.5 | 0.125 | 13.9% |
| `gaussian` sigma0.1 | 0.250 | 0.0% |
| `gaussian` sigma0.25 | 0.250 | 0.8% |
| `gaussian` sigma0.5 | 0.250 | 14.1% |
| `ou` sigmaou0.1 | 0.125 | 0.4% |
| `ou` sigmaou0.2 | 0.125 | 6.8% |
| `ou` sigmaou0.4 | 0.125 | 24.3% |
| `ou` sigmaou0.1 | 0.250 | 0.4% |
| `ou` sigmaou0.2 | 0.250 | 6.8% |
| `ou` sigmaou0.4 | 0.250 | 24.4% |
| `stall` d0.10_ms0.1_hold | 0.125 | 0.0% |
| `stall` d0.10_ms0.3_hold | 0.125 | 0.0% |
| `stall` d0.10_ms1.0_hold | 0.125 | 0.0% |
| `stall` d0.25_ms0.1_hold | 0.125 | 0.0% |
| `stall` d0.25_ms0.3_hold | 0.125 | 0.0% |
| `stall` d0.25_ms1.0_hold | 0.125 | 0.0% |
| `stall` d0.50_ms0.1_hold | 0.125 | 0.0% |
| `stall` d0.50_ms0.3_hold | 0.125 | 0.0% |
| `stall` d0.50_ms1.0_hold | 0.125 | 0.0% |
