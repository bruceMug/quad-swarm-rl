# Tier 1 — severity calibration table

Generated from `byzantine_experiments/results/tier1/*.json`. 120 rollout runs over 3 baseline seeds, 50 episodes each.

Each Byzantine config is compared against the clean rollout of the **same** baseline checkpoint, collected with the same collector. Values are means over seeds; `+-` is the spread across seeds.

> **Read the two degradation columns together.** `reward deg %` is the figure the spec asks for, but mean episode reward is ~1.5 with a per-episode std of ~3.5 and a heavy left tail, so its percentage form divides by a near-zero denominator and is extremely noisy (see the `+-SEM` and `sigma` columns). `goal RMSE deg %` — RMS distance to assigned goal, i.e. formation error — has a true zero and a few-percent SEM, and is the axis that actually ranks severities.

| config | f | reward | reward deg % | ±SEM | σ | goal RMSE | goal RMSE deg % | honest goal RMSE deg % | collisions Δ | action dev | saturation | min sep |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `dropout` p0.05_hold **★** | 0.125 | 0.57 | 35.2 | 67.3 | 0.4 | 1.117 | 5.8 | 6.4 | 0.22 | 0.001 | 0.0% | 0.354 |
| `dropout` p0.15_hold | 0.125 | 0.49 | 43.7 | 65.7 | 0.5 | 1.120 | 6.2 | 5.0 | 0.12 | 0.003 | 0.0% | 0.348 |
| `dropout` p0.15_zero | 0.125 | 1.10 | -54.9 | 65.8 | -0.4 | 1.040 | -1.3 | -1.2 | 0.08 | 0.045 | 0.0% | 0.362 |
| `dropout` p0.3_hold | 0.125 | 1.05 | -94.1 | 60.2 | -0.3 | 1.051 | 0.0 | 0.4 | 0.09 | 0.008 | 0.0% | 0.370 |
| `dropout` p0.5_zero | 0.125 | -0.24 | 138.1 | 72.6 | 1.4 | 1.241 | 17.9 | 3.1 | 0.15 | 0.288 | 0.0% | 0.348 |
| `dropout` p0.6_hold | 0.125 | 1.09 | -47.7 | 64.8 | -0.3 | 1.061 | 0.6 | 0.4 | 0.14 | 0.029 | 0.0% | 0.364 |
| `dropout` p0.85_hold | 0.125 | -0.33 | 182.0 | 67.5 | 1.8 | 1.229 | 16.4 | 0.2 | 0.21 | 0.127 | 0.0% | 0.361 |
| `dropout` p0.85_zero | 0.125 | -2.96 | 540.2 | 71.3 | 5.2 | 1.671 | 58.4 | 11.9 | 0.09 | 0.821 | 0.0% | 0.361 |
| `dropout` p0.98_hold | 0.125 | -3.36 | 557.9 | 63.2 | 6.1 | 1.741 | 65.4 | 5.0 | 0.09 | 0.341 | 0.0% | 0.373 |
| `dropout` p0.05_hold | 0.250 | 0.99 | -59.9 | 64.2 | -0.2 | 1.075 | 2.1 | 2.9 | 0.25 | 0.001 | 0.0% | 0.373 |
| `dropout` p0.15_hold | 0.250 | 0.58 | -1.8 | 65.7 | 0.4 | 1.136 | 8.0 | 8.1 | 0.11 | 0.003 | 0.0% | 0.385 |
| `dropout` p0.15_zero **★** | 0.250 | 0.32 | 26.2 | 68.0 | 0.7 | 1.135 | 7.9 | 3.1 | 0.15 | 0.047 | 0.0% | 0.372 |
| `dropout` p0.3_hold | 0.250 | 1.00 | -89.5 | 60.7 | -0.3 | 1.056 | 0.4 | 0.1 | 0.11 | 0.008 | 0.0% | 0.355 |
| `gaussian` sigma0.1 | 0.125 | 0.82 | -7.5 | 64.8 | 0.1 | 1.062 | 0.7 | 0.5 | 0.17 | 0.188 | 0.0% | 0.341 |
| `gaussian` sigma0.25 | 0.125 | -0.95 | 213.9 | 73.6 | 2.4 | 1.301 | 23.6 | 8.6 | 0.37 | 0.467 | 0.8% | 0.364 |
| `gaussian` sigma0.30 | 0.125 | -1.62 | 335.6 | 77.3 | 3.1 | 1.424 | 35.2 | 12.5 | 0.32 | 0.553 | 2.6% | 0.369 |
| `gaussian` sigma0.35 | 0.125 | -1.91 | 436.1 | 80.7 | 3.7 | 1.487 | 40.7 | 3.3 | 0.17 | 0.632 | 5.3% | 0.358 |
| `gaussian` sigma0.40 | 0.125 | -1.50 | 313.5 | 62.9 | 3.5 | 1.455 | 38.1 | -4.6 | 0.08 | 0.707 | 8.1% | 0.359 |
| `gaussian` sigma0.5 | 0.125 | -2.42 | 491.2 | 69.9 | 4.6 | 1.607 | 52.1 | 0.0 | -0.02 | 0.839 | 13.9% | 0.369 |
| `gaussian` sigma0.1 | 0.250 | 0.69 | -2.6 | 72.3 | 0.3 | 1.096 | 4.0 | 4.1 | 0.17 | 0.188 | 0.0% | 0.359 |
| `gaussian` sigma0.25 | 0.250 | -1.18 | 255.9 | 75.9 | 2.6 | 1.299 | 23.5 | 0.5 | 0.37 | 0.467 | 0.7% | 0.355 |
| `gaussian` sigma0.5 | 0.250 | -6.07 | 976.9 | 67.9 | 9.5 | 2.016 | 91.3 | 0.4 | 0.08 | 0.840 | 13.8% | 0.369 |
| `ou` sigmaou0.04 | 0.125 | 0.12 | 58.7 | 65.8 | 1.0 | 1.175 | 11.7 | 1.5 | 0.36 | 0.142 | 0.0% | 0.359 |
| `ou` sigmaou0.06 | 0.125 | -1.55 | 380.7 | 70.0 | 3.4 | 1.429 | 35.3 | 1.8 | 0.15 | 0.213 | 0.0% | 0.361 |
| `ou` sigmaou0.08 | 0.125 | -2.71 | 508.6 | 67.0 | 5.3 | 1.622 | 53.7 | 4.0 | 0.09 | 0.284 | 0.1% | 0.358 |
| `ou` sigmaou0.1 | 0.125 | -2.71 | 490.3 | 65.1 | 5.2 | 1.657 | 57.2 | -0.0 | 0.03 | 0.355 | 0.4% | 0.370 |
| `ou` sigmaou0.2 | 0.125 | -3.89 | 663.3 | 73.4 | 6.5 | 1.695 | 60.9 | 9.7 | 0.07 | 0.678 | 6.7% | 0.373 |
| `ou` sigmaou0.4 | 0.125 | -3.90 | 625.9 | 64.3 | 6.7 | 1.553 | 47.5 | 0.8 | 0.01 | 1.160 | 24.1% | 0.358 |
| `ou` sigmaou0.1 | 0.250 | -6.64 | 1098.2 | 76.8 | 10.0 | 2.107 | 99.7 | -0.5 | -0.03 | 0.356 | 0.4% | 0.387 |
| `ou` sigmaou0.2 | 0.250 | -7.37 | 1172.1 | 72.9 | 11.0 | 2.005 | 90.2 | 0.9 | 0.10 | 0.679 | 6.8% | 0.392 |
| `ou` sigmaou0.4 | 0.250 | -8.64 | 1350.0 | 75.8 | 12.6 | 1.909 | 81.1 | 0.3 | 0.01 | 1.161 | 24.2% | 0.379 |
| `stall` d0.10_ms0.1_hold | 0.125 | 0.29 | 71.3 | 70.4 | 0.8 | 1.146 | 8.6 | 7.3 | 0.27 | 0.016 | 0.0% | 0.352 |
| `stall` d0.10_ms0.3_hold | 0.125 | -0.23 | 150.1 | 67.1 | 1.6 | 1.222 | 15.9 | 8.9 | 0.33 | 0.036 | 0.0% | 0.371 |
| `stall` d0.10_ms1.0_hold | 0.125 | -0.01 | 65.7 | 67.2 | 1.1 | 1.200 | 14.1 | 5.5 | 0.19 | 0.058 | 0.0% | 0.347 |
| `stall` d0.25_ms0.1_hold **★** | 0.125 | 0.32 | 28.4 | 64.4 | 0.8 | 1.149 | 9.2 | 1.9 | 0.31 | 0.045 | 0.0% | 0.373 |
| `stall` d0.25_ms0.3_hold | 0.125 | -0.90 | 267.9 | 69.2 | 2.4 | 1.325 | 25.5 | 9.1 | 0.31 | 0.089 | 0.0% | 0.351 |
| `stall` d0.25_ms1.0_hold | 0.125 | -0.46 | 135.2 | 69.8 | 1.9 | 1.276 | 21.3 | 3.4 | 0.19 | 0.127 | 0.0% | 0.356 |
| `stall` d0.50_ms0.1_hold | 0.125 | -0.70 | 213.9 | 68.7 | 2.2 | 1.272 | 20.6 | 7.4 | 0.41 | 0.095 | 0.0% | 0.348 |
| `stall` d0.50_ms0.3_hold | 0.125 | -1.63 | 324.4 | 69.4 | 3.4 | 1.432 | 36.1 | 2.4 | 0.19 | 0.165 | 0.0% | 0.357 |
| `stall` d0.50_ms1.0_hold | 0.125 | -2.56 | 496.1 | 72.1 | 4.6 | 1.587 | 50.6 | 9.2 | 0.20 | 0.236 | 0.0% | 0.363 |

★ = mean reward degradation falls in the 20–40 % band.


## Configs in the 20–40 % band by **reward** degradation

- `dropout` p0.05_hold at f=0.125 — 35.2 %
- `dropout` p0.15_zero at f=0.250 — 26.2 %
- `stall` d0.25_ms0.1_hold at f=0.125 — 28.4 %


## Configs in the 20–40 % band by **goal RMSE** degradation

- `gaussian` sigma0.25 at f=0.125 — 23.6 %
- `gaussian` sigma0.30 at f=0.125 — 35.2 %
- `gaussian` sigma0.40 at f=0.125 — 38.1 %
- `gaussian` sigma0.25 at f=0.250 — 23.5 %
- `ou` sigmaou0.06 at f=0.125 — 35.3 %
- `stall` d0.25_ms0.3_hold at f=0.125 — 25.5 %
- `stall` d0.25_ms1.0_hold at f=0.125 — 21.3 %
- `stall` d0.50_ms0.1_hold at f=0.125 — 20.6 %
- `stall` d0.50_ms0.3_hold at f=0.125 — 36.1 %


## Saturation check

The spec flags >~30 % clip saturation as disqualifying for a clean "unbiased noise" control.

| config | f | saturation rate |
|---|---|---|
| `dropout` p0.05_hold | 0.125 | 0.0% |
| `dropout` p0.15_hold | 0.125 | 0.0% |
| `dropout` p0.15_zero | 0.125 | 0.0% |
| `dropout` p0.3_hold | 0.125 | 0.0% |
| `dropout` p0.5_zero | 0.125 | 0.0% |
| `dropout` p0.6_hold | 0.125 | 0.0% |
| `dropout` p0.85_hold | 0.125 | 0.0% |
| `dropout` p0.85_zero | 0.125 | 0.0% |
| `dropout` p0.98_hold | 0.125 | 0.0% |
| `dropout` p0.05_hold | 0.250 | 0.0% |
| `dropout` p0.15_hold | 0.250 | 0.0% |
| `dropout` p0.15_zero | 0.250 | 0.0% |
| `dropout` p0.3_hold | 0.250 | 0.0% |
| `gaussian` sigma0.1 | 0.125 | 0.0% |
| `gaussian` sigma0.25 | 0.125 | 0.8% |
| `gaussian` sigma0.30 | 0.125 | 2.6% |
| `gaussian` sigma0.35 | 0.125 | 5.3% |
| `gaussian` sigma0.40 | 0.125 | 8.1% |
| `gaussian` sigma0.5 | 0.125 | 13.9% |
| `gaussian` sigma0.1 | 0.250 | 0.0% |
| `gaussian` sigma0.25 | 0.250 | 0.7% |
| `gaussian` sigma0.5 | 0.250 | 13.8% |
| `ou` sigmaou0.04 | 0.125 | 0.0% |
| `ou` sigmaou0.06 | 0.125 | 0.0% |
| `ou` sigmaou0.08 | 0.125 | 0.1% |
| `ou` sigmaou0.1 | 0.125 | 0.4% |
| `ou` sigmaou0.2 | 0.125 | 6.7% |
| `ou` sigmaou0.4 | 0.125 | 24.1% |
| `ou` sigmaou0.1 | 0.250 | 0.4% |
| `ou` sigmaou0.2 | 0.250 | 6.8% |
| `ou` sigmaou0.4 | 0.250 | 24.2% |
| `stall` d0.10_ms0.1_hold | 0.125 | 0.0% |
| `stall` d0.10_ms0.3_hold | 0.125 | 0.0% |
| `stall` d0.10_ms1.0_hold | 0.125 | 0.0% |
| `stall` d0.25_ms0.1_hold | 0.125 | 0.0% |
| `stall` d0.25_ms0.3_hold | 0.125 | 0.0% |
| `stall` d0.25_ms1.0_hold | 0.125 | 0.0% |
| `stall` d0.50_ms0.1_hold | 0.125 | 0.0% |
| `stall` d0.50_ms0.3_hold | 0.125 | 0.0% |
| `stall` d0.50_ms1.0_hold | 0.125 | 0.0% |
