# Tier 3 compute accounting (spec item 8)

One Tesla T4 (sequential or shared), 32 CPU cores. **Active** hours exclude pauses longer than 15 min between logged episodes (stops, resumes); **span** is first to last logged episode. GPU-hours = active hours x 1 GPU; runs that overlapped in time shared the device, so their sum is an upper bound on device time -- the union of spans is the device wall clock.

| adversary | env steps (M) | active h (GPU-h) | span h | M steps / active h |
|---|---|---|---|---|
| `adv_seed1_stealth0` | 200 | 31.8 | 31.8 | 6.28 |
| `adv_seed1_stealth0_f250` | 200 | 15.7 | 15.7 | 12.75 |
| `adv_seed1_stealth0_lie` | 200 | 35.2 | 35.2 | 5.69 |
| `adv_seed1_stealth0_lieonly` | 200 | 37.1 | 37.1 | 5.39 |
| `adv_seed1_stealth0_lieonlyplaus` | 200 | 48.9 | 48.9 | 4.09 |
| `adv_seed1_stealth5` | 200 | 32.0 | 32.0 | 6.24 |
| `adv_seed1_stealth5_f250` | 200 | 15.9 | 15.9 | 12.57 |
| `adv_seed2_stealth0_lieonly` | 200 | 49.2 | 49.2 | 4.07 |
| `adv_seed2_stealth0_lieonlyplaus` | 200 | 50.2 | 50.2 | 3.98 |
| `adv_seed3_stealth0_lieonly` | 200 | 57.2 | 57.2 | 3.50 |
| `adv_seed3_stealth0_lieonlyego` | 200 | 36.7 | 36.7 | 5.45 |
| `adv_seed3_stealth0_lieonlyplaus` | 200 | 44.8 | 44.8 | 4.46 |
| **total** | 2400 | **455** | device wall clock (union of spans) 455 h | |

Evaluation (rollouts, CPU-bound): 18,500 episodes of Tier 3 runs and their clean and hand-designed references collected in `byzantine_experiments/results_v2` (sweeps, confirmatory, transfer, triple). At the measured ~20 episodes/min this is ~15 h of node time -- an estimate from throughput, not a measurement.
