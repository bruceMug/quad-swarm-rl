# Phase 3 -- damage per member under distribution shift

Honest goal-RMSE degradation, each run paired with the clean run at the same seed and shift level, pooled over seeds. The first rows give the shift's own effect on the clean swarm (clean at that level vs clean with no shift, same scenarios).

The cm column is the same comparison in absolute terms (added honest goal error). The % column is relative to the clean error at that level, which grows with the shift.

| condition | shift | pooled damage % [95 % CI] | added error, cm [95 % CI] | episodes | runs |
|---|---|---|---|---|---|
| clean swarm (shift effect) | mild | +10.3 [+4.7, +15.9] | +11.2 [+5.1, +17.3] | 150 | 3 |
| clean swarm (shift effect) | mild_v1 | +32.5 [+24.1, +40.9] | +35.5 [+26.3, +44.7] | 150 | 3 |
| clean swarm (shift effect) | severe | +66.2 [+54.9, +77.4] | +72.1 [+59.7, +84.4] | 150 | 3 |
| clean swarm (shift effect) | severe_v1 | +147.2 [+133.6, +160.8] | +160.2 [+145.4, +175.0] | 150 | 3 |
| t1_stall_d0.50_ms0.1_hold_f0125 | none | +7.2 [+2.3, +12.2] | +7.9 [+2.4, +13.3] | 150 | 3 |
| t1_stall_d0.50_ms0.1_hold_f0125 | mild | +0.3 [-3.7, +4.3] | +0.4 [-4.5, +5.2] | 150 | 3 |
| t1_stall_d0.50_ms0.1_hold_f0125 | mild_v1 | +0.8 [-4.0, +5.6] | +1.1 [-5.8, +8.1] | 150 | 3 |
| t1_stall_d0.50_ms0.1_hold_f0125 | severe | +0.4 [-4.2, +5.1] | +0.6 [-8.1, +9.3] | 150 | 3 |
| t1_stall_d0.50_ms0.1_hold_f0125 | severe_v1 | -1.3 [-4.4, +1.9] | -3.3 [-11.8, +5.2] | 150 | 3 |
| t2_liar_mirror_f0125 | none | +6.4 [+2.3, +10.6] | +7.0 [+2.5, +11.6] | 150 | 3 |
| t2_liar_mirror_f0125 | mild | +1.7 [-2.3, +5.8] | +2.1 [-2.7, +7.0] | 150 | 3 |
| t2_liar_mirror_f0125 | mild_v1 | +3.8 [-1.2, +8.7] | +5.6 [-1.8, +13.0] | 150 | 3 |
| t2_liar_mirror_f0125 | severe | -0.7 [-5.2, +3.8] | -1.4 [-9.6, +6.9] | 150 | 3 |
| t2_liar_mirror_f0125 | severe_v1 | -1.1 [-4.3, +2.0] | -3.2 [-11.7, +5.2] | 150 | 3 |
| t3 (cross-seed) | none | +5.6 [+2.5, +8.8] | +6.1 [+2.6, +9.6] | 300 | 6 |
| t3 (cross-seed) | mild | +3.5 [+0.5, +6.5] | +4.2 [+0.6, +7.8] | 300 | 6 |
| t3 (cross-seed) | mild_v1 | +2.1 [-1.1, +5.2] | +3.2 [-1.3, +7.8] | 300 | 6 |
| t3 (cross-seed) | severe | -2.2 [-5.3, +0.9] | -3.8 [-9.5, +1.8] | 300 | 6 |
| t3 (cross-seed) | severe_v1 | -0.7 [-2.9, +1.4] | -2.0 [-7.8, +3.8] | 300 | 6 |
| t3 (own seed) | none | +5.4 [+0.9, +9.8] | +5.8 [+1.0, +10.6] | 150 | 3 |
| t3 (own seed) | mild | +2.9 [-1.7, +7.6] | +3.5 [-2.1, +9.0] | 150 | 3 |
| t3 (own seed) | mild_v1 | +1.8 [-3.2, +6.8] | +2.9 [-4.3, +10.2] | 150 | 3 |
| t3 (own seed) | severe | -1.1 [-5.1, +2.9] | -2.3 [-9.6, +5.0] | 150 | 3 |
| t3 (own seed) | severe_v1 | -1.9 [-5.0, +1.3] | -5.5 [-14.1, +3.2] | 150 | 3 |
