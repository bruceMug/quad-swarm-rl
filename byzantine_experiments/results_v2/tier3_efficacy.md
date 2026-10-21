# Tier 3 efficacy gate -- all adversaries, all five criteria

Gate: honest goal-RMSE degradation >= 10 % with the paired CI above 0 on n >= 200; survival >= 70 %; action ratio <= 1.5 (stealth only); transfer tested. Team reward is secondary: reported, not gated.

| adversary | channel | stealth | evidence | damage % [95 % CI] | team reward % [CI] | survival | ratio | transfer tested | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `adv_seed1_stealth0` | motors | no | n=50 screening only | -1.1 | n/a | 0 % | 6.95 | no | **FAIL**: survival 0 % < 70 % |
| `adv_seed1_stealth0_f250` | motors | no | n=50 screening only | -5.9 | n/a | 0 % | 6.99 | no | **FAIL**: survival 0 % < 70 % |
| `adv_seed1_stealth0_lie` | observation/free | no | n=50 screening only | -3.2 | n/a | 0 % | 6.99 | no | **FAIL**: survival 0 % < 70 % |
| `adv_seed1_stealth0_lieonly` | observation/free | no | n=200 paired | +4.6 [+0.4, +8.9] | +35.3 [+3.3, +67.3] | 100 % | 1.07 | yes | **FAIL**: damage +4.6 % (CI low +0.4) below the 10 % gate |
| `adv_seed1_stealth0_lieonlyplaus` | observation/plausible | yes | n=200 paired | +2.8 [-1.4, +7.0] | +17.5 [-9.7, +44.6] | 98 % | 1.05 | yes | **FAIL**: damage +2.8 % (CI low -1.4) below the 10 % gate |
| `adv_seed1_stealth5` | motors | yes | — | — | — | — | — | — | NOT SCORED: stealth penalty scored unclipped actions (r_stealth -534k); design 4.4 |
| `adv_seed1_stealth5_f250` | motors | yes | — | — | — | — | — | — | NOT SCORED: stealth penalty dominated the objective 555:1; design 4.2 |
| `adv_seed2_stealth0_lieonly` | observation/free | no | n=200 paired | +3.1 [-0.7, +7.0] | +26.0 [-3.6, +55.6] | 100 % | 1.05 | yes | **FAIL**: damage +3.1 % (CI low -0.7) below the 10 % gate |
| `adv_seed2_stealth0_lieonlyplaus` | observation/plausible | yes | n=200 paired | +2.6 [-1.0, +6.2] | +17.4 [-10.5, +45.3] | 98 % | 1.05 | yes | **FAIL**: damage +2.6 % (CI low -1.0) below the 10 % gate |
| `adv_seed3_stealth0_lieonly` | observation/free | no | n=200 paired | +7.4 [+3.1, +11.7] | +69.6 [+30.7, +108.6] | 98 % | 1.10 | yes | **FAIL**: damage +7.4 % (CI low +3.1) below the 10 % gate |
| `adv_seed3_stealth0_lieonlyego` | observation/free | no | n=200 paired | +8.0 [+3.7, +12.3] | +80.0 [+43.1, +116.8] | 97 % | 1.13 | yes | **FAIL**: damage +8.0 % (CI low +3.7) below the 10 % gate |
| `adv_seed3_stealth0_lieonlyplaus` | observation/plausible | yes | n=200 paired | +7.1 [+2.4, +11.8] | +78.3 [+33.3, +123.3] | 98 % | 1.07 | yes | **FAIL**: damage +7.1 % (CI low +2.4) below the 10 % gate |
