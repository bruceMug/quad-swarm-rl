# What limits the attack: the action channel

Honest-drone goal-RMSE degradation vs. the clean run on the **same drone indices**, with a Welch t-test over episodes. "signif worse/better" counts runs with p < 0.05. Efficacy gate is ≥ 10 %.

| tier | channel | f | runs | max deg % | median | **signif worse** | signif better |
|---|---|---|---|---|---|---|---|
| 2 (hand-designed) | motors only | 0.125 | 15 | **6.0 %** | 0.4 % | **0** | 0 |
| 2 (hand-designed) | motors only | 0.250 | 24 | **4.5 %** | -2.9 % | **0** | 0 |
| 2 (hand-designed) | observation channel | 0.125 | 9 | **11.4 %** | 5.1 % | **1** | 0 |
| 2 (hand-designed) | observation channel | 0.250 | 9 | **13.1 %** | 7.6 % | **2** | 0 |
| 3 (**trained**) | observation channel | 0.125 | 28 | **14.2 %** | 4.3 % | **7** | 0 |
| 3 (**trained**) | motors only | 0.125 | 7 | **2.3 %** | -0.8 % | **0** | 0 |
| 3 (**trained**) | motors only | 0.250 | 7 | **-1.8 %** | -6.4 % | **0** | 0 |

## What this data shows

_Every statement below is computed from the rows above; none is fixed text. Interpretation belongs in design/tier3_design.md, where it can cite which data root it was drawn from._

* **Hand-designed, observation channel (Tier 2 liar)** (18 runs): mean +5.7 %, positive in 16/18; significantly worse for the honest team in 3, significantly better in 0; best +13.1 % (`t2_liar_s3_mirror_f025`, p = 0.0181); 3 significant and at or above the 10 % gate.
* **Hand-designed, motors only (Tier 2 signflip / waypoint / collusion)** (39 runs): mean -1.5 %, positive in 14/39; significantly worse for the honest team in 0, significantly better in 0; best +6.0 % (`t2_waypoint_s1_a0.3_f0125`, p = 0.1236); 0 significant and at or above the 10 % gate.
* **Trained, observation channel (Tier 3 rounds 3-5)** (28 runs): mean +3.8 %, positive in 19/28; significantly worse for the honest team in 7, significantly better in 0; best +14.2 % (`t3_adv_seed3_stealth0_lieonly_on_s3_m100M`, p = 0.0189); 5 significant and at or above the 10 % gate.
* **Trained, motors only (Tier 3 rounds 1-2)** (14 runs): mean -3.5 %, positive in 2/14; significantly worse for the honest team in 0, significantly better in 0; best +2.3 % (`t3_adv_seed1_stealth0_on_s1_m006M`, p = 0.5592); 0 significant and at or above the 10 % gate.
* f = 0.125: 59 runs, mean +2.4 %.
* f = 0.250: 40 runs, mean -1.0 %.

## The same data, split by baseline seed

| channel | baseline seed | runs | max deg % | signif worse | clean goal RMSE |
|---|---|---|---|---|---|
| motors only | 1 | 13 | **6.0 %** | 0 | 1.1576 |
| motors only | 2 | 13 | **1.8 %** | 0 | 1.0852 |
| motors only | 3 | 13 | **4.5 %** | 0 | 1.0670 |
| observation channel | 1 | 6 | **7.5 %** | 0 | 1.1576 |
| observation channel | 2 | 6 | **8.6 %** | 0 | 1.0852 |
| observation channel | 3 | 6 | **13.1 %** | 3 | 1.0670 |
| observation channel (**trained**) | 1 | 14 | **6.4 %** | 0 | 1.1576 |
| observation channel (**trained**) | 3 | 14 | **14.2 %** | 7 | 1.0670 |