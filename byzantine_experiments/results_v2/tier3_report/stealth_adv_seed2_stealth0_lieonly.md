# Stealth analysis -- `adv_seed2_stealth0_lieonly` (spec item 6)

Own-seed confirmatory run, n = 200 episodes, paired clean run at seed 2. Broadcasts **recorded**. Thresholds set on the clean run only.

## Action level

| statistic | value |
|---|---|
| action-norm ratio adversary / honest | 1.052 |
| AUC, episode-mean action norm, adversary vs honest | 0.547 |
| control: same slot in the CLEAN run -- ratio / AUC | 1.008 / 0.507 |
| naive action detector (> clean p99 = 0.467): adversary episodes flagged | 0.5 % |
| same detector, honest drones flagged | 1.4 % |

## Broadcast physics (naive plausibility checks)

Thresholds (max over every honest drone-step of the clean run): speed 15.33 m/s, acceleration 1531.2 m/s², kinematic residual 0.1271 m; bounds = the 10 x 10 x 10 m room.

| check | adversary steps failing | adversary episodes flagged | honest episodes flagged |
|---|---|---|---|
| bounds | 100.0 % | 100.0 % | 0.1 % |
| speed | 0.1 % | 18.0 % | 0.1 % |
| consistency | 16.3 % | 100.0 % | 0.1 % |
| accel | 0.0 % | 29.0 % | 0.1 % |
| **any check** | | **100.0 %** | 0.1 % |

Broadcast below the floor: 85.7 % of steps. Broadcast speed mean 10.06 m/s (true 1.03). Offset |claim − truth| mean 17.01 m, p99 17.32 m.

**Reading.** An action-level detector sees this adversary iff the AUC is well away from 0.5 *and* from the same-slot clean control. A broadcast check catches it iff "any check" flags its episodes at a rate well above the honest false-alarm rate. The spec asks to say so plainly if a naive detector catches the stealthy adversary.

![](stealth_adv_seed2_stealth0_lieonly.png)
