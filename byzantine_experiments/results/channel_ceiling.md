# What limits the attack: the action channel

Honest-drone goal-RMSE degradation vs. the clean run on the **same drone indices**, with a Welch t-test over episodes. "signif worse/better" counts runs with p < 0.05. Efficacy gate is ≥ 30 %.

| tier | channel | f | runs | max deg % | median | **signif worse** | signif better |
|---|---|---|---|---|---|---|---|
| 2 (hand-designed) | motors only | 0.125 | 27 | **22.2 %** | 1.7 % | **2** | 2 |
| 2 (hand-designed) | motors only | 0.250 | 24 | **15.3 %** | 3.7 % | **0** | 1 |
| 2 (hand-designed) | observation channel | 0.125 | 9 | **30.4 %** | 2.0 % | **3** | 0 |
| 2 (hand-designed) | observation channel | 0.250 | 9 | **29.9 %** | 9.4 % | **3** | 0 |
| 3 (**trained**) | motors only | 0.125 | 28 | **17.6 %** | -3.8 % | **0** | 2 |
| 3 (**trained**) | motors only | 0.250 | 14 | **7.7 %** | -4.2 % | **0** | 1 |

## What this says

* **Only the observation channel produces attacks that clear the gate.** 6 of 18 `liar` runs significantly damage the honest team, topping out at **30.4 %** (`t2_liar_s3_mirror_f0125`, p = 0.0007) — a hand-designed, unoptimised lie that clears 30 %.
* **Motor authority is not enough, for anyone.** Across 93 motors-only runs — 51 hand-designed plus 42 trained checkpoints — only 2 significantly damage the honest team, and the ceiling is 22.2 %.
* **Motors-only attacks are about as likely to help as to hurt**: 2 significantly worse vs 6 significantly *better* for the honest team. A Byzantine drone that flails and crashes takes a crowding constraint out of an 8-drone formation, and the remaining drones hold station better than in a clean run. In sign, motor-authority faults are indistinguishable from noise.
* **The trained adversary is not underperforming — it has nothing to work with.** Its best checkpoint reaches 17.6 % (p = 0.075), inside the hand-designed motors-only band. 200M steps of direct optimisation neither beat nor lost to hand-designed attacks on the same channel. **The optimiser is not the bottleneck; the channel is.**
* **Doubling f does not help**, and for motors-only attacks it hurts: two drones removed from an 8-drone formation leaves fewer honest neighbours to disturb, and honest damage is measured only over the six that remain.

## The same data, split by baseline seed

| channel | baseline seed | runs | max deg % | signif worse | clean goal RMSE |
|---|---|---|---|---|---|
| motors only | 1 | 17 | **4.9 %** | 0 | 1.1123 |
| motors only | 2 | 17 | **18.6 %** | 0 | 1.0463 |
| motors only | 3 | 17 | **22.2 %** | 2 | 1.0085 |
| observation channel | 1 | 6 | **12.6 %** | 0 | 1.1123 |
| observation channel | 2 | 6 | **17.0 %** | 1 | 1.0463 |
| observation channel | 3 | 6 | **30.4 %** | 5 | 1.0085 |
| observation channel (**trained**) | 1 | 14 | **17.6 %** | 0 | 1.1123 |

**Baseline seed matters more than channel does.** The most attackable baseline (seed 3) is also the one with the *lowest* clean goal RMSE — the tightest formation is the most brittle one, which is a result in its own right. Seed 1 is close to unattackable: 0 significant results across every hand-designed attack on either channel. Any conclusion drawn from a single baseline seed is a conclusion about that seed.


## Implication for Tier 3 round 3

Give the learned adversary the **observation channel** — the `lies_about_state` capability `agents/tier2_structured.py`'s `liar` already implements — rather than more compromised drones. It is the only channel with measured headroom, a hand-designed lie already clears the gate on it, and Tier 3's central claim (trained beats hand-designed) has never actually been testable, because every adversary so far was optimising on a channel where hand-designed attacks do nothing either.

**Caveat to carry forward:** the significant `liar` results are concentrated in baseline seed 3. Round 3 must attack more than one baseline seed before the channel claim can be stated without qualification.
