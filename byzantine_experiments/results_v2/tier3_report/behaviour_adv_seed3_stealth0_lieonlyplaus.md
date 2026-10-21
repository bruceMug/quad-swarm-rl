# Behaviour -- `adv_seed3_stealth0_lieonlyplaus` (spec item 7)

200 paired episodes (own-seed confirmatory vs clean, seed 3); figures show the first 10. Broadcasts recorded. These are the numbers the plain-language description must agree with.

| observation | attacked | clean | paired p |
|---|---|---|---|
| honest goal error, episode mean (m) | 0.781 | 0.741 | 0.0082 |
| adversary's OWN goal error change vs same drone clean (m) | +0.130 | 0 | 0.0002 |
| team collisions per episode | 0.91 | 0.42 | 0.0000 |
| honest–honest minimum separation (m) | 0.130 | 0.133 | 0.4087 |
| collision steps whose closest pair INCLUDES the adversary's slot | 0.54 | 0.11 | 0.0000 |
| collision steps between two honest drones | 0.38 | 0.32 | 0.2823 |
| adversary-slot ↔ nearest honest, minimum separation (m) | 0.131 | 0.174 | 0.0000 |
| honest speed (m/s) | 1.022 | 0.991 | 0.0023 |
| honest jitter, mean ‖Δv‖ per step (m/s) | 0.0242 | 0.0226 | 0.0003 |

**When.** Share of the honest goal-error increase accrued in the first / middle / last third of the episode: +0.15 / +0.33 / +0.52.
**Where.** Spearman correlation of an honest drone's goal-error increase with its mean distance to the adversary's slot in the same scenario, clean: ρ = -0.159 (p = 2.15e-09). Mean increase for the 2 honest drones nearest that slot +0.096 m vs the 2 farthest -0.069 m.
**The phantom** (first 10 episodes). Distance from honest drones to the adversary's broadcast position: median 2.04 m, vs to its true position 0.65 m.

![](behaviour_adv_seed3_stealth0_lieonlyplaus_episodes.png)

![](behaviour_adv_seed3_stealth0_lieonlyplaus_time.png)
