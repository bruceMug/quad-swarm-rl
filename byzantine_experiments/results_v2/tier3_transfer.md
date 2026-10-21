# Tier 3 cross-seed transfer

```
Cross-seed transfer, final checkpoints, n = 200 per cell, paired

  adversary                                       baseline s1              baseline s2              baseline s3
  seed1_stealth0_lieonly              +4.6 [ +0.4, +8.9] *  +0.9 [ -2.6, +4.4]    +5.2 [ +1.1, +9.3] T
  seed2_stealth0_lieonly              +3.5 [ -0.9, +7.9]    +3.1 [ -0.7, +7.0] *  +3.9 [ -0.1, +7.9]  
  seed3_stealth0_lieonly              +4.9 [ +0.5, +9.3] T  +1.6 [ -2.6, +5.9]    +7.4 [ +3.1,+11.7] *
  seed3_stealth0_lieonlyego           +2.0 [ -1.4, +5.3]    +3.8 [ +0.1, +7.5] T  +8.0 [ +3.7,+12.3] *
  seed1_stealth0_lieonlyplaus         +2.8 [ -1.4, +7.0] *  +1.1 [ -2.4, +4.5]    +4.3 [ -0.3, +8.8]  
  seed2_stealth0_lieonlyplaus         +1.9 [ -1.6, +5.5]    +2.6 [ -1.0, +6.2] *  +6.5 [ +2.0,+11.0] T
  seed3_stealth0_lieonlyplaus         +3.1 [ -1.3, +7.5]    +1.8 [ -1.8, +5.3]    +7.1 [ +2.4,+11.8] *

  * own seed (the confirmatory run).  T = transfers: significant damage on a seed the adversary was not trained against.
```
