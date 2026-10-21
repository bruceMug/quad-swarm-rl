"""
Where rollout results live -- defined once.

Everything collected before 2026-09-24 sits under `byzantine_experiments/results/` and was
produced by a collector with two seeding bugs (see design/tier3_design.md 6.16-6.18):

  1. `--seed` did not control the scenario draw at all -- each drone's spawn/goal RNG was
     seeded from OS entropy, so two runs at one seed started 2.9 m apart;
  2. even once seeded, a clean run and an attacked run desynchronised after a few episodes,
     because collisions draw from the global `np.random` stream.

The consequence was a ~7 % run-to-run offset that no reported confidence interval included, and
headline numbers that did not replicate (Tier 2 `mirror`: reported +30.4 %, properly paired
+9.7 %). That data is kept, untouched, as the "before" in the before/after comparison -- but it
must never be mixed with data from the fixed collector, and the sweep scripts skip configs whose
output already exists, so re-collecting into the same folders would silently reuse it.

So re-collection goes to a new root. Set BYZ_RESULTS_ROOT to point any script or analysis
module at a different one; `BYZ_RESULTS_ROOT=byzantine_experiments/results` reproduces the
legacy analyses exactly.
"""

import os

LEGACY_ROOT = 'byzantine_experiments/results'
RESULTS_ROOT = os.environ.get('BYZ_RESULTS_ROOT', 'byzantine_experiments/results_v2')

# Written into every sidecar by the fixed collector. Analyses that pair runs refuse files that
# lack it: a missing tag means the scenarios were not controlled and pairing is meaningless.
SEEDING_SCHEME = 'per_episode_v2'


def results_dir(tier):
    """e.g. results_dir('tier1') -> byzantine_experiments/results_v2/tier1"""
    return os.path.join(RESULTS_ROOT, tier)
