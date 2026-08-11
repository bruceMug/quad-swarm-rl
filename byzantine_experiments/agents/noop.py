"""
Tier 0 control agent: an "attacker" that does nothing.

This exists purely as a null hypothesis. A rollout with `tier0_noop` compromising any
subset of drones must be bit-identical to an unwrapped rollout -- if it is not, the
wrapper is leaking state and every downstream tier's numbers are suspect.
"""

import numpy as np

from byzantine_experiments.agents.base import ByzantineAgent, register_byzantine


@register_byzantine('tier0_noop')
class NoOpAgent(ByzantineAgent):
    """Returns the honest action untouched (as a copy, per the no-in-place-mutation rule)."""

    def perturb_action(self, honest_action: np.ndarray, obs: np.ndarray,
                       t: int) -> np.ndarray:
        return np.array(honest_action, copy=True)
