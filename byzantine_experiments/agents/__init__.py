"""Byzantine agent implementations. Importing this package populates BYZANTINE_REGISTRY."""

from byzantine_experiments.agents.base import (  # noqa: F401
    BYZANTINE_REGISTRY,
    ByzantineAgent,
    clip_to_action_space,
    get_byzantine_class,
    register_byzantine,
)

# Import side effect: each module registers its tiers.
from byzantine_experiments.agents import noop  # noqa: F401,E402
from byzantine_experiments.agents import tier1_stochastic  # noqa: F401,E402
from byzantine_experiments.agents import tier2_structured  # noqa: F401,E402
from byzantine_experiments.agents import tier3_learned  # noqa: F401,E402

__all__ = [
    'BYZANTINE_REGISTRY',
    'ByzantineAgent',
    'clip_to_action_space',
    'get_byzantine_class',
    'register_byzantine',
]
