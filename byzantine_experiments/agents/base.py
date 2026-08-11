"""
Byzantine agent abstraction and registry (Tier 0 infrastructure).

A Byzantine agent is a *post-policy* transform: it receives the action that the frozen
honest policy produced for one drone and returns the action that will actually be executed.
It never sees the policy weights, which is what makes Tiers 1-3 directly comparable --
every tier sits on top of the exact same honest controller.
"""

from abc import ABC, abstractmethod

import numpy as np

# name -> ByzantineAgent subclass. Populated by @register_byzantine.
BYZANTINE_REGISTRY = {}


def register_byzantine(name):
    """Class decorator: make a Byzantine agent addressable by string name."""

    def deco(cls):
        if name in BYZANTINE_REGISTRY and BYZANTINE_REGISTRY[name] is not cls:
            raise ValueError(f"Byzantine tier name '{name}' is already registered "
                             f"to {BYZANTINE_REGISTRY[name].__name__}")
        BYZANTINE_REGISTRY[name] = cls
        return cls

    return deco


def get_byzantine_class(name):
    """Look up a registered Byzantine agent class, with a helpful error message."""
    if name not in BYZANTINE_REGISTRY:
        known = ', '.join(sorted(BYZANTINE_REGISTRY)) or '<none registered>'
        raise KeyError(f"Unknown Byzantine tier '{name}'. Registered tiers: {known}")
    return BYZANTINE_REGISTRY[name]


def clip_to_action_space(action, low=-1.0, high=1.0):
    """Clip an action to the env's legal range. Every tier must call this on its output."""
    return np.clip(action, low, high)


class ByzantineAgent(ABC):
    """
    Transforms an honest policy's output into Byzantine behaviour.

    Contract:
      - perturb_action() is called once per env step per compromised agent
      - it MUST return an action in the same dtype/shape/range as the input
      - it MUST NOT mutate its input array in place
      - it MAY maintain internal state across steps (reset via reset())
    """

    def __init__(self, agent_idx: int, action_dim: int, seed: int = 0,
                 action_low=-1.0, action_high=1.0, **kwargs):
        self.agent_idx = agent_idx
        self.action_dim = action_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed + 1000 * agent_idx)
        # Populated by the wrapper from env.action_space so tiers never hardcode the range.
        self.action_low = action_low
        self.action_high = action_high

    @abstractmethod
    def perturb_action(self, honest_action: np.ndarray, obs: np.ndarray,
                       t: int) -> np.ndarray:
        ...

    def perturb_broadcast_state(self, true_state: np.ndarray, t: int) -> np.ndarray:
        """
        Optional: corrupt what THIS agent reports to neighbours.
        Default is honest (identity). Override in tiers that lie about state.
        """
        return true_state

    def reset(self):
        """
        Called at episode boundaries. Subclasses clear any internal state here.

        Note the RNG stream is deliberately *not* reseeded: episodes within one run should
        see different noise draws, while the run as a whole stays reproducible from `seed`.
        """
        pass

    def clip(self, action):
        """Clip to this env's action range (set by the wrapper from env.action_space)."""
        return clip_to_action_space(action, self.action_low, self.action_high)

    @property
    def name(self) -> str:
        return self.__class__.__name__
