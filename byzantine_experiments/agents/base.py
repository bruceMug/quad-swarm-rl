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


class EgoState:
    """
    One drone's own true state, handed to agents that declare `needs_ego_state`.

    Tier 1 agents never see this -- a stochastic fault has no use for it, and withholding it
    makes "no team state access" trivially true. Tier 2+ agents may be goal-directed, so they
    get their OWN pos/vel/rot/omega (and the inverse control Jacobian of their own airframe,
    for the position controller). It deliberately carries nothing about any other drone.
    """

    __slots__ = ('pos', 'vel', 'rot', 'omega', 'jacobian', 'jacobian_inv', 'room_box',
                 'control_dt')

    def __init__(self, pos, vel, rot, omega, jacobian=None, jacobian_inv=None,
                 room_box=None, control_dt=None):
        self.pos = pos
        self.vel = vel
        self.rot = rot
        self.omega = omega
        self.jacobian = jacobian
        self.jacobian_inv = jacobian_inv
        self.room_box = room_box
        # Seconds per control step. Needed by a plausible liar (Tier 3, D3) to keep its claimed
        # velocity consistent with its claimed trajectory; describes only the drone's own clock.
        self.control_dt = control_dt


class ByzantineAgent(ABC):
    """
    Transforms an honest policy's output into Byzantine behaviour.

    Contract:
      - perturb_action() is called once per env step per compromised agent
      - it MUST return an action in the same dtype/shape/range as the input
      - it MUST NOT mutate its input array in place
      - it MAY maintain internal state across steps (reset via reset())

    Capability flags (all default off, so Tier 0/1 behaviour is unchanged):
      - needs_ego_state: the wrapper sets `self.ego` (an EgoState) before each
        perturb_action call. Own state only -- never a neighbour's.
      - lies_about_state: the wrapper routes this agent's reported position/velocity
        through perturb_broadcast_state() and rewrites the neighbour block of every OTHER
        drone's observation accordingly.
      - needs_coordination: the wrapper hands every such agent the same mutable dict, so
        colluding attackers can share a plan. Only other Byzantine agents can write to it.
    """

    needs_ego_state = False
    lies_about_state = False
    needs_coordination = False

    def __init__(self, agent_idx: int, action_dim: int, seed: int = 0,
                 action_low=-1.0, action_high=1.0, **kwargs):
        self.agent_idx = agent_idx
        self.action_dim = action_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed + 1000 * agent_idx)
        # Populated by the wrapper from env.action_space so tiers never hardcode the range.
        self.action_low = action_low
        self.action_high = action_high
        # Populated by the wrapper when the corresponding capability flag is set.
        self.ego = None
        self.coordination = None

    @abstractmethod
    def perturb_action(self, honest_action: np.ndarray, obs: np.ndarray,
                       t: int) -> np.ndarray:
        ...

    def perturb_broadcast_state(self, true_state: np.ndarray, t: int) -> np.ndarray:
        """
        Optional: corrupt what THIS agent reports to neighbours.

        `true_state` is a 6-vector `[x, y, z, vx, vy, vz]` in world coordinates -- this
        drone's true position and velocity, which is exactly what the env publishes into
        every other drone's neighbour observation block. Return a 6-vector in the same
        layout; the wrapper converts it to the relative form each observer expects.

        Default is honest (identity). Overriding it is only effective if the subclass also
        sets `lies_about_state = True`, which is what tells the wrapper to intercept.
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
