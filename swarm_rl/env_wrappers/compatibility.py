from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api

# Wrapper for compatibility with gym 0.26
# Mostly copied from gym.EnvCompatability
# Modified since swarm_rl does not have a seed, and is a vectorized env
class QuadEnvCompatibility(gym.Wrapper):
    def __init__(self, env, ):
        """A wrapper which converts old-style envs to valid modern envs.

        Some information may be lost in the conversion, so we recommend updating your environment.

        Args:
            env (LegacyEnv): the env to wrap, implemented with the old API
        """
        gym.Wrapper.__init__(self, env)
        # Gymnasium >=1.0 removed Wrapper.__getattr__ attribute forwarding, so we must
        # explicitly expose these so sample_factory detects this env as multi-agent
        # (otherwise it wraps us with NonBatchedMultiAgentWrapper and breaks the data layout).
        self.is_multiagent = env.unwrapped.is_multiagent
        self.num_agents = env.unwrapped.num_agents

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        """Resets the environment.

        Args:
            seed: the seed to reset the environment with
            options: the options to reset the environment with

        Returns:
            (observation, info)
        """
        return self.env.reset(), {}

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Steps through the environment.

        Args:
            action: action to step through the environment with

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # For QuadMultiEnv, truncated is actually integrated in the env,
        # since the termination is tick > ep_len
        obs, reward, done, info = self.env.step(action)

        # sample_factory's non-batched multi-agent sampler expects per-agent
        # terminated/truncated sequences (one entry per agent).
        if isinstance(done, (list, np.ndarray)):
            terminated = [bool(d) for d in done]
            truncated = [False] * len(terminated)
        else:
            terminated = bool(done)
            truncated = False
        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        """Renders the environment.
        Returns:
            The rendering of the environment, depending on the render mode
        """
        return self.env.render()
