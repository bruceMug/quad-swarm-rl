"""
Env construction for Byzantine rollouts.

This mirrors sample_factory's `make_env_func_batched` exactly, with one extra wrapper slid
in between the quadrotor env and `BatchedVecEnv`. Doing it here (rather than editing
swarm_rl/env_wrappers/quad_utils.py) keeps the upstream diff to the CLI flags alone.
"""

import json

from sample_factory.algo.utils.make_env import BatchedVecEnv
from sample_factory.utils.attr_dict import AttrDict

from byzantine_experiments.wrappers.byzantine_env import ByzantineEnvWrapper


def byzantine_kwargs_from_cfg(cfg):
    """Translate the --byzantine_* CLI flags into ByzantineEnvWrapper kwargs.

    Returns None when --byzantine_enabled is False, which the factory reads as 'no wrapper'.
    """
    if not getattr(cfg, 'byzantine_enabled', False):
        return None

    indices = [int(x) for x in str(cfg.byzantine_indices).split(',') if x.strip() != '']
    return dict(
        tier=cfg.byzantine_tier,
        compromised_indices=indices,
        tier_config=json.loads(cfg.byzantine_config or '{}'),
        seed=int(cfg.byzantine_seed),
    )


def make_byzantine_env(cfg, byzantine_kwargs=None, env_config=None, render_mode=None):
    """
    Build the batched quadrotor env, optionally with Byzantine injection.

    Returns (batched_env, byzantine_wrapper_or_None). The second element is the handle used
    to read the perturbation log and the ground-truth mask -- `BatchedVecEnv` hides it.

    If `byzantine_kwargs` is None the --byzantine_* flags on `cfg` are consulted instead.
    """
    from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env

    if byzantine_kwargs is None:
        byzantine_kwargs = byzantine_kwargs_from_cfg(cfg)

    if env_config is None:
        env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)

    env = make_quadrotor_env(cfg.env, cfg=cfg, _env_config=env_config, render_mode=render_mode)

    byz_wrapper = None
    if byzantine_kwargs:
        byz_wrapper = ByzantineEnvWrapper(env, **byzantine_kwargs)
        env = byz_wrapper

    return BatchedVecEnv(env), byz_wrapper
