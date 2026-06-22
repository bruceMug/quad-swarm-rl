"""swarm_rl package init.

Compatibility shim for Gymnasium >= 1.0.

Gymnasium 1.0 removed ``gymnasium.Wrapper.__getattr__``, which used to forward
attribute access to the wrapped environment. Both this codebase and the installed
version of sample_factory were written against the older behaviour and rely on it
to reach env attributes (e.g. ``num_agents``, ``is_multiagent``) through their
wrapper stacks. Without forwarding, sample_factory's own wrappers (e.g.
``BatchedListToDictWrapper``) silently report ``num_agents=1``, which causes an
"Env info mismatch" between the batched env-info probe and the non-batched
rollout workers.

We restore the old forwarding behaviour for all gymnasium wrappers here, since
this module is imported by every process that constructs an environment.
"""

import gymnasium as _gymnasium

if not hasattr(_gymnasium.Wrapper, "__getattr__"):

    def _wrapper_getattr(self, name):
        # Avoid recursion if 'env' itself is not set yet, and don't forward
        # private/dunder attributes (matches the old gymnasium semantics).
        if name == "env" or name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    _gymnasium.Wrapper.__getattr__ = _wrapper_getattr
