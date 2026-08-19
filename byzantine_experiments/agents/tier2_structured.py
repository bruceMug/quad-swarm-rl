"""
Tier 2 — Structured Byzantine attacks (untrained).

The first tier with **intent**. These agents want to disrupt the team, but pursue it with a
fixed hand-designed rule rather than a learned strategy: adversarial without being adaptive.

What separates them from Tier 1:
  - they may read their OWN state (position, velocity, attitude) and act on it, via the
    wrapper's `needs_ego_state` capability. Tier 1 could not.
  - 2C may lie to its neighbours (`lies_about_state`), which Tier 1 was explicitly forbidden.
  - 2D members may coordinate with each other through a shared channel.

What still holds: none of them reads an *honest* drone's state. 2D colluders share only what
they themselves put in the channel.

Survival guard
--------------
An attack that flips the drone into the floor in two seconds removes the attacker from the
episode and yields no detection signal, so every variant supports `survival_guard`: below
`guard_altitude` the emitted action blends back toward the honest one, proportionally to how
far below the threshold the drone is. Trigger rates are counted per agent
(`guard_trigger_rate`) and reported, because a guard that fires constantly means the attack
is really "honest flight with occasional sabotage" and the calibration must say so.
"""

import numpy as np

from byzantine_experiments.agents.base import ByzantineAgent, register_byzantine
from byzantine_experiments.agents.control_utils import position_control_action


class _Tier2Base(ByzantineAgent):
    """Shared survival guard, ego-state plumbing, and diagnostics."""

    needs_ego_state = True

    def __init__(self, *args, survival_guard=True, guard_altitude=0.6,
                 guard_full_altitude=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.survival_guard = bool(survival_guard)
        self.guard_altitude = float(guard_altitude)
        self.guard_full_altitude = float(guard_full_altitude)
        if self.guard_full_altitude >= self.guard_altitude:
            raise ValueError('guard_full_altitude must be below guard_altitude')

        self.guard_steps = 0
        self.total_steps = 0
        self.last_guard_blend = 0.0
        # Lifetime counters survive reset(), because the run-level diagnostics are read
        # AFTER the final episode's reset() -- per-episode counters are zero by then.
        self.lifetime_guard_steps = 0
        self.lifetime_total_steps = 0

    def reset(self):
        super().reset()
        self.guard_steps = 0
        self.total_steps = 0
        self.last_guard_blend = 0.0

    @property
    def guard_trigger_rate(self):
        """Guard rate over the whole run (all episodes), not just the current one."""
        return (self.lifetime_guard_steps / self.lifetime_total_steps
                if self.lifetime_total_steps else 0.0)

    @property
    def episode_guard_trigger_rate(self):
        return self.guard_steps / self.total_steps if self.total_steps else 0.0

    def _apply_guard(self, attack_action, honest_action):
        """
        Blend back toward honest as altitude falls, so the attacker stays airborne.

        blend = 0 above `guard_altitude`, ramping linearly to 1 at `guard_full_altitude`.
        A hard switch would make the attacker's action stream discontinuous in a way that is
        itself a giveaway; a ramp keeps it smooth.
        """
        self.total_steps += 1
        self.lifetime_total_steps += 1
        self.last_guard_blend = 0.0
        if not self.survival_guard or self.ego is None:
            return attack_action

        z = float(self.ego.pos[2])
        if z >= self.guard_altitude:
            return attack_action

        span = self.guard_altitude - self.guard_full_altitude
        blend = float(np.clip((self.guard_altitude - z) / span, 0.0, 1.0))
        self.guard_steps += 1
        self.lifetime_guard_steps += 1
        self.last_guard_blend = blend
        return (1.0 - blend) * np.asarray(attack_action, dtype=np.float64) \
            + blend * np.asarray(honest_action, dtype=np.float64)

    def diagnostics(self):
        return dict(guard_trigger_rate=self.guard_trigger_rate,
                    guard_steps=self.lifetime_guard_steps,
                    total_steps=self.lifetime_total_steps,
                    guard_altitude=self.guard_altitude)


@register_byzantine('tier2_signflip')
class SignFlipAgent(_Tier2Base):
    """
    Variant 2A — sign flip. The canonical Byzantine attack from federated learning.

        a'[d] = clip(-scale * a[d])   for d in dims,   a'[d] = a[d] otherwise

    Args:
        scale: magnitude multiplier, {0.5, 1.0}.
        dims: which dimensions to flip. `None`/'all' flips all four.
        space: `'motor'` (default) flips raw motor commands; `'body'` flips
            [collective, roll, pitch, yaw] accelerations instead. See below.

    **On `space` — a mismatch between the spec and this env.** The spec describes
    `dims=[1,2]` as "flips roll/pitch but leaves thrust honest". That assumes an action
    parameterised as (thrust, roll, pitch, yaw). This env's action is **four raw motor
    thrusts** (`RawControl`), so flipping action dims 1 and 2 flips two *rotors*, not two
    body axes.

    Both are therefore supported:

      - `space='motor'` — literal reading of the spec. `dims='all'` inverts every rotor
        command, which inverts collective thrust and drops the drone out of the sky almost
        immediately. Legitimate but boring: trivially detectable and it removes the attacker
        from the episode.
      - `space='body'` — the spec's *intent*. The action is decomposed through the control
        Jacobian into `[collective, roll, pitch, yaw]` accelerations, the selected components
        are negated, and it is mapped back to motor commands. `dims=[1, 2]` then genuinely
        means "invert roll and pitch response, leave collective thrust honest", which keeps
        the drone airborne while destroying its lateral coordination.

    **`space='body', dims=[1,2]` is the canonical partial setting.** Both are measured.
    """

    def __init__(self, *args, scale=1.0, dims=None, space='motor', **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = float(scale)
        if space not in ('motor', 'body'):
            raise ValueError(f"space must be 'motor' or 'body', got {space!r}")
        self.space = space
        if dims is None or (isinstance(dims, str) and dims.lower() == 'all'):
            self.dims = list(range(self.action_dim))
        else:
            self.dims = [int(d) for d in dims]
        for d in self.dims:
            if not 0 <= d < self.action_dim:
                raise ValueError(f'dim {d} out of range for action_dim={self.action_dim}')
        if len(set(self.dims)) != len(self.dims):
            raise ValueError(f'duplicate dims: {self.dims}')

    def perturb_action(self, honest_action, obs, t):
        honest_action = np.asarray(honest_action, dtype=np.float64)

        if self.space == 'motor':
            out = honest_action.copy()
            out[self.dims] = -self.scale * honest_action[self.dims]
        else:
            if self.ego is None or self.ego.jacobian is None:
                return self.clip(honest_action)      # no ego state yet: stay honest
            from byzantine_experiments.agents.control_utils import action_to_body, body_to_action
            body = action_to_body(honest_action, self.ego.jacobian)
            body[self.dims] = -self.scale * body[self.dims]
            out = body_to_action(body, self.ego.jacobian_inv)

        return self.clip(self._apply_guard(out, honest_action))


@register_byzantine('tier2_waypoint_pull')
class WaypointPullAgent(_Tier2Base):
    """
    Variant 2B — waypoint pull. Ignores the team goal and flies at a fixed adversarial
    waypoint, dragging neighbours off-formation through the collision-avoidance coupling.

        a' = clip((1 - alpha) * a + alpha * position_control_action(ego -> w_adv))

    Args:
        w_adv: adversarial waypoint, world coordinates. Default `None` picks a far corner of
            the room from the agent's own `room_box` at the first step -- deterministic given
            the arena, and safely inside the walls rather than through them.
        alpha: blend weight, {0.3, 0.6, 1.0}. How committed the attacker is.

    The direction->action mapping is the simulator's own Mellinger position controller,
    replicated in `agents/control_utils.py` (it cannot be imported because its `__init__`
    requires tensorflow). Reading own position is allowed for Tier 2; neighbours' states are
    never touched.
    """

    def __init__(self, *args, w_adv=None, alpha=0.6, corner_margin=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f'alpha must be in [0, 1], got {self.alpha}')
        self.w_adv_cfg = None if w_adv is None else np.asarray(w_adv, dtype=np.float64)
        self.corner_margin = float(corner_margin)
        self.w_adv = self.w_adv_cfg

    def resolve_waypoint(self):
        """Pick the adversarial waypoint once the arena is known."""
        if self.w_adv_cfg is not None:
            return self.w_adv_cfg
        if self.ego is None or self.ego.room_box is None:
            return np.array([4.0, 4.0, 0.8])
        lo, hi = np.asarray(self.ego.room_box[0]), np.asarray(self.ego.room_box[1])
        # A low corner: far from the swarm's working volume in x/y, and low enough that
        # dragged neighbours are pulled toward the floor rather than the ceiling.
        target = np.array([hi[0] - self.corner_margin,
                           hi[1] - self.corner_margin,
                           lo[2] + self.corner_margin])
        return target

    def perturb_action(self, honest_action, obs, t):
        honest_action = np.asarray(honest_action, dtype=np.float64)
        if self.ego is None:
            return self.clip(honest_action)
        if self.w_adv is None:
            self.w_adv = self.resolve_waypoint()

        attack = position_control_action(
            pos=self.ego.pos, vel=self.ego.vel, rot=self.ego.rot, omega=self.ego.omega,
            target=self.w_adv, jacobian_inv=self.ego.jacobian_inv,
        )
        blended = (1.0 - self.alpha) * honest_action + self.alpha * attack
        return self.clip(self._apply_guard(blended, honest_action))

    def direction_to_waypoint(self):
        """Unit vector from the agent toward its waypoint (used by the tests)."""
        if self.ego is None:
            return None
        w = self.w_adv if self.w_adv is not None else self.resolve_waypoint()
        d = w - self.ego.pos
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else d


@register_byzantine('tier2_liar')
class StateLiarAgent(_Tier2Base):
    """
    Variant 2C — state misreporting. **Flies honestly, lies about where it is.**

    Its executed action is the honest policy's action, bit-for-bit. The attack is entirely in
    `perturb_broadcast_state()`, which the wrapper uses to rewrite this drone's entry in every
    *other* drone's neighbour observation block.

    Args:
        lie_mode: 'offset'  -- report a constant spatial displacement
                  'stale'   -- report the position from `delay_k` steps ago
                  'mirror'  -- report the position reflected through the arena centre
        magnitude: metres, for 'offset' (and the reflection centre offset for 'mirror').
        delay_k: steps of lag for 'stale'. At 100 Hz, k=50 is half a second.
        lie_velocity: whether the reported velocity is also corrupted. For 'stale' this is
            physically consistent (a delayed report delays both); for 'offset' a constant
            position shift leaves velocity unchanged, so it defaults to False there.

    **Note this variant deliberately does NOT set `needs_ego_state` for its actions** -- it
    needs no ego state to act, because it does not perturb its action at all. The wrapper
    supplies its true pos/vel to `perturb_broadcast_state` separately.
    """

    needs_ego_state = False
    lies_about_state = True

    def __init__(self, *args, lie_mode='offset', magnitude=3.0, delay_k=50,
                 lie_velocity=None, mirror_centre=(0.0, 0.0, 2.0), **kwargs):
        super().__init__(*args, **kwargs)
        if lie_mode not in ('offset', 'stale', 'mirror'):
            raise ValueError(f"lie_mode must be offset/stale/mirror, got {lie_mode!r}")
        self.lie_mode = lie_mode
        self.magnitude = float(magnitude)
        self.delay_k = int(delay_k)
        if self.delay_k < 1:
            raise ValueError(f'delay_k must be >= 1, got {self.delay_k}')
        self.lie_velocity = (lie_mode == 'stale') if lie_velocity is None else bool(lie_velocity)
        self.mirror_centre = np.asarray(mirror_centre, dtype=np.float64)

        # A fixed, seeded direction for the constant offset: deterministic per agent, but not
        # axis-aligned, so the lie is not trivially separable by a single coordinate.
        d = self.rng.normal(size=3)
        self.offset_dir = d / max(np.linalg.norm(d), 1e-9)

        self.history = []
        self.lies_told = 0

    def reset(self):
        super().reset()
        self.history = []
        self.lies_told = 0

    def perturb_action(self, honest_action, obs, t):
        """Honest. The whole point of 2C is that the action stream is clean."""
        return np.array(honest_action, copy=True)

    def perturb_broadcast_state(self, true_state, t):
        true_state = np.asarray(true_state, dtype=np.float64)
        pos, vel = true_state[:3], true_state[3:]
        self.history.append(true_state.copy())
        if len(self.history) > self.delay_k + 1:
            self.history.pop(0)

        if self.lie_mode == 'offset':
            reported = np.concatenate([pos + self.magnitude * self.offset_dir, vel])
            if self.lie_velocity:
                reported[3:] = -vel
        elif self.lie_mode == 'stale':
            past = self.history[0]
            reported = past.copy() if self.lie_velocity else np.concatenate([past[:3], vel])
        else:  # mirror
            reported = np.concatenate([2.0 * self.mirror_centre - pos,
                                       -vel if self.lie_velocity else vel])

        self.lies_told += 1
        return reported

    def diagnostics(self):
        d = super().diagnostics()
        d.update(lies_told=self.lies_told, lie_mode=self.lie_mode)
        return d


@register_byzantine('tier2_collusion')
class CollusionAgent(_Tier2Base):
    """
    Variant 2D — coordinated collusion. Two compromised drones apply correlated
    perturbations. Only meaningful at f = 0.25 (2 of 8).

    Args:
        mode: 'same_waypoint' -- both fly at the same adversarial corner, so they arrive
                                 together and punch a coordinated hole in the formation
              'pincer'        -- they fly at opposite corners, tearing the formation apart
                                 along one axis
              'mirror_pair'   -- one flips the sign of its action, the other flips the
                                 opposite dims, so their motor commands are anti-correlated
        alpha: blend weight for the waypoint modes, as 2B.

    Coordination happens through the wrapper's shared `self.coordination` dict, which only
    Byzantine agents can see. Members register themselves in `reset()` and read their rank
    (0 or 1) from the registration order, which is what makes the roles deterministic. They
    never read an honest drone's state.
    """

    needs_coordination = True

    def __init__(self, *args, mode='same_waypoint', alpha=0.6, corner_margin=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in ('same_waypoint', 'pincer', 'mirror_pair'):
            raise ValueError(f'unknown collusion mode {mode!r}')
        self.mode = mode
        self.alpha = float(alpha)
        self.corner_margin = float(corner_margin)
        self.rank = None
        self.w_adv = None

    def reset(self):
        super().reset()
        self.rank = None
        self.w_adv = None
        if isinstance(self.coordination, dict):
            self.coordination.pop('members', None)

    def _ensure_registered(self):
        """Deterministic role assignment: rank = position in the sorted member list."""
        if self.rank is not None:
            return
        members = self.coordination.setdefault('members', [])
        if self.agent_idx not in members:
            members.append(self.agent_idx)
            members.sort()
        self.rank = members.index(self.agent_idx)

    def _corner(self, sign):
        lo, hi = np.asarray(self.ego.room_box[0]), np.asarray(self.ego.room_box[1])
        m = self.corner_margin
        if sign > 0:
            return np.array([hi[0] - m, hi[1] - m, lo[2] + m])
        return np.array([lo[0] + m, lo[1] + m, lo[2] + m])

    def perturb_action(self, honest_action, obs, t):
        honest_action = np.asarray(honest_action, dtype=np.float64)
        if self.ego is None or not isinstance(self.coordination, dict):
            return self.clip(honest_action)
        self._ensure_registered()

        if self.mode == 'mirror_pair':
            # Anti-correlated motor commands: rank 0 flips rotors {0,1}, rank 1 flips {2,3}.
            dims = [0, 1] if self.rank == 0 else [2, 3]
            attack = honest_action.copy()
            attack[dims] = -honest_action[dims]
        else:
            if self.w_adv is None:
                sign = 1 if (self.mode == 'same_waypoint' or self.rank == 0) else -1
                self.w_adv = self._corner(sign)
            target = position_control_action(
                pos=self.ego.pos, vel=self.ego.vel, rot=self.ego.rot, omega=self.ego.omega,
                target=self.w_adv, jacobian_inv=self.ego.jacobian_inv,
            )
            attack = (1.0 - self.alpha) * honest_action + self.alpha * target

        # Publish intent so the partner (and the tests) can verify the coordination.
        self.coordination[f'target_{self.agent_idx}'] = (
            None if self.w_adv is None else self.w_adv.copy())
        return self.clip(self._apply_guard(attack, honest_action))
