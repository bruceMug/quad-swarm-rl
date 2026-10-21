"""
Tier 3 — the trained adversary, exposed as a ByzantineAgent.

Wrapping the learned policy in the same `ByzantineAgent` interface Tiers 1 and 2 use means it
evaluates through **exactly the same pipeline**: `rollout/collect.py` produces `.npz` files in
the identical schema, `analysis/metrics.py` computes the identical metrics, and the behaviour
descriptors and figures come out comparable. The spec asks for this explicitly ("collect
trajectories using the Tier 0 collector, matching the Tier 1/2 format exactly so Phase 3 can
process all tiers with one code path") -- doing it via the agent interface rather than a
bespoke evaluator is what makes that true rather than approximately true.

Note the asymmetry with training: during *training* the honest team lives inside the env
(`adversary/frozen_team.py`) because Sample Factory has to be handed a 1-agent env. During
*evaluation* the honest team is the ordinary 8-drone env and the adversary is a Byzantine
agent injected by `ByzantineEnvWrapper`. Both arrangements put the same policy on the same
drone with the same frozen neighbours; only the plumbing differs.

The adversary ignores `honest_action` entirely -- it is a policy, not a perturbation.
"""

import glob
import json
import os
import re

import numpy as np

from byzantine_experiments.adversary.lies import (
    DEFAULT_MAX_LIE_POS,
    DEFAULT_MAX_LIE_VEL,
    EGO_AUG_DIM,
    DEFAULT_PLAUSIBLE_MAX_ACCEL,
    DEFAULT_PLAUSIBLE_MAX_OFFSET,
    DEFAULT_PLAUSIBLE_MAX_RATE,
    LIE_DIM,
    PlausibleLie,
    augment_obs,
    decode_lie,
    lie_action_dim,
)
from byzantine_experiments.agents.base import ByzantineAgent, register_byzantine

_INT_RE = re.compile(r'\d+')


def checkpoint_env_steps(path):
    """
    Env-step count encoded in a Sample Factory checkpoint filename.

    Two shapes exist and both put env steps *second*:
        checkpoint_<iteration>_<env_steps>.pth
        best_<iteration>_<env_steps>_reward_<value>.pth
    A trailing-number regex works for the first and silently mis-parses the second (it would
    pick up the reward), so take the second integer rather than the last.
    """
    nums = _INT_RE.findall(os.path.basename(path))
    if len(nums) >= 2:
        return int(nums[1])
    return int(nums[0]) if nums else -1


def list_milestones(run_dir, policy_index=0):
    """(env_steps, path) for every milestone checkpoint, ascending. Used for horizon sweeps."""
    pattern = os.path.join(run_dir, f'checkpoint_p{policy_index}', 'milestones', '*.pth')
    out = []
    for p in glob.glob(pattern):
        out.append((checkpoint_env_steps(p), p))
    return sorted(out)


def resolve_checkpoint(run_dir, policy_index=0, kind='latest', milestone=None):
    """
    Pick which adversary checkpoint to evaluate.

    kind='latest' | 'best', or `milestone=<env_steps>` to select the milestone nearest that
    step count -- which is how the horizon question ("was it better earlier?") gets answered
    without retraining.
    """
    ckpt_dir = os.path.join(run_dir, f'checkpoint_p{policy_index}')
    if milestone is not None:
        stones = list_milestones(run_dir, policy_index)
        if not stones:
            raise FileNotFoundError(f'no milestones under {ckpt_dir}/milestones')
        steps, path = min(stones, key=lambda sp: abs(sp[0] - int(milestone)))
        return path, steps

    prefix = dict(latest='checkpoint', best='best')[kind]
    files = sorted(glob.glob(os.path.join(ckpt_dir, f'{prefix}_*.pth')))
    if not files:
        raise FileNotFoundError(f'no {prefix}_* checkpoint under {ckpt_dir}')
    path = max(files, key=checkpoint_env_steps)
    return path, checkpoint_env_steps(path)


@register_byzantine('tier3_learned')
class LearnedAdversaryAgent(ByzantineAgent):
    """
    A trained adversarial policy acting as one drone in an otherwise honest swarm.

    Args:
        checkpoint: the adversary's run directory, e.g. `train_dir/adv_seed1_stealth0`.
        kind: 'latest' (default) or 'best'.
        milestone: env-step count; selects the nearest milestone checkpoint instead of
            `kind`. This is the knob for the horizon question.
        deterministic: take the policy's argmax (default), matching how every previous tier
            collected its rollouts. Sampling would add a second source of variation and make
            Tier 3 not comparable to Tiers 1/2.
        device: 'cpu' (default) -- rollouts parallelise better on CPU.

    The policy is loaded once per agent instance. With two compromised drones both load the
    same weights; that is a few MB and keeps each drone's inference independent.
    """

    # It reads its own observation, like Tier 2. The wrapper hands `obs` to perturb_action
    # directly, so no extra capability flag is needed.
    needs_ego_state = False

    # Round 3 adversaries broadcast; rounds 1-2 do not. Resolved per checkpoint in
    # `__init__` (an instance attribute shadowing this class default), because
    # `ByzantineEnvWrapper` computes `any_lies` from the constructed agents. Leaving it True
    # for everyone would route a motors-only checkpoint through `_apply_broadcast_lies` --
    # rewriting its neighbour slots with recomputed "true" values that need not be
    # bit-identical to what the simulator wrote, silently changing the 28 sidecars rounds 1-2
    # were measured from.
    lies_about_state = False

    def __init__(self, *args, checkpoint=None, kind='latest', milestone=None,
                 deterministic=True, device='cpu', **kwargs):
        super().__init__(*args, **kwargs)
        if not checkpoint:
            raise ValueError("tier3_learned requires checkpoint='train_dir/adv_seedN_stealthW'")
        self.run_dir = str(checkpoint).rstrip('/')
        self.kind = kind
        self.milestone = milestone
        self.deterministic = bool(deterministic)
        self.device = device

        self.checkpoint_file, self.checkpoint_steps = resolve_checkpoint(
            self.run_dir, kind=kind, milestone=milestone)

        self._policy = None       # built lazily: the wrapper constructs agents eagerly
        self._rnn = None
        self.steps_acted = 0

        # Round 3. Read off the checkpoint's own config so evaluation cannot disagree with
        # training about whether this adversary has the observation channel, or how far its
        # lie reaches. Read eagerly, not with the policy: the wrapper inspects
        # `lies_about_state` at construction time to decide whether to run the broadcast path
        # at all.
        cfg_json = {}
        cfg_path = os.path.join(self.run_dir, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg_json = json.load(f)
        self.enable_lies = bool(cfg_json.get('adv_enable_lies', False))
        self.lie_only = bool(cfg_json.get('adv_lie_only', False))
        # Round 5. Default False, so every rounds 1-4 checkpoint (whose config.json predates
        # the key) keeps its 54-D observation and loads unchanged.
        self.ego_state_obs = bool(cfg_json.get('adv_ego_state_obs', False))
        # D3. A plausible liar is stateful (offset + drift) and needs the room box and control
        # timestep, both delivered through EgoState -- so it declares needs_ego_state too.
        self.lie_mode = str(cfg_json.get('adv_lie_mode', 'free'))
        self.plausible_params = dict(
            max_offset=float(cfg_json.get('adv_plausible_max_offset', DEFAULT_PLAUSIBLE_MAX_OFFSET)),
            max_rate=float(cfg_json.get('adv_plausible_max_rate', DEFAULT_PLAUSIBLE_MAX_RATE)),
            max_accel=float(cfg_json.get('adv_plausible_max_accel', DEFAULT_PLAUSIBLE_MAX_ACCEL)))
        self._plausible = None
        self.needs_ego_state = self.ego_state_obs or (self.enable_lies and self.lie_mode == 'plausible')
        self.max_lie_pos = float(cfg_json.get('adv_max_lie_pos', DEFAULT_MAX_LIE_POS))
        self.max_lie_vel = float(cfg_json.get('adv_max_lie_vel', DEFAULT_MAX_LIE_VEL))
        self.lies_about_state = self.enable_lies
        self._pending_lie = None     # last lie action, set by perturb_action
        self.lies_told = 0

    # ------------------------------------------------------------------ policy loading

    def _ensure_policy(self):
        if self._policy is not None:
            return
        import gymnasium as gym
        import torch
        from sample_factory.model.model_utils import get_rnn_size

        from byzantine_experiments.adversary.env_factory import baseline_cfg_for
        from byzantine_experiments.adversary.frozen_team import FrozenTeamPolicy

        ckpt_dir = os.path.join(self.run_dir, 'checkpoint_p0')
        cfg = baseline_cfg_for(ckpt_dir)          # reads the ADVERSARY's own config.json
        cfg.load_checkpoint_kind = 'latest'

        # `enable_lies` and the lie bounds were resolved from config.json in __init__; do not
        # re-derive them here, or the action space could disagree with what the wrapper was
        # told at construction time.

        # The adversary was trained on the same per-drone observation as everyone else.
        policy_obs_dim = self.obs_dim + (EGO_AUG_DIM if self.ego_state_obs else 0)
        low = np.full(policy_obs_dim, -np.inf, dtype=np.float32)
        obs_space = gym.spaces.Dict(dict(
            obs=gym.spaces.Box(low, -low, dtype=np.float32)))
        act_low = np.asarray(self.action_low, dtype=np.float32).reshape(-1)
        act_high = np.asarray(self.action_high, dtype=np.float32).reshape(-1)
        self.motor_dim = act_low.size
        if self.enable_lies:
            # Must match FrozenTeamEnv's extended space exactly, or the checkpoint's final
            # layer will not load.
            ld = lie_action_dim(self.lie_mode)
            act_low = np.concatenate([act_low, np.full(ld, -1.0, dtype=np.float32)])
            act_high = np.concatenate([act_high, np.full(ld, 1.0, dtype=np.float32)])
        act_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        self._policy = FrozenTeamPolicy(cfg, obs_space, act_space, ckpt_dir,
                                        device=self.device,
                                        checkpoint_file=self.checkpoint_file)
        self._rnn_size = get_rnn_size(cfg)
        self._torch = torch
        self._reset_rnn()

    def _reset_rnn(self):
        self._rnn = self._torch.zeros([1, self._rnn_size], dtype=self._torch.float32,
                                      device=self._policy.device)

    # ------------------------------------------------------------------ agent API

    def reset(self):
        super().reset()
        self._pending_lie = None
        if getattr(self, '_plausible', None) is not None:
            self._plausible.reset()
        if self._policy is not None:
            self._reset_rnn()

    def perturb_action(self, honest_action, obs, t):
        """
        Ignores `honest_action` -- the adversary is a policy, not a perturbation.

        Exception: a `lie_only` round-3 adversary flies the honest action bit-for-bit and
        attacks purely through what it broadcasts, exactly as Tier 2C's `liar` does. The
        broadcast half of the policy output is stashed for `perturb_broadcast_state`, which
        the wrapper calls after the env has stepped.
        """
        if obs is None:
            self._pending_lie = None
            return self.clip(np.asarray(honest_action, dtype=np.float64))

        self.obs_dim = int(np.size(obs))
        self._ensure_policy()

        policy_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.ego_state_obs:
            ego = getattr(self, 'ego', None)
            if ego is None:
                raise RuntimeError(
                    'this adversary was trained with --adv_ego_state_obs but the wrapper did '
                    'not supply an EgoState; needs_ego_state must be True before construction')
            policy_obs = augment_obs(policy_obs, ego.pos, ego.vel,
                                     max_pos=self.max_lie_pos, max_vel=self.max_lie_vel)
        action, self._rnn = self._policy.act(
            policy_obs.reshape(1, -1), self._rnn,
            deterministic=self.deterministic)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        self.steps_acted += 1

        if self.enable_lies:
            self._pending_lie = action[self.motor_dim:]
            action = action[:self.motor_dim]
        if self.lie_only:
            return np.array(honest_action, dtype=np.float64, copy=True)
        return self.clip(action)

    def perturb_broadcast_state(self, true_state, t):
        """
        What this drone claims its position and velocity are.

        Decoded from the policy output cached by `perturb_action` in the same control cycle,
        through the same `decode_lie` the training env uses -- sharing the decoder is what
        makes "trained at f, evaluated at f" true rather than approximately true. A
        motors-only checkpoint has no lie half and broadcasts the truth unchanged, so
        rounds 1-2 evaluate exactly as they did before this capability existed.
        """
        if not self.enable_lies or self._pending_lie is None:
            return np.asarray(true_state, dtype=np.float64)
        self.lies_told += 1
        if getattr(self, 'lie_mode', 'free') == 'plausible':
            if self._plausible is None:
                ego = getattr(self, 'ego', None)
                if ego is None or ego.room_box is None or ego.control_dt is None:
                    raise RuntimeError('plausible liar needs EgoState.room_box and .control_dt; '
                                       'the wrapper must push ego state before the first step')
                self._plausible = PlausibleLie(ego.room_box, ego.control_dt,
                                               **self.plausible_params)
            return self._plausible.step(self._pending_lie, true_state)
        return decode_lie(self._pending_lie, true_state,
                          max_pos=self.max_lie_pos, max_vel=self.max_lie_vel)

    def diagnostics(self):
        return dict(checkpoint_file=os.path.basename(self.checkpoint_file),
                    checkpoint_steps=float(self.checkpoint_steps),
                    steps_acted=float(self.steps_acted),
                    deterministic=self.deterministic,
                    enable_lies=self.enable_lies, lie_only=self.lie_only,
                    lie_mode=self.lie_mode,
                    ego_state_obs=self.ego_state_obs,
                    lies_told=float(self.lies_told))

    # `obs_dim` is discovered from the first observation, so the agent does not need to be
    # told the env's observation size at construction time.
    obs_dim = 54


# The horizon sweep's checkpoint fractions (of each run's CONFIGURED length). With a 200M run
# they give exactly the original 6 25 50 75 100 150 194. The last one is "the final checkpoint"
# that the confirmatory, transfer and matched-triple rules all use (tier3_design 6.22).
HORIZON_FRACTIONS = (0.03, 0.125, 0.25, 0.375, 0.5, 0.75, 0.97)


def horizon_milestones_m(run_dir):
    """Horizon sweep points for a run, in millions of env steps."""
    import json as _json
    with open(os.path.join(run_dir, 'config.json')) as f:
        top = float(_json.load(f).get('train_for_env_steps', 200_000_000))
    return [max(1, int(round(fr * top / 1e6))) for fr in HORIZON_FRACTIONS]


def final_milestone_m(run_dir):
    """The final checkpoint by the pre-registered rule: the last horizon point."""
    return horizon_milestones_m(run_dir)[-1]
