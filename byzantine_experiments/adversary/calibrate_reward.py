"""
Reward-scale calibration for Tier 3 — run this BEFORE launching a training run.

The adversarial reward sums four terms with hand-chosen weights. If one term's
episode-integrated magnitude dwarfs the others, PPO optimises that term and ignores the rest,
and you find out 12-24 GPU-hours later. This measures the balance in a few seconds by driving
the frozen-team env with random actions and integrating each component.

It was not a hypothetical concern: at the spec's literal weights (`w_crash=10` charged every
step the adversary is in a crashed state) the crash term outweighs the objective **2622 : 1**,
which would have produced an adversary that learns "hover, never crash" and nothing else.

Usage:
    python -m byzantine_experiments.adversary.calibrate_reward
    python -m byzantine_experiments.adversary.calibrate_reward --w-team 5 --w-crash 10
    python -m byzantine_experiments.adversary.calibrate_reward --crash-penalty-mode per_step
"""

import argparse
import glob
import json

import numpy as np

EPISODE_STEPS = 1501     # 15 s at 100 Hz


def measure(adv_indices=(0,), n_steps=600, amplitude=0.3, seed=4, checkpoint=None, **reward_kwargs):
    """Drive the env with random actions and integrate each reward component."""
    from byzantine_experiments.tests.test_tier3 import build_adv_env, drive

    env, _ = build_adv_env(adv_indices=adv_indices, **reward_kwargs)
    env.reset()

    rows = []
    for _, _, _, infos in drive(env, n_steps, seed=seed, amplitude=amplitude):
        rows.append(infos[0]['adv_reward_components'])
    env.close()
    if not rows:
        raise RuntimeError('no steps collected')

    scale = EPISODE_STEPS / len(rows)
    out = {}
    for key in ('team', 'crash', 'alive', 'stealth'):
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = dict(
            per_step_mean=float(vals.mean()),
            per_step_std=float(vals.std()),
            episode_abs=float(np.abs(vals).sum() * scale),
        )
    out['steps'] = len(rows)
    out['crash_onsets'] = float(rows[-1].get('crash_onsets', 0)) * scale
    out['crashed_frac'] = float(np.mean([r['adv_crashed'] for r in rows]))
    out['mean_action_deviation'] = float(np.mean([r['action_deviation'] for r in rows]))
    return out


def report(stats, weights):
    print(f"\ndriven {stats['steps']} steps  |  adversary crashed {100*stats['crashed_frac']:.0f}% "
          f"of steps, {stats['crash_onsets']:.0f} crash onsets/episode\n")
    print(f"{'term':10s} {'per-step mean':>14s} {'per-step std':>13s} {'|episode total|':>16s}  share")
    print('-' * 66)
    total = sum(stats[k]['episode_abs'] for k in ('team', 'crash', 'alive', 'stealth')) or 1.0
    for k in ('team', 'crash', 'alive', 'stealth'):
        s = stats[k]
        print(f"{k:10s} {s['per_step_mean']:14.5f} {s['per_step_std']:13.5f} "
              f"{s['episode_abs']:16.2f}  {100*s['episode_abs']/total:5.1f}%")

    team = stats['team']['episode_abs'] or 1e-9
    print()
    for k in ('crash', 'alive', 'stealth'):
        ratio = stats[k]['episode_abs'] / team
        flag = ''
        if ratio > 100:
            flag = '  <-- DROWNS THE OBJECTIVE'
        elif ratio > 20:
            flag = '  <-- dominant'
        print(f"  {k:8s} : team  =  {ratio:9.1f} : 1{flag}")

    print(f"\nweights: {json.dumps(weights)}")
    worst = max(stats[k]['episode_abs'] / team for k in ('crash', 'alive', 'stealth'))
    if worst > 100:
        print('\nVERDICT: unusable. One term is >100x the objective; PPO will optimise it and\n'
              '         ignore team damage. Reduce that weight or change crash_penalty_mode.')
    elif worst > 20:
        print('\nVERDICT: workable but skewed. The objective is visible but a secondary term\n'
              '         still dominates; expect slow learning on team damage.')
    else:
        print('\nVERDICT: balanced enough to train.')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    # Defaults come from reward.DEFAULTS so the calibrator always measures the SAME
    # configuration training will actually use. Hardcoding them here once let the two drift
    # apart, and the calibrator silently reported on a config nobody would run.
    from byzantine_experiments.adversary.reward import DEFAULTS
    p.add_argument('--w-team', type=float, default=DEFAULTS['w_team'])
    p.add_argument('--w-crash', type=float, default=DEFAULTS['w_crash'])
    p.add_argument('--w-alive', type=float, default=DEFAULTS['w_alive'])
    p.add_argument('--w-stealth', type=float, default=DEFAULTS['w_stealth'])
    p.add_argument('--epsilon', type=float, default=DEFAULTS['epsilon'])
    p.add_argument('--crash-penalty-mode', default='onset', choices=['onset', 'per_step'])
    p.add_argument('--team-reward-mode', default='honest', choices=['honest', 'all'])
    p.add_argument('--n-steps', type=int, default=600)
    p.add_argument('--use-clean-mu', action='store_true',
                   help='Load mu_honest from the clean baseline rollouts')
    args = p.parse_args()

    mu = None
    if args.use_clean_mu:
        from byzantine_experiments.adversary.reward import honest_action_stats
        paths = sorted(glob.glob('byzantine_experiments/results/tier1/t1_clean_s*.npz'))
        if paths:
            stats = honest_action_stats(paths[:1])
            mu = stats['mu_honest']
            print(f"mu_honest from clean rollouts: {np.round(mu, 4)}  "
                  f"(mean honest ||a|| = {stats['mean_action_norm']:.3f})")

    weights = dict(w_team=args.w_team, w_crash=args.w_crash, w_alive=args.w_alive,
                   w_stealth=args.w_stealth, epsilon=args.epsilon,
                   crash_penalty_mode=args.crash_penalty_mode,
                   team_reward_mode=args.team_reward_mode)
    stats = measure(n_steps=args.n_steps, mu_honest=mu, **weights)
    report(stats, weights)


if __name__ == '__main__':
    main()
