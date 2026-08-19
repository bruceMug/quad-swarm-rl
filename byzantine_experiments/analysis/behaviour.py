"""
Behavioural characterisation of Tier 1 faults — "what does the degraded drone actually
look like?"

The design doc needs a plain-language answer (does it wobble? drift? freeze?) that feeds the
paper's qualitative section. Rather than eyeballing renders, this derives the answer from
the trajectories with named descriptors, so the description is reproducible and the paper
can cite numbers:

    jitter          mean ||v_{t+1} - v_t||       -- high-frequency shake
    action_jitter   mean ||a_{t+1} - a_t||       -- how twitchy the emitted control is
    action_std      std of the emitted action    -- LOW means frozen/stuck
    goal_error      mean ||pos - goal||          -- how lost it is
    excursion_max   max ||pos - goal||           -- worst departure
    persistence     lag-50 (0.5 s) autocorrelation of the goal-error direction
                    -- HIGH means it drifts steadily one way (OU), LOW means it
                       jitters in place (white noise)
    speed           mean ||v||
    z_mean/z_min    altitude, and the fraction of time below 0.15 m (crash proxy)

Each descriptor is reported for the compromised drone, for the honest drones in the same
run, and for the clean reference — so "abnormal" is always relative to something.

Usage:
    python -m byzantine_experiments.analysis.behaviour --plots --wandb
"""

import argparse
import glob
import json
import os

import numpy as np

RESULTS_DIR = 'byzantine_experiments/results/tier1'
FIG_DIR = 'byzantine_experiments/results/figures'
OUT_JSON = 'byzantine_experiments/results/tier1_behaviour.json'
OUT_JSON_BY_PREFIX = {'t1': OUT_JSON, 't2': 'byzantine_experiments/results/tier2_behaviour.json'}

PERSISTENCE_LAG = 50   # steps = 0.5 s at the env's 100 Hz control rate
FLOOR_Z = 0.15


def _valid(data):
    lengths = np.asarray(data['episode_lengths'])
    t = np.arange(data['rewards'].shape[1])[None, :]
    return t < lengths[:, None]


def _agent_descriptors(positions, velocities, actions, obs, mask, agent_sel):
    """Descriptors for the subset of agents selected by `agent_sel` (a bool array)."""
    if not agent_sel.any():
        return {k: float('nan') for k in
                ('jitter', 'action_jitter', 'action_std', 'goal_error', 'excursion_max',
                 'persistence', 'speed', 'z_mean', 'z_min', 'floor_frac')}

    pos = positions[:, :, agent_sel, :]
    vel = velocities[:, :, agent_sel, :]
    act = actions[:, :, agent_sel, :]
    goal_vec = obs[:, :, agent_sel, 0:3]          # pos - goal, straight from the obs
    m = mask[:, :, None]

    def mean_over(x):                              # x: (E, T, A) -> scalar
        sel = np.broadcast_to(m, x.shape)
        vals = x[sel]
        return float(np.mean(vals)) if vals.size else float('nan')

    # Step-to-step differences: only valid where both t and t+1 are real steps.
    dmask = mask[:, :-1] & mask[:, 1:]
    def mean_diff(x):                              # x: (E, T, A, D)
        d = np.linalg.norm(np.diff(x, axis=1), axis=-1)      # (E, T-1, A)
        sel = np.broadcast_to(dmask[:, :, None], d.shape)
        vals = d[sel]
        return float(np.mean(vals)) if vals.size else float('nan')

    goal_err = np.linalg.norm(goal_vec, axis=-1)               # (E, T, A)

    # Directional persistence: correlation of the goal-error unit vector with itself at a
    # 0.5 s lag. Distinguishes "drifting away steadily" from "shaking about in place".
    norms = np.linalg.norm(goal_vec, axis=-1, keepdims=True)
    unit = np.divide(goal_vec, norms, out=np.zeros_like(goal_vec), where=norms > 1e-9)
    if unit.shape[1] > PERSISTENCE_LAG:
        dots = np.sum(unit[:, :-PERSISTENCE_LAG] * unit[:, PERSISTENCE_LAG:], axis=-1)
        pmask = mask[:, :-PERSISTENCE_LAG] & mask[:, PERSISTENCE_LAG:]
        sel = np.broadcast_to(pmask[:, :, None], dots.shape)
        persistence = float(np.mean(dots[sel])) if sel.any() else float('nan')
    else:
        persistence = float('nan')

    z = pos[..., 2]
    act_sel = np.broadcast_to(m[..., None], act.shape)
    return dict(
        jitter=mean_diff(vel),
        action_jitter=mean_diff(act),
        action_std=float(np.std(act[act_sel])) if act_sel.any() else float('nan'),
        goal_error=mean_over(goal_err),
        excursion_max=float(np.max(np.where(np.broadcast_to(m, goal_err.shape),
                                            goal_err, -np.inf))),
        persistence=persistence,
        speed=mean_over(np.linalg.norm(vel, axis=-1)),
        z_mean=mean_over(z),
        z_min=float(np.min(np.where(np.broadcast_to(m, z.shape), z, np.inf))),
        floor_frac=mean_over((z < FLOOR_Z).astype(float)),
    )


def describe_run(npz_path):
    """Compute compromised / honest descriptors for one rollout file."""
    with np.load(npz_path) as data:
        d = {k: np.asarray(data[k]) for k in data.files}
    mask = _valid(d)
    byz = d['byzantine_mask'].astype(bool)

    out = dict(
        byzantine=_agent_descriptors(d['positions'], d['velocities'],
                                     d['actions_executed'], d['obs'], mask, byz),
        honest=_agent_descriptors(d['positions'], d['velocities'],
                                  d['actions_executed'], d['obs'], mask, ~byz),
    )
    return out


def plain_language(byz, honest, clean):
    """
    Turn descriptors into the sentence the design doc needs.

    Thresholds are relative to the clean reference, so they carry over if the baseline
    policy is retrained. They are heuristics for describing behaviour, not detectors.
    """
    words = []
    ratio = lambda k: (byz[k] / clean[k]) if clean.get(k) and np.isfinite(clean[k]) and clean[k] > 1e-9 else float('nan')

    jit, ajit, err = ratio('jitter'), ratio('action_jitter'), ratio('goal_error')
    astd = ratio('action_std')

    if np.isfinite(ajit) and ajit < 0.5:
        words.append('emits an unusually smooth/stale control signal (frozen output)')
    elif np.isfinite(ajit) and ajit > 2.0:
        words.append('emits a visibly twitchy control signal')

    if np.isfinite(jit) and jit > 1.5:
        words.append(f'wobbles ({jit:.1f}x the clean per-step velocity change)')
    elif np.isfinite(jit) and jit < 0.7:
        words.append('moves unusually sluggishly')

    if np.isfinite(err) and err > 1.3:
        if byz['persistence'] > 0.7:
            words.append(f'drifts persistently away from its goal '
                         f'({err:.1f}x clean goal error, direction autocorrelation '
                         f'{byz["persistence"]:.2f} at 0.5 s)')
        else:
            words.append(f'strays around its goal without a consistent direction '
                         f'({err:.1f}x clean goal error, direction autocorrelation '
                         f'{byz["persistence"]:.2f})')

    if byz['floor_frac'] > 5 * max(clean['floor_frac'], 1e-4):
        words.append(f'spends {100 * byz["floor_frac"]:.1f}% of the time near the floor '
                     f'(vs {100 * clean["floor_frac"]:.1f}% clean)')

    if np.isfinite(astd) and astd < 0.6:
        words.append('with reduced control authority')

    if not words:
        words.append('is behaviourally indistinguishable from a clean drone at this severity')

    honest_err = (honest['goal_error'] / clean['goal_error']) if clean['goal_error'] > 1e-9 else float('nan')
    team = (f'Meanwhile its honest neighbours hold formation ({honest_err:.2f}x clean goal error).'
            if np.isfinite(honest_err) and honest_err < 1.15
            else f'Its neighbours are also affected ({honest_err:.2f}x clean goal error).')

    return 'The degraded drone ' + '; '.join(words) + '. ' + team


def make_plots(configs, results_dir, fig_dir, tier_label='Tier 1'):
    """Trajectory / goal-error / action figures for one representative run per variant."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    paths = []

    for label, npz_path in configs:
        if not os.path.exists(npz_path):
            continue
        with np.load(npz_path) as data:
            pos = np.asarray(data['positions'])
            obs = np.asarray(data['obs'])
            act = np.asarray(data['actions_executed'])
            hon = np.asarray(data['actions_honest'])
            byz = np.asarray(data['byzantine_mask']).astype(bool)
            lengths = np.asarray(data['episode_lengths'])

        b = int(np.argmax(byz))
        h = int(np.argmax(~byz))
        n_show = min(5, len(lengths))

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        fig.suptitle(f'{tier_label} behaviour — {label}\n'
                     f'(drone {b} compromised, drone {h} shown as honest control)',
                     fontsize=12)

        ax = axes[0, 0]
        for e in range(n_show):
            T = lengths[e]
            ax.plot(pos[e, :T, b, 0], pos[e, :T, b, 1], lw=0.8, alpha=0.8)
        ax.plot(pos[0, :lengths[0], h, 0], pos[0, :lengths[0], h, 1],
                'k--', lw=1.2, alpha=0.7, label=f'honest drone {h} (ep 0)')
        ax.set_title(f'XY trajectory, {n_show} episodes'); ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]'); ax.legend(fontsize=8); ax.set_aspect('equal', 'datalim')

        ax = axes[0, 1]
        for e in range(n_show):
            T = lengths[e]
            ax.plot(np.arange(T) / 100.0, pos[e, :T, b, 2], lw=0.8, alpha=0.8)
        ax.plot(np.arange(lengths[0]) / 100.0, pos[0, :lengths[0], h, 2],
                'k--', lw=1.2, alpha=0.7, label=f'honest drone {h}')
        ax.axhline(FLOOR_Z, color='r', ls=':', lw=1, label='floor threshold')
        ax.set_title('Altitude'); ax.set_xlabel('t [s]'); ax.set_ylabel('z [m]')
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        err_b = np.linalg.norm(obs[..., 0:3], axis=-1)
        T = int(lengths.min())
        t = np.arange(T) / 100.0
        for idx, (name, style) in ((b, ('compromised', '-')), (h, ('honest', '--'))):
            m = err_b[:, :T, idx].mean(axis=0)
            s = err_b[:, :T, idx].std(axis=0)
            ax.plot(t, m, style, lw=1.2, label=f'{name} (drone {idx})')
            ax.fill_between(t, m - s, m + s, alpha=0.15)
        ax.set_title('Distance to goal (mean ± std over episodes)')
        ax.set_xlabel('t [s]'); ax.set_ylabel('||pos − goal|| [m]'); ax.legend(fontsize=8)

        ax = axes[1, 1]
        seg = slice(0, min(400, int(lengths[0])))
        ax.plot(np.arange(seg.stop) / 100.0, hon[0, seg, b, 0], lw=1.0,
                label='honest action (motor 0)')
        ax.plot(np.arange(seg.stop) / 100.0, act[0, seg, b, 0], lw=1.0, alpha=0.8,
                label='executed action (motor 0)')
        ax.axhline(1.0, color='r', ls=':', lw=0.8)
        ax.axhline(-1.0, color='r', ls=':', lw=0.8, label='clip boundary')
        ax.set_title('Action trace, first 4 s of episode 0')
        ax.set_xlabel('t [s]'); ax.set_ylabel('action'); ax.legend(fontsize=8)

        fig.tight_layout()
        out = os.path.join(fig_dir, f'behaviour_{label}.png')
        fig.savefig(out, dpi=130)
        plt.close(fig)
        paths.append(out)
        print(f'  wrote {out}')

    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--results-dir', default=RESULTS_DIR)
    p.add_argument('--clean-dir', default=None,
                   help='Where the clean reference lives (default: --results-dir). Tier 2 '
                        'reuses the Tier 1 clean rollouts, so point this at results/tier1.')
    p.add_argument('--prefix', default='t1', help="Run-name prefix: 't1' or 't2'")
    p.add_argument('--pattern', default=None,
                   help='Glob for the runs to characterise (default: <prefix>_*_s<seed>_*f0125)')
    p.add_argument('--fig-dir', default=FIG_DIR)
    p.add_argument('--plots', action='store_true')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--seed', type=int, default=1, help='Which baseline seed to characterise')
    args = p.parse_args()

    clean_dir = args.clean_dir or args.results_dir
    clean_path = os.path.join(clean_dir, f't1_clean_s{args.seed}.npz')
    if not os.path.exists(clean_path):
        raise SystemExit(f'Missing clean reference {clean_path}')
    clean = describe_run(clean_path)['honest']

    pattern = args.pattern or f'{args.prefix}_*_s{args.seed}_*f0125*'
    runs = sorted(glob.glob(os.path.join(args.results_dir, pattern + '.npz'))
                  if not pattern.endswith('.npz') else
                  glob.glob(os.path.join(args.results_dir, pattern)))
    runs = [r for r in runs if '_clean_' not in os.path.basename(r)]
    print(f'Characterising {len(runs)} runs against {clean_path}\n')

    report = {'clean': clean, 'clean_path': clean_path, 'configs': {}}
    for path in runs:
        label = os.path.splitext(os.path.basename(path))[0]
        d = describe_run(path)
        text = plain_language(d['byzantine'], d['honest'], clean)
        report['configs'][label] = dict(descriptors=d, description=text)
        print(f'{label}\n    {text}\n')

    out_json = OUT_JSON_BY_PREFIX.get(args.prefix, OUT_JSON)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f'Wrote {out_json}')

    fig_paths = []
    if args.plots:
        reps = [(os.path.splitext(os.path.basename(p))[0], p) for p in runs]
        tier_label = {'t1': 'Tier 1', 't2': 'Tier 2'}.get(args.prefix, args.prefix)
        fig_paths = make_plots(reps, args.results_dir, args.fig_dir, tier_label=tier_label)

    if args.wandb:
        try:
            import wandb
            from byzantine_experiments.analysis.wandb_logging import DEFAULT_ENTITY, DEFAULT_PROJECT
            run = wandb.init(project=DEFAULT_PROJECT, entity=DEFAULT_ENTITY,
                             name=f'{args.prefix}_behaviour_s{args.seed}',
                             group=f'tier{args.prefix[-1]}_analysis',
                             job_type='analysis',
                             tags=[f'tier{args.prefix[-1]}', 'behaviour', 'analysis'])
            table = wandb.Table(columns=['config', 'description'] +
                                        [f'byz_{k}' for k in clean] +
                                        [f'honest_{k}' for k in clean])
            for label, entry in report['configs'].items():
                d = entry['descriptors']
                table.add_data(label, entry['description'],
                               *[float(d['byzantine'][k]) for k in clean],
                               *[float(d['honest'][k]) for k in clean])
            run.log({'behaviour': table})
            for path in fig_paths:
                run.log({f'figures/{os.path.basename(path)}': wandb.Image(path)})
            print(f'wandb: {run.get_url()}')
            run.finish()
        except Exception as exc:
            print(f'[wandb] failed: {type(exc).__name__}: {exc}')


if __name__ == '__main__':
    main()
