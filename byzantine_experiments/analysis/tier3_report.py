"""
Tier 3 design-document items that need figures or accounting -- spec "Required output" 2, 6, 7, 8.

    rewards    (2) every reward component over training, per adversary, from TensorBoard
    compute    (8) GPU-hours per training run, from TensorBoard wall times
    stealth    (6) action distribution adversary vs honest (histogram overlay) and two naive
               detectors -- an action-magnitude detector and a broadcast physics check -- with
               their detection and false-alarm rates
    behaviour  (7) 10 episodes plotted, plus the numbers the plain-language description of the
               attack has to be checked against

Everything is written under results_v2/tier3_report/ and logged to wandb (group tier3_report)
unless --no-wandb.

    python -m byzantine_experiments.analysis.tier3_report rewards              # all adversaries
    python -m byzantine_experiments.analysis.tier3_report compute
    python -m byzantine_experiments.analysis.tier3_report stealth   --run adv_seed3_stealth0_lieonly
    python -m byzantine_experiments.analysis.tier3_report behaviour --run adv_seed3_stealth0_lieonly

Reading TensorBoard is the slow part (~300 MB of summaries per run): the first `rewards` or
`compute` call caches each run's scalars in tier3_report/curves/<run>.npz; later calls read the
cache (`--refresh` rebuilds it, e.g. after a run trains further).

stealth / behaviour use the adversary's own-seed confirmatory run (n = 200) and the clean run at
that seed, after verifying they are episode-paired. Broadcasts come from the recorded
`broadcast` arrays when the run has them (every liar collected since 2026-09-26). Older runs are
**reconstructed**: the adversary is a feed-forward policy evaluated deterministically, so the lie
on each step is a pure function of that step's recorded observation and true state, and is
recomputed through the same decoder the rollout used. Reconstruction refuses recurrent policies.

Naive-detector thresholds are always set on the CLEAN run (every drone honest), as the maximum
(or p99 for the per-episode action statistic) an honest drone ever shows -- so a threshold never
sees attacked data, and false alarms on clean flight are zero or ~1 % by construction. False
alarms are then measured on the honest drones of the attacked run.
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

from byzantine_experiments.paths import RESULTS_ROOT

OUT_DIR = os.path.join(RESULTS_ROOT, 'tier3_report')
CURVE_DIR = os.path.join(OUT_DIR, 'curves')
TRAIN_DIR = 'train_dir'
REWARD_KEYS = ('adv/r_team', 'adv/r_crash', 'adv/r_alive', 'adv/r_stealth')
EXTRA_KEYS = ('adv/survived', 'len/len', 'adv/lie_magnitude_m', 'adv/lie_wall_limited_frac')
GAP_S = 15 * 60          # a pause longer than this between logged events is downtime, not compute
EPISODES_PLOTTED = 10


def _runs():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(TRAIN_DIR, 'adv_seed*'))
                  if os.path.exists(os.path.join(p, 'config.json')))


def _cfg(run):
    with open(os.path.join(TRAIN_DIR, run, 'config.json')) as f:
        return json.load(f)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


# ------------------------------------------------------------------ TensorBoard cache (2, 8)

def load_curves(run, refresh=False):
    """{key: (steps, values, wall_times)} for the reward/extra keys, cached per run."""
    cache = os.path.join(CURVE_DIR, run + '.npz')
    src = os.path.join(TRAIN_DIR, run, '.summary', '0')
    if (not refresh and os.path.exists(cache) and os.path.isdir(src)
            and os.path.getmtime(cache) >= max(os.path.getmtime(p) for p in glob.glob(src + '/*'))):
        with np.load(cache) as z:
            keys = sorted({k.rsplit('__', 1)[0] for k in z.files})
            return {k.replace('|', '/'): (z[k + '__s'], z[k + '__v'], z[k + '__w']) for k in keys}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(src, size_guidance={'scalars': 20000, 'histograms': 0, 'images': 0,
                                              'audio': 0, 'tensors': 0, 'compressedHistograms': 0})
    ea.Reload()
    have = set(ea.Tags()['scalars'])
    out = {}
    for k in REWARD_KEYS + EXTRA_KEYS:
        if k in have:
            ev = ea.Scalars(k)
            out[k] = (np.array([e.step for e in ev], np.float64),
                      np.array([e.value for e in ev], np.float64),
                      np.array([e.wall_time for e in ev], np.float64))
    os.makedirs(CURVE_DIR, exist_ok=True)
    flat = {}
    for k, (s, v, w) in out.items():
        kk = k.replace('/', '|')
        flat[kk + '__s'], flat[kk + '__v'], flat[kk + '__w'] = s, v, w
    np.savez(cache, **flat)
    return out


def _binned(steps, values, n_bins=40):
    from byzantine_experiments.analysis.training_curve import binned
    return binned(steps, values, n_bins)


def rewards(runs, refresh=False):
    """Item 2: one figure per adversary, one panel per reward component; plus a summary table."""
    plt = _plt()
    os.makedirs(OUT_DIR, exist_ok=True)
    L = ['# Tier 3 reward components over training (spec item 2)\n',
         'Binned means over env steps (40 bins). `first` / `last` are the first and last 10 % of '
         'training. `r_team` is an episode sum: read it with `len/len` (training_curve.py).\n',
         '| adversary | component | first 10 % | last 10 % | change |', '|---|---|---|---|---|']
    figs = []
    for run in runs:
        c = load_curves(run, refresh)
        keys = [k for k in REWARD_KEYS + EXTRA_KEYS if k in c]
        if not keys:
            L.append(f'| `{run}` | (no TensorBoard scalars) | | | |')
            continue
        fig, axes = plt.subplots(2, (len(keys) + 1) // 2, figsize=(4 * ((len(keys) + 1) // 2), 6),
                                 squeeze=False)
        for ax, k in zip(axes.ravel(), keys):
            s, v, _ = c[k]
            x, m, _ = _binned(s, v)
            ax.plot(s / 1e6, v, ',', alpha=0.15, color='grey')
            ax.plot(x / 1e6, m, '-', color='C0' if k in REWARD_KEYS else 'C2', lw=2)
            ax.set_title(k)
            ax.set_xlabel('env steps (M)')
            lo, hi = np.percentile(v, [1, 99])
            ax.set_ylim(lo - 0.05 * (hi - lo + 1e-9), hi + 0.05 * (hi - lo + 1e-9))
            first = v[s <= s.min() + 0.1 * (s.max() - s.min())].mean()
            last = v[s >= s.max() - 0.1 * (s.max() - s.min())].mean()
            L.append(f'| `{run}` | `{k}` | {first:+.3f} | {last:+.3f} | {last - first:+.3f} |')
        for ax in axes.ravel()[len(keys):]:
            ax.axis('off')
        fig.suptitle(run)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f'rewards_{run}.png')
        fig.savefig(path, dpi=110)
        plt.close(fig)
        figs.append(path)
    md = os.path.join(OUT_DIR, 'rewards.md')
    with open(md, 'w') as f:
        f.write('\n'.join(L) + '\n\n' + '\n'.join(f'![]({os.path.basename(p)})' for p in figs) + '\n')
    print('\n'.join(L))
    return md, figs


# ------------------------------------------------------------------ compute accounting (8)

def active_hours(wall):
    """Wall-clock hours with logged activity: pauses longer than GAP_S are downtime."""
    w = np.sort(np.asarray(wall, np.float64))
    if w.size < 2:
        return 0.0, 0.0
    d = np.diff(w)
    return float(d[d <= GAP_S].sum() / 3600.0), float((w[-1] - w[0]) / 3600.0)


def compute(runs, refresh=False):
    """Item 8: GPU-hours per training run, total, and wall clock (one Tesla T4)."""
    rows = []
    for run in runs:
        c = load_curves(run, refresh)
        key = 'len/len' if 'len/len' in c else next(iter(c), None)
        if key is None:
            rows.append(dict(run=run, active_h=float('nan'), span_h=float('nan'), steps_m=float('nan'),
                             start=float('nan'), end=float('nan')))
            continue
        s, _, w = c[key]
        act, span = active_hours(w)
        rows.append(dict(run=run, active_h=act, span_h=span, steps_m=float(s.max() / 1e6),
                         start=float(w.min()), end=float(w.max())))
    # Overlap: runs that trained at the same time shared the one GPU, so their GPU-hours
    # overstate the device time used; the union of spans is the true device wall clock.
    iv = sorted((r['start'], r['end']) for r in rows if np.isfinite(r['start']))
    union, cur = 0.0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur:
                union += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        union += cur[1] - cur[0]
    L = ['# Tier 3 compute accounting (spec item 8)\n',
         'One Tesla T4 (sequential or shared), 32 CPU cores. **Active** hours exclude pauses '
         f'longer than {GAP_S // 60} min between logged episodes (stops, resumes); **span** is first '
         'to last logged episode. GPU-hours = active hours x 1 GPU; runs that overlapped in time '
         'shared the device, so their sum is an upper bound on device time -- the union of spans '
         'is the device wall clock.\n',
         '| adversary | env steps (M) | active h (GPU-h) | span h | M steps / active h |',
         '|---|---|---|---|---|']
    for r in rows:
        rate = r['steps_m'] / r['active_h'] if r['active_h'] else float('nan')
        L.append(f"| `{r['run']}` | {r['steps_m']:.0f} | {r['active_h']:.1f} | {r['span_h']:.1f} "
                 f"| {rate:.2f} |")
    tot = np.nansum([r['active_h'] for r in rows])
    L.append(f'| **total** | {np.nansum([r["steps_m"] for r in rows]):.0f} | **{tot:.0f}** | '
             f'device wall clock (union of spans) {union / 3600:.0f} h | |')
    n_eps = _evaluation_episodes()
    L.append(f'\nEvaluation (rollouts, CPU-bound): {n_eps:,} episodes of Tier 3 runs and their clean and '
             'hand-designed references collected in '
             f'`{RESULTS_ROOT}` (sweeps, confirmatory, transfer, triple). At the measured '
             f'~20 episodes/min this is ~{n_eps / 20 / 60:.0f} h of node time -- an estimate from '
             'throughput, not a measurement.')
    md = os.path.join(OUT_DIR, 'compute.md')
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(md, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    return md, rows


def _evaluation_episodes():
    n = 0
    for p in glob.glob(os.path.join(RESULTS_ROOT, '**', '*.json'), recursive=True):
        base = os.path.basename(p)
        if not (base.startswith('t3_') or base.startswith('confirm_') or 'triple' in p):
            continue
        try:
            with open(p) as f:
                n += int(json.load(f).get('n_episodes', 0))
        except (ValueError, OSError, AttributeError):
            continue
    return n


# ------------------------------------------------------------------ paired run + broadcasts

def confirm_pair(run):
    """(attacked npz, clean npz, adversary seed) for the own-seed n = 200 run, verified paired."""
    import re
    from byzantine_experiments.analysis.paired import check_pairing
    from byzantine_experiments.analysis.transfer_matrix import CONFIRM_DIR, cell_path
    aseed = int(re.search(r'seed(\d+)', run).group(1))
    path, _ = cell_path(run, aseed, 200)
    clean = os.path.join(CONFIRM_DIR, f'confirm_clean_s{aseed}_n200.npz')
    for p in (path, clean):
        if not os.path.exists(p):
            raise SystemExit(f'missing {p} -- run scripts/byz_transfer.sh for {run} first')
    check_pairing(path, clean)
    return path, clean, aseed


def room_box(cfg):
    d = np.asarray(cfg.get('quads_room_dims', [10.0, 10.0, 10.0]), np.float64)
    return np.array([[-d[0] / 2, -d[1] / 2, 0.0], [d[0] / 2, d[1] / 2, d[2]]])


def broadcasts(run, npz, adv, episodes=None):
    """
    (reported, truth, source) for drone `adv`, each (E, T, 6). Recorded when available, else
    reconstructed from the deterministic feed-forward policy (see module docstring).

    Recorded arrays are NaN on each episode's last step. The observation the environment returns
    on its terminal step is already the auto-reset observation of the NEXT episode, so the
    broadcast recorded there is the lie applied to a fresh spawn (up to 6 m from where the drone
    is). No honest drone acts on it within the episode, and keeping it made the plausible liar
    fail the kinematic-consistency check in 199/200 episodes -- every one of them on that step
    alone (2026-10-09). Reconstruction takes truth from the recorded positions, which are
    pre-reset, so it has no such step.
    """
    with np.load(npz) as z:
        eps = np.arange(len(z['episode_lengths'])) if episodes is None else np.asarray(episodes)
        if 'broadcast' in z.files:
            rep = np.asarray(z['broadcast'][eps, :, adv], np.float64)
            truth = np.asarray(z['broadcast_truth'][eps, :, adv], np.float64)
            last = np.asarray(z['episode_lengths'])[eps].astype(int) - 1
            rep[np.arange(len(eps)), last] = np.nan
            truth[np.arange(len(eps)), last] = np.nan
            return rep, truth, 'recorded'
        obs = np.asarray(z['obs'][eps, :, adv], np.float32)
        truth = np.concatenate([z['positions'][eps, :, adv], z['velocities'][eps, :, adv]],
                               axis=-1).astype(np.float64)
        lengths = np.asarray(z['episode_lengths'])[eps]
    return reconstruct(run, obs, truth, lengths), truth, 'reconstructed'


def reconstruct(run, obs, truth, lengths):
    import torch
    from byzantine_experiments.adversary.lies import PlausibleLie, decode_lie
    from byzantine_experiments.agents.tier3_learned import LearnedAdversaryAgent, final_milestone_m
    cfg = _cfg(run)
    if cfg.get('use_rnn'):
        raise SystemExit(f'{run} is recurrent: its lie depends on history and cannot be '
                         'reconstructed from single observations; re-collect with the current '
                         'collector, which records broadcasts')
    if not cfg.get('adv_enable_lies'):
        return truth.copy()
    ag = LearnedAdversaryAgent(0, 4, checkpoint=os.path.join(TRAIN_DIR, run),
                               milestone=final_milestone_m(os.path.join(TRAIN_DIR, run)) * 1_000_000)
    ag.action_low, ag.action_high = -np.ones(4, np.float32), np.ones(4, np.float32)
    ag.obs_dim = obs.shape[-1]
    ag._ensure_policy()
    if getattr(ag, 'ego_state_obs', False):
        raise SystemExit(f'{run} observes its own true state; reconstruction needs the recorded '
                         'broadcast arrays -- re-collect its confirmatory run')
    E, T = obs.shape[:2]
    flat = obs.reshape(E * T, -1)
    acts = []
    with torch.no_grad():
        for i in range(0, len(flat), 50_000):
            a, _ = ag._policy.act(flat[i:i + 50_000], torch.zeros([min(50_000, len(flat) - i),
                                                                  ag._rnn_size]), deterministic=True)
            acts.append(np.asarray(a, np.float64))
    lie = np.concatenate(acts)[:, ag.motor_dim:].reshape(E, T, -1)
    rep = np.full_like(truth, np.nan)
    box = room_box(cfg)
    for e in range(E):
        n = int(lengths[e])
        if getattr(ag, 'lie_mode', 'free') == 'plausible':
            pl = PlausibleLie(box, 1.0 / 100.0, **ag.plausible_params)
            for t in range(n):
                rep[e, t] = pl.step(lie[e, t], truth[e, t])
        else:
            for t in range(n):
                rep[e, t] = decode_lie(lie[e, t], truth[e, t], ag.max_lie_pos, ag.max_lie_vel)
    return rep


def physics(state, lengths, dt):
    """Per-step naive physics statistics of broadcast states (E, T, 6); NaN past episode end."""
    E, T, _ = state.shape
    valid = np.arange(T)[None, :] < np.asarray(lengths)[:, None]
    pos, vel = state[..., :3], state[..., 3:]
    speed = np.where(valid, np.linalg.norm(vel, axis=-1), np.nan)
    acc = np.full((E, T), np.nan)
    acc[:, 1:] = np.linalg.norm(np.diff(vel, axis=1), axis=-1) / dt
    resid = np.full((E, T), np.nan)
    resid[:, 1:] = np.linalg.norm(np.diff(pos, axis=1) - vel[:, 1:] * dt, axis=-1)
    both = valid & np.roll(valid, 1, axis=1)
    both[:, 0] = False
    acc[~both] = np.nan
    resid[~both] = np.nan
    return dict(valid=valid, pos=pos, speed=speed, acc=acc, resid=resid)


def auc(pos_scores, neg_scores):
    """P(score of a random positive > a random negative); 0.5 = the detector sees nothing."""
    p, n = np.asarray(pos_scores), np.asarray(neg_scores)
    u = stats.mannwhitneyu(p, n, alternative='two-sided').statistic
    return float(u / (len(p) * len(n)))


def _drone_state(npz, drones):
    with np.load(npz) as z:
        s = np.concatenate([z['positions'][:, :, drones], z['velocities'][:, :, drones]], -1)
        return np.asarray(s, np.float64), np.asarray(z['episode_lengths'])


# ------------------------------------------------------------------ stealth (6)

def stealth(run, no_wandb=True):
    plt = _plt()
    npz, clean_npz, aseed = confirm_pair(run)
    cfg = _cfg(run)
    box, dt = room_box(cfg), 1.0 / 100.0
    with np.load(npz) as z:
        mask = np.asarray(z['byzantine_mask'])
        a_att = np.asarray(z['actions_executed'], np.float64)
        lengths = np.asarray(z['episode_lengths'])
    with np.load(clean_npz) as z:
        a_cln = np.asarray(z['actions_executed'], np.float64)
        len_c = np.asarray(z['episode_lengths'])
    adv, hon = int(np.flatnonzero(mask)[0]), np.flatnonzero(~mask)
    E, T = a_att.shape[:2]
    valid = np.arange(T)[None, :] < lengths[:, None]
    valid_c = np.arange(a_cln.shape[1])[None, :] < len_c[:, None]

    # --- action-level: the spec's histogram overlay and a naive action-magnitude detector
    norm = np.linalg.norm(a_att, axis=-1)                                   # (E, T, N)
    norm_c = np.linalg.norm(a_cln, axis=-1)
    ep_mean = np.array([[norm[e, :lengths[e], d].mean() for d in range(norm.shape[2])]
                        for e in range(E)])                                  # (E, N)
    ep_mean_c = np.array([[norm_c[e, :len_c[e], d].mean() for d in range(norm_c.shape[2])]
                          for e in range(len(len_c))])
    thr_act = float(np.percentile(ep_mean_c, 99))
    act = dict(ratio=float(norm[..., adv][valid].mean() / norm[..., hon][valid].mean()),
               auc=auc(ep_mean[:, adv], ep_mean[:, hon].ravel()),
               detect=float((ep_mean[:, adv] > thr_act).mean()),
               false_alarm=float((ep_mean[:, hon] > thr_act).mean()),
               threshold=thr_act)
    # Control: the same slot in the clean run. A lie-only adversary flies the honest policy, so
    # any action difference it shows may belong to its slot in the formation, not to the attack.
    act['ratio_clean_slot'] = float(norm_c[..., adv][valid_c].mean() / norm_c[..., hon][valid_c].mean())
    act['auc_clean_slot'] = auc(ep_mean_c[:, adv], ep_mean_c[:, hon].ravel())

    # --- broadcast physics: bounds, speed, kinematic consistency, acceleration
    rep, truth, source = broadcasts(run, npz, adv)
    bvalid = valid & np.isfinite(rep).all(-1) & np.isfinite(truth).all(-1)
    clean_state, _ = _drone_state(clean_npz, np.arange(a_cln.shape[2]))
    pc = [physics(clean_state[:, :, d], len_c, dt) for d in range(clean_state.shape[2])]
    thr = dict(speed=float(np.nanmax([p['speed'] for p in pc])),
               acc=float(np.nanmax([p['acc'] for p in pc])),
               resid=float(np.nanmax([p['resid'] for p in pc])))
    lo, hi = box[0] - 1e-6, box[1] + 1e-6

    def flags(ph):
        v = ph['valid']
        out_room = v & ((ph['pos'] < lo) | (ph['pos'] > hi)).any(-1)
        return dict(bounds=out_room,
                    speed=np.nan_to_num(ph['speed']) > thr['speed'],
                    consistency=np.nan_to_num(ph['resid']) > thr['resid'],
                    accel=np.nan_to_num(ph['acc']) > thr['acc'])

    fa = flags(physics(rep, lengths, dt))
    att_state, _ = _drone_state(npz, hon)
    fh = [flags(physics(att_state[:, :, j], lengths, dt)) for j in range(len(hon))]
    checks = ('bounds', 'speed', 'consistency', 'accel')
    step_rate = {k: float(fa[k][bvalid].mean()) for k in checks}
    ep_detect = {k: float(fa[k].any(1).mean()) for k in checks}
    ep_fa = {k: float(np.mean([f[k].any(1).mean() for f in fh])) for k in checks}
    any_adv = np.any([fa[k].any(1) for k in checks], axis=0)
    any_hon = np.mean([np.any([f[k].any(1) for k in checks], axis=0).mean() for f in fh])
    below_floor = float((bvalid & (rep[..., 2] < box[0, 2] - 1e-6)).sum() / bvalid.sum())
    rep_speed = np.linalg.norm(rep[..., 3:], axis=-1)[bvalid]
    true_speed = np.linalg.norm(truth[..., 3:], axis=-1)[bvalid]
    offset = np.linalg.norm(rep[..., :3] - truth[..., :3], axis=-1)[bvalid]

    # --- figure
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    bins = np.linspace(0, max(np.percentile(norm[valid], 99.5), 1e-3), 60)
    ax[0, 0].hist(norm[..., hon][valid].ravel(), bins, density=True, alpha=.5, label='honest')
    ax[0, 0].hist(norm[..., adv][valid], bins, density=True, alpha=.5, label='adversary')
    ax[0, 0].set_title(f'action norm per step (ratio {act["ratio"]:.2f})')
    ax[0, 0].legend()
    for m in range(a_att.shape[-1]):
        b = np.linspace(-1, 1, 41)
        ax[0, 1].hist(a_att[..., hon, m][valid].ravel(), b, density=True, histtype='step',
                      color=f'C{m}', ls='-', label=f'motor {m} honest')
        ax[0, 1].hist(a_att[..., adv, m][valid], b, density=True, histtype='step',
                      color=f'C{m}', ls='--', label=f'motor {m} adversary')
    ax[0, 1].set_title('per-motor action (solid honest, dashed adversary)')
    ax[0, 1].legend(fontsize=6)
    ax[0, 2].hist(ep_mean[:, hon].ravel(), 40, density=True, alpha=.5, label='honest')
    ax[0, 2].hist(ep_mean[:, adv], 40, density=True, alpha=.5, label='adversary')
    ax[0, 2].axvline(thr_act, color='k', ls=':', label='clean p99')
    ax[0, 2].set_title(f'episode-mean action norm (AUC {act["auc"]:.2f})')
    ax[0, 2].legend()
    zb = np.linspace(min(box[0, 2] - 3, np.nanpercentile(rep[..., 2][bvalid], 0.5)),
                     max(box[1, 2] + 3, np.nanpercentile(rep[..., 2][bvalid], 99.5)), 60)
    ax[1, 0].hist(truth[..., 2][bvalid], zb, density=True, alpha=.5, label='true')
    ax[1, 0].hist(rep[..., 2][bvalid], zb, density=True, alpha=.5, label='broadcast')
    for zz in box[:, 2]:
        ax[1, 0].axvline(zz, color='k', ls=':')
    ax[1, 0].set_title('altitude: true vs broadcast (room at dotted lines)')
    ax[1, 0].legend()
    sb = np.linspace(0, np.percentile(np.r_[rep_speed, true_speed], 99.5), 60)
    ax[1, 1].hist(true_speed, sb, density=True, alpha=.5, label='true')
    ax[1, 1].hist(rep_speed, sb, density=True, alpha=.5, label='broadcast')
    ax[1, 1].axvline(thr['speed'], color='k', ls=':', label='clean max')
    ax[1, 1].set_title('speed: true vs broadcast')
    ax[1, 1].legend()
    ax[1, 2].hist(offset, 60, density=True)
    ax[1, 2].set_title('|broadcast position - true position| (m)')
    fig.suptitle(f'{run} -- stealth analysis (n = {E}, broadcasts {source})')
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_path = os.path.join(OUT_DIR, f'stealth_{run}.png')
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)

    pct = lambda x: f'{100 * x:.1f} %'
    L = [f'# Stealth analysis -- `{run}` (spec item 6)\n',
         f'Own-seed confirmatory run, n = {E} episodes, paired clean run at seed {aseed}. '
         f'Broadcasts **{source}**. Thresholds set on the clean run only.\n',
         '## Action level\n',
         '| statistic | value |', '|---|---|',
         f'| action-norm ratio adversary / honest | {act["ratio"]:.3f} |',
         f'| AUC, episode-mean action norm, adversary vs honest | {act["auc"]:.3f} |',
         f'| control: same slot in the CLEAN run -- ratio / AUC | {act["ratio_clean_slot"]:.3f} '
         f'/ {act["auc_clean_slot"]:.3f} |',
         f'| naive action detector (> clean p99 = {thr_act:.3f}): adversary episodes flagged '
         f'| {pct(act["detect"])} |',
         f'| same detector, honest drones flagged | {pct(act["false_alarm"])} |',
         '\n## Broadcast physics (naive plausibility checks)\n',
         f'Thresholds (max over every honest drone-step of the clean run): speed '
         f'{thr["speed"]:.2f} m/s, acceleration {thr["acc"]:.1f} m/s², kinematic residual '
         f'{thr["resid"]:.4f} m; bounds = the {box[1, 0] - box[0, 0]:.0f} x '
         f'{box[1, 1] - box[0, 1]:.0f} x {box[1, 2]:.0f} m room.\n',
         '| check | adversary steps failing | adversary episodes flagged | honest episodes flagged |',
         '|---|---|---|---|']
    for k in checks:
        L.append(f'| {k} | {pct(step_rate[k])} | {pct(ep_detect[k])} | {pct(ep_fa[k])} |')
    L += [f'| **any check** | | **{pct(float(any_adv.mean()))}** | {pct(float(any_hon))} |',
          '',
          f'Broadcast below the floor: {pct(below_floor)} of steps. Broadcast speed mean '
          f'{rep_speed.mean():.2f} m/s (true {true_speed.mean():.2f}). Offset |claim − truth| '
          f'mean {offset.mean():.2f} m, p99 {np.percentile(offset, 99):.2f} m.',
          '',
          '**Reading.** An action-level detector sees this adversary iff the AUC is well away from '
          '0.5 *and* from the same-slot clean control. A broadcast check catches it iff "any check" flags its episodes at a rate well '
          'above the honest false-alarm rate. The spec asks to say so plainly if a naive '
          'detector catches the stealthy adversary.',
          f'\n![]({os.path.basename(fig_path)})']
    md = os.path.join(OUT_DIR, f'stealth_{run}.md')
    with open(md, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    summary = dict(run=run, source=source, action_ratio=act['ratio'], action_auc=act['auc'],
                   action_ratio_clean_slot=act['ratio_clean_slot'],
                   action_auc_clean_slot=act['auc_clean_slot'],
                   action_detect=act['detect'], action_false_alarm=act['false_alarm'],
                   physics_detect_any=float(any_adv.mean()), physics_false_alarm_any=float(any_hon),
                   **{f'physics_detect_{k}': ep_detect[k] for k in checks},
                   **{f'physics_step_fail_{k}': step_rate[k] for k in checks},
                   below_floor=below_floor, broadcast_speed=float(rep_speed.mean()),
                   true_speed=float(true_speed.mean()), offset_mean=float(offset.mean()))
    return md, [fig_path], summary


# ------------------------------------------------------------------ behaviour (7)

def _goal_error(npz, drones):
    """(E, T, |drones|) distance to goal. obs[:3] is position relative to the goal."""
    with np.load(npz) as z:
        rel = np.asarray(z['obs'][:, :, drones, :3], np.float64)
        lengths = np.asarray(z['episode_lengths'])
    err = np.linalg.norm(rel, axis=-1)
    T = err.shape[1]
    err[np.arange(T)[None, :] >= lengths[:, None]] = np.nan
    return err, lengths


def behaviour(run, no_wandb=True):
    plt = _plt()
    npz, clean_npz, aseed = confirm_pair(run)
    cfg = _cfg(run)
    box, dt = room_box(cfg), 1.0 / 100.0
    with np.load(npz) as z:
        mask = np.asarray(z['byzantine_mask'])
        pos = np.asarray(z['positions'], np.float64)
        vel = np.asarray(z['velocities'], np.float64)
        col = np.asarray(z['collisions'], np.float64)
        rel = np.asarray(z['obs'][:EPISODES_PLOTTED, :, :, :3], np.float64)
        lengths = np.asarray(z['episode_lengths'])
    with np.load(clean_npz) as z:
        pos_c = np.asarray(z['positions'], np.float64)
        vel_c = np.asarray(z['velocities'], np.float64)
        col_c = np.asarray(z['collisions'], np.float64)
        len_c = np.asarray(z['episode_lengths'])
    adv, hon = int(np.flatnonzero(mask)[0]), np.flatnonzero(~mask)
    E, T = pos.shape[:2]
    n = min(E, len(len_c))
    err, _ = _goal_error(npz, np.arange(pos.shape[2]))
    err_c, _ = _goal_error(clean_npz, np.arange(pos.shape[2]))
    err, err_c = err[:n], err_c[:n]

    # 1. when does the damage accrue? honest goal error, attacked minus clean, over time
    d_t = np.nanmean(err[:, :, hon] - err_c[:, :, hon], axis=(0, 2))
    thirds = [np.nansum(d_t[i * T // 3:(i + 1) * T // 3]) for i in range(3)]
    share = np.array(thirds) / (np.nansum(np.abs(thirds)) or 1.0)
    # 2. is it local? per honest drone-episode increase vs mean distance to the adversary's
    # slot, measured in the CLEAN run: the attack itself moves drones, and a drone knocked off
    # course is far from everything, which would make "far from the adversary" look harmful.
    dist = np.linalg.norm(pos_c[:n, :, hon] - pos_c[:n, :, [adv]], axis=-1)
    dist_ep = np.nanmean(np.where(np.isnan(err_c[:, :, hon]), np.nan, dist), axis=1)     # (E, H)
    inc_ep = np.nanmean(err[:, :, hon], axis=1) - np.nanmean(err_c[:, :, hon], axis=1)
    rho, p_rho = stats.spearmanr(dist_ep.ravel(), inc_ep.ravel())
    rank = np.argsort(np.argsort(dist_ep, axis=1), axis=1)
    near = inc_ep[rank < 2].mean()
    far = inc_ep[rank >= len(hon) - 2].mean()
    # 3. collisions, separation, flight quality
    c_att, c_cln = col[:n].sum(1), col_c[:n].sum(1)
    p_col = stats.ttest_rel(c_att, c_cln).pvalue

    def min_sep(P, L):
        out = []
        for e in range(len(L)):
            q = P[e, :L[e]][:, hon]
            dd = np.linalg.norm(q[:, :, None] - q[:, None, :], axis=-1)
            dd[:, np.arange(len(hon)), np.arange(len(hon))] = np.inf
            out.append(dd.min())
        return np.array(out)
    ms, ms_c = min_sep(pos[:n], lengths[:n]), min_sep(pos_c[:n], len_c[:n])

    def adv_sep(P, L):
        return np.array([np.linalg.norm(P[e, :L[e]][:, hon] - P[e, :L[e]][:, [adv]], axis=-1).min()
                         for e in range(len(L))])
    asep, asep_c = adv_sep(pos[:n], lengths[:n]), adv_sep(pos_c[:n], len_c[:n])

    def collision_pairs(P, C, L):
        """Per episode: collision steps whose closest pair includes the adversary, and not."""
        w, wo = np.zeros(len(L)), np.zeros(len(L))
        iu = np.triu_indices(P.shape[2], 1)
        for e in range(len(L)):
            for t in np.flatnonzero(C[e, :L[e]] > 0):
                d = np.linalg.norm(P[e, t][:, None] - P[e, t][None, :], axis=-1)[iu]
                i, j = iu[0][d.argmin()], iu[1][d.argmin()]
                if adv in (i, j):
                    w[e] += 1
                else:
                    wo[e] += 1
        return w, wo
    cw, cwo = collision_pairs(pos[:n], col[:n], lengths[:n])
    cw_c, cwo_c = collision_pairs(pos_c[:n], col_c[:n], len_c[:n])

    def jitter(V, L, idx):
        return np.array([np.linalg.norm(np.diff(V[e, :L[e]][:, idx], axis=0), axis=-1).mean()
                         for e in range(len(L))])
    jit, jit_c = jitter(vel[:n], lengths[:n], hon), jitter(vel_c[:n], len_c[:n], hon)
    spd = np.array([np.linalg.norm(vel[e, :lengths[e]][:, hon], axis=-1).mean() for e in range(n)])
    spd_c = np.array([np.linalg.norm(vel_c[e, :len_c[e]][:, hon], axis=-1).mean() for e in range(n)])
    own = np.nanmean(err[:, :, adv], axis=1) - np.nanmean(err_c[:, :, adv], axis=1)

    # 4. the phantom, for the plotted episodes
    rep, _, source = broadcasts(run, npz, adv, episodes=np.arange(min(EPISODES_PLOTTED, E)))
    phantom = np.linalg.norm(pos[:len(rep), :, hon] - rep[:, :, None, :3], axis=-1)
    true_d = np.linalg.norm(pos[:len(rep), :, hon] - pos[:len(rep), :, [adv]], axis=-1)

    # --- figures: 10 episodes top-down, and the damage-over-time profile
    k = min(EPISODES_PLOTTED, E)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5), squeeze=False)
    for e, ax in zip(range(k), axes.ravel()):
        Le = int(lengths[e])
        goal = pos[e, :Le] - rel[e, :Le]
        for j in hon:
            ax.plot(pos[e, :Le, j, 0], pos[e, :Le, j, 1], '-', color='grey', lw=.8, alpha=.8)
            ax.plot(pos_c[e, :Le, j, 0], pos_c[e, :Le, j, 1], ':', color='C0', lw=.6, alpha=.6)
            ax.plot(goal[:, j, 0], goal[:, j, 1], 'x', color='k', ms=3, alpha=.3)
        ax.plot(pos[e, :Le, adv, 0], pos[e, :Le, adv, 1], '-', color='C3', lw=1.5)
        r = np.clip(rep[e, :Le, :2], box[0, :2] - 1, box[1, :2] + 1)
        ax.plot(r[:, 0], r[:, 1], '.', color='C3', ms=1, alpha=.3)
        ax.add_patch(plt.Rectangle(box[0, :2], *(box[1, :2] - box[0, :2]), fill=False, ls='--'))
        ax.set_xlim(box[0, 0] - 1.2, box[1, 0] + 1.2)
        ax.set_ylim(box[0, 1] - 1.2, box[1, 1] + 1.2)
        ax.set_aspect('equal')
        ax.set_title(f'ep {e}: honest Δerr {np.nanmean(err[e, :, hon] - err_c[e, :, hon]):+.2f} m, '
                     f'coll {int(c_att[e])} vs {int(c_cln[e])}', fontsize=8)
    fig.suptitle(f'{run}: top-down. red = adversary (true), red dots = its broadcast '
                 f'({source}, clipped to 1 m outside the room), grey = honest attacked, '
                 'blue dotted = same honest drones, same scenario, clean; x = goals')
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    ep_fig = os.path.join(OUT_DIR, f'behaviour_{run}_episodes.png')
    fig.savefig(ep_fig, dpi=100)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(np.arange(T) * dt, d_t)
    ax.axhline(0, color='k', lw=.5)
    ax.set_xlabel('time in episode (s)')
    ax.set_ylabel('honest goal error, attacked − clean (m)')
    ax.set_title(f'{run}: when the damage accrues (mean over {n} paired episodes)')
    fig.tight_layout()
    t_fig = os.path.join(OUT_DIR, f'behaviour_{run}_time.png')
    fig.savefig(t_fig, dpi=110)
    plt.close(fig)

    tt = lambda a, b: stats.ttest_rel(a, b).pvalue
    L = [f'# Behaviour -- `{run}` (spec item 7)\n',
         f'{n} paired episodes (own-seed confirmatory vs clean, seed {aseed}); figures show the '
         f'first {k}. Broadcasts {source}. These are the numbers the plain-language description '
         'must agree with.\n',
         '| observation | attacked | clean | paired p |', '|---|---|---|---|',
         f'| honest goal error, episode mean (m) | {np.nanmean(err[:, :, hon]):.3f} '
         f'| {np.nanmean(err_c[:, :, hon]):.3f} | '
         f'{tt(np.nanmean(err[:, :, hon], axis=(1, 2)), np.nanmean(err_c[:, :, hon], axis=(1, 2))):.4f} |',
         f'| adversary\'s OWN goal error change vs same drone clean (m) | {own.mean():+.3f} | 0 '
         f'| {stats.ttest_1samp(own, 0).pvalue:.4f} |',
         f'| team collisions per episode | {c_att.mean():.2f} | {c_cln.mean():.2f} | {p_col:.4f} |',
         f'| honest–honest minimum separation (m) | {ms.mean():.3f} | {ms_c.mean():.3f} '
         f'| {tt(ms, ms_c):.4f} |',
         f'| collision steps whose closest pair INCLUDES the adversary\'s slot | {cw.mean():.2f} '
         f'| {cw_c.mean():.2f} | {tt(cw, cw_c):.4f} |',
         f'| collision steps between two honest drones | {cwo.mean():.2f} | {cwo_c.mean():.2f} '
         f'| {tt(cwo, cwo_c):.4f} |',
         f'| adversary-slot ↔ nearest honest, minimum separation (m) | {asep.mean():.3f} '
         f'| {asep_c.mean():.3f} | {tt(asep, asep_c):.4f} |',
         f'| honest speed (m/s) | {spd.mean():.3f} | {spd_c.mean():.3f} | {tt(spd, spd_c):.4f} |',
         f'| honest jitter, mean ‖Δv‖ per step (m/s) | {jit.mean():.4f} | {jit_c.mean():.4f} '
         f'| {tt(jit, jit_c):.4f} |',
         '',
         f'**When.** Share of the honest goal-error increase accrued in the first / middle / last '
         f'third of the episode: {share[0]:+.2f} / {share[1]:+.2f} / {share[2]:+.2f}.',
         f'**Where.** Spearman correlation of an honest drone\'s goal-error increase with its mean '
         f'distance to the adversary\'s slot in the same scenario, clean: ρ = {rho:+.3f} '
         f'(p = {p_rho:.3g}). Mean increase for the 2 honest drones nearest that slot '
         f'{near:+.3f} m vs the 2 farthest {far:+.3f} m.',
         f'**The phantom** (first {len(rep)} episodes). Distance from honest drones to the '
         f'adversary\'s broadcast position: median {np.nanmedian(phantom):.2f} m, vs to its true '
         f'position {np.nanmedian(true_d):.2f} m.',
         f'\n![]({os.path.basename(ep_fig)})\n\n![]({os.path.basename(t_fig)})']
    md = os.path.join(OUT_DIR, f'behaviour_{run}.md')
    with open(md, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    summary = dict(run=run, honest_err=float(np.nanmean(err[:, :, hon])),
                   honest_err_clean=float(np.nanmean(err_c[:, :, hon])),
                   own_err_change=float(own.mean()), collisions=float(c_att.mean()),
                   collisions_clean=float(c_cln.mean()), min_sep=float(ms.mean()),
                   collision_steps_with_adv=float(cw.mean()),
                   collision_steps_with_adv_clean=float(cw_c.mean()),
                   collision_steps_honest=float(cwo.mean()),
                   collision_steps_honest_clean=float(cwo_c.mean()),
                   adv_min_sep=float(asep.mean()), adv_min_sep_clean=float(asep_c.mean()),
                   min_sep_clean=float(ms_c.mean()), locality_rho=float(rho),
                   damage_share_thirds=[float(x) for x in share])
    return md, [ep_fig, t_fig], summary


# ------------------------------------------------------------------ CLI

def _log(name, rows, files, no_wandb):
    if no_wandb:
        return
    from byzantine_experiments.analysis.wandb_logging import log_summary_table
    url = log_summary_table(rows=rows, run_name=name, group='tier3_report',
                            tags=['tier3', 'report'], artifacts=files)
    if url:
        print(f'wandb: {url}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('item', choices=['rewards', 'compute', 'stealth', 'behaviour'])
    ap.add_argument('--run', nargs='*', default=None,
                    help='adversary run(s); default: every train_dir/adv_seed* for rewards and '
                         'compute, required for stealth and behaviour')
    ap.add_argument('--refresh', action='store_true', help='rebuild the TensorBoard cache')
    ap.add_argument('--no-wandb', action='store_true')
    a = ap.parse_args()
    runs = a.run or _runs()
    if a.item == 'rewards':
        md, figs = rewards(runs, a.refresh)
        _log('tier3_rewards', [dict(run=r) for r in runs], [md] + figs, a.no_wandb)
    elif a.item == 'compute':
        md, rows = compute(runs, a.refresh)
        _log('tier3_compute', rows, [md], a.no_wandb)
    else:
        if not a.run:
            raise SystemExit(f'{a.item} needs --run <adversary>')
        fn = stealth if a.item == 'stealth' else behaviour
        for run in a.run:
            md, figs, summary = fn(run)
            _log(f'tier3_{a.item}_{run}', [summary], [md] + figs, a.no_wandb)


if __name__ == '__main__':
    main()
