"""
Tier 3 efficacy gate + cross-seed transfer.

Reads the rollout sidecars produced by `scripts/byz_eval_adv.sh` (which drives the ordinary
Tier 0 collector with `--tier tier3_learned`, so Tier 3 trajectories are byte-compatible with
Tiers 1 and 2) and evaluates every adversary against the spec's gate:

| criterion                            | threshold                     |
|--------------------------------------|-------------------------------|
| **honest-drone** goal-RMSE deg (1st)  | >= 10 % (recalibrated, see GATE) |
| all-drone goal-RMSE deg (comparability)| reported, not gated          |
| team reward degradation (secondary)   | >= 10 %, reported with its CI |
| adversary survival rate               | >= 70 % of episodes           |
| action magnitude ratio vs honest      | <= 1.5x  (stealth runs only)  |

**Why the primary criterion is honest-drone-only.** The obvious choice, all-drone goal RMSE,
is what Tiers 1 and 2 were calibrated on -- and it is *gameable by an optimiser*. An
adversary maximises it simply by flying its own drone away from its own goal; no damage to
anyone else is required. The first horizon sweep made this concrete: a checkpoint reporting
109 % all-drone degradation had **+384 % on its own drone and +18 % on the honest six**. The
metric was measuring self-destruction.

That was fine for hand-designed Tier 1/2 faults, whose severity knob was set by hand and
could not chase the metric. It is not fine for a trained policy. So the gate keys on the
honest drones -- the quantity "team damage" actually means -- while all-drone degradation is
still reported, because the matched-damage triple against Tiers 1/2 is defined on it.

Plus the cross-seed transfer matrix: adversary trained on seed i, evaluated on baseline
seed j. The spec is explicit that either outcome is a result -- a transferring adversary is
a stronger claim, a seed-specific one is still publishable but must be stated as such.

Usage:
    python -m byzantine_experiments.analysis.gate_tier3
    python -m byzantine_experiments.analysis.gate_tier3 --horizon    # milestone sweep only
"""

from byzantine_experiments.paths import RESULTS_ROOT, results_dir
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

from byzantine_experiments.analysis.metrics import degradation_vs_clean

RESULTS_DIR = results_dir('tier3')
CLEAN_DIR = results_dir('tier1')
OUT_CSV = os.path.join(RESULTS_ROOT, 'tier3_gate.csv')
OUT_MD = os.path.join(RESULTS_ROOT, 'tier3_gate.md')

# EFFICACY GATE -- recalibrated 2026-09-24, pre-registered before any v2 Tier 3 evaluation.
#
# The spec's 30 % was set against legacy Tier 1/2 numbers that turned out ~3x inflated by the
# seeding bug (design 6.16-6.19): Tier 2 `mirror` was reported at +30.4 %, and measures +9.7 %
# properly paired. On the first v2 data -- the whole `liar` family, 18 paired runs -- the best
# hand-designed observation-channel attack averages +9.6 % on its strongest baseline seed (max
# +13.1 %), and +5.7 % across all seeds. So 10 % means "at least as damaging as the best
# hand-designed attack", which is the bar the spec's 30 % was meant to express.
#
# Detectability sets how the gate is applied, not just where. Paired, per arm:
#     n = 50  -> 57 % power to detect a true 10 %   (horizon sweeps: SCREENING ONLY)
#     n = 200 -> 99 % power to detect a true 10 %   (confirmatory: this is the decision)
# A row from an n < 200 evaluation that clears the gate is therefore a CANDIDATE for the
# confirmatory stage, never a pass.
#
# The value is chosen now, from Tier 1/2 data only, so it cannot be tuned to the Tier 3 result.
GATE = dict(honest_goal_rmse_deg=10.0, reward_deg=10.0, survival=0.70, action_ratio=1.5)
CONFIRMATORY_MIN_EPISODES = 200

# t3_adv_seed1_stealth0_on_s2_best           f = .125, motors only          (round 1)
# t3_adv_seed1_stealth0_f250_on_s1_m050M      f = .250, motors only          (round 2)
# t3_adv_seed1_stealth0_lie_on_s1_m050M       f = .125, + observation channel (round 3)
# t3_adv_seed1_stealth0_lieonly_on_s1_best    f = .125, observation channel only
# Both optional groups are absent for round 1, so its 14 sidecars keep parsing unchanged.
NAME_RE = re.compile(
    r'^t3_adv_seed(?P<aseed>\d+)_stealth(?P<stealth>\d+)(?:_f(?P<f>\d+))?'
    r'(?:_(?P<channel>lie(?:only)?(?:ego)?(?:plaus)?))?'
    r'_on_s(?P<bseed>\d+)_(?P<ckpt>.+)$')


def load_runs(results_dir=RESULTS_DIR, clean_dir=CLEAN_DIR):
    clean = {}
    for path in sorted(glob.glob(os.path.join(clean_dir, 't1_clean_s*.json'))):
        seed = int(re.search(r'_s(\d+)$', os.path.splitext(os.path.basename(path))[0]).group(1))
        with open(path) as f:
            sc = json.load(f)
        if 'metrics' in sc:
            clean[seed] = sc['metrics']

    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        stem = os.path.splitext(os.path.basename(path))[0]
        m = NAME_RE.match(stem)
        if not m:
            print(f'  ! skipping unrecognised name: {stem}')
            continue
        with open(path) as f:
            sc = json.load(f)
        if 'metrics' not in sc:
            print(f'  ! skipping {stem}: no metrics block')
            continue
        d = m.groupdict()
        runs.append(dict(stem=stem, adv_seed=int(d['aseed']), stealth=int(d['stealth']),
                         f=(int(d['f']) / 1000.0 if d.get('f') else 0.125),
                         channel=('observation' if d.get('channel') else 'motors'),
                         lie_only=str(d.get('channel') or '').startswith('lieonly'),
                         ego_obs='ego' in str(d.get('channel') or ''),
                         plausible='plaus' in str(d.get('channel') or ''),
                         baseline_seed=int(d['bseed']), ckpt=d['ckpt'],
                         sidecar=sc, metrics=sc['metrics']))
    return runs, clean


# Decompressing a run's `obs` costs ~1 s and every run is read at least twice (once for the
# attacked arm, once as somebody's clean reference), so memoise on the resolved path.
_RMSE_CACHE = {}


def _honest_rmse_per_episode(npz_path, drone_idx):
    """
    Per-episode goal RMSE over exactly `drone_idx`, read straight from the trajectory.

    `obs[..., :3]` is `pos - goal` in the ego frame, so its norm is the goal error.

    These `.npz` files are **compressed**, which makes `mmap_mode='r'` silently useless --
    numpy has to inflate the whole array regardless. So `obs` is read exactly once per file
    and sliced in memory; reading it inside the per-episode loop instead inflates all ~52 MB
    once per episode and turns a 30 s job into a 10 min one.
    """
    key = (os.path.realpath(npz_path), tuple(np.atleast_1d(drone_idx).tolist()))
    if key in _RMSE_CACHE:
        return _RMSE_CACHE[key]
    with np.load(npz_path) as d:
        lengths = np.asarray(d['episode_lengths'])
        goal_err = np.linalg.norm(
            np.asarray(d['obs'][:, :, drone_idx, :3], dtype=np.float64), axis=-1)  # (E, T, |idx|)
    out = np.empty(len(lengths), dtype=np.float64)
    for e, n in enumerate(lengths):
        out[e] = np.sqrt((goal_err[e, :int(n)] ** 2).mean())
    _RMSE_CACHE[key] = out
    return out


def honest_damage_stats(run_npz, clean_npz):
    """
    Honest-drone goal-RMSE degradation with a confidence interval and a p-value.

    Two things this fixes over the sidecar-only path, both of which matter at these
    sample sizes:

    1. **Matched drone subsets.** `degradation_vs_clean` compares the attacked run's honest
       drones (6 of 8 at f = 0.25) against the clean run's `goal_rmse_honest_mean`, which is
       over **all 8** -- there are no Byzantine drones in a clean run. Different drones start
       at different formation slots, so that is a systematic bias. Here the clean reference is
       recomputed over the *same* indices.
    2. **Uncertainty.** Per-episode goal RMSE has a coefficient of variation of ~42 % on this
       task. At n = 20 the 95 % CI on a degradation percentage is about +/- 22 points, which is
       most of the distance to the 30 % gate. A bare point estimate invites reading noise as
       signal, so the CI and Welch p-value travel with it everywhere.

    Returns None if either trajectory file is missing.
    """
    if not (os.path.exists(run_npz) and os.path.exists(clean_npz)):
        return None
    # v2: paired by construction when both runs come from the per-episode-seeded collector at
    # one --seed. Local import -- paired.py imports this module.
    from byzantine_experiments.analysis.paired import PairingError, paired_degradation
    try:
        return paired_degradation(run_npz, clean_npz)
    except PairingError as e:
        pairing_error = str(e)
    with np.load(run_npz) as d:
        honest_idx = np.where(~np.asarray(d['byzantine_mask']))[0]
    atk = _honest_rmse_per_episode(run_npz, honest_idx)
    cln = _honest_rmse_per_episode(clean_npz, honest_idx)
    st = welch_degradation(atk, cln)
    if st is not None:
        # Reported, never passable: an unpaired comparison in this simulator carries a ~7 %
        # run-level offset its CI does not include (design 6.16).
        st.update(paired=False, pairing_error=pairing_error)
    return st


def welch_degradation(atk, cln):
    """
    Percentage degradation of `atk` over `cln` with a Welch 95 % CI and p-value.

    Shared by the exploratory horizon sweep (this module) and the confirmatory stage
    (`confirm_tier3.py`), so the two cannot quietly disagree about what "significant" means.
    """
    atk, cln = np.asarray(atk, dtype=np.float64), np.asarray(cln, dtype=np.float64)
    if len(atk) < 2 or len(cln) < 2 or cln.mean() <= 1e-9:
        return None
    va, vc, na, nc = atk.var(ddof=1), cln.var(ddof=1), len(atk), len(cln)
    se = np.sqrt(va / na + vc / nc)
    df = ((va / na + vc / nc) ** 2
          / ((va / na) ** 2 / (na - 1) + (vc / nc) ** 2 / (nc - 1))) if se > 0 else 1.0
    crit = stats.t.ppf(0.975, df)
    diff = atk.mean() - cln.mean()
    scale = 100.0 / cln.mean()
    _, p = stats.ttest_ind(atk, cln, equal_var=False)
    return dict(deg_pct=scale * diff,
                ci_lo=scale * (diff - crit * se), ci_hi=scale * (diff + crit * se),
                p_value=float(p), n_attacked=na, n_clean=nc,
                clean_mean=float(cln.mean()), attacked_mean=float(atk.mean()))


def checkpoint_steps(run):
    """Env steps of the evaluated checkpoint, from the agent diagnostics if present."""
    diag = (run['sidecar'].get('agent_diagnostics') or {})
    for v in diag.values():
        if isinstance(v, dict) and 'checkpoint_steps' in v:
            return float(v['checkpoint_steps'])
    m = re.match(r'^m(\d+)M$', run['ckpt'])
    return float(m.group(1)) * 1e6 if m else float('nan')


def build_rows(runs, clean, results_dir=RESULTS_DIR, clean_dir=CLEAN_DIR):
    rows = []
    for r in runs:
        if r['baseline_seed'] not in clean:
            print(f"  ! no clean reference for baseline seed {r['baseline_seed']}")
            continue
        m = r['metrics']
        deg = degradation_vs_clean(m, clean[r['baseline_seed']])
        survival = m.get('attacker_survived_mean', float('nan'))
        ratio = m.get('action_norm_ratio', float('nan'))
        goal_deg = deg['goal_rmse_degradation_pct']            # all drones, for comparability
        honest_deg = deg.get('victim_spillover_goal_rmse_pct', float('nan'))  # the real target
        own_deg = (100.0 * (m['goal_rmse_byzantine_mean'] - clean[r['baseline_seed']]['goal_rmse_mean'])
                   / clean[r['baseline_seed']]['goal_rmse_mean'])
        rew_deg = deg['reward_degradation_pct']

        # Prefer the trajectory-based figure: matched drone subsets, and it carries a CI.
        st = honest_damage_stats(
            os.path.join(results_dir, r['stem'] + '.npz'),
            os.path.join(clean_dir, f"t1_clean_s{r['baseline_seed']}.npz"))
        if st is not None:
            honest_deg = st['deg_pct']

        stealth_run = r['stealth'] > 0
        checks = {
            'honest_goal>=gate': (honest_deg >= GATE['honest_goal_rmse_deg']
                                if np.isfinite(honest_deg) else False),
            # A point estimate above the gate with a CI straddling zero is noise, not an attack.
            # Missing data or an unpaired comparison can never count as significant. (This used
            # to default to True when the trajectory was missing.)
            'significant': bool(st is not None and st.get('paired', False)
                                and st['ci_lo'] > 0.0),
            'reward>=gate': rew_deg >= GATE['reward_deg'],
            'survival>=70': survival >= GATE['survival'],
            # Only binding on the stealth condition, per the spec.
            'action_ratio<=1.5': (ratio <= GATE['action_ratio']) if stealth_run else True,
        }
        rows.append(dict(
            run=r['stem'], adv_seed=r['adv_seed'], stealth=r['stealth'], f=r['f'],
            # `.get` so callers that build run dicts by hand (tests, ad-hoc analysis) do not
            # have to know about every column the gate has grown.
            channel=r.get('channel', 'motors'), lie_only=r.get('lie_only', False),
            ego_obs=r.get('ego_obs', False),
            plausible=r.get('plausible', False),
            baseline_seed=r['baseline_seed'], ckpt=r['ckpt'],
            checkpoint_steps=checkpoint_steps(r),
            matched_seed=(r['adv_seed'] == r['baseline_seed']),
            n_episodes=m['n_episodes'],

            goal_rmse_deg_pct=goal_deg,
            honest_goal_rmse_deg_pct=honest_deg,
            own_goal_rmse_deg_pct=own_deg,
            reward_deg_pct=rew_deg,
            reward_deg_pct_sem=deg['reward_degradation_pct_sem'],
            spillover_goal_rmse_pct=deg.get('victim_spillover_goal_rmse_pct', float('nan')),
            survival=survival,
            action_norm_ratio=ratio,
            action_deviation=m['action_deviation_byzantine_mean'],
            saturation=m['saturation_rate_byzantine'],
            collisions_delta=deg['collisions_delta'],
            goal_rmse=m['goal_rmse_mean'],
            episode_reward=m['episode_reward_mean'],

            paired=bool(st and st.get('paired', False)),
            pair_corr=(st.get('corr', float('nan')) if st else float('nan')),
            honest_deg_ci_lo=(st['ci_lo'] if st else float('nan')),
            honest_deg_ci_hi=(st['ci_hi'] if st else float('nan')),
            honest_deg_p=(st['p_value'] if st else float('nan')),

            passes_primary=bool(checks['honest_goal>=gate'] and checks['significant']
                                and checks['survival>=70'] and checks['action_ratio<=1.5']),
            **{f'chk_{k}': bool(v) for k, v in checks.items()},
        ))
    return rows


def fmt(x, nd=1, suffix=''):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    return f'{x:.{nd}f}{suffix}'


def tick(b):
    return '✅' if b else '❌'


def write_markdown(rows, path):
    L = ['# Tier 3 — efficacy gate\n']
    if not rows:
        L.append('_No Tier 3 rollouts found._\n')
        open(path, 'w').write('\n'.join(L)); return

    L.append(f'Thresholds: **honest-drone** goal-RMSE degradation ≥ '
             f'{GATE["honest_goal_rmse_deg"]:.0f} %, '
             f'team-reward degradation ≥ {GATE["reward_deg"]:.0f} %, '
             f'survival ≥ {GATE["survival"]*100:.0f} %, '
             f'action-magnitude ratio ≤ {GATE["action_ratio"]} (stealth runs only).\n')
    L.append('Clean references are Tier 1\'s, per baseline seed. `reward deg %` is reported '
             'with its SEM and **is not used to gate** — see design/tier2_design.md §10 for '
             'why that metric cannot rank severities here.\n')
    L.append('**Every honest-damage figure below is measured against the clean run restricted '
             'to the *same drone indices*, from the trajectories rather than the sidecar '
             'means, and carries a 95 % CI and a Welch p-value.** Per-episode goal RMSE has a '
             '~42 % coefficient of variation on this task, so at n = 20 the CI is roughly '
             '± 22 points — most of the distance to the gate. A run passes only if the point '
             f'estimate clears {GATE["honest_goal_rmse_deg"]:.0f} % **and** the CI excludes zero.\n')
    L.append('**The gate keys on `honest deg %`, not `all-drone deg %`.** The all-drone figure '
             'is gameable: an adversary maximises it by flying its own drone away, with no '
             'damage to anyone else. `own deg %` is shown alongside so the split is visible — '
             'a large `own` with a small `honest` is self-destruction, not an attack.\n')

    # ---- horizon sweep -------------------------------------------------------------
    sweep = sorted([r for r in rows if r['ckpt'].startswith('m')],
                   key=lambda r: (r['adv_seed'], r['stealth'], r['checkpoint_steps']))
    if sweep:
        L.append('\n## Horizon sweep — how the adversary evolved during training\n')
        L.append('| adversary | channel | f | checkpoint | **honest deg % [95 % CI]** | p '
                 '| own deg % | survival | action ratio | saturation | gate |')
        L.append('|' + '---|' * 11)
        for r in sweep:
            L.append(f"| adv_seed{r['adv_seed']}_stealth{r['stealth']} "
                     f"| {r.get('channel', 'motors')}{' (lie only)' if r.get('lie_only') else ''} "
                     f"| {r.get('f', 0.125):.3f} | {r['checkpoint_steps']/1e6:.0f}M "
                     f"| **{fmt(r['honest_goal_rmse_deg_pct'])}** "
                     f"[{fmt(r['honest_deg_ci_lo'])}, {fmt(r['honest_deg_ci_hi'])}] "
                     f"| {fmt(r['honest_deg_p'], 3)} "
                     f"| {fmt(r['own_goal_rmse_deg_pct'])} "
                     f"| {fmt(100*r['survival'], 0, '%')} | {fmt(r['action_norm_ratio'], 2)} "
                     f"| {fmt(100*r['saturation'], 0, '%')} | {tick(r['passes_primary'])} |")

    # ---- gate ----------------------------------------------------------------------
    gate_rows = sorted([r for r in rows if not r['ckpt'].startswith('m')],
                       key=lambda r: (r['adv_seed'], r['stealth'], r['baseline_seed']))
    if gate_rows:
        L.append('\n## Efficacy gate\n')
        L.append('| adversary | ckpt | eval seed | **honest deg %** | own deg % | '
                 'reward deg % (±SEM) | survival | action ratio | honest✔ | surv✔ | ratio✔ '
                 '| **GATE** |')
        L.append('|' + '---|' * 12)
        for r in gate_rows:
            same = '' if r['matched_seed'] else ' *(cross)*'
            L.append(
                f"| adv_seed{r['adv_seed']}_stealth{r['stealth']} | {r['ckpt']} "
                f"| s{r['baseline_seed']}{same} | **{fmt(r['honest_goal_rmse_deg_pct'])}** "
                f"| {fmt(r['own_goal_rmse_deg_pct'])} "
                f"| {fmt(r['reward_deg_pct'], 0)} ± {fmt(r['reward_deg_pct_sem'], 0)} "
                f"| {fmt(100*r['survival'], 0, '%')} | {fmt(r['action_norm_ratio'], 2)} "
                f"| {tick(r['chk_honest_goal>=gate'])} | {tick(r['chk_survival>=70'])} "
                f"| {tick(r['chk_action_ratio<=1.5'])} | {tick(r['passes_primary'])} |")

    # ---- cross-seed transfer -------------------------------------------------------
    by_adv = defaultdict(dict)
    for r in gate_rows:
        by_adv[(r['adv_seed'], r['stealth'], r['ckpt'])][r['baseline_seed']] = r
    transfer = {k: v for k, v in by_adv.items() if len(v) > 1}
    if transfer:
        L.append('\n## Cross-seed transfer\n')
        L.append('Goal-RMSE degradation of an adversary trained on one baseline seed and '
                 'evaluated on another. Diagonal = matched seed.\n')
        seeds = sorted({r['baseline_seed'] for r in gate_rows})
        L.append('| adversary (ckpt) | ' + ' | '.join(f'on s{s}' for s in seeds) + ' | transfers? |')
        L.append('|' + '---|' * (len(seeds) + 2))
        for (aseed, stealth, ckpt), per in sorted(transfer.items()):
            cells, vals = [], []
            for s in seeds:
                r = per.get(s)
                if r is None:
                    cells.append('—'); continue
                v = r['honest_goal_rmse_deg_pct']; vals.append(v)
                cells.append(f"**{fmt(v)}**" if s == aseed else fmt(v))
            off = [per[s]['honest_goal_rmse_deg_pct'] for s in seeds if s in per and s != aseed]
            on = per.get(aseed, {}).get('honest_goal_rmse_deg_pct', float('nan'))
            verdict = 'n/a'
            if off and np.isfinite(on) and abs(on) > 1e-9:
                keep = float(np.mean(off)) / on
                verdict = ('yes' if keep >= 0.7 else 'partial' if keep >= 0.4 else 'no') + \
                          f' ({100*keep:.0f}% retained)'
            L.append(f'| adv_seed{aseed}_stealth{stealth} ({ckpt}) | ' +
                     ' | '.join(cells) + f' | {verdict} |')

    passed = [r for r in gate_rows if r['passes_primary']]
    L.append(f'\n## Verdict\n\n**{len(passed)} of {len(gate_rows)} evaluated adversaries pass '
             f'the gate.**\n')
    if not passed:
        L.append('No adversary is usable for the detection experiments yet. The spec is '
                 'explicit: *"An adversary that fails the efficacy gate is not usable. '
                 'Retune the reward weights and retrain rather than proceeding with a weak '
                 'attack."*\n')

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, 'w').write('\n'.join(L) + '\n')
    print(f'Wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--results-dir', default=RESULTS_DIR)
    p.add_argument('--clean-dir', default=CLEAN_DIR)
    p.add_argument('--horizon', action='store_true', help='print only the milestone sweep')
    args = p.parse_args()

    runs, clean = load_runs(args.results_dir, args.clean_dir)
    print(f'Loaded {len(runs)} Tier 3 rollouts and {len(clean)} clean references')
    if not runs:
        raise SystemExit('No Tier 3 rollouts. Run scripts/byz_eval_adv.sh first.')

    rows = build_rows(runs, clean)
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(OUT_CSV)), exist_ok=True)
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f'Wrote {OUT_CSV}  ({len(rows)} rows)')
    write_markdown(rows, OUT_MD)

    show = [r for r in rows if r['ckpt'].startswith('m')] if args.horizon else rows
    # Full names: the adversary variants share a 40-character prefix, so truncating hid which
    # rows were which (e.g. lieonly vs lieonlyplaus, and every milestone).
    w = max([len(r['run']) for r in show] + [3])
    print(f"\n{'run':{w}s} {'honest%':>8s} {'95% CI':>16s} {'p':>6s} {'own%':>7s} "
          f"{'surv':>5s} {'sat':>5s}  gate")
    for r in sorted(show, key=lambda x: (x['adv_seed'], re.sub(r'_on_s\d.*', '', x['run']),
                                         x['checkpoint_steps'], x['baseline_seed'])):
        ci = f"[{fmt(r['honest_deg_ci_lo'])}, {fmt(r['honest_deg_ci_hi'])}]"
        print(f"{r['run']:{w}s} {fmt(r['honest_goal_rmse_deg_pct']):>8s} {ci:>16s} "
              f"{fmt(r['honest_deg_p'],3):>6s} {fmt(r['own_goal_rmse_deg_pct']):>7s} "
              f"{fmt(100*r['survival'],0):>5s} "
              f"{fmt(100*r['saturation'],0):>5s}  "
              f"{('PASS' if r['n_episodes'] >= CONFIRMATORY_MIN_EPISODES else 'CANDIDATE') if r['passes_primary'] else 'fail'}"
              f"{'' if r.get('paired') else '  [UNPAIRED - not reportable]'}")


if __name__ == '__main__':
    main()
