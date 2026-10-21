"""
Choose the Phase 3 shift levels from their measured effect on the CLEAN swarm (D5).

Rules, fixed on 2026-09-26 before any calibration data existed:

  * The effect of a shift setting is the pooled paired honest goal-RMSE degradation of clean
    rollouts under that setting versus clean rollouts without it, at the same `--seed`
    (identical scenarios), pooled over baseline seeds 1-3, n = 50 per seed.
  * **mild** = dynamics randomisation only, at the SMALLEST ratio whose effect is >= 5 % with
    p < 0.05. If no ratio in the grid gets there, the largest ratio, flagged as weak.
  * **severe** = dynamics randomisation at min(2 x the mild ratio, 0.3) PLUS wind, at the
    SMALLEST mean wind whose combined effect is >= 25 % with p < 0.05. If no wind in the grid
    gets there, the largest, flagged. Gust std = half the mean wind, correlation time 1 s.
  * Any chosen level whose effect exceeds 100 % is flagged: the clean swarm is then failing its
    task outright and an attack's damage would be unmeasurable against it.

Selecting the smallest setting that clears each bar keeps both levels as gentle as the
definition allows, which matters because Phase 3 must still see a ~7 % attack underneath them.

AMENDMENT 2026-10-12 -- the grid, not the rules (written before any v2-grid run, and before any
attacked rollout under shift exists):

  * The v1 grid (DR 0.05-0.3, wind 0.5-2) could not deliver the rules' intent. Its smallest
    ratio already degraded the clean swarm by +32.6 % [+24.3, +40.9] -- past the 25 % *severe*
    bar -- and its severe pick (DR 0.1 + wind 0.5) by +140 %, which the rules above flag as
    destroying the task. The effect is steep in DR (0.05: +33 %, 0.1: +123 %), so the gentle
    end of the scale was never sampled.
  * Amended grid: DR adds 0.01, 0.02, 0.03, 0.04 (stage 1 scans the union with v1); stage 2
    winds are 0.1, 0.25, 0.5, 1.0 at the new severe DR. Thresholds, alpha, DR cap, gust rule,
    seeds and n are unchanged. Its picks are the PRIMARY levels, `mild` and `severe`.
  * The v1 picks are kept, unchanged, as `mild_v1` and `severe_v1` (`--freeze-v1`), and Phase 3
    runs all four levels. Every level is reported, ordered by its clean-swarm effect, whatever
    the attacks or detectors do under it. No level set is chosen after Phase 3 data exist.
  * The amendment used clean-swarm data only: the quantity being calibrated, not any outcome
    of the study (no attacked run under shift had been collected; results_v2/phase3 was empty).

    python -m byzantine_experiments.analysis.shift_calibration --stage 1   # after DR grid
    python -m byzantine_experiments.analysis.shift_calibration --stage 2   # after wind grid
"""

import argparse
import glob
import json
import os
import re

from byzantine_experiments.analysis.paired import per_episode_degradation, pooled
from byzantine_experiments.paths import RESULTS_ROOT, results_dir
from byzantine_experiments.wrappers.shift import SHIFT_LEVELS_FILE

CAL_DIR = os.path.join(RESULTS_ROOT, 'shift_calibration')
SEEDS = (1, 2, 3)
MILD_MIN, SEVERE_MIN, DESTRUCTIVE = 5.0, 25.0, 100.0
ALPHA = 0.05
DR_CAP = 0.3
# The v1 picks recorded by the first calibration (logs_v2/run_block_A_20260926_225548.log).
V1_PICKS = {'mild': (0.05, 0.0), 'severe': (0.1, 0.5)}


def effect(tag):
    """Pooled paired clean degradation for one setting tag, e.g. 'dr0.1' or 'dr0.2_wind1.0'."""
    diffs = []
    for s in SEEDS:
        run = os.path.join(CAL_DIR, f'shiftcal_s{s}_{tag}.npz')
        clean = os.path.join(results_dir('tier1'), f't1_clean_s{s}.npz')
        if not (os.path.exists(run) and os.path.exists(clean)):
            return None
        diffs.append(per_episode_degradation(run, clean, idx=list(range(8)),
                                             allow_different_shift=True))
    return pooled(diffs)


def _tags(pattern):
    out = []
    for p in glob.glob(os.path.join(CAL_DIR, f'shiftcal_s1_{pattern}.npz')):
        out.append(os.path.basename(p)[len('shiftcal_s1_'):-4])
    return out


def _num(tag, key):
    m = re.search(key + r'([\d.]+)', tag)
    return float(m.group(1)) if m else float('nan')


def stage1():
    tags = sorted((t for t in _tags('dr*') if 'wind' not in t), key=lambda t: _num(t, 'dr'))
    rows = [(t, effect(t)) for t in tags]
    rows = [(t, e) for t, e in rows if e is not None]
    if not rows:
        raise SystemExit(f'no stage-1 runs under {CAL_DIR}')
    print('Stage 1 -- dynamics randomisation only (pooled over seeds 1-3, clean vs clean):')
    for t, e in rows:
        print(f'  {t:10s} {e["mean"]:+7.1f} % [{e["ci_lo"]:+6.1f}, {e["ci_hi"]:+6.1f}]  '
              f'p={e["p_value"]:.4f}  n={e["n"]}')
    chosen = next(((t, e) for t, e in rows if e['mean'] >= MILD_MIN and e['p_value'] < ALPHA),
                  None)
    weak = chosen is None
    chosen = chosen or rows[-1]
    r = _num(chosen[0], 'dr')
    print(f'\n  mild = DR {r}{"  (WEAK: no ratio reached the 5 % bar)" if weak else ""}')
    return dict(dr_ratio=r, wind_accel=0.0, gust=0.0, tau=1.0,
                effect_pct=chosen[1]['mean'], ci=[chosen[1]['ci_lo'], chosen[1]['ci_hi']],
                weak=weak, scanned=[t for t, _ in rows])


def severe_dr(mild):
    return min(2.0 * mild['dr_ratio'], DR_CAP)


def stage2(mild):
    dr = severe_dr(mild)
    tags = sorted((t for t in _tags(f'dr{dr:g}_wind*')), key=lambda t: _num(t, 'wind'))
    rows = [(t, effect(t)) for t in tags]
    rows = [(t, e) for t, e in rows if e is not None]
    if not rows:
        raise SystemExit(f'no stage-2 runs for DR {dr:g} under {CAL_DIR}')
    print(f'Stage 2 -- DR {dr:g} + wind (pooled over seeds 1-3, clean vs clean):')
    for t, e in rows:
        print(f'  {t:18s} {e["mean"]:+7.1f} % [{e["ci_lo"]:+6.1f}, {e["ci_hi"]:+6.1f}]  '
              f'p={e["p_value"]:.4f}')
    chosen = next(((t, e) for t, e in rows if e['mean'] >= SEVERE_MIN and e['p_value'] < ALPHA),
                  None)
    weak = chosen is None
    chosen = chosen or rows[-1]
    w = _num(chosen[0], 'wind')
    print(f'\n  severe = DR {dr:g} + wind {w} m/s^2 (gust {w / 2:g})'
          f'{"  (WEAK: no wind reached the 25 % bar)" if weak else ""}')
    return dict(dr_ratio=dr, wind_accel=w, gust=w / 2.0, tau=1.0,
                effect_pct=chosen[1]['mean'], ci=[chosen[1]['ci_lo'], chosen[1]['ci_hi']],
                weak=weak, scanned=[t for t, _ in rows])


def freeze_v1(state):
    """Copy the v1 picks to `mild_v1` / `severe_v1` before the amended grid replaces `mild` /
    `severe`. Idempotent; refuses if `mild` / `severe` are not the recorded v1 picks (i.e. if
    it is run after the amended calibration, when they would be v2 levels)."""
    lv = state['levels']
    if 'mild_v1' in lv and 'severe_v1' in lv:
        print('v1 levels already frozen: ' + ', '.join(
            f"{k} = DR {lv[k]['dr_ratio']:g} + wind {lv[k]['wind_accel']:g} "
            f"({lv[k]['effect_pct']:+.1f} %)" for k in ('mild_v1', 'severe_v1')))
        return False
    for k, (dr, w) in V1_PICKS.items():
        if k not in lv or (lv[k]['dr_ratio'], lv[k]['wind_accel']) != (dr, w):
            raise SystemExit(f'{k!r} in shift_levels.json is not the recorded v1 pick (DR {dr:g} + '
                             f'wind {w:g}); refusing to label it v1')
    for k in V1_PICKS:
        lv[f'{k}_v1'] = dict(lv[k], grid='v1: DR 0.05 0.1 0.2 0.3, wind 0.5 1 2')
        print(f"froze {k}_v1 = DR {lv[k]['dr_ratio']:g} + wind {lv[k]['wind_accel']:g} "
              f"({lv[k]['effect_pct']:+.1f} %)")
    state['amendment'] = ('2026-10-12: grid extended (DR 0.01-0.04; stage-2 wind 0.1-1.0), rules '
                          'unchanged. mild/severe = amended-grid picks (primary); mild_v1/severe_v1 '
                          '= original picks, also run. All levels reported. See module docstring.')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', type=int, choices=[1, 2])
    ap.add_argument('--freeze-v1', action='store_true',
                    help='keep the original picks as mild_v1 / severe_v1 (amendment 2026-10-12)')
    args = ap.parse_args()
    if not (args.stage or args.freeze_v1):
        ap.error('give --stage 1|2 or --freeze-v1')
    path = os.path.join(RESULTS_ROOT, SHIFT_LEVELS_FILE)
    state = json.load(open(path)) if os.path.exists(path) else {'levels': {}}
    if args.freeze_v1:
        if freeze_v1(state):
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            print(f'Wrote {path}')
        return
    if args.stage == 1:
        state['levels']['mild'] = stage1()
        state['levels'].pop('severe', None)
        print(f'SEVERE_DR={severe_dr(state["levels"]["mild"]):g}')   # read by the script
    else:
        if 'mild' not in state['levels']:
            raise SystemExit('run stage 1 first')
        state['levels']['severe'] = stage2(state['levels']['mild'])
        for name, lv in state['levels'].items():
            if lv['effect_pct'] > DESTRUCTIVE:
                print(f'  ! {name} degrades the CLEAN swarm by {lv["effect_pct"]:.0f} % -- it is '
                      'failing its task; attack damage will be unmeasurable at this level')
    state['rules'] = dict(mild_min_pct=MILD_MIN, severe_min_pct=SEVERE_MIN, alpha=ALPHA,
                          dr_cap=DR_CAP, fixed='2026-09-26, before any calibration data')
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)
    print(f'\nWrote {path}')


if __name__ == '__main__':
    main()
