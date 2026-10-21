"""
Read an adversary's training curves as *trends*, not as spot samples.

Written after misreading round 4. `adv/r_team` is logged once per finished episode and is
extremely noisy -- individual samples span roughly +5 to -25 while the local mean sits near
-8. Sampling the series at a few fractions of training (`v[searchsorted(s, s[-1]*0.5)]`) and
reporting those numbers produced a single `+2.166` that looked like the objective had gone
positive for the first time in the project. It had not; the local mean there was -8.7.

So this module only ever reports **binned means**, plus a smoothed peak and an explicit
trend. There is no way to ask it for a single sample.

Round 5 then exposed two more ways to misread the same data, both fixed here:

* **Endpoint comparison.** Round 5's bins ran -9.2 (10M), -6.7 (30-50M), ... -9.6 (110M), ...
  -7.3 (190M). "First -9.2, last -7.3" reads as progress; the curve actually peaked early and
  sagged. The report now names the peak bin and says when the curve is non-monotone.
* **Statistical vs practical significance.** With ~13k logged episodes the slope's SE is tiny,
  so a slope of 1e-5 per 100M steps on `adv/survived` was flagged "moving". A trend is now
  only called moving if it is beyond 2 SE **and** the implied change over the run exceeds the
  bin-to-bin standard deviation -- i.e. it is visible above the curve's own wobble.

The second thing it fixes: `adv/r_team` is an episode **sum**, so it moves when episode
length moves. Round 3 and round 4 both shortened episodes ~30 % while the sum stayed flat,
which means the per-step objective actually changed. Every report here carries the per-step
figure (`r_team / len`) alongside the raw sum, because one of them is always misleading.

    python -m byzantine_experiments.analysis.training_curve adv_seed3_stealth0_lieonly
    python -m byzantine_experiments.analysis.training_curve <run> --key adv/lie_magnitude_m
"""

import argparse
import os

import numpy as np

TRAIN_DIR = 'train_dir'
DEFAULT_KEYS = ('adv/r_team', 'adv/survived', 'adv/lie_magnitude_m', 'len/len')


def load_scalars(run, keys, train_dir=TRAIN_DIR):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    path = os.path.join(train_dir, run, '.summary', '0')
    if not os.path.isdir(path):
        raise FileNotFoundError(f'no tensorboard summary at {path}')
    ea = EventAccumulator(path, size_guidance={
        'scalars': 20000, 'histograms': 0, 'images': 0,
        'audio': 0, 'tensors': 0, 'compressedHistograms': 0})
    ea.Reload()
    available = set(ea.Tags()['scalars'])
    out = {}
    for k in keys:
        if k not in available:
            continue
        ev = ea.Scalars(k)
        out[k] = (np.array([e.step for e in ev], dtype=np.float64),
                  np.array([e.value for e in ev], dtype=np.float64))
    return out


def binned(steps, values, n_bins=20):
    """Mean of `values` in each of `n_bins` equal-width slices of env steps."""
    edges = np.linspace(steps.min(), steps.max(), n_bins + 1)
    centres, means, counts = [], [], []
    for i in range(n_bins):
        m = (steps >= edges[i]) & (steps < edges[i + 1] if i < n_bins - 1 else steps <= edges[i + 1])
        if m.sum():
            centres.append(0.5 * (edges[i] + edges[i + 1]))
            means.append(values[m].mean())
            counts.append(int(m.sum()))
    return np.array(centres), np.array(means), np.array(counts)


def trend(steps, values):
    """
    Least-squares slope over training, reported per 100M env steps, with its standard error.

    The question every round has turned on is "did the objective move at all". A slope with an
    error bar answers that; eyeballing endpoints does not.
    """
    x = steps / 1e8
    n = len(x)
    if n < 3:
        return float('nan'), float('nan')
    slope, intercept = np.polyfit(x, values, 1)
    resid = values - (slope * x + intercept)
    denom = ((x - x.mean()) ** 2).sum()
    se = np.sqrt((resid ** 2).sum() / (n - 2) / denom) if denom > 0 else float('nan')
    return float(slope), float(se)


def _visibility_floor(bin_means):
    """
    Smallest change worth calling a change: the bin-to-bin wobble, but never below 1 % of the
    series' typical magnitude. Without the floor a near-constant series (`adv/survived` sits at
    0.994-0.995 for 200M steps) has wobble ~0 and every rounding error reads as a trend.
    """
    m = np.asarray(bin_means, dtype=np.float64)
    wobble = float(np.std(m, ddof=1)) if m.size > 1 else 0.0
    return max(wobble, 0.01 * float(np.mean(np.abs(m))))


def classify_trend(slope, se, steps, bin_means):
    """
    'flat' unless the trend is BOTH statistically and practically visible.

    Statistical: |slope| > 2 SE. Practical: the change the slope implies over the whole run
    exceeds the bin-to-bin standard deviation -- otherwise the curve's own wobble is larger than
    the trend, and calling it "moving" only reflects how many episodes were logged.
    """
    if not np.isfinite(slope) or not np.isfinite(se) or len(bin_means) < 3:
        return 'insufficient data'
    span_100m = (steps.max() - steps.min()) / 1e8
    implied = abs(slope) * span_100m
    wobble = _visibility_floor(bin_means)
    if abs(slope) <= 2 * se or implied <= wobble:
        return f'flat (implied change {implied:.3f} <= wobble {wobble:.3f})' \
            if abs(slope) > 2 * se else 'flat (within 2 SE of zero)'
    return f'**{"rising" if slope > 0 else "falling"}** (implied change {implied:.3f} > wobble {wobble:.3f})'


def report(run, keys=DEFAULT_KEYS, n_bins=20, train_dir=TRAIN_DIR):
    data = load_scalars(run, tuple(keys) + ('len/len',), train_dir)
    if not data:
        return f'{run}: no matching scalars found'

    L = [f'=== {run}']
    lens = data.get('len/len')
    for k in keys:
        if k not in data:
            L.append(f'  {k}: not logged')
            continue
        s, v = data[k]
        c, m, _ = binned(s, v, n_bins)
        sl, se = trend(s, v)
        verdict = classify_trend(sl, se, s, m)
        peak = int(np.argmax(m))
        L.append(f'\n  {k}')
        L.append(f'    bins: min {m.min():+9.3f}   max {m.max():+9.3f} (peak at ~{c[peak]/1e6:.0f}M)'
                 f'   bin-to-bin sd {np.std(m, ddof=1) if len(m) > 1 else 0.0:.3f}')
        L.append(f'    slope {sl:+.4f} +/- {se:.4f} per 100M steps -> {verdict}')
        if len(m) >= 4 and 0 < peak < len(m) - 2 and m[peak] - m[-1] > _visibility_floor(m):
            L.append(f'    NON-MONOTONE: peaked mid-run at ~{c[peak]/1e6:.0f}M and fell back. '
                     'Do not read first-vs-last bins as a trend; evaluate milestones.')

        # Episode sums are only interpretable next to episode length.
        if k == 'adv/r_team' and lens is not None:
            ls, lv = lens
            _, lm, _ = binned(ls, lv, n_bins)
            n = min(len(m), len(lm))
            per = m[:n] / np.maximum(lm[:n], 1.0)
            L.append(f'    per step (binned): min {per.min():+.5f}  max {per.max():+.5f}  '
                     f'mean {per.mean():+.5f}   (episode length {lm[0]:.0f} -> {lm[-1]:.0f})')
            L.append('    NOTE r_team is an episode SUM; read the per-step line when '
                     'episode length moves.')

        width = 44
        lo, hi = m.min(), m.max()
        span = (hi - lo) or 1.0
        for ci, mi in zip(c, m):
            bar = '#' * max(1, int(width * (mi - lo) / span))
            L.append(f'    {ci/1e6:7.0f}M {mi:+9.3f} {bar}')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='+', help='run directory names under train_dir/')
    ap.add_argument('--key', action='append', dest='keys',
                    help='scalar tag to report (repeatable); defaults to the adversary set')
    ap.add_argument('--bins', type=int, default=20)
    ap.add_argument('--train-dir', default=TRAIN_DIR)
    args = ap.parse_args()
    for run in args.runs:
        print(report(run, tuple(args.keys) if args.keys else DEFAULT_KEYS,
                     args.bins, args.train_dir))
        print()


if __name__ == '__main__':
    main()
