#!/usr/bin/env python3
"""Run one of Stage B: choose tau. spec.md 3.2 配對, plan.md P1 ③.

    python training/SuperPathPoint/cli/inspect_survival.py

Outputs (in result/<SLURM_JOB_NAME or InspectSurvival>/):
    tau_curve.csv, offsets.csv, scores.csv
    figures/tau_curve.png, figures/offsets.png, figures/scores.png

THIS DELIBERATELY DOES NOT REPORT THE SIX PATTERNS OR THE ATTRIBUTION
======================================================================
Both are downstream of tau, and tau is what this run exists to choose. Printing
a band fraction here would be printing `alpha = 1.5`, which nobody measured
(ClaudeRules 8: the first run of a thing with a free parameter is a calibration
run, and its output is the curve rather than the answer). `report_survival.py`
is the run that reports them, after this one has picked a value.

WHAT tau IS AND WHY IT CANNOT BE A CONSTANT
=============================================
Two points at two rungs are the same point when their level-0 distance is
within tau. A coarse rung's pixel IS `ds` level-0 pixels, so a point that
really exists there cannot be located better than that -- and a tau written as
a fixed level-0 number therefore makes every coarse rung unable to match BY
DEFINITION. The output would read as "keypoints all die at coarse resolution":
a wrong answer that looks like a discovery.

So `tau = max(tau_floor_um / mpp_0, alpha * ds)` and this run measures alpha.

WHAT TO READ: THE KNEE, NOT THE PEAK
======================================
The match rate rises with tau until tau is wide enough to catch anything at
all, so the highest rate is always the widest tau and means nothing. The useful
value is where the real rate stops rising faster than the DECOY -- the same
comparison against a set displaced past every tau. Beyond that point the extra
matches are what any point set of that density would have got.

`offsets.png` is the other half of the same choice: it is where the probed peak
actually sits, per rung. A tau below the 90th percentile of that distribution
is declaring a tenth of the real matches dead by construction.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
           os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _paths import RESULT_DIR, job_result_dir, setup_import_paths  # noqa: E402

setup_import_paths()

import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

from PointsAnalysisByMpp import Report, SurvivalTable          # noqa: E402

DEFAULT_TABLE_ROOT = os.path.join(RESULT_DIR, 'BuildSurvival')

#: The alphas swept. tau = alpha * ds level-0 px, so alpha is "how many COARSE
#: PIXELS of slack" -- which is the unit the bound is in, and the reason the
#: ladder is dimensionless.
#:
#: IT RUNS TO 50 BECAUSE THE GAP IS GUARANTEED TO TURN OVER, and a sweep that
#: stops before it reports its own last point as the answer. As tau grows both
#: the real rate and the decoy rate converge to the SAME limit -- every anchor
#: with any partner at all -- so `gap = match - decoy` starts at 0, rises, and
#: returns to 0. There is always an interior maximum; the only question is
#: whether the sweep reached it. The 2026-09-01 sweep stopped at 6 and reported
#: 6 for five of ten rungs, which is what a truncated sweep looks like.
#:
#: The margin (`match / decoy`) has no such maximum and is NOT what the knee is
#: read off: it is enormous where the decoy is zero and decays monotonically to
#: 1, so its "peak" is always the smallest alpha swept. It is printed beside
#: the knee because it says how much better than chance the chosen point is,
#: which the gap alone does not.
DEFAULT_ALPHAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                  8.0, 12.0, 16.0, 24.0, 32.0, 50.0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tables', default=DEFAULT_TABLE_ROOT,
                    help='where build_survival.py wrote its stores')
    ap.add_argument('--stack-kind', nargs='+', default=['F', 'R'],
                    choices=('F', 'R'))
    ap.add_argument('--alphas', type=float, nargs='+',
                    default=list(DEFAULT_ALPHAS))
    ap.add_argument('--threshold', type=float, default=0.015,
                    help='the detection threshold the curve is measured at. '
                         'One value, not a sweep: tau and the threshold are '
                         'two knobs and turning both at once produces a '
                         'surface nobody can read a knee off')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('InspectSurvival')
    figures = os.path.join(out_dir, 'figures')
    os.makedirs(figures, exist_ok=True)

    curve, offsets, scores = [], [], []
    for kind in args.stack_kind:
        paths = SurvivalTable.find(args.tables, stack_kind=kind)
        if not paths:
            print(f'[{kind}] no tables under {args.tables}; skipped', flush=True)
            continue
        for path in paths:
            batch, meta = SurvivalTable.load(path)
            print(f'[{kind}] {meta.wsi_stem}   {len(batch)} points, '
                  f'{meta.n_chains} chains, rungs {meta.rungs}', flush=True)
            curve += Report.tau_curve(batch, meta, alphas=args.alphas,
                                      threshold=args.threshold)
            offsets += Report.offset_quantiles(batch, meta)
            scores += _score_rows(batch, meta)

    if not curve:
        raise SystemExit(
            f'no survival tables under {args.tables}. Run build_survival.py '
            f'first -- this reads its output, it does not produce it')

    _write(os.path.join(out_dir, 'tau_curve.csv'), curve)
    _write(os.path.join(out_dir, 'offsets.csv'), offsets)
    _write(os.path.join(out_dir, 'scores.csv'), scores)

    _plot_tau(curve, os.path.join(figures, 'tau_curve.png'))
    _plot_offsets(offsets, os.path.join(figures, 'offsets.png'))
    _plot_scores(scores, os.path.join(figures, 'scores.png'))

    _report(curve, offsets)
    print(f'\nSaved to {out_dir}')
    return 0


def _score_rows(batch, meta):
    """The score distribution per rung, as quantiles.

    Here rather than in `Report` because it is not a survival quantity -- it is
    what the detector said, and it exists to pick the three thresholds the
    ANALYSIS run sweeps. A sweep whose points all sit in the flat part of this
    distribution moves nothing and proves nothing.
    """
    rows = []
    for j, ds in enumerate(meta.rungs):
        column = batch.score[:, j]
        for q in (0.5, 0.75, 0.9, 0.99):
            rows.append({'wsi_stem': meta.wsi_stem,
                         'stack_kind': meta.stack_kind, 'ds': float(ds),
                         'quantile': q,
                         'score': float(np.quantile(column, q))
                                  if len(column) else float('nan')})
    return rows


def _write(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ── figures ──────────────────────────────────────────────────────────────────

def _by(rows, *keys):
    out = {}
    for row in rows:
        out.setdefault(tuple(row[k] for k in keys), []).append(row)
    return out


def _plot_tau(rows, path):
    """Match rate against alpha, one panel per axis, one line per rung.

    The decoy is drawn dashed in the SAME colour as its rung rather than in one
    colour of its own: what has to be read is the gap between a line and its
    own decoy, and two colour scales make that a comparison across the legend.
    """
    kinds = sorted({r['stack_kind'] for r in rows})
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.5),
                             squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        here = [r for r in rows if r['stack_kind'] == kind]
        rungs = sorted({r['ds'] for r in here})
        colours = plt.cm.viridis(np.linspace(0, 0.9, len(rungs)))
        for colour, ds in zip(colours, rungs):
            line = sorted((r for r in here if r['ds'] == ds),
                          key=lambda r: r['alpha'])
            alpha = [r['alpha'] for r in line]
            ax.plot(alpha, [r['match_rate'] for r in line], '-o', color=colour,
                    markersize=3, label=f'ds {ds:g}')
            ax.plot(alpha, [r['decoy_rate'] for r in line], '--', color=colour,
                    linewidth=1, alpha=0.6)
        ax.set_title(f"'{kind}' axis   (dashed = shifted decoy)")
        ax.set_xlabel('alpha   (tau = alpha * ds level-0 px)')
        ax.set_ylabel('match rate')
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle('READ THE KNEE, NOT THE PEAK: the widest tau always matches '
                 'most', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_offsets(rows, path):
    """Where the probed peak sits, per rung, as quantiles in level-0 px.

    Drawn against `alpha * ds` lines rather than raw pixels, so the question it
    answers is the one being decided: at which alpha does the band cover the
    90th percentile.
    """
    kinds = sorted({r['stack_kind'] for r in rows})
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.5),
                             squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        here = [r for r in rows if r['stack_kind'] == kind]
        rungs = sorted({r['ds'] for r in here})
        for q, style in ((0.5, '-o'), (0.9, '-s'), (0.99, '-^')):
            value = [np.mean([r['offset_l0'] for r in here
                              if r['ds'] == ds and r['quantile'] == q])
                     for ds in rungs]
            ax.plot(rungs, value, style, markersize=4, label=f'p{q * 100:.0f}')
        for alpha, colour in ((1.0, '0.7'), (1.5, '0.5'), (3.0, '0.3')):
            ax.plot(rungs, [alpha * d for d in rungs], ':', color=colour,
                    linewidth=1, label=f'tau at alpha {alpha:g}')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.set_title(f"'{kind}' axis")
        ax.set_xlabel('ds')
        ax.set_ylabel('offset, level-0 px')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7)
    fig.suptitle('a tau below the p90 line declares a tenth of the real '
                 'matches dead by construction', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_scores(rows, path):
    kinds = sorted({r['stack_kind'] for r in rows})
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.5),
                             squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        here = [r for r in rows if r['stack_kind'] == kind]
        rungs = sorted({r['ds'] for r in here})
        for q, style in ((0.5, '-o'), (0.9, '-s'), (0.99, '-^')):
            ax.plot(rungs, [np.mean([r['score'] for r in here
                                     if r['ds'] == ds and r['quantile'] == q])
                            for ds in rungs], style, markersize=4,
                    label=f'p{q * 100:.0f}')
        ax.axhline(0.015, color='r', linestyle='--', linewidth=1,
                   label='0.015 (the label cut)')
        ax.axhline(1 / 65, color='0.5', linestyle=':', linewidth=1,
                   label='1/65 = what a flat detector emits')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(f"'{kind}' axis")
        ax.set_xlabel('ds')
        ax.set_ylabel('probed score')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7)
    fig.suptitle('pick the analysis thresholds from here: a sweep inside the '
                 'flat part moves nothing', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _report(curve, offsets):
    """The knee per axis and rung, printed so the choice is on the record.

    THE RULE: the alpha at which `gap = match - decoy` is largest, with the
    two rates AVERAGED OVER SLIDES first. That is a
    real maximum and not a heuristic cut-off -- both rates converge to the same
    limit as tau grows, so the gap must rise and come back, and its peak is the
    widest tau that is still buying more signal than density.

    A PEAK AT EITHER END OF THE SWEEP IS NOT A PEAK and is called out as such.
    At the top it means the sweep was too short; at the bottom it means the
    decoy is already winning at the tightest tau tried, which is a finding
    about the corpus rather than a value to use.

    It used to be "the first alpha whose gain is not positive", which read a
    FLAT START as a knee (`gains <= 0` includes zero), and then "the first
    alpha reaching 95 per cent of the max gap", which reports the sweep's last
    point whenever the sweep is too short. Both were fixed on 2026-09-01.
    """
    print('\ntau calibration -- the gap peak per axis and rung')
    print(f"  {'axis':5s} {'ds':>5s} {'alpha':>6s} {'gap':>7s} {'match':>7s} "
          f"{'decoy':>7s} {'margin':>7s} {'n':>4s}   note")
    for (kind, ds), rows in sorted(_by(curve, 'stack_kind', 'ds').items()):
        # AVERAGED OVER SLIDES FIRST. `_by` groups every slide's rows together,
        # so a bare argmax over them picks ONE slide's best alpha and prints
        # that slide's rates -- which slide depending on the order they were
        # loaded in. On 2026-09-01 that made F ds 1 and R ds 1 disagree about
        # whether the peak was at the bottom of the sweep, for no reason but
        # which of two slides happened to hold the larger gap.
        alphas = sorted({r['alpha'] for r in rows})
        slides = len({r['wsi_stem'] for r in rows})
        mean = {a: (float(np.mean([r['match_rate'] for r in rows
                                   if r['alpha'] == a])),
                    float(np.mean([r['decoy_rate'] for r in rows
                                   if r['alpha'] == a])))
                for a in alphas}
        gap = [mean[a][0] - mean[a][1] for a in alphas]
        best = int(np.argmax(gap))
        match, decoy = mean[alphas[best]]
        margin = match / max(decoy, 1e-9)

        # Is the match rate flat across the sweep? Then the peak sits at the
        # tightest tau only because the decoy grows, and "the peak" is not a
        # choice the data made. It is the expected shape at ds 1, where one
        # pixel IS one level-0 pixel and there is no localisation slack to
        # tolerate.
        rates = [mean[a][0] for a in alphas]
        flat = (max(rates) - min(rates)) < 1e-6

        note = []
        if best == len(alphas) - 1:
            note.append('AT THE TOP OF THE SWEEP -- sweep further')
        elif best == 0 and flat:
            note.append('match is FLAT -- the tightest tau is best because the '
                        'decoy only grows; expected at ds 1')
        elif best == 0:
            note.append('AT THE BOTTOM -- the decoy already wins')
        print(f"  {kind:5s} {ds:5g} {alphas[best]:6.2f} {gap[best]:7.3f} "
              f"{match:7.3f} {decoy:7.3f} {margin:7.1f} {slides:4d}   "
              f"{'; '.join(note)}")

    print('\n  The peak is where `match - decoy` is largest. Both rates go to '
          'the same limit as tau grows, so the gap always turns over -- a peak '
          'at the end of the sweep means the sweep, not the data.')
    print('  `margin` is `match / decoy` at that point. It has no peak of its '
          'own (it decays monotonically to 1), so it says how much better than '
          'chance the choice is, not where the choice is.')
    print('  Cross-check against offsets.png: a tau under the p90 line kills a '
          'tenth of the real matches by construction.')
    print('\n  NOT REPORTED HERE ON PURPOSE: the six patterns and the '
          'attribution. Both are downstream of tau, so quoting them from this '
          'run quotes the default (plan.md P1).')


if __name__ == '__main__':
    sys.exit(main())
