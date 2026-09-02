#!/usr/bin/env python3
"""Run two of Stage B: the three numbers Stage C turns on. plan.md P1 ④.

    python training/SuperPathPoint/cli/report_survival.py --tau-alpha 1.5

Outputs (in result/<SLURM_JOB_NAME or ReportSurvival>/):
    patterns.csv, attribution.csv, cross.csv
    figures/survival_matrix.png, figures/threshold_sweep.png,
    figures/examples__<pattern>.png

WHAT THIS RUN IS FOR (plan.md P1)
===================================
    A  the band fraction        -> can Stage C's head be two outputs? 0.97 yes,
                                   0.6 and the simplification is a silent error
    B  the late-born fraction   -> is there anything for that head to learn? if
                                   this is near zero, Stage C does not need doing
    C  the one-rung-only share  -> can stage 1's mpp estimate use a scale
                                   signature?

NONE OF THE THREE IS DONE UNTIL IT HAS A SWEEP AND A NULL BESIDE IT. A fraction
at one threshold is not a finding; the finding is either that it barely moves
across a plausible range or that it does, and in the second case there is no
finding. And a fraction with no null is not a finding either: bands are common
on any corpus where most points survive most rungs, and the null holds the
MEASURED per-rung rates fixed so that an excess is about structure rather than
density (`NullModel`).

RUN `inspect_survival.py` FIRST
================================
`--tau-alpha` here is a decision, and run one is what makes it one. Passing the
default 1.5 without having looked at the curve produces the same table with
nothing behind it.

THE EXAMPLE STRIPS ARE THE ONLY FALSIFIABLE PART
==================================================
Every number here is consistent with "晚生型 means a large structure came into
scale" and with "晚生型 means the detector fires on a blur artefact". The
strips -- one point, its tile at every rung, side by side -- are the only thing
in this run that can tell those apart, and the points are chosen BY RULE (fixed
seed, the 10th/50th/90th percentile of score within each pattern) because an
author's picks and a rule's picks look identical on the page.
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

import dataclasses                                             # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402

from PointsAnalysisByMpp import (Attribution, MppStack,        # noqa: E402
                                 NullModel, Report, SurvivalTable)
from PointsAnalysisByMpp.Patterns import (ASCII_NAMES,          # noqa: E402
                                          PATTERNS, classify)

DEFAULT_TABLE_ROOT = os.path.join(RESULT_DIR, 'BuildSurvival')
DEFAULT_TILE_ROOT = os.path.join(RESULT_DIR, 'cache', 'tiles_chains')

#: Three, and they have to straddle something. `inspect_survival.py`'s
#: scores.png is where they come from -- a sweep whose points all sit in the
#: flat part of the score distribution moves nothing and proves nothing.
DEFAULT_THRESHOLDS = (0.005, 0.015, 0.030)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tables', default=DEFAULT_TABLE_ROOT)
    ap.add_argument('--tiles-root', default=DEFAULT_TILE_ROOT,
                    help='for the example strips only')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--tau-alpha', type=float, default=None,
                    help='overrides what the store was built with. Pass the '
                         'value inspect_survival.py chose; leaving it None '
                         "uses the store's own, which is the calibration "
                         'default unless it was rebuilt')
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=list(DEFAULT_THRESHOLDS))
    ap.add_argument('--merge-radius-l0', type=float, default=0.0,
                    help='read-time anchor merge. 0 is the baseline the '
                         'sensitivity to it is measured against')
    ap.add_argument('--examples', type=int, default=3,
                    help='points per pattern in the strips, at evenly spaced '
                         'score percentiles')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('ReportSurvival')
    figures = os.path.join(out_dir, 'figures')
    os.makedirs(figures, exist_ok=True)

    loaded = _load(args)
    if not loaded:
        raise SystemExit(
            f'no survival tables under {args.tables}. Run build_survival.py '
            f'and then inspect_survival.py first')

    patterns, attribution, cross = [], [], []
    for stem, axes in sorted(loaded.items()):
        for kind, (batch, meta) in sorted(axes.items()):
            patterns += Report.pattern_table(
                batch, meta, thresholds=args.thresholds,
                merge_radius_l0=args.merge_radius_l0)
        if 'F' in axes and 'R' in axes:
            (f_batch, f_meta), (r_batch, r_meta) = axes['F'], axes['R']
            attribution += Report.attribution_table(
                f_batch, f_meta, r_batch, r_meta, thresholds=args.thresholds)
            cross += Report.cross_table(f_batch, f_meta, r_batch, r_meta,
                                        threshold=args.thresholds[len(args.thresholds) // 2])
        else:
            print(f'[{stem}] has {sorted(axes)} and attribution needs both '
                  f"'F' and 'R'; the birth cause cannot be told from blur "
                  f'with one axis', flush=True)

    _write(os.path.join(out_dir, 'patterns.csv'), patterns)
    _write(os.path.join(out_dir, 'attribution.csv'), attribution)
    _write(os.path.join(out_dir, 'cross.csv'), cross)

    middle = args.thresholds[len(args.thresholds) // 2]
    _plot_matrix(loaded, middle, os.path.join(figures, 'survival_matrix.png'))
    _plot_sweep(patterns, os.path.join(figures, 'threshold_sweep.png'))
    _strips(loaded, args, figures, middle)

    _report(patterns, attribution, cross, middle)
    print(f'\nSaved to {out_dir}')
    return 0


def _load(args):
    """`{stem: {kind: (batch, meta)}}`, with `--tau-alpha` applied if given."""
    out = {}
    for path in SurvivalTable.find(args.tables):
        batch, meta = SurvivalTable.load(path)
        if args.tau_alpha is not None:
            meta = dataclasses.replace(meta, tau_alpha=float(args.tau_alpha))
        out.setdefault(meta.wsi_stem, {})[meta.stack_kind] = (batch, meta)
    return out


def _write(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ── figures ──────────────────────────────────────────────────────────────────

def _plot_matrix(loaded, threshold, path, cap=4000):
    """Points x rungs, sorted by pattern. The one figure that shows bands.

    Sorted by pattern and then by birth rung, so a contiguous band appears as a
    contiguous block and a flickering population appears as speckle -- which is
    the whole question, and it is answered by looking rather than by a number.

    Capped at `cap` rows because a 500,000-row image is a grey rectangle. The
    rows are taken evenly through the sorted order rather than from the top, so
    the proportions of the picture are the proportions of the corpus.
    """
    panels = [(stem, kind, v) for stem, axes in sorted(loaded.items())
              for kind, v in sorted(axes.items())]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(2.6 * len(panels) + 1, 5), squeeze=False)
    for ax, (stem, kind, (batch, meta)) in zip(axes[0], panels):
        alive = Report.alive_of(batch, meta, threshold)
        order = sorted(range(len(alive)),
                       key=lambda i: (PATTERNS.index(classify(alive[i])[0])
                                      if classify(alive[i])[0] in PATTERNS
                                      else len(PATTERNS),
                                      int(np.argmax(alive[i])) if alive[i].any()
                                      else -1))
        if len(order) > cap:
            order = [order[i] for i in
                     np.linspace(0, len(order) - 1, cap).astype(int)]
        ax.imshow(batch.score[order], aspect='auto', cmap='magma',
                  interpolation='nearest',
                  extent=[0, len(meta.rungs), len(order), 0])
        ax.set_xticks(np.arange(len(meta.rungs)) + 0.5)
        ax.set_xticklabels([f'{d:g}' for d in meta.rungs], fontsize=7)
        ax.set_title(f'{stem[:14]}  {kind}', fontsize=8)
        ax.set_xlabel('ds')
        ax.set_ylabel('points, sorted by pattern')
    fig.suptitle(f'probed score per (point, rung), threshold {threshold:g}   '
                 f'-- a band is a block, a flicker is speckle', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_sweep(rows, path):
    """The three numbers against the threshold. A steep line is a non-finding.

    The null is drawn as a dashed line of the same colour, for the reason the
    tau figure gives: what has to be read is the gap between a fraction and its
    own null, and two colour scales make that a comparison across the legend.
    """
    want = ('一直存活', '晚生型', '只在一階')
    kinds = sorted({r['stack_kind'] for r in rows})
    fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 4.5),
                             squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        here = [r for r in rows if r['stack_kind'] == kind]
        thresholds = sorted({r['threshold'] for r in here})
        for colour, name in zip(plt.cm.tab10(range(len(want))), want):
            measured = [np.mean([r['frac'] for r in here
                                 if r['threshold'] == t and r['pattern'] == name])
                        for t in thresholds]
            null = [np.mean([r['null_frac'] for r in here
                             if r['threshold'] == t and r['pattern'] == name])
                    for t in thresholds]
            ax.plot(thresholds, measured, '-o', color=colour,
                    label=ASCII_NAMES.get(name, name))
            ax.plot(thresholds, null, '--', color=colour, linewidth=1,
                    alpha=0.7)
        ax.set_xscale('log')
        ax.set_title(f"'{kind}' axis   (dashed = independent-rungs null)")
        ax.set_xlabel('detection threshold')
        ax.set_ylabel('share of points')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('a line that moves across this range is a number with no '
                 'finding behind it', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _strips(loaded, args, figures, threshold):
    """One point per row, its tile at every rung. The falsifiable figure.

    CHOSEN BY RULE AND THE RULE IS HERE: within each pattern, the points are
    taken at evenly spaced percentiles of peak score with a fixed seed breaking
    ties. An author picking three convincing examples and a rule picking three
    produce pages that look the same, and only one of them is evidence.
    """
    rng = np.random.default_rng(args.seed)
    for stem, axes in sorted(loaded.items()):
        if 'F' not in axes:
            continue
        batch, meta = axes['F']
        chains = MppStack.chains(args.tiles_root, stem, tile=args.tile,
                                 rungs=list(meta.rungs))
        if not chains:
            print(f'[{stem}] no chains under {args.tiles_root}; no strips',
                  flush=True)
            continue
        alive = Report.alive_of(batch, meta, threshold)
        labels = [classify(row)[0] for row in alive]

        for name in PATTERNS:
            index = [i for i, p in enumerate(labels) if p == name
                     and int(batch.chain[i]) in chains]
            if not index:
                continue
            strength = batch.score[index].max(axis=1)
            jitter = rng.uniform(0, 1e-9, len(index))
            order = np.argsort(strength + jitter)
            picks = [index[order[int(q * (len(order) - 1))]]
                     for q in np.linspace(0.1, 0.9, args.examples)]
            _strip(batch, meta, picks, chains, args.tile, alive,
                   os.path.join(figures,
                                f'examples__{ASCII_NAMES[name]}.png'), name)


def _strip(batch, meta, picks, chains, tile, alive, path, name):
    """One point per row, its tile at every rung, marked ALIVE or NOT.

    THE MARK SAYS WHETHER THE RUNG DETECTED IT, not merely where the anchor
    projects to. Drawing the anchor's position at every rung -- which this did
    until 2026-09-01 -- puts a mark on all five tiles of a `one-rung-only`
    point, which reads as "the point is here at every scale": the exact
    opposite of what that pattern means, on the figure whose whole job is to
    let a reader check the patterns by eye.

    So: a solid green circle where the point is alive, a thin grey dashed one
    where it is not. The dead rungs are still marked, because "what does the
    tissue look like there when the detector says nothing" is the question a
    reader brings to this figure.
    """
    fig, axes = plt.subplots(len(picks), len(meta.rungs),
                             figsize=(1.5 * len(meta.rungs),
                                      1.7 * len(picks)), squeeze=False)
    for row, i in enumerate(picks):
        chain = chains[int(batch.chain[i])]
        stack = MppStack.f_stack(chain, tile=tile)
        for col, ds in enumerate(meta.rungs):
            ax = axes[row][col]
            image = stack.get(float(ds))
            if image is not None:
                ax.imshow(image)
                x = (batch.x0[i] - chain.members[float(ds)][1].x) / float(ds)
                y = (batch.y0[i] - chain.members[float(ds)][1].y) / float(ds)
                if 0 <= x < tile and 0 <= y < tile:
                    if alive[i][col]:
                        ax.plot([x], [y], 'o', markerfacecolor='none',
                                markeredgecolor='lime', markersize=10,
                                markeredgewidth=1.8)
                    else:
                        ax.plot([x], [y], 'o', markerfacecolor='none',
                                markeredgecolor='0.6', markersize=7,
                                markeredgewidth=0.8, linestyle=':')
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f'ds {ds:g}', fontsize=8)
            if col == 0:
                ax.set_ylabel(f'p{[10, 50, 90][row] if len(picks) == 3 else row}',
                              fontsize=7)
    fig.suptitle(f'{ASCII_NAMES.get(name, name)}   green = detected at that '
                 f'rung, grey = not.  Points chosen by rule (score '
                 f'percentiles, fixed seed), not by hand', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _report(patterns, attribution, cross, threshold):
    print(f'\nA / B / C at threshold {threshold:g}, per axis')
    print(f"  {'axis':5s} {'pattern':17s} {'frac':>7s} {'null':>7s} "
          f"{'excess':>7s} {'n':>8s}")
    for kind in sorted({r['stack_kind'] for r in patterns}):
        rows = [r for r in patterns
                if r['stack_kind'] == kind and r['threshold'] == threshold]
        for name in ('一直存活', '晚生型', '只在一階'):
            here = [r for r in rows if r['pattern'] == name]
            if not here:
                continue
            print(f"  {kind:5s} {ASCII_NAMES[name]:17s} "
                  f"{np.mean([r['frac'] for r in here]):7.3f} "
                  f"{np.mean([r['null_frac'] for r in here]):7.3f} "
                  f"{np.mean([r['excess'] for r in here]):7.3f} "
                  f"{sum(r['n'] for r in here):8d}")
        band = np.mean([r['band_fraction'] for r in rows]) if rows else float('nan')
        multi = (np.mean([r['band_fraction_multi'] for r in rows]) if rows
                 else float('nan'))
        n_multi = sum(r['n_multi'] for r in rows) // max(len(PATTERNS), 1)
        print(f"  {kind:5s} {'band, all':12s} {band:7.3f}")
        print(f"  {kind:5s} {'band, >1 rung':12s} {multi:7.3f} "
              f"{'':>15s} {n_multi:8d}   <- spec.md 3.3 reads THIS one")

    if attribution:
        print('\n新生歸因 (F axis; R is the control that subtracts blur)')
        for name in Attribution.CAUSES:
            here = [r for r in attribution
                    if r['cause'] == name and r['threshold'] == threshold]
            if here:
                print(f"  {name:24s} {np.mean([r['frac_of_late'] for r in here]):7.3f}"
                      f"   n={sum(r['n'] for r in here)}")

    if cross:
        cell = [r for r in cross if r['pattern'] == '只在一階'
                and r['cause'] == Attribution.NEIGHBOURHOOD_SCORE]
        if cell:
            print(f"\n  只在一階 x 鄰域新生（分數）: n={sum(r['n'] for r in cell)}, "
                  f"{np.mean([r['frac'] for r in cell]):.4f} of all points")
            print('  That cell is the scale-signature candidate (plan.md P1 '
                  '分析三) -- appears at exactly one rung AND because the '
                  'receptive field widened.')

    print('\n  Read the sweep figure before any of the above. A fraction that '
          'moves across the threshold range is a number with no finding behind '
          'it, and the sweep is the only thing that says which.')
    print('  Then the strips: they are the only part of this run that can '
          'falsify "晚生型 means a large structure came into scale".')


if __name__ == '__main__':
    sys.exit(main())
