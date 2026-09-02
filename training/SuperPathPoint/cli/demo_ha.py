#!/usr/bin/env python3
"""What Homographic Adaptation actually does to a label, as one picture.

    python training/SuperPathPoint/cli/demo_ha.py --wsi <slide> --ds 4
    python training/SuperPathPoint/cli/demo_ha.py --wsi <slide> --num 10 100

Panels, left to right:

    view 0        the identity view -- the tile as stored -- with the teacher's
                  OWN keypoints on it. This is what you get with NO HA
    view 1, 2     two of the homography draws HA made, each with the teacher's
                  own keypoints IN THAT VIEW's frame
    aggregate     the identity view again, carrying the HA label at three
                  thresholds at once

One aggregate panel per `--num`, so `--num 10 100` puts the same three views
beside two aggregates and the only thing that differs is how many views were
averaged.

WHY THE FIRST THREE PANELS ARE THE POINT
------------------------------------------
The teacher is not viewpoint-invariant. Each view is the same tissue presented
differently and the detector fires on a DIFFERENT subset in each -- that is the
premise HA rests on, and it is a premise about these weights on this stain,
which nobody here has checked. If the three view panels show nearly identical
point sets, HA is averaging a hundred copies of one answer and buys nothing. If
they show wildly disjoint sets, the aggregate is mostly noise. The useful case
is in between and this figure is how you tell which one you are in.

THE THRESHOLDS ARE ONE NUMBER SEEN SEVERAL WAYS
-------------------------------------------------
`mean_prob` is the average of the teacher's probability over the views that
could see each pixel, so a pixel found by k of n views at probability p lands
near `k*p/n`. The threshold therefore chooses where the label sits between
"any view found it" and "every view found it":

    0.001   upstream's training/evaluation value. Nearly the union
    0.005   superpoint_pytorch's INFERENCE default -- a different step
    0.015   upstream's HA EXPORT value (magic-point_coco_export.yaml:9), which
            is the step this reproduces, and the one make_ha_labels now cuts at

`--thresholds` replaces that list with any other, and `--cut` says which of them
the store is actually written at (drawn as the red cross). Looking at a value
ABOVE the cut costs nothing and changes nothing: the aggregate is computed once
and every threshold is a comparison on the same array, and this script writes no
store.

    --thresholds 0.001 0.005 0.015 0.025 --cut 0.015

They nest: every 0.015 point is also a 0.005 point and a 0.001 point, so the
markers are drawn largest-set-first and the nesting is visible.

The single-view panels are drawn at the same 0.015 for one number throughout,
and that comparison needs stating: a single view's probability and a 100-view
mean are both in [0, 1] but they are not the same quantity. Suppressing what
only one view saw is the entire mechanism, so the aggregate having FEWER points
than a single view at the same threshold is the mechanism working, not a bug.

NO PRE-TILE STORE NEEDED
--------------------------
`--wsi` cuts a pre-tile straight off the slide through the same `DsLadder` plan
that `extract_pretiles` uses, so this runs while step 3c is still going.
`--pretile-root` reads a stored one instead, once there is one.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # training/SuperPathPoint/

import numpy as np                                              # noqa: E402
import matplotlib                                               # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402

from cli import job_result_dir, setup_import_paths              # noqa: E402

setup_import_paths()

import torch                                                    # noqa: E402

import PreTileStore                                 # noqa: E402
from common.Homography import sample_homography                 # noqa: E402
from common.HomographyConfig import HomographyConfig            # noqa: E402
from common.KeypointLabelStore import points_from_prob          # noqa: E402
from PreTileStore import (PRE_TILE_FACTOR, centre_crop,  # noqa: E402
                                 centre_margin, pre_tile_px,
                                 warp_from_pretile)
from SuperPoint.HomographicAdaptation import HaConfig           # noqa: E402
from SuperPoint.Teacher import TeacherConfig                    # noqa: E402

#: The DEFAULT rungs, the same three as `make_ha_labels.THRESHOLD_LADDER`. Not
#: imported from it, and that is the one duplication here worth arguing about:
#: importing would drag a CLI's argparse into a figure script. `--thresholds`
#: replaces them, which is how a value ABOVE the cut gets looked at without
#: re-cutting anything -- nothing here writes a store.
THRESHOLDS = (0.001, 0.005, 0.015)

# ── the three places to change how this looks ────────────────────────────────
#
# Every dict below is handed straight to `plt.scatter(**style)`, so any scatter
# keyword works: marker, s (area, not radius), c, facecolors, edgecolors,
# linewidths, alpha, zorder. Nothing else in this file needs touching.

#: The most permissive rung. It marks EVERY point, including the ones the
#: stricter rungs also mark, so it is the one that disappears -- painted first
#: it sits under every larger marker, and at s=6 grey it was invisible even
#: where nothing covered it. Fixed both ways: the highest zorder, and a light
#: fill with a thin dark edge so it reads on pink, purple and brown alike.
#:
#: The cost of putting it on top is that every cross and circle now has a dot
#: in its middle. That is a target, not a collision -- and the alternative
#: (painting it under) is the version that could not be seen.
_DOT = dict(marker='.', s=16, c='#f5f5f5', edgecolors='#202124',
            linewidths=0.35, zorder=5)

#: The cut -- what `make_ha_labels` actually writes into the store. The only
#: cross, the only red, and the one the legend calls out in capitals. Everything
#: else on the figure is a report.
_CUT = dict(marker='+', s=48, c='#006000', linewidths=1.2, zorder=4)

#: Everything between the two, permissive to strict. Open circles so they do
#: not hide what is inside them, growing outward so the nesting reads as rings.
#: Add colours here to support a longer ladder; the last one repeats if the
#: ladder outruns the list.
_RAMP = ('#613030', '#336666', '#6C3365', '#977C00')


def styles_for(thresholds, cut: float) -> dict:
    """A marker per threshold, and one of them marked as the store's cut.

    Generated rather than tabulated, because `--thresholds` is a list of any
    length. Three things are load-bearing and none is decoration:

    SIZE GROWS WITH THE THRESHOLD, so the nesting reads as rings inside rings --
    every 0.025 point is also a 0.015 point and a 0.001 point.

    ZORDER RUNS THE OTHER WAY. Paint order would bury the permissive rung under
    every stricter one, because it is drawn at every point. `zorder` overrides
    paint order, so the dot goes on top and the rings stay open.

    THE CUT IS THE ONLY CROSS AND THE ONLY RED, so a reader does not have to
    work out from the legend which rung is the one that was written down.
    """
    rungs = sorted(float(t) for t in thresholds)
    out = {}
    ring = 0
    for i, t in enumerate(rungs):
        if abs(t - float(cut)) < 1e-12:
            out[t] = dict(_CUT)
        elif i == 0:
            out[t] = dict(_DOT)
        else:
            out[t] = dict(marker='o', s=24 + 10 * ring, facecolors='none',
                          edgecolors=_RAMP[min(ring, len(_RAMP) - 1)],
                          linewidths=0.9, zorder=3)
            ring += 1
    return out


def read_pre_tile(args) -> tuple:
    """The pre-tile to demonstrate on, and a one-line description of it."""
    pre_px = pre_tile_px(args.tile, args.factor)

    if args.pretile_root:
        folder = PreTileStore.find_one(args.pretile_root,
                                       wsi_stem=args.wsi_stem, ds=args.ds,
                                       tile=args.tile)
        meta = PreTileStore.load_meta(folder)
        records = PreTileStore.load_index(folder)
        record = records[int(args.index) % len(records)]
        pre = PreTileStore.read_tile(folder, record)
        return pre, (f'{meta.wsi_stem}  ds {meta.ds:g}  '
                     f'pre-tile #{record.index} at level-0 '
                     f'({record.x}, {record.y})')

    from demo_homography import wsi_tile                        # noqa: PLC0415

    pre = wsi_tile(args.wsi, args.ds, pre_px, args.seed)
    return pre, f'{os.path.basename(args.wsi)}  ds {args.ds:g}  seed {args.seed}'


def replay_draws(tile: int, seed: int, count: int) -> list:
    """The first `count` homographies HA drew, redrawn.

    `HomographicAdaptation.run` takes a generator and consumes it in a loop, so
    a second generator seeded the same way yields the same sequence. That is
    what makes the view panels the ACTUAL views the aggregate was built from
    rather than a fresh sample that merely looks like them -- and it costs a
    second `default_rng(seed)` instead of an API change.
    """
    rng = np.random.default_rng(seed)
    shape = (int(tile), int(tile))
    return [sample_homography(shape, rng=rng, **HomographyConfig().kwargs())
            for _ in range(int(count))]


def single_view_points(teacher, image, args) -> np.ndarray:
    """The teacher's own keypoints in one view's own frame. No HA."""
    prob = teacher.dense_prob([image]).detach().float().cpu().numpy()[0]
    xy, _, _ = points_from_prob(prob, None,
                               score_threshold=args.view_threshold,
                               nms_radius=args.nms_radius, border=args.border,
                               max_points=None)
    return xy


def draw_points(axis, xy, style) -> None:
    if not len(xy):
        return
    axis.scatter(xy[:, 0], xy[:, 1], **style)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wsi',
                    default='/work/u26130998/datasets/histoimage.na.icar.cnr.it/'
                            'BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs')
    ap.add_argument('--ds', type=float, default=4.0)
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--factor', type=int, default=PRE_TILE_FACTOR)
    ap.add_argument('--num', type=int, nargs='+', default=[100],
                    help='views to aggregate. One panel per value, so '
                         '`--num 10 100` shows what the tenfold cost buys')
    ap.add_argument('--views', type=int, default=2,
                    help='how many WARPED views to show beside the identity')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--view-threshold', type=float, default=0.015,
                    help="the cut on a SINGLE view's own probability. Same "
                         'number as the aggregate cut, deliberately -- see the '
                         'module docstring on why they are not the same '
                         'quantity')
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=list(THRESHOLDS),
                    help='which cuts to draw on the aggregate, all in one '
                         'panel. They nest, so any number of them is readable. '
                         '--cut is added automatically if it is not listed')
    ap.add_argument('--cut', type=float, default=0.015,
                    help='which of them make_ha_labels writes into the store. '
                         'Drawn as the red cross and named in the legend')
    ap.add_argument('--nms-radius', type=int, default=4)
    ap.add_argument('--border', type=int, default=4)
    ap.add_argument('--pretile-root', default=None,
                    help='read a stored pre-tile instead of cutting one')
    ap.add_argument('--wsi-stem', default=None)
    ap.add_argument('--index', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available()
                    else 'cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    tile = int(args.tile)
    margin = centre_margin(tile, args.factor)
    shape = (tile, tile)
    rungs = sorted(set(float(t) for t in args.thresholds) | {float(args.cut)})
    style = styles_for(rungs, args.cut)
    # The view panels are cut at --view-threshold, which is a different flag
    # from --cut and only defaults to the same number. Reaching for
    # `style[max(rungs)]` was right while the ladder ended at the cut and became
    # wrong the moment a rung was added above it.
    view_style = style.get(float(args.view_threshold), dict(_CUT))

    pre, source = read_pre_tile(args)
    print(f'pre-tile {pre.shape[0]}x{pre.shape[1]}  tile {tile}  '
          f'factor {args.factor}  margin {margin}')
    print(f'  {source}')

    device = torch.device(args.device)
    print(f'\nloading the teacher on {device} ...')
    teacher = TeacherConfig().build(device)

    identity = centre_crop(pre, tile)
    samples = replay_draws(tile, args.seed, args.views)
    views = [('view 0  identity', identity)]
    for i, sample in enumerate(samples, 1):
        views.append((f'view {i}  drawn',
                      warp_from_pretile(pre, sample.matrix, margin, shape)))

    print('\nsingle-view keypoints (no HA), cut at '
          f'{args.view_threshold:g}:')
    view_points = []
    for title, image in views:
        xy = single_view_points(teacher, image, args)
        view_points.append(xy)
        print(f'  {title:20s} {len(xy):5d} pts')

    # How much the views actually disagree, which is the premise HA rests on.
    # Same-frame comparison only makes sense for the identity view against
    # itself, so this is reported as a spread rather than as a match rate.
    counts = [len(xy) for xy in view_points]
    print(f'  spread across the {len(counts)} views: min {min(counts)}, '
          f'max {max(counts)}, mean {np.mean(counts):.0f}')

    aggregates = []
    for num in args.num:
        ha = HaConfig(num=int(num)).build(teacher)
        result = ha.run(pre, tile, rng=np.random.default_rng(args.seed),
                        factor=args.factor)
        per_threshold = {}
        for t in rungs:
            xy, _, _ = points_from_prob(result.mean_prob, result.counts,
                                        score_threshold=t,
                                        nms_radius=args.nms_radius,
                                        border=args.border, max_points=None)
            per_threshold[t] = xy
        aggregates.append((num, result, per_threshold))

        base = max(len(per_threshold[rungs[0]]), 1)
        print(f'\nHA aggregate, num={num}   '
              f'counts {result.counts.min():.0f}-{result.counts.max():.0f} '
              f'views per pixel')
        for t in rungs:
            n = len(per_threshold[t])
            mark = '  <- the store cuts here' if abs(t - args.cut) < 1e-12 else ''
            print(f'  {t:<7g} {n:5d} pts   {100.0 * n / base:5.1f}% of the '
                  f'{rungs[0]:g} set   '
                  f'{100.0 * (1 - n / base):5.1f}% filtered{mark}')

    n_panels = len(views) + len(aggregates)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.8))
    axes = np.atleast_1d(axes)

    for axis, (title, image), xy in zip(axes, views, view_points):
        axis.imshow(image)
        draw_points(axis, xy, view_style)
        axis.set_title(f'{title}\n{len(xy)} pts, single view @ '
                       f'{args.view_threshold:g}', fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    for axis, (num, result, per_threshold) in zip(axes[len(views):], aggregates):
        axis.imshow(identity)
        for t in rungs:                           # permissive first: they nest
            draw_points(axis, per_threshold[t], style[t])
        line = '  '.join(f'{t:g}:{len(per_threshold[t])}' for t in rungs)
        axis.set_title(f'HA aggregate  num={num}\n{line}', fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    notes = {0.001: 'upstream training/eval', 0.005: 'pytorch inference default',
             0.015: 'upstream HA export'}
    handles = []
    for t in rungs:
        st = style[t]
        note = notes.get(t, '')
        if abs(t - args.cut) < 1e-12:
            note = (note + ', ' if note else '') + 'WHAT THE STORE CUTS AT'
        handles.append(plt.Line2D(
            [], [], linestyle='none', marker=st['marker'],
            markerfacecolor='none',
            color=st.get('c', st.get('edgecolors', '#000000')),
            label=f'{t:g}' + (f'  {note}' if note else '')))
    fig.legend(handles=handles, loc='lower center',
               ncol=min(len(handles), 4), frameon=False, fontsize=9)
    fig.suptitle(f'Homographic Adaptation   {source}   tile {tile}, '
                 f'pre-tile {pre.shape[0]}', fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    out = args.out or os.path.join(job_result_dir('SuperPathPointDemo'),
                                   f'ha_demo__num{"-".join(map(str, args.num))}.png')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved  {out}')
    print('\n  Read the three view panels first. If they carry nearly the same '
          'points,\n  HA is averaging a hundred copies of one answer; if they '
          'are disjoint, the\n  aggregate is mostly noise. The useful case is '
          'in between.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
