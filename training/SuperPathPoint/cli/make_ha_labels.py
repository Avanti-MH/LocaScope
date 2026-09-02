#!/usr/bin/env python3
"""spec.md 12 step 4: run Homographic Adaptation over a pre-tile store.

    python training/SuperPathPoint/cli/make_ha_labels.py --tile 256
    python training/SuperPathPoint/cli/make_ha_labels.py --tile 256 --limit 100

Outputs (in result/cache/keypoint_labels/ by default):
    <wsi_stem>__ds<d>__<cfg8>.safetensors
    make_ha_labels.csv          in result/<SLURM_JOB_NAME or MakeHaLabels>/

argparse, a loop, printed progress. What decides anything is in
`SuperPoint/Teacher.py` (the network and the dense decode),
`SuperPoint/HomographicAdaptation.py` (the N views and the aggregation) and
`common/KeypointLabelStore.py` (what a point is and how it lands).

THE FIRST RUN IS A MEASUREMENT, AND `--limit` IS WHAT MAKES IT ONE
-------------------------------------------------------------------
spec.md 13 lists the HA wall clock as undecided, and what it decides is whether
Stage A is one job or a project: whether R > 1 is affordable at all, and whether
`model_512` and `model_1024` are worth opening. The cost is
`tiles x N x forward(tile_size)` and the forward goes with tile AREA, so one
measurement extrapolates to all three: linear in N and in tile count, quadratic
in tile size.

    --limit 100 on one slide and one rung, N=100.  Ten minutes, not hours.

Run that before the full pass. The per-tile seconds this prints is the number
the extrapolation is made of.

WHAT THE PRINTED `n_kp` DISTRIBUTION IS FOR
--------------------------------------------
`points_per_megapixel` is a MEMORY cap and the score threshold is what selects
(spec.md 6.3). The two are distinguishable in exactly one way: if `n_kp` sits at
the cap, the threshold is not what selected -- the cap is, and the store is
lossy in a way that cannot be undone without re-running. `at_cap` is printed per
rung for that reason, and it should be near zero.

If it is not, RAISE THE CAP rather than the threshold. Lowering the cap and
raising the threshold both make the store smaller; only the second one is a
decision about what a keypoint is.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, '..', '..', '..', 'utilities'),
           os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _paths import RESULT_DIR, job_result_dir, setup_import_paths  # noqa: E402

setup_import_paths()

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

from common import KeypointLabelStore                # noqa: E402
import PreTileStore
from common.KeypointLabelStore import (LabelMeta, batch_from_lists,  # noqa: E402
                                       cap_for, points_from_prob)
from SuperPoint.HomographicAdaptation import HaConfig              # noqa: E402
from SuperPoint.Teacher import TeacherConfig                       # noqa: E402

DEFAULT_TILE_ROOT = os.path.join(RESULT_DIR, 'cache', 'tiles')
DEFAULT_LABEL_ROOT = os.path.join(RESULT_DIR, 'cache', 'keypoint_labels')

#: Upstream's HA EXPORT value, `magic-point_coco_export.yaml:9`, which is the
#: config that generates the pseudo-labels this step reproduces. It carries
#: `# 0.001` beside it as the value it was raised FROM, so 15x is deliberate:
#: every other upstream config -- training, evaluation, repeatability -- uses
#: 0.001, and only the export uses 0.015.
#:
#: It was 0.005 here, which is `superpoint_pytorch.py`'s INFERENCE default. That
#: is a different step, and reading a value off the wrong step of the same
#: project is the easiest kind of mistake to make: the number exists, it is
#: upstream's, and it is 3x too permissive for this one. THRESHOLD_LADDER below
#: is what makes the choice visible rather than assumed.
DEFAULT_SCORE_THRESHOLD = 0.015

#: The DEFAULT rungs of the report: the three values that exist in upstream and
#: in this repo. `--threshold-ladder` replaces them, which is how a value above
#: the cut gets looked at without re-cutting the store -- the ladder is a
#: REPORT and `--score-threshold` is the store, and only the second is hashed. The threshold is where the aggregate sits between "any view
#: found it" and "every view found it", so it is the single number that decides
#: what a label IS -- and it is a hashed field, so a store cut at one cannot be
#: re-cut at another. Reporting all three costs one extra NMS per tile, against
#: `--num` teacher forwards, and turns "we picked 0.015" into "0.015 keeps this
#: fraction of what 0.001 would have".
THRESHOLD_LADDER = (0.001, 0.005, 0.015)

#: <!-- PENDING-MEASUREMENT: the real point density. This is a MEMORY cap, not a
#: threshold (spec.md 6.3), so a generous value costs bytes and a tight one
#: costs information -- 30000/Mpx is 1966 points in a 256 tile, well above what
#: `detection_threshold=0.005` is expected to pass. The first full rung's `n_kp`
#: histogram (cli/inspect_ha_labels.py) replaces it: take the knee, or leave the
#: cap generous and record that the threshold is what selected. -->
DEFAULT_POINTS_PER_MPX = 30000.0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tiles-root', default=DEFAULT_TILE_ROOT)
    ap.add_argument('--labels-root', default=DEFAULT_LABEL_ROOT)
    ap.add_argument('--tile', type=int, default=256,
                    help='which extraction to read. v1 is 256')
    ap.add_argument('--wsi-stem', nargs='*', default=None,
                    help='slides to do. Default: every store under --tiles-root '
                         'with this tile size')
    ap.add_argument('--ds', type=float, nargs='*', default=None,
                    help='rungs to do. Default: every one present')
    ap.add_argument('--num', type=int, default=100,
                    help="homographies per tile, upstream's N. The cost is "
                         'strictly linear in this')
    ap.add_argument('--limit', type=int, default=0,
                    help='stop after this many tiles per rung. The wall-clock '
                         'measurement of spec.md 12 step 4; 0 means all')
    ap.add_argument('--score-threshold', type=float,
                    default=DEFAULT_SCORE_THRESHOLD,
                    help='THE number that decides the label -- it is where '
                         'the aggregate sits between "any view found it" and '
                         '"every view found it". Default 0.015 is upstream\'s '
                         'HA EXPORT value (magic-point_coco_export.yaml:9); '
                         '0.001 is what every other upstream config uses and '
                         '0.005 is superpoint_pytorch\'s inference default. '
                         'All three are reported per rung. Identity: a store '
                         'cut at one value cannot be re-cut at another')
    ap.add_argument('--points-per-megapixel', type=float,
                    default=DEFAULT_POINTS_PER_MPX,
                    help='the per-tile cap, as a DENSITY so that three tile '
                         'sizes get comparable label densities')
    ap.add_argument('--sampler-id', default=None,
                    help='which corpus, when result/cache/tiles/ holds more '
                         'than one. Not a filter for convenience: without it a '
                         'mixed store is REFUSED, because processing both is '
                         'the expensive step run twice')
    ap.add_argument('--threshold-ladder', type=float, nargs='+',
                    default=list(THRESHOLD_LADDER),
                    help='which thresholds to REPORT beside the one the store '
                         'is cut at. Costs one extra NMS per tile whatever the '
                         'length, because every value is a comparison on the '
                         'same score array. The cut is added automatically if '
                         'it is not listed')
    ap.add_argument('--nms-radius', type=int, default=4)
    ap.add_argument('--border', type=int, default=4)
    ap.add_argument('--batch', type=int, default=16,
                    help='warped views per teacher forward. Throughput only')
    ap.add_argument('--seed', type=int, default=0,
                    help='the homography RNG. Not identity of the store: two '
                         'seeds give two samples of the same distribution, and '
                         'N=100 is what makes them agree')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('MakeHaLabels')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher = TeacherConfig().build(device)
    ha = HaConfig(num=args.num, batch=args.batch).build(teacher)
    print(teacher.summary(), flush=True)
    print(f'ha {ha.identity_id()}   N={args.num}   device {device}', flush=True)

    stores = _stores(args)
    if not stores:
        print(f'no pre-tile stores under {args.tiles_root} for tile '
              f'{args.tile}. Run cli/extract_pretiles.py first.')
        return 1
    print(f'{len(stores)} (slide, rung) stores', flush=True)

    rows, failures = [], []
    for index, folder in enumerate(stores, 1):
        meta = PreTileStore.load_meta(folder)
        print(f'\n[{index}/{len(stores)}] {meta.wsi_stem}  ds {meta.ds:g}',
              flush=True)
        try:
            rows.append(_label_one(folder, meta, ha, args))
        except Exception as e:                                   # noqa: BLE001
            print(f'    FAILED  {type(e).__name__}: {e}', flush=True)
            failures.append((folder.name, f'{type(e).__name__}: {e}'))

    summary = os.path.join(out_dir, 'make_ha_labels.csv')
    if rows:
        with open(summary, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved {summary}   ({len(rows)} rungs)')
        _extrapolate(rows, args)

    if failures:
        print(f'\n{len(failures)} rung(s) failed:')
        for what, why in failures:
            print(f'  {what}: {why}')
    return 1 if failures else 0


def _stores(args):
    """Every finished pre-tile store matching the filters, in a stable order.

    REFUSES A MIXED sampler_id RATHER THAN PROCESSING BOTH. `find` matches on
    tile size, and `sampler_id` is part of the store's cfg hash, so a corpus
    cut at 0.75 and one cut at 0.5 sit side by side as two complete sets of
    directories -- which is the store design working: neither overwrote the
    other. What does NOT work is this function quietly returning both, because
    the caller then spends the hours-to-days step twice and writes labels for a
    corpus that was rejected. Nothing would raise; the run would just take twice
    as long and the extra labels would look exactly like the wanted ones.

    So: name one with `--sampler-id`, or delete the set you do not want.
    """
    hits = PreTileStore.find(args.tiles_root, tile=int(args.tile))
    ratios = {}
    for folder in hits:
        meta = PreTileStore.load_meta(folder)
        ratios.setdefault(str(meta.sampler_id), []).append(folder)
    if args.sampler_id is None and len(ratios) > 1:
        listing = '   '.join(
            f'{r:g}: {len(v)} stores' for r, v in sorted(ratios.items()))
        raise SystemExit(
            f'{args.tiles_root} holds pre-tiles cut at {len(ratios)} different '
            f'sampler_ids and none was named.\n  {listing}\n'
            f'Pass --sampler-id, or delete the set you are not training on. '
            f'Running both is not a slower version of the right answer -- it '
            f'is HA over a corpus that was rejected, at full price.')

    kept = []
    for folder in hits:
        meta = PreTileStore.load_meta(folder)
        if (args.sampler_id is not None
                and str(meta.sampler_id) != str(args.sampler_id)):
            continue
        if args.wsi_stem and meta.wsi_stem not in args.wsi_stem:
            continue
        if args.ds and not any(abs(meta.ds - d) < 1e-6 for d in args.ds):
            continue
        kept.append(folder)
    return sorted(kept, key=lambda p: PreTileStore.load_meta(p).ds)


def _ladder_of(args):
    """The thresholds to report, and a zeroed accumulator for them.

    The cut is folded in whether or not it was listed, because a report that
    does not contain the value the store was cut at cannot say what that value
    did. Sorted, so `ladder[0]` is the most permissive and is the one reference
    extraction everything else is counted from.
    """
    ladder = sorted(set(float(t) for t in args.threshold_ladder)
                    | {float(args.score_threshold)})
    return ladder, np.zeros(len(ladder), np.int64)


def _label_one(folder, meta, ha, args):
    """One (slide, rung): every pre-tile through HA, one store file out."""
    existing = KeypointLabelStore.find(
        args.labels_root, wsi_stem=meta.wsi_stem, ds=meta.ds,
        ha_id=ha.identity_id(), pretile_id=meta.cfg_hash())
    if existing and not args.overwrite:
        print(f'    have it: {existing[0].name}   (--overwrite to redo)',
              flush=True)
        _, have = KeypointLabelStore.load(existing[0])
        # The ladder with no counts: same columns, blank values. Same columns
        # because `csv.DictWriter` takes its fieldnames from the FIRST row and
        # raises on any later row that carries a different set -- a reused rung
        # landing first would otherwise decide the schema for the whole file.
        return _row(meta, have, seconds=0.0, reused=True,
                    ladder=_ladder_of(args)[0])

    records = PreTileStore.load_index(folder)
    if args.limit:
        records = records[:int(args.limit)]
    cap = cap_for(meta.tile, args.points_per_megapixel)
    # One generator for the whole rung, so the draws of tile k do not depend on
    # how many tiles came before it in a previous run -- reproducible per rung,
    # which is the unit that gets re-run.
    rng = np.random.default_rng(args.seed)

    # `>` and not `>=`, because that is what points_from_prob uses
    # (`superpoint_pytorch.py:126-137`, and the -1 border sentinel depends on
    # it). Counting with the other comparison would put the ladder a hair off
    # the store it is describing.
    ladder, ladder_kept = _ladder_of(args)

    positions, points, scores, counts = [], [], [], []
    started = time.time()
    for i, record in enumerate(records):
        pre = PreTileStore.read_tile(folder, record)
        result = ha.run(pre, meta.tile, rng=rng, factor=meta.pre_tile_factor)
        xy, score, seen = points_from_prob(
            result.prob, result.counts,
            score_threshold=args.score_threshold,
            nms_radius=args.nms_radius, border=args.border, max_points=cap)

        # One extra extraction at the most permissive value on the ladder, with
        # no cap: its scores are all the NMS-and-border survivors above that
        # value, so every higher threshold is a comparison on the same array
        # rather than another pass. `max_points=None` matters -- with the cap
        # on, the denominator would be truncated by the cap and the ratio would
        # describe the budget instead of the threshold.
        _, ref_score, _ = points_from_prob(
            result.prob, result.counts, score_threshold=ladder[0],
            nms_radius=args.nms_radius, border=args.border, max_points=None)
        ladder_kept += np.array([(ref_score > t).sum() for t in ladder],
                                np.int64)

        positions.append((record.x, record.y))
        points.append(xy)
        scores.append(score)
        counts.append(seen)

        if (i + 1) % 50 == 0 or i + 1 == len(records):
            per = (time.time() - started) / (i + 1)
            print(f'    {i + 1}/{len(records)}   {per:.2f} s/tile   '
                  f'{np.mean([len(p) for p in points]):.0f} pts/tile', flush=True)
    seconds = time.time() - started

    batch = batch_from_lists(positions, points, scores, counts, cap)
    label_meta = LabelMeta.of(
        batch, wsi_stem=meta.wsi_stem, ds=meta.ds, tile=meta.tile, ha=ha,
        pretile_meta=meta, score_threshold=args.score_threshold,
        points_per_megapixel=args.points_per_megapixel,
        nms_radius=args.nms_radius, border=args.border,
        aggregation=ha.cfg.aggregation, wsi_path=meta.wsi_path)
    path = KeypointLabelStore.save(args.labels_root, batch, label_meta)

    print(f'    n_kp  mean {label_meta.mean_n_kp:.0f}  min {batch.n_kp.min()}  '
          f'max {batch.n_kp.max()}  cap {cap}  at-cap {batch.at_cap}', flush=True)
    _print_ladder(ladder, ladder_kept, float(args.score_threshold),
                  len(records))
    if batch.at_cap:
        print(f'    WARNING  {batch.at_cap} tiles hit the cap, so the CAP '
              f'selected rather than the threshold. Raise '
              f'--points-per-megapixel and redo this rung', flush=True)
    print(f'    wrote {path.name}   ({seconds / 60:.1f} min)', flush=True)
    return _row(meta, label_meta, seconds=seconds, reused=False,
                ladder=ladder, ladder_kept=ladder_kept)


def _print_ladder(ladder, kept, chosen: float, n_tiles: int) -> None:
    """What the threshold actually cost, as a fraction and not as a claim.

    The denominator is STATED rather than absolute: "every NMS peak" is not a
    denominator worth having, because a softmax gives every pixel some
    probability and a threshold of 0 would count noise. `ladder[0]` -- 0.001,
    upstream's training and evaluation value -- is the most permissive number
    anyone in this lineage actually uses, so it is the reference.
    """
    base = max(int(kept[0]), 1)
    per_tile = [k / max(n_tiles, 1) for k in kept]
    print(f'    threshold ladder, against {ladder[0]:g} as the reference:')
    for t, k, per in zip(ladder, kept, per_tile):
        mark = '  <- cut at this' if abs(t - chosen) < 1e-12 else ''
        print(f'      {t:<7g} {int(k):9d} pts   {per:7.1f}/tile   '
              f'{100.0 * k / base:5.1f}% kept   '
              f'{100.0 * (1 - k / base):5.1f}% filtered{mark}')


def _row(pre_meta, label_meta, *, seconds, reused, ladder=None,
         ladder_kept=None):
    tiles = max(label_meta.n_tiles, 1)
    # The ladder as three fixed columns rather than one packed string, so a
    # plot can read them. A reused rung has no ladder -- it was not recomputed
    # -- and writes blanks rather than zeros, which would read as "the
    # threshold removed everything".
    rungs = list(ladder) if ladder is not None else list(THRESHOLD_LADDER)
    ladder_cols = {f'n_gt_{t:g}': '' for t in rungs}
    ladder_cols['kept_frac_vs_ref'] = ''
    if ladder_kept is not None and len(ladder_kept):
        table = dict(zip(rungs, (int(k) for k in ladder_kept)))
        for t in rungs:
            ladder_cols[f'n_gt_{t:g}'] = table[t]
        chosen = table.get(float(label_meta.score_threshold))
        base = max(int(ladder_kept[0]), 1)
        if chosen is not None:
            ladder_cols['kept_frac_vs_ref'] = round(chosen / base, 4)
    return {'wsi_stem': label_meta.wsi_stem, 'ds': label_meta.ds,
            'tile': label_meta.tile, 'n_tiles': label_meta.n_tiles,
            'ha_id': label_meta.ha_id, 'pretile_id': label_meta.pretile_id,
            'score_threshold': label_meta.score_threshold,
            'cap': label_meta.cap, 'n_at_cap': label_meta.n_at_cap,
            'mean_n_kp': round(label_meta.mean_n_kp, 1),
            'seconds': round(seconds, 1),
            'sec_per_tile': round(seconds / tiles, 3),
            'reused': int(reused), **ladder_cols}


def _extrapolate(rows, args):
    """Turn the measured seconds per tile into the three numbers spec.md 13 asks
    for. Printed, not stored: it is arithmetic on the CSV, and burying it in a
    column would make it look like something that was measured."""
    timed = [r for r in rows if not r['reused'] and r['seconds'] > 0]
    if not timed:
        return
    per_tile = sum(r['sec_per_tile'] for r in timed) / len(timed)
    print(f'\n{per_tile:.2f} s/tile at tile {args.tile}, N={args.num}')
    print('  extrapolation (linear in tiles and in N, quadratic in tile size):')
    for tile, rungs, budget in ((256, 6, 500), (512, 5, 500), (1024, 4, 500)):
        area = (tile / args.tile) ** 2
        hours = per_tile * area * rungs * budget * 6 / 3600
        print(f'    model_{tile:<5}{rungs} rungs x {budget} tiles x 6 slides '
              f'-> {hours:7.1f} GPU-hours')
    print('  These decide whether R > 1 is affordable and whether 512/1024 are '
          'worth opening (spec.md 13).')


if __name__ == '__main__':
    sys.exit(main())
