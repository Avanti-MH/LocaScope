#!/usr/bin/env python3
"""One SLURM job, every WSI diagnostic, in funnel order.

    python utilities/cli/diagnostics/wsi_health_check.py --dataset ki67_pure
    python utilities/cli/diagnostics/wsi_health_check.py --dataset bracs/test --stop-after holes
    python utilities/cli/diagnostics/wsi_health_check.py --wsi /path/one.mrxs /path/two.svs --skip-scale

Five diagnostics already exist in this directory, each answering a different
question about the same slides:

    tier        class               question                          cost
    --------    -----------------   --------------------------------   --------
    metadata    WsiInfoCheck        vendor/levels/size/mpp -- sane?     seconds
    holes       WsiHolesScan        exhaustive: which blocks fail?      minutes-hours
    mask        MaskValidityCheck   does the tissue mask match reality? minutes, GPU for a model --seg
    yield       TileYieldProbe      does extraction get enough tiles?   minutes, needs a mask STORE
    scale       WsiScaleCheck       base_mpp spread, native rungs       seconds (parallel to the above)

They were built and stay separate on purpose -- see this directory's own
history: SlideProbe.py holds only the low-level primitives (readable(),
bounds_rect(), get_mpp()) the tools above build on, and stays that way rather
than growing the five checks into itself. This script is the thin, later
layer: it does not reimplement any of them, it only resolves ONE WSI list via
WsiSelection and routes it through however many tiers the caller wants,
in the order that makes each one's answer trustworthy before the next runs.

WHY THIS ORDER
--------------
metadata is a sanity check on the file the other four all open -- a WSI
with no mpp or one level fails loudly here in seconds instead of obscurely
later. holes is exhaustive and can run for hours on a badly-scanned MIRAX,
so it comes right after the cheap check and before anything that would waste
GPU time on a slide holes would have flagged first. mask depends on nothing
holes produced (it builds its own probe/mask), but reading it AFTER holes
means a reader already knows whether OpenSlideError gaps are the likely
explanation for whatever mask sees. yield is the deepest and slowest tier,
and the one with a real prerequisite outside this script -- a mask STORE
built by utilities/cli/build_cache/build_mask_store.py -- so it runs last
and degrades to a clear skip message, never a crash, when that store is
missing or has nothing for a given slide. scale asks a question none of the
other four touch (pyramid geometry / base_mpp), so it does not depend on or
feed into the funnel -- it runs whenever --skip-scale is not given, tier
budget allowing.

--stop-after {metadata,holes,mask,yield} runs the funnel up through and
including that tier and no further -- named stages rather than a numeric
--tier because the four are not interchangeable steps of one process, they
are four different questions, and a number would have to be memorised against
this table to be read at all. Omit it to run the whole funnel. scale is
controlled separately (--skip-scale) because it never was IN the funnel.

Tuning any one tier beyond its essentials here (probe grid density, the
sampler's overlap policy, ...) is what that tool's own CLI is for -- run it
directly. This script exposes only what a first-pass health check needs:
--seg for the mask recipe both mask tiers use, and --block/--levels/--sweep
for holes, because BLOCK decides whether an exhaustive scan fits the SLURM
walltime at all -- see jobscripts/WsiHealthCheck.sh's own calibration step
for how to pick it.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _paths import job_result_dir, setup_import_paths               # noqa: E402
setup_import_paths()

from WsiSelection import resolve_wsi_paths                            # noqa: E402
from wsi_info import WsiInfoCheck                                     # noqa: E402
from scan_wsi_holes import WsiHolesScan                               # noqa: E402
from diag_mask_validity import MaskValidityCheck                      # noqa: E402
from probe_tile_yield import TileYieldProbe, DEFAULT_MASK_CACHE_JOB     # noqa: E402
from TissueMaskConfig import MASK_RECIPES                             # noqa: E402
from diag_wsi_scale import (                                          # noqa: E402
    WsiScaleCheck, print_mpp_summary, print_native_summary, write_csv as write_scale_csv)


TIERS = ('metadata', 'holes', 'mask', 'yield')   # funnel order; scale is separate


def _heading(title: str) -> None:
    print(f'\n{"=" * 74}\nTIER: {title}\n{"=" * 74}', flush=True)


def run_metadata(entries: list, out_dir: str) -> list:
    _heading('metadata  (WsiInfoCheck)')
    rows = WsiInfoCheck(full=False).run(entries)
    if rows:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'wsi_info.csv'), 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    return rows


def run_holes(entries: list, out_dir: str, levels, block: int, sweep: int,
             figure_slides: str) -> list:
    _heading('holes  (WsiHolesScan -- exhaustive, can be slow)')
    rows = WsiHolesScan(levels=levels, block=block, sweep=sweep,
                        figure_slides=figure_slides, out_dir=out_dir).run(entries)

    # slides_with_holes.csv: one row per SCANNED slide, not per broken block --
    # holes.csv (written by WsiHolesScan itself) answers "where is the damage",
    # this answers "which slides do I have to worry about", which a summary
    # built only from holes.csv could not: a slide with zero broken blocks
    # appears NOWHERE in it, so "clean" has to come from the entries actually
    # scanned, not from absence in a file.
    if rows:
        by_slide = {}
        for r in rows:
            key = (r['dataset'], r['wsi_name'])
            agg = by_slide.setdefault(key, dict(dataset=r['dataset'],
                                                wsi_name=r['wsi_name'],
                                                n_broken_blocks=0, levels_affected=[]))
            agg['n_broken_blocks'] += r['bad']
            if r['bad']:
                agg['levels_affected'].append(r['level'])
        path = os.path.join(out_dir, 'slides_with_holes.csv')
        with open(path, 'w', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['dataset', 'wsi_name', 'has_holes', 'n_broken_blocks',
                        'levels_affected'])
            for agg in by_slide.values():
                wr.writerow([agg['dataset'], agg['wsi_name'],
                            int(agg['n_broken_blocks'] > 0), agg['n_broken_blocks'],
                            ' '.join(str(lv) for lv in sorted(agg['levels_affected']))])
        print(f'Saved {path}')
    return rows


def run_mask(entries: list, out_dir: str, seg: str) -> list:
    _heading(f'mask  (MaskValidityCheck, seg={seg})')
    check = MaskValidityCheck(MASK_RECIPES[seg], out_dir=out_dir)
    try:
        return check.run(entries)
    finally:
        check.masks.close()


def run_yield(entries: list, out_dir: str, seg: str, mask_cache_job: str) -> list:
    _heading('yield  (TileYieldProbe -- needs a prebuilt mask cache)')
    probe = TileYieldProbe(seg=seg, mask_cache_job=mask_cache_job, out_dir=out_dir)
    if not os.path.isdir(probe.seg_dir):
        print(f'  [SKIP] no mask cache at {probe.seg_dir} -- run '
              f'utilities/cli/build_cache/build_mask_store.py first, or pass '
              f'--seg/--mask-cache-job. yield answers nothing without it.',
              flush=True)
        return []
    return probe.run(entries)


def run_scale(entries: list, out_dir: str) -> list:
    _heading('scale  (WsiScaleCheck -- parallel, not part of the funnel)')
    rows = WsiScaleCheck(quiet=True).run(entries)
    if rows:
        print_mpp_summary(rows)
        print_native_summary(rows)
        write_scale_csv(rows, os.path.join(out_dir, 'base_mpp.csv'))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', nargs='+', default=None)
    ap.add_argument('--wsi', nargs='+', default=None, help='explicit WSI path(s)')
    ap.add_argument('--val-only', action='store_true')
    ap.add_argument('--stop-after', choices=TIERS, default=TIERS[-1],
                    help='run the funnel through and including this tier. '
                         'Default: the whole funnel (yield)')
    ap.add_argument('--skip-scale', action='store_true',
                    help='scale runs by default alongside the funnel -- it is '
                         'cheap and independent. Skip it if only the funnel '
                         'tiers are wanted')
    ap.add_argument('--seg', choices=sorted(MASK_RECIPES), default='hest',
                    help='mask recipe (MASK_RECIPES): the mask tier segments '
                         'with it, the yield tier reads its cached masks -- '
                         'one recipe, so both tiers look at the same mask')
    ap.add_argument('--mask-cache-job', default=DEFAULT_MASK_CACHE_JOB,
                    help='the job that made the mask cache the yield tier '
                         'reads: result/cache/<this>/')
    ap.add_argument('--levels', type=int, nargs='+', default=[0, 1, 2, 3],
                    help='holes tier: which pyramid levels to scan')
    ap.add_argument('--block', type=int, default=4096,
                    help='holes tier: block size in level-0 px. Halving it '
                         'quadruples the reads -- see the walltime table in '
                         'jobscripts/WsiHealthCheck.sh before lowering this '
                         'on a large corpus')
    ap.add_argument('--sweep', type=int, default=4,
                    help='holes tier: report block, 2x block, 4x block, ... '
                         'this many doublings, pooled from the same scan')
    ap.add_argument('--figure-slides', choices=('holed', 'all', 'none'),
                    default='holed',
                    help="holes tier: which slides get a PNG row. 'holed' "
                         'avoids all-white panels and the matplotlib raster '
                         'limit on a large corpus')
    ap.add_argument('--out', default=None,
                    help='parent dir. Each tier writes to <out>/<tier>/. '
                         'Default: result/<SLURM_JOB_NAME or WsiHealthCheck>/')
    args = ap.parse_args()

    if not args.dataset and not args.wsi:
        ap.error('need --dataset and/or --wsi')

    entries = resolve_wsi_paths(dataset=args.dataset, wsi=args.wsi,
                                val_only=args.val_only)
    print(f'{len(entries)} WSI(s) selected', flush=True)
    if not entries:
        return 1

    out_dir = args.out or job_result_dir('WsiHealthCheck')
    os.makedirs(out_dir, exist_ok=True)

    ran = {}
    stop_at = TIERS.index(args.stop_after)

    ran['metadata'] = run_metadata(entries, os.path.join(out_dir, 'metadata'))
    if stop_at >= TIERS.index('holes'):
        ran['holes'] = run_holes(entries, os.path.join(out_dir, 'holes'),
                                 args.levels, args.block, args.sweep,
                                 args.figure_slides)
    if stop_at >= TIERS.index('mask'):
        ran['mask'] = run_mask(entries, os.path.join(out_dir, 'mask'), args.seg)
    if stop_at >= TIERS.index('yield'):
        ran['yield'] = run_yield(entries, os.path.join(out_dir, 'yield'), args.seg,
                                  args.mask_cache_job)

    if not args.skip_scale:
        ran['scale'] = run_scale(entries, os.path.join(out_dir, 'scale'))

    _heading('summary')
    for tier in list(TIERS) + ['scale']:
        if tier in ran:
            print(f'  {tier:10s} {len(ran[tier]):5d} row(s)   -> {out_dir}/{tier}/')
        elif tier == 'scale':
            print(f'  {tier:10s} skipped  (--skip-scale)')
        else:
            print(f'  {tier:10s} skipped  (--stop-after {args.stop_after})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
