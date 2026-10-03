#!/usr/bin/env python3
"""Multi-WSI x per-pyramid-level batch: one Render per (WSI, level).

For each input WSI, opens it once, iterates pyramid levels, builds a Render (camera)
with query_mpp = level's native mpp, and generates `--per-camera` shots into
a single unified out_dir + gt.csv.

The tissue mask is built ONCE per WSI, not once per level: the segmentation
and the recipe's region prep depend only on the slide. Each level's camera draws
its FoV positions from that one mask (`generator.FovSupply`: richness buckets
and overlap, through `TileSampler` placing the camera's spec), batch after batch
with the next seed until --per-camera shots exist.

A (WSI, level) with no position at all -- no region can hold the FoV -- is
skipped, and skips.csv carries what the sampler saw: per bucket, asked vs taken.
(The figure `diag_camera_skip.py` drew explained the retired tissue-ratio
sampler and went with it.) A level whose slide runs out of new positions
before --per-camera repeats some, each pass with a fresh domain gap, and says so.

Usage:
    python query_sim/cli/multi_batch.py <wsi1> [<wsi2> ...] \\
        [--per-camera 30] [--jitter 0.05] \\
        [--wh-ratio 4:3] [--MPixels 12] \\
        [--richness default|open] \\
        [--seg hest] [--mask-ds 1.0] [--seg-chunk-px 4000000] \\
        [--seed 0] [--out DIR]

Outputs (in result/<SLURM_JOB_NAME or MultiBatch>/):
    images/<wsi_tag>_L<lvl>_syn00000.png ...
    gt.csv        one row per shot (wsi, level, nominal_mpp, effective_mpp, gt_x, gt_y, ...)
    skips.csv     one row per skipped camera with reason (the sampler's
                  per-bucket report when no FoV fits)

Both csv files are appended as each camera finishes, not written at the end.
A PNG whose gt row was never flushed carries no position, mpp or rotation, so
a run that dies part-way would otherwise leave every image it produced
unusable.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from typing import List

import openslide
from PIL import Image

# ── query_sim/ + utilities/ onto sys.path ─────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                                # query_sim/
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
for _d in ('utilities', 'aiNNModel'):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SafeSlide          import SafeSlide                # noqa: E402
from TissueMask import TissueMask       # noqa: E402
from _memprobe          import mem_line                 # noqa: E402
from cli       import job_result_dir                    # noqa: E402
from config    import DomainGapConfig                   # noqa: E402
from record    import FOVRecord                         # noqa: E402
from camera    import Render                            # noqa: E402
from SlideReader import SlideReader                     # noqa: E402
from generator import (RICHNESS_PRESETS, FovSupply,           # noqa: E402
                       base_mask as build_base_mask, _record_from_shot,
                       sampler_cfg_for)

import Cache                                                         # noqa: E402
from TissueMaskConfig import MaskMaker, add_mask_args, mask_cfg_from_args  # noqa: E402


def _slide_tag(wsi_path: str, max_len: int = 20) -> str:
    return os.path.splitext(os.path.basename(wsi_path))[0][:max_len]


def _append_rows(path: str, rows: list, fieldnames: list) -> None:
    """Append dict rows to a CSV, writing the header only into an empty file.

    Written per camera rather than once at the end because the end is not
    guaranteed to arrive: an OOM kill during the fifth slide mask build used to
    throw away four slides of images, since every gt row was still in a list in
    memory. A PNG without its gt row carries no position, mpp or rotation -- it
    is unrecoverable, not merely unlabelled.

    Emptiness rather than a first-call flag, because one process is no longer
    the unit: under a job array each slide is its own process and each would
    think it was first. main() truncates its own two files once at startup, so
    a re-run still replaces rather than appends.
    """
    if not rows:
        return
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _run_camera(
    slide:          openslide.OpenSlide,
    wsi_path:       str,
    wsi_tag:        str,
    level:          int,
    per_camera:     int,
    cfg:            DomainGapConfig,
    richness:       str,
    base_mask:      TissueMask,
    seed:           int,
    img_dir:        str,
) -> tuple:
    """Build one Render at this (wsi, level). Return (records, skip_reason).

    `base_mask` is built once per WSI by the caller and SHARED across every
    level; the camera only reads it.
    """
    cam = Render(SlideReader(slide), cfg=cfg, seed=seed)
    print(f'\n[{wsi_tag} L{level}] mpp={cfg.query_mpp:.4f}  '
          f'rect_l0={cam.rect_w_l0}x{cam.rect_h_l0}  '
          f'bounding_l0={cam.bounding_square_side_l0}', flush=True)

    supply = FovSupply(cam, base_mask, sampler_cfg_for(per_camera, seed, richness))
    try:
        sampler = supply.sampler                    # the draw happens here
    except RuntimeError as exc:
        # One line per row: skips.csv is read by people and by csv readers, and
        # the sampler's report is the reason.
        reason = ' | '.join(line.strip() for line in str(exc).splitlines()
                            if line.strip())
        print(f'  SKIP: {reason}', flush=True)
        return [], reason
    print(next(iter(sampler.reports.values())).line(), flush=True)

    records: List[FOVRecord] = []
    n_repeats = 0
    for shot in supply:
        if len(records) >= per_camera:
            break
        n_repeats += int(shot.pass_index > 0)
        idx = len(records)
        fname = f'{wsi_tag}_L{level}_syn{idx:05d}.png'
        Image.fromarray(shot.image).save(os.path.join(img_dir, fname))
        records.append(_record_from_shot(
            shot, fname, wsi_path, cfg,
            cam.output_w, cam.output_h, level=level,
        ))
        if len(records) % max(1, per_camera // 5) == 0 or len(records) == per_camera:
            print(f'  [saved] {len(records)}/{per_camera}  {fname}', flush=True)

    if n_repeats:
        # Not a skip -- the rows belong in gt.csv -- but the slide ran out of
        # new positions and per_camera was met by repeating some, and that has
        # to be visible.
        print(f'  [WARN] the slide ran out of new positions: {n_repeats} of '
              f'{per_camera} shots repeat one with a fresh domain gap',
              flush=True)
    return records, None


def main():
    ap = argparse.ArgumentParser(description='Multi-WSI x per-level camera batch.')
    ap.add_argument('wsi_paths', nargs='+')
    ap.add_argument('--per-camera',       type=int,   default=30)
    ap.add_argument('--jitter',           type=float, default=0.05,
                    help='cfg.query_mpp_jitter fraction (0.05 = +/-5%%). 0 disables.')
    ap.add_argument('--wh-ratio',         default='4:3')
    ap.add_argument('--MPixels',          type=float, default=12.0)
    ap.add_argument('--richness', choices=sorted(RICHNESS_PRESETS), default='default',
                    help="which FoVs: 'default' mostly tissue-dense with a "
                         "share of edges, 'open' any FoV up to 85%% background")
    # --seg and its overrides. The recipe's read is chunked by default, which
    # is what keeps --mask-ds 1.0 from costing 16 bytes per level-0 pixel.
    add_mask_args(ap)
    ap.add_argument('--mask-cache-job', default=None,
                    help='whose mask cache: result/cache/<this>_mask/. '
                         'Default: this job')
    ap.add_argument('--device', default=None,
                    help='segmenter device; default cuda when available')
    ap.add_argument('--seed',             type=int,   default=0)
    ap.add_argument('--out',              default=None)
    ap.add_argument('--append',           action='store_true',
                    help='add to gt.csv and skips.csv instead of replacing '
                         'them. For a caller that runs this once per slide in '
                         'a loop, where every invocation after the first would '
                         'otherwise wipe the ones before it.')
    args = ap.parse_args()

    out_dir  = args.out or job_result_dir('MultiBatch')
    img_dir  = os.path.join(out_dir, 'images')
    os.makedirs(img_dir,  exist_ok=True)
    # Under a job array every task shares out_dir, so the csv files get the task
    # id and are merged afterwards. Appending to one shared file would interleave
    # rows: a few hundred at a time is far past the size an O_APPEND write is
    # atomic at. images/ needs no such care, the names already carry wsi and level.
    _task = os.environ.get('SLURM_ARRAY_TASK_ID')
    _sfx  = f'_{_task}' if _task else ''
    gt_path    = os.path.join(out_dir, f'gt{_sfx}.csv')
    skips_path = os.path.join(out_dir, f'skips{_sfx}.csv')
    # Truncate our own two files here, once. _append_rows writes a header into
    # an empty file, so without this a re-run would append a second corpus onto
    # the first. --append is for the caller that drives one slide per process:
    # there the invocations are the loop, and only the loop knows where the
    # corpus starts.
    if not args.append:
        for _p in (gt_path, skips_path):
            if os.path.exists(_p):
                os.remove(_p)
    print(f'Output    -> {out_dir}', flush=True)
    print(f'per-camera={args.per_camera}  jitter={args.jitter}  '
          f'wh={args.wh_ratio}  MP={args.MPixels}', flush=True)

    # One MaskMaker for the whole job: a segmentation model is seconds of
    # startup and the weights are the same for every WSI. It is built on the
    # first cache miss, and a slide already masked is never segmented again.
    import torch                                                     # noqa: PLC0415
    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    masks = MaskMaker(mask_cfg_from_args(args), Cache.cache_root(
        args.mask_cache_job or Cache.job_name('MultiBatch'), 'mask'), device)
    print(f'mask seg  -> {masks.cfg.seg_id()} on {device}', flush=True)

    # Counts, not lists: the rows go to disk as they are produced, so keeping
    # them here as well would only be a second copy waiting to be lost.
    n_records = 0
    n_skips   = 0
    SKIP_FIELDS = ['wsi', 'level', 'mpp', 'reason']

    for wsi_path in args.wsi_paths:
        wsi_tag = _slide_tag(wsi_path)
        try:
            # SafeSlide, not OpenSlide. A MIRAX read that lands on a cell the
            # scanner never wrote raises OpenSlideError, and that error latches
            # on the handle -- every later call fails, metadata included. There
            # is no except around _run_camera, so one hole used to take the
            # whole job down. It matters most at a small --mask-ds: a read of a
            # large rect fails whole when any tile inside it is missing, so at
            # ds=1 a single hole covers a whole read. SafeSlide halves the rect on failure and
            # only the genuinely missing leaves come back blank.
            slide = SafeSlide(wsi_path)
        except Exception as e:
            reason = f'SafeSlide open failed: {type(e).__name__}: {e}'
            print(f'\n[{wsi_tag}] SKIP whole WSI: {reason}', flush=True)
            _append_rows(skips_path, [{'wsi': wsi_tag, 'level': -1,
                                       'mpp': -1, 'reason': reason}],
                         SKIP_FIELDS)
            n_skips += 1
            continue

        base_mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        print(f'\n===== {wsi_tag}  levels={slide.level_count}  base_mpp={base_mpp:.4f} =====',
              flush=True)

        # Once per WSI, not once per level. The segmentation and the recipe's
        # region prep do not depend on the level; only `patchable` does, and
        # _run_camera takes that view itself. This is also the one heavy step
        # that can fail on its own (a level read, and with a model a forward
        # pass), so it gets its own except and takes down just this WSI.
        try:
            print(f'  building base mask ({masks.cfg.seg_id()}) ...', flush=True)
            # Drop the previous slide's mask BEFORE allocating this one. The
            # assignment below evaluates its right-hand side in full before it
            # rebinds the name, so without this the old main_mask (1 byte/px,
            # 18.7 GB on the largest slide here) is still alive throughout the
            # new slide's read.
            base_mask = None
            _t0 = time.perf_counter()
            base_mask = build_base_mask(slide, masks)
            # Timed because this is the one per-WSI fixed cost, and at ds=1
            # with a model it is the number that decides whether ds=1 is
            # affordable at all. One line per WSI, comparable across slides.
            print(f'  base mask: {base_mask.main_mask.shape}  '
                  f'tissue_frac={base_mask.tissue_fraction()*100:.1f}%  '
                  f'regions={len(base_mask.tissue_regions)}  '
                  f'[{time.perf_counter() - _t0:.1f}s]', flush=True)
            print(f'  {mem_line("after base mask")}', flush=True)
        except Exception as e:
            reason = f'base mask build failed: {type(e).__name__}: {e}'
            print(f'[{wsi_tag}] SKIP whole WSI: {reason}', flush=True)
            _append_rows(skips_path, [{'wsi': wsi_tag, 'level': -1,
                                       'mpp': round(base_mpp, 4),
                                       'reason': reason}], SKIP_FIELDS)
            n_skips += 1
            slide.close()
            continue

        for lvl in range(slide.level_count):
            ds        = slide.level_downsamples[lvl]
            level_mpp = base_mpp * ds
            cfg = DomainGapConfig(
                wh_ratio        = args.wh_ratio,
                MPixels         = args.MPixels,
                query_mpp       = level_mpp,
                query_mpp_jitter= args.jitter,
            )
            recs, skip_reason = _run_camera(
                slide=slide, wsi_path=wsi_path, wsi_tag=wsi_tag, level=lvl,
                per_camera=args.per_camera, cfg=cfg,
                richness=args.richness,
                base_mask=base_mask, seed=args.seed,
                img_dir=img_dir,
            )
            if skip_reason is not None:
                _append_rows(skips_path, [{'wsi':    wsi_tag,
                                           'level':  lvl,
                                           'mpp':    round(level_mpp, 4),
                                           'reason': skip_reason}],
                             SKIP_FIELDS)
                n_skips += 1
            if recs:
                _append_rows(gt_path, [asdict(r) for r in recs],
                             list(asdict(recs[0]).keys()))
                n_records += len(recs)
            # Per level, so growth inside one slide is attributable to a level
            # rather than only visible as a total at slide_done.
            print(f'  {mem_line(f"L{lvl} done")}', flush=True)

        # One line per WSI saying whether any read failed and where. The mask
        # composites holes onto the background colour rather than gating on the
        # validity plane, which is a bet that blank glass never reads as tissue.
        # This is how the bet gets checked: SafeSlide.holes carries the level-0
        # box of every failed read, so a mask that claims tissue there is
        # visible after the fact instead of silently producing blank queries.
        print(f'  reads: {slide.hole_summary()}  (reopens={slide.reopens})',
              flush=True)
        slide.close()
        # Paired with the line after the base mask build: the difference
        # between the two is this slide's working set, and the difference
        # between successive slides at THIS point is whatever did not come
        # back. peak is the cgroup's exact high-water mark, so one run answers
        # what --mem should have been without waiting for sacct to sample it.
        print(f'  {mem_line("slide done")}', flush=True)

    # Both files were written as the run went; this only reports the totals.
    if n_records:
        print(f'\ngt.csv    -> {gt_path}  ({n_records} rows)', flush=True)
    if n_skips:
        print(f'skips.csv -> {skips_path}  ({n_skips} skipped cameras)', flush=True)

    masks.close()
    print(f'\n{mem_line("job end")}', flush=True)


if __name__ == '__main__':
    main()
