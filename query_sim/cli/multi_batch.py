#!/usr/bin/env python3
"""Multi-WSI x per-pyramid-level batch: one Camera per (WSI, level).

For each input WSI, opens it once, iterates pyramid levels, builds a Camera
with query_mpp = level's native mpp, and generates `--per-camera` shots into
a single unified out_dir + gt.csv.

The tissue mask is built ONCE per WSI, not once per level: from_wsi plus
filter_regions plus merge_overlapping depend only on the slide, and only
filter_patchable depends on the level. The one mask object is then shared
across the levels, so each level must leave it exactly as it found it.

Cameras whose mask ends up with zero usable regions after all filters
(filter_regions -> merge_overlapping -> filter_patchable) are skipped with a
clear reason line. Use query_sim/cli/diag_camera_skip.py to visualise WHY a
specific (wsi, level) got skipped.

Usage:
    python query_sim/cli/multi_batch.py <wsi1> [<wsi2> ...] \\
        [--per-camera 30] [--jitter 0.05] \\
        [--wh-ratio 4:3] [--MPixels 12] \\
        [--tissue-ratio 0.3] [--region-protrusion 0.5] [--mask-ds 1.0] \\
        [--hest] [--seg-chunk-px 4000000] \\
        [--seed 0] [--out DIR]

Outputs (in result/<SLURM_JOB_NAME or MultiBatch>/):
    images/<wsi_tag>_L<lvl>_syn00000.png ...
    gt.csv        one row per shot (wsi, level, nominal_mpp, effective_mpp, gt_x, gt_y, ...)
    skips.csv     one row per skipped Camera with reason

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
from typing import List, Optional

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
from TissuesRegionsMask import TissuesRegionsMask       # noqa: E402
from _memprobe          import mem_line                 # noqa: E402
from cli       import job_result_dir                    # noqa: E402
from config    import DomainGapConfig                   # noqa: E402
from record    import FOVRecord                         # noqa: E402
from camera    import Camera                            # noqa: E402
from generator import _build_base_mask, _record_from_shot   # noqa: E402
from cli.diag_camera_skip import diagnose_skip          # noqa: E402


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
    tissue_ratio:   float,
    region_prot:    float,
    base_mask:      TissuesRegionsMask,
    seed:           int,
    img_dir:        str,
    diag_dir:       str,
) -> tuple:
    """Build one Camera at this (wsi, level). Return (records, skip_reason).

    `base_mask` is built once per WSI by the caller and SHARED across every
    level, so this function must hand it back exactly as it found it: it
    pushes one snapshot (filter_patchable, the only level-dependent filter)
    and the finally-block pops exactly that one.
    """
    cam = Camera(slide, cfg=cfg, seed=seed,
                 tissue_ratio=tissue_ratio,
                 region_protrusion_ratio=region_prot)
    print(f'\n[{wsi_tag} L{level}] mpp={cfg.query_mpp:.4f}  '
          f'rect_l0={cam.rect_w_l0}x{cam.rect_h_l0}  '
          f'bounding_l0={cam.bounding_square_side_l0}  '
          f'required_region_side_l0={cam.required_region_side_l0}',
          flush=True)

    cam.mask = base_mask
    cam.mask.filter_patchable(tile_size=cam.required_region_side_l0, ds=1.0)
    n_regions = len(cam.mask.tissue_regions)
    print(f'  mask: tissue_frac={cam.mask.tissue_fraction()*100:.1f}%  '
          f'usable_regions={n_regions}', flush=True)

    if n_regions == 0:
        reason = (f'filter_patchable emptied the mask: no tissue region can host '
                  f'required_region_side_l0={cam.required_region_side_l0} '
                  f'(mask ds={base_mask.mask_ds_x:.1f}, tile too large for this level)')
        print(f'  SKIP: {reason}', flush=True)
        # diagnose_skip works on its own copy and leaves base_mask untouched,
        # so undo filter_patchable here -- this early return skips the
        # try/finally below that would otherwise have done it.
        cam.mask.regions_undo()
        # Bracketed by probes on purpose: _draw hands the whole main_mask to
        # imshow, and at ds=1 that is a 6.6 Gpx bool per panel which matplotlib
        # promotes to float to colour-map. It is the prime suspect for the
        # ~25 GB that did not come back during the first run's level loop.
        print(f'  {mem_line("before diag")}', flush=True)
        diagnose_skip(
            cam     = cam,
            level   = level,
            out_dir = diag_dir,
            wsi_tag = wsi_tag,
            verdict = 'SKIP: filter_patchable emptied the mask',
        )
        print(f'  {mem_line("after diag")}', flush=True)
        return [], reason

    records: List[FOVRecord] = []
    try:
        for shot in cam:
            if len(records) >= per_camera:
                break
            idx = len(records)
            fname = f'{wsi_tag}_L{level}_syn{idx:05d}.png'
            Image.fromarray(shot.image).save(os.path.join(img_dir, fname))
            records.append(_record_from_shot(
                shot, fname, wsi_path, cfg,
                cam.output_w, cam.output_h, level=level,
            ))
            if len(records) % max(1, per_camera // 5) == 0 or len(records) == per_camera:
                print(f'  [saved] {len(records)}/{per_camera}  {fname}', flush=True)
    finally:
        # Exactly one: the shared base_mask arrived with filter_regions and
        # merge_overlapping already applied and must leave with them intact.
        cam.mask.regions_undo()                          # filter_patchable

    if not records:
        reason = (f'Camera iterator yielded nothing (all sampled positions '
                  f'failed has_tissue check or bounding-square read went out-of-WSI)')
        print(f'  SKIP: {reason}', flush=True)
        # diagnose_skip copies cam.mask before rewinding it, so base_mask keeps
        # its two filters and the next level starts from the same state.
        print(f'  {mem_line("before diag")}', flush=True)
        diagnose_skip(
            cam     = cam,
            level   = level,
            out_dir = diag_dir,
            wsi_tag = wsi_tag,
            verdict = 'SKIP: Camera yielded nothing',
        )
        print(f'  {mem_line("after diag")}', flush=True)
        return [], reason
    if len(records) < per_camera:
        # The camera gave up part-way: it produced usable shots, so this is not
        # a skip and the rows belong in gt.csv, but per_camera silently became
        # something smaller and that has to be visible in the log.
        print(f'  [WARN] only {len(records)}/{per_camera} shots; the camera hit '
              f'its consecutive-failure fuse after producing some', flush=True)
    return records, None


def main():
    ap = argparse.ArgumentParser(description='Multi-WSI x per-level Camera batch.')
    ap.add_argument('wsi_paths', nargs='+')
    ap.add_argument('--per-camera',       type=int,   default=30)
    ap.add_argument('--jitter',           type=float, default=0.05,
                    help='cfg.query_mpp_jitter fraction (0.05 = +/-5%%). 0 disables.')
    ap.add_argument('--wh-ratio',         default='4:3')
    ap.add_argument('--MPixels',          type=float, default=12.0)
    ap.add_argument('--tissue-ratio',     type=float, default=0.3)
    ap.add_argument('--region-protrusion',type=float, default=0.5)
    ap.add_argument('--mask-ds',          type=float, default=1.0)
    ap.add_argument('--hest',             action='store_true',
                    help='segment tissue with the HEST DeepLabV3 model instead '
                         'of the default HSV threshold')
    ap.add_argument('--seg-chunk-px',       type=int,   default=4_000_000,
                    help='tile budget for the segmentation forward pass; only '
                         'used with --hest, where the whole level image would '
                         'otherwise go through the model in one go')
    ap.add_argument('--read-chunk-px',  type=int,   default=268_435_456,
                    help='tile budget for READING the level (host RAM, as '
                         'opposed to --seg-chunk-px which is VRAM). Above this '
                         'the level is read and segmented tile by tile and is '
                         'never in memory whole. 0 restores the single read, '
                         'which at --mask-ds 1.0 costs 16 bytes per level-0 '
                         'pixel and OOMs on any large slide.')
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
    diag_dir = os.path.join(out_dir, 'diag')
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)
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

    # Segmentation method, chosen once for the whole job: loading DeepLabV3 is
    # seconds of startup and the weights are the same for every WSI.
    mask_method = None
    if args.hest:
        import torch
        from TissueSegFunc import HestSegConfig
        _dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'mask seg  -> HEST DeepLabV3 on {_dev}  '
              f'(seg_chunk_px={args.seg_chunk_px})', flush=True)
        mask_method = HestSegConfig().build(_dev)
    else:
        print('mask seg  -> HSV threshold (default)', flush=True)

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
            # whole job down. It matters most at a small --mask-ds: from_wsi
            # reads the level in ONE read_region, and openslide fails the whole
            # rect when any tile inside it is missing, so at ds=1 a single hole
            # covers the entire slide. SafeSlide halves the rect on failure and
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

        # Once per WSI, not once per level. from_wsi + filter_regions +
        # merge_overlapping do not depend on the level; only filter_patchable
        # does, and _run_camera applies that itself on the mask handed to it.
        # This is also the one heavy step that can now fail on its own (a full
        # level read, and with --hest a model forward pass), so it gets its own
        # except and takes down just this WSI.
        try:
            print(f'  building base mask (ds={args.mask_ds}) ...', flush=True)
            # Drop the previous slide's mask BEFORE allocating this one. The
            # assignment below evaluates its right-hand side in full before it
            # rebinds the name, so without this the old main_mask (1 byte/px,
            # 18.7 GB on the largest slide here) is still alive throughout the
            # new slide's read.
            base_mask = None
            _t0 = time.perf_counter()
            base_mask = _build_base_mask(
                slide, args.mask_ds, mask_method, 0.01,
                seg_chunk_px=args.seg_chunk_px if mask_method is not None else None,
                read_chunk_px=args.read_chunk_px or None,
            )
            # Timed because this is now the one per-WSI fixed cost, and at
            # ds=1 with --hest it is the number that decides whether ds=1 is
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
                tissue_ratio=args.tissue_ratio, region_prot=args.region_protrusion,
                base_mask=base_mask, seed=args.seed,
                img_dir=img_dir, diag_dir=diag_dir,
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
        print(f'To visualise a skip: python query_sim/cli/diag_camera_skip.py '
              f'<wsi_path> --level <lvl>', flush=True)

    print(f'\n{mem_line("job end")}', flush=True)


if __name__ == '__main__':
    main()
