#!/usr/bin/env python3
"""Multi-WSI x per-pyramid-level batch: one Camera per (WSI, level).

For each input WSI, opens it once, iterates pyramid levels, builds a Camera
with query_mpp = level's native mpp, and generates `--per-camera` shots into
a single unified out_dir + gt.csv.

Cameras whose mask ends up with zero usable regions after all filters
(filter_regions -> merge_overlapping -> filter_patchable) are skipped with a
clear reason line. Use query_sim/cli/diag_camera_skip.py to visualise WHY a
specific (wsi, level) got skipped.

Usage:
    python query_sim/cli/multi_batch.py <wsi1> [<wsi2> ...] \\
        [--per-camera 30] [--jitter 0.05] \\
        [--wh-ratio 4:3] [--MPixels 12] \\
        [--tissue-ratio 0.5] [--region-protrusion 0.5] [--mask-ds 32] \\
        [--seed 0] [--out DIR]

Outputs (in result/<SLURM_JOB_NAME or MultiBatch>/):
    images/<wsi_tag>_L<lvl>_syn00000.png ...
    gt.csv        one row per shot (wsi, level, nominal_mpp, effective_mpp, gt_x, gt_y, ...)
    skips.csv     one row per skipped Camera with reason
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict
from typing import List, Optional

import openslide
from PIL import Image

# ── query_sim/ + utilities/ onto sys.path ─────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                                # query_sim/
_UTILITIES = os.path.abspath(os.path.join(_HERE, '..', '..', 'utilities'))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)

from cli       import job_result_dir                    # noqa: E402
from config    import DomainGapConfig                   # noqa: E402
from record    import FOVRecord                         # noqa: E402
from camera    import Camera                            # noqa: E402
from generator import _prep_mask_for_camera, _record_from_shot   # noqa: E402
from cli.diag_camera_skip import diagnose_skip          # noqa: E402


def _slide_tag(wsi_path: str, max_len: int = 20) -> str:
    return os.path.splitext(os.path.basename(wsi_path))[0][:max_len]


def _run_camera(
    slide:          openslide.OpenSlide,
    wsi_path:       str,
    wsi_tag:        str,
    level:          int,
    per_camera:     int,
    cfg:            DomainGapConfig,
    tissue_ratio:   float,
    region_prot:    float,
    mask_ds:        float,
    seed:           int,
    img_dir:        str,
    diag_dir:       str,
) -> tuple:
    """Build one Camera at this (wsi, level). Return (records, skip_reason)."""
    cam = Camera(slide, cfg=cfg, seed=seed,
                 tissue_ratio=tissue_ratio,
                 region_protrusion_ratio=region_prot)
    print(f'\n[{wsi_tag} L{level}] mpp={cfg.query_mpp:.4f}  '
          f'rect_l0={cam.rect_w_l0}x{cam.rect_h_l0}  '
          f'bounding_l0={cam.bounding_square_side_l0}  '
          f'required_region_side_l0={cam.required_region_side_l0}',
          flush=True)

    cam.mask = _prep_mask_for_camera(
        cam, mask_ds=mask_ds, mask_method=None, min_region_ratio=0.01,
    )
    n_regions = len(cam.mask.tissue_regions)
    print(f'  mask: tissue_frac={cam.mask.tissue_fraction()*100:.1f}%  '
          f'usable_regions={n_regions}', flush=True)

    if n_regions == 0:
        reason = (f'filter_patchable emptied the mask: no tissue region can host '
                  f'required_region_side_l0={cam.required_region_side_l0} '
                  f'(mask thumb ds={mask_ds}, tile too large for this level)')
        print(f'  SKIP: {reason}', flush=True)
        # diagnose_skip rewinds cam.mask to raw internally (the 3 snapshots
        # from _prep_mask_for_camera are still in history at this point).
        diagnose_skip(
            cam     = cam,
            level   = level,
            out_dir = diag_dir,
            wsi_tag = wsi_tag,
            verdict = 'SKIP: filter_patchable emptied the mask',
        )
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
        cam.mask.regions_undo()                          # filter_patchable
        cam.mask.regions_undo()                          # merge_overlapping
        cam.mask.regions_undo()                          # filter_regions

    if not records:
        reason = (f'Camera iterator yielded nothing (all sampled positions '
                  f'failed has_tissue check or bounding-square read went out-of-WSI)')
        print(f'  SKIP: {reason}', flush=True)
        # try/finally above already undid the 3 filters; cam.mask is at raw
        # (empty history). diagnose_skip's while-undo loop is a no-op; it
        # then re-applies the 3 filters, plots, and undoes them again.
        diagnose_skip(
            cam     = cam,
            level   = level,
            out_dir = diag_dir,
            wsi_tag = wsi_tag,
            verdict = 'SKIP: Camera yielded nothing',
        )
        return [], reason
    return records, None


def main():
    ap = argparse.ArgumentParser(description='Multi-WSI x per-level Camera batch.')
    ap.add_argument('wsi_paths', nargs='+')
    ap.add_argument('--per-camera',       type=int,   default=30)
    ap.add_argument('--jitter',           type=float, default=0.05,
                    help='cfg.query_mpp_jitter fraction (0.05 = +/-5%%). 0 disables.')
    ap.add_argument('--wh-ratio',         default='4:3')
    ap.add_argument('--MPixels',          type=float, default=12.0)
    ap.add_argument('--tissue-ratio',     type=float, default=0.5)
    ap.add_argument('--region-protrusion',type=float, default=0.5)
    ap.add_argument('--mask-ds',          type=float, default=32.0)
    ap.add_argument('--seed',             type=int,   default=0)
    ap.add_argument('--out',              default=None)
    args = ap.parse_args()

    out_dir  = args.out or job_result_dir('MultiBatch')
    img_dir  = os.path.join(out_dir, 'images')
    diag_dir = os.path.join(out_dir, 'diag')
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)
    gt_path    = os.path.join(out_dir, 'gt.csv')
    skips_path = os.path.join(out_dir, 'skips.csv')
    print(f'Output    -> {out_dir}', flush=True)
    print(f'per-camera={args.per_camera}  jitter={args.jitter}  '
          f'wh={args.wh_ratio}  MP={args.MPixels}', flush=True)

    all_records: List[FOVRecord] = []
    skip_rows: List[dict] = []

    for wsi_path in args.wsi_paths:
        wsi_tag = _slide_tag(wsi_path)
        try:
            slide = openslide.OpenSlide(wsi_path)
        except Exception as e:
            reason = f'openslide.OpenSlide failed: {type(e).__name__}: {e}'
            print(f'\n[{wsi_tag}] SKIP whole WSI: {reason}', flush=True)
            skip_rows.append({'wsi': wsi_tag, 'level': -1,
                              'mpp': -1, 'reason': reason})
            continue

        base_mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        print(f'\n===== {wsi_tag}  levels={slide.level_count}  base_mpp={base_mpp:.4f} =====',
              flush=True)

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
                mask_ds=args.mask_ds, seed=args.seed,
                img_dir=img_dir, diag_dir=diag_dir,
            )
            if skip_reason is not None:
                skip_rows.append({
                    'wsi':    wsi_tag,
                    'level':  lvl,
                    'mpp':    round(level_mpp, 4),
                    'reason': skip_reason,
                })
            all_records.extend(recs)

        slide.close()

    # gt.csv
    if all_records:
        with open(gt_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_records[0]).keys()))
            writer.writeheader()
            for r in all_records:
                writer.writerow(asdict(r))
        print(f'\ngt.csv    -> {gt_path}  ({len(all_records)} rows)', flush=True)

    # skips.csv
    if skip_rows:
        with open(skips_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['wsi', 'level', 'mpp', 'reason'])
            writer.writeheader()
            for r in skip_rows:
                writer.writerow(r)
        print(f'skips.csv -> {skips_path}  ({len(skip_rows)} skipped cameras)', flush=True)
        print(f'To visualise a skip: python query_sim/cli/diag_camera_skip.py '
              f'<wsi_path> --level <lvl>', flush=True)


if __name__ == '__main__':
    main()
