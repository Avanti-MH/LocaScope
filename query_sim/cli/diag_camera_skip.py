#!/usr/bin/env python3
"""Visualise WHY a specific (WSI, level) got skipped by multi_batch.

Callable API (preferred — Camera already has mask + tile_l0 + cfg):
    from cli.diag_camera_skip import diagnose_skip
    diagnose_skip(cam, level=5, out_dir='result/Diag', wsi_tag='S1151088')

CLI (opens the WSI, builds a fresh Camera + mask, then calls diagnose_skip):
    python query_sim/cli/diag_camera_skip.py <wsi_path> --level <L>
        [--wh-ratio 4:3] [--MPixels 12] [--mask-ds 32]
        [--region-protrusion 0.5] [--min-region-ratio 0.01]
        [--out result/DiagCameraSkip]

Output: <out>/<wsi_tag>_L<lvl>_diag.png
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Optional

import openslide
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── query_sim/ + utilities/ onto sys.path ─────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                                # query_sim/
_UTILITIES = os.path.abspath(os.path.join(_HERE, '..', '..', 'utilities'))
if _UTILITIES not in sys.path:
    sys.path.insert(0, _UTILITIES)
_TM = os.path.abspath(os.path.join(_HERE, '..', '..', 'utilities', 'test_modules'))
if _TM not in sys.path:
    sys.path.insert(0, _TM)
from _paths import job_result_dir                                   # noqa: E402

from TissuesRegionsMask import TissuesRegionsMask     # noqa: E402
from config             import DomainGapConfig         # noqa: E402
from camera             import Camera                  # noqa: E402


_DRAW_MAX_SIDE = 2000        # decimate the mask to about this before plotting


def _draw(mask, regions, title, ax, tile_l0):
    """Regions come from `mask` but are passed separately: the stages are
    snapshots taken at different points of the filter pipeline, while the
    coordinate conversion belongs to the mask they all came from.

    The mask is decimated first because imshow copies whatever it is handed
    (safe_masked_invalid(A, copy=True)). At mask_ds=1 the full array is
    61197x107568 bool = 6.6 GB, and four panels cost 26 GB that measurably did
    not come back -- to fill a 4.5 inch panel that is about 630 px across. The
    slice is a stride view, so nothing is copied until imshow sees the small
    version. Every coordinate below is in mask pixels, so they all divide by
    the same step; at the mask_ds=32 that most callers use, step is 1 and this
    is the old function exactly.
    """
    step = max(1, int(max(mask.main_mask.shape) / _DRAW_MAX_SIDE))
    ax.imshow(mask.main_mask[::step, ::step], cmap='gray', aspect='equal')
    ax.set_title(f'{title}\nn_regions={len(regions)}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in regions:
        rx, ry, rw, rh = mask.region_box(r)
        ax.add_patch(Rectangle((rx / step, ry / step), rw / step, rh / step,
                               linewidth=1.0, edgecolor='crimson', facecolor='none'))
    tile_mask_side = tile_l0 / mask.mask_ds_x / step
    ax.add_patch(Rectangle((5, 5), tile_mask_side, tile_mask_side,
                           linewidth=1.5, edgecolor='deepskyblue',
                           facecolor='none', linestyle='--'))
    ax.text(5, 5 + tile_mask_side + 8,
            f'req tile {tile_l0}px lv0\n= {tile_mask_side * step:.0f}px in mask',
            fontsize=7, color='deepskyblue')


def diagnose_skip(
    cam:              Camera,
    level:            int,
    out_dir:          str,
    wsi_tag:          Optional[str] = None,
    min_region_ratio: float         = 0.01,
    verdict:          Optional[str] = None,
) -> str:
    """Save a 4-panel figure showing cam.mask at each filter stage.

    `cam` must have `cam.mask` set, in any filter state. This function does
    NOT touch it: it works on a shallow copy with its own region list and undo
    stack, rewinds that copy to raw, and replays raw → filter_regions →
    merge_overlapping → filter_patchable(tile=cam.required_region_side_l0)
    to snapshot each stage. `cam.mask` is left exactly as it was found.

    The copy is not cosmetic. multi_batch builds one mask per WSI and shares it
    across every pyramid level, so rewinding it here would silently strip
    filter_regions and merge_overlapping from every later level. The copy is
    shallow on purpose -- main_mask is large and read-only for our purposes,
    only the two mutable lists need their own identity.

    `level` is used only for the filename + title (Camera doesn't carry a
    level attr — the level lives in cfg.query_mpp indirectly).
    `verdict` overrides the auto verdict text in the title.

    Returns the saved PNG path.
    """
    os.makedirs(out_dir, exist_ok=True)
    if wsi_tag is None:
        wsi_tag = 'wsi'

    mask = copy.copy(cam.mask)
    mask.tissue_regions   = list(cam.mask.tissue_regions)
    mask._regions_history = [list(h) for h in cam.mask._regions_history]

    tile_l0   = cam.required_region_side_l0
    level_mpp = cam.cfg.query_mpp
    base_mpp  = float(cam.wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, level_mpp))

    # Rewind the COPY to raw, no matter what filter chain the caller applied.
    while mask.regions_undo():
        pass

    # Snapshot chain
    stages = [('raw', list(mask.tissue_regions))]
    mask.filter_regions(min_ratio=min_region_ratio)
    stages.append((f'filter_regions(min_ratio={min_region_ratio})',
                   list(mask.tissue_regions)))
    mask.merge_overlapping()
    stages.append(('merge_overlapping', list(mask.tissue_regions)))
    mask.filter_patchable(tile_size=tile_l0, ds=1.0)
    stages.append((f'filter_patchable(tile={tile_l0})',
                   list(mask.tissue_regions)))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    try:
        for ax, (title, regions) in zip(axes, stages):
            _draw(mask, regions, title, ax, tile_l0)

        if verdict is None:
            n_final = len(stages[-1][1])
            verdict = 'PASS' if n_final > 0 else 'SKIP (0 regions after filter_patchable)'
        fig.suptitle(
            f'{wsi_tag}  L{level}  '
            f'(mpp={level_mpp:.3f}, req_tile={tile_l0}px = {tile_l0*base_mpp/1000:.1f}mm)  '
            f'-> {verdict}',
            fontsize=12,
        )
        fig.tight_layout()

        out_path = os.path.join(out_dir, f'{wsi_tag}_L{level}_diag.png')
        fig.savefig(out_path, dpi=140, bbox_inches='tight')

        print(f'  [diag] {wsi_tag} L{level}: '
              + ' -> '.join(f'{len(regs)}' for _, regs in stages)
              + f'   saved {out_path}', flush=True)
    finally:
        plt.close(fig)
        # No regions_undo here: everything above happened on our own copy, so
        # there is nothing of the caller's to restore.

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wsi_path')
    ap.add_argument('--level',             type=int, required=True)
    ap.add_argument('--wh-ratio',          default='4:3')
    ap.add_argument('--MPixels',           type=float, default=12.0)
    ap.add_argument('--mask-ds',           type=float, default=32.0)
    ap.add_argument('--region-protrusion', type=float, default=0.5)
    ap.add_argument('--min-region-ratio',  type=float, default=0.01)
    ap.add_argument('--out',               default='',
                    help='output directory. Empty means result/<SLURM_JOB_NAME or DiagCameraSkip>/, via _paths.job_result_dir -- results live outside the checkout')
    args = ap.parse_args()

    # job_result_dir honours SLURM_JOB_NAME, so a job's output lands under
    # result/<job>/ without the jobscript spelling the path twice, and it makes
    # the directory itself.
    args.out = args.out or job_result_dir('DiagCameraSkip')

    slide = openslide.OpenSlide(args.wsi_path)
    try:
        base_mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        level_mpp = base_mpp * slide.level_downsamples[args.level]
        cfg = DomainGapConfig(
            wh_ratio=args.wh_ratio, MPixels=args.MPixels, query_mpp=level_mpp,
        )
        cam = Camera(slide, cfg=cfg,
                     region_protrusion_ratio=args.region_protrusion)
        cam.mask = TissuesRegionsMask.from_wsi(slide, ds=args.mask_ds)

        wsi_tag = os.path.splitext(os.path.basename(args.wsi_path))[0]
        diagnose_skip(
            cam=cam, level=args.level, out_dir=args.out, wsi_tag=wsi_tag,
            min_region_ratio=args.min_region_ratio,
        )
    finally:
        slide.close()


if __name__ == '__main__':
    main()
