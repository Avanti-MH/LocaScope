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
# Through the package, the way batch, demo and multi_batch already do it. This
# file used to reach past them into utilities/test_modules for a second copy of
# the same function.
from cli import job_result_dir                                      # noqa: E402

from SafeSlide import SafeSlide                        # noqa: E402
from TissueMaskConfig import add_mask_args, mask_cfg_from_args  # noqa: E402
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

    `cam` must have `cam.mask` set, in any filter state. Each stage is a view
    derived from `cam.mask.raw()` -- raw → filtered → merged →
    patchable(cam.required_region_side_l0) -- so `cam.mask` itself is never
    touched, and the base mask multi_batch shares across levels cannot lose
    its region prep here.

    `level` is used only for the filename + title (Camera doesn't carry a
    level attr — the level lives in cfg.query_mpp indirectly).
    `verdict` overrides the auto verdict text in the title.

    Returns the saved PNG path.
    """
    os.makedirs(out_dir, exist_ok=True)
    if wsi_tag is None:
        wsi_tag = 'wsi'

    mask = cam.mask.raw()
    tile_l0   = cam.required_region_side_l0
    level_mpp = cam.cfg.query_mpp
    base_mpp  = float(cam.wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, level_mpp))

    filtered = mask.filtered(min_region_ratio)
    merged = filtered.merged()
    stages = [('raw', mask.tissue_regions),
              (f'filtered(min_ratio={min_region_ratio})', filtered.tissue_regions),
              ('merged', merged.tissue_regions),
              (f'patchable({tile_l0})', merged.patchable(tile_l0).tissue_regions)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    try:
        for ax, (title, regions) in zip(axes, stages):
            _draw(mask, regions, title, ax, tile_l0)

        if verdict is None:
            n_final = len(stages[-1][1])
            verdict = 'PASS' if n_final > 0 else 'SKIP (0 regions after patchable)'
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

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wsi_path')
    ap.add_argument('--level',             type=int, required=True)
    ap.add_argument('--wh-ratio',          default='4:3')
    ap.add_argument('--MPixels',           type=float, default=12.0)
    ap.add_argument('--region-protrusion', type=float, default=0.5)
    add_mask_args(ap)
    ap.add_argument('--device', default=None,
                    help='segmenter device; default cuda when available')
    ap.add_argument('--out',               default='',
                    help='output directory. Empty means result/<SLURM_JOB_NAME or DiagCameraSkip>/, via _paths.job_result_dir -- results live outside the checkout')
    args = ap.parse_args()

    # job_result_dir honours SLURM_JOB_NAME, so a job's output lands under
    # result/<job>/ without the jobscript spelling the path twice, and it makes
    # the directory itself.
    args.out = args.out or job_result_dir('DiagCameraSkip')

    mask_cfg = mask_cfg_from_args(args)
    import torch                                                     # noqa: PLC0415
    device = torch.device(args.device or
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    slide = SafeSlide(args.wsi_path)
    try:
        base_mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        level_mpp = base_mpp * slide.level_downsamples[args.level]
        cfg = DomainGapConfig(
            wh_ratio=args.wh_ratio, MPixels=args.MPixels, query_mpp=level_mpp,
        )
        cam = Camera(slide, cfg=cfg,
                     region_protrusion_ratio=args.region_protrusion)
        cam.mask = mask_cfg.build(slide, device)

        wsi_tag = os.path.splitext(os.path.basename(args.wsi_path))[0]
        diagnose_skip(
            cam=cam, level=args.level, out_dir=args.out, wsi_tag=wsi_tag,
            min_region_ratio=mask_cfg.min_region_ratio,
        )
    finally:
        slide.close()


if __name__ == '__main__':
    main()
