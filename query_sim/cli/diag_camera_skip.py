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

from TissuesRegionsMask import TissuesRegionsMask     # noqa: E402
from config             import DomainGapConfig         # noqa: E402
from camera             import Camera                  # noqa: E402


def _draw(mask_snap, regions, title, ax, tile_l0, mask_ds_x, mask_ds_y):
    ax.imshow(mask_snap, cmap='gray', aspect='equal')
    ax.set_title(f'{title}\nn_regions={len(regions)}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in regions:
        rx = r.x / mask_ds_x
        ry = r.y / mask_ds_y
        rw = r.w / mask_ds_x
        rh = r.h / mask_ds_y
        ax.add_patch(Rectangle((rx, ry), rw, rh,
                               linewidth=1.0, edgecolor='crimson', facecolor='none'))
    tile_mask_side = tile_l0 / mask_ds_x
    ax.add_patch(Rectangle((5, 5), tile_mask_side, tile_mask_side,
                           linewidth=1.5, edgecolor='deepskyblue',
                           facecolor='none', linestyle='--'))
    ax.text(5, 5 + tile_mask_side + 8,
            f'req tile {tile_l0}px lv0\n= {tile_mask_side:.0f}px in mask',
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

    `cam` must have `cam.mask` set (any filter state — this function rewinds
    to raw via regions_undo, snapshots the raw → filter_regions →
    merge_overlapping → filter_patchable(tile=cam.required_region_side_l0)
    chain, saves the plot, then undoes those 3 filters again so cam.mask
    ends where diagnose_skip found it (raw state).

    `level` is used only for the filename + title (Camera doesn't carry a
    level attr — the level lives in cfg.query_mpp indirectly).
    `verdict` overrides the auto verdict text in the title.

    Returns the saved PNG path.
    """
    os.makedirs(out_dir, exist_ok=True)
    if wsi_tag is None:
        wsi_tag = 'wsi'

    mask      = cam.mask
    tile_l0   = cam.required_region_side_l0
    level_mpp = cam.cfg.query_mpp
    base_mpp  = float(cam.wsi.properties.get(openslide.PROPERTY_NAME_MPP_X, level_mpp))

    # Rewind mask to raw state, no matter what filter chain the caller applied.
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

    try:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        for ax, (title, regions) in zip(axes, stages):
            _draw(mask.main_mask, regions, title, ax, tile_l0,
                  mask.mask_ds_x, mask.mask_ds_y)

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
        plt.close(fig)

        print(f'  [diag] {wsi_tag} L{level}: '
              + ' -> '.join(f'{len(regs)}' for _, regs in stages)
              + f'   saved {out_path}', flush=True)
    finally:
        # Leave mask in raw state for the caller (undo the 3 we just applied)
        mask.regions_undo()   # filter_patchable
        mask.regions_undo()   # merge_overlapping
        mask.regions_undo()   # filter_regions

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
    ap.add_argument('--out',               default='result/DiagCameraSkip')
    args = ap.parse_args()

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
