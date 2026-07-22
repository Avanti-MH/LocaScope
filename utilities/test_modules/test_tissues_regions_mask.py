#!/usr/bin/env python3
"""
Tests for TissuesRegionsMask.

Synthetic tests (always run):
  1. constructor  — all fields assigned correctly
  2. tissue_fraction — correct ratio on known mask
  3. _search_tissue_regions — blobs found with correct level-0 bboxes
  4. has_tissue / has_tissue_l0 — correct acceptance / rejection
  5. _levelCoordinate_converter — level-N → mask coords
  6. _mppCoordinate_converter  — mpp-space → mask coords
  7. loc / levelloc / mpploc   — mask slices at various coord systems

Real WSI test (requires --wsi):
  8. from_wsi — loads real slide, tissue_fraction > 0, tissue_regions not empty

Output figure: mask + TissueRegion bboxes overlay.

Usage:
    python test_modules/test_tissues_regions_mask.py
    python test_modules/test_tissues_regions_mask.py --wsi /path/to/slide.mrxs
"""

import argparse
import os
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from _paths import job_result_dir, setup_import_paths
setup_import_paths()

from TissuesRegionsMask import TissueRegion, TissuesRegionsMask


# ── Synthetic mask builder ────────────────────────────────────────────────────

def make_mask_with_blobs(H: int, W: int,
                         blobs: list[tuple[int, int, int, int]]) -> np.ndarray:
    """
    Return a bool mask (H, W) with rectangular blobs set to True.
    blobs: list of (row, col, h, w) in mask pixels.
    """
    mask = np.zeros((H, W), dtype=bool)
    for r, c, h, w in blobs:
        mask[r:r+h, c:c+w] = True
    return mask


def make_trm(mask: np.ndarray, mask_ds_x: float, mask_ds_y: float,
             level_downsamples: Optional[list] = None,
             mask_mpp: float = 0.0) -> TissuesRegionsMask:
    tissue_regions = TissuesRegionsMask._search_tissue_regions(
        mask, mask_ds_x, mask_ds_y
    )
    return TissuesRegionsMask(
        main_mask=mask,
        mask_ds_x=mask_ds_x,
        mask_ds_y=mask_ds_y,
        mask_mpp=mask_mpp,
        tissue_regions=tissue_regions,
        wsi_width=int(mask.shape[1] * mask_ds_x),
        wsi_height=int(mask.shape[0] * mask_ds_y),
        wsi_mpp_x=0.0,
        wsi_mpp_y=0.0,
        wsi_level_downsamples=level_downsamples or [1.0],
    )


# ── 1. Constructor ────────────────────────────────────────────────────────────

def validate_constructor():
    mask = np.ones((10, 20), dtype=bool)
    trm = make_trm(mask, mask_ds_x=4.0, mask_ds_y=4.0,
                   level_downsamples=[1.0, 2.0, 4.0])

    assert trm.main_mask is mask
    assert trm.mask_ds_x == 4.0
    assert trm.mask_ds_y == 4.0
    assert trm.wsi_width  == 80
    assert trm.wsi_height == 40
    assert trm.wsi_level_downsamples == [1.0, 2.0, 4.0]
    print('[PASS] constructor: all fields assigned correctly')


# ── 2. tissue_fraction ────────────────────────────────────────────────────────

def validate_tissue_fraction():
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :] = True   # top half = tissue → 50%
    trm = make_trm(mask, 1.0, 1.0)
    frac = trm.tissue_fraction()
    assert abs(frac - 0.5) < 1e-6, f'expected 0.5, got {frac}'

    mask_all = np.ones((8, 8), dtype=bool)
    assert abs(make_trm(mask_all, 1.0, 1.0).tissue_fraction() - 1.0) < 1e-6

    mask_none = np.zeros((8, 8), dtype=bool)
    assert make_trm(mask_none, 1.0, 1.0).tissue_fraction() == 0.0

    print('[PASS] tissue_fraction: 0%, 50%, 100% cases correct')


# ── 3. _search_tissue_regions ─────────────────────────────────────────────────

def validate_search_tissue_regions():
    """
    Mask (50×100) with two blobs.
    ds_x=4, ds_y=4 → level-0 coords = mask coords × 4.

    Blob A: row=5,  col=10, h=10, w=20  → level-0: x=40,  y=20,  w=80,  h=40
    Blob B: row=30, col=60, h=8,  w=15  → level-0: x=240, y=120, w=60,  h=32
    """
    H, W = 50, 100
    ds = 4.0
    blob_a = (5,  10, 10, 20)
    blob_b = (30, 60,  8, 15)
    mask = make_mask_with_blobs(H, W, [blob_a, blob_b])
    regions = TissuesRegionsMask._search_tissue_regions(mask, ds, ds)

    assert len(regions) == 2, f'expected 2 regions, got {len(regions)}'

    # Sort by x for deterministic comparison
    regions.sort(key=lambda r: r.x)
    r_a, r_b = regions

    assert r_a.x == int(blob_a[1] * ds), f'A.x={r_a.x}'
    assert r_a.y == int(blob_a[0] * ds), f'A.y={r_a.y}'
    assert r_a.w == int(blob_a[3] * ds), f'A.w={r_a.w}'
    assert r_a.h == int(blob_a[2] * ds), f'A.h={r_a.h}'

    assert r_b.x == int(blob_b[1] * ds), f'B.x={r_b.x}'
    assert r_b.y == int(blob_b[0] * ds), f'B.y={r_b.y}'
    assert r_b.w == int(blob_b[3] * ds), f'B.w={r_b.w}'
    assert r_b.h == int(blob_b[2] * ds), f'B.h={r_b.h}'

    # index field is set
    assert all(isinstance(r.index, int) for r in regions)

    # Small blob below min_area_px should be filtered
    tiny = make_mask_with_blobs(50, 100, [(0, 0, 5, 5)])   # area=25 < 100
    regions_tiny = TissuesRegionsMask._search_tissue_regions(tiny, ds, ds,
                                                             min_area_px=100)
    assert len(regions_tiny) == 0, 'tiny blob should be filtered'

    print(f'[PASS] _search_tissue_regions: 2 blobs found, coords correct, tiny filtered')


# ── 4. has_tissue / has_tissue_l0 ────────────────────────────────────────────

def validate_has_tissue():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True   # tissue square in mask coords
    trm = make_trm(mask, mask_ds_x=2.0, mask_ds_y=2.0,
                   level_downsamples=[1.0])

    # Direct mask coords
    assert     trm.has_tissue(5, 5, 10, 10),  'centre should have tissue'
    assert not trm.has_tissue(0, 0,  4,  4),  'corner should have no tissue'
    assert not trm.has_tissue(0, 0, 20, 20, tissue_ratio=0.99), 'full mask < 99% tissue'

    print('[PASS] has_tissue: centre/corner/ratio cases correct')


def validate_has_tissue_l0():
    """
    Mask (20×20), ds=2. Tissue at mask [5:15, 5:15].
    In level-0: tissue at x=10..30, y=10..30 (ds=2 → level-0 = mask×2).
    """
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    trm = make_trm(mask, mask_ds_x=2.0, mask_ds_y=2.0,
                   level_downsamples=[1.0])

    # Level-0 coords: centre of tissue blob
    assert     trm.has_tissue_l0(10, 10, 20, 20), 'centre l0 should have tissue'
    # Level-0 coords: far corner (no tissue)
    assert not trm.has_tissue_l0(0, 0, 8, 8),     'corner l0 should have no tissue'

    print('[PASS] has_tissue_l0: level-0 coords correctly converted')


# ── 5. _levelCoordinate_converter ────────────────────────────────────────────

def validate_level_converter():
    """
    ds=[1.0, 4.0, 16.0], mask_ds_x=mask_ds_y=8.0
    Level-2 point (x=16, y=32) in level-2 pixels.
    level_downsamples[2] = 16.0
    → level-0 x = 16 * 16 = 256 → mask col = 256/8 = 32
    → level-0 y = 32 * 16 = 512 → mask row = 512/8 = 64
    """
    mask = np.zeros((100, 100), dtype=bool)
    trm = make_trm(mask, mask_ds_x=8.0, mask_ds_y=8.0,
                   level_downsamples=[1.0, 4.0, 16.0])

    col, row = trm._levelCoordinate_converter(16, 32, level=2)
    assert col == 32, f'col={col}'
    assert row == 64, f'row={row}'

    # Level-0 (ds=1.0): x/mask_ds_x, y/mask_ds_y
    col0, row0 = trm._levelCoordinate_converter(80, 160, level=0)
    assert col0 == 10, f'col0={col0}'
    assert row0 == 20, f'row0={row0}'

    print('[PASS] _levelCoordinate_converter: level-2 and level-0 correct')


# ── 6. _mppCoordinate_converter ──────────────────────────────────────────────

def validate_mpp_converter():
    """
    mask_mpp=8.0. Input at mpp=2.0 (4× finer than mask).
    x_at_mpp2=40 → x_at_mask = 40 * 2.0 / 8.0 = 10
    """
    mask = np.zeros((50, 50), dtype=bool)
    trm = TissuesRegionsMask(
        main_mask=mask, mask_ds_x=4.0, mask_ds_y=4.0, mask_mpp=8.0,
        tissue_regions=[], wsi_width=200, wsi_height=200,
        wsi_mpp_x=0.5, wsi_mpp_y=0.5, wsi_level_downsamples=[1.0],
    )
    col, row = trm._mppCoordinate_converter(40.0, 24.0, mpp=2.0)
    assert col == 10, f'col={col}'
    assert row ==  6, f'row={row}'

    # Tuple mpp
    col2, row2 = trm._mppCoordinate_converter(40.0, 24.0, mpp=(2.0, 4.0))
    assert col2 == 10, f'col2={col2}'
    assert row2 == 12, f'row2={row2}'

    print('[PASS] _mppCoordinate_converter: scalar and tuple mpp correct')


# ── 7. loc / levelloc / mpploc ────────────────────────────────────────────────

def validate_loc_methods():
    """
    Mask (20×20) with a 4×4 tissue block at mask[8:12, 8:12].
    ds_x=ds_y=2, mask_mpp=4, level_ds=[1.0, 2.0].
    """
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    trm = TissuesRegionsMask(
        main_mask=mask, mask_ds_x=2.0, mask_ds_y=2.0, mask_mpp=4.0,
        tissue_regions=[], wsi_width=40, wsi_height=40,
        wsi_mpp_x=0.5, wsi_mpp_y=0.5, wsi_level_downsamples=[1.0, 2.0],
    )

    # loc: direct mask coords
    patch = trm.loc(8, 8, 4, 4)
    assert patch.shape == (4, 4)
    assert patch.all(), 'loc should return all-tissue patch'

    empty = trm.loc(0, 0, 4, 4)
    assert not empty.any(), 'loc corner should be empty'

    # levelloc: level-1 (ds=2.0), mask_ds=2.0 → ratio=1.0
    # level-1 point (8,8) → mask col=8*2/2=8, row=8*2/2=8
    patch_lv = trm.levelloc(8, 8, 4, 4, level=1)
    assert patch_lv.shape == (4, 4)
    assert patch_lv.all(), 'levelloc should hit tissue block'

    # mpploc: mpp=2.0, mask_mpp=4.0 → ratio=0.5 → mask col=x*2/4
    # x=16 → mask col=8; same for y
    patch_mpp = trm.mpploc(16, 16, 8, 8, mpp=2.0)
    assert patch_mpp.shape == (4, 4)
    assert patch_mpp.all(), 'mpploc should hit tissue block'

    print('[PASS] loc / levelloc / mpploc: all return correct mask slices')


# ── 8. Real WSI test (Otsu) ──────────────────────────────────────────────────

def test_hest_seg(path: str) -> tuple:
    '''Run HEST DeepLabV3 tissue segmentation via from_wsi(method=...).'''
    import torch
    import openslide
    from HESTSegFunc import hest_seg_model, make_hest_method

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n[HEST] seg  device={device}')

    wsi   = openslide.OpenSlide(path)
    model = hest_seg_model(device)
    trm   = TissuesRegionsMask.from_wsi(
        wsi, ds=64.0, method=make_hest_method(model, device),
    )
    Ht, Wt = trm.main_mask.shape
    thumb  = np.array(wsi.get_thumbnail((Wt, Ht)).convert('RGB'))
    wsi.close()

    print(f'  tissue={trm.tissue_fraction()*100:.1f}%  '
          f'{len(trm.tissue_regions)} regions  mask={trm.main_mask.shape}')
    return trm, thumb


def test_real_wsi(path: str) -> tuple:
    import openslide
    wsi = openslide.OpenSlide(path)
    trm = TissuesRegionsMask.from_wsi(wsi)
    Ht, Wt = trm.main_mask.shape
    thumb = np.array(wsi.get_thumbnail((Wt, Ht)).convert('RGB'))
    wsi.close()

    assert trm.main_mask is not None
    assert trm.main_mask.dtype == bool
    assert trm.tissue_fraction() > 0.0, 'real WSI should have some tissue'
    assert len(trm.tissue_regions) > 0, 'real WSI should have at least one region'
    assert trm.mask_ds_x > 0 and trm.mask_ds_y > 0
    assert trm.wsi_width > 0 and trm.wsi_height > 0
    assert len(trm.wsi_level_downsamples) > 0

    for i, r in enumerate(trm.tissue_regions):
        assert r.w > 0 and r.h > 0, f'region {i} has zero size'
        assert 0 <= r.x < trm.wsi_width,  f'region {i} x={r.x} out of bounds'
        assert 0 <= r.y < trm.wsi_height, f'region {i} y={r.y} out of bounds'

    print(f'[PASS] from_wsi: tissue={trm.tissue_fraction()*100:.1f}%, '
          f'{len(trm.tissue_regions)} regions, mask={trm.main_mask.shape}')
    return trm, thumb


# ── 9. from_wsi level / ds sweep ─────────────────────────────────────────────

def test_from_wsi_params(path: str) -> list:
    '''
    Sweep from_wsi over level=-1,-2,-3 and ds=8,4,1.
    Returns list of (label, trm).
    '''
    import openslide
    wsi = openslide.OpenSlide(path)
    n_levels = len(wsi.level_dimensions)

    print(f'\n[sweep] WSI has {n_levels} levels:')
    for lv in range(n_levels):
        W, H = wsi.level_dimensions[lv]
        print(f'  level {lv}: {W}×{H}  ds={wsi.level_downsamples[lv]:.1f}')

    configs = []
    for lv in [-1, -2, -3]:
        real = n_levels + lv
        if real >= 0:
            configs.append((f'level={lv}', dict(level=lv)))
        else:
            print(f'  level={lv}: SKIP (only {n_levels} levels)')
    for ds_val in [4, 16, 32, 64, 128]:
        configs.append((f'ds={ds_val}', dict(ds=float(ds_val))))

    results = []
    for label, kwargs in configs:
        trm = TissuesRegionsMask.from_wsi(wsi, **kwargs)
        Ht, Wt = trm.main_mask.shape
        print(f'  {label:10s}: mask={Wt}×{Ht}  '
              f'ds_x={trm.mask_ds_x:.1f}  '
              f'tissue={trm.tissue_fraction()*100:.1f}%  '
              f'regions={len(trm.tissue_regions)}')
        assert trm.tissue_fraction() > 0,      f'{label}: no tissue'
        assert len(trm.tissue_regions) > 0,    f'{label}: no regions'
        results.append((label, trm))

    wsi.close()
    print(f'[PASS] from_wsi sweep: {len(results)}/{len(configs)} configs ok')
    return results


# ── Figure ────────────────────────────────────────────────────────────────────

def draw_mask_with_regions(ax, trm: TissuesRegionsMask, title: str,
                           show_index: bool = True, linewidth: float = 1.5):
    ax.imshow(trm.main_mask, cmap='gray', vmin=0, vmax=1)
    for r in trm.tissue_regions:
        # Convert level-0 bbox to mask coords
        mx = r.x / trm.mask_ds_x
        my = r.y / trm.mask_ds_y
        mw = r.w / trm.mask_ds_x
        mh = r.h / trm.mask_ds_y
        ax.add_patch(mpatches.Rectangle(
            (mx, my), mw, mh,
            fill=False, edgecolor='red', linewidth=linewidth,
        ))
        if show_index:
            ax.text(mx + 2, my + 8, str(r.index), color='yellow', fontsize=7)
    ax.set_title(f'{title}\n{len(trm.tissue_regions)} regions, '
                 f'tissue={trm.tissue_fraction()*100:.1f}%', fontsize=9)
    ax.axis('off')


def draw_thumb_with_regions(ax, trm: TissuesRegionsMask, thumb: np.ndarray, title: str,
                            show_index: bool = True, linewidth: float = 1.5):
    ax.imshow(thumb)
    for r in trm.tissue_regions:
        rx = r.x / trm.mask_ds_x
        ry = r.y / trm.mask_ds_y
        rw = r.w / trm.mask_ds_x
        rh = r.h / trm.mask_ds_y
        ax.add_patch(mpatches.Rectangle(
            (rx, ry), rw, rh,
            fill=False, edgecolor='red', linewidth=linewidth,
        ))
        if show_index:
            ax.text(rx + 2, ry + 8, str(r.index), color='yellow', fontsize=7)
    ax.set_title(f'{title}\n{len(trm.tissue_regions)} regions, '
                 f'tissue={trm.tissue_fraction()*100:.1f}%', fontsize=9)
    ax.axis('off')


def draw_synthetic_figure(ax):
    H, W = 50, 100
    ds = 4.0
    blobs = [(5, 10, 10, 20), (30, 60, 8, 15)]
    mask = make_mask_with_blobs(H, W, blobs)
    trm = make_trm(mask, ds, ds, level_downsamples=[1.0])
    draw_mask_with_regions(ax, trm, 'Synthetic (2 blobs, ds=4)')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi', default='/work/u26130998/datasets/Ki67/S1151088,G7E,111220.mrxs',
    # ap.add_argument('--wsi', default='/work/u26130998/datasets/Ki67/S1103037,G7E,110122.mrxs',
                    help='real WSI path for integration test')
    ap.add_argument('--hest', action=argparse.BooleanOptionalAction, default=False,
                    help='also run HEST DeepLabV3 seg and compare with Otsu')
    ap.add_argument('--sweep', action=argparse.BooleanOptionalAction, default=False,
                    help='sweep from_wsi over level=-1,-2,-3 and ds=8,4,1')
    ap.add_argument('--region-index', action=argparse.BooleanOptionalAction, default=True,
                    help='show region index labels on bounding boxes')
    ap.add_argument('--bbox-lw', type=float, default=1.5,
                    help='bounding box linewidth (default 1.5)')
    ap.add_argument('--out', default=None, help='output figure path')
    args = ap.parse_args()

    # Synthetic tests
    validate_constructor()
    validate_tissue_fraction()
    validate_search_tissue_regions()
    validate_has_tissue()
    validate_has_tissue_l0()
    validate_level_converter()
    validate_mpp_converter()
    validate_loc_methods()

    # Real WSI — Otsu
    real_trm = None
    real_thumb = None
    if args.wsi and os.path.exists(args.wsi):
        real_trm, real_thumb = test_real_wsi(args.wsi)
    elif args.wsi:
        print(f'[SKIP] wsi not found: {args.wsi}')

    # Real WSI — HEST
    hest_trm = None
    hest_thumb = None
    if args.hest and args.wsi and os.path.exists(args.wsi):
        hest_trm, hest_thumb = test_hest_seg(args.wsi)

    # from_wsi level/ds sweep
    sweep_results = []
    if args.sweep and args.wsi and os.path.exists(args.wsi):
        sweep_results = test_from_wsi_params(args.wsi)

    # Figure layout
    SWEEP_PER_ROW = 4
    n_top  = 1                                  # synthetic always shown
    if real_trm  is not None: n_top += 2
    if hest_trm  is not None: n_top += 2
    n_bot       = len(sweep_results)
    n_sweep_rows = (n_bot + SWEEP_PER_ROW - 1) // SWEEP_PER_ROW if n_bot > 0 else 0
    n_cols = max(n_top, min(n_bot, SWEEP_PER_ROW), 1)
    n_rows = 1 + n_sweep_rows

    fig, ax_all = plt.subplots(n_rows, n_cols,
                               figsize=(7 * n_cols, 5 * n_rows),
                               squeeze=False)

    wsi_name = os.path.basename(args.wsi)
    col = 0
    si = args.region_index
    lw = args.bbox_lw
    draw_synthetic_figure(ax_all[0, col]); col += 1
    if real_trm is not None:
        draw_mask_with_regions( ax_all[0, col], real_trm, f'Otsu mask ({wsi_name})', show_index=si, linewidth=lw);  col += 1
        draw_thumb_with_regions(ax_all[0, col], real_trm, real_thumb, 'Otsu thumb',  show_index=si, linewidth=lw);  col += 1
    if hest_trm is not None:
        draw_mask_with_regions( ax_all[0, col], hest_trm, f'HEST mask ({wsi_name})', show_index=si, linewidth=lw); col += 1
        draw_thumb_with_regions(ax_all[0, col], hest_trm, hest_thumb, 'HEST thumb',  show_index=si, linewidth=lw); col += 1
    for c in range(col, n_cols):
        ax_all[0, c].axis('off')

    if sweep_results:
        for ro in range(n_sweep_rows):          # hide all sweep cells first
            for c in range(n_cols):
                ax_all[1 + ro, c].axis('off')
        for i, (label, trm) in enumerate(sweep_results):
            ro = i // SWEEP_PER_ROW
            c  = i %  SWEEP_PER_ROW
            draw_mask_with_regions(ax_all[1 + ro, c], trm, label, show_index=si, linewidth=lw)

    fig.tight_layout()
    out = args.out or os.path.join(job_result_dir('TissueMaskTest'),
                                    'tissue_mask__regions.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out}')
    print('All checks passed.')


if __name__ == '__main__':
    main()
