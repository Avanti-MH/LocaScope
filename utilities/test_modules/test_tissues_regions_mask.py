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


def validate_merge_overlapping():
    """
    Synthetic test for merge_overlapping (partial overlap only + union-find).

    Setup 4 regions:
      A = (  0,   0, 100, 100)      partial overlap with B
      B = ( 50,  50, 100, 100)      partial overlap with A and C
      C = (120, 120, 100, 100)      partial overlap with B, disjoint from A
      D = (400, 400,  50,  50)      isolated -> stays alone
      E = ( 60,  60,  20,  20)      fully inside B  -> NOT merged, stays alone
    Expected after merge:
      - {A, B, C} -> one region with union bbox (0, 0, 220, 220)
      - D unchanged
      - E unchanged (containment excluded)
    """
    from TissuesRegionsMask import TissueRegion
    trm = TissuesRegionsMask(
        main_mask=np.ones((500, 500), dtype=bool),
        mask_ds_x=1.0, mask_ds_y=1.0, mask_mpp=1.0,
        tissue_regions=[
            TissueRegion(  0,   0, 100, 100, index=0),  # A
            TissueRegion( 50,  50, 100, 100, index=1),  # B
            TissueRegion(120, 120, 100, 100, index=2),  # C
            TissueRegion(400, 400,  50,  50, index=3),  # D isolated
            TissueRegion( 60,  60,  20,  20, index=4),  # E fully inside B
        ],
        wsi_width=500, wsi_height=500,
        wsi_mpp_x=1.0, wsi_mpp_y=1.0, wsi_level_downsamples=[1.0],
    )

    trm.merge_overlapping()

    labels = sorted((r.x, r.y, r.w, r.h) for r in trm.tissue_regions)
    expect = sorted([
        (  0,   0, 220, 220),   # A + B + C union
        (400, 400,  50,  50),   # D
        ( 60,  60,  20,  20),   # E untouched
    ])
    assert len(trm.tissue_regions) == 3, \
        f'expected 3 regions, got {len(trm.tissue_regions)}'
    assert labels == expect, f'expected {expect}, got {labels}'
    for i, r in enumerate(trm.tissue_regions):
        assert r.index == i, f'index should be reset 0..N-1, got r{i}.index={r.index}'

    # No-op cases
    empty_trm = TissuesRegionsMask(
        main_mask=np.zeros((10, 10), dtype=bool), mask_ds_x=1, mask_ds_y=1,
        mask_mpp=1, tissue_regions=[], wsi_width=10, wsi_height=10,
        wsi_mpp_x=1, wsi_mpp_y=1, wsi_level_downsamples=[1.0],
    )
    empty_trm.merge_overlapping()
    assert empty_trm.tissue_regions == []

    print('[PASS] merge_overlapping: chain A-B-C merged; nested / isolated kept')


# ── 8. Real WSI test (Otsu) ──────────────────────────────────────────────────

def test_hest_seg(path: str, method: callable, ds: float = 64.0,
                  max_pixels: int = None) -> tuple:
    '''Run HEST DeepLabV3 tissue segmentation via from_wsi(method=...).'''
    import openslide

    print(f'\n[HEST] seg  ds={ds}  max_pixels={max_pixels}')

    wsi = openslide.OpenSlide(path)
    trm = TissuesRegionsMask.from_wsi(
        wsi, ds=ds, method=method, max_pixels=max_pixels,
    )
    Ht, Wt = trm.main_mask.shape
    thumb  = np.array(wsi.get_thumbnail((Wt, Ht)).convert('RGB'))
    wsi.close()

    print(f'  tissue={trm.tissue_fraction()*100:.1f}%  '
          f'{len(trm.tissue_regions)} regions  mask={trm.main_mask.shape}')
    return trm, thumb


def test_real_wsi(path: str, ds: float = 32.0) -> tuple:
    import openslide
    wsi = openslide.OpenSlide(path)
    trm = TissuesRegionsMask.from_wsi(wsi, ds=ds)
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

def test_from_wsi_params(path: str,
                         ds_list: list = None,
                         level_list: list = None,
                         method: callable = None,
                         max_pixels: int = None) -> list:
    '''
    Sweep from_wsi over the given level list and ds list.
    method=None -> HSV (default); method=<callable> -> HEST or custom.
    max_pixels enables tiled inference when method is a heavy model.
    Returns list of (label, trm).
    '''
    import openslide
    ds_list    = list(ds_list)    if ds_list    is not None else [4, 16, 32, 64, 128]
    level_list = list(level_list) if level_list is not None else [-1, -2, -3]

    wsi = openslide.OpenSlide(path)
    n_levels = len(wsi.level_dimensions)

    method_tag = 'HEST' if method is not None else 'HSV'
    print(f'\n[sweep {method_tag}] WSI has {n_levels} levels  '
          f'max_pixels={max_pixels}:')
    for lv in range(n_levels):
        W, H = wsi.level_dimensions[lv]
        print(f'  level {lv}: {W}x{H}  ds={wsi.level_downsamples[lv]:.1f}')

    configs = []
    for lv in level_list:
        real = n_levels + lv if lv < 0 else lv
        if 0 <= real < n_levels:
            configs.append((f'level={lv}', dict(level=lv)))
        else:
            print(f'  level={lv}: SKIP (only {n_levels} levels)')
    for ds_val in ds_list:
        configs.append((f'ds={ds_val:g}', dict(ds=float(ds_val))))

    results = []
    for label, kwargs in configs:
        try:
            trm = TissuesRegionsMask.from_wsi(
                wsi, method=method, max_pixels=max_pixels, **kwargs,
            )
        except Exception as e:
            print(f'  {label:10s}: FAIL ({type(e).__name__}: {e})')
            continue
        Ht, Wt = trm.main_mask.shape
        print(f'  {label:10s}: mask={Wt}x{Ht}  '
              f'ds_x={trm.mask_ds_x:.1f}  '
              f'tissue={trm.tissue_fraction()*100:.1f}%  '
              f'regions={len(trm.tissue_regions)}')
        results.append((label, trm))

    wsi.close()
    print(f'[PASS] from_wsi sweep: {len(results)}/{len(configs)} configs ok')
    return results


# ── 10. Ops pipeline: filter_regions / filter_patchable / merge_overlapping ──

def test_operations_pipeline(path: str,
                             ds: float = 32.0,
                             min_ratio: float = 0.01,
                             tile_size: int = 256,
                             ds_for_patchable: float = 4.0,
                             method: callable = None,
                             max_pixels: int = None) -> list:
    '''
    Ops pipeline: baseline mask -> each of filter_regions / filter_patchable /
    merge_overlapping in isolation -> all three combined.

    method=None -> HSV; method=<callable> -> HEST or custom (with tiled
    inference if max_pixels is set).
    Returns list of (label, trm) for figure rendering.
    '''
    import openslide
    from copy import deepcopy

    method_tag = 'HEST' if method is not None else 'HSV'
    wsi = openslide.OpenSlide(path)
    base = TissuesRegionsMask.from_wsi(
        wsi, ds=ds, method=method, max_pixels=max_pixels,
    )
    wsi.close()
    n0 = len(base)
    print(f'\n[ops pipeline {method_tag}] baseline: {n0} regions')

    results = [(f'baseline ({n0})', base)]

    t = deepcopy(base); t.filter_regions(min_ratio=min_ratio)
    print(f'  [1] filter_regions({min_ratio}): {n0} -> {len(t)}')
    results.append((f'[1] filter_regions({min_ratio})  {n0}->{len(t)}', t))

    t = deepcopy(base); t.merge_overlapping()
    print(f'  [2] merge_overlapping: {n0} -> {len(t)}')
    results.append((f'[2] merge_overlapping  {n0}->{len(t)}', t))

    t = deepcopy(base); t.filter_patchable(tile_size=tile_size, ds=ds_for_patchable)
    print(f'  [3] filter_patchable({tile_size}, ds={ds_for_patchable}): {n0} -> {len(t)}')
    results.append((f'[3] filter_patchable({tile_size},ds={ds_for_patchable:g})  {n0}->{len(t)}', t))

    t = deepcopy(base)
    t.filter_regions(min_ratio=min_ratio)
    t.merge_overlapping()
    t.filter_patchable(tile_size=tile_size, ds=ds_for_patchable)
    print(f'  pipeline [1]->[2]->[3]: {n0} -> {len(t)}')
    results.append((f'pipeline [1]->[2]->[3]  {n0}->{len(t)}', t))

    print(f'[PASS] ops pipeline: {len(results)} states rendered')
    return results


# ── 11. Tiling (adaptive halving) effect ─────────────────────────────────────

def _plan_tile_grid(H: int, W: int, max_pixels: int) -> tuple[int, int]:
    '''Mirror TissuesRegionsMask._adaptive_apply's grid planning (for visualisation).'''
    n_h = n_w = 1
    while (H // n_h) * (W // n_w) > max_pixels:
        if H // n_h >= W // n_w:
            n_h *= 2
        else:
            n_w *= 2
    return n_h, n_w


def test_tiling_effect(path: str, ds: float = 32.0,
                       max_pixels_list: tuple = (16_000_000, 4_000_000, 1_000_000),
                       overlap: int = 128,
                       method: callable = None) -> tuple:
    '''
    Three tiling effect views (works on HSV by default, or pass method=HEST
    to prove tiled inference matches whole-image inference on DeepLabV3):
      (a) seam artifact: no-tile vs tiled at same ds (no-tile may OOM
          if method is heavy and ds too low -- catch and skip)
      (b) tile grid overlay: mask + tile boundary + overlap zone
      (c) max_pixels sweep: several budgets -> different grids

    Returns (seam_list, grid_pack, sweep_list).
    seam_list  = [(label, trm), ...]                              # 0-2 items
    grid_pack  = (trm, n_h, n_w, overlap, max_pixels)             # or None
    sweep_list = [(label, trm, n_h, n_w), ...]
    '''
    import openslide

    method_tag = 'HEST' if method is not None else 'HSV'
    wsi = openslide.OpenSlide(path)

    # (a) Seam: no-tile baseline vs tiled at the reference max_pixels
    ref_mp = max_pixels_list[1] if len(max_pixels_list) > 1 else max_pixels_list[0]
    print(f'\n[tiling {method_tag}] seam test  ds={ds}  '
          f'ref max_pixels={ref_mp/1e6:.1f}M  overlap={overlap}')
    seam_list = []
    try:
        trm_no = TissuesRegionsMask.from_wsi(wsi, ds=ds, method=method)
        seam_list.append((f'{method_tag} no-tile ds={ds:g}', trm_no))
        print(f'  no-tile  regions={len(trm_no)}  tissue={trm_no.tissue_fraction()*100:.1f}%')
    except Exception as e:
        trm_no = None
        print(f'  no-tile  SKIP ({type(e).__name__}: {e})')

    trm_til = TissuesRegionsMask.from_wsi(
        wsi, ds=ds, method=method, max_pixels=ref_mp, overlap=overlap,
    )
    H, W = trm_til.main_mask.shape
    if trm_no is not None:
        diff_frac = float((trm_no.main_mask != trm_til.main_mask).mean())
        print(f'  tiled    regions={len(trm_til)}  tissue={trm_til.tissue_fraction()*100:.1f}%')
        print(f'  pixel disagreement fraction = {diff_frac*100:.3f}%')
        seam_list.append((
            f'{method_tag} tiled max={ref_mp/1e6:.0f}M ov={overlap}  '
            f'diff={diff_frac*100:.2f}%',
            trm_til,
        ))
    else:
        seam_list.append((
            f'{method_tag} tiled max={ref_mp/1e6:.0f}M ov={overlap}',
            trm_til,
        ))
        print(f'  tiled    regions={len(trm_til)}  tissue={trm_til.tissue_fraction()*100:.1f}%')

    # (b) Grid overlay: mask at ref max_pixels + tile boundary lines
    n_h_ref, n_w_ref = _plan_tile_grid(H, W, ref_mp)
    print(f'  tile grid @ max={ref_mp/1e6:.0f}M: {n_h_ref}x{n_w_ref} = {n_h_ref*n_w_ref} tiles')
    grid_pack = (trm_til, n_h_ref, n_w_ref, overlap, ref_mp)

    # (c) max_pixels sweep
    print(f'[tiling {method_tag}] max_pixels sweep '
          f'{[f"{mp/1e6:.0f}M" for mp in max_pixels_list]}')
    sweep_list = []
    for mp in max_pixels_list:
        trm = TissuesRegionsMask.from_wsi(
            wsi, ds=ds, method=method, max_pixels=mp, overlap=overlap,
        )
        n_h, n_w = _plan_tile_grid(H, W, mp)
        sweep_list.append((
            f'{method_tag} max={mp/1e6:.0f}M grid={n_h}x{n_w}', trm, n_h, n_w,
        ))
        print(f'  max={mp/1e6:.0f}M  grid={n_h}x{n_w}  regions={len(trm)}')

    wsi.close()
    print(f'[PASS] tiling: {len(sweep_list)} sweep configs')
    return seam_list, grid_pack, sweep_list


def draw_tile_grid_overlay(ax, trm, n_h, n_w, overlap_px, max_pixels, title):
    '''Draw mask + tile core boundaries (solid) + overlap zone edges (dashed).'''
    H, W = trm.main_mask.shape
    ax.imshow(trm.main_mask, cmap='gray', vmin=0, vmax=1)
    # Convert overlap from mask units (already at the same scale as mask)
    tile_h = H // n_h
    tile_w = W // n_w
    ov = overlap_px
    for i in range(n_h + 1):
        y = i * tile_h if i < n_h else H
        ax.axhline(y, color='cyan', linewidth=1.0)
        if 0 < i < n_h:
            ax.axhline(min(H, y + ov), color='magenta', linewidth=0.5, linestyle='--')
            ax.axhline(max(0, y - ov), color='magenta', linewidth=0.5, linestyle='--')
    for j in range(n_w + 1):
        x = j * tile_w if j < n_w else W
        ax.axvline(x, color='cyan', linewidth=1.0)
        if 0 < j < n_w:
            ax.axvline(min(W, x + ov), color='magenta', linewidth=0.5, linestyle='--')
            ax.axvline(max(0, x - ov), color='magenta', linewidth=0.5, linestyle='--')
    ax.set_title(f'{title}\n{n_h}x{n_w}={n_h*n_w} tiles, '
                 f'~{tile_h}x{tile_w}, overlap={ov}px',
                 fontsize=9)
    ax.axis('off')


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

def _parse_pixel_count(s):
    '''Parse "16M" / "4M" / "500K" / "1500000" into int.'''
    s = str(s).strip()
    if not s:
        return None
    tail = s[-1].upper()
    if tail == 'M':
        return int(float(s[:-1]) * 1_000_000)
    if tail == 'K':
        return int(float(s[:-1]) * 1_000)
    return int(float(s))


def _parse_int_list(s):
    return [int(float(x)) for x in s.split(',') if x.strip()]


def _parse_float_list(s):
    return [float(x) for x in s.split(',') if x.strip()]


def _parse_pixel_list(s):
    return [_parse_pixel_count(x) for x in s.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi', default='/work/u26130998/datasets/Ki67/S1151088,G7E,111220.mrxs',
                    help='real WSI path for integration test')
    ap.add_argument('--out', default=None, help='output figure path')

    # -- Sub-test toggles --
    ap.add_argument('--hest', action=argparse.BooleanOptionalAction, default=False,
                    help='enable HEST DeepLabV3 seg (also feeds sweep/ops/tiling)')
    ap.add_argument('--sweep', action=argparse.BooleanOptionalAction, default=False,
                    help='sweep from_wsi over level list x ds list')
    ap.add_argument('--ops', action=argparse.BooleanOptionalAction, default=False,
                    help='ops pipeline: filter_regions / filter_patchable / merge_overlapping')
    ap.add_argument('--tiling', action=argparse.BooleanOptionalAction, default=False,
                    help='tiling effect: seam vs no-tile + grid overlay + max_pixels sweep')

    # -- Value knobs (option C: all exposed) --
    # Otsu baseline
    ap.add_argument('--otsu-ds', type=float, default=32.0,
                    help='Otsu baseline mask ds (default 32)')
    # Sweep matrix
    ap.add_argument('--sweep-ds',    default='4,16,32,64,128',
                    help='comma list of ds for --sweep (default 4,16,32,64,128)')
    ap.add_argument('--sweep-level', default='-1,-2,-3',
                    help='comma list of level for --sweep (default -1,-2,-3)')
    # Ops pipeline
    ap.add_argument('--ops-ds',         type=float, default=32.0)
    ap.add_argument('--ops-min-ratio',  type=float, default=0.05)
    ap.add_argument('--ops-patch-tile', type=int,   default=256)
    ap.add_argument('--ops-patch-ds',   type=float, default=4.0)
    # Tiling
    ap.add_argument('--tiling-ds',         type=float, default=32.0)
    ap.add_argument('--tiling-max-pixels', default='16M,4M,1M',
                    help='comma list of tile budgets, e.g. "16M,4M,1M"')
    ap.add_argument('--tiling-overlap',    type=int,   default=128)
    # HEST
    ap.add_argument('--hest-ds',         type=float, default=64.0)
    ap.add_argument('--hest-max-pixels', default='4M',
                    help='HEST tile budget for hest-only / ops / sweep (single value)')
    # Visualization
    ap.add_argument('--per-row', type=int, default=4)
    ap.add_argument('--dpi',     type=int, default=600)
    ap.add_argument('--figure-scale', default='7,5',
                    help='(col-scale,row-scale) for figsize, e.g. "7,5"')
    ap.add_argument('--region-index', action=argparse.BooleanOptionalAction, default=True,
                    help='show region index labels on bounding boxes')
    ap.add_argument('--bbox-lw', type=float, default=1.5,
                    help='bounding box linewidth (default 1.5)')

    args = ap.parse_args()

    # Parsed list/pixel values
    sweep_ds_list      = _parse_float_list(args.sweep_ds)
    sweep_level_list   = _parse_int_list(args.sweep_level)
    tiling_max_pixels  = _parse_pixel_list(args.tiling_max_pixels)
    hest_max_pixels    = _parse_pixel_count(args.hest_max_pixels)
    fig_col_s, fig_row_s = _parse_float_list(args.figure_scale)

    # Synthetic tests
    validate_constructor()
    validate_tissue_fraction()
    validate_search_tissue_regions()
    validate_has_tissue()
    validate_has_tissue_l0()
    validate_level_converter()
    validate_mpp_converter()
    validate_loc_methods()
    validate_merge_overlapping()

    # HEST model loaded once, shared across all sub-tests
    hest_method = None
    if args.hest:
        import torch
        from HESTSegFunc import hest_seg_model, make_hest_method
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'\n[HEST] loading model on {device}')
        hest_model = hest_seg_model(device)
        hest_method = make_hest_method(hest_model, device)

    # Real WSI -- Otsu
    real_trm = None
    real_thumb = None
    if args.wsi and os.path.exists(args.wsi):
        real_trm, real_thumb = test_real_wsi(args.wsi, ds=args.otsu_ds)
    elif args.wsi:
        print(f'[SKIP] wsi not found: {args.wsi}')

    # Real WSI -- HEST-only mask panel (single ds)
    hest_trm = None
    hest_thumb = None
    if hest_method is not None and args.wsi and os.path.exists(args.wsi):
        hest_trm, hest_thumb = test_hest_seg(
            args.wsi, method=hest_method,
            ds=args.hest_ds, max_pixels=hest_max_pixels,
        )

    # from_wsi level/ds sweep (HEST if enabled, else HSV)
    sweep_results = []
    if args.sweep and args.wsi and os.path.exists(args.wsi):
        sweep_results = test_from_wsi_params(
            args.wsi,
            ds_list=sweep_ds_list, level_list=sweep_level_list,
            method=hest_method,
            max_pixels=hest_max_pixels if hest_method is not None else None,
        )

    # ops pipeline (HEST if enabled, else HSV)
    ops_results = []
    if args.ops and args.wsi and os.path.exists(args.wsi):
        ops_results = test_operations_pipeline(
            args.wsi,
            ds=args.ops_ds,
            min_ratio=args.ops_min_ratio,
            tile_size=args.ops_patch_tile,
            ds_for_patchable=args.ops_patch_ds,
            method=hest_method,
            max_pixels=hest_max_pixels if hest_method is not None else None,
        )

    # tiling effect (HEST if enabled, else HSV)
    tiling_seam = tiling_grid = tiling_sweep = None
    if args.tiling and args.wsi and os.path.exists(args.wsi):
        tiling_seam, tiling_grid, tiling_sweep = test_tiling_effect(
            args.wsi,
            ds=args.tiling_ds,
            max_pixels_list=tiling_max_pixels,
            overlap=args.tiling_overlap,
            method=hest_method,
        )

    # Figure layout: one row per section
    PER_ROW = args.per_row
    si = args.region_index
    lw = args.bbox_lw

    # Row 0: synthetic + Otsu (+thumb) + HEST (+thumb)
    row0_cells = 1 + (2 if real_trm is not None else 0) + (2 if hest_trm is not None else 0)
    n_top_cols = max(row0_cells, 1)

    # Sweep rows
    n_sweep_rows = (len(sweep_results) + PER_ROW - 1) // PER_ROW if sweep_results else 0

    # Ops rows (may wrap when ops > PER_ROW)
    n_ops_rows = (len(ops_results) + PER_ROW - 1) // PER_ROW if ops_results else 0

    # Tiling rows: seam (2 panels) + grid (1) + sweep (len sweep)
    n_tiling_rows = 0
    if tiling_seam or tiling_grid or tiling_sweep:
        n_tiling_seam_cells  = len(tiling_seam or [])
        n_tiling_sweep_cells = len(tiling_sweep or [])
        n_tiling_cells       = n_tiling_seam_cells + 1 + n_tiling_sweep_cells
        n_tiling_rows        = (n_tiling_cells + PER_ROW - 1) // PER_ROW
    else:
        n_tiling_cells = 0

    n_cols = max(
        n_top_cols,
        PER_ROW if sweep_results   else 1,
        PER_ROW if ops_results     else 1,
        PER_ROW if n_tiling_cells  else 1,
    )
    n_rows = 1 + n_sweep_rows + n_ops_rows + n_tiling_rows

    fig, ax_all = plt.subplots(n_rows, n_cols,
                               figsize=(fig_col_s * n_cols, fig_row_s * n_rows),
                               squeeze=False)

    # blank everything first
    for r in range(n_rows):
        for c in range(n_cols):
            ax_all[r, c].axis('off')

    wsi_name = os.path.basename(args.wsi)

    # Row 0
    col = 0
    draw_synthetic_figure(ax_all[0, col]); col += 1
    if real_trm is not None:
        draw_mask_with_regions( ax_all[0, col], real_trm, f'Otsu mask ({wsi_name})', show_index=si, linewidth=lw);  col += 1
        draw_thumb_with_regions(ax_all[0, col], real_trm, real_thumb, 'Otsu thumb',  show_index=si, linewidth=lw);  col += 1
    if hest_trm is not None:
        draw_mask_with_regions( ax_all[0, col], hest_trm, f'HEST mask ({wsi_name})', show_index=si, linewidth=lw); col += 1
        draw_thumb_with_regions(ax_all[0, col], hest_trm, hest_thumb, 'HEST thumb',  show_index=si, linewidth=lw); col += 1

    # Sweep rows
    row_base = 1
    for i, (label, trm) in enumerate(sweep_results):
        r = row_base + i // PER_ROW
        c = i %  PER_ROW
        draw_mask_with_regions(ax_all[r, c], trm, label, show_index=si, linewidth=lw)
    row_base += n_sweep_rows

    # Ops rows
    for i, (label, trm) in enumerate(ops_results):
        r = row_base + i // PER_ROW
        c = i %  PER_ROW
        draw_mask_with_regions(ax_all[r, c], trm, f'[ops] {label}', show_index=si, linewidth=lw)
    row_base += n_ops_rows

    # Tiling rows: seam panels, grid overlay, sweep panels
    if n_tiling_cells:
        panels = []
        for label, trm in (tiling_seam or []):
            panels.append(('mask', trm, f'[seam] {label}'))
        if tiling_grid is not None:
            panels.append(('grid', tiling_grid, '[grid overlay]'))
        for label, trm, n_h, n_w in (tiling_sweep or []):
            panels.append(('mask', trm, f'[sweep] {label}'))
        for i, (kind, payload, title) in enumerate(panels):
            r = row_base + i // PER_ROW
            c = i %  PER_ROW
            if kind == 'mask':
                draw_mask_with_regions(ax_all[r, c], payload, title, show_index=si, linewidth=lw)
            else:
                trm_g, n_h, n_w, ov, mp = payload
                draw_tile_grid_overlay(ax_all[r, c], trm_g, n_h, n_w, ov, mp, title)

    fig.tight_layout()
    out = args.out or os.path.join(job_result_dir('TissueMaskTest'),
                                    'tissue_mask__regions.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved {out}')
    print('All checks passed.')


if __name__ == '__main__':
    main()
