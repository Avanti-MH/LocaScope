#!/usr/bin/env python3
"""Comprehensive tests for utilities/PatchingLib (PatchGrid, PatchInfo, containers).

Merged from test_patchgrid_index.py, test_patch_info_coords.py, and
test_tissue_patch_container.py.

Sections:
  1. PatchGrid — layout counts, flat/unified indexing, offset metadata
  2. PatchInfo — for_query/for_wsi, to_level0(), grid offset coordinates
  3. Containers — QueryPatchContainer & TissuePatchContainer extraction, crop, real data
  4. Scale — resolve_scale / from_ds: which level, which downsample, which
     regions survive it. No model needed; the first two checks need no WSI.

Usage:
  python utilities/test_modules/test_patching_lib.py
  python utilities/test_modules/test_patching_lib.py --only grid coords
  python utilities/test_modules/test_patching_lib.py --only containers --size 64

Outputs (under result/PatchingLibTest/ by default):
  patch_grid__index.png
  patch_info__coords.png
  patch_container__grid.png
  patch_container__reconstruction.png
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage

from _paths import job_result_dir, setup_import_paths

setup_import_paths()

from PatchingLib import PatchGrid, PatchInfo, QueryPatchContainer, TissuePatchContainer
from TissuesRegionsMask import TissueRegion


# ══════════════════════════════════════════════════════════════════════════════
# 1. PatchGrid
# ══════════════════════════════════════════════════════════════════════════════

"""
Exhaustive indexing test for PatchGrid.

Tests:
  1.  Layout counts: grid_rows / grid_cols / overlap_rows / overlap_cols / __len__
  2.  Flat ↔ unified roundtrip: flat_to_unified then back must recover original flat i
  3.  patch_info_at(int) == patch_info_at(flat_to_unified(int))
  4.  flat_index_at(tuple) == flat_index_at(flat_index_at(tuple))  (idempotent)
  5.  flat_index_for_main / flat_index_for_overlap roundtrip
  6.  iter_infos() yields patch_info_at(i) for every flat i
  7.  IndexError for out-of-range flat int
  8.  IndexError for out-of-range unified tuple
  9.  IndexError for mixed-parity unified tuple (with overlap)
  10. No-overlap: any (r, c) is valid; no parity restriction
  11. Edge: image smaller than tile → empty grid
  12. Edge: image exactly one tile → 1 main patch, 0 overlap
  13. Edge: single row (height < 2*tile) → 0 overlap even if cols >= 2
  14. Edge: single col  (width  < 2*tile) → 0 overlap even if rows >= 2
  15. Non-divisible dimensions: tail must NOT produce a partial-tile patch
  16. x_offset / y_offset: PatchInfo.x/y include offset; roundtrip still works
  17. ds / level / mpp forwarded correctly to every PatchInfo

Output figure: flat/unified index diagram for a sample 3×3 grid with overlap.

Usage:
    python test_modules/test_patchgrid_index.py [--out PATH]
"""






# ── Layout count validation ───────────────────────────────────────────────────

def expected_count(length: int, tile: int) -> int:
    return sum(1 for s in range(0, length, tile) if s + tile <= length)


def validate_layout_counts(W, H, tile, has_overlap_expected=None):
    grid = PatchGrid.from_size(W, H, tile, overlap=True)
    er = expected_count(H, tile)
    ec = expected_count(W, tile)

    assert grid.grid_rows == er, f'grid_rows {grid.grid_rows} != {er} (W={W},H={H},tile={tile})'
    assert grid.grid_cols == ec, f'grid_cols {grid.grid_cols} != {ec}'

    if er >= 2 and ec >= 2:
        assert grid.overlap_rows == er - 1
        assert grid.overlap_cols == ec - 1
        assert grid.has_overlap
    else:
        assert grid.overlap_rows == 0 or not grid.has_overlap
        assert not grid.has_overlap

    expected_len = (
        er * ec + (er - 1) * (ec - 1) if grid.has_overlap else er * ec
    )
    assert len(grid) == expected_len, f'__len__ {len(grid)} != {expected_len}'
    assert len(grid.main_patch_infos)    == er * ec
    assert len(grid.overlap_patch_infos) == (
        (er - 1) * (ec - 1) if grid.has_overlap else 0
    )

    if has_overlap_expected is not None:
        assert grid.has_overlap == has_overlap_expected

    return grid


# ── Flat ↔ unified roundtrip ──────────────────────────────────────────────────

def validate_flat_unified_roundtrip(grid: PatchGrid):
    """For every flat i: flat_to_unified(i) → patch_info_at → flat_index_at → i"""
    for flat_i in range(len(grid)):
        u = grid.flat_to_unified(flat_i)
        info_via_flat   = grid.patch_info_at(flat_i)
        info_via_unified = grid.patch_info_at(u)
        assert info_via_flat == info_via_unified, (
            f'flat {flat_i} → unified {u}: '
            f'patch_info mismatch {info_via_flat} vs {info_via_unified}'
        )
        recovered = grid.flat_index_at(u)
        assert recovered == flat_i, (
            f'flat {flat_i} → unified {u} → flat {recovered} (mismatch)'
        )


# ── flat_index_for_main / overlap roundtrip ───────────────────────────────────

def validate_main_overlap_roundtrip(grid: PatchGrid):
    for info in grid.main_patch_infos:
        fi = grid.flat_index_for_main(info.row, info.col)
        assert grid.patch_info_at(fi) == info, (
            f'main ({info.row},{info.col}) flat={fi} roundtrip failed'
        )
        unified = (2 * info.row, 2 * info.col) if grid.has_overlap else (info.row, info.col)
        assert grid.flat_index_at(unified) == fi

    for info in grid.overlap_patch_infos:
        fi = grid.flat_index_for_overlap(info.row, info.col)
        assert grid.patch_info_at(fi) == info, (
            f'overlap ({info.row},{info.col}) flat={fi} roundtrip failed'
        )
        unified = (2 * info.row + 1, 2 * info.col + 1)
        assert grid.flat_index_at(unified) == fi


# ── iter_infos consistency ────────────────────────────────────────────────────

def validate_iter_infos(grid: PatchGrid):
    infos = list(grid.iter_infos())
    assert len(infos) == len(grid)
    for i, info in enumerate(infos):
        assert grid.patch_info_at(i) == info, f'iter_infos[{i}] mismatch'


# ── Index error cases ─────────────────────────────────────────────────────────

def validate_index_errors(grid: PatchGrid):
    # OOB flat index
    for bad in [-1, len(grid), len(grid) + 1]:
        try:
            grid.flat_index_at(bad)
            raise AssertionError(f'expected IndexError for flat {bad}')
        except IndexError:
            pass

    # OOB unified tuple (even row, valid col)
    oob_r = grid.grid_rows * 2
    try:
        grid.patch_info_at((oob_r, 0))
        raise AssertionError(f'expected IndexError for unified ({oob_r}, 0)')
    except IndexError:
        pass

    if grid.has_overlap:
        # mixed parity
        try:
            grid.patch_info_at((0, 1))
            raise AssertionError('expected IndexError for mixed parity (0, 1)')
        except IndexError:
            pass
        try:
            grid.patch_info_at((1, 0))
            raise AssertionError('expected IndexError for mixed parity (1, 0)')
        except IndexError:
            pass


# ── Offset: PatchInfo.x/y include offset ────────────────────────────────────

def validate_offset(W, H, tile, ox, oy, ds=1.0, level=2):
    # No mpp here, and none on PatchInfo. This test used to pass mpp= and assert
    # info.mpp, against a field PatchGrid has not had for a long time -- it was
    # already failing at eee3412, so nothing downstream ever depended on it.
    # Not reinstated: a patch knows its ds and its level, and mpp is
    # ds * wsi.base_mpp, so storing it would be the same quantity written twice
    # and free to disagree. TileInfo carries mpp legitimately -- it is a
    # sampling record whose entire purpose is mpp estimation -- and that is a
    # different dataclass.
    grid = PatchGrid.from_size(W, H, tile, overlap=True,
                               x_offset=ox, y_offset=oy, ds=ds, level=level)
    for info in grid.iter_infos():
        local_x = info.x - ox
        local_y = info.y - oy
        assert 0 <= local_x, f'local_x={local_x} < 0 (x={info.x}, ox={ox})'
        assert 0 <= local_y, f'local_y={local_y} < 0 (y={info.y}, oy={oy})'
        assert local_x + tile <= W, f'patch right edge {local_x+tile} > W={W}'
        assert local_y + tile <= H, f'patch bottom edge {local_y+tile} > H={H}'
        assert info.ds    == ds
        assert info.level == level

    # Roundtrip still works after offset
    validate_flat_unified_roundtrip(grid)
    validate_main_overlap_roundtrip(grid)


# ── Non-divisible dimensions ──────────────────────────────────────────────────

def validate_non_divisible(tile):
    """Tail pixels that don't fit a full tile must be excluded."""
    for W, H in [(tile + 1, tile + 1), (2 * tile + 1, tile + 1), (3 * tile - 1, 2 * tile - 1)]:
        grid = PatchGrid.from_size(W, H, tile, overlap=False)
        ec = expected_count(W, tile)
        er = expected_count(H, tile)
        assert grid.grid_cols == ec, f'W={W} tile={tile}: cols {grid.grid_cols} != {ec}'
        assert grid.grid_rows == er, f'H={H} tile={tile}: rows {grid.grid_rows} != {er}'
        # No patch should extend beyond (W, H)
        for info in grid.main_patch_infos:
            assert info.x + tile <= W, f'patch right {info.x+tile} > W={W}'
            assert info.y + tile <= H, f'patch bottom {info.y+tile} > H={H}'


# ── Figure: index diagram for a 3×3 overlap grid ─────────────────────────────

def draw_index_diagram(ax, grid: PatchGrid, tile: int):
    """Draw each patch cell with its flat index and unified (r,c) label."""
    ax.set_xlim(-0.5, grid.width + 0.5)
    ax.set_ylim(grid.height + 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')

    colors = {'main': '#4CAF50', 'overlap': '#F44336'}
    for i, info in enumerate(grid.iter_infos()):
        u = grid.flat_to_unified(i)
        rect = mpatches.Rectangle(
            (info.x, info.y), tile, tile,
            linewidth=1.2, edgecolor='white', facecolor=colors[info.kind], alpha=0.5,
        )
        ax.add_patch(rect)
        cx, cy = info.x + tile / 2, info.y + tile / 2
        ax.text(cx, cy - tile * 0.12, f'flat={i}', ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')
        ax.text(cx, cy + tile * 0.18, f'u={u}', ha='center', va='center',
                fontsize=6, color='#FFD700')

    legend = [
        mpatches.Patch(facecolor='#4CAF50', alpha=0.6, label='main'),
        mpatches.Patch(facecolor='#F44336', alpha=0.6, label='overlap'),
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=8,
              facecolor='#333', labelcolor='white')
    ax.set_title(
        f'PatchGrid {grid.grid_rows}×{grid.grid_cols} (overlap)\n'
        f'flat order: m,o,m,o,...  unified: even=main, odd=overlap',
        color='white', fontsize=9,
    )
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')


# ── All cases ─────────────────────────────────────────────────────────────────

def run_all_patchgrid(tile: int = 128):
    results = []

    # 1. Standard grids: various sizes
    cases = [
        (512, 512, tile, True),
        (384, 256, tile, True),
        (256, 256, tile, True),   # 2×2 main → 1×1 overlap
        (tile, tile, tile, False), # single patch, no overlap
        (tile - 1, tile, tile, False),  # image < tile in one dim
        (0, 0, tile, False),       # empty
    ]
    for W, H, t, expected_ovl in cases:
        grid = validate_layout_counts(W, H, t, expected_ovl)
        if len(grid) > 0:
            validate_flat_unified_roundtrip(grid)
            validate_main_overlap_roundtrip(grid)
            validate_iter_infos(grid)
            validate_index_errors(grid)
        results.append((W, H, grid))
        print(f'[PASS] layout+index ({W}x{H}, tile={t}): '
              f'{grid.grid_rows}x{grid.grid_cols} main, '
              f'{grid.overlap_rows}x{grid.overlap_cols} overlap, '
              f'len={len(grid)}')

    # 2. Single row / single col
    for W, H in [(3 * tile, tile), (tile, 3 * tile)]:
        grid = PatchGrid.from_size(W, H, tile, overlap=True)
        assert not grid.has_overlap, f'{W}x{H}: expected no overlap (only 1 row or col)'
        validate_flat_unified_roundtrip(grid)
        validate_index_errors(grid)
        print(f'[PASS] single-{"row" if H==tile else "col"} ({W}x{H}): no overlap as expected')

    # 3. Non-divisible dimensions
    validate_non_divisible(tile)
    print(f'[PASS] non-divisible: tail pixels correctly excluded')

    # 4. Offset + ds/level/mpp forwarding
    validate_offset(256, 256, tile, ox=128, oy=64, ds=4.0, level=2)
    print(f'[PASS] offset + ds/level/mpp: coordinates and metadata verified')

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. PatchInfo / coordinates
# ══════════════════════════════════════════════════════════════════════════════

# ── PatchInfo factory validation ──────────────────────────────────────────────

def validate_for_query():
    info = PatchInfo.for_query(row=1, col=2, x=64, y=32, size_px=128, kind='main')
    assert info.ds == 1.0,    f'for_query ds={info.ds}, expected 1.0'
    assert info.level is None, f'for_query level={info.level}, expected None'
    assert info.x == 64 and info.y == 32
    assert info.size_px == 128
    assert info.kind == 'main'
    print('[PASS] PatchInfo.for_query')


def validate_for_wsi():
    info = PatchInfo.for_wsi(row=0, col=0, x=100, y=200, size_px=256,
                             kind='main', ds=4.0, level=2)
    assert info.ds == 4.0
    assert info.level == 2
    assert info.x == 100 and info.y == 200
    print('[PASS] PatchInfo.for_wsi')


def validate_to_level0():
    cases = [
        # (x, y, size_px, ds)  ->  expected level-0 (x0, y0, s0)
        (100, 200, 256, 4.0,  400,  800, 1024),
        (50,   75, 128, 2.0,  100,  150,  256),
        (64,   64, 128, 1.0,   64,   64,  128),  # ds=1: no change
        (33,   17, 100, 3.0,   99,   51,  300),
    ]
    for x, y, s, ds, ex, ey, es in cases:
        info = PatchInfo.for_wsi(0, 0, x, y, s, 'main', ds=ds)
        l0 = info.to_level0()
        assert l0.x == ex, f'to_level0 x: got {l0.x}, expected {ex} (ds={ds})'
        assert l0.y == ey, f'to_level0 y: got {l0.y}, expected {ey} (ds={ds})'
        assert l0.size_px == es, f'to_level0 size_px: got {l0.size_px}, expected {es}'
        assert l0.ds == 1.0
        assert l0.level == 0
    print('[PASS] PatchInfo.to_level0 (4 cases)')


# ── PatchGrid offset validation ───────────────────────────────────────────────

def validate_grid_offset(size: int):
    """
    PatchGrid built with x_offset / y_offset:
    PatchInfo.x/y must equal offset + local position.
    Extracting from a full image using the offset must match direct slicing.
    """
    W, H = 512, 512
    ox, oy = 256, 128  # offset in level-N space

    # Region: w=256, h=384 starting at (ox, oy)
    rw, rh = W - ox, H - oy
    grid = PatchGrid.from_size(rw, rh, size, overlap=False,
                               x_offset=ox, y_offset=oy, ds=1.0)

    for info in grid.iter_infos():
        local_x = info.x - ox
        local_y = info.y - oy
        assert 0 <= local_x and local_x + size <= rw, (
            f'grid offset x out of region: info.x={info.x}, ox={ox}'
        )
        assert 0 <= local_y and local_y + size <= rh, (
            f'grid offset y out of region: info.y={info.y}, oy={oy}'
        )
        assert info.x == ox + local_x
        assert info.y == oy + local_y

    print(f'[PASS] PatchGrid x_offset/y_offset: {len(grid)} patches, coords verified')
    return grid, ox, oy, rw, rh


def validate_grid_offset_pixels(size: int):
    """
    TissuePatchContainer (full + region): patches from full image with region offset
    must equal direct numpy slicing.
    """
    W, H = 512, 512
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Unique pixel values: encode (y, x) in R and G channels
    ys = np.arange(H, dtype=np.uint8)[:, None] * np.ones(W, dtype=np.uint8)[None, :]
    xs = np.ones(H, dtype=np.uint8)[:, None] * np.arange(W, dtype=np.uint8)[None, :]
    img[:, :, 0] = ys
    img[:, :, 1] = xs
    img[:, :, 2] = 128

    region = TissueRegion(x=128, y=64, w=256, h=384, index=0)
    ds = 1.0
    rx, ry = int(region.x / ds), int(region.y / ds)
    rw, rh = int(region.w / ds), int(region.h / ds)

    tc = TissuePatchContainer(img.copy(), region=region, img_ds=ds, is_crop=False)
    tc.extract_all(size, overlap=False)

    main_patches = list(tc.iter_main())
    row_starts = [i for i in range(0, rh, size) if i + size <= rh]
    col_starts = [j for j in range(0, rw, size) if j + size <= rw]

    idx = 0
    for i in row_starts:
        for j in col_starts:
            expected = img[ry + i:ry + i + size, rx + j:rx + j + size]
            assert np.array_equal(main_patches[idx], expected), (
                f'pixel mismatch at region-local ({i},{j})'
            )
            idx += 1

    print(f'[PASS] PatchGrid offset pixel correctness: {idx} patches verified')
    return img, region, ds


# ── Figure ────────────────────────────────────────────────────────────────────

def draw_info_rects(ax, infos, size, color, lw=1.2):
    for info in infos:
        rect = mpatches.Rectangle(
            (info.x, info.y), size, size,
            fill=False, edgecolor=color, linewidth=lw,
        )
        ax.add_patch(rect)


# ══════════════════════════════════════════════════════════════════════════════
# 3. QueryPatchContainer / TissuePatchContainer
# ══════════════════════════════════════════════════════════════════════════════


# ── Synthetic image ───────────────────────────────────────────────────────────

def make_gradient_image(width: int, height: int) -> np.ndarray:
    """Each pixel encodes (x, y) in R/G channels → unique values everywhere."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = (np.arange(width,  dtype=np.float32) * 255 / max(width  - 1, 1)
                    ).astype(np.uint8)[np.newaxis, :]
    img[:, :, 1] = (np.arange(height, dtype=np.float32) * 255 / max(height - 1, 1)
                    ).astype(np.uint8)[:, np.newaxis]
    img[:, :, 2] = 128
    return img


# ── Grid geometry helpers ─────────────────────────────────────────────────────

def main_origins(w: int, h: int, size: int):
    rows = [i for i in range(0, h, size) if i + size <= h]
    cols = [j for j in range(0, w, size) if j + size <= w]
    return [(i, j) for i in rows for j in cols]


def overlap_origins(w: int, h: int, size: int):
    rows = [i for i in range(0, h, size) if i + size <= h]
    cols = [j for j in range(0, w, size) if j + size <= w]
    half = size // 2
    return [
        (rows[ri] + half, cols[ci] + half)
        for ri in range(len(rows) - 1)
        for ci in range(len(cols) - 1)
    ]


# ── Universal helpers ─────────────────────────────────────────────────────────

def validate_patch_shapes(container, size: int, label: str = ''):
    bad = [(i, p.shape) for i, p in enumerate(container)
           if p.shape != (size, size, 3)]
    assert not bad, (
        f'{label}shape mismatch at indices {[i for i,_ in bad]}: '
        f'{[s for _,s in bad]}, expected ({size},{size},3)'
    )
    print(f'[PASS] {label}shapes: all {len(list(container))} patches are ({size},{size},3)')


def validate_iterators(container, label: str = ''):
    flat = list(container)
    assert flat == [container[i] for i in range(len(container))]

    grid = container.grid
    main_by_iter = list(container.iter_main())
    main_by_idx  = [container[grid.flat_index_for_main(info.row, info.col)]
                    for info in grid.main_patch_infos]
    assert main_by_iter == main_by_idx

    if grid.has_overlap:
        ovl_by_iter = list(container.iter_overlap())
        ovl_by_idx  = [container[grid.flat_index_for_overlap(info.row, info.col)]
                       for info in grid.overlap_patch_infos]
        assert ovl_by_iter == ovl_by_idx
        assert len(flat) == len(main_by_iter) + len(ovl_by_iter)

    assert [p for b in container.iter_batches(batch_size=3) for p in b] == flat
    print(f'[PASS] {label}iterators: __iter__ / iter_main / iter_overlap / iter_batches OK')


# ── QueryPatchContainer ───────────────────────────────────────────────────────

def validate_qpc_main(qc: QueryPatchContainer, size: int, label: str = 'QPC '):
    grid = qc.grid
    origins = main_origins(qc.width, qc.height, size)
    main_patches = list(qc.iter_main())
    assert len(main_patches) == len(origins)
    for idx, (y, x) in enumerate(origins):
        r, c = divmod(idx, grid.grid_cols)
        expected = qc.img[y:y + size, x:x + size]
        flat_i = grid.flat_index_for_main(r, c) if grid.has_overlap else idx
        grid_i = (2*r, 2*c) if grid.has_overlap else (r, c)
        for lbl, patch in (
            (f'iter_main[{idx}]', main_patches[idx]),
            (f'[{flat_i}]', qc[flat_i]),
            (f'[{grid_i}]', qc[grid_i]),
        ):
            assert np.array_equal(patch, expected), f'{label}{lbl} mismatch at ({r},{c})'
    print(f'[PASS] {label}main: {len(origins)} patches, 3 access methods verified')


def validate_qpc_overlap(qc: QueryPatchContainer, size: int, label: str = 'QPC '):
    grid = qc.grid
    half = size // 2
    origins = overlap_origins(qc.width, qc.height, size)
    ovl_patches = list(qc.iter_overlap())
    assert len(ovl_patches) == len(origins)
    for idx, (y, x) in enumerate(origins):
        r, c = divmod(idx, grid.overlap_cols)
        expected = qc.img[y:y + size, x:x + size]
        flat_i = grid.flat_index_for_overlap(r, c)
        for lbl, patch in (
            (f'iter_overlap[{idx}]', ovl_patches[idx]),
            (f'[{flat_i}]', qc[flat_i]),
            (f'[{2*r+1},{2*c+1}]', qc[2*r+1, 2*c+1]),
        ):
            assert np.array_equal(patch, expected), f'{label}{lbl} mismatch at ({r},{c})'
        # Corner-pixel 4-neighbour relationship
        p = qc[2*r+1, 2*c+1]
        assert np.array_equal(p[:half, :half],  qc[2*r,   2*c  ][half:, half:])
        assert np.array_equal(p[:half, half:],  qc[2*r,   2*c+2][half:, :half])
        assert np.array_equal(p[half:, :half],  qc[2*r+2, 2*c  ][:half, half:])
        assert np.array_equal(p[half:, half:],  qc[2*r+2, 2*c+2][:half, :half])
    print(f'[PASS] {label}overlap: {len(origins)} patches, pixel + corner-pixel OK')


def validate_qpc_no_overlap(img: np.ndarray, size: int):
    qc = QueryPatchContainer(img.copy())
    qc.extract_all(size, overlap=False)
    assert not qc.grid.has_overlap
    assert list(qc.iter_overlap()) == []
    assert len(qc) == len(list(qc.iter_main()))
    assert list(qc) == list(qc.iter_main())
    # Without overlap, any in-range (r, c) is valid — no parity restriction
    qc[0, 1]
    print(f'[PASS] QPC overlap=False: {len(qc)} main patches, mixed-parity tuple OK')


def validate_qpc_factory_methods(img: np.ndarray, size: int):
    ref = QueryPatchContainer(img.copy())
    ref.extract_all(size, overlap=True)
    ref_patches = list(ref)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tmppath = f.name
    try:
        PILImage.fromarray(img).save(tmppath)
        cases = [
            ('from_path',  QueryPatchContainer.from_path(tmppath)),
            ('from_pil',   QueryPatchContainer.from_pil(PILImage.open(tmppath).convert('RGB'))),
            ('from_array', QueryPatchContainer.from_array(img.copy())),
        ]
        for label, qc in cases:
            qc.extract_all(size, overlap=True)
            for i, (p, r) in enumerate(zip(qc, ref_patches)):
                assert np.array_equal(p, r), f'QPC {label}: patch[{i}] differs'
            print(f'[PASS] QPC {label}: identical to direct constructor')
    finally:
        os.unlink(tmppath)


def validate_qpc_multichannel(size: int):
    H, W = 256, 256
    # RGBA → drop alpha, keep RGB
    rgba = np.random.randint(0, 200, (H, W, 4), dtype=np.uint8)
    qc_rgba = QueryPatchContainer(rgba.copy())
    assert qc_rgba.img.shape == (H, W, 3), f'RGBA shape: {qc_rgba.img.shape}'
    assert np.array_equal(qc_rgba.img, rgba[:, :, :3])
    print('[PASS] QPC RGBA→RGB: alpha dropped, RGB preserved')

    # Grayscale (2D) → stack 3 identical channels
    gray = np.random.randint(0, 200, (H, W), dtype=np.uint8)
    qc_gray = QueryPatchContainer(gray.copy())
    assert qc_gray.img.shape == (H, W, 3), f'gray shape: {qc_gray.img.shape}'
    assert np.all(qc_gray.img[:, :, 0] == gray)
    assert np.all(qc_gray.img[:, :, 1] == gray)
    assert np.all(qc_gray.img[:, :, 2] == gray)
    print('[PASS] QPC grayscale→RGB: all 3 channels equal original')


def validate_qpc_errors(img: np.ndarray, size: int):
    fresh = QueryPatchContainer(img.copy())
    try:
        _ = fresh[0]
        raise AssertionError('expected RuntimeError before extract_all')
    except RuntimeError:
        pass

    qc = QueryPatchContainer(img.copy())
    qc.extract_all(size, overlap=True)

    try:
        _ = qc[len(qc)]
        raise AssertionError('expected IndexError for OOB flat')
    except IndexError:
        pass

    try:
        _ = qc[0, 1]
        raise AssertionError('expected IndexError for mixed parity (0,1) with overlap')
    except IndexError:
        pass

    print('[PASS] QPC errors: RuntimeError / OOB IndexError / mixed-parity IndexError')


# ── TissuePatchContainer ─────────────────────────────────────────────────────

def validate_tpc_case1(tc: TissuePatchContainer, img: np.ndarray, size: int):
    origins = main_origins(tc.width, tc.height, size)
    patches = list(tc.iter_main())
    assert len(patches) == len(origins)
    for idx, (y, x) in enumerate(origins):
        assert np.array_equal(patches[idx], img[y:y+size, x:x+size]), (
            f'case1 mismatch at ({y},{x})')
    print(f'[PASS] TPC case1 (full, no region): {len(origins)} patches')


def validate_tpc_case2(tc: TissuePatchContainer, img: np.ndarray,
                        region: TissueRegion, size: int, ds: float):
    rx, ry = int(region.x / ds), int(region.y / ds)
    rw, rh = int(region.w / ds), int(region.h / ds)
    origins = main_origins(rw, rh, size)
    patches = list(tc.iter_main())
    assert len(patches) == len(origins)
    for idx, (y, x) in enumerate(origins):
        expected = img[ry+y:ry+y+size, rx+x:rx+x+size]
        assert np.array_equal(patches[idx], expected), (
            f'case2 mismatch at global ({ry+y},{rx+x})')
    print(f'[PASS] TPC case2 (full + region): {len(origins)} patches, offset verified')
    return patches


def validate_tpc_case3(tc: TissuePatchContainer, size: int, ref: list):
    patches = list(tc.iter_main())
    assert len(patches) == len(ref)
    for i, (p3, p2) in enumerate(zip(patches, ref)):
        assert np.array_equal(p3, p2), f'case3 main[{i}] differs from case2'
    print(f'[PASS] TPC case3 (is_crop + region): {len(patches)} main patches match case2')
    return patches


def validate_tpc_case3_overlap(tc2: TissuePatchContainer, tc3: TissuePatchContainer):
    """Overlap patches from is_crop must be pixel-identical to full-image + region."""
    ovl2 = list(tc2.iter_overlap())
    ovl3 = list(tc3.iter_overlap())
    assert len(ovl2) == len(ovl3), (
        f'overlap count: case2={len(ovl2)}, case3={len(ovl3)}')
    for i, (p2, p3) in enumerate(zip(ovl2, ovl3)):
        assert np.array_equal(p2, p3), f'case3 overlap[{i}] differs from case2'
    print(f'[PASS] TPC case3 overlap: {len(ovl2)} overlap patches match case2')


def validate_tpc_ds_not_1(size: int):
    """
    img_ds=4.0 (level-2 equivalent): verify both x and y region offsets are
    correctly divided by ds, and that at_level / ds are forwarded to PatchInfo.

    Synthetic image: 512×512 at level-N (ds=4), representing a 2048×2048 level-0 WSI.
    Region (level-0): x=256, y=384, w=1024, h=768
    → level-N:        x=64,  y=96,  w=256,  h=192
    """
    W, H = 512, 512
    img   = make_gradient_image(W, H)
    ds    = 4.0
    level = 2
    # Level-0 region coords
    region = TissueRegion(x=256, y=384, w=1024, h=768)

    tc = TissuePatchContainer(img.copy(), region=region, img_ds=ds,
                              is_crop=False, at_level=level)
    tc.extract_all(size, overlap=False)

    rx_n = int(region.x / ds)   # 64
    ry_n = int(region.y / ds)   # 96
    rw_n = int(region.w / ds)   # 256
    rh_n = int(region.h / ds)   # 192

    origins = main_origins(rw_n, rh_n, size)
    patches = list(tc.iter_main())
    assert len(patches) == len(origins), (
        f'ds=4 patch count {len(patches)} != {len(origins)}')
    for idx, (y, x) in enumerate(origins):
        expected = img[ry_n+y:ry_n+y+size, rx_n+x:rx_n+x+size]
        assert np.array_equal(patches[idx], expected), (
            f'ds=4 mismatch at level-N ({ry_n+y},{rx_n+x})')

    # PatchInfo metadata must reflect the constructor arguments
    for info in tc.grid.main_patch_infos:
        assert info.ds    == ds,    f'PatchInfo.ds={info.ds}'
        assert info.level == level, f'PatchInfo.level={info.level}'
        # x/y in PatchInfo are level-N global coords (include grid offset)
        assert info.x >= rx_n, f'PatchInfo.x={info.x} < rx_n={rx_n}'
        assert info.y >= ry_n, f'PatchInfo.y={info.y} < ry_n={ry_n}'

    print(f'[PASS] TPC ds=4.0 level={level}: {len(origins)} patches, '
          f'x/y offset ({rx_n},{ry_n}), ds/level in PatchInfo verified')


def validate_tpc_region_y_offset(img: np.ndarray, size: int):
    """Region with non-zero y: both x and y offsets must be applied."""
    H, W = img.shape[:2]
    ds = 1.0
    region = TissueRegion(x=W // 2, y=H // 2, w=W // 2, h=H // 2)
    tc = TissuePatchContainer(img.copy(), region=region, img_ds=ds, is_crop=False)
    tc.extract_all(size, overlap=False)

    rx, ry = W // 2, H // 2
    rw, rh = W // 2, H // 2
    origins = main_origins(rw, rh, size)
    patches = list(tc.iter_main())
    assert len(patches) == len(origins)
    for idx, (y, x) in enumerate(origins):
        expected = img[ry+y:ry+y+size, rx+x:rx+x+size]
        assert np.array_equal(patches[idx], expected), (
            f'y-offset mismatch at global ({ry+y},{rx+x})')

    print(f'[PASS] TPC region y_offset={H//2}: {len(origins)} patches verified')


def validate_tpc_patchinfo_meta(img: np.ndarray, size: int):
    """at_level must be forwarded to every PatchInfo in the grid."""
    ds, lv = 4.0, 2
    tc = TissuePatchContainer(img.copy(), img_ds=ds, at_level=lv)
    tc.extract_all(size, overlap=True)
    for info in tc.grid.iter_infos():
        assert info.ds    == ds,  f'PatchInfo.ds={info.ds}, expected {ds}'
        assert info.level == lv,  f'PatchInfo.level={info.level}, expected {lv}'
    print(f'[PASS] TPC PatchInfo meta: ds/level forwarded to all {len(tc.grid)} patches')


def validate_tpc_no_overlap(img: np.ndarray, region: TissueRegion, ds: float, size: int):
    tc = TissuePatchContainer(img.copy(), region=region, img_ds=ds, is_crop=False)
    tc.extract_all(size, overlap=False)
    assert not tc.grid.has_overlap
    assert list(tc.iter_overlap()) == []
    assert len(tc) == len(list(tc.iter_main()))
    print(f'[PASS] TPC overlap=False: {len(tc)} main patches, no overlap')


def validate_tpc_factory_methods(img: np.ndarray, size: int):
    ref = TissuePatchContainer(img.copy())
    ref.extract_all(size, overlap=True)
    ref_patches = list(ref)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tmppath = f.name
    try:
        PILImage.fromarray(img).save(tmppath)
        cases = [
            ('from_path',  TissuePatchContainer.from_path(tmppath)),
            ('from_pil',   TissuePatchContainer.from_pil(PILImage.open(tmppath).convert('RGB'))),
            ('from_array', TissuePatchContainer.from_array(img.copy())),
        ]
        for label, tc in cases:
            tc.extract_all(size, overlap=True)
            for i, (p, r) in enumerate(zip(tc, ref_patches)):
                assert np.array_equal(p, r), f'TPC {label}: patch[{i}] differs'
            print(f'[PASS] TPC {label}: identical to direct constructor')
    finally:
        os.unlink(tmppath)


def validate_tpc_overlap_corner(tc: TissuePatchContainer, size: int, label: str = 'TPC '):
    grid = tc.grid
    if not grid.has_overlap:
        print(f'[SKIP] {label}overlap corner: grid too small for overlap')
        return
    half = size // 2
    for info in grid.overlap_patch_infos:
        r, c = info.row, info.col
        p = tc[2*r+1, 2*c+1]
        assert np.array_equal(p[:half, :half],  tc[2*r,   2*c  ][half:, half:])
        assert np.array_equal(p[:half, half:],  tc[2*r,   2*c+2][half:, :half])
        assert np.array_equal(p[half:, :half],  tc[2*r+2, 2*c  ][:half, half:])
        assert np.array_equal(p[half:, half:],  tc[2*r+2, 2*c+2][:half, :half])
    print(f'[PASS] {label}overlap corner-pixel: {len(grid.overlap_patch_infos)} patches verified')


def validate_tpc_errors(img: np.ndarray, size: int):
    try:
        TissuePatchContainer(img.copy(), is_crop=True, region=None)
        raise AssertionError('expected ValueError for is_crop without region')
    except ValueError:
        pass

    fresh = TissuePatchContainer(img.copy())
    try:
        _ = fresh[0]
        raise AssertionError('expected RuntimeError before extract_all')
    except RuntimeError:
        pass

    tc = TissuePatchContainer(img.copy())
    tc.extract_all(size, overlap=True)
    try:
        _ = tc[len(tc)]
        raise AssertionError('expected IndexError for OOB flat')
    except IndexError:
        pass

    print('[PASS] TPC errors: ValueError / RuntimeError / OOB IndexError')


# ── crop() ────────────────────────────────────────────────────────────────────

def validate_container_crop(container, label: str = ''):
    """
    Verify container.crop() correctness across grid / pixel units, size / bottom_right,
    padding, clamping, and error cases.

    Assertions:
      - grid + bottom_right: img_origin & new-grid offset correct, patches identical
      - grid + size + pad:   pad clamps within grid bounds
      - pixel + size:        top-left floor / bottom-right ceil (round-up rule)
      - pixel + pad:         pad_unit='pixel' ceils to full grid tiles
      - PatchInfo.x/y stays in level-N global coords after crop
      - 6 invalid argument combos raise ValueError
    """
    grid = container.grid
    ts   = grid.tile_size
    R, C = grid.grid_rows, grid.grid_cols
    if R < 2 or C < 2:
        print(f'[SKIP] {label}crop: grid too small ({R}x{C})')
        return

    # ── (1) grid unit: (r0,c0) + bottom_right ─────────────────────────────────
    r0, c0, r1, c1 = 0, 0, min(2, R), min(2, C)
    sub = container.crop((r0, c0), bottom_right=(r1, c1), unit='grid')

    assert sub.width  == (c1 - c0) * ts
    assert sub.height == (r1 - r0) * ts
    assert sub.img_origin_x == grid.x_offset + c0 * ts
    assert sub.img_origin_y == grid.y_offset + r0 * ts
    assert sub.grid.x_offset == sub.img_origin_x   # invariant: grid starts at img[0,0]
    assert sub.grid.y_offset == sub.img_origin_y

    for r in range(r1 - r0):
        for c in range(c1 - c0):
            new_idx = (2*r, 2*c)         if sub.grid.has_overlap else (r, c)
            old_idx = (2*(r0+r), 2*(c0+c)) if grid.has_overlap    else (r0+r, c0+c)
            assert np.array_equal(sub[new_idx], container[old_idx]), \
                f'{label}crop main patch ({r},{c}) mismatch'

    # PatchInfo.x/y is still level-N global
    for info in sub.grid.main_patch_infos:
        orig_r = (info.y - grid.y_offset) // ts
        orig_c = (info.x - grid.x_offset) // ts
        assert r0 <= orig_r < r1 and c0 <= orig_c < c1, \
            f'{label}crop PatchInfo.x/y not global: got ({info.y},{info.x})'
    print(f'[PASS] {label}crop grid+corners: {sub.grid.grid_rows}x{sub.grid.grid_cols}, '
          f'origin=({sub.img_origin_x},{sub.img_origin_y}), patches match, x/y stays global')

    # ── (2) grid unit: size + pad ─────────────────────────────────────────────
    sub2 = container.crop((0, 0), size=(1, 1), unit='grid', pad=1)
    # size=(1,1) → [0,1); pad=1 → [-1,2); clamp → [0, min(R,2))
    assert sub2.grid.grid_rows == min(2, R)
    assert sub2.grid.grid_cols == min(2, C)
    print(f'[PASS] {label}crop grid+size+pad: {sub2.grid.grid_rows}x{sub2.grid.grid_cols} '
          'after clamp')

    # ── (3) pixel unit: non-aligned size → round up ───────────────────────────
    x0, y0 = grid.x_offset + 1, grid.y_offset + 1
    w, h   = ts + 3, ts + 3
    sub3 = container.crop((x0, y0), size=(w, h), unit='pixel')
    # top-left floor: (1//ts, 1//ts) = (0, 0)
    # bottom-right ceil: ceil((ts+4)/ts) = 2
    assert sub3.grid.grid_rows == min(2, R)
    assert sub3.grid.grid_cols == min(2, C)
    print(f'[PASS] {label}crop pixel+size+roundup: {sub3.grid.grid_rows}x{sub3.grid.grid_cols} '
          '(bottom-right ceiled to tile boundary)')

    # ── (4) pixel unit: pad (should ceil to whole grid tiles) ─────────────────
    sub4 = container.crop((grid.x_offset, grid.y_offset), size=(ts, ts),
                          unit='pixel', pad=ts + 1, pad_unit='pixel')
    # size=(ts,ts) → grid [0,1); pad_g = ceil((ts+1)/ts) = 2 → [-2, 3); clamp → [0, min(R,3))
    assert sub4.grid.grid_rows == min(3, R)
    assert sub4.grid.grid_cols == min(3, C)
    print(f'[PASS] {label}crop pixel-pad (ceil to grid): '
          f'{sub4.grid.grid_rows}x{sub4.grid.grid_cols}')

    # ── (5) invalid arg combos → ValueError ───────────────────────────────────
    bad_cases = [
        dict(top_left=(0, 0)),                                          # no br, no size
        dict(top_left=(0, 0), bottom_right=(1, 1), size=(1, 1)),        # both
        dict(top_left=(0, 0), size=(1, 1), unit='what'),                # bad unit
        dict(top_left=(0, 0), size=(1, 1), pad=1, pad_unit='what'),     # bad pad_unit
        dict(top_left=(0, 0), bottom_right=(0, 0)),                     # empty (r0==r1)
        dict(top_left=(R, C), size=(1, 1)),                             # out-of-bounds → empty
    ]
    for kwargs in bad_cases:
        try:
            container.crop(**kwargs)
            raise AssertionError(f'{label}crop expected ValueError for {kwargs}')
        except ValueError:
            pass
    print(f'[PASS] {label}crop errors: {len(bad_cases)} invalid arg combos raise ValueError')

    return sub  # for downstream figure demo


def validate_crop_before_extract():
    """crop() before extract_all() must raise RuntimeError (same contract as __getitem__)."""
    img  = make_gradient_image(128, 128)
    fresh_qc = QueryPatchContainer(img.copy())
    fresh_tc = TissuePatchContainer(img.copy())
    for c, name in [(fresh_qc, 'QPC'), (fresh_tc, 'TPC')]:
        try:
            c.crop((0, 0), size=(1, 1), unit='grid')
            raise AssertionError(f'{name} crop before extract must raise RuntimeError')
        except RuntimeError:
            pass
    print('[PASS] crop before extract: QPC / TPC both raise RuntimeError')


def validate_tpc_crop_extra_fields(tc: TissuePatchContainer, label: str = 'TPC '):
    """_copy_extra_after_crop must shallow-copy tissue_region / img_ds / is_crop / at_level."""
    sub = tc.crop((0, 0), size=(2, 2), unit='grid')
    assert sub.tissue_region is tc.tissue_region, f'{label}crop lost tissue_region'
    assert sub.img_ds        == tc.img_ds,        f'{label}crop img_ds mismatch'
    assert sub.is_crop       == tc.is_crop,       f'{label}crop is_crop mismatch'
    assert sub.at_level      == tc.at_level,      f'{label}crop at_level mismatch'
    print(f'[PASS] {label}crop extra fields shallow-copied '
          '(tissue_region / img_ds / is_crop / at_level)')


# ── Real data tests ───────────────────────────────────────────────────────────

def test_real_query(path: str, size: int) -> QueryPatchContainer:
    qc = QueryPatchContainer(path)
    qc.extract_all(size, overlap=True)
    validate_patch_shapes(qc, size, f'real-query({os.path.basename(path)}) ')
    validate_iterators(qc, f'real-query ')
    validate_qpc_main(qc, size, label=f'real-query ')
    if qc.grid.has_overlap:
        validate_qpc_overlap(qc, size, label=f'real-query ')
    # from_path vs from_array must yield identical patches
    qc2 = QueryPatchContainer.from_array(qc.img.copy())
    qc2.extract_all(size, overlap=True)
    for i, (p1, p2) in enumerate(zip(qc, qc2)):
        assert np.array_equal(p1, p2), f'real query from_array patch[{i}] differs'
    print(f'[PASS] Real query {os.path.basename(path)}: {qc.width}x{qc.height}, '
          f'{len(qc)} patches (size={size})')
    return qc


def test_real_roi_as_query(path: str, size: int) -> QueryPatchContainer:
    """RoI PNG used as a plain query image (no region info)."""
    qc = QueryPatchContainer(path)
    qc.extract_all(size, overlap=True)
    validate_patch_shapes(qc, size, 'roi-as-query ')
    validate_iterators(qc, 'roi-as-query ')
    validate_qpc_main(qc, size, label='roi-as-query ')
    if qc.grid.has_overlap:
        validate_qpc_overlap(qc, size, label='roi-as-query ')
    print(f'[PASS] RoI as query {os.path.basename(path)}: {qc.width}x{qc.height}, '
          f'{len(qc)} patches (size={size})')
    return qc


def test_real_roi(path: str, size: int) -> TissuePatchContainer:
    tc = TissuePatchContainer(path)
    tc.extract_all(size, overlap=True)
    validate_patch_shapes(tc, size, 'real-roi ')
    validate_iterators(tc, 'real-roi ')
    validate_tpc_overlap_corner(tc, size, 'real-roi ')
    print(f'[PASS] Real RoI {os.path.basename(path)}: {tc.width}x{tc.height}, '
          f'{len(tc)} patches (size={size})')
    return tc


def test_real_wsi(path: str, level: int, size: int) -> TissuePatchContainer:
    """Load the full WSI level image."""
    import openslide
    wsi = openslide.OpenSlide(path)
    W_l, H_l = wsi.level_dimensions[level]
    ds = wsi.level_downsamples[level]
    arr = np.array(wsi.read_region((0, 0), level, (W_l, H_l)).convert('RGB'))
    wsi.close()

    tc = TissuePatchContainer(arr, img_ds=ds)
    tc.extract_all(size, overlap=True)
    validate_patch_shapes(tc, size, 'real-wsi-full ')
    validate_iterators(tc, 'real-wsi-full ')
    validate_tpc_overlap_corner(tc, size, 'real-wsi-full ')
    print(f'[PASS] Real WSI {os.path.basename(path)} level={level} (ds={ds:.0f}) '
          f'{W_l}x{H_l}: {len(tc)} patches (size={size})')
    return tc


def test_real_wsi_from_openslide(path: str, level: int, size: int) -> TissuePatchContainer:
    """Test from_openslide factory method."""
    import openslide
    wsi = openslide.OpenSlide(path)
    level = min(level, wsi.level_count - 1)
    W_l, H_l = wsi.level_dimensions[level]
    tc = TissuePatchContainer.from_openslide(wsi, at_level=level)
    wsi.close()
    tc.extract_all(size, overlap=True)
    validate_patch_shapes(tc, size, 'from_openslide ')
    validate_iterators(tc, 'from_openslide ')
    validate_tpc_overlap_corner(tc, size, 'from_openslide ')
    print(f'[PASS] from_openslide level={level} ({W_l}x{H_l}): {len(tc)} patches')
    return tc


# ── Reconstruction ────────────────────────────────────────────────────────────

def reconstruct_image(container, main_only: bool = True):
    """
    Stitch patches back using PatchInfo.x/y as the destination coordinates.

    For QPC   : info.x/y are image-local coords (img_origin = 0).
    For TPC   : info.x/y are level-N global; subtract img_origin to get local.

    Returns
    -------
    canvas   : (H, W, 3) uint8 — reconstructed image (uncovered pixels = black)
    coverage : (H, W) bool    — True where at least one patch was written
    """
    canvas   = np.zeros((container.height, container.width, 3), dtype=np.uint8)
    coverage = np.zeros((container.height, container.width), dtype=bool)
    ox = getattr(container, 'img_origin_x', 0)
    oy = getattr(container, 'img_origin_y', 0)
    grid = container.grid

    if main_only:
        pairs = [(grid.flat_index_for_main(info.row, info.col), info)
                 for info in grid.main_patch_infos]
    else:
        pairs = [(i, grid.patch_info_at(i)) for i in range(len(grid))]

    for flat_i, info in pairs:
        patch = container[flat_i]
        lx = info.x - ox
        ly = info.y - oy
        s  = info.size_px
        canvas[ly:ly + s, lx:lx + s] = patch
        coverage[ly:ly + s, lx:lx + s] = True
    return canvas, coverage


def draw_reconstruction_row(axes_row, container, source_img: np.ndarray,
                             size: int, title: str = ''):
    """
    Fill one row of 4 axes with the reconstruction comparison:
      col 0 : source image + main-grid overlay
      col 1 : reconstructed from main patches
      col 2 : reconstructed from main + overlap patches (overlap overwrites)
      col 3 : per-pixel max abs-diff between source and main-reconstruction,
               masked to covered area; uncovered pixels shown as grey
    """
    ax_src, ax_main, ax_all, ax_diff = axes_row

    grid = container.grid
    ox = getattr(container, 'img_origin_x', 0)
    oy = getattr(container, 'img_origin_y', 0)

    # col 0: source + grid overlay
    ax_src.imshow(source_img)
    for info in grid.main_patch_infos:
        lx, ly = info.x - ox, info.y - oy
        ax_src.add_patch(mpatches.Rectangle(
            (lx, ly), size, size,
            fill=False, edgecolor='lime', linewidth=1.0,
        ))
    for info in grid.overlap_patch_infos:
        lx, ly = info.x - ox, info.y - oy
        ax_src.add_patch(mpatches.Rectangle(
            (lx, ly), size, size,
            fill=False, edgecolor='red', linewidth=1.0, linestyle='--',
        ))
    ax_src.set_title(f'{title}\noriginal + grid\n'
                     f'{grid.grid_rows}×{grid.grid_cols} main, '
                     f'{grid.overlap_rows}×{grid.overlap_cols} overlap')
    ax_src.legend(handles=[
        mpatches.Patch(edgecolor='lime', facecolor='none', label='main'),
        mpatches.Patch(edgecolor='red',  facecolor='none', label='overlap'),
    ], loc='upper right', fontsize=6)

    # col 1: reconstruct from main only
    recon_main, cov_main = reconstruct_image(container, main_only=True)
    ax_main.imshow(recon_main)
    pct = cov_main.mean() * 100
    ax_main.set_title(f'Reconstructed (main only)\ncoverage {pct:.1f}%')

    # col 2: reconstruct from main + overlap
    recon_all, cov_all = reconstruct_image(container, main_only=False)
    ax_all.imshow(recon_all)
    pct_all = cov_all.mean() * 100
    ax_all.set_title(f'Reconstructed (main + overlap)\ncoverage {pct_all:.1f}%')

    # col 3: diff (source vs main recon, covered pixels only)
    src_crop = source_img[:container.height, :container.width]
    diff = np.abs(src_crop.astype(np.int16) - recon_main.astype(np.int16)).max(axis=-1)
    # Show diff only where covered; grey elsewhere
    diff_vis = np.full((*diff.shape, 3), 180, dtype=np.uint8)
    diff_vis[cov_main] = np.stack([diff[cov_main]] * 3, axis=-1).clip(0, 255)
    ax_diff.imshow(diff_vis, vmin=0, vmax=20)
    max_d = int(diff[cov_main].max()) if cov_main.any() else 0
    ax_diff.set_title(f'|source − recon| (main)\nmax diff = {max_d} (expect 0)')
    ax_diff.text(source_img.shape[1] // 2, source_img.shape[0] // 2,
                 f'max={max_d}',
                 ha='center', va='center', fontsize=14,
                 color='lime' if max_d == 0 else 'red')

    for ax in axes_row:
        ax.axis('off')


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_rects(ax, origins, size, color, lw=1.2, linestyle='-'):
    for y, x in origins:
        ax.add_patch(mpatches.Rectangle(
            (x, y), size, size,
            fill=False, edgecolor=color, linewidth=lw, linestyle=linestyle,
        ))


def draw_region_bbox(ax, region: TissueRegion, ds: float, color='yellow'):
    rx, ry = int(region.x / ds), int(region.y / ds)
    rw, rh = int(region.w / ds), int(region.h / ds)
    ax.add_patch(mpatches.Rectangle(
        (rx, ry), rw, rh, fill=False, edgecolor=color, linewidth=2,
    ))


def show_patch_grid(ax, patches, n_cols: int = 4, title: str = ''):
    n = len(patches)
    if n == 0:
        ax.set_title(title + '\n(no patches)')
        ax.axis('off')
        return
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols
    s = patches[0].shape[0]
    canvas = np.ones((n_rows * s, n_cols * s, 3), dtype=np.uint8) * 220
    for idx, p in enumerate(patches[:n_cols * n_rows]):
        r, c = divmod(idx, n_cols)
        canvas[r*s:(r+1)*s, c*s:(c+1)*s] = p
    ax.imshow(canvas)
    ax.set_title(title)
    ax.axis('off')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_patchgrid_section(tile: int, out_dir: str) -> None:
    print('\n=== PatchGrid ===')
    results = run_all_patchgrid(tile)
    diagram_grid = PatchGrid.from_size(3 * tile, 3 * tile, tile, overlap=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#1a1a2e')
    draw_index_diagram(axes[0], diagram_grid, tile)
    axes[1].set_facecolor('#1a1a2e')
    axes[1].axis('off')
    headers = ['W', 'H', 'rows', 'cols', 'ovl_r', 'ovl_c', 'len']
    table_data = [
        [str(W), str(H), str(g.grid_rows), str(g.grid_cols),
         str(g.overlap_rows), str(g.overlap_cols), str(len(g))]
        for W, H, g in results
    ]
    tbl = axes[1].table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#223366' if r == 0 else '#1a1a2e')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#444')
    axes[1].set_title('PatchGrid layout summary', color='white', fontsize=10)
    fig.tight_layout()
    out = os.path.join(out_dir, 'patch_grid__index.png')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved {out}')


def run_patchinfo_section(size: int, out_dir: str) -> None:
    print('\n=== PatchInfo / coordinates ===')
    validate_for_query()
    validate_for_wsi()
    validate_to_level0()
    grid, ox, oy, rw, rh = validate_grid_offset(size)
    img, region, ds = validate_grid_offset_pixels(size)
    W, H = 512, 512
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:, :, 0] = np.linspace(30, 200, W, dtype=np.uint8)[None, :]
    bg[:, :, 1] = np.linspace(30, 200, H, dtype=np.uint8)[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(bg)
    rx_n, ry_n = int(region.x / ds), int(region.y / ds)
    rw_n, rh_n = int(region.w / ds), int(region.h / ds)
    axes[0].add_patch(mpatches.Rectangle((rx_n, ry_n), rw_n, rh_n,
                                         fill=False, edgecolor='yellow', linewidth=2))
    tc_vis = TissuePatchContainer(bg.copy(), region=region, img_ds=ds, is_crop=False)
    tc_vis.extract_all(size, overlap=False)
    draw_info_rects(axes[0], tc_vis.grid.main_patch_infos, size, color='cyan')
    axes[0].set_title(f'PatchGrid with offset ({rx_n},{ry_n})\n'
                      f'{len(tc_vis.grid.main_patch_infos)} patches inside region')
    ds_vals = [1.0, 2.0, 4.0]
    colors = ['lime', 'orange', 'red']
    x_before = [50, 50, 50]
    x_after = [50, 100, 200]
    axes[1].set_xlim(0, 300)
    axes[1].set_ylim(-1, len(ds_vals))
    axes[1].set_facecolor('#111111')
    for i, (ds_val, col, xb, xa) in enumerate(zip(ds_vals, colors, x_before, x_after)):
        axes[1].annotate('', xy=(xa, i), xytext=(xb, i),
                         arrowprops=dict(arrowstyle='->', color=col, lw=2))
        axes[1].text(xb - 5, i, f'x={xb}', ha='right', va='center', color='white', fontsize=9)
        axes[1].text(xa + 5, i, f'x0={xa}', ha='left', va='center', color=col, fontsize=9)
        axes[1].text(150, i + 0.3, f'ds={ds_val}', ha='center', color=col, fontsize=8, alpha=0.8)
    axes[1].set_yticks(range(len(ds_vals)))
    axes[1].set_yticklabels([f'ds={d}' for d in ds_vals], color='white')
    axes[1].tick_params(colors='white')
    axes[1].set_title('to_level0: x * ds → level-0 x', color='white')
    axes[1].set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    axes[0].set_facecolor('#1a1a2e')
    axes[0].legend(handles=[
        mpatches.Patch(edgecolor='yellow', facecolor='none', label='region bbox'),
        mpatches.Patch(edgecolor='cyan', facecolor='none', label='patch grid'),
    ], loc='upper left', fontsize=8, facecolor='#333', labelcolor='white')
    axes[0].axis('off')
    axes[1].spines[:].set_color('#444')
    fig.tight_layout()
    out = os.path.join(out_dir, 'patch_info__coords.png')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved {out}')


# ══════════════════════════════════════════════════════════════════════════════
#  4. Scale resolution — which level, and which regions survive it
# ══════════════════════════════════════════════════════════════════════════════
#
# `WsiTissuesContainer.from_ds` answers two questions that used to be answered
# by whoever called it, in three places, differently:
#
#     which level does this ds mean, and what is that level's OWN downsample
#     which regions can host a tile at that downsample
#
# Getting either wrong is quiet. A ds that is 0.04% off the level's real one --
# BRACS_1228 level 1 reports 4.00003 -- turns `int(w / ds)` into one missing
# tile for a region sized near a multiple of 256, and the region then reaches
# the encoder with an empty batch and dies inside torch.cat naming neither the
# region nor the level. Skipping the filter entirely does the same thing.
#
# None of the checks below need a model, and the first three need no WSI at
# all, so the whole section runs in about a second.


def _mask_with_regions(regions_wh, slide_w, slide_h):
    """A mask whose regions are placed by hand, with a token raster.

    NOT test_tissues_regions_mask.make_trm: that derives regions from the
    raster, which would need one mask pixel per level-0 pixel of the slide --
    40 GB for a 200k x 200k canvas. Here only `tissue_regions` is read, so the
    raster exists purely because `regions_view` shares rather than copies it.
    The rebind semantics that makes that safe is tested where it belongs, in
    test_tissues_regions_mask.validate_regions_view_isolation.
    """
    from TissuesRegionsMask import TissuesRegionsMask
    regions, x = [], 0
    for i, (w, h) in enumerate(regions_wh):
        regions.append(TissueRegion(x=x, y=0, w=w, h=h, index=i))
        x += w + 1000
    return TissuesRegionsMask(
        main_mask=np.ones((16, 16), dtype=bool),
        # mask_mpp=0: nothing here converts through mpp, and a non-zero
        # value would have to satisfy wsi_mpp * mask_ds, which for a
        # 16-pixel raster over a whole slide is a meaningless number.
        mask_ds_x=slide_w / 16, mask_ds_y=slide_h / 16, mask_mpp=0.0,
        tissue_regions=regions, wsi_width=slide_w, wsi_height=slide_h,
        wsi_mpp_x=0.25, wsi_mpp_y=0.25,
        wsi_level_downsamples=[1.0, 4.0, 16.0])


def validate_resolve_scale(wsi) -> None:
    """resolve_scale must return a real level and THAT level's own downsample."""
    print('\n[scale] resolve_scale')
    from PatchingLib import WsiTissuesContainer

    for bad in ({}, {'mpp': 0.5, 'ds': 2.0}):
        try:
            WsiTissuesContainer.resolve_scale(wsi, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'resolve_scale({bad}) should have raised')

    downsamples = [float(d) for d in wsi.level_downsamples]
    for level, ds_true in enumerate(downsamples):
        # Ask for something 0.04% off -- the size of the gap that broke
        # filter_patchable against from_mpp -- and demand the level's own value.
        got_level, got_ds = WsiTissuesContainer.resolve_scale(
            wsi, ds=ds_true * 1.0004)
        assert got_level == level, (
            f'ds {ds_true * 1.0004:.5f} resolved to level {got_level}, '
            f'expected {level}')
        assert got_ds == ds_true, (
            f'level {level} reported ds {got_ds!r}, expected the slide\'s own '
            f'{ds_true!r}')
    print(f'  ok   {len(downsamples)} levels, each returns its own downsample')


def validate_ds_gate(wsi) -> None:
    """A ds that is on no level must be refused, level given or not."""
    print('\n[scale] constructor rejects a ds no level has')
    from PatchingLib import WsiTissuesContainer

    downsamples = sorted(float(d) for d in wsi.level_downsamples)
    bogus = (downsamples[0] + downsamples[1]) / 2 if len(downsamples) > 1 else 3.7
    for kwargs in ({'ds': bogus}, {'ds': bogus, 'level': 0}):
        try:
            WsiTissuesContainer(wsi, tile_size=256, overlap=True, **kwargs)
        except ValueError:
            continue
        raise AssertionError(
            f'WsiTissuesContainer({kwargs}) was accepted; ds {bogus} is on no '
            f'level, and with `level` given this used to pass silently because '
            f'the two were only compared when _find_level found something')
    print(f'  ok   ds={bogus:g} refused both with and without an explicit level')


def validate_container_contract(wsi, tile: int, level: int) -> None:
    """What from_ds promises: a real level, its own ds, and usable regions."""
    from PatchingLib import WsiTissuesContainer

    ds = float(wsi.level_downsamples[level])
    width, height = wsi.level_dimensions[0]
    # One region comfortably above the tile, one just below it, one far below.
    big = int(min(width, height, tile * ds * 6))
    mask = _mask_with_regions(
        [(big, big), (int(tile * ds) - 1, int(tile * ds) - 1), (16, 16)],
        slide_w=width, slide_h=height)
    kept_before = len(mask.tissue_regions)

    container = WsiTissuesContainer.from_ds(
        wsi, ds * 1.0004, tile_size=tile, overlap=True, mask=mask)

    assert container.level == level, (
        f'from_ds picked level {container.level}, expected {level}')
    assert container.ds == ds, (
        f'container.ds is {container.ds!r}, not the level\'s own {ds!r}')
    assert len(container.tissue_patches) == len(container.tissue_regions), (
        f'{len(container.tissue_patches)} patch containers against '
        f'{len(container.tissue_regions)} regions -- callers zip these')
    for i, (region, patches) in enumerate(
            zip(container.tissue_regions, container.tissue_patches)):
        assert int(region.w / container.ds) >= tile, (
            f'region {i} is {region.w} level-0 px, which is '
            f'{int(region.w / container.ds)} px at ds {container.ds} -- under '
            f'one {tile} px tile, so it should have been filtered out')
        assert len(patches) > 0, (
            f'region {i} survived the filter and still yielded no patches; '
            f'this is what reaches gigapath_encode as an empty batch')
    assert len(mask.tissue_regions) == kept_before, (
        'from_ds narrowed the caller\'s mask instead of a view of it')
    print(f'  ok   level {container.level}  ds {container.ds:<9.4g}  '
          f'{len(container.tissue_regions)}/{kept_before} regions kept, '
          f'all patchable')


def validate_container_contract_every_level(wsi, tile: int) -> None:
    """The contract at every level the slide has, not at one chosen number.

    It used to take `--openslide-level`, whose default of 9 belongs to
    test_real_wsi_from_openslide and is off the end of a 4-level SVS. Sweeping
    instead removes the argument rather than picking a safer constant: the
    number of levels is a property of the slide, and 2x and 4x pyramids do not
    have the same ones -- BRACS steps 4x per level and Ki67 MRXS 2x, which is
    the difference that made ds_target != ds_actual matter in the first place.

    Bounded whatever the level. The big region is `tile * ds * 6` in level-0
    units, so the container reads `6 * tile` px per side at every level; the
    coarse end is not the expensive end here.
    """
    print(f'\n[scale] from_ds contract, all {wsi.level_count} levels')
    for level in range(wsi.level_count):
        validate_container_contract(wsi, tile, level)


def run_scale_section(args, out_dir: str) -> None:
    # The mask-side half of this -- regions_view isolation and the
    # fine/coarse/fine round trip -- lives in test_tissues_regions_mask.py,
    # since it is filter_patchable's rebind semantics rather than anything
    # about a container.
    print('\n=== Scale resolution ===')
    if not os.path.exists(args.wsi):
        print(f'  [SKIP] WSI not found, container checks skipped: {args.wsi}')
        return
    from SafeSlide import SafeSlide
    wsi = SafeSlide(args.wsi)
    try:
        print(f'  {os.path.basename(args.wsi)}  {wsi.mpp_summary()}')
        validate_resolve_scale(wsi)
        validate_ds_gate(wsi)
        validate_container_contract_every_level(wsi, args.rsize)
    finally:
        wsi.close()


def run_containers_section(args, out_dir: str) -> None:
    print('\n=== QueryPatchContainer / TissuePatchContainer ===')
    size = args.size
    W, H = 512, 512
    img = make_gradient_image(W, H)
    qc = QueryPatchContainer(img.copy())
    qc.extract_all(size, overlap=True)
    validate_qpc_main(qc, size)
    validate_qpc_overlap(qc, size)
    validate_iterators(qc, 'QPC ')
    validate_patch_shapes(qc, size, 'QPC ')
    validate_qpc_no_overlap(img, size)
    validate_qpc_factory_methods(img, size)
    validate_qpc_multichannel(size)
    validate_qpc_errors(img, size)
    region = TissueRegion(x=W // 2, y=0, w=W // 2, h=H, index=0)
    ds = 1.0
    tc1 = TissuePatchContainer(img.copy(), img_ds=ds, is_crop=False)
    tc1.extract_all(size, overlap=True)
    validate_tpc_case1(tc1, img, size)
    validate_iterators(tc1, 'TPC-case1 ')
    validate_patch_shapes(tc1, size, 'TPC-case1 ')
    validate_tpc_overlap_corner(tc1, size, 'TPC-case1 ')
    tc2 = TissuePatchContainer(img.copy(), region=region, img_ds=ds, is_crop=False)
    tc2.extract_all(size, overlap=True)
    case2_patches = validate_tpc_case2(tc2, img, region, size, ds)
    validate_iterators(tc2, 'TPC-case2 ')
    validate_patch_shapes(tc2, size, 'TPC-case2 ')
    validate_tpc_overlap_corner(tc2, size, 'TPC-case2 ')
    rx = int(region.x / ds)
    crop_img = img[:, rx:].copy()
    tc3 = TissuePatchContainer(crop_img, region=region, img_ds=ds, is_crop=True)
    tc3.extract_all(size, overlap=True)
    validate_tpc_case3(tc3, size, case2_patches)
    validate_tpc_case3_overlap(tc2, tc3)
    validate_patch_shapes(tc3, size, 'TPC-case3 ')
    validate_tpc_ds_not_1(size)
    validate_tpc_region_y_offset(img, size)
    validate_tpc_patchinfo_meta(img, size)
    validate_tpc_no_overlap(img, region, ds, size)
    validate_tpc_factory_methods(img, size)
    validate_tpc_errors(img, size)
    qc_sub = validate_container_crop(qc, label='QPC ')
    tc1_sub = validate_container_crop(tc1, label='TPC-case1 ')
    tc2_sub = validate_container_crop(tc2, label='TPC-case2 ')
    tc3_sub = validate_container_crop(tc3, label='TPC-case3 ')
    validate_crop_before_extract()
    validate_tpc_crop_extra_fields(tc2, 'TPC-case2 ')
    validate_tpc_crop_extra_fields(tc3, 'TPC-case3 ')
    rsize = args.rsize
    real_qc = real_roi_qc = real_roi_tc = real_wsi_tc = None
    if args.query and os.path.exists(args.query):
        real_qc = test_real_query(args.query, rsize)
    elif args.query:
        print(f'[SKIP] query not found: {args.query}')
    if args.roi and os.path.exists(args.roi):
        real_roi_qc = test_real_roi_as_query(args.roi, rsize)
        real_roi_tc = test_real_roi(args.roi, rsize)
    elif args.roi:
        print(f'[SKIP] roi not found: {args.roi}')
    if args.wsi and os.path.exists(args.wsi):
        real_wsi_tc = test_real_wsi(args.wsi, args.level, rsize)
        test_real_wsi_from_openslide(args.wsi, args.openslide_level, rsize)
    elif args.wsi:
        print(f'[SKIP] wsi not found: {args.wsi}')
    has_real = any(x is not None for x in [real_qc, real_roi_qc, real_roi_tc, real_wsi_tc])
    nrows = 3 if has_real else 2
    fig, axes = plt.subplots(nrows, 4, figsize=(24, 6 * nrows))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f'QPC original\n{W}x{H}')
    axes[0, 1].imshow(img)
    draw_rects(axes[0, 1], main_origins(W, H, size), size, 'lime')
    axes[0, 1].set_title(f'QPC main grid\n{qc.grid.grid_rows}x{qc.grid.grid_cols} '
                         f'= {len(list(qc.iter_main()))} patches')
    axes[0, 2].imshow(img)
    draw_rects(axes[0, 2], main_origins(W, H, size), size, 'lime')
    draw_rects(axes[0, 2], overlap_origins(W, H, size), size, 'red', lw=1.5, linestyle='--')
    axes[0, 2].set_title(f'QPC +overlap\n+{len(list(qc.iter_overlap()))} corner patches')
    axes[0, 2].legend(handles=[
        mpatches.Patch(edgecolor='lime', facecolor='none', label='main'),
        mpatches.Patch(edgecolor='red', facecolor='none', label='overlap'),
    ], loc='upper right', fontsize=7)
    show_patch_grid(axes[0, 3], list(qc.iter_main())[:8], n_cols=4,
                    title=f'QPC first 8 main patches (size={size})')
    axes[1, 0].imshow(img)
    draw_rects(axes[1, 0], main_origins(W, H, size), size, 'lime')
    axes[1, 0].set_title(f'TPC case1: full, no region\n{len(list(tc1.iter_main()))} patches')
    axes[1, 1].imshow(img)
    draw_region_bbox(axes[1, 1], region, ds)
    rw_n, rh_n = int(region.w / ds), int(region.h / ds)
    glob_orig = [(y, rx + x) for y, x in main_origins(rw_n, rh_n, size)]
    draw_rects(axes[1, 1], glob_orig, size, 'cyan')
    axes[1, 1].set_title(f'TPC case2: full + region\n{len(case2_patches)} patches')
    axes[1, 1].legend(handles=[
        mpatches.Patch(edgecolor='yellow', facecolor='none', label='region bbox'),
        mpatches.Patch(edgecolor='cyan', facecolor='none', label='region grid'),
    ], loc='upper left', fontsize=7)
    axes[1, 2].imshow(crop_img)
    draw_rects(axes[1, 2], main_origins(rw_n, rh_n, size), size, 'cyan')
    axes[1, 2].set_title('TPC case3: is_crop + region\n(same pixels as case2)')
    diffs = [np.abs(p2.astype(int) - p3.astype(int)).max()
             for p2, p3 in zip(case2_patches, list(tc3.iter_main()))]
    max_diff = max(diffs) if diffs else 0
    axes[1, 3].imshow(np.zeros((size, size, 3), dtype=np.uint8))
    axes[1, 3].text(size // 2, size // 2, f'case2 vs case3\nmax diff={max_diff}\n(expect 0)',
                    ha='center', va='center', fontsize=13,
                    color='lime' if max_diff == 0 else 'red')
    axes[1, 3].set_title('Pixel diff panel')
    if has_real:
        real_items = [
            (real_qc, args.query, 'Real query (QPC)'),
            (real_roi_qc, args.roi, 'RoI as query (QPC)'),
            (real_roi_tc, args.roi, 'Real RoI (TPC)'),
            (real_wsi_tc, args.wsi, 'Real WSI crop (TPC)'),
        ]
        for col, (container, path, label) in enumerate(real_items):
            ax = axes[2, col]
            if container is None:
                ax.axis('off')
                continue
            patches = list(container.iter_main())
            show_patch_grid(ax, patches[:8], n_cols=4,
                            title=f'{label}\n{os.path.basename(path or "")}\n'
                                  f'{container.width}x{container.height} '
                                  f'→ {len(patches)} main / '
                                  f'{len(list(container.iter_overlap()))} ovl')
    for row in axes:
        for ax in row:
            ax.axis('off')
    fig.tight_layout()
    out = os.path.join(out_dir, 'patch_container__grid.png')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
    recon_cases = [
        (qc, img, 'QPC synthetic'),
        (tc1, img, 'TPC case1 (full, no region)'),
        (tc2, img, 'TPC case2 (full + region)'),
        (tc3, crop_img, 'TPC case3 (is_crop + region)'),
    ]
    for sub, title in [(qc_sub, 'QPC crop (2x2 grid)'), (tc2_sub, 'TPC-case2 crop (2x2 grid)')]:
        if sub is not None:
            recon_cases.append((sub, sub.img, title))
    for container, _, lbl in [
        (real_qc, None, 'Real query (QPC)'),
        (real_roi_qc, None, 'RoI as query (QPC)'),
        (real_roi_tc, None, 'Real RoI (TPC)'),
        (real_wsi_tc, None, 'Real WSI crop (TPC)'),
    ]:
        if container is not None:
            recon_cases.append((container, container.img, lbl))
    fig2, axes2 = plt.subplots(len(recon_cases), 4, figsize=(24, 6 * len(recon_cases)))
    if len(recon_cases) == 1:
        axes2 = axes2[np.newaxis, :]
    for row_axes, (container, src, title) in zip(axes2, recon_cases):
        draw_reconstruction_row(row_axes, container, src, container.grid.tile_size, title)
    fig2.suptitle('Patch reconstruction comparison', fontsize=11)
    fig2.tight_layout()
    out2 = os.path.join(out_dir, 'patch_container__reconstruction.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f'Saved {out2}')


def main() -> int:
    ap = argparse.ArgumentParser(description='PatchingLib comprehensive tests')
    ap.add_argument('--only', nargs='+',
                    choices=['grid', 'coords', 'containers', 'scale'],
                    default=['grid', 'coords', 'containers', 'scale'],
                    help='which sections to run (default: all)')
    ap.add_argument('--size', type=int, default=128, help='tile size (synthetic + coords)')
    ap.add_argument('--tile', type=int, default=None, help='PatchGrid tile size (default: --size)')
    ap.add_argument('--rsize', type=int, default=256, help='tile size for real-data container tests')
    ap.add_argument('--query', default='/work/u26130998/datasets/Ki67/S1103037_ki67/2.bmp')
    ap.add_argument('--roi',
                    default='/work/u26130998/datasets/histoimage.na.icar.cnr.it/'
                            'BRACS_RoI/latest_version/test/0_N/BRACS_264_N_5.png')
    ap.add_argument('--wsi',
                    default='/work/u26130998/datasets/Ki67/S1103037,G7E,110122.mrxs')
    ap.add_argument('--level', type=int, default=2)
    ap.add_argument('--openslide-level', type=int, default=9)
    ap.add_argument('--out-dir', default=None, help='figure output directory')
    args = ap.parse_args()
    tile = args.tile if args.tile is not None else args.size
    out_dir = args.out_dir or job_result_dir('PatchingLibTest')
    sections = set(args.only)
    if 'grid' in sections:
        run_patchgrid_section(tile, out_dir)
    if 'coords' in sections:
        run_patchinfo_section(args.size, out_dir)
    if 'containers' in sections:
        run_containers_section(args, out_dir)
    if 'scale' in sections:
        run_scale_section(args, out_dir)
    print('\nAll checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
