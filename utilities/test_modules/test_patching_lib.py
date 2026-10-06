#!/usr/bin/env python3
"""Comprehensive tests for utilities/PatchingLib (PatchGrid, PatchInfo, QueryPatchContainer).

Merged from test_patchgrid_index.py, test_patch_info_coords.py, and
test_tissue_patch_container.py.

Sections:
  1. PatchGrid — layout counts, flat/unified indexing, offset metadata
  2. PatchInfo — for_query/for_wsi, grid offset coordinates
  3. Containers — QueryPatchContainer extraction, synthetic and real data

TissuePatchContainer and WsiTissuesContainer were retired on 2026-10-06; their
tests (cases 1-3, the from_ds scale contract) went with them. A slide's tiles
are read by SlideReader.read_grid, tested in test_slide_reader.

Usage:
  python utilities/test_modules/test_patching_lib.py
  python utilities/test_modules/test_patching_lib.py --only grid coords
  python utilities/test_modules/test_patching_lib.py --only containers --size 64

Outputs (under result/TestPatchingLib/ by default):
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

# _paths holds the one definition of OUTPUT_ROOT for every package, so it lives
# in utilities/ rather than beside this file. That directory goes on sys.path
# here, because setup_import_paths -- which puts the rest there -- is inside it.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image as PILImage

from _paths import job_result_dir, setup_import_paths

setup_import_paths()

from PatchingLib import FeaturesMap, PatchGrid, PatchInfo, QueryPatchContainer
from TissueMask import TissueRegion


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
    # and free to disagree.
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


# ── Figure ────────────────────────────────────────────────────────────────────

def draw_info_rects(ax, infos, size, color, lw=1.2):
    for info in infos:
        rect = mpatches.Rectangle(
            (info.x, info.y), size, size,
            fill=False, edgecolor=color, linewidth=lw,
        )
        ax.add_patch(rect)


# ══════════════════════════════════════════════════════════════════════════════
# 3. QueryPatchContainer
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

# ── Closed-form index sums and one-gather grids, against the old loops ──────

def _old_prefix(grid: PatchGrid, r: int) -> int:
    '''FROZEN: PatchGrid._flat_prefix before 2026-10-06 -- a sum over rows.'''
    return sum(grid._row_scan_width(i) for i in range(r))


def _old_len(grid: PatchGrid) -> int:
    '''FROZEN: PatchGrid.__len__ before 2026-10-06.'''
    if not grid.has_overlap:
        return len(grid.main_patch_infos)
    return sum(grid._row_scan_width(r) for r in range(grid.grid_rows))


def _old_lattice_grid(fm: FeaturesMap, lattice: str):
    '''FROZEN: FeaturesMap.main/overlap_feature_grid before 2026-10-06 -- one
    copy per cell, each through __getitem__.'''
    g = fm.grid
    if lattice == 'main':
        rows, cols, flat = g.grid_rows, g.grid_cols, g.flat_index_for_main
    else:
        rows, cols, flat = g.overlap_rows, g.overlap_cols, g.flat_index_for_overlap
    out = fm.features.new_empty(rows, cols, fm.feat_dim)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = fm[flat(r, c)]
    return out


def validate_closed_form_index(tile: int):
    '''`__len__` and `_flat_prefix` in closed form, and the feature grids as
    one gather, must give exactly what the loops gave: the same counts, and
    bit-identical grids. Shapes cover overlap on and off, one row, one column,
    a non-divisible edge, and no tile at all.'''
    import torch
    shapes = [(4 * tile, 3 * tile), (5 * tile + 7, 2 * tile + 3), (tile, 4 * tile),
              (4 * tile, tile), (tile, tile), (tile - 1, tile), (0, 0),
              (13 * tile, 9 * tile)]
    n = 0
    for W, H in shapes:
        for overlap in (True, False):
            g = PatchGrid.from_size(W, H, tile, overlap=overlap)
            assert len(g) == _old_len(g), (W, H, overlap, len(g), _old_len(g))
            for r in range(g.grid_rows + 1):
                assert g._flat_prefix(r) == _old_prefix(g, r), (W, H, overlap, r)
            fm = FeaturesMap(g, torch.randn(len(g), 5))
            for lattice, new in (('main', fm.main_feature_grid()),
                                 ('offset', fm.overlap_feature_grid())):
                old = _old_lattice_grid(fm, lattice)
                assert new.shape == old.shape and torch.equal(new, old), \
                    (W, H, overlap, lattice, tuple(new.shape), tuple(old.shape))
            n += 1
    print(f'[PASS] closed-form len/prefix and one-gather grids == the old loops '
          f'({n} grids)')


def run_patchgrid_section(tile: int, out_dir: str) -> None:
    print('\n=== PatchGrid ===')
    results = run_all_patchgrid(tile)
    validate_closed_form_index(tile)
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
    grid, ox, oy, rw, rh = validate_grid_offset(size)
    region, ds = TissueRegion(x=128, y=64, w=256, h=384, index=0), 1.0
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
    grid_vis = PatchGrid.for_region(region, ds, size, overlap=False)
    draw_info_rects(axes[0], grid_vis.main_patch_infos, size, color='cyan')
    axes[0].set_title(f'PatchGrid with offset ({rx_n},{ry_n})\n'
                      f'{len(grid_vis.main_patch_infos)} patches inside region')
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
    axes[1].set_title('x * ds (scale only; a read starts at tile_origin_l0)', color='white')
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


def run_containers_section(args, out_dir: str) -> None:
    print('\n=== QueryPatchContainer ===')
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
    rsize = args.rsize
    real_qc = real_roi_qc = None
    if args.query and os.path.exists(args.query):
        real_qc = test_real_query(args.query, rsize)
    elif args.query:
        print(f'[SKIP] query not found: {args.query}')
    if args.roi and os.path.exists(args.roi):
        real_roi_qc = test_real_roi_as_query(args.roi, rsize)
    elif args.roi:
        print(f'[SKIP] roi not found: {args.roi}')
    has_real = real_qc is not None or real_roi_qc is not None
    nrows = 2 if has_real else 1
    fig, axes = plt.subplots(nrows, 4, figsize=(24, 6 * nrows), squeeze=False)
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
    if has_real:
        real_items = [
            (real_qc, args.query, 'Real query (QPC)'),
            (real_roi_qc, args.roi, 'RoI as query (QPC)'),
        ]
        for col, (container, path, label) in enumerate(real_items):
            ax = axes[1, col]
            if container is None:
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
    recon_cases = [(qc, img, 'QPC synthetic')]
    for container, lbl in [(real_qc, 'Real query (QPC)'),
                           (real_roi_qc, 'RoI as query (QPC)')]:
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
                    choices=['grid', 'coords', 'containers'],
                    default=['grid', 'coords', 'containers'],
                    help='which sections to run (default: all)')
    ap.add_argument('--size', type=int, default=128, help='tile size (synthetic + coords)')
    ap.add_argument('--tile', type=int, default=None, help='PatchGrid tile size (default: --size)')
    ap.add_argument('--rsize', type=int, default=256, help='tile size for real-data container tests')
    ap.add_argument('--query', default='/work/u26130998/datasets/Ki67_with_photo/S1103037_G7E_110122_mrxs/S1103037_ki67/2.bmp')
    ap.add_argument('--roi',
                    default='/work/u26130998/datasets/histoimage.na.icar.cnr.it/'
                            'BRACS_RoI/latest_version/test/0_N/BRACS_264_N_5.png')
    ap.add_argument('--out-dir', default=None, help='figure output directory')
    args = ap.parse_args()
    tile = args.tile if args.tile is not None else args.size
    out_dir = args.out_dir or job_result_dir('TestPatchingLib')
    sections = set(args.only)
    if 'grid' in sections:
        run_patchgrid_section(tile, out_dir)
    if 'coords' in sections:
        run_patchinfo_section(args.size, out_dir)
    if 'containers' in sections:
        run_containers_section(args, out_dir)
    print('\nAll checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
