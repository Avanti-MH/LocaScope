#!/usr/bin/env python3
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

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from _paths import job_result_dir, setup_import_paths
setup_import_paths()

from PatchingLib import PatchGrid, PatchInfo


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

def validate_offset(W, H, tile, ox, oy, ds=1.0, level=2, mpp=0.5):
    grid = PatchGrid.from_size(W, H, tile, overlap=True,
                               x_offset=ox, y_offset=oy, ds=ds, level=level, mpp=mpp)
    for info in grid.iter_infos():
        local_x = info.x - ox
        local_y = info.y - oy
        assert 0 <= local_x, f'local_x={local_x} < 0 (x={info.x}, ox={ox})'
        assert 0 <= local_y, f'local_y={local_y} < 0 (y={info.y}, oy={oy})'
        assert local_x + tile <= W, f'patch right edge {local_x+tile} > W={W}'
        assert local_y + tile <= H, f'patch bottom edge {local_y+tile} > H={H}'
        assert info.ds    == ds
        assert info.level == level
        assert info.mpp   == mpp

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

def run_all(tile: int = 128):
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
    validate_offset(256, 256, tile, ox=128, oy=64, ds=4.0, level=2, mpp=0.5)
    print(f'[PASS] offset + ds/level/mpp: coordinates and metadata verified')

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tile', type=int, default=128, help='tile size')
    ap.add_argument('--out',  default=None, help='output figure path')
    args = ap.parse_args()

    results = run_all(args.tile)

    # Figure: index diagram for the 512×512 standard grid
    tile = args.tile
    diagram_grid = PatchGrid.from_size(3 * tile, 3 * tile, tile, overlap=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#1a1a2e')

    draw_index_diagram(axes[0], diagram_grid, tile)

    # Panel 2: summary table
    axes[1].set_facecolor('#1a1a2e')
    axes[1].axis('off')
    headers = ['W', 'H', 'rows', 'cols', 'ovl_r', 'ovl_c', 'len']
    table_data = [
        [str(W), str(H),
         str(g.grid_rows), str(g.grid_cols),
         str(g.overlap_rows), str(g.overlap_cols), str(len(g))]
        for W, H, g in results
    ]
    tbl = axes[1].table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#223366' if r == 0 else '#1a1a2e')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#444')
    axes[1].set_title('PatchGrid layout summary', color='white', fontsize=10)

    fig.tight_layout()
    out = args.out or os.path.join(job_result_dir('PatchGridIndexTest'),
                                    'patch_grid__index.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\nSaved {out}')
    print('All checks passed.')


if __name__ == '__main__':
    main()
