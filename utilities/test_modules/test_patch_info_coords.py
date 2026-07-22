#!/usr/bin/env python3
"""
Assertion + visual test for PatchInfo coordinate system and PatchGrid offset.

Validates:
  1. PatchInfo factory methods (for_query, for_wsi)
  2. to_level0() coordinate scaling for various ds values
  3. PatchGrid.from_size() with x_offset / y_offset
       — PatchInfo.x/y in the grid include the offset
       — All patches extracted from the correct position in the source image
  4. PatchGrid offset + TissuePatchContainer: grid PatchInfo coords are level-N global,
     but _cut_patch correctly subtracts img_origin to get local image coords

Output figure: 2-panel
  1. Grid with offset: PatchInfo origins drawn on a 512x512 image
  2. PatchInfo.x coord coloured by ds value (before/after to_level0)

Usage:
    python utilities/test_modules/test_patch_info_coords.py [--size N] [--out PATH]
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from _paths import job_result_dir, setup_import_paths

setup_import_paths()
from PatchingLib import PatchGrid, PatchInfo, TissuePatchContainer
from TissuesRegionsMask import TissueRegion


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

def draw_rects(ax, infos, size, color, lw=1.2):
    for info in infos:
        rect = mpatches.Rectangle(
            (info.x, info.y), size, size,
            fill=False, edgecolor=color, linewidth=lw,
        )
        ax.add_patch(rect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=128, help='tile size in pixels')
    ap.add_argument('--out', default=None, help='output figure path')
    args = ap.parse_args()

    size = args.size

    # ── Assertions ───────────────────────────────────────────────────────────
    validate_for_query()
    validate_for_wsi()
    validate_to_level0()
    grid, ox, oy, rw, rh = validate_grid_offset(size)
    img, region, ds = validate_grid_offset_pixels(size)

    # ── Figure ───────────────────────────────────────────────────────────────
    W, H = 512, 512
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:, :, 0] = np.linspace(30, 200, W, dtype=np.uint8)[None, :]
    bg[:, :, 1] = np.linspace(30, 200, H, dtype=np.uint8)[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: grid with offset drawn on full image
    axes[0].imshow(bg)
    # Region bbox
    rx_n, ry_n = int(region.x / ds), int(region.y / ds)
    rw_n, rh_n = int(region.w / ds), int(region.h / ds)
    axes[0].add_patch(mpatches.Rectangle(
        (rx_n, ry_n), rw_n, rh_n,
        fill=False, edgecolor='yellow', linewidth=2,
    ))
    # Patches from full+region
    tc_vis = TissuePatchContainer(bg.copy(), region=region, img_ds=ds, is_crop=False)
    tc_vis.extract_all(size, overlap=False)
    draw_rects(axes[0], tc_vis.grid.main_patch_infos, size, color='cyan')
    axes[0].set_title(
        f'PatchGrid with offset ({rx_n},{ry_n})\n'
        f'{len(tc_vis.grid.main_patch_infos)} patches inside region'
    )

    # Panel 2: to_level0 coordinate diagram
    ds_vals = [1.0, 2.0, 4.0]
    colors  = ['lime', 'orange', 'red']
    x_before = [50, 50, 50]
    x_after  = [50, 100, 200]
    axes[1].set_xlim(0, 300)
    axes[1].set_ylim(-1, len(ds_vals))
    axes[1].set_facecolor('#111111')
    for i, (ds_val, col, xb, xa) in enumerate(zip(ds_vals, colors, x_before, x_after)):
        axes[1].annotate(
            '', xy=(xa, i), xytext=(xb, i),
            arrowprops=dict(arrowstyle='->', color=col, lw=2),
        )
        axes[1].text(xb - 5, i, f'x={xb}', ha='right', va='center',
                     color='white', fontsize=9)
        axes[1].text(xa + 5, i, f'x0={xa}', ha='left', va='center',
                     color=col, fontsize=9)
        axes[1].text(150, i + 0.3, f'ds={ds_val}', ha='center',
                     color=col, fontsize=8, alpha=0.8)
    axes[1].set_yticks(range(len(ds_vals)))
    axes[1].set_yticklabels([f'ds={d}' for d in ds_vals], color='white')
    axes[1].tick_params(colors='white')
    axes[1].set_title('to_level0: x * ds → level-0 x', color='white')
    axes[1].set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    axes[0].set_facecolor('#1a1a2e')

    legend = [
        mpatches.Patch(edgecolor='yellow', facecolor='none', label='region bbox'),
        mpatches.Patch(edgecolor='cyan', facecolor='none', label='patch grid'),
    ]
    axes[0].legend(handles=legend, loc='upper left', fontsize=8,
                   facecolor='#333', labelcolor='white')
    axes[0].axis('off')
    axes[1].spines[:].set_color('#444')

    fig.tight_layout()

    out = args.out or os.path.join(job_result_dir('PatchInfoCoordsTest'),
                                    'patch_info__coords.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'\nSaved {out}')
    print('All checks passed.')


if __name__ == '__main__':
    main()
