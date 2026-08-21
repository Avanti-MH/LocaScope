#!/usr/bin/env python3
"""Exhaustively find every block a WSI cannot return, per slide and per level.

A MIRAX scanner photographs only the grid cells its pre-scan flagged, so a read
that touches a skipped cell fails with

    OpenSlideError: Not a JPEG file: starts with 0x00 0x00

and openslide.bounds-* does not protect you: it is the outer envelope of the
photographed cells, and the inside of that rectangle has gaps. Coverage also
differs per pyramid level -- the levels are stored separately, so a coarse level
is not a downsample of a complete level 0 and can be MORE holed, not less.

This tiles the scanned rectangle COMPLETELY -- every pixel belongs to exactly
one block, no sampling -- reads each block in full, and reports which fail.
`broken / total` is then a real areal fraction, not an estimate.

Two things make the numbers mean something:

  * A block is read in FULL, so it is broken exactly when a read of that block
    would fail. That is the question the pipeline asks: WsiTissuesContainer
    reads a whole region in one call, and one bad tile inside takes the lot.

  * --block is in LEVEL-0 pixels, so every level is cut on the same grid over
    the same physical area. Maps line up across a row and the percentages are
    comparable; a block in level pixels would make each level cover a different
    footprint and the comparison would be meaningless.

The block-size sweep costs no extra reads: a coarse block is readable exactly
when every fine block inside it is, so coarser results are derived by pooling.

CRITICAL, and the reason a naive version of this reports nonsense: an OpenSlide
handle that has raised once is dead. openslide checks its error state on every
call, so after the first bad read every later call raises the same error -- even
level_count. A scan sharing one handle reports everything after the first hole
as broken. SlideProbe replaces the handle on each failure, which is also why
`reopens` equals the number of broken blocks.

Usage:
    python utilities/cli/scan_wsi_holes.py \\
        "/work/u26130998/datasets/Ki67/S1103037,G7E,110122.mrxs" \\
        "/work/u26130998/datasets/Ki67/S1104360,G7E,110208.mrxs" \\
        --levels 0,1,2,3,4 --block 4096 --out result/WsiHoles

Figure: one ROW per slide, one COLUMN per level.

Outputs, in --out:
    holes_grid.png    the slide x level map grid
    holes_sweep.png   damage vs block size, one line per (slide, level)
    holes.csv         one row per broken block: slide, level, level-0 x/y/w/h
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))
from _paths import job_result_dir                                   # noqa: E402

import numpy as np
import openslide
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SlideProbe:
    """An OpenSlide handle that survives a failed read by replacing itself."""

    def __init__(self, path: str):
        self.path = path
        self.osl = openslide.OpenSlide(path)
        self.reopens = 0

    def close(self):
        try:
            self.osl.close()
        except Exception:
            pass

    def readable(self, x: int, y: int, level: int, size: tuple) -> bool:
        try:
            self.osl.read_region((x, y), level, size)
            return True
        except Exception:
            self.close()
            self.osl = openslide.OpenSlide(self.path)
            self.reopens += 1
            return False


def scan(probe: SlideProbe, x0: int, y0: int, w: int, h: int,
         level: int, ds: float, block_l0: int,
         progress: bool = False) -> np.ndarray:
    """Tile a level-0 rect on a block_l0 grid, reading each block at `level`.

    Returns (ny, nx) bool, True = the whole block read. Cells cover the rect
    exactly once, so ok.mean() is an areal fraction. The grid does not depend on
    `level`, only the read size does, which is what makes levels comparable.
    """
    nx = max(1, math.ceil(w / block_l0))
    ny = max(1, math.ceil(h / block_l0))
    ok = np.zeros((ny, nx), dtype=bool)

    t0 = time.time()
    for i in range(ny):
        for j in range(nx):
            px, py = x0 + j * block_l0, y0 + i * block_l0
            bw = max(1, int(min(block_l0, x0 + w - px) / ds))
            bh = max(1, int(min(block_l0, y0 + h - py) / ds))
            ok[i, j] = probe.readable(px, py, level, (bw, bh))
        if progress and (i % 10 == 0 or i == ny - 1):
            print(f'    row {i+1:4d}/{ny}  broken {int((~ok[:i+1]).sum()):5d}'
                  f'  {time.time()-t0:5.0f}s', flush=True)
    return ok


def pool_broken(ok: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen by `factor`: a coarse block is readable iff every fine one is.

    Exact, not an approximation -- reading a coarse block touches precisely the
    tiles its fine blocks touch. Padded with True so partial edge cells do not
    invent breakage.
    """
    ny, nx = ok.shape
    py, px = (-ny) % factor, (-nx) % factor
    padded = np.pad(ok, ((0, py), (0, px)), constant_values=True)
    return padded.reshape(padded.shape[0] // factor, factor,
                          padded.shape[1] // factor, factor).all(axis=(1, 3))


def slide_rect(path: str, whole_canvas: bool) -> tuple:
    """(x, y, w, h, mpp, n_levels, scope) for the area worth scanning."""
    wsi = openslide.OpenSlide(path)
    W0, H0 = wsi.dimensions
    p = wsi.properties
    # mpp-x alone, NOT SafeSlide.base_mpp: this tool opens raw on purpose --
    # it hunts holes, and SafeSlide exists to survive them -- and it wants nan
    # rather than a raise when the slide carries no mpp.
    mpp = float(p.get('openslide.mpp-x', 0)) or float('nan')
    n_levels = wsi.level_count
    dss = [float(d) for d in wsi.level_downsamples]
    if whole_canvas or p.get('openslide.bounds-width') is None:
        rect, scope = (0, 0, W0, H0), 'canvas'
    else:
        rect = (int(p['openslide.bounds-x']), int(p['openslide.bounds-y']),
                int(p['openslide.bounds-width']), int(p['openslide.bounds-height']))
        scope = 'bounds'
    wsi.close()
    return rect, mpp, n_levels, dss, scope, (W0, H0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wsi', nargs='+', help='one or more slides; one row each')
    ap.add_argument('--levels', default='0,1,2,3',
                    help='comma-separated pyramid levels; one column each')
    ap.add_argument('--block', type=int, default=4096,
                    help='block side in LEVEL-0 pixels, same grid at every level')
    ap.add_argument('--whole-canvas', action='store_true',
                    help='scan the full canvas instead of openslide.bounds-*')
    ap.add_argument('--sweep', type=int, default=4,
                    help='how many doublings of --block to report')
    ap.add_argument('--out', default='',
                    help='output directory. Empty means result/<SLURM_JOB_NAME or WsiHoles>/, via _paths.job_result_dir -- results live outside the checkout')
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    # job_result_dir honours SLURM_JOB_NAME, so a job's output lands under
    # result/<job>/ without the jobscript spelling the path twice. The makedirs
    # is for the other branch: `or` short-circuits, so an explicit --out never
    # reaches job_result_dir and nothing else would create that directory.
    args.out = args.out or job_result_dir('WsiHoles')
    os.makedirs(args.out, exist_ok=True)
    levels = [int(v) for v in args.levels.split(',') if v.strip()]

    results = {}       # (slide_tag, level) -> dict
    csv_rows = []

    for path in args.wsi:
        tag = os.path.splitext(os.path.basename(path))[0]
        (bx, by, bw, bh), mpp, n_levels, dss, scope, canvas = \
            slide_rect(path, args.whole_canvas)
        nx = math.ceil(bw / args.block)
        ny = math.ceil(bh / args.block)

        print(f'\n{tag}')
        print(f'  canvas {canvas[0]} x {canvas[1]}   '
              f'{scope} {bw} x {bh} at ({bx}, {by})')
        print(f'  grid {ny} x {nx} = {ny*nx} blocks of {args.block} lv0 px '
              f'({args.block*mpp/1000:.2f} mm)')

        for lv in levels:
            if lv >= n_levels:
                print(f'  [SKIP] level {lv}: slide has {n_levels} levels')
                continue
            ds = dss[lv]
            probe = SlideProbe(path)
            t0 = time.time()
            ok = scan(probe, bx, by, bw, bh, lv, ds, args.block)
            dt = time.time() - t0
            n_bad = int((~ok).sum())
            print(f'  level {lv} (ds={ds:6.1f})  broken {n_bad:5d}/{ok.size:<5d} '
                  f'= {100.0*n_bad/ok.size:6.2f}%   reopens={probe.reopens:<5d} '
                  f'{dt:5.0f}s', flush=True)
            probe.close()

            results[(tag, lv)] = dict(ok=ok, bad=n_bad, ds=ds, mpp=mpp,
                                      rect=(bx, by, bw, bh))
            for i, j in zip(*np.nonzero(~ok)):
                px, py = bx + int(j) * args.block, by + int(i) * args.block
                csv_rows.append([tag, lv, int(i), int(j), px, py,
                                 min(args.block, bx + bw - px),
                                 min(args.block, by + bh - py),
                                 round(px * mpp / 1000.0, 3),
                                 round(py * mpp / 1000.0, 3)])

    if not results:
        print('nothing scanned')
        return

    csv_path = os.path.join(args.out, 'holes.csv')
    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['slide', 'level', 'row', 'col',
                     'x_l0', 'y_l0', 'w_l0', 'h_l0', 'x_mm', 'y_mm'])
        wr.writerows(csv_rows)

    tags = []
    for path in args.wsi:
        t = os.path.splitext(os.path.basename(path))[0]
        if t not in tags and any(k[0] == t for k in results):
            tags.append(t)
    used_levels = [lv for lv in levels if any(k[1] == lv for k in results)]

    # ── grid: one row per slide, one column per level ────────────────────────
    nr, nc = len(tags), len(used_levels)
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 7.5 * nr),
                             squeeze=False)
    for r, tag in enumerate(tags):
        for c, lv in enumerate(used_levels):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            res = results.get((tag, lv))
            if res is None:
                ax.axis('off')
                ax.set_title(f'L{lv}\n(no such level)', fontsize=8)
                continue
            _, _, w, h = res['rect']
            mm_w, mm_h = w * res['mpp'] / 1000.0, h * res['mpp'] / 1000.0
            # extent in mm so every level of a slide shares one physical frame
            ax.imshow(res['ok'], cmap='gray', vmin=0, vmax=1,
                      interpolation='nearest', extent=(0, mm_w, mm_h, 0))
            ax.set_aspect('equal')
            ax.set_facecolor('0.15')
            frac = 100.0 * res['bad'] / res['ok'].size
            ax.set_title(f'L{lv}  ds={res["ds"]:g}\n'
                         f'{res["bad"]}/{res["ok"].size} broken  {frac:.2f}%',
                         fontsize=9, color=('crimson' if res['bad'] else 'black'))
            if c == 0:
                ax.set_ylabel(f'{tag}\n{mm_w:.1f} x {mm_h:.1f} mm', fontsize=8)
    fig.suptitle(f'Unreadable blocks, {args.block} level-0 px grid '
                 f'(black = read_region fails)', fontsize=12)
    fig.tight_layout()
    grid_path = os.path.join(args.out, 'holes_grid.png')
    fig.savefig(grid_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    # ── sweep: damage vs block size, derived from the same scans ─────────────
    fig2, ax = plt.subplots(figsize=(9, 6))
    print(f'\n{"slide":<28} {"lv":>3}' +
          ''.join(f'{args.block * 2**k:>10}' for k in range(args.sweep)))
    for (tag, lv), res in sorted(results.items()):
        xs, ys = [], []
        for k in range(args.sweep):
            cur = res['ok'] if k == 0 else pool_broken(res['ok'], 2 ** k)
            xs.append(args.block * 2 ** k)
            ys.append(100.0 * int((~cur).sum()) / cur.size)
        ax.plot(xs, ys, 'o-', label=f'{tag[:18]} L{lv}', alpha=0.85)
        print(f'{tag[:28]:<28} {lv:>3}' + ''.join(f'{v:>9.2f}%' for v in ys))
    ax.set_xscale('log', base=2)
    ax.set_xlabel('block size (level-0 px)')
    ax.set_ylabel('% of scanned area inside a broken block')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Damage vs block size\n'
                 'one bad tile condemns its whole block, so smaller chunks '
                 'recover area\nthis is where a read chunk size comes from',
                 fontsize=10)
    fig2.tight_layout()
    sweep_path = os.path.join(args.out, 'holes_sweep.png')
    fig2.savefig(sweep_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig2)

    print(f'\nSaved {grid_path}')
    print(f'Saved {sweep_path}')
    print(f'Saved {csv_path}')


if __name__ == '__main__':
    main()
