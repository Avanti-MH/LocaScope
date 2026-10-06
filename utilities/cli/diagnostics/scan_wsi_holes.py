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
    would fail: one bad tile inside a single read takes the lot.
    SlideReader.read_grid reads a region in blocks of tile rows.

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
    python utilities/cli/diagnostics/scan_wsi_holes.py \\
        --wsi-path "/work/u26130998/datasets/Ki67_with_photo/S1103037_G7E_110122_mrxs/S1103037,G7E,110122.mrxs" \\
        --levels 0,1,2,3,4 --block 4096 --out result/WsiHoles
    python utilities/cli/diagnostics/scan_wsi_holes.py --dataset ki67_pure bracs/test

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
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _paths import job_result_dir                                   # noqa: E402

import numpy as np
import openslide
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from SlideProbe import SlideProbe, bounds_rect                       # noqa: E402
from WsiSelection import resolve_wsi_paths                            # noqa: E402


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
    bx, by, bw, bh, scope = bounds_rect(wsi, whole_canvas)
    wsi.close()
    return (bx, by, bw, bh), mpp, n_levels, dss, scope, (W0, H0)


class WsiHolesScan:
    """`run(entries)` is the shape every diagnostic in this directory shares
    (see `WsiSelection.py`). Batch-native like the original: the PNGs
    compare slides against each other, so they are drawn once per call, not
    once per slide -- `run()` returns one summary row per (slide, level),
    not the per-broken-block detail (still written to `holes.csv` in full,
    same as always)."""

    def __init__(self, levels=(0, 1, 2, 3), block: int = 4096,
                whole_canvas: bool = False, sweep: int = 4, dpi: int = 200,
                figure_slides: str = 'holed', out_dir=None):
        self.levels = list(levels)
        self.block = block
        self.whole_canvas = whole_canvas
        self.sweep = sweep
        self.dpi = dpi
        self.figure_slides = figure_slides
        self.out_dir = out_dir or job_result_dir('WsiHoles')
        os.makedirs(self.out_dir, exist_ok=True)

    def run(self, entries: list) -> list:
        results = {}       # (slide_tag, level) -> dict
        csv_rows = []
        summary = []        # one row per (slide, level) -- this call's return

        for e in entries:
            path = e['path']
            tag = e.get('wsi_name') or os.path.splitext(os.path.basename(path))[0]
            (bx, by, bw, bh), mpp, n_levels, dss, scope, canvas = \
                slide_rect(path, self.whole_canvas)
            nx = math.ceil(bw / self.block)
            ny = math.ceil(bh / self.block)

            print(f'\n{tag}')
            print(f'  canvas {canvas[0]} x {canvas[1]}   '
                  f'{scope} {bw} x {bh} at ({bx}, {by})')
            print(f'  grid {ny} x {nx} = {ny*nx} blocks of {self.block} lv0 px '
                  f'({self.block*mpp/1000:.2f} mm)')

            for lv in self.levels:
                if lv >= n_levels:
                    print(f'  [SKIP] level {lv}: slide has {n_levels} levels')
                    continue
                ds = dss[lv]
                probe = SlideProbe(path)
                t0 = time.time()
                ok = scan(probe, bx, by, bw, bh, lv, ds, self.block)
                dt = time.time() - t0
                n_bad = int((~ok).sum())
                print(f'  level {lv} (ds={ds:6.1f})  broken {n_bad:5d}/{ok.size:<5d} '
                      f'= {100.0*n_bad/ok.size:6.2f}%   reopens={probe.reopens:<5d} '
                      f'{dt:5.0f}s', flush=True)
                probe.close()

                results[(tag, lv)] = dict(ok=ok, bad=n_bad, ds=ds, mpp=mpp,
                                          rect=(bx, by, bw, bh))
                summary.append(dict(dataset=e.get('dataset'), wsi_name=tag,
                                    level=lv, ds=ds, bad=n_bad, total=int(ok.size),
                                    frac=n_bad / ok.size if ok.size else 0.0))
                for i, j in zip(*np.nonzero(~ok)):
                    px, py = bx + int(j) * self.block, by + int(i) * self.block
                    csv_rows.append([tag, lv, int(i), int(j), px, py,
                                     min(self.block, bx + bw - px),
                                     min(self.block, by + bh - py),
                                     round(px * mpp / 1000.0, 3),
                                     round(py * mpp / 1000.0, 3)])

        if not results:
            print('nothing scanned')
            return summary

        csv_path = os.path.join(self.out_dir, 'holes.csv')
        with open(csv_path, 'w', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['slide', 'level', 'row', 'col',
                        'x_l0', 'y_l0', 'w_l0', 'h_l0', 'x_mm', 'y_mm'])
            wr.writerows(csv_rows)

        tags = []
        for e in entries:
            t = e.get('wsi_name') or os.path.splitext(os.path.basename(e['path']))[0]
            if t not in tags and any(k[0] == t for k in results):
                tags.append(t)
        used_levels = [lv for lv in self.levels if any(k[1] == lv for k in results)]

        # ── which slides get DRAWN ───────────────────────────────────────
        # Not all of them. The figure is one row per slide at 7.5 inches;
        # at 145 slides (two whole datasets) that is 1087 inches, which Agg
        # refuses to render. Only the slides that HAVE a broken block are
        # drawn, rather than merely shrinking the rows, because 137 all-white
        # panels are not the information anyone opens this file for.
        # `--figure-slides all` is there for when the question really is
        # "show me the clean ones too".
        if self.figure_slides == 'none':
            print('\nfigures: skipped (--figure-slides none). holes.csv has '
                  'every broken block.')
            print(f'\nSaved {csv_path}')
            return summary

        holed = {tag for (tag, _), res in results.items() if res['bad']}
        draw_tags = tags if self.figure_slides == 'all' else [
            t for t in tags if t in holed]
        skipped = len(tags) - len(draw_tags)
        if skipped:
            print(f'\nfigures: {len(draw_tags)} slide(s) with holes drawn, '
                  f'{skipped} clean slide(s) omitted '
                  f'(--figure-slides all to include them)')
        if not draw_tags:
            print('\nfigures: no slide has a broken block -- nothing to draw.')
            print(f'\nSaved {csv_path}')
            return summary

        # Agg's hard limit is 2**16 px per side; leave room for the
        # suptitle and for `bbox_inches='tight'`. Splitting into numbered
        # files rather than shrinking rows to fit: a row squeezed to a
        # tenth of an inch is present but unreadable, which is a worse
        # answer than a second file.
        rows_per_fig = max(1, int(62000 / (7.5 * self.dpi)))
        pages = [draw_tags[i:i + rows_per_fig]
                for i in range(0, len(draw_tags), rows_per_fig)]

        grid_paths = []
        for page_no, page_tags in enumerate(pages, 1):
            grid_paths.append(self._draw_grid(page_tags, used_levels, results,
                                              page_no, len(pages)))

        self._draw_sweep(draw_tags, results)
        print(f'\nSaved {csv_path}')
        for gp in grid_paths:
            print(f'Saved {gp}')
        print(f'Saved {os.path.join(self.out_dir, "holes_sweep.png")}')
        return summary

    def _draw_grid(self, tags, used_levels, results, page_no, n_pages):
        '''One page of the per-slide grid: one row per slide, one column per
        level.'''
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
        page = '' if n_pages == 1 else f'   (page {page_no} of {n_pages})'
        fig.suptitle(f'Unreadable blocks, {self.block} level-0 px grid '
                     f'(black = read_region fails){page}', fontsize=12)
        fig.tight_layout()
        stem = 'holes_grid' if n_pages == 1 else f'holes_grid_{page_no}'
        grid_path = os.path.join(self.out_dir, f'{stem}.png')
        fig.savefig(grid_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return grid_path

    def _draw_sweep(self, tags, results):
        '''Damage vs block size, one line per (slide, level) -- restricted
        to the same slides the grid draws.'''
        keep = set(tags)
        fig2, ax = plt.subplots(figsize=(9, 6))
        print(f'\n{"slide":<28} {"lv":>3}' +
              ''.join(f'{self.block * 2**k:>10}' for k in range(self.sweep)))
        for (tag, lv), res in sorted(results.items()):
            if tag not in keep:
                continue
            xs, ys = [], []
            for k in range(self.sweep):
                cur = res['ok'] if k == 0 else pool_broken(res['ok'], 2 ** k)
                xs.append(self.block * 2 ** k)
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
        sweep_path = os.path.join(self.out_dir, 'holes_sweep.png')
        fig2.savefig(sweep_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig2)
        return sweep_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wsi', nargs='*', default=None,
                    help='original positional calling convention: one or '
                         'more explicit paths; combined with --wsi-path/'
                         '--dataset if those are also given')
    ap.add_argument('--dataset', nargs='+', default=None)
    ap.add_argument('--wsi-path', dest='wsi_paths', nargs='+', default=None,
                    help='explicit WSI path(s)')
    ap.add_argument('--val-only', action='store_true')
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
    ap.add_argument('--figure-slides', choices=('holed', 'all', 'none'),
                    default='holed',
                    help="which slides the PNGs draw. 'holed' (default) draws "
                         'only slides with at least one broken block; '
                         "'all' draws every scanned slide, which at 145 "
                         "slides is several pages of mostly white; 'none' "
                         'skips the PNGs entirely, for a run whose answer is '
                         'the CSV. The CSVs always cover every slide '
                         'whichever is chosen')
    args = ap.parse_args()

    wsi_paths = list(args.wsi_paths or [])
    wsi_paths += list(args.wsi or [])
    if not args.dataset and not wsi_paths:
        ap.error('need --dataset and/or --wsi-path (or explicit positional paths)')

    entries = resolve_wsi_paths(dataset=args.dataset, wsi=wsi_paths or None,
                                val_only=args.val_only)
    levels = [int(v) for v in args.levels.split(',') if v.strip()]

    scanner = WsiHolesScan(levels=levels, block=args.block,
                           whole_canvas=args.whole_canvas, sweep=args.sweep,
                           dpi=args.dpi, figure_slides=args.figure_slides,
                           out_dir=args.out or None)
    scanner.run(entries)


if __name__ == '__main__':
    main()
