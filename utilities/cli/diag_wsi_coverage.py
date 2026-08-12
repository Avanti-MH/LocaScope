#!/usr/bin/env python3
"""Where on the WSI does read_region fail, and does alpha predict it?

A MIRAX scanner pre-scans the slide, then photographs only the grid cells it
judged to hold something. Cells it skipped have no JPEG in the .dat, so

    OpenSlideError: Not a JPEG file: starts with 0x00 0x00

comes back for any level-0 read that touches one. openslide.bounds-* is only
the outer envelope of the photographed cells -- the inside of that rectangle
is ragged and full of gaps. A segmentation model run on read_region(...)
.convert('RGB') sees those gaps as pure black (transparent -> black) and can
call them tissue, which puts tissue_regions over areas that were never
photographed.

Three questions in one pass:

  1. WHERE are the unreadable spots, and which tissue_regions sit on them?
  2. Does the alpha channel at the mask level predict level-0 readability?
  3. Which mask op removes which regions?

On (2) the answer for S1103037 is already in: no. Over 2291 probes alpha and
level-0 readability agreed 51.3% of the time, missing all 17 real holes while
calling 1104 readable points empty. The reason is the opposite of what was
assumed -- alpha at level 4 covers 8.4% of the canvas while bounds covers
15.9%, so the COARSE level is sparser than level 0. `mask &= (alpha > 0)` would
delete about half the real tissue. Validity has to come from attempting the
read, not from alpha. The panel stays because the comparison is worth
re-running per slide before trusting either source.

Layout (2 rows x 5):
  row 1  slide / alpha / mask after [1]->[2] / level-0 probe map
         region colour = MEASURED readability: lime >=90%, orange >=50%, red <50%
  row 2  baseline / [1] filter_regions / [2] merge_overlapping /
         [3] filter_patchable / pipeline [1]->[2]
         each from its own deepcopy of baseline, so a panel shows what THAT
         step does rather than the accumulated effect. Titles carry n0 -> n.

Every panel shares the mask coordinate frame, including the probe map, which is
placed with extent= rather than drawn as a raw nx-by-ny image -- a square grid
over the 1:2.8 bounds rect would stretch the slide sideways and the failures
could not be read against the regions.

Usage:
    python utilities/cli/diag_wsi_coverage.py \\
        "/work/u26130998/datasets/Ki67/S1103037,G7E,110122.mrxs" \\
        --hest --mask-ds 4 --grid 48 --out result/DiagWsiCoverage

Without --hest the regions come from the default HSV threshold, which is enough
to see the coverage map but will not reproduce the pipeline's regions. Match
--mask-ds and --min-region-ratio to LocaScopePipeline, and --patch-ds to the
level it routes to, or the panels describe a mask the pipeline never builds.

Output: <out>/<wsi_tag>_coverage.png  and  <wsi_tag>_regions.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from copy import deepcopy

import numpy as np
import openslide
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
for _d in ('utilities', 'aiNNModel'):
    p = os.path.join(_ROOT, _d)
    if p not in sys.path:
        sys.path.insert(0, p)

from TissuesRegionsMask import TissuesRegionsMask     # noqa: E402


# ── Probing ───────────────────────────────────────────────────────────────────

class SlideProbe:
    """An OpenSlide handle that survives a failed read.

    openslide checks its error state on EVERY call, so one read that lands on a
    tile the scanner never wrote kills the handle permanently -- not just for
    reads but for metadata too: level_count and level_downsamples raise the same
    error afterwards. A probe loop sharing one handle therefore reports every
    point after the first hole as unreadable, and the resulting map measures
    nothing but where the first failure was.

    Replacing the handle after each failure is what makes the map real. Reopening
    re-parses Index.dat, which is small, so the cost scales with the number of
    holes rather than the number of probes.
    """

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


def aspect_grid(w: int, h: int, budget: int) -> tuple:
    """Split `budget` probes into (nx, ny) so each cell is roughly square.

    A square grid over a rect that is not square gives cells as elongated as
    the rect, and drawing that array as an image rescales it to a square --
    the scanned area of a MIRAX slide is about 1:2.8, so a square grid stretches
    it sideways by 2.8x and the map can no longer be read against the slide.
    """
    nx = max(2, int(round((budget * w / max(h, 1)) ** 0.5)))
    ny = max(2, int(round(budget / nx)))
    return nx, ny


def probe_grid(probe: SlideProbe, x0: int, y0: int, w: int, h: int,
               nx: int, ny: int, block: int = 64) -> np.ndarray:
    """Try a level-0 read at nx x ny points over a rect. True = readable.

    Returns an (ny, nx) array, row-major like an image. Each probe is
    deliberately tiny: the cost is openslide decoding whichever JPEG tile the
    point lands in, not the block size. Note this samples a point per cell, not
    the cell area, so it detects holes but does not measure their extent.
    """
    ok = np.zeros((ny, nx), dtype=bool)
    for i in range(ny):
        for j in range(nx):
            ok[i, j] = probe.readable(x0 + w * j // nx, y0 + h * i // ny,
                                      0, (block, block))
    return ok


def region_readable_fraction(probe: SlideProbe, region, budget: int = 256) -> float:
    """Fraction of a region bbox that reads at level 0, over `budget` probes."""
    nx, ny = aspect_grid(region.w, region.h, budget)
    ok = probe_grid(probe, region.x, region.y, region.w, region.h, nx, ny)
    return float(ok.mean())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wsi')
    ap.add_argument('--mask-ds', type=float, default=4.0,
                    help='ds for the mask, matching LocaScopePipeline.mask_ds')
    ap.add_argument('--seg-chunk-px', type=int, default=4_000_000,
                    help='tile-and-stitch budget, only used with --hest')
    ap.add_argument('--hest', action='store_true',
                    help='segment with HEST DeepLabV3 instead of HSV (needs GPU)')
    ap.add_argument('--grid', type=int, default=48,
                    help='slide-wide probe budget is --grid squared, split by '
                         'bounds aspect so cells come out square')
    ap.add_argument('--region-probes', type=int, default=256,
                    help='probes per region, split by bbox aspect (total, not per side)')
    ap.add_argument('--min-region-ratio', type=float, default=0.01,
                    help='filter_regions threshold, as in LocaScopePipeline')
    ap.add_argument('--patch-tile', type=int, default=256,
                    help='filter_patchable tile_size, for the ops panel')
    ap.add_argument('--patch-ds', type=float, default=1.0,
                    help='filter_patchable target level ds (1.0 = level 0)')
    ap.add_argument('--out', default='result/DiagWsiCoverage')
    ap.add_argument('--dpi', type=int, default=400)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.wsi))[0]
    wsi = openslide.OpenSlide(args.wsi)

    W0, H0 = wsi.dimensions
    p = wsi.properties
    bx = int(p.get('openslide.bounds-x', 0))
    by = int(p.get('openslide.bounds-y', 0))
    bw = int(p.get('openslide.bounds-width',  W0))
    bh = int(p.get('openslide.bounds-height', H0))
    mpp = float(p.get('openslide.mpp-x', 0)) or float('nan')

    print(f'{tag}')
    print(f'  canvas  {W0} x {H0}')
    print(f'  bounds  {bw} x {bh} at ({bx}, {by})  '
          f'= {100.0 * bw * bh / (W0 * H0):.1f}% of canvas')

    # ── mask ─────────────────────────────────────────────────────────────────
    method = None
    if args.hest:
        import torch
        from HESTSegFunc import hest_seg_model, make_hest_method
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'  loading HEST on {dev}', flush=True)
        method = make_hest_method(hest_seg_model(dev), dev)

    base = TissuesRegionsMask.from_wsi(
        wsi, ds=args.mask_ds, method=method,
        seg_chunk_px=args.seg_chunk_px if method is not None else None,
    )
    n_base = len(base.tissue_regions)
    print(f'  mask {base.main_mask.shape}  baseline regions = {n_base}')

    # Each mutation on its own deep copy, so a panel shows what THAT step does
    # rather than the accumulated effect of everything before it. Same shape as
    # test_tissues_regions_mask --ops.
    mr, pt, pds = args.min_region_ratio, args.patch_tile, args.patch_ds
    ops = [(f'baseline  ({n_base})', base)]

    t = deepcopy(base); t.filter_regions(min_ratio=mr)
    ops.append((f'[1] filter_regions({mr})  {n_base}->{len(t)}', t))

    t = deepcopy(base); t.merge_overlapping()
    ops.append((f'[2] merge_overlapping  {n_base}->{len(t)}', t))

    t = deepcopy(base); t.filter_patchable(tile_size=pt, ds=pds)
    ops.append((f'[3] filter_patchable({pt},ds={pds:g})  {n_base}->{len(t)}', t))

    # [1] -> [2] is what LocaScopePipeline.build() does. filter_patchable runs
    # later and per level, in _level_mask, so it stays out of the state the
    # coverage panels are drawn against.
    trm = deepcopy(base)
    trm.filter_regions(min_ratio=mr)
    trm.merge_overlapping()
    n_final = len(trm.tissue_regions)
    ops.append((f'pipeline [1]->[2]  {n_base}->{n_final}', trm))

    for label, t in ops:
        print(f'  {label}')
    print(f'  coverage panels use [1]->[2]: {n_final} regions '
          f'(indices renumbered by merge_overlapping)')

    # The level from_wsi actually read, so alpha is sampled the same way.
    lv = wsi.get_best_level_for_downsample(args.mask_ds)
    Wl, Hl = wsi.level_dimensions[lv]
    rgba = np.array(wsi.read_region((0, 0), lv, (Wl, Hl)))
    alpha = rgba[:, :, 3]
    has_data = alpha > 0
    print(f'  level {lv} alpha>0 covers {100.0 * has_data.mean():.1f}% '
          f'of the canvas')

    thumb = rgba[:, :, :3]

    # ── slide-wide probe over the bounds rect ────────────────────────────────
    # Its own handle: probing deliberately triggers failures, and a failed read
    # poisons the handle it ran on. `wsi` above must stay usable for metadata.
    probe = SlideProbe(args.wsi)

    nx, ny = aspect_grid(bw, bh, args.grid * args.grid)
    print(f'  probing {ny} x {nx} points across bounds at level 0 '
          f'(cells ~{bw // nx} x {bh // ny} px) ...', flush=True)
    ok = probe_grid(probe, bx, by, bw, bh, nx, ny)
    print(f'  readable {100.0 * ok.mean():.1f}% of probes inside bounds  '
          f'(handle reopened {probe.reopens}x)')

    # Does alpha at the mask level predict level-0 readability?
    alpha_at_probe = np.zeros_like(ok)
    for i in range(ny):
        for j in range(nx):
            px = bx + bw * j // nx
            py = by + bh * i // ny
            mxx = min(int(px / trm.mask_ds_x), has_data.shape[1] - 1)
            myy = min(int(py / trm.mask_ds_y), has_data.shape[0] - 1)
            alpha_at_probe[i, j] = has_data[myy, mxx]

    agree = (alpha_at_probe == ok).mean()
    a_ok_r_bad = int((alpha_at_probe & ~ok).sum())
    a_bad_r_ok = int((~alpha_at_probe & ok).sum())
    print(f'\n  alpha vs level-0 readability over {ok.size} probes:')
    print(f'    agree                        {100.0 * agree:.1f}%')
    print(f'    alpha says data, read FAILS  {a_ok_r_bad}   <- alpha too optimistic')
    print(f'    alpha says none, read works  {a_bad_r_ok}   <- alpha too pessimistic')
    if a_ok_r_bad == 0:
        print('    => alpha never over-promises: `mask &= (alpha > 0)` is safe')
    else:
        print('    => alpha alone does NOT catch every hole')

    # ── per-region readability ───────────────────────────────────────────────
    print(f'\n  probing each of {n_final} regions '
          f'(~{args.region_probes} points each, aspect-matched) ...', flush=True)
    rows = []
    for r in trm.tissue_regions:
        frac = region_readable_fraction(probe, r, budget=args.region_probes)
        rows.append({
            'index': r.index, 'x': r.x, 'y': r.y, 'w': r.w, 'h': r.h,
            'bbox_area': r.w * r.h,
            'mm_w': r.w * mpp / 1000.0, 'mm_h': r.h * mpp / 1000.0,
            'readable_frac': round(frac, 4),
        })
    rows.sort(key=lambda d: -d['bbox_area'])

    csv_path = os.path.join(args.out, f'{tag}_regions.csv')
    with open(csv_path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    print(f'\n  {"index":>6} {"bbox_area":>12} {"size_mm":>14} {"readable":>9}')
    for d in rows[:15]:
        print(f'  {d["index"]:>6} {d["bbox_area"]:>12.3e} '
              f'{d["mm_w"]:>6.2f}x{d["mm_h"]:<7.2f} '
              f'{100 * d["readable_frac"]:>8.1f}%')
    n_phantom = sum(1 for d in rows if d['readable_frac'] < 0.5)
    print(f'\n  {n_phantom}/{n_final} regions are under 50% readable '
          f'(phantom: segmented where nothing was photographed)')

    # ── figure ───────────────────────────────────────────────────────────────
    # Row 1 tells the coverage story, colour = measured readability.
    # Row 2 tells the ops story, colour = uniform, titles carry the counts.
    fig, axes = plt.subplots(2, 5, figsize=(32, 18))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    dsx, dsy = base.mask_ds_x, base.mask_ds_y
    MH, MW = base.main_mask.shape

    def frame(ax, title):
        ax.set_title(title, fontsize=10)

    def bounds_rect(ax):
        mx, my = base.to_mask_xy(bx, by)
        ax.add_patch(Rectangle(
            (mx, my), bw / dsx, bh / dsy,
            fill=False, edgecolor='cyan', linestyle='--', linewidth=1.6))

    def coverage_boxes(ax, label_bad: bool = True):
        """Regions of the [1]->[2] state, coloured by measured readability."""
        for d in rows:
            f = d['readable_frac']
            c = 'lime' if f >= 0.9 else ('orange' if f >= 0.5 else 'red')
            mx, my = base.to_mask_xy(d['x'], d['y'])
            ax.add_patch(Rectangle(
                (mx, my), d['w'] / dsx, d['h'] / dsy,
                fill=False, edgecolor=c, linewidth=1.4))
            if label_bad and f < 1.0:
                ax.text(mx, my - 4,
                        f'{d["index"]}: {100*f:.0f}%', color=c, fontsize=6)

    def plain_boxes(ax, t):
        """Region bboxes of one ops state, uniform colour + index."""
        for r in t.tissue_regions:
            rx, ry, rw_, rh_ = t.region_box(r)
            ax.add_patch(Rectangle(
                (rx, ry), rw_, rh_,
                fill=False, edgecolor='red', linewidth=1.2))
            ax.text(rx + 2, ry + 8, str(r.index),
                    color='yellow', fontsize=6)

    # ── Row 1: context + coverage ────────────────────────────────────────────
    axes[0, 0].imshow(thumb)
    bounds_rect(axes[0, 0]); plain_boxes(axes[0, 0], base)
    frame(axes[0, 0], f'{tag}\nslide at level {lv} (transparent -> black), '
                      f'{n_base} baseline regions')

    axes[0, 1].imshow(has_data, cmap='gray', vmin=0, vmax=1)
    bounds_rect(axes[0, 1])
    frame(axes[0, 1], f'alpha > 0 at level {lv}\n'
                      f'white = photographed ({100.0*has_data.mean():.1f}%)')

    axes[0, 2].imshow(trm.main_mask, cmap='gray', vmin=0, vmax=1)
    bounds_rect(axes[0, 2]); coverage_boxes(axes[0, 2])
    frame(axes[0, 2], f'{"HEST" if method is not None else "HSV"} mask after '
                      f'[1]->[2], {n_final} regions\n'
                      f'lime >=90% readable, orange >=50%, red <50%')

    # Drawn in MASK coordinates, not as a raw nx-by-ny image: extent pins the
    # probe grid onto the bounds rect and the axis limits match every other
    # panel, so a failed probe sits where it actually is on the slide and can be
    # read against the region boxes on top of it.
    ex0, ey0 = base.to_mask_xy(bx, by)
    ex1, ey1 = base.to_mask_xy(bx + bw, by + bh)
    axes[0, 3].imshow(ok, cmap='gray', vmin=0, vmax=1, interpolation='nearest',
                      extent=(ex0, ex1, ey1, ey0))
    axes[0, 3].set_facecolor('0.15')
    bounds_rect(axes[0, 3]); coverage_boxes(axes[0, 3])
    n_bad = int((~ok).sum())
    frame(axes[0, 3], f'level-0 probe {ny}x{nx} inside bounds\n'
                      f'white = readable ({100.0*ok.mean():.1f}%), '
                      f'{n_bad} failed, cells ~{bw // nx}x{bh // ny} px')

    axes[0, 4].axis('off')

    # ── Row 2: ops stages, each from its own deep copy of baseline ───────────
    for k, (label, t) in enumerate(ops):
        ax = axes[1, k]
        ax.imshow(t.main_mask, cmap='gray', vmin=0, vmax=1)
        bounds_rect(ax); plain_boxes(ax, t)
        frame(ax, f'{label}\ntissue={t.tissue_fraction()*100:.1f}%')

    # Every panel in the same mask frame, so the two rows stack meaningfully.
    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xlim(0, MW)
            ax.set_ylim(MH, 0)
            ax.set_aspect('equal')

    fig.tight_layout()
    fig_path = os.path.join(args.out, f'{tag}_coverage.png')
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    wsi.close()
    probe.close()

    print(f'\nSaved {fig_path}')
    print(f'Saved {csv_path}')


if __name__ == '__main__':
    main()
