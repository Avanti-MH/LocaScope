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
  row 2  baseline / [1] filtered / [2] merged / [3] patchable /
         pipeline [1]->[2]
         each its own view of the raw baseline, so a panel shows what THAT
         step does rather than the accumulated effect. Titles carry n0 -> n.

Every panel shares the mask coordinate frame, including the probe map, which is
placed with extent= rather than drawn as a raw nx-by-ny image -- a square grid
over the 1:2.8 bounds rect would stretch the slide sideways and the failures
could not be read against the regions.

Usage:
    python utilities/cli/diagnostics/diag_mask_validity.py \\
        --wsi "/work/u26130998/datasets/Ki67_with_photo/S1103037_G7E_110122_mrxs/S1103037,G7E,110122.mrxs" \\
        --seg hest --grid 48 --out result/DiagMaskValidity
    python utilities/cli/diagnostics/diag_mask_validity.py --dataset ki67_pure --seg hsv --grid 24

`--seg` names the recipe (default hest, LocaScopePipeline's own), so the
panels are the regions the pipeline builds. Match --patch-ds to the level it
routes to, or the patchable panel describes a scale the pipeline never uses.

Output, per slide: <out>/<wsi_tag>_coverage.png  and  <wsi_tag>_regions.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import openslide
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
for _d in ('utilities', 'aiNNModel'):
    p = os.path.join(_ROOT, _d)
    if p not in sys.path:
        sys.path.insert(0, p)
from _paths import job_result_dir                                   # noqa: E402

from TissueMask import TissueMask                      # noqa: E402
from TissueMaskConfig import add_mask_args, mask_cfg_from_args  # noqa: E402
from TissueSegFunc import nearest_level                 # noqa: E402
from SlideProbe import SlideProbe, bounds_rect         # noqa: E402
from WsiSelection import resolve_wsi_paths             # noqa: E402


# ── Probing ───────────────────────────────────────────────────────────────────

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


# ── Check ────────────────────────────────────────────────────────────────────

class MaskValidityCheck:
    """`run(entries)` is the shape every diagnostic in this directory shares
    (see `WsiSelection.py`). The segmenter is built ONCE, on the first slide,
    by the MaskMaker this holds -- a model is a real GPU load, and a batch of
    slides shares it the same way `LocaScopePipeline` would.
    """

    def __init__(self, mask_cfg, device=None, grid: int = 48,
                region_probes: int = 256, patch_tile: int = 256,
                patch_ds: float = 1.0, dpi: int = 400, out_dir=None):
        from TissueMaskConfig import MaskMaker                     # noqa: PLC0415
        self.mask_cfg = mask_cfg
        self.masks = MaskMaker(mask_cfg, device=device)
        self.grid = grid
        self.region_probes = region_probes
        self.min_region_ratio = mask_cfg.min_region_ratio
        self.patch_tile = patch_tile
        self.patch_ds = patch_ds
        self.dpi = dpi
        self.out_dir = out_dir or job_result_dir('DiagMaskValidity')
        os.makedirs(self.out_dir, exist_ok=True)


    def run_one(self, entry: dict) -> dict:
        wsi_path = entry['path']
        tag = entry.get('wsi_name') or os.path.splitext(
            os.path.basename(wsi_path))[0]
        wsi = openslide.OpenSlide(wsi_path)

        W0, H0 = wsi.dimensions
        p = wsi.properties
        bx, by, bw, bh, _scope = bounds_rect(wsi)
        # mpp-x alone, NOT SafeSlide.base_mpp -- see SlideProbe.py's own
        # module docstring for why.
        mpp = float(p.get('openslide.mpp-x', 0)) or float('nan')

        print(f'\n{tag}')
        print(f'  canvas  {W0} x {H0}')
        print(f'  bounds  {bw} x {bh} at ({bx}, {by})  '
              f'= {100.0 * bw * bh / (W0 * H0):.1f}% of canvas')

        # ── mask ─────────────────────────────────────────────────────────
        # RAW regions: the panels below show what each step of the recipe's
        # region prep does to them.
        base = TissueMask(wsi, self.masks.slide_mask(wsi)[0])
        n_base = len(base.tissue_regions)
        print(f'  mask {base.main_mask.shape}  baseline regions = {n_base}')

        # Each step on its own view of the raw mask, so a panel shows what THAT
        # step does rather than the accumulated effect of everything before it.
        mr, pt, pds = self.min_region_ratio, self.patch_tile, self.patch_ds
        ops = [(f'baseline  ({n_base})', base)]

        t = base.filtered(mr)
        ops.append((f'[1] filtered({mr})  {n_base}->{len(t)}', t))

        t = base.merged()
        ops.append((f'[2] merged  {n_base}->{len(t)}', t))

        t = base.patchable(pt * pds)
        ops.append((f'[3] patchable({pt},ds={pds:g})  {n_base}->{len(t)}', t))

        # [1] -> [2] is the recipe's region prep, what every caller is handed.
        # `patchable` runs later and per scale, so it stays out of the state
        # the coverage panels are drawn against.
        trm = self.mask_cfg.regions(wsi, base.slide_mask)
        n_final = len(trm.tissue_regions)
        ops.append((f'pipeline [1]->[2]  {n_base}->{n_final}', trm))

        for label, t in ops:
            print(f'  {label}')
        print(f'  coverage panels use [1]->[2]: {n_final} regions '
              f'(indices renumbered by merged)')

        # The level nearest the mask's resolution -- for a plane segmenter the
        # level it read -- so alpha is sampled the same way.
        lv = nearest_level(wsi, base.mask_ds_x)
        Wl, Hl = wsi.level_dimensions[lv]
        rgba = np.array(wsi.read_region((0, 0), lv, (Wl, Hl)))
        alpha = rgba[:, :, 3]
        has_data = alpha > 0
        print(f'  level {lv} alpha>0 covers {100.0 * has_data.mean():.1f}% '
              f'of the canvas')

        thumb = rgba[:, :, :3]

        # ── slide-wide probe over the bounds rect ───────────────────────────
        # Its own handle: probing deliberately triggers failures, and a
        # failed read poisons the handle it ran on. `wsi` above must stay
        # usable for metadata.
        probe = SlideProbe(wsi_path)

        nx, ny = aspect_grid(bw, bh, self.grid * self.grid)
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

        # ── per-region readability ───────────────────────────────────────
        print(f'\n  probing each of {n_final} regions '
              f'(~{self.region_probes} points each, aspect-matched) ...',
              flush=True)
        rows = []
        for r in trm.tissue_regions:
            frac = region_readable_fraction(probe, r, budget=self.region_probes)
            rows.append({
                'index': r.index, 'x': r.x, 'y': r.y, 'w': r.w, 'h': r.h,
                'bbox_area': r.w * r.h,
                'mm_w': r.w * mpp / 1000.0, 'mm_h': r.h * mpp / 1000.0,
                'readable_frac': round(frac, 4),
            })
        rows.sort(key=lambda d: -d['bbox_area'])

        csv_path = os.path.join(self.out_dir, f'{tag}_regions.csv')
        if rows:
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

        fig_path = self._draw_figure(tag, wsi, base, trm, ops, thumb, has_data,
                                     ok, rows, bx, by, bw, bh, lv, n_base, n_final)
        wsi.close()
        probe.close()

        print(f'\nSaved {fig_path}')
        print(f'Saved {csv_path}')

        return dict(dataset=entry.get('dataset'), wsi_name=tag,
                   n_base=n_base, n_final=n_final, n_phantom=n_phantom,
                   alpha_vs_read_agree=round(float(agree), 4),
                   readable_frac_mean=(round(float(np.mean(
                       [d['readable_frac'] for d in rows])), 4) if rows else float('nan')),
                   png=fig_path, csv=csv_path)

    def _draw_figure(self, tag, wsi, base, trm, ops, thumb, has_data, ok, rows,
                     bx, by, bw, bh, lv, n_base, n_final) -> str:
        """Row 1 tells the coverage story, colour = measured readability.
        Row 2 tells the ops story, colour = uniform, titles carry the
        counts."""
        fig, axes = plt.subplots(2, 5, figsize=(32, 18))
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])

        dsx, dsy = base.mask_ds_x, base.mask_ds_y
        MH, MW = base.main_mask.shape

        def frame(ax, title):
            ax.set_title(title, fontsize=10)

        # Named for what it draws, not `bounds_rect` -- that name is the
        # MODULE-LEVEL import from SlideProbe.py used above in `run_one`; a
        # nested `def bounds_rect(...)` here would shadow it for the WHOLE
        # enclosing scope (Python resolves a name as local to a function the
        # moment any `def`/assignment for it appears anywhere in that
        # function, regardless of execution order), which is exactly the
        # `UnboundLocalError` this rename avoids.
        def draw_bounds_box(ax):
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

        # ── Row 1: context + coverage ─────────────────────────────────────
        axes[0, 0].imshow(thumb)
        draw_bounds_box(axes[0, 0]); plain_boxes(axes[0, 0], base)
        frame(axes[0, 0], f'{tag}\nslide at level {lv} (transparent -> black), '
                          f'{n_base} baseline regions')

        axes[0, 1].imshow(has_data, cmap='gray', vmin=0, vmax=1)
        draw_bounds_box(axes[0, 1])
        frame(axes[0, 1], f'alpha > 0 at level {lv}\n'
                          f'white = photographed ({100.0*has_data.mean():.1f}%)')

        axes[0, 2].imshow(trm.main_mask, cmap='gray', vmin=0, vmax=1)
        draw_bounds_box(axes[0, 2]); coverage_boxes(axes[0, 2])
        frame(axes[0, 2], f'{self.mask_cfg.seg_id()} mask '
                          f'after [1]->[2], {n_final} regions\n'
                          f'lime >=90% readable, orange >=50%, red <50%')

        # Drawn in MASK coordinates, not as a raw nx-by-ny image: extent pins
        # the probe grid onto the bounds rect and the axis limits match
        # every other panel, so a failed probe sits where it actually is on
        # the slide and can be read against the region boxes on top of it.
        ex0, ey0 = base.to_mask_xy(bx, by)
        ex1, ey1 = base.to_mask_xy(bx + bw, by + bh)
        ny, nx = ok.shape
        axes[0, 3].imshow(ok, cmap='gray', vmin=0, vmax=1, interpolation='nearest',
                          extent=(ex0, ex1, ey1, ey0))
        axes[0, 3].set_facecolor('0.15')
        draw_bounds_box(axes[0, 3]); coverage_boxes(axes[0, 3])
        n_bad = int((~ok).sum())
        frame(axes[0, 3], f'level-0 probe {ny}x{nx} inside bounds\n'
                          f'white = readable ({100.0*ok.mean():.1f}%), '
                          f'{n_bad} failed, cells ~{bw // nx}x{bh // ny} px')

        axes[0, 4].axis('off')

        # ── Row 2: ops stages, each from its own deep copy of baseline ─────
        for k, (label, t) in enumerate(ops):
            ax = axes[1, k]
            ax.imshow(t.main_mask, cmap='gray', vmin=0, vmax=1)
            draw_bounds_box(ax); plain_boxes(ax, t)
            frame(ax, f'{label}\ntissue={t.tissue_fraction()*100:.1f}%')

        # Every panel in the same mask frame, so the two rows stack
        # meaningfully.
        for ax in axes.ravel():
            if ax.has_data():
                ax.set_xlim(0, MW)
                ax.set_ylim(MH, 0)
                ax.set_aspect('equal')

        fig.tight_layout()
        fig_path = os.path.join(self.out_dir, f'{tag}_coverage.png')
        fig.savefig(fig_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        return fig_path

    def run(self, entries: list) -> list:
        return [self.run_one(e) for e in entries]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wsi', nargs='?', default=None,
                    help='original single-file calling convention; '
                         'combined with --wsi/--dataset if those are also given')
    ap.add_argument('--dataset', nargs='+', default=None)
    ap.add_argument('--wsi-path', dest='wsi_paths', nargs='+', default=None,
                    help='explicit WSI path(s)')
    ap.add_argument('--val-only', action='store_true')
    add_mask_args(ap)
    ap.add_argument('--grid', type=int, default=48,
                    help='slide-wide probe budget is --grid squared, split by '
                         'bounds aspect so cells come out square')
    ap.add_argument('--region-probes', type=int, default=256,
                    help='probes per region, split by bbox aspect (total, not per side)')
    ap.add_argument('--patch-tile', type=int, default=256,
                    help='patchable tile size, for the ops panel')
    ap.add_argument('--patch-ds', type=float, default=1.0,
                    help='patchable target ds (1.0 = level 0)')
    ap.add_argument('--out', default='',
                    help='output directory. Empty means result/<SLURM_JOB_NAME or DiagMaskValidity>/, via _paths.job_result_dir -- results live outside the checkout')
    ap.add_argument('--dpi', type=int, default=400)
    args = ap.parse_args()

    wsi_paths = list(args.wsi_paths or [])
    if args.wsi:
        wsi_paths.append(args.wsi)
    if not args.dataset and not wsi_paths:
        ap.error('need --dataset and/or --wsi-path (or a single positional path)')

    entries = resolve_wsi_paths(dataset=args.dataset, wsi=wsi_paths or None,
                                val_only=args.val_only)
    print(f'{len(entries)} WSI(s)')

    check = MaskValidityCheck(
        mask_cfg_from_args(args), grid=args.grid,
        region_probes=args.region_probes, patch_tile=args.patch_tile,
        patch_ds=args.patch_ds, dpi=args.dpi, out_dir=args.out or None)
    try:
        check.run(entries)
    finally:
        check.masks.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
