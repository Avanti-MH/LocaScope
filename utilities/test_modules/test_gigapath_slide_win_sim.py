#!/usr/bin/env python3
"""
End-to-end pipeline test:
  1. QueryFromWSI      — crop a query at known (x, y, mpp) from the WSI
  2. estimate_mpp      — estimate the query MPP from GigaPath features
  3. compute_gigapath_sliding_win_similarity — sliding window similarity at estimated MPP
  4. Verify            — best match should be near the ground-truth crop location
  5. GigaPathSlidingWinSimRot — the rotation-aware retriever LocaScopePipeline
                          actually calls, checked against the same crop and a
                          decoy comparison against step 3's non-rotated result

Steps 1-4 are the base module's original test: one query, mpp estimated the way
production would, a visual 8-panel figure, and a soft PASS/WARN print that was
never wired to the exit code. Step 5 is a harder, assertion-based check that
started as a separate script and was folded back in here before it was ever
committed on its own, because running two scripts that each crop the same
query and load the same model was paying GPU + WSI cost twice for one fact.

Why step 5 exists and what it checks
-------------------------------------
LocaScopePipeline (utilities/LocaScopePipeline.py) calls GigaPathSlidingWinSimRot
exclusively -- never the base module steps 1-4 exercise. A regression in
compute_sim_maps, find_best, or the rotation bookkeeping could ship unnoticed
while this file stayed green, because green here only ever meant "the
non-rotated path still works."

Two things are checked, and the second is the reason Rot exists rather than the
base module:

    recovers a rotated query   the SAME crop is fed in at four synthetic
                               "photo" rotations (0/90/180/270, via np.rot90 --
                               the same operation GigaPathSlidingWinSimRot uses
                               internally). find_best must land near the true
                               crop location for all four, not just 0, which is
                               all steps 1-4 ever ran.

    beats the base module on the ones that need rotation   the base module
                               places the query as given and never searches an
                               orientation, so a 90/180/270 photo should
                               mismatch its own un-rotated features and land far
                               from the truth. If the base module recovers it
                               too, Rot is not earning its 4x cost and that is
                               worth knowing, not hiding. Reuses this file's own
                               `_find_best` rather than a second copy of it.

best_rotation is checked against a predicted value, not just its own
consistency: GigaPathSlidingWinSimRot._rotate_np composes as np.rot90(img, k),
and two np.rot90 calls compose by ADDING their k mod 4, so undoing a k-step
photo needs a (4-k)-step search hit -- best_rotation should equal
(360 - applied) % 360. That is discrete array-transpose composition, not a
continuous-angle sign convention, so it does not carry the ambiguity that made
Camera.output_to_level0's R(+/-rot) choice guessable rather than derivable
(test_camera_output_to_level0.py) -- but it is still asserted against a real
run rather than trusted on paper, because a wrong answer here would look
exactly like the same kind of invisible-at-0/180 sign bug.

Step 5 deliberately uses the KNOWN ground-truth mpp, not step 2's estimate:
mpp estimation has its own test (test_gigapath_knn_esti_mpp.py), and conflating
the two here would leave a step-5 failure unable to say which stage broke.
Steps 1-4 keep using the estimate, unchanged, since that is closer to how a
real shot actually reaches the base module's rotation-blind path when nothing
else calls it directly.

Step 5's checks drive the exit code; steps 1-4's PASS/WARN print stays
informational, as it always has been.

Usage:
    python utilities/test_modules/test_gigapath_slide_win_sim.py
    python utilities/test_modules/test_gigapath_slide_win_sim.py \\
        --wsi /path/to/slide.svs --x 10000 --y 20000 --mpp 0.5
"""

import argparse
import os
import math
import sys
import time

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide

import torch
from _paths import job_result_dir, setup_import_paths
setup_import_paths()

from PatchingLib import QueryPatchContainer
from TissuesRegionsMask import TissuesRegionsMask
from QueryFromWSI import QueryFromWSI
from GigaPathKnnEstiMpp import GigaPathKnnEstiMpp
from GigaPathSlidingWinSim import compute_gigapath_sliding_win_similarity
from GigaPathSlidingWinSimRot import GigaPathSlidingWinSimRot
from GigaPathFunc import gigapath_model, gigapath_encode


ROTATIONS = (0, 90, 180, 270)


# ── Canvas builder ────────────────────────────────────────────────────────────

def build_sim_canvas(
    mask: TissuesRegionsMask,
    sim_maps: list,
    ds: float,
    tile_size: int,
    thumb_w: int,
    thumb_h: int,) -> np.ndarray:
    """Paint per-region mean similarity onto a thumbnail-sized canvas.
    sim_maps is list[tuple[main_sim, overlap_sim]]; overlap painted at half-tile offset."""
    canvas = np.full((thumb_h, thumb_w), np.nan)
    cell_l0 = tile_size * ds
    half_l0 = cell_l0 / 2

    for region, (main_sim, overlap_sim) in zip(mask.tissue_regions, sim_maps):
        for hm, x_off, y_off in ((main_sim, 0.0, 0.0), (overlap_sim, half_l0, half_l0)):
            if hm.numel() == 0:
                continue
            heatmap = hm.mean(dim=(-2, -1)).numpy()   # [H_out, W_out]
            H_out, W_out = heatmap.shape
            for r in range(H_out):
                for c in range(W_out):
                    x0 = region.x + c * cell_l0 + x_off
                    y0 = region.y + r * cell_l0 + y_off
                    tx, ty = mask.to_mask_xy(x0, y0)   # position: offset removed
                    tw = max(1, int(cell_l0 / mask.mask_ds_x))   # length: no offset
                    th = max(1, int(cell_l0 / mask.mask_ds_y))
                    if tx < thumb_w and ty < thumb_h:
                        canvas[ty:min(ty+th, thumb_h), tx:min(tx+tw, thumb_w)] = heatmap[r, c]

    return canvas


def _find_best(mask, sim_maps, ds, tile_size, use_overlap: bool):
    """Return (x_l0, y_l0, score) for either the main or overlap grid."""
    best_score = -np.inf
    best_x = best_y = 0
    cell_l0 = tile_size * ds
    half_l0 = cell_l0 / 2

    for region, (main_sim, overlap_sim) in zip(mask.tissue_regions, sim_maps):
        hm = overlap_sim if use_overlap else main_sim
        x_off = half_l0 if use_overlap else 0.0
        y_off = half_l0 if use_overlap else 0.0
        if hm.numel() == 0:
            continue
        heatmap = hm.mean(dim=(-2, -1))
        idx = int(heatmap.argmax())
        r, c = divmod(idx, heatmap.shape[1])
        score = float(heatmap[r, c])
        if score > best_score:
            best_score = score
            best_x = int(region.x + c * cell_l0 + x_off)
            best_y = int(region.y + r * cell_l0 + y_off)

    return best_x, best_y, best_score


# ── step 5: rotation-aware retriever ────────────────────────────────────────

ROT_PASS, ROT_FAIL = [], []


def rot_check(label: str, condition: bool, detail: str = '') -> None:
    tag = 'ok  ' if condition else 'FAIL'
    print(f'  {tag}  {label}' + (f'   {detail}' if detail else ''))
    (ROT_PASS if condition else ROT_FAIL).append(label)


def run_rotation_checks(wsi, mask, query_np, encoder, args, base_mpp) -> None:
    """Step 5. Reuses the already-open wsi/mask/encoder from steps 0-3; crops
    nothing new. Ground-truth mpp, not the step-2 estimate -- see module
    docstring for why the two must not be conflated."""
    print('\n[5] GigaPathSlidingWinSimRot -- rotation-aware retrieval...')
    ds = args.mpp / base_mpp
    tol_um = args.tile * ds * base_mpp

    def dist_um(x0, y0):
        return math.hypot(x0 - args.x, y0 - args.y) * base_mpp

    retriever = GigaPathSlidingWinSimRot(wsi, encoder, mask=mask, mpp=args.mpp,
                                         tile_size=args.tile, overlap=True)
    retriever.build_wsi_features()

    print('  recovers a rotated query')
    for applied_rot in ROTATIONS:
        photo = GigaPathSlidingWinSimRot._rotate_np(query_np, applied_rot)
        retriever.build_query_features(photo)
        retriever.compute_sim_maps()
        result = retriever.find_best()

        err_um = dist_um(result.best_x0, result.best_y0)
        rot_check(f'rot={applied_rot:>3}  position within one tile',
                  err_um <= tol_um, f'err={err_um:.0f}um  tol={tol_um:.0f}um')

        expected_rot = (360 - applied_rot) % 360
        rot_check(f'rot={applied_rot:>3}  best_rotation == {expected_rot}',
                  result.best_rotation == expected_rot,
                  f'got {result.best_rotation}')

    print('  beats the non-rotated module on rotations it cannot search')
    for applied_rot in (90, 180, 270):
        photo = GigaPathSlidingWinSimRot._rotate_np(query_np, applied_rot)
        qc = QueryPatchContainer(photo)
        qc.extract_all(args.tile, overlap=True)
        sim_maps = compute_gigapath_sliding_win_similarity(
            qc, wsi, mpp=args.mpp, tile_size=args.tile, overlap=True,
            mask=mask, encoder=encoder)
        mx, my, m_score = _find_best(mask, sim_maps, ds, args.tile, use_overlap=False)
        ox, oy, o_score = _find_best(mask, sim_maps, ds, args.tile, use_overlap=True)
        bx, by = (ox, oy) if o_score > m_score else (mx, my)
        base_err_um = dist_um(bx, by)
        rot_check(f'rot={applied_rot:>3}  base module misses (Rot exists for this)',
                  base_err_um > tol_um,
                  f'base err={base_err_um:.0f}um  tol={tol_um:.0f}um')

    total = len(ROT_PASS) + len(ROT_FAIL)
    print(f'\n  [5] {len(ROT_PASS)}/{total} passed')
    if ROT_FAIL:
        print(f'      failed: {", ".join(ROT_FAIL)}')


# ── Visualization ─────────────────────────────────────────────────────────────

def draw_figure(thumb, mask, query_img_np, query_qpc,
                sim_maps, ds, tile_size,
                gt_x, gt_y, est_x, est_y, error_um,
                wsi_name, mpp_gt, mpp_est,
                crop_img, crop_x0, crop_y0, crop_ds,
                bm_x, bm_y, bo_x, bo_y, pad,
                out):

    Ht, Wt = mask.main_mask.shape
    sim_canvas = build_sim_canvas(mask, sim_maps, ds, tile_size, Wt, Ht)
    valid = ~np.isnan(sim_canvas)
    vmin = float(np.nanmin(sim_canvas)) if valid.any() else -1.0
    vmax = float(np.nanmax(sim_canvas)) if valid.any() else  1.0

    fig, axes = plt.subplots(2, 4, figsize=(28, 13))

    # [0,0] WSI thumbnail + tissue regions + GT + best match
    ax = axes[0, 0]
    ax.imshow(thumb)
    for r in mask.tissue_regions:
        rx, ry, rw, rh = mask.region_box(r)
        ax.add_patch(mpatches.Rectangle(
            (rx, ry), rw, rh, fill=False, edgecolor='red', linewidth=1.2))
    # Ground truth (cyan)
    gt_tx, gt_ty = mask.to_mask_xy(gt_x, gt_y)
    ax.plot(gt_tx, gt_ty, '+', color='cyan', ms=14, mew=2.5, label='GT')
    # Best match (yellow)
    est_tx, est_ty = mask.to_mask_xy(est_x, est_y)
    ax.plot(est_tx, est_ty, 'x', color='yellow', ms=14, mew=2.5, label='Best match')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f'WSI + GT vs Best match\n{wsi_name}\nerror={error_um:.1f} µm')

    # [0,1] Tissue mask
    ax = axes[0, 1]
    ax.imshow(mask.main_mask, cmap='gray')
    ax.plot(gt_tx, gt_ty, '+', color='cyan', ms=12, mew=2)
    ax.set_title(f'Tissue mask\n{len(mask.tissue_regions)} regions  '
                 f'tissue={mask.tissue_fraction()*100:.1f}%')

    # [0,2] Query image
    ax = axes[0, 2]
    ax.imshow(query_img_np)
    ax.set_title(f'Query image  (GT mpp={mpp_gt:.3f})\n'
                 f'Estimated mpp={mpp_est:.3f}  '
                 f'({query_img_np.shape[1]}×{query_img_np.shape[0]})')

    # [1,0] Query patches
    ax = axes[1, 0]
    s = tile_size
    q_patches = list(query_qpc.iter_main())[:16]
    ncols = min(4, len(q_patches))
    nrows = (len(q_patches) + ncols - 1) // ncols
    canvas_q = np.ones((nrows * s, ncols * s, 3), dtype=np.uint8) * 220
    for idx, p in enumerate(q_patches):
        ri, ci = divmod(idx, ncols)
        canvas_q[ri*s:(ri+1)*s, ci*s:(ci+1)*s] = p
    ax.imshow(canvas_q)
    ax.set_title(f'Query patches (first {len(q_patches)})\n'
                 f'{query_qpc.grid.grid_rows}×{query_qpc.grid.grid_cols} main  tile={s}')

    # [1,1] Similarity heatmap
    ax = axes[1, 1]
    ax.imshow(thumb, alpha=0.45)
    hmap_rgba = cm.hot((sim_canvas - vmin) / max(vmax - vmin, 1e-6))
    hmap_rgba[..., 3] = np.where(valid, 0.75, 0.0)
    ax.imshow(hmap_rgba)
    ax.plot(gt_tx,  gt_ty,  '+', color='cyan',   ms=14, mew=2.5, label='GT')
    ax.plot(est_tx, est_ty, 'x', color='yellow', ms=14, mew=2.5, label='Best match')
    ax.legend(fontsize=8, loc='upper right')
    sm = plt.cm.ScalarMappable(cmap='hot',
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='mean cos-sim')
    n_main = sum(m.shape[0] * m.shape[1] for m, _ in sim_maps if m.numel() > 0)
    n_ov   = sum(o.shape[0] * o.shape[1] for _, o in sim_maps if o.numel() > 0)
    ax.set_title(f'Similarity heatmap  mpp_est={mpp_est:.3f}\n'
                 f'{n_main} main + {n_ov} overlap windows  range=[{vmin:.3f}, {vmax:.3f}]')

    # [1,2] Full query image with tile grid overlay
    ax = axes[1, 2]
    ax.imshow(query_img_np)
    H_q, W_q = query_img_np.shape[:2]
    for i in range(0, W_q + 1, tile_size):
        ax.axvline(i, color='white', lw=0.5, alpha=0.5)
    for i in range(0, H_q + 1, tile_size):
        ax.axhline(i, color='white', lw=0.5, alpha=0.5)
    ax.set_title(f'Query image  ({W_q}×{H_q})\n'
                 f'{query_qpc.grid.grid_rows}×{query_qpc.grid.grid_cols} tiles  tile={tile_size}px')

    # [0,3] hidden
    axes[0, 3].axis('off')

    # [1,3] Zoomed crop: GT area ± pad tiles with grid lines + colored boxes
    ax = axes[1, 3]
    ax.imshow(crop_img)
    R_q = query_qpc.grid.grid_rows
    C_q = query_qpc.grid.grid_cols
    tile_px = tile_size * ds / crop_ds        # one tile in crop-image pixels

    # grid lines: main grid anchored at bm_x/bm_y; overlap shifted by half tile
    bm_px = (bm_x - crop_x0) / crop_ds
    bm_py = (bm_y - crop_y0) / crop_ds
    gt_px = (gt_x - crop_x0) / crop_ds
    gt_py = (gt_y - crop_y0) / crop_ds
    crop_h_px, crop_w_px = crop_img.shape[:2]
    half_px = tile_px / 2

    for offset, color, lw, ls in (
        (0,       'white', 0.6, 'solid'),   # main grid
        (half_px, 'wheat', 0.5, 'dashed'),  # overlap grid
    ):
        x0 = (bm_px + offset) % tile_px
        xi = x0
        while xi <= crop_w_px:
            ax.axvline(xi, color=color, lw=lw, alpha=0.55, linestyle=ls)
            xi += tile_px
        y0 = (bm_py + offset) % tile_px
        yi = y0
        while yi <= crop_h_px:
            ax.axhline(yi, color=color, lw=lw, alpha=0.55, linestyle=ls)
            yi += tile_px

    box_w = C_q * tile_px
    box_h = R_q * tile_px

    # Main best box (yellow)
    ax.add_patch(mpatches.Rectangle(
        (bm_px, bm_py), box_w, box_h,
        fill=False, edgecolor='yellow', linewidth=2.0, label='Main best'))

    # Overlap best box (orange)
    bo_px = (bo_x - crop_x0) / crop_ds
    bo_py = (bo_y - crop_y0) / crop_ds
    ax.add_patch(mpatches.Rectangle(
        (bo_px, bo_py), box_w, box_h,
        fill=False, edgecolor='orange', linewidth=2.0, label='Overlap best'))

    # GT box (cyan)
    ax.add_patch(mpatches.Rectangle(
        (gt_px, gt_py), box_w, box_h,
        fill=False, edgecolor='cyan', linewidth=2.0, label='GT'))

    ax.legend(fontsize=7, loc='upper right', framealpha=0.6)
    ax.set_title(f'Zoomed crop ±{pad} tiles around GT\n'
                 f'cyan=GT  yellow=main  orange=overlap')

    for ax in axes.flat:
        ax.axis('off')

    status = 'PASS' if error_um < 2000 else 'WARN'
    fig.suptitle(
        f'[{status}] LocaScope end-to-end  |  WSI: {wsi_name}\n'
        f'GT=({gt_x},{gt_y})  BestMatch=({est_x},{est_y})  '
        f'error={error_um:.1f} µm  '
        f'mpp_gt={mpp_gt:.3f}  mpp_est={mpp_est:.3f}',
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi',
                    default='/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs')
    ap.add_argument('--x',       type=int,   default=31700,
                    help='Ground-truth crop top-left x in level-0 pixels')
    ap.add_argument('--y',       type=int,   default=33600,
                    help='Ground-truth crop top-left y in level-0 pixels')
    ap.add_argument('--mpp',     type=float, default=0.252,
                    help='Ground-truth query MPP (µm/px)')
    ap.add_argument('--ratio',   type=str,   default='45:32',
                    help='W:H ratio of the query image (e.g. 45:32 for 1440×1024)')
    ap.add_argument('--mpixels', type=float, default=1.475,
                    help='Query size in megapixels (1440×1024 ≈ 1.475 MP)')
    ap.add_argument('--tile',            type=int,   default=256)
    ap.add_argument('--overlap', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--filter', action=argparse.BooleanOptionalAction, default=True,
                    help='apply filter_regions to remove small/contained tissue regions')
    ap.add_argument('--batch',           type=int,   default=1024)
    ap.add_argument('--min-region-ratio',type=float, default=0.10,
                    help='Skip regions smaller than this fraction of the largest region (default 0.10)')
    ap.add_argument('--skip-rot', action='store_true',
                    help='skip step 5 (GigaPathSlidingWinSimRot checks)')
    ap.add_argument('--out',             default=None)
    args = ap.parse_args()

    if not os.path.exists(args.wsi):
        print(f'[SKIP] WSI not found: {args.wsi}')
        return 0

    wsi_name = os.path.basename(args.wsi)
    print(f'WSI : {args.wsi}')
    print(f'GT  : x={args.x}  y={args.y}  mpp={args.mpp}')

    timings: dict[str, float] = {}

    # ── Step 1: Crop query from WSI at known location ─────────────────────────
    print('\n[1] Cropping query from WSI...')
    t0 = time.perf_counter()
    qfwsi = QueryFromWSI(
        args.wsi,
        wh_ratio=args.ratio,
        MPixels=args.mpixels,
        mpp=args.mpp,
    )
    query_pil = qfwsi.crop(args.x, args.y)
    if query_pil is None:
        print('[FAIL] QueryFromWSI.crop returned None')
        return 1
    query_np = np.array(query_pil)
    query_qpc = QueryPatchContainer(query_np)
    query_qpc.extract_all(args.tile, overlap=args.overlap)
    timings['1. crop query'] = time.perf_counter() - t0
    print(f'  Query size: {query_pil.width}×{query_pil.height}  mpp_gt={args.mpp}')
    print(f'  Patches: {query_qpc.grid.grid_rows}×{query_qpc.grid.grid_cols} main')

    if query_qpc.grid.grid_rows == 0 or query_qpc.grid.grid_cols == 0:
        print('[FAIL] Query too small for even one patch — use larger --mpixels or smaller --tile')
        return 1

    wsi      = qfwsi.wsi
    base_mpp = float(wsi.properties.get('openslide.mpp-x', 0))

    # ── Step 0: Build encoder (shared for all stages) ────────────────────────
    print('\n[0] Loading GigaPath model...')
    t0 = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  device={device}')
    _model = gigapath_model(device)
    encoder = lambda patches: gigapath_encode(patches, _model, device, batch_size=args.batch)
    timings['0. load model'] = time.perf_counter() - t0

    # ── Step 2: Estimate MPP ──────────────────────────────────────────────────
    print('\n[2] Estimating MPP...')
    t0 = time.perf_counter()
    est = GigaPathKnnEstiMpp(wsi, encoder=encoder, tile_size=args.tile)
    mpp_result = est.estimate(query_qpc)
    mpp_est = mpp_result.estimated_mpp
    ds_est  = mpp_est / base_mpp
    timings['2. estimate mpp'] = time.perf_counter() - t0
    print(f'  mpp_gt={args.mpp:.4f}  mpp_est={mpp_est:.4f}  '
          f'error={abs(mpp_est - args.mpp) / args.mpp * 100:.1f}%')

    # ── Step 3: Tissue mask ───────────────────────────────────────────────────
    print('\n[3] Building tissue mask...')
    t0 = time.perf_counter()
    mask = TissuesRegionsMask.from_wsi(wsi)
    before = len(mask.tissue_regions)
    if args.filter:
        mask.filter_regions(args.min_region_ratio)
    mask.filter_patchable(tile_size=args.tile, ds=ds_est)
    timings['3. tissue mask'] = time.perf_counter() - t0
    print(f'  {before} regions  tissue={mask.tissue_fraction()*100:.1f}%')
    if len(mask.tissue_regions) < before:
        print(f'  filtered → {len(mask.tissue_regions)} regions '
              f'(removed {before - len(mask.tissue_regions)}'
              + (f': <{args.min_region_ratio*100:.0f}% of largest or contained' if args.filter else '')
              + '  + unpatchable)')
    elif not args.filter:
        print(f'  filter disabled — using all {len(mask.tissue_regions)} regions')

    # ── Step 4: Sliding window similarity ────────────────────────────────────
    print('\n[4] Running compute_gigapath_sliding_win_similarity...')
    t0 = time.perf_counter()
    sim_maps = compute_gigapath_sliding_win_similarity(
        query_qpc, wsi, mpp=mpp_est,
        tile_size=args.tile,
        overlap=args.overlap,
        mask=mask,
        encoder=encoder,
    )
    timings['4. slide win sim'] = time.perf_counter() - t0
    for i, (main_s, ov_s) in enumerate(sim_maps):
        if main_s.numel() == 0:
            print(f'  region {i}: [EMPTY]')
        else:
            hm = main_s.mean(dim=(-2, -1))
            msg = (f'  region {i}: main {tuple(main_s.shape[:2])} windows  '
                   f'mean={hm.mean():.4f}  max={hm.max():.4f}')
            if ov_s.numel() > 0:
                hm_ov = ov_s.mean(dim=(-2, -1))
                msg += (f'  |  overlap mean={hm_ov.mean():.4f}  max={hm_ov.max():.4f}')
            print(msg)

    # ── Step 4b: Verify ───────────────────────────────────────────────────────
    print('\n[4b] Verifying...')
    t0 = time.perf_counter()

    mx, my, m_score = _find_best(mask, sim_maps, ds_est, args.tile, use_overlap=False)
    ox, oy, o_score = _find_best(mask, sim_maps, ds_est, args.tile, use_overlap=True)

    def dist_um(x, y):
        return math.sqrt((x - args.x) ** 2 + (y - args.y) ** 2) * base_mpp

    m_um = dist_um(mx, my)
    o_um = dist_um(ox, oy)
    tol_um = args.tile * ds_est * base_mpp

    has_overlap = o_score > -np.inf
    if has_overlap and o_score > m_score:
        est_x, est_y, best_score, err_um = ox, oy, o_score, o_um
    else:
        est_x, est_y, best_score, err_um = mx, my, m_score, m_um

    timings['4b. verify'] = time.perf_counter() - t0

    print(f'  GT location   : ({args.x}, {args.y})')
    print(f'  Main best     : ({mx}, {my})  score={m_score:.4f}  dist={m_um:.1f} µm')
    if has_overlap:
        tag = '  <- closer' if o_um < m_um else ''
        print(f'  Overlap best  : ({ox}, {oy})  score={o_score:.4f}  dist={o_um:.1f} µm{tag}')
        diff = abs(m_um - o_um)
        winner = 'overlap' if o_um < m_um else ('main' if m_um < o_um else 'tied')
        print(f'  Winner        : {winner}  (diff={diff:.1f} µm)')
    print(f'  Tolerance     : {tol_um:.1f} µm  (one tile at estimated level)')

    status = 'PASS' if err_um <= tol_um else 'WARN'
    print(f'\n  [{status}] best error={err_um:.1f} µm  tol={tol_um:.1f} µm')

    # ── Step 5: GigaPathSlidingWinSimRot ─────────────────────────────────────
    if not args.skip_rot:
        run_rotation_checks(wsi, mask, query_np, encoder, args, base_mpp)

    # ── Figure ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    Ht, Wt = mask.main_mask.shape
    thumb = mask.read_matching_rgb(wsi)

    # zoomed crop around best-match (read before wsi.close)
    pad = 4
    tile_l0 = args.tile * ds_est
    crop_x0 = max(0, int(est_x - pad * tile_l0))
    crop_y0 = max(0, int(est_y - pad * tile_l0))
    R_q = query_qpc.grid.grid_rows
    C_q = query_qpc.grid.grid_cols
    crop_level = wsi.get_best_level_for_downsample(ds_est)
    crop_ds    = wsi.level_downsamples[crop_level]
    crop_w_n   = int((C_q + pad * 2) * args.tile * ds_est / crop_ds)
    crop_h_n   = int((R_q + pad * 2) * args.tile * ds_est / crop_ds)
    crop_img   = np.array(
        wsi.read_region((crop_x0, crop_y0), crop_level, (crop_w_n, crop_h_n)).convert('RGB')
    )

    wsi.close()

    tag = f"{'ov' if args.overlap else 'nov'}_{'flt' if args.filter else 'noflt'}"
    out = args.out or os.path.join(job_result_dir('SlideWinTest'),
                                    f'slide_win_sim__{tag}.png')
    draw_figure(
        thumb, mask, query_np, query_qpc,
        sim_maps, ds_est, args.tile,
        args.x, args.y, est_x, est_y, err_um,
        wsi_name, args.mpp, mpp_est,
        crop_img, crop_x0, crop_y0, crop_ds, mx, my, ox, oy, pad,
        out,
    )
    timings['6. figure'] = time.perf_counter() - t0

    # ── Timing summary ────────────────────────────────────────────────────────
    total = sum(timings.values())
    print('\n' + '─' * 42)
    print(f'  {"Step":<22}  {"Time":>7}  {"% total":>7}')
    print('─' * 42)
    for name, t in timings.items():
        print(f'  {name:<22}  {t:>6.1f}s  {t/total*100:>6.1f}%')
    print('─' * 42)
    print(f'  {"Total":<22}  {total:>6.1f}s')
    print('─' * 42)
    print('\nDone.')

    return 1 if ROT_FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
