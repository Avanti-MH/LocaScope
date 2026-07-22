#!/usr/bin/env python3
"""
End-to-end SIFT+RANSAC localization test:
  1. QueryFromWSI      — crop a query at known (x, y, mpp) from the WSI
  2. estimate_mpp      — estimate the query MPP from GigaPath features
  3. GigaPathSlidingWinSim — sliding window similarity → SlideWinSimResult
  4. SiftRansacLocalizer   — sub-tile refinement → SiftRansacResult
  5. Verify            — compare retrieval error vs SIFT error vs GT

Usage:
    python test_modules/test_sift_ransac.py
    python test_modules/test_sift_ransac.py \\
        --wsi /path/to/slide.svs --x 10000 --y 20000 --mpp 0.5
"""

import argparse
import math
import os
import time

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from _paths import job_result_dir, setup_import_paths
setup_import_paths()

from PatchingLib import QueryPatchContainer
from TissuesRegionsMask import TissuesRegionsMask
from QueryFromWSI import QueryFromWSI
from GigaPathKnnEstiMpp import GigaPathKnnEstiMpp
from GigaPathSlideWinSim import GigaPathSlidingWinSim, SlideWinSimResult
from SIFT_RANSAC import SiftRansacLocalizer, SiftRansacResult
from GigaPathFunc import gigapath_model, gigapath_encode


# ── Visualization ─────────────────────────────────────────────────────────────

def _kp_img(img: np.ndarray, kps, max_kp: int = 300) -> np.ndarray:
    """Draw SIFT keypoints on a copy of img (RGB)."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    drawn = cv2.drawKeypoints(
        bgr, kps[:max_kp], None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)


def _match_img(query_img, query_kps, wsi_crop, crop_kps, matches,
               max_m: int = 60, gap: int = 128) -> np.ndarray:
    """Compose query | gap | wsi_crop with vertical centers aligned; draw match lines."""
    h_q, w_q = query_img.shape[:2]
    h_c, w_c = wsi_crop.shape[:2]
    H = max(h_q, h_c)
    W = w_q + gap + w_c

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    q_y0 = (H - h_q) // 2
    c_y0 = (H - h_c) // 2
    c_x0 = w_q + gap

    canvas[q_y0:q_y0 + h_q, :w_q]          = query_img
    canvas[c_y0:c_y0 + h_c, c_x0:c_x0 + w_c] = wsi_crop

    for m in matches[:max_m]:
        pt_q = query_kps[m.queryIdx].pt
        pt_c = crop_kps[m.trainIdx].pt
        p1 = (int(pt_q[0]),        int(pt_q[1]) + q_y0)
        p2 = (int(pt_c[0]) + c_x0, int(pt_c[1]) + c_y0)
        cv2.line(canvas, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
        cv2.circle(canvas, p2, 3, (0, 255, 0), -1)

    return canvas


def draw_figure(
    thumb, mask,
    query_img, query_kps,
    wsi_crop, crop_kps, good_matches,
    retrieval: SlideWinSimResult,
    sift: SiftRansacResult,
    gt_x, gt_y, base_mpp, tile_size,
    wsi_name, mpp_gt, mpp_est,
    crop_img_12, crop_x0_12, crop_y0_12, crop_ds_12, zoom_pad,
    query_rows, query_cols,
    crop_origin_x: int = 0, crop_origin_y: int = 0,
    h_decomp: bool = True, patch_grid: bool = True, trans_arrow: bool = True,
    out: str = 'out.png',
):
    def dist_um(x, y):
        return math.sqrt((x - gt_x) ** 2 + (y - gt_y) ** 2) * base_mpp

    ret_err = dist_um(retrieval.best_x0, retrieval.best_y0)
    sift_err = dist_um(sift.x0, sift.y0) if sift.success else float('nan')

    fig, axes = plt.subplots(2, 4, figsize=(30, 14))

    # ── [0,0] WSI overview ───────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.imshow(thumb)
    for r in mask.tissue_regions:
        ax.add_patch(mpatches.Rectangle(
            (r.x / mask.mask_ds_x, r.y / mask.mask_ds_y),
            r.w / mask.mask_ds_x, r.h / mask.mask_ds_y,
            fill=False, edgecolor='red', linewidth=1.0,
        ))
    def to_thumb(x0, y0):
        return x0 / mask.mask_ds_x, y0 / mask.mask_ds_y

    gt_tx, gt_ty = to_thumb(gt_x, gt_y)
    ret_tx, ret_ty = to_thumb(retrieval.best_x0, retrieval.best_y0)
    ax.plot(gt_tx,  gt_ty,  '+', color='cyan',   ms=14, mew=2.5, label=f'GT')
    ax.plot(ret_tx, ret_ty, 'x', color='yellow', ms=14, mew=2.5,
            label=f'Retrieval ({ret_err:.0f} µm)')
    if sift.success:
        sift_tx, sift_ty = to_thumb(sift.x0, sift.y0)
        ax.plot(sift_tx, sift_ty, 'D', color='lime', ms=10, mew=2.0,
                label=f'SIFT ({sift_err:.0f} µm)')
    ax.legend(fontsize=7, loc='upper right', framealpha=0.7)
    ax.set_title(f'Overview  {wsi_name}\nmpp_gt={mpp_gt:.3f}  mpp_est={mpp_est:.3f}')

    # ── [0,1] Tissue mask ────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.imshow(mask.main_mask, cmap='gray')
    ax.plot(gt_tx, gt_ty, '+', color='cyan', ms=12, mew=2)
    ax.set_title(f'Tissue mask\n{len(mask.tissue_regions)} regions  '
                 f'tissue={mask.tissue_fraction()*100:.1f}%')

    # ── [0,2] Query image with SIFT keypoints ────────────────────────────────
    ax = axes[0, 2]
    ax.imshow(_kp_img(query_img, query_kps) if query_kps else query_img)
    ax.set_title(f'Query  ({query_img.shape[1]}×{query_img.shape[0]})\n'
                 f'{len(query_kps) if query_kps else 0} SIFT keypoints')

    # ── [0,3] WSI crop with SIFT keypoints ──────────────────────────────────
    ax = axes[0, 3]
    ax.imshow(_kp_img(wsi_crop, crop_kps) if crop_kps else wsi_crop)
    ax.set_title(f'WSI crop (region {sift.region_index})\n'
                 f'{len(crop_kps) if crop_kps else 0} SIFT keypoints')

    # ── [1,0] SIFT matches ───────────────────────────────────────────────────
    ax = axes[1, 0]
    if good_matches and query_kps and crop_kps:
        match_img = _match_img(query_img, query_kps, wsi_crop, crop_kps, good_matches)
        ax.imshow(match_img)
        status = 'SUCCESS' if sift.success else 'FAIL'
        ax.set_title(f'SIFT matches [{status}]\n'
                     f'{sift.match_count} good  {sift.inlier_count} inliers')
    else:
        ax.set_title('No SIFT matches')

    # ── [1,1] Retrieval result stats ─────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis('off')
    lines = [
        f'Retrieval (GigaPath sliding window)',
        f'  best_x0 = {retrieval.best_x0}',
        f'  best_y0 = {retrieval.best_y0}',
        f'  score   = {retrieval.best_score:.4f}',
        f'  overlap = {retrieval.from_overlap}',
        f'  error   = {ret_err:.1f} µm',
        '',
        f'SIFT + RANSAC',
        f'  x0      = {sift.x0}',
        f'  y0      = {sift.y0}',
        f'  success = {sift.success}',
        f'  matches = {sift.match_count}',
        f'  inliers = {sift.inlier_count}',
        f'  error   = {sift_err:.1f} µm' if sift.success else '  error   = N/A',
        '',
        f'GT  x={gt_x}  y={gt_y}',
    ]
    ax.text(0.05, 0.95, '\n'.join(lines),
            transform=ax.transAxes, va='top', fontsize=9,
            fontfamily='monospace')
    ax.set_title('Result summary')

    # ── [1,2] Zoomed WSI crop with tile grids + result boxes ─────────────────
    ax = axes[1, 2]
    ax.imshow(crop_img_12)
    tile_px   = tile_size * retrieval.ds / crop_ds_12
    bm_px     = (retrieval.main_x0    - crop_x0_12) / crop_ds_12
    bm_py     = (retrieval.main_y0    - crop_y0_12) / crop_ds_12
    crop_h_px, crop_w_px = crop_img_12.shape[:2]
    half_px   = tile_px / 2

    for offset, color, lw, ls in (
        (0,       'white', 0.6, 'solid'),
        (half_px, 'wheat', 0.5, 'dashed'),
    ):
        xi = (bm_px + offset) % tile_px
        while xi <= crop_w_px:
            ax.axvline(xi, color=color, lw=lw, alpha=0.55, linestyle=ls)
            xi += tile_px
        yi = (bm_py + offset) % tile_px
        while yi <= crop_h_px:
            ax.axhline(yi, color=color, lw=lw, alpha=0.55, linestyle=ls)
            yi += tile_px

    box_w = query_cols * tile_px
    box_h = query_rows * tile_px

    ax.add_patch(mpatches.Rectangle(
        (bm_px, bm_py), box_w, box_h,
        fill=False, edgecolor='yellow', linewidth=2.0, label='Main best'))
    bo_px = (retrieval.overlap_x0 - crop_x0_12) / crop_ds_12
    bo_py = (retrieval.overlap_y0 - crop_y0_12) / crop_ds_12
    ax.add_patch(mpatches.Rectangle(
        (bo_px, bo_py), box_w, box_h,
        fill=False, edgecolor='orange', linewidth=2.0, label='Overlap best'))
    gt_bpx = (gt_x - crop_x0_12) / crop_ds_12
    gt_bpy = (gt_y - crop_y0_12) / crop_ds_12
    ax.add_patch(mpatches.Rectangle(
        (gt_bpx, gt_bpy), box_w, box_h,
        fill=False, edgecolor='green', linewidth=2.0, linestyle='--', label='GT'))
    if sift.success:
        sx_bpx = (sift.x0 - crop_x0_12) / crop_ds_12
        sy_bpy = (sift.y0 - crop_y0_12) / crop_ds_12
        ax.add_patch(mpatches.Rectangle(
            (sx_bpx, sy_bpy), box_w, box_h,
            fill=False, edgecolor='dodgerblue', linewidth=2.0, label='SIFT'))

    ax.legend(fontsize=7, loc='upper right', framealpha=0.6)
    ax.set_title(f'Zoomed ±{zoom_pad} tiles\n'
                 f'green--=GT  yellow=main  orange=overlap  blue=SIFT')

    # ── [1,3] Homography analysis ─────────────────────────────────────────────
    ax = axes[1, 3]
    fallback_img = wsi_crop if wsi_crop is not None else np.zeros((64, 64, 3), np.uint8)
    ax.imshow(fallback_img)

    if sift.success and sift.H is not None:
        H = sift.H
        h_q, w_q = query_img.shape[:2]

        # ── Query boundary (always shown) ────────────────────────────────────
        corners = np.float32([[0,0],[w_q,0],[w_q,h_q],[0,h_q]]).reshape(-1,1,2)
        mapped  = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        poly    = np.vstack([mapped, mapped[0]])
        ax.plot(poly[:, 0], poly[:, 1], 'lime', linewidth=2.0, label='Query boundary')

        # ── Decompose H ──────────────────────────────────────────────────────
        A  = H[:2, :2]
        sx = float(np.linalg.norm(A[:, 0]))
        sy = float(np.linalg.norm(A[:, 1]))
        theta_deg = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
        dx_c, dy_c = float(H[0, 2]), float(H[1, 2])
        p20, p21   = float(H[2, 0]), float(H[2, 1])

        sift_err_ln = (sift_err / base_mpp / sift.ds) if sift.success else float('nan')

        # ── H decomposition text overlay ─────────────────────────────────────
        if h_decomp:
            txt = (f'θ={theta_deg:+.2f}°\n'
                   f'sx={sx:.4f}  sy={sy:.4f}\n'
                   f'dx={dx_c:+.1f}px  dy={dy_c:+.1f}px\n'
                   f'persp=({p20:.2e},{p21:.2e})\n'
                   f'err={sift_err:.0f}µm  ({sift_err_ln:.1f}px@lvl{sift.level})')
            ax.text(4, 4, txt, color='white', fontsize=6.5, va='top',
                    fontfamily='monospace',
                    bbox=dict(facecolor='black', alpha=0.55, pad=2, boxstyle='round'))

        # ── Patch grid projection ─────────────────────────────────────────────
        if patch_grid:
            for x in range(tile_size, w_q, tile_size):
                pts = np.float32([[[float(x), 0.]], [[float(x), float(h_q)]]])
                p   = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                ax.plot(p[:, 0], p[:, 1], color='yellow', lw=0.6, alpha=0.65)
            for y in range(tile_size, h_q, tile_size):
                pts = np.float32([[[0., float(y)]], [[float(w_q), float(y)]]])
                p   = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                ax.plot(p[:, 0], p[:, 1], color='yellow', lw=0.6, alpha=0.65)

        # ── Translation arrow: retrieval → SIFT ──────────────────────────────
        if trans_arrow:
            ret_cx = float(retrieval.best_x - crop_origin_x)
            ret_cy = float(retrieval.best_y - crop_origin_y)
            tl = cv2.perspectiveTransform(np.float32([[[0., 0.]]]), H)[0, 0]
            ax.annotate('', xy=(float(tl[0]), float(tl[1])),
                        xytext=(ret_cx, ret_cy),
                        arrowprops=dict(arrowstyle='->', color='dodgerblue', lw=1.5))
            ax.plot(ret_cx, ret_cy, 'x', color='yellow', ms=8, mew=2, label='Retrieval')
            ax.plot(tl[0], tl[1], 'D', color='dodgerblue', ms=5, label='SIFT tl')

        ax.legend(fontsize=6.5, loc='lower right', framealpha=0.65)
        ax.set_title(f'Homography  θ={theta_deg:+.1f}°  sx={sx:.3f}  sy={sy:.3f}\n'
                     f'err={sift_err:.0f}µm  ({sift_err_ln:.1f}px@lvl{sift.level})')
    else:
        ax.set_title('Homography failed')

    for ax in axes.flat:
        ax.axis('off')

    sift_tag = f'SIFT={sift_err:.0f}µm' if sift.success else 'SIFT=FAIL'
    fig.suptitle(
        f'SIFT+RANSAC Localization  |  {wsi_name}\n'
        f'GT=({gt_x},{gt_y})  Retrieval err={ret_err:.1f}µm  {sift_tag}  '
        f'matches={sift.match_count}  inliers={sift.inlier_count}',
        fontsize=10,
    )
    fig.tight_layout()
    fig.subplots_adjust(right=0.87)

    # ── Right-side feature legend ─────────────────────────────────────────────
    feat_lines = [
        'Panel [1,3] features',
        '─' * 20,
        f'{"✓" if h_decomp    else "✗"} H decomposition',
        f'{"✓" if patch_grid  else "✗"} Patch grid proj.',
        f'{"✓" if trans_arrow else "✗"} Translation arrow',
        '✓ Pixel error (always)',
        '',
        'Disable via flag:',
        '  --no-h-decomp',
        '  --no-patch-grid',
        '  --no-trans-arrow',
    ]
    fig.text(0.882, 0.30, '\n'.join(feat_lines),
             fontsize=7, va='center', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5e8', alpha=0.88))

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi',
                    default='/work/u26130998/datasets/histoimage.na.icar.cnr.it/BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs')
    ap.add_argument('--x',       type=int,   default=31700)
    ap.add_argument('--y',       type=int,   default=33600)
    ap.add_argument('--mpp',     type=float, default=0.252)
    ap.add_argument('--ratio',   type=str,   default='45:32')
    ap.add_argument('--mpixels', type=float, default=1.475)
    ap.add_argument('--tile',    type=int,   default=256)
    ap.add_argument('--overlap', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--filter',  action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--min-region-ratio', type=float, default=0.10)
    ap.add_argument('--batch',   type=int,   default=1024)
    ap.add_argument('--padding', type=int,   default=2,
                    help='Padding around retrieval best match for SIFT crop (in tiles)')
    ap.add_argument('--min-inliers', type=int, default=10)
    ap.add_argument('--h-decomp',    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--patch-grid',  action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--trans-arrow', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--out',     default=None)
    args = ap.parse_args()

    if not os.path.exists(args.wsi):
        print(f'[SKIP] WSI not found: {args.wsi}')
        return

    wsi_name = os.path.basename(args.wsi)
    print(f'WSI : {args.wsi}')
    print(f'GT  : x={args.x}  y={args.y}  mpp={args.mpp}')

    timings: dict[str, float] = {}

    # ── Step 1: Crop query ────────────────────────────────────────────────────
    print('\n[1] Cropping query from WSI...')
    t0 = time.perf_counter()
    qfwsi = QueryFromWSI(
        args.wsi, WH_ratio=args.ratio, MPixels=args.mpixels,
        mpp=args.mpp, x_top_left=args.x, y_top_left=args.y,
    )
    query_pil = qfwsi.load_query_image()
    if query_pil is None:
        print('[FAIL] QueryFromWSI returned None')
        return
    query_np  = np.array(query_pil)
    query_qpc = QueryPatchContainer(query_np)
    query_qpc.extract_all(args.tile, overlap=args.overlap)
    timings['1. crop query'] = time.perf_counter() - t0
    print(f'  {query_pil.width}×{query_pil.height}  patches={query_qpc.grid.grid_rows}×{query_qpc.grid.grid_cols}')

    wsi      = qfwsi.wsi
    base_mpp = float(wsi.properties.get('openslide.mpp-x', 0))

    # ── Step 0: Load model ────────────────────────────────────────────────────
    print('\n[0] Loading GigaPath model...')
    t0 = time.perf_counter()
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _model  = gigapath_model(device)
    encoder = lambda patches: gigapath_encode(patches, _model, device, batch_size=args.batch)
    timings['0. load model'] = time.perf_counter() - t0
    print(f'  device={device}')

    # ── Step 2: Estimate MPP ──────────────────────────────────────────────────
    print('\n[2] Estimating MPP...')
    t0 = time.perf_counter()
    est      = GigaPathKnnEstiMpp(wsi, encoder=encoder, tile_size=args.tile)
    mpp_result = est.estimate(query_qpc)
    mpp_est  = mpp_result.estimated_mpp
    ds_est   = mpp_est / base_mpp
    timings['2. estimate mpp'] = time.perf_counter() - t0
    print(f'  mpp_gt={args.mpp:.4f}  mpp_est={mpp_est:.4f}  '
          f'error={abs(mpp_est - args.mpp) / args.mpp * 100:.1f}%')

    # ── Step 3: Tissue mask ───────────────────────────────────────────────────
    print('\n[3] Building tissue mask...')
    t0 = time.perf_counter()
    mask = TissuesRegionsMask.from_wsi(wsi)
    if args.filter:
        mask.filter_regions(args.min_region_ratio)
    mask.filter_patchable(tile_size=args.tile, ds=ds_est)
    timings['3. tissue mask'] = time.perf_counter() - t0
    print(f'  {len(mask.tissue_regions)} regions after filtering')

    # ── Step 4: Sliding window retrieval ─────────────────────────────────────
    print('\n[4] Running GigaPathSlidingWinSim...')
    t0 = time.perf_counter()
    retrieval_pipeline = GigaPathSlidingWinSim(
        wsi, encoder, mask=mask, mpp=mpp_est,
        tile_size=args.tile, overlap=args.overlap,
    )
    retrieval_pipeline.build_wsi_features()
    retrieval_pipeline.build_query_features(query_qpc)
    retrieval_pipeline.compute_sim_maps()
    retrieval_result = retrieval_pipeline.find_best()
    timings['4. retrieval'] = time.perf_counter() - t0
    print(f'  best=({retrieval_result.best_x0}, {retrieval_result.best_y0})  '
          f'score={retrieval_result.best_score:.4f}  '
          f'region={retrieval_result.best_region_index}  '
          f'overlap={retrieval_result.from_overlap}')

    # ── Step 5: SIFT + RANSAC ─────────────────────────────────────────────────
    print('\n[5] Running SIFT+RANSAC...')
    t0 = time.perf_counter()
    localizer = SiftRansacLocalizer(
        retrieval_pipeline.wsi_container, query_qpc, retrieval_result,
        min_inliers=args.min_inliers, padding=args.padding,
    )
    localizer.read_wsi_crop()
    localizer.detect_and_match()
    sift_result = localizer.estimate_homography()
    timings['5. sift ransac'] = time.perf_counter() - t0
    print(f'  success={sift_result.success}  '
          f'matches={sift_result.match_count}  inliers={sift_result.inlier_count}')
    print(f'  SIFT location=({sift_result.x0}, {sift_result.y0})')

    # ── Step 6: Verify ────────────────────────────────────────────────────────
    print('\n[6] Verifying...')
    def dist_um(x, y):
        return math.sqrt((x - args.x) ** 2 + (y - args.y) ** 2) * base_mpp

    ret_err  = dist_um(retrieval_result.best_x0, retrieval_result.best_y0)
    sift_err = dist_um(sift_result.x0, sift_result.y0) if sift_result.success else float('nan')
    tol_um   = args.tile * ds_est * base_mpp

    print(f'  GT              : ({args.x}, {args.y})')
    print(f'  Retrieval       : ({retrieval_result.best_x0}, {retrieval_result.best_y0})  '
          f'error={ret_err:.1f} µm')
    if sift_result.success:
        improvement = ret_err - sift_err
        tag = f'  ({"+" if improvement > 0 else ""}{improvement:.1f} µm improvement)'
        print(f'  SIFT refined    : ({sift_result.x0}, {sift_result.y0})  '
              f'error={sift_err:.1f} µm{tag}')
    else:
        print(f'  SIFT refined    : FAILED '
              f'(matches={sift_result.match_count}  inliers={sift_result.inlier_count})')
    print(f'  Tolerance       : {tol_um:.1f} µm (one tile)')

    # ── Figure ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    Ht, Wt = mask.main_mask.shape
    thumb = np.array(wsi.get_thumbnail((Wt, Ht)).convert('RGB'))

    # Zoomed crop for panel [1,2] — must read before wsi.close()
    zoom_pad  = 4
    tile_l0   = args.tile * retrieval_result.ds
    crop_x0_12 = max(0, int(retrieval_result.best_x0 - zoom_pad * tile_l0))
    crop_y0_12 = max(0, int(retrieval_result.best_y0 - zoom_pad * tile_l0))
    R_q = query_qpc.grid.grid_rows
    C_q = query_qpc.grid.grid_cols
    crop_level_12 = wsi.get_best_level_for_downsample(retrieval_result.ds)
    crop_ds_12    = wsi.level_downsamples[crop_level_12]
    crop_w_12     = int((C_q + zoom_pad * 2) * args.tile * retrieval_result.ds / crop_ds_12)
    crop_h_12     = int((R_q + zoom_pad * 2) * args.tile * retrieval_result.ds / crop_ds_12)
    crop_img_12   = np.array(
        wsi.read_region((crop_x0_12, crop_y0_12), crop_level_12,
                        (crop_w_12, crop_h_12)).convert('RGB')
    )

    tag = f"{'ov' if args.overlap else 'nov'}_{'flt' if args.filter else 'noflt'}"
    out = args.out or os.path.join(job_result_dir('SiftRansacTest'),
                                    f'sift_ransac__{tag}.png')

    draw_figure(
        thumb=thumb, mask=mask,
        query_img=query_np,
        query_kps=localizer.query_kps,
        wsi_crop=localizer.wsi_crop,
        crop_kps=localizer.crop_kps,
        good_matches=localizer.good_matches,
        retrieval=retrieval_result,
        sift=sift_result,
        gt_x=args.x, gt_y=args.y,
        base_mpp=base_mpp, tile_size=args.tile,
        wsi_name=wsi_name, mpp_gt=args.mpp, mpp_est=mpp_est,
        crop_img_12=crop_img_12, crop_x0_12=crop_x0_12, crop_y0_12=crop_y0_12,
        crop_ds_12=crop_ds_12, zoom_pad=zoom_pad,
        query_rows=R_q, query_cols=C_q,
        crop_origin_x=localizer.crop_origin_x or 0,
        crop_origin_y=localizer.crop_origin_y or 0,
        h_decomp=args.h_decomp,
        patch_grid=args.patch_grid,
        trans_arrow=args.trans_arrow,
        out=out,
    )
    timings['6. figure'] = time.perf_counter() - t0

    wsi.close()

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


if __name__ == '__main__':
    main()
