#!/usr/bin/env python3
"""
End-to-end pipeline test:
  0. _sim_tensors      — the window kernel's fast contraction against the
                          original, scored on decoys. Pure tensors, no slide,
                          no model; runs first because a failure invalidates
                          every number below it.
  1. QueryFromWSI      — crop a query at known (x, y, mpp) from the WSI
  2. estimate_mpp      — estimate the query MPP from GigaPath features
  3. compute_gigapath_sliding_win_similarity — sliding window similarity at estimated MPP
  4. Verify            — best match should be near the ground-truth crop location
  5. GigaPathSlidingWinSimRot — the rotation-aware retriever LocaScopePipeline
                          actually calls, checked against the same crop and a
                          decoy comparison against step 3's non-rotated result
  6. build_wsi_features    — rebuilding at another scale: the features cache
                          hits, and the similarity maps of the old scale do not
                          survive into the new one. Stub encoder, ~1 second.

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

from PatchingLib import QueryPatchContainer, WsiTissuesContainer
from TissuesRegionsMask import TissuesRegionsMask
from QueryFromWSI import QueryFromWSI
from GigaPathKnnEstiMpp import GigaPathKnnEstiMpp
from GigaPathSlidingWinSim import compute_gigapath_sliding_win_similarity
from GigaPathSlidingWinSimRot import GigaPathSlidingWinSimRot
from GigaPathFunc import GigaPathEncoderConfig


ROTATIONS = (0, 90, 180, 270)


# ── Canvas builder ────────────────────────────────────────────────────────────

def build_sim_canvas(
    mask: TissuesRegionsMask,
    regions: list,
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

    # `regions` is the container's list, which from_mpp has already narrowed to
    # what can host a tile at this ds; `mask` is still needed for to_mask_xy and
    # mask_ds_*. They are different lists now and only the first lines up with
    # sim_maps.
    for region, (main_sim, overlap_sim) in zip(regions, sim_maps):
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


def _find_best(regions, sim_maps, ds, tile_size, use_overlap: bool):
    """Return (x_l0, y_l0, score) for either the main or overlap grid.

    `regions` must be the list the sim_maps were computed over -- that is
    `container.tissue_regions`, not `mask.tissue_regions`. from_mpp drops the
    regions too small to host a tile, so the two differ, and pairing them the
    wrong way puts every window on a neighbouring region's coordinates without
    raising anything.
    """
    best_score = -np.inf
    best_x = best_y = 0
    cell_l0 = tile_size * ds
    half_l0 = cell_l0 / 2

    for region, (main_sim, overlap_sim) in zip(regions, sim_maps):
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


# ── step 0: _sim_tensors, the fast contraction against the original ─────────

ROT_PASS, ROT_FAIL = [], []


def run_sim_tensors_equivalence(seed: int = 0) -> None:
    """Step 0. Does contracting D first give the same windows?

    `_sim_tensors` used to write `windows * q` and sum afterwards, which
    materialises [D, H, W, R_q, C_q] -- 2.62 GB for a 145x147 region against a
    4x5 kernel, to produce 1.71 MB. It now contracts D first with einsum and
    indexes the windows out. Same definition, different order, and matmul does
    not reduce in the same order as an elementwise product, so the two answers
    are close rather than equal.

    Close is not a thing to assert against a number somebody chose. The
    comparison is against DECOYS instead: the same computation with the query
    rolled by one tile, and with the WSI rolled by one tile. If the rewrite is
    right, its disagreement with the original is orders of magnitude below its
    disagreement with either decoy. If it is wrong, the three land in the same
    range and no threshold would have separated them.

    Pure tensors, no model and no slide, so it runs first -- before the WSI is
    opened or GigaPath is loaded. A failure here means every number the rest of
    this file produces was computed with a broken kernel.
    """
    print('\n[0] _sim_tensors: einsum contraction vs the original unfold...')
    from GigaPathSlidingWinSim import _sim_tensors, _sim_tensors_unfold

    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
        print(f'  torch.backends.cuda.matmul.allow_tf32 = '
              f'{torch.backends.cuda.matmul.allow_tf32}   '
              f'(True keeps 10 mantissa bits, so expect ~1e-3 not ~1e-7)')

    R_w, C_w, R_q, C_q, D = 12, 11, 4, 5, 64
    for device in devices:
        generator = torch.Generator(device='cpu').manual_seed(seed)
        wsi = torch.randn(R_w, C_w, D, generator=generator)
        query = torch.randn(R_q, C_q, D, generator=generator)
        wsi = (wsi / wsi.norm(dim=-1, keepdim=True)).to(device)
        query = (query / query.norm(dim=-1, keepdim=True)).to(device)

        fast = _sim_tensors(query, wsi)
        reference = _sim_tensors_unfold(query, wsi)
        rot_check(f'sim_tensors[{device.type}]: same shape',
                  tuple(fast.shape) == tuple(reference.shape),
                  f'{tuple(fast.shape)} vs {tuple(reference.shape)}')

        gap = float((fast - reference).abs().max())
        decoys = {
            'query rolled one tile': _sim_tensors_unfold(
                torch.roll(query, shifts=1, dims=0), wsi),
            'wsi rolled one tile': _sim_tensors_unfold(
                query, torch.roll(wsi, shifts=1, dims=0)),
        }
        for name, decoy in decoys.items():
            spread = float((fast - decoy).abs().max())
            rot_check(f'sim_tensors[{device.type}]: below the {name} decoy',
                      spread > gap * 1000,
                      f'gap {gap:.2e} vs decoy {spread:.2e} '
                      f'({spread / max(gap, 1e-30):.0f}x)')

        # The empty case is a contract, not an accident: a region smaller than
        # the query kernel must come back as an empty tensor, which every
        # caller tests with .numel().
        small = _sim_tensors(query, wsi[:R_q - 1])
        rot_check(f'sim_tensors[{device.type}]: region under the kernel is empty',
                  small.numel() == 0, f'numel {small.numel()}')


# ── step 5: rotation-aware retriever ────────────────────────────────────────


def rot_check(label: str, condition: bool, detail: str = '') -> None:
    tag = 'ok  ' if condition else 'FAIL'
    print(f'  {tag}  {label}' + (f'   {detail}' if detail else ''))
    (ROT_PASS if condition else ROT_FAIL).append(label)


def run_scale_switch_checks(wsi, mask, args) -> None:
    """Step 6. Rebuilding at a different scale: what is reused, what expires.

    `build_wsi_features` grew a second entrance -- give it a new mpp or ds and
    it rebuilds without re-segmenting, and the same scale twice reuses the
    features. Two things can go quietly wrong with that, and neither raises:

        the cache misses     every scale change pays for a full re-encode,
                             which is only a cost, but an invisible one
        the maps survive     sim_maps_by_rot belongs to the scale that made it.
                             find_best() only recomputes when the dict is
                             EMPTY, so a stale dict pairs the previous scale's
                             similarity maps with this scale's regions and
                             returns a confident wrong location

    No GigaPath here. The encoder is a counting stub: what is being tested is
    bookkeeping, and a real forward pass would only make it slower and hide the
    count. That also means this runs in about a second even though it builds
    three times.
    """
    print('\n[6] build_wsi_features: cache and invalidation...')

    calls = {'n': 0}

    def counting_encoder(patches):
        calls['n'] += 1
        # 8 dims is plenty: nothing here looks at the values, and the sliding
        # window never runs. L2-normalised so a FeaturesMap is well formed.
        feats = torch.randn(len(patches), 8)
        return feats / feats.norm(dim=-1, keepdim=True)

    downsamples = [float(d) for d in wsi.level_downsamples]
    if len(downsamples) < 2:
        rot_check('scale switch: slide has two levels', False,
                  f'only {len(downsamples)}')
        return
    # The two coarsest levels: the same code path as any other pair, and the
    # cheapest possible read.
    ds_a, ds_b = downsamples[-1], downsamples[-2]

    r = GigaPathSlidingWinSimRot(wsi, counting_encoder, mask=mask,
                                 tile_size=args.tile, overlap=True)
    r.build_wsi_features(ds=ds_a)
    after_a = calls['n']
    regions_a, feats_a = r.regions, r.wsi_features
    rot_check('scale switch: first build encodes', after_a > 0,
              f'{after_a} encoder calls')
    rot_check('scale switch: ds is the level\'s own', r.ds == ds_a,
              f'asked {ds_a}, built {r.ds}')

    r.build_wsi_features(ds=ds_b)
    after_b = calls['n']
    rot_check('scale switch: a new scale re-encodes', after_b > after_a,
              f'{after_b - after_a} more calls')
    rot_check('scale switch: regions follow the scale',
              r.regions is not regions_a, f'{len(r.regions)} regions at ds={r.ds}')

    # Plant a stale map and confirm the rebuild clears it. Without this,
    # find_best() would skip compute_sim_maps and zip last scale's maps against
    # this scale's regions.
    r.sim_maps_by_rot = {0: ['stale']}
    r.build_wsi_features(ds=ds_a)
    rot_check('scale switch: rebuild empties sim_maps_by_rot',
              r.sim_maps_by_rot == {}, f'got {list(r.sim_maps_by_rot)}')
    rot_check('scale switch: returning to a built scale is free',
              calls['n'] == after_b, f'{calls["n"] - after_b} extra calls')
    rot_check('scale switch: the cache returns the same features',
              r.wsi_features is feats_a and r.regions is regions_a)
    rot_check('scale switch: the container comes back with them',
              r.wsi_container is not None and r.wsi_container.ds == ds_a,
              'stage 3 reads retriever.wsi_container for the pixels')

    before = len(mask.tissue_regions)
    rot_check('scale switch: the caller\'s mask is untouched',
              len(mask.tissue_regions) == before, f'{before} regions')


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
        base_container = WsiTissuesContainer.from_mpp(
            wsi, args.mpp, tile_size=args.tile, overlap=True, mask=mask)
        base_regions = base_container.tissue_regions
        sim_maps = compute_gigapath_sliding_win_similarity(
            qc, base_container, mpp=args.mpp, tile_size=args.tile,
            overlap=True, encoder=encoder)
        mx, my, m_score = _find_best(base_regions, sim_maps, ds, args.tile, use_overlap=False)
        ox, oy, o_score = _find_best(base_regions, sim_maps, ds, args.tile, use_overlap=True)
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

def draw_figure(thumb, mask, regions, query_img_np, query_qpc,
                sim_maps, ds, tile_size,
                gt_x, gt_y, est_x, est_y, error_um,
                wsi_name, mpp_gt, mpp_est,
                crop_img, crop_x0, crop_y0, crop_ds,
                bm_x, bm_y, bo_x, bo_y, pad,
                out):

    Ht, Wt = mask.main_mask.shape
    # `regions` and not mask.tissue_regions: sim_maps was computed over the
    # container's list, which from_ds narrowed to what can host a tile. Panel
    # [0,0] below draws mask.tissue_regions on purpose -- that panel is about
    # the segmentation, this canvas is about the scores.
    sim_canvas = build_sim_canvas(mask, regions, sim_maps, ds, tile_size, Wt, Ht)
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
    ap.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16',
                    help='encoder precision. fp16 is what production ships '
                         '(log/TODO.log: cos 0.99995 against fp32, 5.5x '
                         'faster) and is ignored on CPU. Output is fp32 '
                         'either way -- autocast only changes the forward '
                         'pass (GigaPathFunc.py:174).')
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

    # ── Step 0: _sim_tensors kernel ──────────────────────────────────────────
    # Milliseconds, no slide, no model -- and if it fails, everything below was
    # computed with a broken window kernel, so it goes first.
    run_sim_tensors_equivalence()
    if ROT_FAIL:
        print('\n[0] FAILED -- not spending a GPU hour on top of a broken kernel')
        return 1

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
    base_mpp = wsi.base_mpp  # SafeSlide.base_mpp: mean of mpp-x/y, one definition

    # ── Step 0: Build encoder (shared for all stages) ────────────────────────
    print('\n[0] Loading GigaPath model...')
    t0 = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Same rule as utilities/cli/locate_photo.py:442 -- fp16 on CUDA, fp32
    # elsewhere, because autocast has nothing to offer on CPU. This file claims
    # to exercise the path LocaScopePipeline takes, and the two callers that
    # actually take it (locate_photo, bench_locascope) both hand it an fp16
    # encoder. Running fp32 here left the 5e-5 between the two precisions
    # untested by the one test that says it covers production.
    dtype = (torch.float16
             if args.precision == 'fp16' and device.type == 'cuda'
             else torch.float32)
    print(f'  device={device}  precision={str(dtype).replace("torch.", "")}')
    # From the resolved `dtype`, not from args.precision: the rule above already
    # demoted fp16 to fp32 on CPU, and passing the raw flag would put that back.
    encoder = GigaPathEncoderConfig(batch_size=args.batch)\
        .with_model(dtype='fp16' if dtype is torch.float16 else 'fp32')\
        .build(device)
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
    from TissueSegFunc import TissueSegConfig
    # hsv: these need real tissue tiles, so blank glass has to
    # be excluded. Retrieval does not -- see GigaPathSlidingWinSim.
    mask = TissuesRegionsMask.from_wsi(
        wsi, method=TissueSegConfig('hsv').build())
    before = len(mask.tissue_regions)
    if args.filter:
        mask.filter_regions(args.min_region_ratio)

    # No filter_patchable here. Steps 4 and 5 deliberately run at different
    # scales -- the estimate and the ground truth -- and each one now narrows
    # the mask itself, at the ds it actually resolves to
    # (WsiTissuesContainer.from_ds). This line used to filter at ds_est alone,
    # which let a region 256 level-0 px wide through and then handed step 5 a
    # container with zero patches; the fix after that filtered at
    # max(ds_est, ds_rot), which was still one caller guessing on behalf of
    # two. Neither is needed once the filter lives where the ds is known.
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
    # Build the container here rather than letting the function do it, because
    # its regions are the ones sim_maps line up with: from_mpp drops whatever
    # cannot host a tile at this ds, so mask.tissue_regions is a longer list.
    wsi_container = WsiTissuesContainer.from_mpp(
        wsi, mpp_est, tile_size=args.tile, overlap=args.overlap, mask=mask)
    regions = wsi_container.tissue_regions
    sim_maps = compute_gigapath_sliding_win_similarity(
        query_qpc, wsi_container, mpp=mpp_est,
        tile_size=args.tile,
        overlap=args.overlap,
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

    mx, my, m_score = _find_best(regions, sim_maps, ds_est, args.tile, use_overlap=False)
    ox, oy, o_score = _find_best(regions, sim_maps, ds_est, args.tile, use_overlap=True)

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

    # ── Step 6: rebuilding at another scale ──────────────────────────────────
    # Runs on a stub encoder, so it costs a second and is worth having even
    # when --skip-rot has turned the expensive half off.
    run_scale_switch_checks(wsi, mask, args)

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
        thumb, mask, regions, query_np, query_qpc,
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
