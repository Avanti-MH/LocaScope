"""Reusable localization diagnostic panels (the bottom row of test_sift_ransac's figure).

`draw_localization_row(...)` renders 4 panels for one shot:

    [0] SIFT matches      query | wsi_crop with green match lines
    [1] Result summary    retrieval + SIFT numbers as text
    [2] Zoomed +-N tiles  green--=GT  yellow=main  orange=overlap  blue=SIFT
    [3] Homography        query boundary + patch grid + translation arrow

Works with either SlideWinSimResult (2_retrieval/GigaPathSlidingWinSim) or
SlideWinSimRotResult (GigaPathSlidingWinSimRot) — both expose the
best_*/main_*/overlap_* fields this module reads. When the result carries a
`best_rotation`, it is shown in the summary panel.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import matplotlib.patches as mpatches


# ── helpers ───────────────────────────────────────────────────────────────────

def query_quad(query_shape, H) -> np.ndarray:
    """The query's 4 corners in crop px, from the origin round to (0, h).

    A rotated quad, not an axis-aligned box: a box drawn from the mapped (0, 0)
    extends in the wrong direction once the query came in rotated, and pushes
    the mapped centre outside its own footprint.

    Every caller draws with it, which is why it sits here, while `is_invertible`
    -- the other question worth asking about an H -- sits in
    3_localization/SIFT_RANSAC.py, where the code that must not import
    matplotlib can reach it.
    """
    h_q, w_q = query_shape[:2]
    corners = np.float32(
        [[0, 0], [w_q, 0], [w_q, h_q], [0, h_q]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(
        corners, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def warp_query_into(query_img, crop_shape, H):
    """Warp the query into the WSI crop's frame. Returns (warped, cover).

    This is the direction the figures want: the question is where the photo sits
    on the slide, so the slide is the frame that stays still and the photo is
    what moves into it. It also uses H as estimated -- query px -> crop px --
    instead of its inverse.

    `cover` is the footprint the query actually lands on, and it is needed
    because the crop is bigger than the query. SIFT_RANSAC pads the crop by 2
    tiles on every side, so a 1440x1024 photo covers about 28% of the resulting
    2560x2048 crop; a blend or a checker run over the whole crop would turn the
    other 72% into half-dark noise and hide the thing being checked.

    Only call this on an H that `is_invertible`. `warpPerspective` inverts M
    internally and ignores the return code, so a singular one paints black in
    silence rather than raising the way `np.linalg.inv` would.
    """
    ch, cw = crop_shape[:2]
    M = np.asarray(H, dtype=np.float64)
    warped = cv2.warpPerspective(query_img, M, (cw, ch))
    ones = np.full(query_img.shape[:2], 255, np.uint8)
    cover = cv2.warpPerspective(ones, M, (cw, ch)) > 127
    return warped, cover


def blend_in_footprint(crop, warped, cover, alpha: float = 0.5) -> np.ndarray:
    """The crop with the warped query blended over it, inside `cover` only."""
    out = crop.copy()
    out[cover] = cv2.addWeighted(crop, 1.0 - alpha, warped, alpha, 0.0)[cover]
    return out


def checker_in_footprint(crop, warped, cover, n: int = 8) -> np.ndarray:
    """Interleave the warped query with the crop in an n x n checker.

    The cells are laid out over the footprint's bounding box rather than over
    the whole crop: an 8x8 grid across a crop that the query covers a quarter of
    would put only about 2x2 cells on the thing being checked.
    """
    out = crop.copy()
    ys, xs = np.nonzero(cover)
    if ys.size == 0:
        return out
    y0, x0 = int(ys.min()), int(xs.min())
    ch = max(1, (int(ys.max()) + 1 - y0) // n)
    cw = max(1, (int(xs.max()) + 1 - x0) // n)
    rows = ((np.arange(crop.shape[0], dtype=np.int32) - y0) // ch)[:, None]
    cols = ((np.arange(crop.shape[1], dtype=np.int32) - x0) // cw)[None, :]
    take = (((rows + cols) % 2) == 1) & cover
    out[take] = warped[take]
    return out


def match_img(query_img, query_kps, wsi_crop, crop_kps, matches,
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

    canvas[q_y0:q_y0 + h_q, :w_q]            = query_img
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


def read_anchored_crop(wsi, x0_l0: float, y0_l0: float, ds: float,
                       tile_size: int, query_rows: int, query_cols: int,
                       zoom_pad: int = 4):
    """Read a (cols + 2*pad) x (rows + 2*pad) tile crop anchored at a level-0 point.

    `ds` is the downsample the TILES are measured in (the retrieval level's),
    which is not the same as `crop_ds`, the downsample of the level the pixels
    are actually read from. Those two are separate on purpose: mixing them is
    how a crop ends up the right size at the wrong scale.

    Returns (crop_img, crop_x0, crop_y0, crop_ds) — all level-0 anchored.
    """
    tile_l0 = tile_size * ds
    crop_x0 = max(0, int(x0_l0 - zoom_pad * tile_l0))
    crop_y0 = max(0, int(y0_l0 - zoom_pad * tile_l0))
    crop_level = wsi.get_best_level_for_downsample(ds)
    crop_ds    = wsi.level_downsamples[crop_level]
    crop_w = int((query_cols + zoom_pad * 2) * tile_size * ds / crop_ds)
    crop_h = int((query_rows + zoom_pad * 2) * tile_size * ds / crop_ds)
    crop_img = np.array(
        wsi.read_region((crop_x0, crop_y0), crop_level, (crop_w, crop_h)).convert('RGB')
    )
    return crop_img, crop_x0, crop_y0, crop_ds


def read_zoom_crop(wsi, retrieval, tile_size: int, query_rows: int,
                   query_cols: int, zoom_pad: int = 4):
    """Read the panel-[2] zoom crop around the retrieval best match.

    Returns (crop_img, crop_x0, crop_y0, crop_ds) — all level-0 anchored.
    """
    return read_anchored_crop(wsi, retrieval.best_x0, retrieval.best_y0,
                              retrieval.ds, tile_size, query_rows, query_cols,
                              zoom_pad)


def _retrieval_center(retrieval, query_img, tile_size, query_rows, query_cols,
                      override=None):
    """Centre @ level-0 of the retrieval's matched window.

    `override` lets the caller pass the value it already computed (the bench
    derives it from the FoV footprint, not the grid-rounded one) so the figure
    and metrics.csv never disagree.
    """
    if override is not None:
        return override
    w_l0 = query_cols * tile_size * retrieval.ds
    h_l0 = query_rows * tile_size * retrieval.ds
    return retrieval.best_x0 + w_l0 / 2.0, retrieval.best_y0 + h_l0 / 2.0


# ── the 4 panels ──────────────────────────────────────────────────────────────

def draw_localization_row(
    axes,                       # sequence of 4 matplotlib Axes
    query_img:   np.ndarray,
    query_kps,
    wsi_crop:    Optional[np.ndarray],
    crop_kps,
    good_matches,
    retrieval,                  # SlideWinSimResult | SlideWinSimRotResult
    sift,                       # SiftRansacResult
    gt_x: int, gt_y: int,
    base_mpp: float,
    tile_size: int,
    crop_img: np.ndarray, crop_x0: int, crop_y0: int, crop_ds: float,
    zoom_pad: int,
    query_rows: int, query_cols: int,
    crop_origin_x: int = 0, crop_origin_y: int = 0,
    gt_center:   Optional[tuple] = None,  # (cx, cy) @ level-0 — rotation-invariant GT
    retr_center: Optional[tuple] = None,  # (cx, cy) @ level-0 — caller's own value
    gt_box_wh:   Optional[tuple] = None,  # (w, h) @ level-0 of the UNROTATED GT rect
    h_decomp: bool = True, patch_grid: bool = True, trans_arrow: bool = True,
) -> None:
    """Render the 4 localization panels into `axes` (len 4)."""

    def dist_um(x, y):
        return math.sqrt((x - gt_x) ** 2 + (y - gt_y) ** 2) * base_mpp

    def center_err_um(x, y):
        if gt_center is None:
            return float('nan')
        return math.sqrt((x - gt_center[0]) ** 2 + (y - gt_center[1]) ** 2) * base_mpp

    ret_err  = dist_um(retrieval.best_x0, retrieval.best_y0)
    sift_err = dist_um(sift.x0, sift.y0) if sift.success else float('nan')
    best_rot = getattr(retrieval, 'best_rotation', None)

    # ── [0] SIFT matches ──────────────────────────────────────────────────────
    ax = axes[0]
    if wsi_crop is not None and good_matches:
        ax.imshow(match_img(query_img, query_kps, wsi_crop, crop_kps, good_matches))
        tag = 'SUCCESS' if sift.success else 'FAIL'
        ax.set_title(f'SIFT matches [{tag}]\n'
                     f'{sift.match_count} good  {sift.inlier_count} inliers')
    else:
        ax.set_title('SIFT matches — no crop / no matches')

    # ── [1] Result summary ────────────────────────────────────────────────────
    ax = axes[1]
    lines = [
        'Retrieval (GigaPath sliding window)',
        f'  best_x0 = {retrieval.best_x0}',
        f'  best_y0 = {retrieval.best_y0}',
        f'  score   = {retrieval.best_score:.4f}',
        f'  overlap = {retrieval.from_overlap}',
        f'  error   = {ret_err:.1f} um',
    ]
    if best_rot is not None:
        lines.append(f'  rotation= {best_rot} deg')
        scores = getattr(retrieval, 'scores_by_rotation', None)
        if scores:
            for r in sorted(scores):
                mark = ' <-' if r == best_rot else ''
                lines.append(f'    rot {r:>3}: {scores[r]:.4f}{mark}')
    lines += [
        '',
        'SIFT + RANSAC',
        f'  x0      = {sift.x0}',
        f'  y0      = {sift.y0}',
        f'  success = {sift.success}',
        f'  matches = {sift.match_count}',
        f'  inliers = {sift.inlier_count}',
        (f'  error   = {sift_err:.1f} um' if sift.success else '  error   = N/A'),
        '',
        f'GT  x={gt_x}  y={gt_y}',
    ]
    if gt_center is not None:
        c_ret = center_err_um(*_retrieval_center(retrieval, query_img, tile_size,
                                                 query_rows, query_cols,
                                                 override=retr_center))
        c_sft = center_err_um(sift.center_x0, sift.center_y0)
        lines += [
            f'GT  centre=({gt_center[0]:.0f}, {gt_center[1]:.0f})',
            '',
            'CENTRE error (rotation-invariant)',
            f'  retrieval = {c_ret:.1f} um',
            f'  SIFT      = {c_sft:.1f} um',
        ]
    ax.text(0.05, 0.95, '\n'.join(lines), transform=ax.transAxes,
            va='top', fontsize=9, fontfamily='monospace')
    ax.set_title('Result summary')

    # ── [2] Zoomed WSI crop with tile grids + result boxes ────────────────────
    ax = axes[2]
    ax.imshow(crop_img)
    tile_px = tile_size * retrieval.ds / crop_ds
    bm_px   = (retrieval.main_x0 - crop_x0) / crop_ds
    bm_py   = (retrieval.main_y0 - crop_y0) / crop_ds
    crop_h_px, crop_w_px = crop_img.shape[:2]
    half_px = tile_px / 2

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
    bo_px = (retrieval.overlap_x0 - crop_x0) / crop_ds
    bo_py = (retrieval.overlap_y0 - crop_y0) / crop_ds
    ax.add_patch(mpatches.Rectangle(
        (bo_px, bo_py), box_w, box_h,
        fill=False, edgecolor='orange', linewidth=2.0, label='Overlap best'))
    # gt_box_wh is the footprint the shot actually covers, so its dims are
    # swapped for a 90/270 shot. Anchor on the GT CENTRE, not on gt_x/gt_y:
    # the rotation happens about that centre, and the pre-rotation corner no
    # longer bounds the swapped box.
    gt_bw, gt_bh = ((gt_box_wh[0] / crop_ds, gt_box_wh[1] / crop_ds)
                    if gt_box_wh else (box_w, box_h))
    if gt_center is not None:
        gt_bx = (gt_center[0] - crop_x0) / crop_ds - gt_bw / 2
        gt_by = (gt_center[1] - crop_y0) / crop_ds - gt_bh / 2
    else:
        gt_bx = (gt_x - crop_x0) / crop_ds
        gt_by = (gt_y - crop_y0) / crop_ds
    ax.add_patch(mpatches.Rectangle(
        (gt_bx, gt_by), gt_bw, gt_bh,
        fill=False, edgecolor='green', linewidth=2.0, linestyle='--', label='GT'))

    # SIFT footprint is the query's 4 corners through H — a rotated quad, not an
    # axis-aligned box. Drawing a box from the mapped (0,0) would extend in the
    # wrong direction and push the centre marker outside it.
    if sift.success and sift.H is not None:
        mapped = query_quad(query_img.shape, sift.H)
        # crop px (level-n) -> level-n global -> level-0 -> zoom-crop px
        qx = ((mapped[:, 0] + crop_origin_x) * sift.ds - crop_x0) / crop_ds
        qy = ((mapped[:, 1] + crop_origin_y) * sift.ds - crop_y0) / crop_ds
        ax.plot(np.append(qx, qx[0]), np.append(qy, qy[0]),
                color='dodgerblue', lw=2.0, label='SIFT')
    elif sift.success:
        ax.add_patch(mpatches.Rectangle(
            ((sift.x0 - crop_x0) / crop_ds, (sift.y0 - crop_y0) / crop_ds),
            box_w, box_h,
            fill=False, edgecolor='dodgerblue', linewidth=2.0, label='SIFT'))

    # Rotation-invariant centres — these are what the bench actually scores
    def to_crop(x0, y0):
        return (x0 - crop_x0) / crop_ds, (y0 - crop_y0) / crop_ds

    if gt_center is not None:
        gx, gy = to_crop(*gt_center)
        ax.plot(gx, gy, '*', color='lime', ms=16, mec='black', mew=0.8,
                label='GT centre')
        rx, ry = to_crop(*_retrieval_center(retrieval, query_img, tile_size,
                                            query_rows, query_cols,
                                            override=retr_center))
        ax.plot(rx, ry, 'x', color='yellow', ms=10, mew=2.2, label='Retr centre')
        if sift.success:
            sx_, sy_ = to_crop(sift.center_x0, sift.center_y0)
            ax.plot(sx_, sy_, 'D', color='dodgerblue', ms=7, mec='black', mew=0.6,
                    label='SIFT centre')

    # Lock the view to the crop itself. Any of the boxes/markers above can land
    # far outside it (e.g. the overlap-grid best when the main grid won), and
    # matplotlib would otherwise autoscale to include them, squashing the image.
    ax.set_xlim(0, crop_w_px)
    ax.set_ylim(crop_h_px, 0)

    ax.legend(fontsize=7, loc='upper right', framealpha=0.6)
    ax.set_title(f'Zoomed +-{zoom_pad} tiles\n'
                 f'boxes: green--=GT yellow=main orange=overlap blue=SIFT   '
                 f'markers = centres')

    # ── [3] Homography analysis ───────────────────────────────────────────────
    ax = axes[3]
    ax.imshow(wsi_crop if wsi_crop is not None else np.zeros((64, 64, 3), np.uint8))

    if sift.success and sift.H is not None:
        H = sift.H
        h_q, w_q = query_img.shape[:2]

        mapped = query_quad(query_img.shape, H)
        poly   = np.vstack([mapped, mapped[0]])
        ax.plot(poly[:, 0], poly[:, 1], 'lime', linewidth=2.0, label='Query boundary')

        A  = H[:2, :2]
        sx = float(np.linalg.norm(A[:, 0]))
        sy = float(np.linalg.norm(A[:, 1]))
        theta_deg = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
        dx_c, dy_c = float(H[0, 2]), float(H[1, 2])
        p20, p21   = float(H[2, 0]), float(H[2, 1])
        sift_err_ln = sift_err / base_mpp / sift.ds
        # Centre error is the one to read. The top-left number compares
        # gt_x/gt_y (pre-rotation corner) against the mapped query origin
        # (post-rotation corner), so a 90/180/270 shot inflates it by roughly
        # one FoV even when the localisation is spot on.
        c_err    = center_err_um(sift.center_x0, sift.center_y0)
        c_err_ln = c_err / base_mpp / sift.ds

        if h_decomp:
            txt = (f'theta={theta_deg:+.2f} deg\n'
                   f'sx={sx:.4f}  sy={sy:.4f}\n'
                   f'dx={dx_c:+.1f}px  dy={dy_c:+.1f}px\n'
                   f'persp=({p20:.2e},{p21:.2e})\n'
                   f'centre err={c_err:.1f}um ({c_err_ln:.1f}px@lvl{sift.level})\n'
                   f'top-left  ={sift_err:.0f}um (rot-naive)')
            ax.text(4, 4, txt, color='white', fontsize=6.5, va='top',
                    fontfamily='monospace',
                    bbox=dict(facecolor='black', alpha=0.55, pad=2, boxstyle='round'))

        if patch_grid:
            for x in range(tile_size, w_q, tile_size):
                pts = np.float32([[[float(x), 0.]], [[float(x), float(h_q)]]])
                p = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                ax.plot(p[:, 0], p[:, 1], color='yellow', lw=0.6, alpha=0.65)
            for y in range(tile_size, h_q, tile_size):
                pts = np.float32([[[0., float(y)]], [[float(w_q), float(y)]]])
                p = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                ax.plot(p[:, 0], p[:, 1], color='yellow', lw=0.6, alpha=0.65)

        if trans_arrow:
            # Compare CENTRES, not top-lefts: for a 90/180/270 shot the two
            # top-lefts are different corners of the same footprint, so an arrow
            # between them shows the rotation rather than the correction.
            rc = _retrieval_center(retrieval, query_img, tile_size,
                                   query_rows, query_cols, override=retr_center)
            ret_cx = float(rc[0] / sift.ds - crop_origin_x)
            ret_cy = float(rc[1] / sift.ds - crop_origin_y)
            sc = cv2.perspectiveTransform(
                np.float32([[[w_q / 2.0, h_q / 2.0]]]), H)[0, 0]
            ax.annotate('', xy=(float(sc[0]), float(sc[1])), xytext=(ret_cx, ret_cy),
                        arrowprops=dict(arrowstyle='->', color='dodgerblue', lw=1.5))
            ax.plot(ret_cx, ret_cy, 'x', color='yellow', ms=8, mew=2,
                    label='Retrieval centre')
            ax.plot(sc[0], sc[1], 'D', color='dodgerblue', ms=5, label='SIFT centre')

        ax.legend(fontsize=6.5, loc='lower right', framealpha=0.65)
        rot_tag = f'  rot={best_rot}deg' if best_rot is not None else ''
        ax.set_title(f'Homography  theta={theta_deg:+.1f}  sx={sx:.3f}  sy={sy:.3f}{rot_tag}\n'
                     f'CENTRE err={c_err:.1f}um ({c_err_ln:.1f}px@lvl{sift.level})   '
                     f'| top-left {sift_err:.0f}um (rot-naive)')
    else:
        ax.set_title('Homography failed')


# ── recall-failure panels ─────────────────────────────────────────────────────

def _box_in_crop(ax, cx_l0, cy_l0, w_l0, h_l0, crop_x0, crop_y0, crop_ds,
                 colour, label):
    """Draw a level-0 centred box in a crop's own pixel coordinates."""
    if cx_l0 is None or cy_l0 is None:
        return
    x = (cx_l0 - w_l0 / 2.0 - crop_x0) / crop_ds
    y = (cy_l0 - h_l0 / 2.0 - crop_y0) / crop_ds
    ax.add_patch(mpatches.Rectangle((x, y), w_l0 / crop_ds, h_l0 / crop_ds,
                                    fill=False, ec=colour, lw=2, label=label))
    ax.plot((cx_l0 - crop_x0) / crop_ds, (cy_l0 - crop_y0) / crop_ds,
            'x', color=colour, ms=9, mew=2)


def draw_recall_row(
    axes,                       # sequence of 4 matplotlib Axes
    query_img:    np.ndarray,
    gt_crop:      Optional[np.ndarray],
    gt_anchor:    tuple,        # (crop_x0, crop_y0, crop_ds) of gt_crop
    gt_center:    tuple,        # (cx, cy) @ level-0
    pick_crop:    Optional[np.ndarray],
    pick_anchor:  tuple,
    pick_center:  Optional[tuple],
    box_wh:       tuple,        # (w, h) @ level-0 of the FoV footprint
    summary:      list,         # lines of text, assembled by the caller
) -> None:
    """Render the 4-panel RETRIEVAL diagnostic for one shot.

    The stage-3 row answers "did SIFT finish the job". This one answers the
    question before it: the truth was never proposed, so what does the place it
    should have found look like next to the place retrieval preferred? Nothing
    here comes from SIFT -- on a recall failure stage 3 was handed the wrong
    window and its picture says nothing about why.

        [0] query
        [1] WSI at the TRUTH        green box
        [2] WSI at retrieval's pick yellow box
        [3] the numbers
    """
    w_l0, h_l0 = box_wh

    ax = axes[0]
    ax.imshow(query_img)
    ax.set_title(f'query  {query_img.shape[1]}x{query_img.shape[0]}')
    ax.axis('off')

    ax = axes[1]
    if gt_crop is not None:
        ax.imshow(gt_crop)
        _box_in_crop(ax, gt_center[0], gt_center[1], w_l0, h_l0,
                     gt_anchor[0], gt_anchor[1], gt_anchor[2], 'lime', 'ground truth')
        ax.legend(fontsize=7, loc='lower right', framealpha=0.65)
        ax.set_title('WSI at the TRUTH  (never proposed)')
    else:
        ax.set_title('truth crop unavailable')
    ax.axis('off')

    ax = axes[2]
    if pick_crop is not None:
        ax.imshow(pick_crop)
        _box_in_crop(ax, (pick_center or (None, None))[0],
                     (pick_center or (None, None))[1], w_l0, h_l0,
                     pick_anchor[0], pick_anchor[1], pick_anchor[2],
                     'yellow', 'retrieval pick')
        ax.legend(fontsize=7, loc='lower right', framealpha=0.65)
        ax.set_title('WSI at what retrieval PREFERRED')
    else:
        ax.set_title('pick crop unavailable')
    ax.axis('off')

    ax = axes[3]
    ax.axis('off')
    ax.text(0.02, 0.98, '\n'.join(summary), va='top', ha='left',
            family='monospace', fontsize=9, transform=ax.transAxes)
    ax.set_title('numbers')

    for ax in axes:
        ax.axis('off')
