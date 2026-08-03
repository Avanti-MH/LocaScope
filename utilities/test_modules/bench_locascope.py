#!/usr/bin/env python3
"""End-to-end LocaScope bench: read a query_sim multi_batch corpus, run each
shot through LocaScopePipeline, collect per-stage errors, and produce plots.

Inputs (from query_sim/cli/multi_batch.py):
    <gt-csv>         result/MultiBatch/gt.csv
    <images-dir>     result/MultiBatch/images/
    (WSI paths are read from gt.csv's wsi_path column)

Outputs (in --out DIR, default result/BenchLocaScope/):
    metrics.csv               per-shot: mpp/retrieval/refine errors (px + um)
    summary.txt               aggregate stats + failure counts
    stage1_mpp_cdf.png        Stage 1 CDF (aggregate + per-WSI subplots)
    stage2_retr_cdf.png       Stage 2 CDF
    stage3_refine_cdf.png     Stage 3 CDF
    heatmap.png               (WSI, level) x 3 stages, median error, column-normalized colour

Usage:
    python utilities/test_modules/bench_locascope.py \\
        --gt-csv result/MultiBatch/gt.csv \\
        --images-dir result/MultiBatch/images \\
        --out result/BenchLocaScope \\
        [--limit N] [--batch-size 128] [--device auto]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / 'utilities'))
sys.path.insert(0, str(_ROOT / 'aiNNModel'))

from _paths            import job_result_dir                            # noqa: E402
from _sift_plot        import draw_localization_row, read_zoom_crop     # noqa: E402
from _locascope_plots  import write_metrics_csv, render_all             # noqa: E402
from LocaScopePipeline import LocaScopePipeline, LocaScopeShotResult    # noqa: E402
from GigaPathFunc     import gigapath_model, make_gigapath_encoder      # noqa: E402


# ── metric helpers ────────────────────────────────────────────────────────────

def _dist_px(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _fmt(v: Optional[float], fmt: str = '{:.3f}', na: str = '   N/A') -> str:
    return fmt.format(v) if v is not None else na


def _gt_footprint_wh(row: dict, base_mpp: float) -> tuple:
    """(w, h) @ level-0 of the slide area the shot actually covers.

    The Camera rotates the read square about its centre before centre-cropping,
    so a 90/270 shot covers a footprint whose width and height are swapped
    relative to the FoV rect. Uses the ground-truth rot_deg, not the
    retriever's vote.
    """
    nominal = float(row['nominal_mpp'])
    w = int(row['fov_width'])  * nominal / base_mpp
    h = int(row['fov_height']) * nominal / base_mpp
    return (h, w) if int(row['rot_deg']) % 180 == 90 else (w, h)


def _theta_from_H(H) -> Optional[float]:
    """Rotation (deg) encoded in a homography's linear part."""
    if H is None:
        return None
    return float(math.degrees(math.atan2(float(H[1, 0]), float(H[0, 0]))))


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a - b, wrapped to (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def _pick_device(name: str):
    import torch
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def compute_metrics(row: dict, result: LocaScopeShotResult, base_mpp: float) -> dict:
    """Build one metrics.csv row from a gt-row + LocaScopeShotResult.

    Two families of position error are recorded:

      *_err_px      top-left based. Only meaningful when the shot was NOT
                    rotated: gt_x/gt_y is the pre-rotation top-left, whereas
                    the prediction is where the shot's own (0,0) landed, which
                    is a different corner for the 90/180/270 steps.
      *_center_err_px  centre based. Rotation-invariant — the FoV is rotated
                    about its own centre, so this is comparable for every
                    orientation. Prefer this one when reading results.
    """
    gt_x         = int(row['gt_x'])
    gt_y         = int(row['gt_y'])
    effective    = float(row['effective_mpp'])

    # GT centre @ level-0. The rect QFW read is fov_width x fov_height at the
    # NOMINAL mpp; rotation and the final centre-crop are both centred on it,
    # so its centre is the shot's centre regardless of rotation / scale.
    fov_w    = int(row['fov_width'])
    fov_h    = int(row['fov_height'])
    nominal  = float(row['nominal_mpp'])
    rect_w_l0 = fov_w * nominal / base_mpp
    rect_h_l0 = fov_h * nominal / base_mpp
    gt_cx = gt_x + rect_w_l0 / 2.0
    gt_cy = gt_y + rect_h_l0 / 2.0

    m: dict = {
        'filename':       row['filename'],
        'wsi_path':       row['wsi_path'],
        'level':          int(row['level']),
        'nominal_mpp':    float(row['nominal_mpp']),
        'effective_mpp':  effective,
        'gt_x':           gt_x,
        'gt_y':           gt_y,
        'gt_center_x':    round(gt_cx, 1),
        'gt_center_y':    round(gt_cy, 1),
        'est_mpp':        result.est_mpp,
        'routed_level':   result.routed_level,
        'unusable_level': result.unusable_level,
        'error':          result.error or '',
        'mpp_err_rel':    None,   # |est-eff| / eff
        'gt_rot_deg':     int(row['rot_deg']),
        'retr_rotation':  None,   # rotation the retriever voted for
        'rot_correct':    None,   # retr_rotation == gt_rot_deg
        'retr_x0':        None,
        'retr_y0':        None,
        'retr_score':     None,
        'retr_err_px':    None,
        'retr_err_um':    None,
        'retr_center_x':      None,
        'retr_center_y':      None,
        'retr_center_err_px': None,
        'retr_center_err_um': None,
        'refine_x0':      None,
        'refine_y0':      None,
        'refine_success': None,
        'refine_inliers': None,
        'refine_matches': None,
        'refine_err_px':  None,
        'refine_err_um':  None,
        'refine_center_err_px': None,
        'refine_center_err_um': None,
        # Orientation recovered by the homography itself — independent of the
        # retriever's 4-way vote, so it still scores when that vote is wrong.
        'gt_theta_deg':     round(int(row['rot_deg']) + float(row['angle_jitter']), 3),
        'sift_theta_deg':   None,
        'theta_err_deg':    None,
    }
    if result.est_mpp is not None:
        m['mpp_err_rel'] = abs(result.est_mpp - effective) / effective
    if result.retrieval is not None:
        r = result.retrieval
        m['retr_rotation'] = int(r.best_rotation)
        m['rot_correct']   = (int(r.best_rotation) == m['gt_rot_deg'])
        m['retr_x0']    = int(r.best_x0)
        m['retr_y0']    = int(r.best_y0)
        m['retr_score'] = float(r.best_score)
        d = _dist_px(r.best_x0, r.best_y0, gt_x, gt_y)
        m['retr_err_px'] = d
        m['retr_err_um'] = d * base_mpp
        # Retrieval centre: the matched window holds the ROTATED query, so the
        # footprint's width/height swap for the 90/270 steps.
        w_l0, h_l0 = ((rect_h_l0, rect_w_l0) if r.best_rotation in (90, 270)
                      else (rect_w_l0, rect_h_l0))
        rcx = r.best_x0 + w_l0 / 2.0
        rcy = r.best_y0 + h_l0 / 2.0
        dc = _dist_px(rcx, rcy, gt_cx, gt_cy)
        m['retr_center_x']      = round(rcx, 1)
        m['retr_center_y']      = round(rcy, 1)
        m['retr_center_err_px'] = dc
        m['retr_center_err_um'] = dc * base_mpp
    if result.refine is not None:
        rf = result.refine
        m['refine_x0']      = int(rf.x0)
        m['refine_y0']      = int(rf.y0)
        m['refine_success'] = bool(rf.success)
        m['refine_inliers'] = int(rf.inlier_count)
        m['refine_matches'] = int(rf.match_count)
        d = _dist_px(rf.x0, rf.y0, gt_x, gt_y)
        m['refine_err_px'] = d
        m['refine_err_um'] = d * base_mpp
        dc = _dist_px(rf.center_x0, rf.center_y0, gt_cx, gt_cy)
        m['refine_center_err_px'] = dc
        m['refine_center_err_um'] = dc * base_mpp
        theta = _theta_from_H(rf.H) if rf.success else None
        if theta is not None:
            m['sift_theta_deg'] = round(theta, 3)
            m['theta_err_deg']  = round(_angle_diff(theta, m['gt_theta_deg']), 3)
    return m


def draw_shot_figure(
    pl, row: dict, img: np.ndarray, result: LocaScopeShotResult,
    out_dir: str, zoom_pad: int = 4, metrics: Optional[dict] = None,
) -> Optional[str]:
    """Render the 4-panel localization diagnostic for one shot.

    Requires result to carry the diagnostic objects (run(..., keep_objects=True)).
    Returns the saved path, or None when the shot lacks retrieval/refine data.
    """
    if result.retrieval is None or result.refine is None or result.localizer is None:
        return None

    loc       = result.localizer
    retriever = result.retriever
    # Query grid dims come from the WINNING rotation's container
    qc_win = retriever.qc_by_rot[result.retrieval.best_rotation]
    q_rows, q_cols = qc_win.grid.grid_rows, qc_win.grid.grid_cols

    crop_img, crop_x0, crop_y0, crop_ds = read_zoom_crop(
        pl.wsi, result.retrieval, pl.tile_size, q_rows, q_cols, zoom_pad=zoom_pad,
    )

    fig, axes = plt.subplots(1, 4, figsize=(30, 7))
    draw_localization_row(
        axes,
        query_img     = img,
        query_kps     = loc.query_kps,
        wsi_crop      = loc.wsi_crop,
        crop_kps      = loc.crop_kps,
        good_matches  = loc.good_matches,
        retrieval     = result.retrieval,
        sift          = result.refine,
        gt_x          = int(row['gt_x']),
        gt_y          = int(row['gt_y']),
        base_mpp      = pl.base_mpp,
        tile_size     = pl.tile_size,
        crop_img      = crop_img, crop_x0 = crop_x0,
        crop_y0       = crop_y0,  crop_ds = crop_ds,
        zoom_pad      = zoom_pad,
        query_rows    = q_rows, query_cols = q_cols,
        crop_origin_x = loc.crop_origin_x or 0,
        crop_origin_y = loc.crop_origin_y or 0,
        gt_center     = ((metrics['gt_center_x'], metrics['gt_center_y'])
                         if metrics else None),
        retr_center   = ((metrics['retr_center_x'], metrics['retr_center_y'])
                         if metrics and metrics.get('retr_center_x') is not None
                         else None),
        gt_box_wh     = _gt_footprint_wh(row, pl.base_mpp),
    )
    fig.suptitle(
        f'{row["filename"]}   L{row["level"]} -> routed L{result.routed_level}   '
        f'gt_rot={row["rot_deg"]} deg  retr_rot={result.retrieval.best_rotation} deg   '
        f'est_mpp={result.est_mpp:.4f}  effective_mpp={float(row["effective_mpp"]):.4f}',
        fontsize=11,
    )
    fig.tight_layout()

    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, os.path.splitext(row['filename'])[0] + '_diag.png')
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ── output writers + plotting live in _locascope_plots (no torch, reusable by
#    plot_locascope_metrics.py to re-plot an existing metrics.csv offline) ─────


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-csv',     required=True)
    ap.add_argument('--images-dir', required=True)
    ap.add_argument('--out',        default=None,
                    help='Output dir. Default: result/<SLURM_JOB_NAME or BenchLocaScope>/')
    ap.add_argument('--limit',      type=int, default=None,
                    help='Process only the first N shots (debug).')
    ap.add_argument('--draw-figures', type=int, default=0, metavar='N',
                    help='Save the 4-panel localization diagnostic for the first '
                         'N shots into <out>/figures/. 0 = off, -1 = all.')
    ap.add_argument('--zoom-pad',   type=int, default=4,
                    help='Zoom-crop padding in tiles for the diagnostic panel.')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--device',     default='auto')
    ap.add_argument('--precision',  choices=['fp16', 'fp32'], default='fp16',
                    help='GigaPath autocast precision. fp16 is the validated '
                         'production setting (~5.5x faster, cos=0.99995, '
                         'top-5=0.99 vs fp32 — see TODO 2026-07-23 AccuracyV1). '
                         'Ignored on CPU.')
    args = ap.parse_args()

    out_dir = args.out or job_result_dir('BenchLocaScope')
    os.makedirs(out_dir, exist_ok=True)
    print(f'gt-csv     : {args.gt_csv}')
    print(f'images-dir : {args.images_dir}')
    print(f'out        : {out_dir}')

    import torch
    device = _pick_device(args.device)
    # autocast fp16 is CUDA-only; fall back silently on CPU
    dtype = (torch.float16
             if args.precision == 'fp16' and device.type == 'cuda'
             else torch.float32)
    print(f'device     : {device}')
    print(f'precision  : {str(dtype).replace("torch.", "")}'
          f'{"  (requested fp16, CPU -> fp32)" if args.precision == "fp16" and dtype is torch.float32 else ""}')
    print('Loading GigaPath model ...', flush=True)
    model   = gigapath_model(device)
    encoder = make_gigapath_encoder(model, device,
                                    batch_size=args.batch_size, dtype=dtype)

    with open(args.gt_csv) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    print(f'Shots      : {len(rows)}\n', flush=True)

    by_wsi: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_wsi[r['wsi_path']].append(r)

    all_metrics: List[dict] = []
    n_drawn = 0
    t_start = time.time()

    for wsi_path, wsi_rows in by_wsi.items():
        wsi_tag = os.path.splitext(os.path.basename(wsi_path))[0]
        print(f'== {wsi_tag}  n_shots={len(wsi_rows)}  path={wsi_path}', flush=True)
        try:
            pl = LocaScopePipeline(wsi_path, encoder).build()
        except Exception as e:
            print(f'  [pipeline build failed] {type(e).__name__}: {e}', flush=True)
            for row in wsi_rows:
                stub = LocaScopeShotResult(None, None, False, None, None,
                                           f'pipeline build failed: {type(e).__name__}: {e}')
                all_metrics.append(compute_metrics(row, stub, 1.0))
            continue

        print(f'  pipeline built (base_mpp={pl.base_mpp:.4f}  '
              f'mask_regions={len(pl.mask.tissue_regions)})', flush=True)

        for i, row in enumerate(wsi_rows, 1):
            img_path = os.path.join(args.images_dir, row['filename'])
            try:
                img = np.array(Image.open(img_path).convert('RGB'))
            except Exception as e:
                stub = LocaScopeShotResult(None, None, False, None, None,
                                           f'image read failed: {type(e).__name__}: {e}')
                m = compute_metrics(row, stub, pl.base_mpp)
                all_metrics.append(m)
                print(f'  [{i:4d}/{len(wsi_rows)}] {row["filename"]}  FAIL: image read', flush=True)
                continue

            want_fig = (args.draw_figures == -1
                        or n_drawn < args.draw_figures)
            t0 = time.time()
            result = pl.run(img, keep_objects=want_fig)
            dt = time.time() - t0
            m = compute_metrics(row, result, pl.base_mpp)
            all_metrics.append(m)

            if want_fig:
                try:
                    p = draw_shot_figure(pl, row, img, result, out_dir,
                                         zoom_pad=args.zoom_pad, metrics=m)
                    if p:
                        n_drawn += 1
                        print(f'      [fig] {p}', flush=True)
                except Exception as e:
                    print(f'      [fig failed] {type(e).__name__}: {e}', flush=True)

            print(f'  [{i:4d}/{len(wsi_rows)}] {row["filename"]:36s}  '
                  f'L={row["level"]:>2}  '
                  f'route=L{_fmt(m["routed_level"], "{:>1d}", "  ")}  '
                  f'mpp_err={_fmt(m["mpp_err_rel"], "{:.3f}")}  '
                  f'rot {m["gt_rot_deg"]:>3}->{_fmt(m["retr_rotation"], "{:>3d}", "  ?")}'
                  f'{"" if m["rot_correct"] else "*"}  '
                  f'retr_ctr={_fmt(m["retr_center_err_px"], "{:>7.0f}")}  '
                  f'refine_ctr={_fmt(m["refine_center_err_px"], "{:>7.0f}")}  '
                  f'({dt:.1f}s)'
                  + (f'\n      ERR: {m["error"]}' if m['error'] else ''),
                  flush=True)

    print(f'\nTotal wall time: {time.time() - t_start:.1f}s', flush=True)

    write_metrics_csv(all_metrics, os.path.join(out_dir, 'metrics.csv'))
    render_all(all_metrics, out_dir)
    print(f'\nRe-plot later without re-running the pipeline:\n'
          f'  python utilities/cli/plot_locascope_metrics.py '
          f'{os.path.join(out_dir, "metrics.csv")}')


if __name__ == '__main__':
    main()
