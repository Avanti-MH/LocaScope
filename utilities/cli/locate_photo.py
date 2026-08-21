#!/usr/bin/env python3
"""Locate real microscope photos in their WSI -- one photo, or a whole folder.

There is no recorded (x, y) for these photos. The sidecar .json files next to
them hold Ki-67 nucleus polygons in PHOTO pixel coordinates, not slide
positions, so nothing here can be scored against a ground truth. Correctness is
judged two ways instead:

  1. SIFT+RANSAC inlier count. A correct match on tissue yields hundreds to
     thousands of geometrically consistent keypoints; a wrong one yields single
     digits.
  2. Visual: the homography is inverted to warp the WSI back into the photo's
     own frame. If the localisation is right the two images overlap
     pixel-for-pixel, which the blend and checkerboard panels make obvious.

Why a folder mode exists
------------------------
LocaScopePipeline.build() segments the whole slide and encodes a KNN reference
bank, and the first shot routed to a pyramid level makes that level's retriever
encode every tile of the slide. All of it is per-WSI, none of it is per-photo.
Running one process per photo therefore repeats the entire expensive part once
per photo -- with 170 to 478 photos per slide that is roughly two orders of
magnitude of wasted work. Folder mode builds the pipeline once and loops.

Outputs, per invocation
-----------------------
    <out>/predictions.csv      one row per photo -- the actual deliverable
    <out>/_overview.png        every prediction on the slide, coloured by inliers
    <out>/success/             per-photo figures for the photos SIFT located
    <out>/fail/                per-photo figures for the ones it did not
        <stem>_locate.png      5-panel per-photo figure  (see --figures)
        <stem>_on_slide.png    that photo's predicted point on the slide

The success/fail split is by the SIFT verdict, not by the confidence label, so
the failure folder is browsable on its own -- which is the point, since there is
no ground truth for these photos and the figures are the only way to see why a
photo failed.

Usage:
    # one photo
    python utilities/cli/locate_photo.py \\
        /work/u26130998/datasets/Ki67/S1104360_ki67/1.bmp \\
        /work/u26130998/datasets/Ki67/S1104360,G7E,110208.mrxs

    # every photo of one slide
    python utilities/cli/locate_photo.py \\
        /work/u26130998/datasets/Ki67/S1104360_ki67 \\
        /work/u26130998/datasets/Ki67/S1104360,G7E,110208.mrxs \\
        --out result/RealTest/S1104360_ki67
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import openslide
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / 'utilities'))
from _paths import job_result_dir                                   # noqa: E402
sys.path.insert(0, str(_ROOT / 'aiNNModel'))

from _sift_plot        import match_img                              # noqa: E402
from LocaScopePipeline import LocaScopePipeline                      # noqa: E402
from GigaPathFunc      import GigaPathEncoderConfig                  # noqa: E402
from TissueSegFunc     import HestSegConfig                          # noqa: E402


PHOTO_EXTS = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')


# ── result schema ─────────────────────────────────────────────────────────────
#
# One row per photo. Column order is the reading order: what it is, then the
# answer, then each stage's evidence for that answer, then run bookkeeping.
#
# The raw `inliers` / `matches` / `retr_margin` columns are kept even though
# `confidence` summarises them, because the confidence thresholds below are a
# guess -- with no ground truth they cannot be calibrated yet. Keeping the raw
# numbers means reclassifying later is a pandas one-liner, not a 2296-photo
# rerun.
FIELDS = [
    # identity
    'slide', 'photo', 'photo_w', 'photo_h',
    # the answer
    'status', 'confidence', 'pred_source',
    'pred_x0', 'pred_y0', 'pred_um_x', 'pred_um_y',
    # stage 1 -- mpp estimation and level routing
    'est_mpp', 'routed_level', 'level_mpp', 'ds', 'unusable_level',
    # stage 2 -- coarse retrieval
    'retr_x0', 'retr_y0', 'retr_cx0', 'retr_cy0',
    'retr_score', 'retr_rot', 'retr_margin', 'retr_region', 'retr_from_overlap',
    'score_rot0', 'score_rot90', 'score_rot180', 'score_rot270',
    # stage 3 -- SIFT+RANSAC
    'sift_x0', 'sift_y0', 'sift_cx0', 'sift_cy0',
    'sift_success', 'inliers', 'matches', 'inlier_ratio',
    # bookkeeping
    't_total_s', 'retriever_built', 'reopens', 'error',
]


def classify(sift_success: bool, inliers: int) -> str:
    """Self-assessed quality, in the absence of any ground truth.

    Thresholds are provisional. They come from the observed split between a
    correct match (hundreds of inliers) and a wrong one (single digits), not
    from a calibration run -- see the note on FIELDS.
    """
    if not sift_success:
        return 'failed'
    if inliers >= 100:
        return 'high'
    if inliers >= 30:
        return 'medium'
    return 'low'


def _natural_key(p: Path):
    """Sort 2.bmp before 10.bmp, and keep prefixed names grouped."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', p.name)]


def collect_photos(target: Path, limit: int, stride: int) -> list[Path]:
    """One file, or every image in a folder."""
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise SystemExit(f'no such photo or folder: {target}')
    photos = sorted(
        (p for p in target.iterdir()
         if p.is_file() and p.suffix.lower() in PHOTO_EXTS),
        key=_natural_key,
    )
    if stride > 1:
        photos = photos[::stride]
    if limit > 0:
        photos = photos[:limit]
    return photos


def _checkerboard(a: np.ndarray, b: np.ndarray, n: int = 8) -> np.ndarray:
    """Interleave two same-size images in an n x n checker pattern."""
    out = a.copy()
    h, w = a.shape[:2]
    ch, cw = h // n, w // n
    for i in range(n):
        for j in range(n):
            if (i + j) % 2:
                out[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw] = \
                    b[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
    return out


# ── the per-photo row ─────────────────────────────────────────────────────────

def build_row(slide_tag, photo_path, img, res, base_mpp, tile_size,
              t_total, retriever_built, reopens, level_mpp):
    """Flatten a LocaScopeQueryResult into one CSV row."""
    row = {k: '' for k in FIELDS}
    row.update(
        slide=slide_tag, photo=photo_path.name,
        photo_w=img.shape[1], photo_h=img.shape[0],
        status='error' if res.error else 'ok',
        error=res.error or '',
        est_mpp=res.est_mpp, routed_level=res.routed_level,
        level_mpp=level_mpp, unusable_level=int(bool(res.unusable_level)),
        t_total_s=round(t_total, 2),
        retriever_built=int(bool(retriever_built)),
        reopens=reopens,
        confidence='failed', pred_source='none',
    )

    pred_x0 = pred_y0 = None

    r = res.retrieval
    if r is not None:
        # The query footprint in level-0 px, so the retrieval top-left can be
        # turned into a centre. extract_all's main grid is floor(size/tile)
        # tiles, and at 90/270 degrees the query was rotated before matching,
        # so its footprint in WSI space is transposed.
        cols = img.shape[1] // tile_size
        rows_ = img.shape[0] // tile_size
        if r.best_rotation in (90, 270):
            cols, rows_ = rows_, cols
        retr_cx0 = int(r.best_x0 + cols * tile_size * r.ds / 2)
        retr_cy0 = int(r.best_y0 + rows_ * tile_size * r.ds / 2)

        sc = dict(r.scores_by_rotation)
        ordered = sorted(sc.values(), reverse=True)
        # Best minus runner-up across rotations. With no ground truth this is
        # the only confidence signal stage 2 produces on its own: a flat
        # profile means the retriever could not tell the orientations apart.
        margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else float('nan')

        row.update(
            ds=r.ds,
            retr_x0=r.best_x0, retr_y0=r.best_y0,
            retr_cx0=retr_cx0, retr_cy0=retr_cy0,
            retr_score=round(float(r.best_score), 6),
            retr_rot=r.best_rotation,
            retr_margin=round(float(margin), 6),
            retr_region=r.best_region_index,
            retr_from_overlap=int(bool(r.from_overlap)),
            score_rot0=round(float(sc.get(0, float('nan'))), 6),
            score_rot90=round(float(sc.get(90, float('nan'))), 6),
            score_rot180=round(float(sc.get(180, float('nan'))), 6),
            score_rot270=round(float(sc.get(270, float('nan'))), 6),
        )
        pred_x0, pred_y0 = retr_cx0, retr_cy0
        row['pred_source'] = 'retrieval'

    s = res.refine
    if s is not None:
        row.update(
            sift_x0=s.x0, sift_y0=s.y0,
            sift_cx0=s.center_x0, sift_cy0=s.center_y0,
            sift_success=int(bool(s.success)),
            inliers=s.inlier_count, matches=s.match_count,
            inlier_ratio=(round(s.inlier_count / s.match_count, 4)
                          if s.match_count else 0.0),
        )
        row['confidence'] = classify(s.success, s.inlier_count)
        if s.success:
            # The CENTRE, not the top-left. The query is rotated about its own
            # centre, so the centre is the only anchor that stays comparable
            # across the 4 orientations -- a corner is a different corner of
            # the same footprint once rotated.
            pred_x0, pred_y0 = s.center_x0, s.center_y0
            row['pred_source'] = 'sift'

    if pred_x0 is not None:
        row.update(
            pred_x0=int(pred_x0), pred_y0=int(pred_y0),
            pred_um_x=round(pred_x0 * base_mpp, 2),
            pred_um_y=round(pred_y0 * base_mpp, 2),
        )
    return row


# ── figures ───────────────────────────────────────────────────────────────────

def save_shot_figure(img, res, stem, wsi_tag, out_dir) -> str | None:
    """5 panels: photo, WSI warped into the photo frame, blend, checker, matches."""
    if res.refine is None or res.retrieval is None or res.localizer is None:
        return None
    loc = res.localizer
    H = res.refine.H

    fig, axes = plt.subplots(1, 5, figsize=(30, 6.5))

    axes[0].imshow(img)
    axes[0].set_title(f'Photo\n{stem}  {img.shape[1]}x{img.shape[0]}')

    if H is not None and loc.wsi_crop is not None:
        # Invert H (query px -> crop px) to pull the WSI into the photo frame.
        # A correct localisation makes this panel a copy of the photo.
        H_inv = np.linalg.inv(H)
        warped = cv2.warpPerspective(loc.wsi_crop, H_inv,
                                     (img.shape[1], img.shape[0]))
        axes[1].imshow(warped)
        axes[1].set_title('WSI warped into photo frame\n(should match panel 1)')

        axes[2].imshow(cv2.addWeighted(img, 0.5, warped, 0.5, 0))
        axes[2].set_title('Blend 50/50\n(ghosting = misalignment)')

        axes[3].imshow(_checkerboard(img, warped))
        axes[3].set_title('Checkerboard\n(discontinuities = misalignment)')
    else:
        for i, t in ((1, 'no homography'), (2, ''), (3, '')):
            axes[i].set_title(t)

    if loc.wsi_crop is not None and loc.good_matches:
        axes[4].imshow(match_img(img, loc.query_kps, loc.wsi_crop,
                                 loc.crop_kps, loc.good_matches))
        axes[4].set_title(f'SIFT matches\n{res.refine.match_count} good  '
                          f'{res.refine.inlier_count} inliers')

    for ax in axes:
        ax.axis('off')

    s = res.refine
    fig.suptitle(
        f'{stem}  ->  {wsi_tag}    '
        f'est_mpp={res.est_mpp:.4f} (L{res.routed_level})   '
        f'retr score={res.retrieval.best_score:.4f} '
        f'rot={res.retrieval.best_rotation}deg   '
        f'SIFT {s.inlier_count}/{s.match_count} inliers   '
        f'predicted centre @ level-0 = ({s.center_x0}, {s.center_y0})',
        fontsize=12,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, f'{stem}_locate.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def save_on_slide_figure(pl, backdrop, res, stem, wsi_tag, out_dir) -> str | None:
    """One photo's predicted point on the slide."""
    if res.refine is None:
        return None
    s = res.refine
    Ht, Wt = pl.mask.main_mask.shape
    fig, ax = plt.subplots(figsize=(7, 7 * Ht / max(Wt, 1)))
    ax.imshow(backdrop)
    for r in pl.mask.tissue_regions:
        mx, my, mw, mh = pl.mask.region_box(r)
        ax.add_patch(mpatches.Rectangle((mx, my), mw, mh, fill=False,
                                        edgecolor='red', lw=0.6, alpha=0.5))
    cx, cy = pl.mask.to_mask_xy(s.center_x0, s.center_y0)
    ax.plot(cx, cy, '*', color='lime', ms=20, mec='black', mew=1.0)
    ax.set_title(f'{wsi_tag}\npredicted location of {stem}')
    ax.axis('off')
    fig.tight_layout()
    path = os.path.join(out_dir, f'{stem}_on_slide.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def save_overview(pl, backdrop, rows, wsi_tag, out_dir) -> str:
    """Every prediction on one slide -- the 'did this slide work at all' view.

    Points are coloured by inlier count, so a slide that localised well shows a
    scatter of bright points sitting on tissue, and a slide that did not shows
    dark points piled on whatever the retriever's favourite region was.
    """
    Ht, Wt = pl.mask.main_mask.shape
    fig, ax = plt.subplots(figsize=(9, 9 * Ht / max(Wt, 1)))
    ax.imshow(backdrop)
    for r in pl.mask.tissue_regions:
        mx, my, mw, mh = pl.mask.region_box(r)
        ax.add_patch(mpatches.Rectangle((mx, my), mw, mh, fill=False,
                                        edgecolor='red', lw=0.6, alpha=0.4))

    good_x, good_y, good_c = [], [], []
    bad_x, bad_y = [], []
    for row in rows:
        if row['pred_x0'] == '' or row['pred_x0'] is None:
            continue
        mx, my = pl.mask.to_mask_xy(int(row['pred_x0']), int(row['pred_y0']))
        if row['pred_source'] == 'sift':
            good_x.append(mx)
            good_y.append(my)
            good_c.append(int(row['inliers'] or 0))
        else:
            bad_x.append(mx)
            bad_y.append(my)

    if bad_x:
        ax.scatter(bad_x, bad_y, marker='x', c='crimson', s=28, linewidths=1.0,
                   label=f'SIFT failed, retrieval only ({len(bad_x)})')
    if good_x:
        sc = ax.scatter(good_x, good_y, c=good_c, cmap='viridis', s=26,
                        edgecolors='black', linewidths=0.3,
                        label=f'SIFT located ({len(good_x)})')
        fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label='inliers')

    n = len(rows)
    n_ok = len(good_x)
    ax.set_title(f'{wsi_tag}\n{n_ok}/{n} photos localised by SIFT')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
    ax.axis('off')
    fig.tight_layout()
    path = os.path.join(out_dir, '_overview.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo', help='A photo file, or a folder of photos')
    ap.add_argument('wsi',   help='The WSI the photos were taken from')
    ap.add_argument('--out',        default='',
                    help='output directory. Empty means result/<SLURM_JOB_NAME or LocatePhoto>/, via _paths.job_result_dir -- results live outside the checkout')
    ap.add_argument('--csv',        default='predictions.csv')
    ap.add_argument('--precision',  choices=['fp16', 'fp32'], default='fp16')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--device',     default='auto')
    ap.add_argument('--limit',  type=int, default=0,
                    help='max photos from a folder (0 = all)')
    ap.add_argument('--stride', type=int, default=1,
                    help='take every Nth photo')
    ap.add_argument('--figures', choices=['none', 'fail', 'sample', 'all'],
                    default='sample',
                    help='sample = first --figure-limit photos plus every '
                         'failure; all = one pair of figures per photo. '
                         'Figures land in <out>/success/ or <out>/fail/ by the '
                         'SIFT verdict. Budget ~2.3 MB per photo -- "all" over '
                         'the whole Ki67 set is roughly 5 GB')
    ap.add_argument('--figure-limit', type=int, default=5)
    ap.add_argument('--no-resume', action='store_true',
                    help='rewrite the CSV instead of skipping photos already in it')
    args = ap.parse_args()

    # job_result_dir honours SLURM_JOB_NAME, so a job's output lands under
    # result/<job>/ without the jobscript spelling the path twice.
    #
    # The makedirs is NOT redundant with the one inside job_result_dir: `or`
    # short-circuits, so an explicit --out never reaches it and nothing would
    # create that directory. That cost a run -- --out pointed at a fresh
    # result/RealTest/<tag>/ and the FileNotFoundError landed on the first CSV
    # write, after GigaPath had loaded and the mask + MPP bank were built.
    args.out = args.out or job_result_dir('LocatePhoto')
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, args.csv)
    wsi_tag = os.path.splitext(os.path.basename(args.wsi))[0]

    photos = collect_photos(Path(args.photo), args.limit, args.stride)
    if not photos:
        print(f'[skip] no images under {args.photo}')
        return 0

    # Resume: a full folder is hours of GPU time, and a requeue should not
    # start over. Rows already written are trusted and skipped.
    done: set[str] = set()
    if not args.no_resume and os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            done = {r['photo'] for r in csv.DictReader(f) if r.get('photo')}
    todo = [p for p in photos if p.name not in done]

    print(f'wsi    : {args.wsi}')
    print(f'photos : {len(photos)} found, {len(done)} already done, '
          f'{len(todo)} to run')
    print(f'out    : {args.out}')
    if not todo:
        print('nothing to do')
        return 0

    import torch
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    dtype = (torch.float16
             if args.precision == 'fp16' and device.type == 'cuda'
             else torch.float32)
    print(f'device : {device}  precision={str(dtype).replace("torch.", "")}')

    print('\nLoading GigaPath ...', flush=True)
    # From the resolved dtype, not args.precision: the rule above already
    # demoted fp16 to fp32 on CPU.
    encoder  = GigaPathEncoderConfig(batch_size=args.batch_size)\
        .with_model(dtype='fp16' if dtype is torch.float16 else 'fp32')\
        .build(device)
    from TissueMaskConfig import TissueMaskConfig
    mask_cfg = TissueMaskConfig(seg=HestSegConfig(), ds=4.0)

    # Built ONCE for the whole folder -- see the module docstring.
    print('Building pipeline (mask + mpp reference bank) ...', flush=True)
    t0 = time.perf_counter()
    pl = LocaScopePipeline(args.wsi,
                           encoder,
                           tile_size=256,
                           mask_cfg=mask_cfg,
                           knn_samples=100,
                           knn_k=5,
                           retriever_overlap=True,
                           refiner_min_inliers=10,
                           refiner_padding=2).build()
    print(f'  base_mpp={pl.base_mpp:.4f}  levels={pl.wsi.level_count}  '
          f'mask_regions={len(pl.mask.tissue_regions)}  '
          f'({time.perf_counter() - t0:.1f}s)', flush=True)

    # One backdrop read for every figure of this slide.
    backdrop = pl.mask.read_matching_rgb(pl.wsi)

    # Existence is not the same question as "has a header". writeheader() only
    # buffers, and the first flush happens after photo 1 -- which can take
    # minutes because it builds that level's retriever. A run killed inside that
    # window leaves a file that exists and is empty; the next run then appended
    # headerless rows, and csv.DictReader read data row 1 as the field names.
    # Test the size, and flush the header immediately so the window closes.
    write_header = (args.no_resume
                    or not os.path.exists(csv_path)
                    or os.path.getsize(csv_path) == 0)
    mode = 'w' if args.no_resume else 'a'
    fh = open(csv_path, mode, newline='')
    writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
    if write_header:
        writer.writeheader()
        fh.flush()

    rows: list[dict] = []
    seen_levels: set[int] = set()
    prev_reopens = getattr(pl.wsi, 'reopens', 0)
    t_start = time.perf_counter()

    for i, photo_path in enumerate(todo, 1):
        stem = photo_path.stem
        try:
            img = np.array(Image.open(photo_path).convert('RGB'))
        except Exception as e:
            print(f'[{i}/{len(todo)}] {photo_path.name}: unreadable: {e}',
                  flush=True)
            continue

        t_shot = time.perf_counter()
        try:
            # keep_objects is always on: the heavy object it retains is the
            # retriever, which is the pipeline's own cached one, and the rest is
            # a few MB that goes away when `res` is dropped at the end of the
            # loop. Deciding it up front would mean knowing whether this photo
            # is going to fail, which is exactly what --figures fail needs.
            res = pl.run(img, keep_objects=True)
        except Exception as e:
            # One unlucky photo must not take the other 477 with it.
            traceback.print_exc()
            row = {k: '' for k in FIELDS}
            row.update(slide=wsi_tag, photo=photo_path.name,
                       photo_w=img.shape[1], photo_h=img.shape[0],
                       status='crash', confidence='failed', pred_source='none',
                       error=f'{type(e).__name__}: {e}',
                       t_total_s=round(time.perf_counter() - t_shot, 2))
            writer.writerow(row)
            fh.flush()
            rows.append(row)
            continue
        t_total = time.perf_counter() - t_shot

        built = res.routed_level is not None and res.routed_level not in seen_levels
        if res.routed_level is not None:
            seen_levels.add(res.routed_level)
        level_mpp = ('' if res.routed_level is None else
                     round(pl.base_mpp * pl.wsi.level_downsamples[res.routed_level], 4))
        reopens = getattr(pl.wsi, 'reopens', 0)
        d_reopens, prev_reopens = reopens - prev_reopens, reopens

        row = build_row(wsi_tag, photo_path, img, res, pl.base_mpp, pl.tile_size,
                        t_total, built, d_reopens, level_mpp)
        writer.writerow(row)
        fh.flush()          # a killed job keeps every row it finished
        rows.append(row)

        print(f'[{i}/{len(todo)}] {photo_path.name:24s} '
              f'L{row["routed_level"]} '
              f'conf={row["confidence"]:7s} '
              f'inliers={row["inliers"] or "-":>5} '
              f'pred=({row["pred_x0"] or "-"}, {row["pred_y0"] or "-"}) '
              f'{t_total:.1f}s'
              + (f'  ERR {row["error"]}' if row['status'] != 'ok' else ''),
              flush=True)

        failed = row['status'] != 'ok' or not row['sift_success']
        want = (args.figures == 'all'
                or (args.figures == 'fail' and failed)
                or (args.figures == 'sample'
                    and (i <= args.figure_limit or failed)))
        if want:
            # Split by the SIFT verdict rather than by the confidence label. The
            # inlier count is near-binary on this corpus -- a correct match runs
            # to hundreds, a wrong one to single digits -- so these two folders
            # really are "located" and "did not", and the failure folder can be
            # flipped through on its own looking for what the failures share.
            fig_dir = os.path.join(args.out, 'fail' if failed else 'success')
            os.makedirs(fig_dir, exist_ok=True)
            save_shot_figure(img, res, stem, wsi_tag, fig_dir)
            save_on_slide_figure(pl, backdrop, res, stem, wsi_tag, fig_dir)

        del res

    fh.close()

    # The overview wants every row for this slide, including ones a previous
    # run wrote, so read the CSV back rather than using only this run's rows.
    with open(csv_path, newline='') as f:
        all_rows = list(csv.DictReader(f))
    if all_rows:
        print(f'\nSaved -> {save_overview(pl, backdrop, all_rows, wsi_tag, args.out)}')

    # ── summary ──────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    tally = {}
    for r in all_rows:
        tally[r['confidence']] = tally.get(r['confidence'], 0) + 1
    print(f'\n--- {wsi_tag} ---')
    print(f'photos     : {len(all_rows)}')
    for k in ('high', 'medium', 'low', 'failed'):
        if k in tally:
            print(f'  {k:8s} : {tally[k]:4d}  ({100 * tally[k] / len(all_rows):.1f}%)')
    print(f'time       : {elapsed / 60:.1f} min for {len(todo)} photos '
          f'({elapsed / max(len(todo), 1):.1f} s/photo)')
    if getattr(pl.wsi, 'holes', None):
        print(f'slide holes: {pl.wsi.hole_summary()}')
    print(f'csv        : {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
