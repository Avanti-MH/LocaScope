#!/usr/bin/env python3
"""Is brute-force SIFT over a whole WSI fast enough, and does it find the photo?

Two questions, two modes, and a stopwatch in both.

  --gt-csv / --gt-xy   GOLDEN. The answer is known, so the run is scored: how
                       far the top hit is from the truth, and at what rank the
                       truth first appears among the top-k. This is the
                       "does it work" mode.

  (neither)            NO GOLDEN. Nothing to score against, so the top-k are
                       drawn instead -- each one warped back into the photo's
                       frame, blended and checkerboarded, next to a map of where
                       all k landed on the slide. This is the "look at it and
                       judge" mode, and it is the only one available for the
                       real Ki-67 photos, which have no recorded position.

Timing is broken into read / detect / match / ransac in both modes, because
"too slow" and "too slow at the wrong thing" call for different responses.

Start with --max-windows
------------------------
A full level-0 scan of one Ki-67 slide is ~53000 windows. Run a few hundred
first to get milliseconds-per-window, multiply, and decide whether the full run
is worth submitting. A capped run says nothing about accuracy -- the truth is
almost certainly outside the windows it looked at -- so cap for speed, uncap for
correctness.

A query_sim images/ folder is flat: MultiBatch1440 holds 2500 files spanning 7
slides and 3-4 levels each. Point this at the whole folder and it keeps only the
photos whose gt row names the slide and level being scanned, reporting what it
dropped. Passing the folder is therefore the correct thing to do, not a shortcut.

Usage:
    # does it work, against query_sim ground truth
    python utilities/cli/slide_win_sift.py \\
        result/MultiBatch1440/images \\
        /path/to/BRACS_1228.svs \\
        --gt-csv result/MultiBatch1440/gt.csv --level 0 --limit 3

    # how fast -- same thing, capped at 300 windows
    python utilities/cli/slide_win_sift.py \\
        result/MultiBatch1440/images \\
        /path/to/BRACS_1228.svs \\
        --gt-csv result/MultiBatch1440/gt.csv --level 0 --limit 1 \\
        --max-windows 300

    # no ground truth: one photo, one slide, draw the best 5 and look
    python utilities/cli/slide_win_sift.py \\
        /work/u26130998/datasets/Ki67/S1103037_ki67/1.bmp \\
        "/work/u26130998/datasets/Ki67/S1103037,G7E,110122.mrxs"

Outputs (into --out, default result/SlideWinSift):
    results.csv                one row per photo: hits, distances, timings
    <stem>_swsift.png          the top-k figure (no-golden mode, or --figures)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / 'utilities'))

from SlideWinSift import SlideWinSift          # noqa: E402
from _sift_plot   import match_img             # noqa: E402

Image.MAX_IMAGE_PIXELS = None

_PHOTO_EXT = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'}


# ── inputs ────────────────────────────────────────────────────────────────────

def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', p.name)]


def collect_photos(target: Path) -> list:
    """Every photo under target, natural-sorted. Subsetting happens later, after
    the ground truth has had its say about which ones belong to this slide."""
    if target.is_file():
        return [target]
    return sorted((p for p in target.iterdir()
                   if p.suffix.lower() in _PHOTO_EXT), key=_natural_key)


def load_gt(gt_csv: str) -> dict:
    """filename -> the raw gt.csv row. The centre needs the slide, so not here."""
    with open(gt_csv, newline='') as f:
        return {r['filename']: r for r in csv.DictReader(f)}


def filter_by_gt(photos: list, gt_map: dict, wsi: str, level: int) -> tuple:
    """Keep only the photos that belong to THIS slide at THIS level.

    A query_sim images/ directory holds every slide and every level in one flat
    folder -- 2500 files across 7 slides for MultiBatch1440. Two of them would
    be nonsense to run here and the filter exists to make that impossible rather
    than merely unlikely:

      wrong slide: the photo simply is not in this WSI, so the scan can only
                   report whatever noise scores highest.
      wrong level: the photo was rendered at its own level's mpp and nothing in
                   this tool rescales it. Against a 4x pyramid a one-level
                   mismatch hands SIFT a query at 4x the scale of everything it
                   is being compared against.

    Returns (kept, n_wrong_slide, n_wrong_level, n_no_gt).
    """
    tag = Path(wsi).name
    kept, wrong_slide, wrong_level, no_gt = [], 0, 0, 0
    for p in photos:
        row = gt_map.get(p.name)
        if row is None:
            no_gt += 1
        elif Path(row['wsi_path']).name != tag:
            wrong_slide += 1
        elif int(row['level']) != level:
            wrong_level += 1
        else:
            kept.append(p)
    return kept, wrong_slide, wrong_level, no_gt


def gt_center(row: dict, base_mpp: float) -> tuple:
    """(cx, cy) @ level-0 of the shot's centre.

    Copied from bench_locascope._gt_center rather than imported, because that
    module pulls in torch and the whole pipeline and this tool deliberately
    depends on neither.

    The rect query_sim read is fov_width x fov_height at the NOMINAL mpp, and
    both the rotation and the final centre-crop are centred on it, so its centre
    is the shot's centre whatever the orientation or the scale. gt_x/gt_y is a
    CORNER of the unrotated rect and is NOT comparable across rotations, which
    is why every distance below is centre-based.
    """
    nominal = float(row['nominal_mpp'])
    w = int(row['fov_width']) * nominal / base_mpp
    h = int(row['fov_height']) * nominal / base_mpp
    return (int(row['gt_x']) + w / 2.0, int(row['gt_y']) + h / 2.0)


# ── figure ────────────────────────────────────────────────────────────────────

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


def backdrop(sws: SlideWinSift, max_side: int = 2500) -> tuple:
    """(rgb, ds) of the scanned rectangle, small enough to draw on.

    Read from the deepest pyramid level that is still above max_side, so the
    cost is a thumbnail's rather than a level-0 read. ds is level-0 px per
    backdrop px, which is all a caller needs to place a point on it.
    """
    n = len(sws.wsi.level_dimensions)
    lv = n - 1
    for i in range(n):
        d = sws.wsi.level_downsamples[i]
        if max(sws.span_w / d, sws.span_h / d) <= max_side:
            lv = i
            break
    d = sws.wsi.level_downsamples[lv]
    w = max(1, int(sws.span_w / d))
    h = max(1, int(sws.span_h / d))
    rgb = sws.wsi.read_region_rgb((sws.origin_x, sws.origin_y), lv, (w, h))
    return rgb, d


def save_figure(img, res, sws, stem, wsi_tag, out_dir,
                bd=None, gt=None) -> str:
    """One figure: each of the top-k as a row, plus where they all landed.

    The photo is drawn once in a column of its own rather than once per row --
    it is the same image every time, and the comparison the rows exist for is
    photo-versus-warped, which reads better with the reference held still.
    """
    hits = [h for h in res.hits if h.H is not None and h.crop is not None]
    k = max(1, len(hits))
    if bd is None:
        bd = backdrop(sws)
    bd_img, bd_ds = bd

    fig = plt.figure(figsize=(30, 4.6 * k + 1.6))
    gs = fig.add_gridspec(k, 6, width_ratios=[1.3, 1.3, 1.3, 1.3, 2.6, 1.1])

    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img)
    ax.set_title(f'Photo  {stem}\n{img.shape[1]}x{img.shape[0]}')
    ax.axis('off')

    for r, h in enumerate(hits):
        H_inv = np.linalg.inv(h.H)
        warped = cv2.warpPerspective(h.crop, H_inv, (img.shape[1], img.shape[0]))
        for c, (pic, title) in enumerate((
                (h.crop,  f'#{r+1} window @L{res.level}\n'
                          f'({h.win_x0}, {h.win_y0}) level-0'),
                (warped,  'WSI warped into photo frame\n(should match the photo)'),
                (cv2.addWeighted(img, 0.5, warped, 0.5, 0),
                          'Blend 50/50\n(ghosting = misalignment)'),
                (_checkerboard(img, warped),
                          'Checkerboard\n(discontinuities = misalignment)'))):
            a = fig.add_subplot(gs[r, c])
            a.imshow(pic)
            a.set_title(title, fontsize=9)
            a.axis('off')
        a = fig.add_subplot(gs[r, 4])
        a.imshow(match_img(img, res.query_kps, h.crop, h.crop_kps, h.good_matches)
                 if h.crop_kps is not None and res.query_kps is not None
                 else h.crop)
        a.set_title(f'#{r+1}   {h.inlier_count}/{h.match_count} inliers   '
                    f'theta={h.theta_deg:+.1f} deg  scale={h.scale:.3f}\n'
                    f'centre @level-0 = ({h.center_x0}, {h.center_y0})', fontsize=9)
        a.axis('off')

    ax = fig.add_subplot(gs[:, 5])
    ax.imshow(bd_img)
    for r, h in enumerate(hits):
        bx = (h.center_x0 - sws.origin_x) / bd_ds
        by = (h.center_y0 - sws.origin_y) / bd_ds
        ax.plot(bx, by, '*', color='lime', ms=17, mec='black', mew=1.0)
        ax.annotate(str(r + 1), (bx, by), color='black', fontsize=11,
                    weight='bold', xytext=(6, 4), textcoords='offset points')
    if gt is not None:
        ax.plot((gt[0] - sws.origin_x) / bd_ds, (gt[1] - sws.origin_y) / bd_ds,
                'x', color='red', ms=16, mew=2.5)
    ax.set_title(f'{wsi_tag}\ntop-{len(hits)} landings'
                 + ('   red x = truth' if gt is not None else ''), fontsize=10)
    ax.axis('off')

    fig.suptitle(
        f'{stem} -> {wsi_tag}   SlideWinSift L{res.level} '
        f'(ds={res.ds:g}, plane {res.plane_w}x{res.plane_h})   '
        f'{res.timing_line()}',
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = os.path.join(out_dir, f'{stem}_swsift.png')
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo', help='a photo, or a folder of photos')
    ap.add_argument('wsi')
    ap.add_argument('--out', default='result/SlideWinSift')
    ap.add_argument('--level', type=int, default=0)
    ap.add_argument('--step-frac', type=float, default=0.5)
    ap.add_argument('--top', type=int, default=5)
    ap.add_argument('--min-inliers', type=int, default=10)
    ap.add_argument('--nms-frac', type=float, default=0.0,
                    help='0 = top-k are simply the k best windows (default); '
                         '>0 = collapse windows landing in the same place')
    ap.add_argument('--ratio', type=float, default=0.75)
    ap.add_argument('--ransac-thresh', type=float, default=5.0)
    ap.add_argument('--sift-nfeatures', type=int, default=0)
    ap.add_argument('--min-std', type=float, default=0.0,
                    help='skip SIFT on windows flatter than this (0 = off)')
    ap.add_argument('--sample-windows', type=int, default=0,
                    help='scan an evenly spaced sample of N windows instead of '
                         'all of them -- for timing, not for accuracy. Spread '
                         'across the slide, not the first N: the first N are '
                         'the blank top-left corner, where SIFT finds nothing '
                         'and so runs far faster than it will on tissue.')
    ap.add_argument('--stop-at-inliers', type=int, default=0)
    ap.add_argument('--no-limit-bounds', action='store_true')
    ap.add_argument('--gt-csv', default=None,
                    help='query_sim gt.csv -> golden mode')
    ap.add_argument('--gt-xy', type=float, nargs=2, default=None,
                    metavar=('CX', 'CY'),
                    help='level-0 centre of the truth, for a single photo')
    ap.add_argument('--tol-px', type=float, default=0.0,
                    help='level-0 distance within which a hit counts as '
                         'correct. 0 = half the query\'s short side expressed '
                         'at level-0, which keeps the verdict comparable '
                         'across levels: an L2 shot covers 16x the level-0 '
                         'area of an L0 shot, so a fixed px tolerance would '
                         'silently be four times stricter for it.')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--figures', action='store_true',
                    help='draw even in golden mode (always on without golden)')
    ap.add_argument('--progress-every', type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    photos = collect_photos(Path(args.photo))
    if not photos:
        sys.exit(f'no photos under {args.photo}')
    gt_map = load_gt(args.gt_csv) if args.gt_csv else {}
    golden = bool(gt_map) or args.gt_xy is not None
    draw = args.figures or not golden
    wsi_tag = Path(args.wsi).stem

    if gt_map:
        n_all = len(photos)
        photos, n_slide, n_level, n_missing = \
            filter_by_gt(photos, gt_map, args.wsi, args.level)
        print(f'ground truth: {n_all} file(s) seen  ->  {len(photos)} for '
              f'{Path(args.wsi).name} L{args.level}   '
              f'(dropped {n_slide} other slides, {n_level} other levels, '
              f'{n_missing} with no gt row)')
        if not photos:
            sys.exit(f'no photo in {args.photo} is from {Path(args.wsi).name} '
                     f'at level {args.level}')

    photos = photos[::max(1, args.stride)]
    if args.limit:
        photos = photos[:args.limit]

    print(f'SlideWinSift  {len(photos)} photo(s)  ->  {wsi_tag}')
    print(f'  level={args.level}  step_frac={args.step_frac}  '
          f'min_inliers={args.min_inliers}  top={args.top}  '
          f'limit_bounds={not args.no_limit_bounds}')
    print(f'  mode={"GOLDEN" if golden else "no golden, figures only"}'
          + (f'  (TIMING PROBE: {args.sample_windows} windows sampled across '
             f'the slide -- accuracy from this run is meaningless)'
             if args.sample_windows else ''), flush=True)

    sws = SlideWinSift(
        args.wsi, level=args.level, limit_bounds=not args.no_limit_bounds,
        step_frac=args.step_frac, min_inliers=args.min_inliers, top_k=args.top,
        ratio=args.ratio, ransac_thresh=args.ransac_thresh,
        nms_frac=args.nms_frac, sift_nfeatures=args.sift_nfeatures,
        min_std=args.min_std, stop_at_inliers=args.stop_at_inliers,
        sample_windows=args.sample_windows).build()

    xs, ys, ww, wh = sws.window_grid(*_probe_shape(photos[0]))
    print(f'  scan rect {sws.span_w}x{sws.span_h} @level-0 '
          f'-> {sws.plane_w}x{sws.plane_h} @L{sws.lv} (ds={sws.ds_lv:g})')
    print(f'  grid {len(xs)}x{len(ys)} = {len(xs)*len(ys)} windows of {ww}x{wh}',
          flush=True)

    # gt.csv records gt_x/gt_y in level-0 px of THIS slide, and the FoV size in
    # px at the nominal mpp, so turning the pair into a centre needs the slide's
    # own level-0 mpp -- which only exists after build().
    try:                       # SafeSlide.base_mpp: mean of mpp-x/y, one definition
        base_mpp = sws.wsi.base_mpp
    except RuntimeError:
        base_mpp = None
    if gt_map and base_mpp is None:
        sys.exit('slide carries no mpp metadata; cannot place the gt.csv centre')

    bd = backdrop(sws) if draw else None
    rows = []
    csv_path = os.path.join(args.out, 'results.csv')
    t_all = time.time()

    for i, p in enumerate(photos, 1):
        img = np.array(Image.open(p).convert('RGB'))
        stem = p.stem

        gt = None
        if args.gt_xy is not None:
            gt = (args.gt_xy[0], args.gt_xy[1])
        elif p.name in gt_map:
            gt = gt_center(gt_map[p.name], base_mpp)

        def _prog(done, total, best):
            b = f'best {best.inlier_count} inliers' if best else 'nothing yet'
            print(f'    [{stem}] {done}/{total} windows  {b}', flush=True)

        res = sws.run(img, progress=_prog, progress_every=args.progress_every)

        row = {
            'filename': p.name, 'wsi': wsi_tag, 'level': res.level,
            'plane_w': res.plane_w, 'plane_h': res.plane_h,
            'n_windows': res.n_windows, 'n_scanned': res.n_scanned,
            'n_skipped_flat': res.n_skipped_flat,
            'n_with_homography': res.n_with_homography,
            'n_hits_raw': res.n_hits_raw, 'n_hits_kept': len(res.hits),
            'query_kp_n': res.query_kp_n, 'capped': int(res.stopped_early),
            't_total_s': round(res.t_total_s, 2),
            't_read_s': round(res.t_read_s, 2),
            't_detect_s': round(res.t_detect_s, 2),
            't_match_s': round(res.t_match_s, 2),
            't_ransac_s': round(res.t_ransac_s, 2),
            'ms_per_window': round(res.t_total_s / res.n_scanned * 1000, 1)
                             if res.n_scanned else None,
        }
        for r in range(args.top):
            h = res.hits[r] if r < len(res.hits) else None
            row[f'top{r+1}_inliers'] = h.inlier_count if h else None
            row[f'top{r+1}_cx'] = h.center_x0 if h else None
            row[f'top{r+1}_cy'] = h.center_y0 if h else None
            row[f'top{r+1}_theta'] = round(h.theta_deg, 2) if h else None
            row[f'top{r+1}_dist_px'] = (
                round(math.hypot(h.center_x0 - gt[0], h.center_y0 - gt[1]), 1)
                if h and gt else None)
        if gt:
            tol = args.tol_px or 0.5 * min(img.shape[:2]) * sws.ds_lv
            row['gt_cx'], row['gt_cy'] = round(gt[0], 1), round(gt[1], 1)
            row['tol_px'] = round(tol, 1)
            row['hit_rank'] = next(
                (r + 1 for r, h in enumerate(res.hits)
                 if math.hypot(h.center_x0 - gt[0], h.center_y0 - gt[1]) <= tol),
                None)
        rows.append(row)

        b = res.best
        verdict = ''
        if gt:
            d = row['top1_dist_px']
            verdict = (f'   top1_dist={d} px  hit@'
                       f'{row["hit_rank"] if row["hit_rank"] else "-"}')
        print(f'  [{i}/{len(photos)}] {p.name}  '
              f'{("best %d/%d inliers @ (%d, %d)" % (b.inlier_count, b.match_count, b.center_x0, b.center_y0)) if b else "NO HIT"}'
              f'{verdict}\n      {res.timing_line()}', flush=True)

        if draw and res.hits:
            print(f'      [fig] {save_figure(img, res, sws, stem, wsi_tag, args.out, bd, gt)}',
                  flush=True)

        fields = sorted({k for r_ in rows for k in r_}, key=_field_order)
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    sws.close()
    _summary(rows, golden, time.time() - t_all)
    print(f'\n-> {csv_path}')


def _field_order(k: str) -> tuple:
    head = ['filename', 'wsi', 'level', 'plane_w', 'plane_h', 'n_windows',
            'n_scanned', 'n_skipped_flat', 'n_with_homography', 'n_hits_raw',
            'n_hits_kept', 'query_kp_n', 'capped', 'gt_cx', 'gt_cy', 'tol_px',
            'hit_rank', 't_total_s', 't_read_s', 't_detect_s', 't_match_s',
            't_ransac_s', 'ms_per_window']
    return (head.index(k), '') if k in head else (len(head), k)


def _probe_shape(p: Path) -> tuple:
    with Image.open(p) as im:
        return im.size          # (w, h) -- window_grid takes (qw, qh)


def _summary(rows: list, golden: bool, wall: float) -> None:
    print(f'\n=== {len(rows)} photo(s), {wall:.1f}s wall ===')
    ms = [r['ms_per_window'] for r in rows if r['ms_per_window']]
    if ms:
        print(f'  per window : {np.median(ms):.1f} ms (median)')
        n = rows[0]['n_windows']
        print(f'  a full scan: {n} windows -> '
              f'{n * np.median(ms) / 1000 / 60:.1f} min per photo')
    for k in ('t_read_s', 't_detect_s', 't_match_s', 't_ransac_s'):
        tot = sum(r[k] for r in rows)
        all_t = sum(r['t_total_s'] for r in rows) or 1
        print(f'  {k:12s} {tot:8.1f}s  ({tot / all_t * 100:4.1f}%)')
    hit = [r for r in rows if r.get('top1_inliers')]
    print(f'  found something : {len(hit)}/{len(rows)}')
    if golden:
        ok1 = sum(1 for r in rows if r.get('hit_rank') == 1)
        okk = sum(1 for r in rows if r.get('hit_rank'))
        print(f'  top-1 correct   : {ok1}/{len(rows)}')
        print(f'  correct in top-k: {okk}/{len(rows)}')
        d = [r['top1_dist_px'] for r in rows if r.get('top1_dist_px') is not None]
        if d:
            print(f'  top-1 dist px   : p50={np.median(d):.0f}  '
                  f'p90={np.percentile(d, 90):.0f}')
    if rows and rows[0]['capped']:
        print('  NOTE: --sample-windows was on. The timing is real; the '
              'accuracy is not -- most of the slide was stepped over.')


if __name__ == '__main__':
    main()
