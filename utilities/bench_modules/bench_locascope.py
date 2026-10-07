#!/usr/bin/env python3
"""End-to-end LocaScope bench: render synthetic shots with known positions,
run each through LocaScopePipeline, collect per-stage errors, and plot them.

Inputs: `split_shots` below -- the first --n-wsi slides of each of
--datasets' recorded --split, --per-level FoVs at every native level from a
TileSampler draw (placed on the hest masks of --fov-mask-cache-job),
each photographed on demand through `FovSupply` (TileSampler -> Render).
Nothing is read from a stored corpus, which would go stale whenever the camera
changed. A shot's `filename` is an id.

Outputs (in --out DIR/<encoder>/, default result/BenchLocaScope/<encoder>/):
    metrics.csv               per-shot: mpp/retrieval/refine errors (px + um)
    summary.txt               aggregate stats + failure counts
    stage1_mpp_cdf.png        Stage 1 CDF (aggregate + per-WSI subplots)
    stage2_retr_cdf.png       Stage 2 CDF
    stage3_refine_cdf.png     Stage 3 CDF
    heatmap.png               (WSI, level) x 3 stages, median error, column-normalized colour
    recall_at_k.png           Stage 2 recall@K, overall and per routed level

Usage:
    python utilities/bench_modules/bench_locascope.py \\
        --datasets bracs/test ki67_with_photo --split test --n-wsi 5 \\
        --sampler-n-per-rung 50 --out result/BenchLocaScope \\
        [--limit N] [--batch-size 128] [--device auto] \\
        [--topk 20] [--sift-topk 5]

--topk is free: it reads similarity scores find_best already computed and
discarded. --sift-topk is not: it is one SIFT pass per candidate per shot, and
it is what measures whether a retrieval-proposes / SIFT-verifies loop would
work without ground truth to lean on.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # utilities/
import _paths                                                       # noqa: E402
_paths.setup_import_paths()

from _paths            import encoder_tag, job_result_dir               # noqa: E402
import Cache                                                           # noqa: E402
from dump_function._sift_plot import (draw_localization_row,            # noqa: E402
                                      draw_recall_row, read_anchored_crop,
                                      read_zoom_crop)
from dump_function._locascope_plots import (append_metrics_row,         # noqa: E402
                                            load_metrics_csv, render_all)
from LocaScopePipeline import LocaScopePipeline, LocaScopeQueryResult    # noqa: E402
from stage1_estimation.KnnEstMpp import knn_estimator                     # noqa: E402
from stage2_retrieval.SlidingWinSimRot import (                   # noqa: E402
    SlidingWinSimRot, SlidingWinSimRotConfig)
from TissueMaskConfig import add_mask_args, mask_cfg_from_args           # noqa: E402
from stage3_localization.SIFT_RANSAC       import SiftRansacLocalizer                       # noqa: E402
from TileEncoderFunc   import encoder_config, encoder_names             # noqa: E402
from CpuBudget         import CpuBudget                                 # noqa: E402
from TileSampler       import PlanSpec                              # noqa: E402
from TissueMaskConfig  import MASK_RECIPES, MaskMaker                   # noqa: E402
from AccessDatasets    import list_names, locate                        # noqa: E402
from SlideReader       import SlideReader                               # noqa: E402
from camera            import Render                                    # noqa: E402
from record            import FOVRecord                                 # noqa: E402
from FovSupply         import (FovRecipe, FovSupply, add_fov_args,  # noqa: E402
                               fov_from_args)


# ── the shots: one FovSupply per slide, every level in one draw ─────────────

def split_shots(datasets, split: str, n_wsi: int, fov: FovRecipe, *, masks,
                skip=None):
    """`(row, image)` per synthetic FoV, rendered on demand: `row` is the
    FOVRecord of its ground truth, `image` uint8 RGB.

    SLIDES: the first `n_wsi` of each dataset's recorded `split`
    (`<dataset>#<split>`), the slides Stage1MppBench and OffGridScore score,
    every one with a cached mask in `masks` (a MaskMaker). LEVELS: the
    recipe's (`fov.rungs_for`), all in ONE draw per slide: a microscope
    (`Render` at ds 1) and a `FovSupply` over `PlanSpec('ladder', <the levels'
    own ds>)`, each position photographed through the objective at its level.
    A level with no room for a FoV comes back short in the sampler's report; a
    slide with none at all is skipped and says so.

    `row['filename']` is `<slide>_L<level>_syn<i>.png`, `i` counting within
    the level, an id; nothing is written. `skip(filename)` true passes a shot
    over BEFORE it is rendered -- a resumed run's done shots -- and the rest
    are the same pictures, since a FoV's rng is its own position's
    (`FovSupply.photo`)."""
    for dataset in datasets:
        dataset_split = f'{dataset}#{split}'
        names = list_names(dataset=dataset_split)
        if n_wsi > len(names):
            raise ValueError(f'n_wsi {n_wsi} but {dataset_split} holds '
                             f'{len(names)} slide(s)')
        for name in names[:n_wsi]:
            path = str(locate(name, dataset=dataset_split).path)
            reader = SlideReader(path)
            mask, _ = masks.mask(reader.slide)
            rungs = fov.rungs_for(reader.level_downsamples)
            microscope = Render(reader, fov.sensor, fov.gap, ds=1.0,
                                seed=fov.sampler.seed)
            supply = FovSupply(microscope, PlanSpec('ladder', rungs,
                                                    camera=microscope.spec),
                               fov.sampler, mask)
            try:
                sampler = supply.sampler
            except RuntimeError as exc:
                print(f'  {name}: no FoV position at any level -- skipped '
                      f'({str(exc).splitlines()[0]})', flush=True)
                continue
            counted = {}
            for s in sampler:
                m = s.meta
                i = counted[m.level] = counted.get(m.level, -1) + 1
                filename = f'{name}_L{m.level}_syn{i:05d}.png'
                if skip is not None and skip(filename):
                    continue
                image, params = supply.photo(m)
                yield (dataclasses.asdict(FOVRecord.from_capture(
                    filename, path, supply.camera_for(m.ds), m.fov_rect[0],
                    m.fov_rect[1], params, level=m.level)), image)


# ── run identity: what a resumed metrics.csv must have been made with ────────

#: How `split_shots` places and photographs. Bumped whenever the same filename
#: would stop naming the same FoV: 'fov-supply-per-slide' is one draw per
#: slide across its levels.
SHOTS_RECIPE = 'fov-supply-per-slide'


def run_identity(args, fov: FovRecipe, encoder, mask_cfg) -> dict:
    """Everything that decides which shots a row is about and how it is
    scored. A resume skips shots by filename, and a filename is only an id:
    under another value of any of these, the same name is another FoV or
    another score, and rows of one bench would be mixed into another."""
    return {'shots': SHOTS_RECIPE,
            'datasets': list(args.datasets), 'split': args.split,
            'n_wsi': args.n_wsi, 'max_ds': fov.max_ds, 'rungs': fov.rungs,
            'sampler': dataclasses.asdict(fov.sampler),
            'sensor': list(fov.sensor), 'camera': dataclasses.asdict(fov.gap),
            'fov_mask': MASK_RECIPES['hest'].seg_id(),
            'encoder': encoder.identity_id(),
            'mask': [mask_cfg.seg_id(), mask_cfg.region_id()],
            'topk': args.topk, 'sift_topk': args.sift_topk}


def refuse_foreign_resume(run: dict, run_path: str) -> None:
    """Exit unless the metrics.csv being resumed was made by this same run."""
    if not os.path.exists(run_path):
        sys.exit(f'[refused] --resume, but {run_path} is missing: the metrics '
                 f'beside it predate the run identity, so which shots they hold '
                 f'cannot be told. Run without --resume, or give a new --out.')
    with open(run_path) as f:
        old = json.load(f)
    fresh = json.loads(json.dumps(run, default=str))
    differs = sorted(k for k in set(old) | set(fresh) if old.get(k) != fresh.get(k))
    if differs:
        sys.exit(f'[refused] --resume into a run made with other {differs} '
                 f'({run_path}). Run without --resume, or give a new --out.')


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


def _gt_center(row: dict, base_mpp: float) -> tuple:
    """(cx, cy) @ level-0 of the shot's centre.

    The rect the camera read is fov_width x fov_height at the NOMINAL mpp,
    and both the rotation and the final centre-crop are centred on it, so its
    centre is the shot's centre whatever the orientation or the scale. That
    invariance is why every position metric here is centre-based.

    Deliberately NOT built on _gt_footprint_wh: that one swaps w and h for the
    90/270 steps, which is right for a footprint and irrelevant to a centre,
    since gt_x/gt_y is the corner of the unswapped rect.
    """
    nominal = float(row['nominal_mpp'])
    w = int(row['fov_width'])  * nominal / base_mpp
    h = int(row['fov_height']) * nominal / base_mpp
    return (int(row['gt_x']) + w / 2.0, int(row['gt_y']) + h / 2.0)


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


def verify_candidates(pl, retriever, qc, candidates: list, n: int,
                      gt_cx: float, gt_cy: float,
                      hit_tol_px: float) -> dict:
    """Run SIFT on the first n candidates and report where each rank lands.

    This is the measurement behind "retrieval proposes K, SIFT verifies": it
    reports two ranks that must be read against each other.

      sift_hit_rank       first candidate SIFT localised CORRECTLY, judged
                          against ground truth. The ceiling of the design.
      sift_verified_rank  first candidate SIFT ACCEPTED, judged only by its own
                          inlier count. What a system with no ground truth
                          could actually pick -- so this is the number that
                          transfers to the real photographs.

    When the two agree, the inlier count is a trustworthy verifier and the loop
    is worth building. When verified fires earlier than hit, SIFT is confidently
    wrong somewhere in the list and a verification loop would lock onto it.

    Every candidate is tried rather than stopping at the first acceptance,
    because stopping would hide exactly that disagreement.
    """
    out = {
        'sift_topk_n':        0,
        'sift_hit_rank':      None,
        'sift_verified_rank': None,
        'sift_best_inliers':  None,
        'sift_best_rank':     None,
    }
    # A localizer of its own, sharing the pipeline's reader: pl.localizer still
    # holds rank 1's crop and keypoints, which the shot figure draws afterwards.
    loc = SiftRansacLocalizer(min_inliers=pl.localizer.min_inliers,
                              padding=pl.localizer.padding).build(pl.localizer.reader)
    for i in range(min(n, len(candidates))):
        rank = i + 1
        try:
            rf = loc.localize(qc, candidates, rank=i)
        except Exception:
            # A candidate whose crop is empty or whose match set is degenerate
            # is simply a candidate that failed verification, which is the same
            # outcome as a low inlier count. It must not stop the sweep.
            continue
        out['sift_topk_n'] += 1
        inl = int(rf.inlier_count)
        if out['sift_best_inliers'] is None or inl > out['sift_best_inliers']:
            # Recorded with its rank so the summary can score the other obvious
            # picker -- take the most inliers anywhere in the list, rather than
            # the first candidate that clears min_inliers. Neither is obviously
            # right and both are free to evaluate once this is here.
            out['sift_best_inliers'] = inl
            out['sift_best_rank']    = rank
        if out['sift_verified_rank'] is None and rf.success:
            out['sift_verified_rank'] = rank
        if out['sift_hit_rank'] is None and \
                _dist_px(rf.center_x0, rf.center_y0, gt_cx, gt_cy) <= hit_tol_px:
            out['sift_hit_rank'] = rank
    return out


def _round_s(seconds):
    return None if seconds is None else round(float(seconds), 3)


def compute_metrics(row: dict, result: LocaScopeQueryResult, base_mpp: float,
                    tile_l0: Optional[float] = None,
                    candidates: Optional[list] = None,
                    hit_tol_px: Optional[float] = None,
                    hit_tol_strict_px: Optional[float] = None,
                    sift_topk: Optional[dict] = None) -> dict:
    """Build one metrics.csv row from a gt-row + LocaScopeQueryResult.

    Two families of position error are recorded:

      *_err_px      top-left based. Only meaningful when the shot was NOT
                    rotated: gt_x/gt_y is the pre-rotation top-left, whereas
                    the prediction is where the shot's own (0,0) landed, which
                    is a different corner for the 90/180/270 steps.
      *_center_err_px  centre based. Rotation-invariant — the FoV is rotated
                    about its own centre, so this is comparable for every
                    orientation. Prefer this one when reading results.

    `candidates` is the retriever's top-K (retriever.candidate_set(k=...)). It turns one
    number, did the winner land, into the rank at which the truth first
    appears, which is the difference between a ranking that can be rescued by
    verifying K candidates and features that never scored the right window at
    all. A candidate's centre is derived exactly as the winner's is, from the
    ground-truth footprint, so rank 1 always agrees with retr_center_err_px.
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
    gt_cx, gt_cy = _gt_center(row, base_mpp)

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
        # Stage 2 top-K. rank of the first candidate near the truth, 1-based;
        # None means it was not in the K enumerated, which is a different
        # failure from being ranked low.
        'retr_topk_n':          None,
        'retr_hit_rank':        None,   # within hit_tol (the SIFT search crop)
        'retr_hit_rank_strict': None,   # within one tile (the window is right)
        'retr_hit_tol_px':      None,
        # One retrieval tile in LEVEL-0 pixels. Per shot, because it is
        # tile_size * ds and ds is the routed level's downsample: the same
        # 256 px tile is 256 level-0 px at L0 and 2048 at a 3-step 2x pyramid.
        # It is the natural unit for a retrieval error -- the grid cannot
        # resolve better than one tile -- and a fixed px threshold would mean
        # different things on different shots.
        'retr_tile_l0':         (round(tile_l0, 1) if tile_l0 else None),
        # Stage 2 + 3 combined: SIFT run on each of the first --sift-topk
        # candidates. hit is judged against ground truth, verified only against
        # SIFT's own inlier count -- the gap between them is whether the inlier
        # count can be trusted to pick the winner where there is no truth.
        'sift_topk_n':          None,
        'sift_hit_rank':        None,
        'sift_verified_rank':   None,
        'sift_best_inliers':    None,
        'sift_best_rank':       None,
        't_verify_s':           None,
        # the bench's own clock around the pipeline, per shot: the photo (on
        # a slide's first shot also its draw), the top-K enumeration, the
        # figures; `t_build_s` is the slide's pipeline build, on its first
        # shot only, so the column sums to the run's builds
        't_photo_s':            None,
        't_topk_s':             None,
        't_fig_s':              None,
        't_build_s':            None,
        # the pipeline's own clock, per stage of this shot (LocaScopeQueryResult)
        't_stage1_s':           _round_s(result.t_stage1_s),
        't_level_s':            _round_s(result.t_level_s),
        't_stage2_s':           _round_s(result.t_stage2_s),
        't_stage3_s':           _round_s(result.t_stage3_s),
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
        cs = result.retrieval
        best = cs.best
        bx0, by0 = cs.origin_l0(best)
        m['retr_rotation'] = int(best.rotation)
        m['rot_correct']   = (int(best.rotation) == m['gt_rot_deg'])
        m['retr_x0']    = int(bx0)
        m['retr_y0']    = int(by0)
        m['retr_score'] = float(best.score)
        d = _dist_px(bx0, by0, gt_x, gt_y)
        m['retr_err_px'] = d
        m['retr_err_um'] = d * base_mpp
        # Retrieval centre: the matched window holds the ROTATED query, so the
        # footprint's width/height swap for the 90/270 steps.
        w_l0, h_l0 = ((rect_h_l0, rect_w_l0) if best.rotation in (90, 270)
                      else (rect_w_l0, rect_h_l0))
        rcx = bx0 + w_l0 / 2.0
        rcy = by0 + h_l0 / 2.0
        dc = _dist_px(rcx, rcy, gt_cx, gt_cy)
        m['retr_center_x']      = round(rcx, 1)
        m['retr_center_y']      = round(rcy, 1)
        m['retr_center_err_px'] = dc
        m['retr_center_err_um'] = dc * base_mpp

        if candidates:
            m['retr_topk_n']     = len(candidates)
            m['retr_hit_tol_px'] = round(hit_tol_px, 1)
            for rank, c in enumerate(candidates, 1):
                cw, ch = ((rect_h_l0, rect_w_l0) if c.rotation in (90, 270)
                          else (rect_w_l0, rect_h_l0))
                cx0, cy0 = candidates.origin_l0(c)
                d = _dist_px(cx0 + cw / 2.0, cy0 + ch / 2.0, gt_cx, gt_cy)
                if m['retr_hit_rank'] is None and d <= hit_tol_px:
                    m['retr_hit_rank'] = rank
                if m['retr_hit_rank_strict'] is None and d <= hit_tol_strict_px:
                    m['retr_hit_rank_strict'] = rank
                if m['retr_hit_rank'] and m['retr_hit_rank_strict']:
                    break
        if sift_topk:
            m.update(sift_topk)
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
    pl, row: dict, img: np.ndarray, result: LocaScopeQueryResult,
    out_dir: str, zoom_pad: int = 4, metrics: Optional[dict] = None,
    subdir: str = '',
) -> Optional[str]:
    """Render the 4-panel localization diagnostic for one shot.

    Requires result to carry the diagnostic objects (run(..., keep_objects=True)).
    Returns the saved path, or None when the shot lacks retrieval/refine data.
    """
    if result.retrieval is None or result.refine is None or result.localizer is None:
        return None

    loc       = result.localizer
    # The best window's tile grid: the query's, turned to the winning rotation
    q_rows, q_cols = result.retrieval.window_tiles(result.retrieval.best,
                                                   result.query_qc)

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
        query_qc      = result.query_qc,
        sift          = result.refine,
        gt_x          = int(row['gt_x']),
        gt_y          = int(row['gt_y']),
        base_mpp      = pl.base_mpp,
        tile_size     = pl.tile_size,
        crop_img      = crop_img, crop_x0 = crop_x0,
        crop_y0       = crop_y0,  crop_ds = crop_ds,
        zoom_pad      = zoom_pad,
        query_rows    = q_rows, query_cols = q_cols,
        crop_origin_l0 = loc.crop_origin_l0,
        gt_center     = ((metrics['gt_center_x'], metrics['gt_center_y'])
                         if metrics else None),
        retr_center   = ((metrics['retr_center_x'], metrics['retr_center_y'])
                         if metrics and metrics.get('retr_center_x') is not None
                         else None),
        gt_box_wh     = _gt_footprint_wh(row, pl.base_mpp),
    )
    fig.suptitle(
        f'{row["filename"]}   L{row["level"]} -> routed L{result.routed_level}   '
        f'gt_rot={row["rot_deg"]} deg  retr_rot={result.retrieval.best.rotation} deg   '
        f'est_mpp={result.est_mpp:.4f}  effective_mpp={float(row["effective_mpp"]):.4f}',
        fontsize=11,
    )
    fig.tight_layout()

    fig_dir = os.path.join(out_dir, 'figures', subdir) if subdir \
              else os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, os.path.splitext(row['filename'])[0] + '_diag.png')
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ── failure classification ────────────────────────────────────────────────────
#
# "Failure" is not one thing here, and the categories need different pictures.
# The first two are stage-3 questions and reuse the 4-panel row; no_recall is a
# stage-2 question and gets its own, because on a recall failure SIFT was handed
# the wrong window and its picture says nothing about why.
#
# Note the axes are independent: a shot can be no_recall AND still land correct
# (the ruler retr_hit_rank uses is tighter than the crop SIFT actually reads),
# so a shot may be filed under more than one category on purpose.

FAILURE_MODES = ('confident-wrong', 'wrong', 'no-recall')

_MODE_DIR = {
    'confident-wrong': 'confident_wrong',
    'wrong':           'wrong_abstained',
    'no-recall':       'no_recall',
}


def classify_failure(m: dict, modes, tol_um: float) -> list:
    """Which of the requested failure modes this shot belongs to.

    `wrong` is reported as wrong_abstained so it never doubles up with
    confident-wrong: a shot past tolerance is one or the other, never both.
    """
    if not modes:
        return []
    err  = m.get('refine_center_err_um')
    past = err is not None and err > tol_um
    succ = bool(m.get('refine_success'))
    hit  = []
    if past and succ and 'confident-wrong' in modes:
        hit.append('confident-wrong')
    if past and not succ and 'wrong' in modes:
        hit.append('wrong')
    if 'no-recall' in modes and m.get('retr_topk_n') and m.get('retr_hit_rank') is None:
        hit.append('no-recall')
    return hit


def draw_recall_figure(
    pl, row: dict, img: np.ndarray, result: LocaScopeQueryResult,
    out_dir: str, m: dict, zoom_pad: int = 4,
) -> Optional[str]:
    """Render the RETRIEVAL diagnostic: the truth's window beside the chosen one.

    Reads two crops at the SAME scale (the retrieval level's ds) so the two
    panels are directly comparable; anchoring both through read_anchored_crop
    keeps that arithmetic in one place.
    """
    if result.retrieval is None or result.query_qc is None:
        return None
    gt_cx, gt_cy = m.get('gt_center_x'), m.get('gt_center_y')
    if gt_cx is None or gt_cy is None:
        return None

    q_rows, q_cols = result.retrieval.window_tiles(result.retrieval.best,
                                                   result.query_qc)
    ds = result.retrieval.ds
    w_l0, h_l0 = _gt_footprint_wh(row, pl.base_mpp)

    gt_crop, gx0, gy0, gds = read_anchored_crop(
        pl.wsi, gt_cx - w_l0 / 2.0, gt_cy - h_l0 / 2.0, ds,
        pl.tile_size, q_rows, q_cols, zoom_pad)
    pk_crop, px0, py0, pds = read_anchored_crop(
        pl.wsi, *result.retrieval.origin_l0(result.retrieval.best), ds,
        pl.tile_size, q_rows, q_cols, zoom_pad)

    d_um = m.get('retr_center_err_um')
    summary = [
        f'{row["filename"]}',
        '',
        f'true level      L{row["level"]}',
        f'routed level    L{result.routed_level}',
        f'est_mpp         {result.est_mpp:.4f}',
        f'nominal_mpp     {float(row["nominal_mpp"]):.4f}',
        '',
        f'retr_score      {result.retrieval.best.score:.4f}',
        f'retr_region     {result.retrieval.best.region_index} '
        f'{result.retrieval.best.lattice} r{result.retrieval.best.row} c{result.retrieval.best.col}',
        f'retr_rot        {result.retrieval.best.rotation} deg'
        f'   (gt {row["rot_deg"]} deg)',
        f'top-K enumerated {m.get("retr_topk_n")}',
        f'truth rank      {m.get("retr_hit_rank") or "NOT IN ANY CANDIDATE"}',
        '',
        f'pick -> truth   {d_um:,.0f} um' if d_um is not None else 'pick -> truth   n/a',
        f'FoV footprint   {w_l0:.0f} x {h_l0:.0f} px @L0',
        f'one tile        {pl.tile_size * ds:.0f} px @L0',
    ]

    fig, axes = plt.subplots(1, 4, figsize=(30, 7))
    draw_recall_row(
        axes,
        query_img   = img,
        gt_crop     = gt_crop,  gt_anchor   = (gx0, gy0, gds),
        gt_center   = (gt_cx, gt_cy),
        pick_crop   = pk_crop,  pick_anchor = (px0, py0, pds),
        pick_center = (m.get('retr_center_x'), m.get('retr_center_y')),
        box_wh      = (w_l0, h_l0),
        summary     = summary,
    )
    fig.suptitle(f'{row["filename"]}   RECALL FAILURE: the truth was never '
                 f'proposed among {m.get("retr_topk_n")} candidates', fontsize=12)
    fig.tight_layout()

    fig_dir = os.path.join(out_dir, 'figures', _MODE_DIR['no-recall'])
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir,
                            os.path.splitext(row['filename'])[0] + '_recall.png')
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ── output writers + plotting live in _locascope_plots (no torch, reusable by
#    plot_locascope_metrics.py to re-plot an existing metrics.csv offline) ─────


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument('--datasets', nargs='+', default=['bracs/test', 'ki67_with_photo'])
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--n-wsi', type=int, default=5, help='slides per dataset')
    # --fov, --max-ds, --sampler-* (--sampler-n-per-rung, --sampler-seed, ...)
    # and --camera-*, each given flag replacing that field of the recipe.
    add_fov_args(ap)
    ap.add_argument('--fov-mask-cache-job', default='MppRoutingHead',
                    help='hest masks the FoVs are placed on')
    ap.add_argument('--out',        default=None,
                    help='Output dir, used verbatim. Default: '
                         'result/<SLURM_JOB_NAME or BenchLocaScope>/<encoder>/ '
                         '-- the encoder level is added only to that derived '
                         'path, so put it in --out yourself when you give one.')
    ap.add_argument('--limit',      type=int, default=None,
                    help='Process only the first N shots (debug).')
    ap.add_argument('--draw-figures', type=int, default=0, metavar='N',
                    help='Save the 4-panel localization diagnostic for the first '
                         'N shots into <out>/figures/. 0 = off, -1 = all.')
    ap.add_argument('--draw-failures', nargs='+', default=[], metavar='MODE',
                    choices=list(FAILURE_MODES),
                    help='draw a figure for every shot that fails in one of '
                         'these ways, on top of --draw-figures. Files go to '
                         '<out>/figures/<category>/ so the categories stay '
                         'apart. confident-wrong: SIFT claimed success and was '
                         'past tolerance -- the only class a deployment cannot '
                         'detect. wrong: past tolerance but SIFT abstained. '
                         'no-recall: the truth was never proposed, which gets '
                         'the retrieval figure instead, because on a recall '
                         'failure the SIFT picture explains nothing.')
    ap.add_argument('--fail-tol-um', type=float, default=100.0, metavar='UM',
                    help='centre error above which --draw-failures counts a '
                         'shot as wrong (default 100). The error distribution '
                         'is bimodal by a factor of ~630, so anything from 50 '
                         'to 20000 selects the same shots.')
    ap.add_argument('--zoom-pad',   type=int, default=4,
                    help='Zoom-crop padding in tiles for the diagnostic panel.')
    ap.add_argument('--topk',       type=int, default=20, metavar='K',
                    help='Record the rank at which the truth first appears in '
                         'stage 2, over the K best-scoring windows. Costs no '
                         'GPU: the scores already exist and find_best throws '
                         'all but one away. 0 = off.')
    ap.add_argument('--sift-topk',  type=int, default=0, metavar='K',
                    help='Also run SIFT on the first K candidates and record '
                         'where it lands, both against ground truth and by its '
                         'own inlier count. Unlike --topk this is NOT free: it '
                         'is K SIFT passes per shot. 0 = off.')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--device',     default='auto')
    ap.add_argument(
        '--encoder', default='gigapath', choices=encoder_names(),
        help='which tile encoder. Only the module for THIS one is imported: '
             'every implementation sets HF_HOME above its own timm import and '
             'setdefault is first-one-wins, so importing all three would point '
             'two of them at the wrong weight cache -- silently. See '
             'TileEncoderFunc._IMPLEMENTATIONS. This is the synthetic '
             'counterpart of cli/driver/locate_photo, which takes the same two '
             'options: what runs on real photos has to be measurable here.')
    ap.add_argument(
        '--head', default='',
        help="which exit of the model, empty for its own default. Only CONCH "
             "has two, and BOTH run here -- the pipeline wants one vector per "
             "tile and either exit is that -- so this is a choice between two "
             "embeddings rather than a prerequisite. Its attentional pooler is "
             "512-d and its trunk 768-d, the shape GigaPath and UNI2 have. The "
             "head is part of identity_id and of the output directory.")
    # --seg: the mask stage 2 searches. `none` gives it ONE region covering the
    # whole scanned rectangle; a window on blank glass loses on its own
    # mean-cosine, but find_best takes a global maximum, so a region with more
    # placements wins on sample count alone, and at a routed level of 0 every
    # tile of the plane is encoded. A model recipe is read from
    # --mask-cache-job's cache when it is there (MppRoutingHead's holds hest for
    # every #val slide and the first test slides) and segmented, then written
    # back, when it is not -- that costs a mask build before the slide's first
    # shot and shares the GPUs with the tile encoder.
    add_mask_args(ap)
    ap.add_argument('--mask-cache-job', default='MppRoutingHead',
                    help="whose mask cache the pipeline's --seg mask is read "
                         'from and written to')
    ap.add_argument('--resume',     action='store_true',
                    help='carry on from an existing metrics.csv instead of '
                         'replacing it: every shot already recorded there is '
                         'skipped and the new rows are appended. For a run that '
                         'hit the walltime -- 2500 shots at about a minute each '
                         'does not fit 24 hours. The retrievers still have to be '
                         'rebuilt, so resume at a WSI boundary loses least. '
                         'Refused unless run.json beside it names the same '
                         'shots, mask, encoder and scoring (run_identity).')
    ap.add_argument('--features-cache-job', default=None, metavar='JOB',
                    help='cache each (slide, level) WSI feature map under '
                         'result/cache/<JOB>_features/<encoder>/, keyed on the '
                         'mask recipe and the grid. Off when absent. A hit skips the '
                         'ENCODE, not the read -- stage 3 reads pixels back out '
                         'of the container, so it is built either way '
                         '(278s read + 285s encode on BRACS_1228 L0). Every '
                         'miss prints which field differed, so a permanently '
                         'cold cache does not read like a correctly invalidated '
                         'one.')
    ap.add_argument('--feature-store-mode', choices=['r', 'w', 'rw'], default='rw',
                    help="'w' fills the cache without trusting it, which is how "
                         'to populate it the first time: otherwise the first run '
                         'has to trust the write and the read at once, and a '
                         'failure cannot say which.')
    ap.add_argument('--multi-gpu',  action='store_true',
                    help='wrap the tile encoder in DataParallel when the '
                         'allocation has more than one GPU. Raise --batch-size '
                         'with it: '
                         'DataParallel splits one batch across the cards, so an '
                         'unchanged batch just halves the work each card gets.')
    ap.add_argument('--precision',  choices=['fp16', 'fp32'], default='fp16',
                    help='autocast precision for the tile encoder. fp16 is the '
                         'validated production setting, but the validation is '
                         "GigaPath's: ~5.5x faster at cos=0.99995 and top-5=0.99 "
                         'against fp32 (log/TODO.log, AccuracyV1). Nobody has '
                         'measured the other two, so read fp16 there as an '
                         'assumption rather than a result. Ignored on CPU.')
    args = ap.parse_args()

    # A candidate that was never enumerated cannot be verified, and silently
    # verifying nothing is the worst of the three outcomes.
    if args.sift_topk > args.topk:
        print(f'[note] --sift-topk {args.sift_topk} > --topk {args.topk}; '
              f'raising --topk to match')
        args.topk = args.sift_topk

    # Only on the derived path -- an explicit --out is used verbatim, so a
    # caller that already composes the tag does not get a second one.
    enc_tag = encoder_tag(args.encoder, args.head)
    out_dir = args.out or job_result_dir('BenchLocaScope', encoder=enc_tag)
    os.makedirs(out_dir, exist_ok=True)
    fov = fov_from_args(args)
    sampler_cfg = fov.sampler
    print(f'shots      : {" ".join(args.datasets)}  #{args.split}  n_wsi={args.n_wsi}  '
          f'per level {sampler_cfg.n_per_rung}  seed {sampler_cfg.seed}  '
          f'sampler {sampler_cfg.identity_id()}')
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
    print(f'Loading {enc_tag} model ...', flush=True)
    # Passed only when given, so gigapath and uni2 build byte-identical configs
    # to before this option existed and their identity_id does not move.
    over = {'head': args.head} if args.head else {}
    encoder_cfg = encoder_config(args.encoder, batch_size=args.batch_size, **over)\
        .with_model(dtype='fp16' if dtype is torch.float16 else 'fp32')
    # Stage 2 builds its encoder from that config; stage 1 builds its own
    # (fp32, one card) from the registry name. All three stages are built once
    # here and bound to each slide by its pipeline's build().
    # Reads on DataLoader workers, so the slide is read while the GPU encodes;
    # the rest of the cpus are this process's torch threads.
    budget = CpuBudget.for_job(processes=1).apply()
    print(f'  {budget.line()}', flush=True)
    retriever = SlidingWinSimRot(
        SlidingWinSimRotConfig(encoder_cfg), device,
        multi_gpu=args.multi_gpu, read_workers=budget.workers)
    encoder = retriever.encoder
    if args.multi_gpu:
        import torch as _t
        print(f'  DataParallel over {_t.cuda.device_count()} GPU(s)', flush=True)
    print(f'  encoder    : {encoder.identity_id()}', flush=True)

    mask_cfg = mask_cfg_from_args(args)
    print(f'Mask       : {args.seg}   seg_id {mask_cfg.seg_id()}   region_id '
          f'{mask_cfg.region_id()}', flush=True)
    estimator = knn_estimator(args.encoder, mask_cfg, device=device)
    localizer = SiftRansacLocalizer()

    all_metrics: List[dict] = []
    metrics_path = os.path.join(out_dir, 'metrics.csv')
    run_path = os.path.join(out_dir, 'run.json')
    run = run_identity(args, fov, encoder, mask_cfg)
    metrics_fields: Optional[List[str]] = None
    done: set = set()
    if args.resume and os.path.exists(metrics_path):
        refuse_foreign_resume(run, run_path)
        # Take the column order from the existing header, not from the first
        # new row: append_metrics_row will not re-write a header into a
        # non-empty file, so a different order would silently misalign every
        # row appended from here on.
        with open(metrics_path) as f:
            rdr = csv.DictReader(f)
            metrics_fields = rdr.fieldnames
            done = {r['filename'] for r in rdr}
        # The resumed rows come back into memory too, so render_all at the end
        # plots the whole bench rather than only the resumed tail.
        all_metrics.extend(load_metrics_csv(metrics_path))
        print(f'Resume     : {len(done)} shots already in metrics.csv', flush=True)
    elif os.path.exists(metrics_path):
        # Truncate once, here. append_metrics_row writes a header into an empty
        # file, so without this a re-run would stack a second bench onto the first.
        os.remove(metrics_path)
    if not done:
        with open(run_path, 'w') as f:
            json.dump(run, f, indent=1, sort_keys=True, default=str)

    fov_masks = MaskMaker(MASK_RECIPES['hest'],
                          Cache.cache_root(args.fov_mask_cache_job, 'mask'), device)
    # The pipeline's own mask, through the same cache mechanism: under the
    # default --seg hest and one cache job it is the FoVs' mask, read once more
    # from the same file and never segmented twice.
    pipeline_masks = MaskMaker(mask_cfg, Cache.cache_root(args.mask_cache_job,
                                                          'mask'), device)
    shots = split_shots(args.datasets, args.split, args.n_wsi, fov,
                        masks=fov_masks, skip=lambda name: name in done)
    if done:
        print(f'Shots      : {len(done)} already done are passed over', flush=True)

    def record(m: dict) -> dict:
        """Keep the row for the plots AND put it on disk now.

        render_all still wants the whole list, so it is not either/or -- but
        the list alone is what made a crash on the last shot of a multi-hour
        run return nothing.
        """
        nonlocal metrics_fields
        if metrics_fields is not None and set(m) != set(metrics_fields):
            # a resumed metrics.csv written by a bench with other columns:
            # appending would misalign every row, or fail half-way
            sys.exit(f'[refused] {metrics_path} has the columns of another '
                     f'version of this bench (missing '
                     f'{sorted(set(m) - set(metrics_fields))}, extra '
                     f'{sorted(set(metrics_fields) - set(m))}). Run without '
                     f'--resume, or give a new --out.')
        all_metrics.append(m)
        metrics_fields = append_metrics_row(m, metrics_path, metrics_fields)
        return m
    n_drawn = 0
    n_fail_drawn = 0
    t_start = time.time()

    pl, cur_wsi, build_error, i = None, None, None, 0
    shot_iter = iter(shots)
    while True:
        t_p = time.time()
        try:
            row, img = next(shot_iter)
        except StopIteration:
            break
        t_photo = time.time() - t_p
        if args.limit and i >= args.limit:
            break
        i += 1
        t_build = None
        if row['wsi_path'] != cur_wsi:
            # A new slide: one pipeline per slide, built when its first shot
            # arrives (split_shots yields a slide's shots together).
            cur_wsi = row['wsi_path']
            wsi_tag = os.path.splitext(os.path.basename(cur_wsi))[0]
            print(f'== {wsi_tag}  path={cur_wsi}', flush=True)
            pl, build_error = None, None
            t_b = time.time()
            try:
                pl = LocaScopePipeline(
                    cur_wsi, estimator, retriever, localizer, pipeline_masks,
                    feature_store_root=(
                        None if not args.features_cache_job else
                        Cache.cache_root(args.features_cache_job, 'features') / enc_tag),
                    feature_store_mode=args.feature_store_mode).build()
                print(f'  pipeline built (base_mpp={pl.base_mpp:.4f}  '
                      f'mask_regions={len(pl.mask.tissue_regions)})', flush=True)
            except Exception as e:
                build_error = f'pipeline build failed: {type(e).__name__}: {e}'
                print(f'  [{build_error}]', flush=True)
            t_build = time.time() - t_b
        if pl is None:
            m = compute_metrics(row, LocaScopeQueryResult(
                None, None, False, None, None, build_error), 1.0)
            m.update(t_photo_s=round(t_photo, 2),
                     t_build_s=None if t_build is None else round(t_build, 2))
            record(m)
            continue

        want_fig = (args.draw_figures == -1
                    or n_drawn < args.draw_figures)
        t0 = time.time()
        # candidate_set reads the retriever's similarity maps, which only live on
        # the retriever object and only until the next shot overwrites
        # them, so it has to be kept and read here rather than later.
        # --draw-failures only knows the shot failed after the metrics
        # exist, so the diagnostic objects have to be kept before that.
        result = pl.run(img, keep_objects=(want_fig or args.topk > 0
                                           or bool(args.draw_failures)))
        dt = time.time() - t0

        # One tile at level-0 says the window itself is right. The refiner's
        # padding says the truth is inside the crop SIFT would search, which
        # is what a top-K plus verification loop could actually recover.
        tile_l0 = strict_tol = hit_tol = None
        if result.retrieval is not None:
            tile_l0 = strict_tol = pl.tile_size * result.retrieval.ds
            hit_tol = pl.localizer.padding * strict_tol

        cands = None
        t_k = time.time()
        if args.topk > 0 and result.retriever is not None and tile_l0:
            try:
                # the maps of THIS shot are still on the retriever
                cands = result.retriever.candidate_set(k=args.topk)
            except Exception as e:
                print(f'      [topk failed] {type(e).__name__}: {e}', flush=True)
        t_topk = time.time() - t_k

        sift_topk = None
        if cands and args.sift_topk > 0 and result.query_qc is not None:
            gt_cx, gt_cy = _gt_center(row, pl.base_mpp)
            t_v = time.time()
            sift_topk = verify_candidates(
                pl, result.retriever, result.query_qc, cands,
                args.sift_topk, gt_cx, gt_cy, hit_tol)
            sift_topk['t_verify_s'] = round(time.time() - t_v, 2)

        m = compute_metrics(row, result, pl.base_mpp, tile_l0=tile_l0,
                            candidates=cands, hit_tol_px=hit_tol,
                            hit_tol_strict_px=strict_tol,
                            sift_topk=sift_topk)

        t_f = time.time()
        if want_fig:
            try:
                p = draw_shot_figure(pl, row, img, result, out_dir,
                                     zoom_pad=args.zoom_pad, metrics=m)
                if p:
                    n_drawn += 1
                    print(f'      [fig] {p}', flush=True)
            except Exception as e:
                print(f'      [fig failed] {type(e).__name__}: {e}', flush=True)

        for mode in classify_failure(m, args.draw_failures, args.fail_tol_um):
            try:
                if mode == 'no-recall':
                    p = draw_recall_figure(pl, row, img, result, out_dir, m,
                                           zoom_pad=args.zoom_pad)
                else:
                    p = draw_shot_figure(pl, row, img, result, out_dir,
                                         zoom_pad=args.zoom_pad, metrics=m,
                                         subdir=_MODE_DIR[mode])
                if p:
                    n_fail_drawn += 1
                    print(f'      [fig {mode}] {p}', flush=True)
            except Exception as e:
                print(f'      [fig {mode} failed] {type(e).__name__}: {e}',
                      flush=True)
        m.update(t_photo_s=round(t_photo, 2), t_topk_s=round(t_topk, 2),
                 t_fig_s=round(time.time() - t_f, 2),
                 t_build_s=None if t_build is None else round(t_build, 2))
        record(m)

        print(f'  [{i:4d}] {row["filename"]:36s}  '
              f'L={row["level"]:>2}  '
              f'route=L{_fmt(m["routed_level"], "{:>1d}", "  ")}  '
              f'mpp_err={_fmt(m["mpp_err_rel"], "{:.3f}")}  '
              f'rot {m["gt_rot_deg"]:>3}->{_fmt(m["retr_rotation"], "{:>3d}", "  ?")}'
              f'{"" if m["rot_correct"] else "*"}  '
              f'retr_ctr={_fmt(m["retr_center_err_px"], "{:>7.0f}")}  '
              f'hit@{_fmt(m["retr_hit_rank"], "{:>2d}", " -")}  '
              f'sift@{_fmt(m["sift_hit_rank"], "{:>2d}", " -")}'
              f'/{_fmt(m["sift_verified_rank"], "{:<2d}", "- ")}  '
              f'refine_ctr={_fmt(m["refine_center_err_px"], "{:>7.0f}")}  '
              f'({dt:.1f}s: s1 {_fmt(m["t_stage1_s"], "{:.1f}")} '
              f'lvl {_fmt(m["t_level_s"], "{:.1f}")} '
              f's2 {_fmt(m["t_stage2_s"], "{:.1f}")} '
              f's3 {_fmt(m["t_stage3_s"], "{:.1f}")} '
              f'| photo {m["t_photo_s"]:.1f} topk {m["t_topk_s"]:.1f} '
              f'verify {_fmt(m["t_verify_s"], "{:.1f}")} fig {m["t_fig_s"]:.1f}'
              + ('' if t_build is None else f' build {t_build:.1f}') + ')'
              + (f'\n      ERR: {m["error"]}' if m['error'] else ''),
              flush=True)

    print(f'\nTotal wall time: {time.time() - t_start:.1f}s', flush=True)
    spent = {k: sum(float(m.get(k) or 0) for m in all_metrics)
             for k in ('t_build_s', 't_photo_s', 't_stage1_s', 't_level_s',
                       't_stage2_s', 't_stage3_s', 't_topk_s', 't_verify_s',
                       't_fig_s')}
    print('  where it went: ' + '  '.join(f'{k[2:-2]} {v:.0f}s'
                                          for k, v in spent.items()), flush=True)

    print(f'metrics.csv -> {metrics_path}  ({len(all_metrics)} rows)')
    render_all(all_metrics, out_dir)
    if args.draw_failures:
        print(f'\nFailure figures: {n_fail_drawn} written to '
              f'{os.path.join(out_dir, "figures")}/<category>/'
              f'   modes={" ".join(args.draw_failures)}  '
              f'tol={args.fail_tol_um:g}um')
    print(f'\nRe-plot later without re-running the pipeline:\n'
          f'  python utilities/cli/metrics/plot_locascope_metrics.py '
          f'{os.path.join(out_dir, "metrics.csv")}')


if __name__ == '__main__':
    main()
