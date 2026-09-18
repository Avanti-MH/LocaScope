#!/usr/bin/env python3
"""Merged end-to-end test for the 3 LocaScope stages -- replaces
`test_gigapath_knn_esti_mpp.py`, `test_gigapath_slide_win_sim.py` and
`test_sift_ransac.py`, which each cropped the same kind of query, loaded the
same kind of encoder, and re-wrote most of the same argparse flags.

    python utilities/test_modules/test_locascope_stages.py --stages 1
    python utilities/test_modules/test_locascope_stages.py --stages 1,2
    python utilities/test_modules/test_locascope_stages.py --stages 2,3
    python utilities/test_modules/test_locascope_stages.py --stages 1,2,3

`--stages` is a CONTIGUOUS run: `1`, `1,2`, `1,2,3`, `2,3` are the only legal
values -- never a bare `3` (stage 3 needs stage 2's retrieval result to
refine) and never `1,3` (skipping 2 leaves stage 3 nothing to read).

Starting above stage 1 does NOT re-derive what stage 1 would have produced --
it substitutes that stage's own required input directly, so a stage-2/3
failure can never be blamed on stage 1:

    stages start at 1   KnnEstMpp actually estimates the query's mpp
    stages start at 2   `--mpp` (the ground truth) is used AS the estimate

WHAT THIS MERGE DROPPED, ON PURPOSE. `test_gigapath_slide_win_sim.py` carried
~250 lines of rotation-recovery and sim-tensor-kernel-equivalence checks
specific to `GigaPathSlidingWinSimRot`'s internals, and both retrieval-facing
source files carried their own ~150-line multi-panel matplotlib figure.
Stage 2/3 are getting the same interface redesign stage 1 just got
(`StageInterface.py`), so polishing tests against the CURRENT interface is
effort spent against a target that is about to move. This file keeps the
load-bearing numeric checks -- does the estimate/retrieval/localisation land
near ground truth -- and drops the rest. Redo the finer-grained guards once
stage 2/3's real shape is settled; this is a placeholder, not a replacement
for that work.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Tuple

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
from _paths import setup_import_paths
setup_import_paths()

from PatchingLib import QueryPatchContainer                          # noqa: E402
from QueryFromWSI import QueryFromWSI                                # noqa: E402
from TileEncoderFunc import encoder_config, encoder_names             # noqa: E402
from TileSampler import OverlapConfig, SamplerConfig                  # noqa: E402
from TissueMaskConfig import TissueMaskConfig                          # noqa: E402
from KnnEstMpp import KnnEstMpp, KnnEstMppConfig, REFERENCE_BANK_RICHNESS  # noqa: E402
from GigaPathSlidingWinSimRot import GigaPathSlidingWinSimRot          # noqa: E402
from SIFT_RANSAC import SiftRansacLocalizer                            # noqa: E402


# ── --stages ─────────────────────────────────────────────────────────────────

_LEGAL_STAGES = ({1}, {1, 2}, {1, 2, 3}, {2, 3})


def parse_stages(text: str) -> Tuple[int, ...]:
    """`'1'` / `'1,2'` / `'1,2,3'` / `'2,3'` -- anything else is refused rather
    than silently run as whatever it happened to parse to. A bare `3` or a
    `1,3` both look like a typo for one of the four legal shapes, and running
    the wrong one would look like a stage-3 result with no stage 3 input."""
    try:
        stages = tuple(sorted({int(s) for s in text.split(',')}))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'--stages must be a comma list of ints, got {text!r}')
    if set(stages) not in _LEGAL_STAGES:
        raise argparse.ArgumentTypeError(
            f"--stages {text!r} is not one of the legal shapes: "
            f"'1', '1,2', '1,2,3', '2,3' (a contiguous run starting at 1 or "
            f"2 -- stage 3 always needs stage 2's own result)")
    return stages


# ── steps shared by every --stages combination ────────────────────────────────

def crop_query(args):
    qfwsi = QueryFromWSI(args.wsi, wh_ratio=args.ratio, MPixels=args.mpixels,
                         mpp=args.mpp)
    query_pil = qfwsi.crop(args.x, args.y)
    if query_pil is None:
        sys.exit('[FAIL] QueryFromWSI.crop returned None')
    query_np = np.array(query_pil)
    qc = QueryPatchContainer(query_np)
    qc.extract_all(args.tile, overlap=args.overlap)
    if qc.grid.grid_rows == 0 or qc.grid.grid_cols == 0:
        sys.exit('[FAIL] query too small for even one patch -- use a larger '
                 '--mpixels or a smaller --tile')
    print(f'  query {query_pil.width}x{query_pil.height}  '
         f'patches={qc.grid.grid_rows}x{qc.grid.grid_cols}')
    return qfwsi.wsi, query_np, qc


def load_encoder(args, device):
    over = {'head': args.head} if args.head else {}
    return encoder_config(args.encoder, batch_size=args.batch, **over)\
        .with_model(dtype='fp32').build(device)


def run_stage1(wsi, mask, query_qc, args, device) -> Tuple[float, object]:
    """Returns (mpp_est, the encoder KnnEstMpp built -- reused by stage 2/3
    rather than building a second copy; see `KnnEstMpp.py`'s own docstring
    for why that sharing is the caller's job, not a guarantee of the class)."""
    cfg = KnnEstMppConfig(
        encoder=args.encoder, mask_cfg=TissueMaskConfig(),
        sampler_cfg=SamplerConfig(tile=args.tile, n_per_rung=args.samples,
                                  seed=args.seed, richness=REFERENCE_BANK_RICHNESS,
                                  overlap=OverlapConfig()),
        k=args.k)
    est = KnnEstMpp(cfg, device=device).build(wsi, mask=mask)
    result = est.estimate(query_qc)
    err_pct = abs(result.estimated_mpp - args.mpp) / args.mpp * 100
    print(f'  mpp_gt={args.mpp:.4f}  mpp_est={result.estimated_mpp:.4f}  '
         f'error={err_pct:.1f}%')
    return result.estimated_mpp, est.encoder


def run_stage2(wsi, encoder, mask, query_np, mpp_est, args):
    retriever = GigaPathSlidingWinSimRot(
        wsi, encoder=encoder, mask=mask, mpp=mpp_est,
        tile_size=args.tile, overlap=args.overlap)
    retriever.build_wsi_features()
    retriever.build_query_features(query_np)
    retriever.compute_sim_maps()
    result = retriever.find_best()
    print(f'  best=({result.best_x0}, {result.best_y0})  '
         f'score={result.best_score:.4f}  rotation={result.best_rotation}  '
         f'region={result.best_region_index}')
    return retriever, result


def run_stage3(retriever, query_qc, retrieval_result, args):
    localizer = SiftRansacLocalizer(
        retriever.wsi_container, query_qc, retrieval_result,
        min_inliers=args.min_inliers, padding=args.padding)
    localizer.read_wsi_crop()
    localizer.detect_and_match()
    result = localizer.estimate_homography()
    print(f'  success={result.success}  matches={result.match_count}  '
         f'inliers={result.inlier_count}')
    return result


def dist_um(x, y, args, base_mpp) -> float:
    return math.sqrt((x - args.x) ** 2 + (y - args.y) ** 2) * base_mpp


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stages', type=parse_stages, default=(1, 2, 3))
    ap.add_argument('--wsi',
                    default='/work/u26130998/datasets/histoimage.na.icar.cnr.it/'
                            'BRACS_WSI/test/Group_AT/Type_ADH/BRACS_1228.svs')
    ap.add_argument('--x',       type=int,   default=31700)
    ap.add_argument('--y',       type=int,   default=33600)
    ap.add_argument('--mpp',     type=float, default=0.252,
                    help='ground truth. Also the STAGE-1 SUBSTITUTE when '
                         '--stages starts at 2 -- see this module\'s docstring')
    ap.add_argument('--ratio',   default='45:32')
    ap.add_argument('--mpixels', type=float, default=1.475)
    ap.add_argument('--tile',    type=int,   default=256)
    ap.add_argument('--overlap', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--filter', action=argparse.BooleanOptionalAction, default=True,
                    help='apply filter_regions to drop small/contained tissue regions')
    ap.add_argument('--min-region-ratio', type=float, default=0.10)
    ap.add_argument('--batch',   type=int,   default=1024)
    ap.add_argument('--encoder', default='gigapath', choices=encoder_names())
    ap.add_argument('--head',    default='')
    ap.add_argument('--samples', type=int, default=40, help='stage 1: reference tiles per level')
    ap.add_argument('--k',       type=int, default=5,  help='stage 1: KNN neighbours')
    ap.add_argument('--seed',    type=int, default=42, help='stage 1: reference bank sampling')
    ap.add_argument('--padding', type=int, default=2,  help='stage 3: tiles of context around the match')
    ap.add_argument('--min-inliers', type=int, default=10)
    args = ap.parse_args()

    if not os.path.exists(args.wsi):
        print(f'[SKIP] WSI not found: {args.wsi}')
        return 0

    print(f'WSI    : {args.wsi}')
    print(f'GT     : x={args.x}  y={args.y}  mpp={args.mpp}')
    print(f'Stages : {args.stages}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timings: dict = {}

    print('\n[crop] cropping query...')
    t0 = time.perf_counter()
    wsi, query_np, query_qc = crop_query(args)
    base_mpp = wsi.base_mpp
    timings['crop'] = time.perf_counter() - t0

    encoder = None
    mpp_est = args.mpp   # ground-truth substitute unless stage 1 runs below
    ok = True

    # ONE mask, built once, whichever stages run -- stage 1's own reference
    # bank and stage 2's retriever have to agree on what counts as tissue,
    # the same reason `LocaScopePipeline.build()` builds it once and hands it
    # to both rather than letting each stage segment its own.
    print('\n[mask] building tissue mask...')
    t0 = time.perf_counter()
    mask = TissueMaskConfig().build(wsi, device)
    before = len(mask.tissue_regions)
    if args.filter:
        mask.filter_regions(min_ratio=args.min_region_ratio)
    timings['mask'] = time.perf_counter() - t0
    print(f'  {before} regions'
         + (f' -> {len(mask.tissue_regions)} after filtering'
            if args.filter else ' (filter disabled)'))

    if 1 in args.stages:
        print('\n[1] KnnEstMpp...')
        t0 = time.perf_counter()
        mpp_est, encoder = run_stage1(wsi, mask, query_qc, args, device)
        timings['1. estimate mpp'] = time.perf_counter() - t0
        ok &= abs(mpp_est - args.mpp) / args.mpp < 0.20
    else:
        print('\n[1] SKIPPED -- using ground-truth mpp as the stage-1 substitute')

    if 2 in args.stages:
        if encoder is None:
            print('\n[encoder] loading (no stage 1 to reuse one from)...')
            encoder = load_encoder(args, device)

        print('\n[2] GigaPathSlidingWinSimRot...')
        t0 = time.perf_counter()
        retriever, retrieval_result = run_stage2(
            wsi, encoder, mask, query_np, mpp_est, args)
        timings['2. retrieval'] = time.perf_counter() - t0
        ret_err = dist_um(retrieval_result.best_x0, retrieval_result.best_y0,
                          args, base_mpp)
        tol_um = args.tile * retrieval_result.ds * base_mpp
        print(f'  distance to GT: {ret_err:.1f} um  (tolerance {tol_um:.1f} um)')
        ok &= ret_err <= tol_um

    if 3 in args.stages:
        print('\n[3] SIFT + RANSAC...')
        t0 = time.perf_counter()
        sift_result = run_stage3(retriever, query_qc, retrieval_result, args)
        timings['3. sift ransac'] = time.perf_counter() - t0
        if sift_result.success:
            sift_err = dist_um(sift_result.x0, sift_result.y0, args, base_mpp)
            improvement = ret_err - sift_err
            print(f'  distance to GT: {sift_err:.1f} um  '
                 f'({"+" if improvement > 0 else ""}{improvement:.1f} um vs '
                 f'retrieval alone)')
            ok &= sift_err <= tol_um
        else:
            print('  [FAIL] SIFT+RANSAC did not converge')
            ok = False

    wsi.close()

    print('\n' + '-' * 42)
    total = sum(timings.values())
    for name, t in timings.items():
        print(f'  {name:<22}  {t:>6.1f}s')
    print(f'  {"total":<22}  {total:>6.1f}s')
    print('-' * 42)
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
