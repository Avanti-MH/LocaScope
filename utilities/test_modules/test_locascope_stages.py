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
    stages start at 2   the drawn FoV's own mpp (the ground truth) is used AS
                        the estimate

WHAT THIS MERGE DROPPED, ON PURPOSE. `test_gigapath_slide_win_sim.py` carried
~250 lines of rotation-recovery and sim-tensor-kernel-equivalence checks
specific to `SlidingWinSimRot`'s internals, and both retrieval-facing
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
import random
import sys
import time
from types import SimpleNamespace
from typing import Tuple

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
from _paths import setup_import_paths
setup_import_paths()

from PatchingLib import QueryPatchContainer                          # noqa: E402
from camera import sensor_size                                      # noqa: E402
from SlideReader import SlideReader                                 # noqa: E402
from TileEncoderFunc import encoder_config, encoder_names             # noqa: E402
from TileSampler import (OverlapConfig, PlanSpec, RichnessConfig,     # noqa: E402
                         SamplerConfig, TileSampler)
from AccessDatasets import list_names, locate                        # noqa: E402
from Cache import cache_root, job_name                               # noqa: E402
from SafeSlide import SafeSlide                                     # noqa: E402
from ReadGeometry import ReadSpec                                   # noqa: E402
from TissueMaskConfig import MaskMaker, add_mask_args, mask_cfg_from_args  # noqa: E402
from stage1_estimation.KnnEstMpp import KnnEstMpp, KnnEstMppConfig, REFERENCE_BANK_RICHNESS  # noqa: E402
from stage2_retrieval.SlidingWinSimRot import (                     # noqa: E402
    SlidingWinSimRot, SlidingWinSimRotConfig)
from stage3_localization.SIFT_RANSAC import SiftRansacLocalizer                            # noqa: E402


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

def pick_fov(args, masks) -> None:
    """Set args.wsi / x / y / mpp: a slide of `--dataset` and one FoV drawn on
    it at `--rung`, as Stage1MppBench draws its FoVs (a TileSampler draw for
    the query camera, cached). `x, y` is the FoV's own top-left."""
    names = list_names(dataset=args.dataset, split_job=args.split_cache_job)
    name = args.slide or random.Random(args.pick_seed).choice(names)
    entry = locate(name, dataset=args.dataset, split_job=args.split_cache_job)
    sampler_root = cache_root(
        args.sampler_cache_job or job_name('TestLocaScopeStages'), 'sampler')
    sampler = TileSampler.cached(
        entry.path,
        SamplerConfig(n_per_rung=1, seed=args.pick_seed,
                      richness=RichnessConfig(), overlap=OverlapConfig()),
        PlanSpec('ladder', (args.rung,),
                 camera=ReadSpec(*sensor_size(args.ratio, args.mpixels))),
        sampler_root, masks=masks)
    drawn = list(sampler)
    if not drawn:
        sys.exit(f'[FAIL] no FoV fits {name} at rung {args.rung:g}')
    meta = drawn[0].meta
    args.wsi = str(entry.path)
    args.x, args.y = int(meta.fov_rect[0]), int(meta.fov_rect[1])
    args.mpp = SafeSlide(entry.path).base_mpp * float(meta.ds)


def crop_query(args):
    reader = SlideReader(args.wsi)
    query_np = reader.read(args.x, args.y,
                           ReadSpec(*sensor_size(args.ratio, args.mpixels)),
                           args.mpp / reader.base_mpp)
    if query_np is None:
        sys.exit('[FAIL] SlideReader.read returned None (off the slide)')
    qc = QueryPatchContainer(query_np)
    qc.extract_all(args.tile, overlap=args.overlap)
    if qc.grid.grid_rows == 0 or qc.grid.grid_cols == 0:
        sys.exit('[FAIL] query too small for even one patch -- use a larger '
                 '--mpixels or a smaller --tile')
    print(f'  query {query_np.shape[1]}x{query_np.shape[0]}  '
         f'patches={qc.grid.grid_rows}x{qc.grid.grid_cols}')
    return reader.slide, query_np, qc


def stage2_encoder_cfg(args):
    over = {'head': args.head} if args.head else {}
    return encoder_config(args.encoder, batch_size=args.batch, **over)\
        .with_model(dtype='fp32')


def run_stage1(wsi, mask, query_qc, args, device):
    """Returns the EstMppResult."""
    cfg = KnnEstMppConfig(
        encoder=args.encoder, mask_cfg=mask_cfg_from_args(args),
        sampler_cfg=SamplerConfig(n_per_rung=args.samples,
                                  seed=args.seed, richness=REFERENCE_BANK_RICHNESS,
                                  overlap=OverlapConfig()),
        k=args.k, tile_size=args.tile)
    est = KnnEstMpp(cfg, device=device).build(wsi, mask=mask)
    result = est.estimate(query_qc)
    err_pct = abs(result.estimated_mpp - args.mpp) / args.mpp * 100
    print(f'  mpp_gt={args.mpp:.4f}  mpp_est={result.estimated_mpp:.4f}  '
         f'error={err_pct:.1f}%  level {result.chosen_level}')
    return result


def run_stage2(wsi, mask, query_np, estimate, args, device):
    """Stage 2 on stage 1's output (or its ground-truth substitute). Builds
    its own encoder, as stage 1 does."""
    retriever = SlidingWinSimRot(
        SlidingWinSimRotConfig(stage2_encoder_cfg(args), tile_size=args.tile,
                                       overlap=args.overlap),
        device).build(wsi, mask)
    cs = retriever.retrieve(query_np, estimate)
    best = cs.best
    x0, y0 = cs.origin_l0(best)
    print(f'  best=({x0}, {y0})  score={best.score:.4f}  rotation={best.rotation}  '
         f'region={best.region_index} {best.lattice}  level {cs.level}')
    return retriever, cs


def run_stage3(wsi, query_qc, cs, args):
    """Stage 3 on stage 2's output."""
    localizer = SiftRansacLocalizer(
        min_inliers=args.min_inliers, padding=args.padding).build(wsi)
    result = localizer.localize(query_qc, cs)
    print(f'  success={result.success}  matches={result.match_count}  '
         f'inliers={result.inlier_count}')
    return result


def dist_um(x, y, args, base_mpp) -> float:
    return math.sqrt((x - args.x) ** 2 + (y - args.y) ** 2) * base_mpp


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stages', type=parse_stages, default=(1, 2, 3))
    # The slide comes from a recorded split whose masks are already cached, and
    # the query from a TileSampler draw on it -- the way Stage1MppBench picks
    # both -- so the test segments nothing and its ground truth is a placed FoV.
    ap.add_argument('--dataset', default='bracs/test#val',
                    help='<dataset>#<split>; every val slide of bracs/test and '
                         'ki67_with_photo has a mask in the MppRoutingHead cache')
    ap.add_argument('--slide', default=None,
                    help='a name in --dataset; default: one picked by --pick-seed')
    ap.add_argument('--rung', type=float, default=1.0,
                    help='the ds the FoV is drawn at. Its mpp is the ground '
                         'truth, and the STAGE-1 SUBSTITUTE when --stages '
                         'starts at 2 -- see this module\'s docstring')
    ap.add_argument('--pick-seed', type=int, default=0,
                    help='which slide (when --slide is not given) and which FoV')
    ap.add_argument('--mask-cache-job', default='MppRoutingHead',
                    help='whose mask cache is read')
    ap.add_argument('--sampler-cache-job', default=None,
                    help="whose sampler cache the FoV draw goes in; default this job's")
    ap.add_argument('--split-cache-job', default=None,
                    help='whose recorded split --dataset is read from; default MakeSplit')
    ap.add_argument('--ratio',   default='45:32')
    ap.add_argument('--mpixels', type=float, default=1.475)
    ap.add_argument('--tile',    type=int,   default=256)
    ap.add_argument('--overlap', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--filter', action=argparse.BooleanOptionalAction, default=True,
                    help="the recipe's region prep (filtered + merged); "
                         '--no-filter hands the stages the raw components')
    add_mask_args(ap, default='hest')
    ap.add_argument('--batch',   type=int,   default=1024)
    ap.add_argument('--encoder', default='gigapath', choices=encoder_names())
    ap.add_argument('--head',    default='')
    ap.add_argument('--samples', type=int, default=40, help='stage 1: reference tiles per level')
    ap.add_argument('--k',       type=int, default=5,  help='stage 1: KNN neighbours')
    ap.add_argument('--seed',    type=int, default=42, help='stage 1: reference bank sampling')
    ap.add_argument('--padding', type=int, default=2,  help='stage 3: tiles of context around the match')
    ap.add_argument('--min-inliers', type=int, default=10)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timings: dict = {}
    masks = MaskMaker(mask_cfg_from_args(args),
                      cache_root(args.mask_cache_job, 'mask'), device)
    pick_fov(args, masks)
    print(f'WSI    : {args.wsi}')
    print(f'GT     : x={args.x}  y={args.y}  mpp={args.mpp:.4f}  (rung {args.rung:g})')
    print(f'Stages : {args.stages}')

    print('\n[crop] cropping query...')
    t0 = time.perf_counter()
    wsi, query_np, query_qc = crop_query(args)
    base_mpp = wsi.base_mpp
    timings['crop'] = time.perf_counter() - t0

    # Stage 1's output, or its ground-truth substitute when stage 1 is skipped:
    # the true mpp, routed by the rule KnnEstMpp routes by.
    estimate = SimpleNamespace(
        estimated_mpp=args.mpp,
        chosen_level=wsi.coarser_level_for_downsample(args.mpp / base_mpp))
    ok = True

    # ONE mask, built once, whichever stages run -- stage 1's own reference
    # bank and stage 2's retriever have to agree on what counts as tissue,
    # the same reason `LocaScopePipeline.build()` builds it once and hands it
    # to both rather than letting each stage segment its own.
    print('\n[mask] reading the tissue mask...')
    t0 = time.perf_counter()
    mask, hit = masks.mask(wsi)
    masks.close()
    print(f'  {"cache hit" if hit else "segmented (cache miss)"}: '
          f'{cache_root(args.mask_cache_job, "mask")}')
    before = len(mask.raw())
    if not args.filter:
        mask = mask.raw()
    timings['mask'] = time.perf_counter() - t0
    print(f'  {before} regions'
         + (f' -> {len(mask.tissue_regions)} after the recipe\'s region prep'
            if args.filter else ' (filter disabled)'))

    if 1 in args.stages:
        print('\n[1] KnnEstMpp...')
        t0 = time.perf_counter()
        estimate = run_stage1(wsi, mask, query_qc, args, device)
        timings['1. estimate mpp'] = time.perf_counter() - t0
        ok &= abs(estimate.estimated_mpp - args.mpp) / args.mpp < 0.20
    else:
        print('\n[1] SKIPPED -- using ground-truth mpp as the stage-1 substitute')

    if 2 in args.stages:
        print('\n[2] SlidingWinSimRot...')
        t0 = time.perf_counter()
        retriever, retrieval_result = run_stage2(
            wsi, mask, query_np, estimate, args, device)
        timings['2. retrieval'] = time.perf_counter() - t0
        ret_err = dist_um(*retrieval_result.origin_l0(retrieval_result.best),
                          args, base_mpp)
        tol_um = args.tile * retrieval_result.ds * base_mpp
        print(f'  distance to GT: {ret_err:.1f} um  (tolerance {tol_um:.1f} um)')
        ok &= ret_err <= tol_um

    if 3 in args.stages:
        print('\n[3] SIFT + RANSAC...')
        t0 = time.perf_counter()
        sift_result = run_stage3(wsi, query_qc, retrieval_result, args)
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
