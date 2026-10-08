"""LocaScope end-to-end 3-stage pipeline glue.

Wraps the three stage primitives into one WSI-scoped object:

    Stage 1 — mpp estimation    via the estimator handed in
    Stage 2 — retrieval         via SlidingWinSimRot (cached per level)
    Stage 3 — SIFT+RANSAC       via SiftRansacLocalizer

Design:

* build() does the WSI-wide one-time work (mask + the estimator's build).
* The mask comes from `masks`, a `TissueMaskConfig.MaskMaker`: its recipe (a
  `MASK_RECIPES` entry) and, when it has one, its cache -- so a slide whose
  mask is cached is never segmented again. The recipe's segmenter reads the
  level tile by tile -- a heavy method such as HEST DeepLabV3 OOMs on a whole
  MRXS level otherwise: at mask_ds=16 one slide's level image is ~313 MP, and a
  single ResNet layer1 activation on that is 18.6 GiB. The chunk budgets are
  fields of the recipe's segmenter config.
* A retriever is built lazily on first use for each pyramid level; the
  routed level is the estimator's own `chosen_level` --
  `StageInterface.routed_level` over `ReadGeometry.coarser_level`, the
  measured, coarse-biased routing rule (91.1% recovered by stage 3 at one
  level coarse against 15.7% at one level fine, 1398 shots). It is read
  off the Result, never recomputed here.
* If a level's retriever build fails (e.g. `patchable` emptied the mask
  because tiles are too big at that level), the shot is marked
  `unusable_level` and its stage 2 / 3 metrics are None.
* Errors in any stage produce a LocaScopeQueryResult with `.error` set;
  earlier stages' results are preserved.

The three stages come in BUILT, each from its own config, and not yet bound
to a slide: `estimator` (any stage-1 method -- KnnEstMpp, ClassifierEstMpp,
PrototypeEstMpp), `retriever` (SlidingWinSimRot) and `localizer`
(SiftRansacLocalizer). This class builds the mask, binds all three to the
slide, and runs a shot through them; it does not choose a method or a
parameter of any of them. Each stage builds its own encoder, so stage 1 and
stage 2 never share one: a ClassifierEstMpp checkpoint may carry a fine-tuned
trunk, and stage 2 has to score with the encoder its feature cache was written
by. A second copy of the weights in GPU memory is the price. The stages are
reusable across slides: a loop over slides builds them once and a pipeline per
slide.

Usage:

    masks = MaskMaker(MASK_RECIPES['hest'], job, device)   # job's mask cache
    est = KnnEstMpp(KNN_RECIPES['gigapath'], device)
    ret = SlidingWinSimRot(SLIDEWIN_RECIPES['gigapath'], device)
    loc = SiftRansacLocalizer(SIFT_RECIPES['default'])
    pl  = LocaScopePipeline(wsi, est, ret, loc, masks).build()
    result = pl.run(shot_img)
    # result.est_mpp, result.routed_level, result.retrieval, result.refine, result.ranks

    r1 = pl.stage1(shot_img)                  # or each stage alone, for a caller
    qc, cs = pl.stage2(shot_img, r1.chosen_level)   # that keeps what each made
    ranks = pl.stage3(qc, cs)
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import openslide
import torch

from PatchingLib            import QueryPatchContainer                                # noqa: E402
from SafeSlide               import SafeSlide                                          # noqa: E402
from TissueMask      import TissueMask                                 # noqa: E402
from stage2_retrieval.SlidingWinSimRot import SlidingWinSimRot     # noqa: E402
from stage2_retrieval.StageInterface import CandidateSet                        # noqa: E402
from stage3_localization.SIFT_RANSAC             import SiftRansacLocalizer, SiftRansacResult              # noqa: E402


@dataclass
class LocaScopeQueryResult:
    """Per-shot pipeline output.

    Metrics that couldn't be computed are None; `error` carries the reason
    when a stage errored. `unusable_level` is True when the routed pyramid
    level has no retriever available (mask filter yielded 0 regions).
    """
    est_mpp:        Optional[float]
    routed_level:   Optional[int]
    unusable_level: bool
    retrieval:      Optional[CandidateSet]
    refine:         Optional[SiftRansacResult]
    error:          Optional[str]
    # Wall seconds per stage of THIS shot. `t_level_s` is the routed level's
    # one-time build (features encoded or read from the cache) when this shot
    # is the first to need it, else ~0; `t_stage2_s` is the search alone.
    # None for a stage that did not run.
    t_stage1_s:     Optional[float] = None
    t_level_s:      Optional[float] = None
    t_stage2_s:     Optional[float] = None
    t_stage3_s:     Optional[float] = None
    # Heavyweight refs kept only for diagnostics/plotting (not for bulk storage).
    # Populated when run(..., keep_objects=True).
    retriever: object = None      # SlidingWinSimRot
    localizer: object = None      # SiftRansacLocalizer
    query_qc:  object = None      # QueryPatchContainer at the winning rotation
    # Every stage's own output: the EstMppResult, and one SiftRansacResult per
    # verified candidate (`refine` is the first of them).
    stage1:    object = None
    ranks:     Optional[list] = None



class LocaScopePipeline:
    """3-stage LocaScope pipeline over one WSI."""

    def __init__(
        self,
        wsi:                 Union[openslide.OpenSlide, str],
        estimator,                    # stage 1, built, not yet bound; None: not run here
        retriever:           Optional[SlidingWinSimRot],
        localizer:           Optional[SiftRansacLocalizer],
        masks:               'MaskMaker',
        feature_cache_job:   Optional[str] = None,
        feature_store_mode:  str = 'rw',
        bank_cache_job:      Optional[str] = None,
    ):
        # SafeSlide, not OpenSlide: a MIRAX read that lands on a cell the
        # scanner never wrote raises, and that raise latches on the handle, so
        # every later call fails -- metadata included. Since this one object is
        # handed to TissueMask, TileSampler and SlidingWinSimRot,
        # the recovery has to live inside it; healing swaps the native handle in
        # place and every holder keeps working.
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi                 = wsi
        # Where the mask comes from; its recipe is how the mask was built,
        # which is what a cache has to ask before trusting a stored feature map.
        self.masks               = masks
        self.mask_cfg            = masks.cfg
        self.feature_cache_job   = feature_cache_job
        self.feature_store_mode  = feature_store_mode
        # Whose cache stage 1 keeps its reference bank in (the draw and its
        # features); None: made in memory every build.
        self.bank_cache_job      = bank_cache_job

        # SafeSlide.base_mpp: the one definition of the slide's scale.
        self.base_mpp = wsi.base_mpp   # raises if the slide carries no mpp

        self.mask:      Optional[TissueMask] = None
        self.estimator           = estimator
        self._built = False
        # None value == "tried, unusable"; missing key == "not tried yet"
        # One retriever and one localizer for the slide: the retriever keeps
        # every level it has built (per ds), so there is no per-level object.
        self.retriever           = retriever
        self.localizer           = localizer
        # The tile the query is cut at is the retriever's, which compares it
        # against tiles of that size.
        self.tile_size           = None if retriever is None else retriever.tile_size
        self._level_reason: Dict[int, Optional[str]] = {}   # None = usable

    # ── One-time setup ────────────────────────────────────────────────────────
    def build(self) -> 'LocaScopePipeline':
        """Build the mask, then bind the estimator and stages 2-3 to this slide
        (per-WSI one-time)."""
        # Segment (or read the cached raw mask), filter and merge in one place
        # and in one order: `MaskMaker.mask`. merge is incomplete without filter
        # having run first -- it skips nested boxes on the assumption they are
        # already gone.
        self.mask, _ = self.masks.mask(self.wsi)

        # The same mask to stage 1, so its reference bank and stage 2's
        # retriever agree on which regions are tissue (a method that samples
        # nothing ignores it -- StageInterface.MppEstimator).
        if self.estimator is not None:
            self.estimator.build(self.wsi, mask=self.mask, masks=self.masks,
                                 cache_job=self.bank_cache_job)
        # Stage 2 and 3 bound to the same slide and mask. No level is built
        # here: the retriever builds the one stage 1 routes a shot to, once.
        if self.retriever is not None:
            self.retriever.build(self.wsi, self.mask,
                                 feature_store=self._feature_store())
        if self.localizer is not None:
            self.localizer.build(self.wsi)
        self._level_reason = {}
        self._built = True
        return self

    # ── Lazy per-level retriever cache ────────────────────────────────────────
    #
    # The retriever narrows the mask itself (`build_wsi_features` takes a
    # `mask.patchable(...)` view), at the ds it builds at -- which this class
    # does not know.

    def _feature_store(self):
        '''The feature-map cache for this slide, or None when no root was given.

        `feature_cache_job` is whose cache the grid features are read from and
        written to. Built here because the address needs the whole mask recipe
        -- the segmentation and the region prep -- and this is the only object
        that holds it.
        '''
        if not self.feature_cache_job:
            return None
        from Store import FeatureMapCache
        return FeatureMapCache(
            self.feature_cache_job, getattr(self.wsi, '_filename', ''),
            self.retriever.encoder, self.mask_cfg, mode=self.feature_store_mode)

    def _level_ready(self, level: int) -> Optional[str]:
        """Build the retriever's features for `level` if not yet, and say why
        the level is unusable (None when it is usable). Remembered per level:
        a level that failed once is not rebuilt for every shot routed to it."""
        if level in self._level_reason:
            return self._level_reason[level]
        print(f'  [retriever L{level}] mpp='
              f'{self.base_mpp * self.wsi.level_downsamples[level]:.4f}', flush=True)
        reason = None
        try:
            self.retriever.build_wsi_features(level=level)
            print(f'  [retriever L{level}] regions '
                  f'{len(self.retriever.regions)}/{len(self.mask.tissue_regions)} '
                  f'patchable at ds={self.retriever.ds:g}', flush=True)
            if not self.retriever.wsi_features:
                reason = 'build_wsi_features produced no feature maps'
        except Exception as e:
            # Never swallow this silently -- a failed level turns every shot
            # routed to it into a bare `unusable_level` with no reason.
            reason = f'build failed: {type(e).__name__}: {e}'
            traceback.print_exc()
        if reason:
            print(f'  [retriever L{level}] UNUSABLE: {reason}', flush=True)
        self._level_reason[level] = reason
        return reason

    # ── One stage at a time ───────────────────────────────────────────────────
    #
    # A caller that keeps each stage's output (the bench) runs them one by one:
    # a stage it already has is read back instead, and the next one is handed
    # what was read. `run` is the three in a row.

    def stage1(self, img_np: np.ndarray):
        """Stage 1: the estimator's EstMppResult -- the mpp and the routed level."""
        return self.estimator.estimate(img_np)

    def stage2(self, img_np: np.ndarray, level: int
               ) -> Tuple[QueryPatchContainer, CandidateSet]:
        """Stage 2 at `level`: the query cut at the retriever's tile, and the
        candidate windows. Raises `UnusableLevel` when the level has no
        feature map. The retriever's similarity maps stay on it until the next
        query, for whatever the caller reads off them."""
        reason = self._level_ready(level)
        if reason:
            raise UnusableLevel(reason)
        self.retriever.build_wsi_features(level=level)
        qc = QueryPatchContainer(img_np)
        qc.extract_all(self.tile_size, overlap=self.retriever.overlap)
        self.retriever.build_query_features(qc)
        self.retriever.compute_sim_maps()
        return qc, self.retriever.candidate_set()

    def candidates_at(self, level: int, candidates) -> CandidateSet:
        """A CandidateSet of `candidates` (stage 2's, read back) in `level`'s
        frame -- what stage 3 takes. The level's grids come from its feature
        map, read from the cache when it is there."""
        reason = self._level_ready(level)
        if reason:
            raise UnusableLevel(reason)
        self.retriever.build_wsi_features(level=level)
        return CandidateSet(candidates=tuple(candidates), level=self.retriever.level,
                            ds=self.retriever.ds, grids=tuple(self.retriever.grids))

    def stage3(self, qc: QueryPatchContainer, cs: CandidateSet
               ) -> List[SiftRansacResult]:
        """Stage 3: one SiftRansacResult per verified candidate, in rank order."""
        return self.localizer.localize_top(qc, cs)

    # ── Per-shot end-to-end ───────────────────────────────────────────────────
    def run(self, img_np: np.ndarray, keep_objects: bool = False) -> LocaScopeQueryResult:
        """Run all 3 stages on one shot image, each on the previous one's
        output: EstMppResult -> CandidateSet -> SiftRansacResult per verified
        candidate.

        `keep_objects=True` attaches the retriever / localizer / query container
        to the result so diagnostics can plot keypoints, matches and homography.
        Leave False for bulk runs — those objects hold large tensors.
        """
        if not self._built:
            raise RuntimeError('LocaScopePipeline not built; call .build() first.')

        # Each stage ends in host values (an mpp, topk lists, a homography), so
        # the GPU has finished by the time a clock is read.
        t = {}
        t0 = time.perf_counter()
        try:
            r1 = self.stage1(img_np)
            est_mpp = float(r1.estimated_mpp)
            level = r1.chosen_level
        except Exception as e:
            return LocaScopeQueryResult(
                None, None, False, None, None,
                f'stage1 failed: {type(e).__name__}: {e}')
        t['t_stage1_s'] = time.perf_counter() - t0

        # Stage 2 — candidate windows at the routed level; the level's build
        # (first shot to need it) timed apart from the search
        t0 = time.perf_counter()
        reason = self._level_ready(level)
        t['t_level_s'] = time.perf_counter() - t0
        if reason:
            return LocaScopeQueryResult(est_mpp, level, True, None, None, reason,
                                        stage1=r1, **t)
        t0 = time.perf_counter()
        try:
            qc, retrieval = self.stage2(img_np, level)
        except Exception as e:
            return LocaScopeQueryResult(
                est_mpp, level, False, None, None,
                f'stage2 failed: {type(e).__name__}: {e}', stage1=r1, **t)
        t['t_stage2_s'] = time.perf_counter() - t0

        # Stage 3 — SIFT+RANSAC inside the first n_verify candidates
        t0 = time.perf_counter()
        try:
            ranks = self.stage3(qc, retrieval)
        except Exception as e:
            return LocaScopeQueryResult(
                est_mpp, level, False, retrieval, None,
                f'stage3 failed: {type(e).__name__}: {e}', stage1=r1, **t)
        t['t_stage3_s'] = time.perf_counter() - t0

        return LocaScopeQueryResult(
            est_mpp, level, False, retrieval, ranks[0] if ranks else None, None,
            stage1=r1, ranks=ranks, **t,
            retriever = self.retriever if keep_objects else None,
            localizer = self.localizer if keep_objects else None,
            query_qc  = qc             if keep_objects else None,
        )


class UnusableLevel(RuntimeError):
    """The routed level has no feature map to search (the mask left no region
    that holds a tile there, or its build failed)."""
