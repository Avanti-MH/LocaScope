"""LocaScope end-to-end 3-stage pipeline glue.

Wraps the three stage primitives into one WSI-scoped object:

    Stage 1 — mpp estimation    via the estimator handed in
    Stage 2 — retrieval         via SlidingWinSimRot (cached per level)
    Stage 3 — SIFT+RANSAC       via SiftRansacLocalizer

Design:

* build() does the WSI-wide one-time work (mask + the estimator's build).
* The mask comes from `mask_cfg` (a `MASK_RECIPES` entry, hest by default),
  whose segmenter reads the level tile by tile -- a heavy method such as HEST
  DeepLabV3 OOMs on a whole MRXS level otherwise: at mask_ds=16 one slide's
  level image is ~313 MP, and a single ResNet layer1 activation on that is
  18.6 GiB. The chunk budgets are fields of the recipe's segmenter config.
* A retriever is built lazily on first use for each pyramid level; the
  routed level is the estimator's own `chosen_level` --
  `wsi.coarser_level_for_downsample`, the repo's own measured, coarse-biased
  routing rule (`SafeSlide.py`'s own docstring: 91.1% recovered by stage 3 at
  one level coarse against 15.7% at one level fine, 1398 shots). This file
  used to snap with `wsi.get_best_level_for_downsample` instead -- the
  FINE-biased general-purpose openslide rule, not the one measured for this
  exact job -- inline, a second implementation of the same snap. Fixed by
  reading `chosen_level` off the Result rather than recomputing it; see
  `StageInterface.py`'s docstring for why that recompute is not a shared
  function either, now that there is nothing left here to share it with.
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

    est = knn_estimator('gigapath', mask_cfg, device=device)
    ret = SlidingWinSimRot(
        SlidingWinSimRotConfig(encoder_config('gigapath')), device)
    loc = SiftRansacLocalizer()
    pl  = LocaScopePipeline(wsi, est, ret, loc, mask_cfg=mask_cfg).build()
    result = pl.run(shot_img)
    # result.est_mpp, result.routed_level, result.retrieval, result.refine
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import openslide
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _d in ('utilities', ''):     # '' = the root, which resolves the stage packages
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

from PatchingLib             import QueryPatchContainer                                # noqa: E402
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
    # Heavyweight refs kept only for diagnostics/plotting (not for bulk storage).
    # Populated when run(..., keep_objects=True).
    retriever: object = None      # SlidingWinSimRot
    localizer: object = None      # SiftRansacLocalizer
    query_qc:  object = None      # QueryPatchContainer at the winning rotation



class LocaScopePipeline:
    """3-stage LocaScope pipeline over one WSI."""

    def __init__(
        self,
        wsi:                 Union[openslide.OpenSlide, str],
        estimator,                    # stage 1, built, not yet bound
        retriever:           SlidingWinSimRot,
        localizer:           SiftRansacLocalizer,
        mask_cfg:            'TissueMaskConfig' = None,
        feature_store_root:  Optional[str] = None,
        feature_store_mode:  str = 'rw',
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
        # One value instead of five parameters and two remembered method calls.
        # The mask a pipeline builds can now say how it was built, which is what
        # a cache has to ask before trusting a stored feature map.
        from TissueMaskConfig import MASK_RECIPES
        self.mask_cfg            = mask_cfg or MASK_RECIPES['hest']
        self.feature_store_root  = feature_store_root
        self.feature_store_mode  = feature_store_mode

        # SafeSlide.base_mpp: the mean of mpp-x and mpp-y, with an aperio
        # fallback. This line used to read mpp-x alone, which disagreed with
        # QueryFromWSI on every slide in use.
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
        self.tile_size           = retriever.tile_size
        self._level_reason: Dict[int, Optional[str]] = {}   # None = usable

    # ── One-time setup ────────────────────────────────────────────────────────
    def build(self) -> 'LocaScopePipeline':
        """Build the mask, then bind the estimator and stages 2-3 to this slide
        (per-WSI one-time)."""
        # Segment, filter and merge in one place and in one order. merge is
        # incomplete without filter having run first -- it skips nested boxes on
        # the assumption they are already gone -- and that dependency used to be
        # two lines every caller wrote out.
        self.mask = self.mask_cfg.build(
            self.wsi, getattr(self.retriever.encoder, 'device', None))

        # The same mask to stage 1, so its reference bank and stage 2's
        # retriever agree on which regions are tissue (a method that samples
        # nothing ignores it -- StageInterface.MppEstimator).
        self.estimator.build(self.wsi, mask=self.mask)
        # Stage 2 and 3 bound to the same slide and mask. No level is built
        # here: the retriever builds the one stage 1 routes a shot to, once.
        self.retriever.build(self.wsi, self.mask, feature_store=self._feature_store())
        self.localizer.build(self.wsi)
        self._level_reason = {}
        self._built = True
        return self

    # ── Lazy per-level retriever cache ────────────────────────────────────────
    #
    # `_level_mask` used to live here: a per-level copy of the mask keeping only
    # regions that can host a tile. It was the patchable filter written out a
    # second time, because the mask's filter then mutated in place and this
    # needed a copy. It moved into the retriever -- `build_wsi_features`
    # takes a `mask.patchable(...)` view -- so the
    # retriever now narrows the mask itself, at the ds it is actually going to
    # build at. Which is the point: this class did not know that ds, it only
    # knew the one it was asking for.

    def _feature_store(self):
        '''The feature-map cache for this slide, or None when no root was given.

        `feature_store_root` is `Cache.cache_root(<job>, 'features') /
        encoder_tag`. Built here because the file's address needs the whole
        mask recipe -- the segmentation and the region prep -- and this is the
        only object that holds it.
        '''
        if not self.feature_store_root:
            return None
        from Store import FeatureMapCache
        return FeatureMapCache(
            self.feature_store_root, getattr(self.wsi, '_filename', ''),
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

    # ── Per-shot end-to-end ───────────────────────────────────────────────────
    def run(self, img_np: np.ndarray, keep_objects: bool = False) -> LocaScopeQueryResult:
        """Run all 3 stages on one shot image, each on the previous one's
        output: EstMppResult -> CandidateSet -> SiftRansacResult.

        `keep_objects=True` attaches the retriever / localizer / query container
        to the result so diagnostics can plot keypoints, matches and homography.
        Leave False for bulk runs — those objects hold large tensors.
        """
        if not self._built:
            raise RuntimeError('LocaScopePipeline not built; call .build() first.')

        # Stage 1 — estimate mpp AND route to a pyramid level (`chosen_level`,
        # the estimator's own snap). Stage 2 searches that level; it does not
        # choose again.
        try:
            r1 = self.estimator.estimate(img_np)
            est_mpp = float(r1.estimated_mpp)
            level = r1.chosen_level
        except Exception as e:
            return LocaScopeQueryResult(
                None, None, False, None, None,
                f'stage1 failed: {type(e).__name__}: {e}')

        # Stage 2 — candidate windows at the routed level
        reason = self._level_ready(level)
        if reason:
            return LocaScopeQueryResult(est_mpp, level, True, None, None, reason)
        try:
            qc = QueryPatchContainer(img_np)
            qc.extract_all(self.tile_size, overlap=self.retriever.overlap)
            retrieval = self.retriever.retrieve(qc, r1)
        except Exception as e:
            return LocaScopeQueryResult(
                est_mpp, level, False, None, None,
                f'stage2 failed: {type(e).__name__}: {e}')

        # Stage 3 — SIFT+RANSAC inside the best candidate
        try:
            refine = self.localizer.localize(qc, retrieval, rank=0)
        except Exception as e:
            return LocaScopeQueryResult(
                est_mpp, level, False, retrieval, None,
                f'stage3 failed: {type(e).__name__}: {e}')

        return LocaScopeQueryResult(
            est_mpp, level, False, retrieval, refine, None,
            retriever = self.retriever if keep_objects else None,
            localizer = self.localizer if keep_objects else None,
            query_qc  = qc             if keep_objects else None,
        )
