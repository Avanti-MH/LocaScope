"""LocaScope end-to-end 3-stage pipeline glue.

Wraps the three stage primitives into one WSI-scoped object:

    Stage 1 — mpp estimation    via KnnEstMpp
    Stage 2 — retrieval         via GigaPathSlidingWinSimRot (cached per level)
    Stage 3 — SIFT+RANSAC       via SiftRansacLocalizer

Design:

* build() does the WSI-wide one-time work (mask + KNN reference bank).
* The mask comes from `mask_cfg` (a `MASK_RECIPES` entry, hest by default),
  whose segmenter reads the level tile by tile -- a heavy method such as HEST
  DeepLabV3 OOMs on a whole MRXS level otherwise: at mask_ds=16 one slide's
  level image is ~313 MP, and a single ResNet layer1 activation on that is
  18.6 GiB. The chunk budgets are fields of the recipe's segmenter config.
* A retriever is built lazily on first use for each pyramid level; the
  routed level is `KnnEstMpp.estimate`'s own `chosen_level` --
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

`encoder` IS A REGISTRY NAME (`TileEncoderFunc`'s, e.g. `'gigapath'`), not a
built object -- `KnnEstMpp` builds its own from it (`KnnEstMppConfig`'s own
design: an estimator does not need to be handed an already-built encoder to
be usable standing alone). This pipeline still wants ONE encoder shared
across mask-building, stage 1 and stage 2 rather than three separate copies
in GPU memory, so it does not build a second one itself: `build()` reads the
one `self.estimator` already built off `self.estimator.encoder` and reuses
THAT for everything downstream. The sharing was always incidental to what
`KnnEstMpp` needs for itself, never a requirement of it -- this is the
pipeline arranging for it, not the estimator promising it.

Usage:

    from utilities.LocaScopePipeline import LocaScopePipeline

    pl = LocaScopePipeline(wsi, encoder='gigapath').build()
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
for _d in ('utilities', '1_estimate_query_mpp', '2_retrieval', '3_localization'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

from PatchingLib             import QueryPatchContainer                                # noqa: E402
from SafeSlide               import SafeSlide                                          # noqa: E402
from TissueMask      import TissueMask                                 # noqa: E402
from TileSampler             import OverlapConfig, SamplerConfig                       # noqa: E402
from KnnEstMpp                import (KnnEstMpp, KnnEstMppConfig,                       # noqa: E402
                                      REFERENCE_BANK_RICHNESS)
from GigaPathSlidingWinSimRot import GigaPathSlidingWinSimRot, SlideWinSimRotResult     # noqa: E402
from SIFT_RANSAC             import SiftRansacLocalizer, SiftRansacResult              # noqa: E402


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
    retrieval:      Optional[SlideWinSimRotResult]
    refine:         Optional[SiftRansacResult]
    error:          Optional[str]
    # Heavyweight refs kept only for diagnostics/plotting (not for bulk storage).
    # Populated when run(..., keep_objects=True).
    retriever: object = None      # GigaPathSlidingWinSimRot
    localizer: object = None      # SiftRansacLocalizer
    query_qc:  object = None      # QueryPatchContainer at the winning rotation


class LocaScopePipeline:
    """3-stage LocaScope pipeline over one WSI."""

    def __init__(
        self,
        wsi:                 Union[openslide.OpenSlide, str],
        encoder:             str,
        device:              Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
        tile_size:           int   = 256,
        mask_cfg:            'TissueMaskConfig' = None,
        feature_store_root:  Optional[str] = None,
        feature_store_mode:  str = 'rw',
        knn_samples:         int   = 40,
        knn_k:               int   = 5,
        knn_seed:            int   = 42,
        retriever_overlap:   bool  = True,
        refiner_min_inliers: int   = 10,
        refiner_padding:     int   = 2,
    ):
        # SafeSlide, not OpenSlide: a MIRAX read that lands on a cell the
        # scanner never wrote raises, and that raise latches on the handle, so
        # every later call fails -- metadata included. Since this one object is
        # handed to TissueMask, TileSampler and GigaPathSlidingWinSimRot,
        # the recovery has to live inside it; healing swaps the native handle in
        # place and every holder keeps working.
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi                 = wsi
        # A REGISTRY NAME, not a built object -- see this module's own
        # docstring. Resolved to the real thing in build(), and read off
        # `self.estimator.encoder` from there on so mask-building and stage 2
        # share the exact object `KnnEstMpp` built for itself.
        self.encoder_name        = encoder
        self.device              = torch.device(device)
        self.encoder = None
        self.tile_size           = tile_size
        # One value instead of five parameters and two remembered method calls.
        # The mask a pipeline builds can now say how it was built, which is what
        # a cache has to ask before trusting a stored feature map.
        from TissueMaskConfig import MASK_RECIPES
        self.mask_cfg            = mask_cfg or MASK_RECIPES['hest']
        self.feature_store_root  = feature_store_root
        self.feature_store_mode  = feature_store_mode
        self.knn_samples         = knn_samples
        self.knn_k               = knn_k
        self.knn_seed            = knn_seed
        self.retriever_overlap   = retriever_overlap
        self.refiner_min_inliers = refiner_min_inliers
        self.refiner_padding     = refiner_padding

        # SafeSlide.base_mpp: the mean of mpp-x and mpp-y, with an aperio
        # fallback. This line used to read mpp-x alone, which disagreed with
        # QueryFromWSI on every slide in use.
        self.base_mpp = wsi.base_mpp   # raises if the slide carries no mpp

        self.mask:      Optional[TissueMask] = None
        self.estimator: Optional[KnnEstMpp] = None
        # None value == "tried, unusable"; missing key == "not tried yet"
        self._retrievers: Dict[int, Optional[GigaPathSlidingWinSimRot]] = {}
        self._retriever_reason: Dict[int, str] = {}   # why a level is unusable

    # ── One-time setup ────────────────────────────────────────────────────────
    def build(self) -> 'LocaScopePipeline':
        """Build the encoder + mask + mpp KNN reference bank (per-WSI one-time)."""
        # KnnEstMpp builds its own encoder from cfg.encoder -- constructing
        # it here, before the mask, is what lets mask-building below reuse
        # THAT object instead of building a second encoder of its own.
        # `mask_cfg=self.mask_cfg` even though `build(wsi, mask=...)` below
        # hands the estimator an already-built mask directly: identity should
        # still name the recipe that mask actually came from, not whatever
        # KnnEstMppConfig's own default happens to be.
        cfg = KnnEstMppConfig(
            encoder=self.encoder_name, mask_cfg=self.mask_cfg,
            sampler_cfg=SamplerConfig(
                n_per_rung=self.knn_samples,
                seed=self.knn_seed, richness=REFERENCE_BANK_RICHNESS,
                overlap=OverlapConfig()),
            k=self.knn_k, tile_size=self.tile_size)
        self.estimator = KnnEstMpp(cfg, device=self.device)
        self.encoder = self.estimator.encoder

        # Segment, filter and merge in one place and in one order. merge is
        # incomplete without filter having run first -- it skips nested boxes on
        # the assumption they are already gone -- and that dependency used to be
        # two lines every caller wrote out.
        self.mask = self.mask_cfg.build(
            self.wsi, getattr(self.encoder, 'device', None))

        # Reuses `self.mask` rather than letting KnnEstMpp segment its own --
        # same mask, so stage 1's reference bank and stage 2's retriever agree
        # on which regions are tissue.
        self.estimator.build(self.wsi, mask=self.mask)
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
            self.encoder, self.mask_cfg, mode=self.feature_store_mode)

    def _get_retriever(self, level: int) -> Optional[GigaPathSlidingWinSimRot]:
        """Return cached rotation-aware retriever for this level, or None if unusable."""
        if level in self._retrievers:
            return self._retrievers[level]

        level_mpp = self.base_mpp * self.wsi.level_downsamples[level]
        print(f'  [retriever L{level}] mpp={level_mpp:.4f}', flush=True)

        try:
            r = GigaPathSlidingWinSimRot(
                self.wsi, encoder=self.encoder, mask=self.mask,
                mpp=level_mpp, tile_size=self.tile_size,
                overlap=self.retriever_overlap,
                feature_store=self._feature_store(),
            )
            r.build_wsi_features()
            # How many regions survived is only knowable after the build now,
            # because the retriever filters at the ds it resolved rather than
            # at the one asked for. An empty feature list is the same condition
            # the old n_ok == 0 check caught, one step later and for the same
            # reason.
            print(f'  [retriever L{level}] regions '
                  f'{len(r.regions)}/{len(self.mask.tissue_regions)} patchable '
                  f'at ds={r.ds:g}', flush=True)
            if not r.wsi_features:
                reason = 'build_wsi_features produced no feature maps'
                print(f'  [retriever L{level}] UNUSABLE: {reason}', flush=True)
                self._retrievers[level] = None
                self._retriever_reason[level] = reason
                return None
        except Exception as e:
            # Never swallow this silently — a failed retriever turns every shot
            # routed to this level into a bare `unusable_level` with no reason.
            reason = f'build failed: {type(e).__name__}: {e}'
            print(f'  [retriever L{level}] {reason}', flush=True)
            traceback.print_exc()
            self._retrievers[level] = None
            self._retriever_reason[level] = reason
            return None
        self._retrievers[level] = r
        return r

    # ── Per-shot end-to-end ───────────────────────────────────────────────────
    def run(self, img_np: np.ndarray, keep_objects: bool = False) -> LocaScopeQueryResult:
        """Run all 3 stages on one shot image.

        `keep_objects=True` attaches the retriever / localizer / query container
        to the result so diagnostics can plot keypoints, matches and homography.
        Leave False for bulk runs — those objects hold large tensors.
        """
        if self.estimator is None:
            raise RuntimeError('LocaScopePipeline not built; call .build() first.')

        # Stage 1 — estimate mpp AND route to a pyramid level.
        # `chosen_level` is `KnnEstMpp.estimate`'s own snap
        # (`wsi.coarser_level_for_downsample`) -- there is no separate
        # routing step left to fail on its own, so a routing failure now
        # surfaces as `stage1 failed` rather than its own category. See this
        # module's own docstring for why that snap moved off this file.
        try:
            r1 = self.estimator.estimate(img_np, overlap=True)
            est_mpp = float(r1.estimated_mpp)
            level = r1.chosen_level
        except Exception as e:
            return LocaScopeQueryResult(
                None, None, False, None, None,
                f'stage1 failed: {type(e).__name__}: {e}')

        # Stage 2 — retrieve (cached retriever per level)
        retriever = self._get_retriever(level)
        if retriever is None:
            return LocaScopeQueryResult(
                est_mpp, level, True, None, None,
                self._retriever_reason.get(level, 'retriever unavailable'))

        try:
            qc = QueryPatchContainer(img_np)
            qc.extract_all(self.tile_size, overlap=self.retriever_overlap)
            retriever.build_query_features(qc)
            retriever.compute_sim_maps()
            retrieval = retriever.find_best()
        except Exception as e:
            return LocaScopeQueryResult(
                est_mpp, level, False, None, None,
                f'stage2 failed: {type(e).__name__}: {e}')

        # Stage 3 — SIFT+RANSAC refine
        try:
            localizer = SiftRansacLocalizer(
                reader=retriever.reader, grids=retriever.grids,
                level=retriever.level,
                query=qc, location=retrieval,
                min_inliers=self.refiner_min_inliers,
                padding=self.refiner_padding,
            )
            localizer.read_wsi_crop()
            localizer.detect_and_match()
            refine = localizer.estimate_homography()
        except Exception as e:
            return LocaScopeQueryResult(
                est_mpp, level, False, retrieval, None,
                f'stage3 failed: {type(e).__name__}: {e}')

        return LocaScopeQueryResult(
            est_mpp, level, False, retrieval, refine, None,
            retriever = retriever if keep_objects else None,
            localizer = localizer if keep_objects else None,
            query_qc  = qc        if keep_objects else None,
        )
