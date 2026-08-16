"""LocaScope end-to-end 3-stage pipeline glue.

Wraps the three stage primitives into one WSI-scoped object:

    Stage 1 — mpp estimation    via GigaPathKnnEstiMpp
    Stage 2 — retrieval         via GigaPathSlidingWinSimRot (cached per level)
    Stage 3 — SIFT+RANSAC       via SiftRansacLocalizer

Design:

* build() does the WSI-wide one-time work (mask + KNN reference bank).
* The mask is always built through `from_wsi`'s tile-and-stitch path
  (`mask_seg_chunk_px`). A heavy `mask_method` such as HEST DeepLabV3 OOMs on a
  whole MRXS level otherwise — at mask_ds=16 one slide's level image is
  ~313 MP, and a single ResNet layer1 activation on that is 18.6 GiB.
  Pass mask_seg_chunk_px=None to opt out and segment the level in one call.
* A retriever is built lazily on first use for each pyramid level; est_mpp
  from stage 1 is fuzzy-snapped to a level via
  `wsi.get_best_level_for_downsample`. WsiTissuesContainer requires an exact
  match to a level downsample, so the retriever is constructed at that
  level's NATIVE mpp — not at the raw est_mpp.
* If a level's retriever build fails (e.g. filter_patchable emptied the mask
  because tiles are too big at that level), the shot is marked
  `unusable_level` and its stage 2 / 3 metrics are None.
* Errors in any stage produce a LocaScopeQueryResult with `.error` set;
  earlier stages' results are preserved.

Usage:

    from utilities.LocaScopePipeline import LocaScopePipeline

    pl = LocaScopePipeline(wsi, encoder).build()
    result = pl.run(shot_img)
    # result.est_mpp, result.routed_level, result.retrieval, result.refine
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import openslide

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _d in ('utilities', '1_estimate_query_mpp', '2_retrieval', '3_localization'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

from PatchingLib             import QueryPatchContainer                                # noqa: E402
from SafeSlide               import SafeSlide                                          # noqa: E402
from TissuesRegionsMask      import TissuesRegionsMask                                 # noqa: E402
from GigaPathKnnEstiMpp      import GigaPathKnnEstiMpp                                 # noqa: E402
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
        encoder:             Callable,
        tile_size:           int   = 256,
        mask_cfg:            'TissueMaskConfig' = None,
        feature_store_root:  Optional[str] = None,
        feature_store_mode:  str = 'rw',
        knn_samples:         int   = 40,
        knn_k:               int   = 5,
        knn_seed:            Optional[int] = 42,
        knn_tissue_ratio:    float = 0.5,
        retriever_overlap:   bool  = True,
        refiner_min_inliers: int   = 10,
        refiner_padding:     int   = 2,
    ):
        # SafeSlide, not OpenSlide: a MIRAX read that lands on a cell the
        # scanner never wrote raises, and that raise latches on the handle, so
        # every later call fails -- metadata included. Since this one object is
        # handed to TissuesRegionsMask, TileSampler and GigaPathSlidingWinSimRot,
        # the recovery has to live inside it; healing swaps the native handle in
        # place and every holder keeps working.
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi                 = wsi
        self.encoder             = encoder
        self.tile_size           = tile_size
        # One value instead of five parameters and two remembered method calls.
        # The mask a pipeline builds can now say how it was built, which is what
        # a cache has to ask before trusting a stored feature map.
        from TissueMaskConfig import TissueMaskConfig
        self.mask_cfg            = mask_cfg or TissueMaskConfig()
        self.feature_store_root  = feature_store_root
        self.feature_store_mode  = feature_store_mode
        self.knn_samples         = knn_samples
        self.knn_k               = knn_k
        self.knn_seed            = knn_seed
        self.knn_tissue_ratio    = knn_tissue_ratio
        self.retriever_overlap   = retriever_overlap
        self.refiner_min_inliers = refiner_min_inliers
        self.refiner_padding     = refiner_padding

        # SafeSlide.base_mpp: the mean of mpp-x and mpp-y, with an aperio
        # fallback. This line used to read mpp-x alone, which disagreed with
        # QueryFromWSI on every slide in use.
        self.base_mpp = wsi.base_mpp   # raises if the slide carries no mpp

        self.mask:      Optional[TissuesRegionsMask] = None
        self.estimator: Optional[GigaPathKnnEstiMpp] = None
        # None value == "tried, unusable"; missing key == "not tried yet"
        self._retrievers: Dict[int, Optional[GigaPathSlidingWinSimRot]] = {}
        self._retriever_reason: Dict[int, str] = {}   # why a level is unusable

    # ── One-time setup ────────────────────────────────────────────────────────
    def build(self) -> 'LocaScopePipeline':
        """Build mask + mpp KNN reference bank (per-WSI one-time)."""
        # Segment, filter and merge in one place and in one order. merge is
        # incomplete without filter having run first -- it skips nested boxes on
        # the assumption they are already gone -- and that dependency used to be
        # two lines every caller wrote out.
        self.mask = self.mask_cfg.build(
            self.wsi, getattr(self.encoder, 'device', None))

        self.estimator = GigaPathKnnEstiMpp(
            self.wsi, encoder=self.encoder, mask=self.mask,
            tile_size=self.tile_size,
            samples_per_level=self.knn_samples, k=self.knn_k,
            seed=self.knn_seed, tissue_ratio=self.knn_tissue_ratio,
        )
        self.estimator.build_samples()
        self.estimator.build_ref_features()
        return self

    # ── Lazy per-level retriever cache ────────────────────────────────────────
    #
    # `_level_mask` used to live here: a per-level copy of the mask keeping only
    # regions that can host a tile. It was TissuesRegionsMask.filter_patchable
    # written out a second time, because that method mutates in place and this
    # needed a copy. Both halves moved into the library -- `regions_view()` for
    # the copy, and `WsiTissuesContainer.from_ds` for the filter -- so the
    # retriever now narrows the mask itself, at the ds it is actually going to
    # build at. Which is the point: this class did not know that ds, it only
    # knew the one it was asking for.

    def _feature_store(self):
        '''The cache for this slide, or None when no root was given.

        Built here and nowhere else because mask_id needs the whole recipe --
        segmentation method, its ds, the region filter, whether merging ran --
        and this is the only object that holds all of it.
        '''
        if not self.feature_store_root:
            return None
        from WsiFeaturesMapStore import WsiFeaturesMapStore
        return WsiFeaturesMapStore(
            self.feature_store_root,
            getattr(self.wsi, '_filename', self.mask_cfg and ''),
            self.encoder, self.mask_cfg.mask_id(),
            mode=self.feature_store_mode)

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

        # Stage 1 — estimate mpp
        try:
            r1 = self.estimator.estimate(img_np, overlap=True)
            est_mpp = float(r1.estimated_mpp)
        except Exception as e:
            return LocaScopeQueryResult(
                None, None, False, None, None,
                f'stage1 failed: {type(e).__name__}: {e}')

        # Route est_mpp to the closest pyramid level (fuzzy snap)
        try:
            ds_target = est_mpp / self.base_mpp
            level = self.wsi.get_best_level_for_downsample(ds_target)
        except Exception as e:
            return LocaScopeQueryResult(
                est_mpp, None, False, None, None,
                f'level routing failed: {type(e).__name__}: {e}')

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
                wsi_container=retriever.wsi_container,
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
