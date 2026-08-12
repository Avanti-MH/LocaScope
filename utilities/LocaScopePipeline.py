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

import copy
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
        mask_method:         Callable = None,
        mask_ds:             float = 4.0,
        mask_seg_chunk_px:   int   = 4_000_000,
        mask_overlap:        int   = 128,
        min_region_ratio:    float = 0.01,
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
        self.mask_method         = mask_method
        self.mask_ds             = mask_ds
        self.mask_seg_chunk_px   = mask_seg_chunk_px
        self.mask_overlap        = mask_overlap
        self.min_region_ratio    = min_region_ratio
        self.knn_samples         = knn_samples
        self.knn_k               = knn_k
        self.knn_seed            = knn_seed
        self.knn_tissue_ratio    = knn_tissue_ratio
        self.retriever_overlap   = retriever_overlap
        self.refiner_min_inliers = refiner_min_inliers
        self.refiner_padding     = refiner_padding

        self.base_mpp = float(wsi.properties.get('openslide.mpp-x', 0))
        if self.base_mpp == 0:
            raise ValueError('WSI missing openslide.mpp-x metadata')

        self.mask:      Optional[TissuesRegionsMask] = None
        self.estimator: Optional[GigaPathKnnEstiMpp] = None
        # None value == "tried, unusable"; missing key == "not tried yet"
        self._retrievers: Dict[int, Optional[GigaPathSlidingWinSimRot]] = {}
        self._retriever_reason: Dict[int, str] = {}   # why a level is unusable

    # ── One-time setup ────────────────────────────────────────────────────────
    def build(self) -> 'LocaScopePipeline':
        """Build mask + mpp KNN reference bank (per-WSI one-time)."""
        self.mask = TissuesRegionsMask.from_wsi(self.wsi,
                                                ds=self.mask_ds,
                                                method=self.mask_method,
                                                seg_chunk_px=self.mask_seg_chunk_px,
                                                overlap=self.mask_overlap,
                                                )
        self.mask.filter_regions(min_ratio=self.min_region_ratio)
        self.mask.merge_overlapping()

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
    def _level_mask(self, level: int) -> TissuesRegionsMask:
        """A per-level view of the mask keeping only regions that can host a tile.

        `filter_regions` gates on AREA only, so a long thin region (say
        5000x100 px) survives it while being too short to yield a single
        256 px tile — `extract_all` then returns zero patches and the encoder
        raises on an empty batch.

        This returns a shallow copy with its own `tissue_regions` list rather
        than mutating `self.mask`: each cached retriever keeps a reference to
        its mask and re-reads `mask.tissue_regions` in `find_best()`, so
        filtering the shared mask in place would desync already-built
        retrievers from their similarity maps.
        """
        ds      = self.wsi.level_downsamples[level]
        tile_l0 = self.tile_size * ds
        lvl = copy.copy(self.mask)
        lvl._regions_history = []
        lvl.tissue_regions = [
            r for r in self.mask.tissue_regions
            if r.w >= tile_l0 and r.h >= tile_l0
        ]
        return lvl

    def _get_retriever(self, level: int) -> Optional[GigaPathSlidingWinSimRot]:
        """Return cached rotation-aware retriever for this level, or None if unusable."""
        if level in self._retrievers:
            return self._retrievers[level]

        level_mpp = self.base_mpp * self.wsi.level_downsamples[level]
        lvl_mask  = self._level_mask(level)
        n_all, n_ok = len(self.mask.tissue_regions), len(lvl_mask.tissue_regions)
        print(f'  [retriever L{level}] mpp={level_mpp:.4f}  '
              f'regions {n_ok}/{n_all} patchable', flush=True)

        if n_ok == 0:
            reason = (f'no tissue region can host a {self.tile_size}px tile '
                      f'at level {level}')
            print(f'  [retriever L{level}] UNUSABLE: {reason}', flush=True)
            self._retrievers[level] = None
            self._retriever_reason[level] = reason
            return None

        try:
            r = GigaPathSlidingWinSimRot(
                self.wsi, encoder=self.encoder, mask=lvl_mask,
                mpp=level_mpp, tile_size=self.tile_size,
                overlap=self.retriever_overlap,
            )
            r.build_wsi_features()
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
