"""Rotation-aware GigaPath sliding-window retrieval.

Same staged API as GigaPathSlidingWinSim but tries all 4 cardinal rotations
(0, 90, 180, 270 deg) of the query image and picks the one with the highest
similarity score. Downstream refinement (SIFT+RANSAC) can read
`result.best_rotation` and rotate the query image for alignment.

Cost:
    4x query patch extraction + encoding + sim-map computation.
    WSI features are encoded ONCE and shared across the 4 query orientations.

Usage:
    r = GigaPathSlidingWinSimRot(wsi, encoder, mask=mask, mpp=level_mpp,
                                 tile_size=256, overlap=True)
    r.build_wsi_features()
    r.build_query_features(shot_img)
    r.compute_sim_maps()
    result = r.find_best()    # SlideWinSimRotResult with .best_rotation
    cands  = r.top_k(20)      # the same ranking, truncated instead of collapsed
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'utilities'))
sys.path.insert(0, str(ROOT / 'aiNNModel'))

import openslide                                                            # noqa: E402
from PatchingLib          import (QueryPatchContainer, WsiTissuesContainer,   # noqa: E402
                                  FeaturesMap, WsiFeaturesMap)
from SafeSlide            import SafeSlide                                                # noqa: E402
from TissuesRegionsMask   import TissueRegion, TissuesRegionsMask            # noqa: E402
from GigaPathSlidingWinSim  import SlidingWindowSimilarity                   # noqa: E402


# ── Result dataclass (adds best_rotation vs the base module's result) ─────────

@dataclass(frozen=True)
class SlideWinSimRotResult:
    # Best match — union of (rotation, main / overlap grid) with highest score
    best_x:             int          # top-left X @ level-n (rotated-query space)
    best_y:             int
    best_x0:            int          # top-left X @ level-0
    best_y0:            int
    best_score:         float
    from_overlap:       bool
    best_region_index:  int
    best_rotation:      int          # 0 / 90 / 180 / 270 (deg) that produced this match
    ds:                 float
    # main / overlap grid bests OF THE WINNING ROTATION (debug / visualization)
    main_x:               int
    main_y:               int
    main_x0:              int
    main_y0:              int
    main_score:           float
    main_region_index:    int
    overlap_x:            int
    overlap_y:            int
    overlap_x0:           int
    overlap_y0:           int
    overlap_score:        float
    overlap_region_index: int
    # Per-rotation winning scores (for debugging / picking margins)
    scores_by_rotation: Dict[int, float]


@dataclass(frozen=True)
class SlideWinSimCandidate:
    """One scored window from top_k, in the same coordinates as the result.

    Geometry and score only -- no notion of being right; ranking a candidate
    against a truth is the caller's business. Rank 1 carries the same x/y/x0/y0
    and score find_best returns.

    win_w0 and win_h0 come from the ROTATED query's tile grid, so they are
    already transposed for the 90 and 270 steps and want no further swap. They
    are a whole number of tiles and therefore not the query footprint, which is
    not; a caller that knows the true footprint should prefer its own.

    The best_* aliases below make a candidate usable directly as
    SiftRansacLocalizer's `location`. That refiner reads exactly five
    attributes -- best_region_index, best_x, best_y, ds, best_rotation
    (SIFT_RANSAC.py:127-290, the last one via getattr) -- and already accepts
    either result type by duck typing, so verifying a candidate is a matter of
    handing it over rather than of building a fake result around it.
    """
    rank:          int        # 1-based, by descending score
    score:         float
    rotation:      int        # 0 / 90 / 180 / 270
    from_overlap:  bool       # main grid or the half-tile-shifted one
    region_index:  int
    x:             int        # top-left @ level-n
    y:             int
    x0:            int        # top-left @ level-0
    y0:            int
    win_w0:        int        # window footprint @ level-0, rotated query grid
    win_h0:        int
    ds:            float      # level-n downsample, as on the result

    @property
    def best_x(self) -> int:
        return self.x

    @property
    def best_y(self) -> int:
        return self.y

    @property
    def best_region_index(self) -> int:
        return self.region_index

    @property
    def best_rotation(self) -> int:
        return self.rotation


# ── Pipeline class ────────────────────────────────────────────────────────────

class GigaPathSlidingWinSimRot:
    """Rotation-aware version of GigaPathSlidingWinSim.

    Stages (same shape as base class):
        1. build_wsi_features(mpp)   — tile WSI → encode  (once)
        2. build_query_features()    — extract patches from 4 rotations → encode 4x
        3. compute_sim_maps()        — 4 sim-map sets
        4. find_best()               — best (rotation, position) across all 4
    """

    ROTATIONS = (0, 90, 180, 270)

    def __init__(
        self,
        wsi:       Union[openslide.OpenSlide, str],
        encoder:   Callable,
        mask:      Optional[TissuesRegionsMask] = None,
        mpp:       Optional[float]              = None,
        tile_size: int                          = 256,
        overlap:   bool                         = True,
        feature_store                           = None,
    ):
        # SafeSlide so a hole in a MIRAX cannot kill the handle mid-run; see
        # utilities/SafeSlide.py. Only reached when constructed from a path --
        # LocaScopePipeline passes an already-open slide.
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi       = wsi
        self.encoder   = encoder
        self.mask      = mask
        self.mpp       = mpp
        self.tile_size = tile_size
        self.overlap   = overlap
        #: Optional. Anything with load(container) -> WsiFeaturesMap | None and
        #: save(WsiFeaturesMap). This class does not know about paths, ids or
        #: safetensors -- it asks, and rebuilds when the answer is None, which
        #: is also what happens when the store decides it does not match.
        self.feature_store = feature_store

        # Scale of the CURRENT build. `mpp` above is what the caller asked for;
        # these are what was actually used, and they are what everything
        # downstream reads.
        self.level:   Optional[int]                = None
        self.ds:      Optional[float]              = None
        self.regions: Optional[list[TissueRegion]] = None
        self.wsi_container: Optional[WsiTissuesContainer] = None
        self.wsi_features:  Optional[list[FeaturesMap]]   = None
        #: {ds: (container, features)} for every scale built so far.
        #:
        #: The container is kept, not released after encoding. It holds each
        #: region's pixels, which looks like the obvious thing to drop -- 19.7
        #: GB for a level-0 BRACS region -- but stage 3 reads them:
        #: SiftRansacLocalizer takes `retriever.wsi_container`
        #: (LocaScopePipeline.py:264) and crops the matched window out of
        #: `wsi_container[best_region_index].img` to run SIFT on it. Dropping
        #: it hands stage 3 a None.
        #:
        #: Cached as a pair because the features, the region list and the
        #: pixels all belong to one scale, and find_best zips the first two.
        self._by_ds: Dict[float, tuple] = {}

        # Per-rotation state (dict keyed by rotation degree)
        self.qc_by_rot:            Dict[int, QueryPatchContainer]                                = {}
        self.query_features_by_rot: Dict[int, FeaturesMap]                                       = {}
        self.sim_maps_by_rot:      Dict[int, list[tuple[torch.Tensor, torch.Tensor]]]            = {}
        self.result: Optional[SlideWinSimRotResult]                                              = None

    # ── Stage 1: WSI features (single orientation) ───────────────────────────
    def build_wsi_features(self, mpp: Optional[float] = None,
                           ds: Optional[float] = None) -> list[FeaturesMap]:
        """Encode every usable tissue region at one scale.

        Two ways in, and the difference is only what gets recomputed:

            first call     `mask` may be None, and then the tissue mask is
                           segmented here -- about a minute, and by far the
                           most expensive thing this method does.
            later call     a different scale, given as `mpp` or `ds`. The
                           segmentation is reused. The same scale a second
                           time reuses the features too.

        `self.mask` is the caller's mask and is never replaced. `from_ds` takes
        a `regions_view()` of it and filters THAT, so every build starts from
        the whole segmentation. It has to: filtering is monotone in ds, so a
        build at 1.0 that narrowed the original in place would leave a later
        build at 0.25 unable to see the regions the coarse pass dropped.
        0.25 -> 1.0 -> 0.25 must return the same regions as a fresh 0.25.

        `ds` must land on a level the slide actually has; a request between
        levels is snapped to the nearest and reported (WsiTissuesContainer).
        """
        if mpp is None and ds is None:
            mpp = self.mpp
            if mpp is None:
                raise ValueError('mpp must be provided in __init__ or build_wsi_features()')

        # Resolve first: it is free, and it gives the cache key before anything
        # is read or encoded.
        self.level, self.ds = WsiTissuesContainer.resolve_scale(
            self.wsi, mpp=mpp, ds=ds)
        self.mpp = self.wsi.base_mpp * self.ds   # mpp/ds/level now say one thing

        if self.mask is None:
            # '' and not a threshold: this scores a window by the mean cosine
            # over the query's tiles, so a window on blank glass loses on its
            # own merits and the mask is an optimisation here, not a
            # correctness requirement. '' skips the read AND the array.
            from TissueSegFunc import TissueSegConfig
            self.mask = TissuesRegionsMask.from_wsi(
                self.wsi, method=TissueSegConfig('').build())
            print(f'  [Rot] no mask given; one region over the whole scanned '
                  f'rectangle, which is NOT what LocaScopePipeline uses '
                  f'(HEST, mask_ds=4, filter_regions, merge_overlapping). '
                  f'{len(self.mask.tissue_regions)} regions, '
                  f'tissue {self.mask.tissue_fraction() * 100:.1f}%', flush=True)

        # Three ways to end up with features, cheapest first.
        if self.ds in self._by_ds:
            self.wsi_container, self.wsi_features = self._by_ds[self.ds]
        else:
            # The container is built either way. It is not only the source of
            # the tiles -- stage 3 reads pixels back out of it
            # (SIFT_RANSAC.py:150), so a cache hit skips the ENCODE and not the
            # read. Half of a level-0 build, measured: 278s read + 285s encode
            # on BRACS_1228. The other half needs lazy region reads; see
            # log/TODO.log.
            self.wsi_container = WsiTissuesContainer.from_ds(
                self.wsi, self.ds, tile_size=self.tile_size,
                overlap=self.overlap, mask=self.mask)

            self.wsi_features = None
            if self.feature_store is not None:
                self.wsi_features = self.feature_store.load(self.wsi_container)

            if self.wsi_features is None:
                self.wsi_features = self.wsi_container.to_features(self.encoder)
                if self.feature_store is not None:
                    self.feature_store.save(self.wsi_features)

            self._by_ds[self.ds] = (self.wsi_container, self.wsi_features)
        # `regions` is the container's list -- the FILTERED one the features and
        # similarity maps line up with, not self.mask's. Kept as an attribute
        # because find_best zips it and reaching two levels down to say so
        # obscures which list is meant.
        self.regions = self.wsi_container.tissue_regions

        # Similarity maps are query features x WSI features, so they belong to
        # the scale that produced them. compute_sim_maps() already rebuilds
        # from scratch; the reason to empty this is find_best(), which only
        # calls it when the dict is EMPTY and would otherwise pair last scale's
        # maps with this scale's regions.
        self.sim_maps_by_rot = {}
        self.result = None
        return self.wsi_features

    # ── Stage 2: query features at 4 rotations ───────────────────────────────
    @staticmethod
    def _rotate_np(img: np.ndarray, rot_deg: int) -> np.ndarray:
        """Lossless 90-deg-step rotation of an RGB image (H, W, C)."""
        if rot_deg == 0:
            return img
        if rot_deg == 90:
            return np.rot90(img, k=1)
        if rot_deg == 180:
            return np.rot90(img, k=2)
        if rot_deg == 270:
            return np.rot90(img, k=3)
        raise ValueError(f'unsupported rotation {rot_deg}; must be 0/90/180/270')

    def build_query_features(
        self,
        query: Union[QueryPatchContainer, Image.Image, np.ndarray],
    ) -> Dict[int, FeaturesMap]:
        """Extract + encode query patches at each of 4 cardinal rotations."""
        if isinstance(query, QueryPatchContainer):
            img_np = np.array(query.img)
        elif isinstance(query, Image.Image):
            img_np = np.array(query.convert('RGB'))
        else:
            img_np = np.asarray(query)

        self.qc_by_rot            = {}
        self.query_features_by_rot = {}
        for rot in self.ROTATIONS:
            rotated = self._rotate_np(img_np, rot)
            qc = QueryPatchContainer(rotated)
            qc.extract_all(tile_size=self.tile_size, overlap=self.overlap)
            self.qc_by_rot[rot]             = qc
            self.query_features_by_rot[rot] = qc.to_features(self.encoder)
        return self.query_features_by_rot

    # ── Stage 3a: sim maps for each rotation ─────────────────────────────────
    def compute_sim_maps(self) -> Dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
        if self.wsi_features is None:
            raise RuntimeError('call build_wsi_features() first')
        if not self.query_features_by_rot:
            raise RuntimeError('call build_query_features() first')
        self.sim_maps_by_rot = {}
        for rot, qfm in self.query_features_by_rot.items():
            self.sim_maps_by_rot[rot] = [
                SlidingWindowSimilarity(qfm, wf) for wf in self.wsi_features
            ]
        return self.sim_maps_by_rot

    # ── Stage 3b: helpers reused across rotations ────────────────────────────
    def _window_xy(self, region, r: int, c: int,
                   use_overlap: bool) -> tuple[int, int]:
        """Top-left @ level-n of grid cell (r, c) inside `region`.

        The overlap grid is the main grid shifted half a tile on both axes;
        that offset is the only difference between them. It lives here alone
        because the same formula written out twice is how one code path ends up
        reporting a position half a tile from the other and nothing says so.
        """
        ds  = self.ds          # the built level's own downsample
        off = self.tile_size // 2 if use_overlap else 0
        return (int(region.x / ds) + c * self.tile_size + off,
                int(region.y / ds) + r * self.tile_size + off)

    def _grid_means(
        self,
        sim_maps:    list[tuple[torch.Tensor, torch.Tensor]],
        use_overlap: bool,
        ):
        """Yield (region_index, region, per-window mean-cosine grid).

        The score of a window is the mean cosine over the query's tiles, so the
        two trailing dims collapse and what is left is one score per placement.
        Empty grids are skipped: a region smaller than the query produces none.
        """
        for ri, ((region, _), (main_sim, overlap_sim)) in enumerate(
            # wsi_features.items(), not self.mask.tissue_regions: the maps
            # were computed over the FILTERED regions and the caller's mask
            # still holds all of them. The WsiFeaturesMap carries its own
            # regions, so there is no longer a second list to pick wrongly
            # from -- which is what this comment used to be guarding.
            zip(self.wsi_features.items(), sim_maps)
        ):
            hm = overlap_sim if use_overlap else main_sim
            if hm.numel() == 0:
                continue
            yield ri, region, hm.mean(dim=(-2, -1))

    def _find_best_in_grid(
        self,
        sim_maps:    list[tuple[torch.Tensor, torch.Tensor]],
        use_overlap: bool,
        ) -> tuple[int, int, float, int]:
        """(x @ level-n, y @ level-n, score, region_index) of the strongest window."""
        best_score = -float('inf')
        best_x = best_y = 0
        best_region_idx = 0
        for ri, region, hm_mean in self._grid_means(sim_maps, use_overlap):
            idx = int(hm_mean.argmax())
            r, c = divmod(idx, hm_mean.shape[1])
            score = float(hm_mean[r, c])
            if score > best_score:
                best_score = score
                best_x, best_y = self._window_xy(region, r, c, use_overlap)
                best_region_idx = ri
        return best_x, best_y, best_score, best_region_idx

    def top_k(self, k: int = 20,
              min_sep_px: Optional[float] = None) -> list[SlideWinSimCandidate]:
        """The k best-scoring windows across every rotation, grid and region.

        Free. compute_sim_maps already produced every score this reads and
        find_best discards all but one of them; nothing is encoded again.

        Exists to answer a question find_best cannot: when the winner is wrong,
        is the truth further down the list or absent from it? Those two call for
        opposite fixes -- verify K candidates geometrically, versus repair the
        features so the right window scores higher at all.

        min_sep_px suppresses a candidate whose top-left is within that many
        LEVEL-0 pixels of one already kept AT THE SAME ROTATION. A strong peak
        otherwise fills the list with itself, seen from the main grid and from
        the half-tile-shifted one, so a nominal k of 5 can be two real places.
        Suppression is per-rotation on purpose: the same spot at two
        orientations is two hypotheses, not a duplicate, and this retriever's
        rotation vote is exactly what is not trusted. Defaults to one tile at
        level-0; pass 0 for the raw ranking.

        Rank 1 is the find_best winner, recomputed here rather than read off
        self.result so top_k does not depend on find_best having run.
        """
        if k <= 0:
            return []
        if not self.sim_maps_by_rot:
            self.compute_sim_maps()

        ds = self.ds           # the built level's own downsample
        if min_sep_px is None:
            min_sep_px = self.tile_size * ds

        # topk per grid rather than over the union: a slide holds far more
        # windows than fit in memory as a Python list, and only k of them can
        # survive from any single grid anyway.
        raw: list[tuple] = []
        for rot, sim_maps in self.sim_maps_by_rot.items():
            for use_overlap in (False, True):
                for ri, region, hm_mean in self._grid_means(sim_maps, use_overlap):
                    flat = hm_mean.reshape(-1)
                    vals, idxs = torch.topk(flat, min(k, flat.numel()))
                    n_cols = hm_mean.shape[1]
                    for v, i in zip(vals.tolist(), idxs.tolist()):
                        r, c = divmod(int(i), n_cols)
                        x, y = self._window_xy(region, r, c, use_overlap)
                        raw.append((float(v), rot, use_overlap, ri, x, y))
        raw.sort(key=lambda t: -t[0])

        out: list[SlideWinSimCandidate] = []
        for score, rot, use_overlap, ri, x, y in raw:
            x0, y0 = int(x * ds), int(y * ds)
            if min_sep_px > 0 and any(
                c.rotation == rot
                and math.hypot(x0 - c.x0, y0 - c.y0) < min_sep_px
                for c in out
            ):
                continue
            grid = self.qc_by_rot[rot].grid
            out.append(SlideWinSimCandidate(
                rank         = len(out) + 1,
                score        = score,
                rotation     = rot,
                from_overlap = use_overlap,
                region_index = ri,
                x = x, y = y, x0 = x0, y0 = y0,
                win_w0 = int(grid.grid_cols * self.tile_size * ds),
                win_h0 = int(grid.grid_rows * self.tile_size * ds),
                ds     = ds,
            ))
            if len(out) == k:
                break
        return out

    def find_best(self) -> SlideWinSimRotResult:
        """Best match across (4 rotations) x (main + overlap grids)."""
        if not self.sim_maps_by_rot:
            self.compute_sim_maps()

        ds = self.ds           # the built level's own downsample

        # Per-rotation main / overlap winners
        scores_by_rot: Dict[int, float] = {}
        per_rot: Dict[int, tuple] = {}
        for rot, sim_maps in self.sim_maps_by_rot.items():
            m = self._find_best_in_grid(sim_maps, use_overlap=False)   # (x, y, score, ri)
            o = self._find_best_in_grid(sim_maps, use_overlap=True)
            per_rot[rot] = (m, o)
            scores_by_rot[rot] = max(m[2], o[2])

        # Overall winner across rotations
        best_rot = max(scores_by_rot, key=scores_by_rot.get)
        (mx, my, m_s, m_ri), (ox, oy, o_s, o_ri) = per_rot[best_rot]

        from_overlap = o_s > m_s
        bx, by, bs, b_ri = (ox, oy, o_s, o_ri) if from_overlap else (mx, my, m_s, m_ri)

        # No region produced a single valid window: the sliding kernel needs the
        # WSI region grid to be at least as large as the query grid, and every
        # region here is smaller. _find_best_in_grid would otherwise hand back
        # its (0, 0) sentinel, which downstream code cannot distinguish from a
        # real match at the slide origin.
        if bs == -float('inf'):
            qc = self.qc_by_rot[best_rot]
            q_r, q_c = qc.grid.grid_rows, qc.grid.grid_cols
            biggest = max(
                ((fm.main_feature_grid().shape[0], fm.main_feature_grid().shape[1])
                 for fm in self.wsi_features),
                default=(0, 0),
            )
            raise ValueError(
                f'query does not fit any tissue region at this level: query grid '
                f'is {q_r}x{q_c} tiles ({self.tile_size}px each), the largest '
                f'region grid is {biggest[0]}x{biggest[1]}. Route to a finer '
                f'level, lower the query MPixels, or relax the mask filtering.'
            )

        self.result = SlideWinSimRotResult(
            best_x             = bx,
            best_y             = by,
            best_x0            = int(bx * ds),
            best_y0            = int(by * ds),
            best_score         = bs,
            from_overlap       = from_overlap,
            best_region_index  = b_ri,
            best_rotation      = best_rot,
            ds                 = ds,
            main_x    = mx, main_y  = my,
            main_x0   = int(mx * ds), main_y0 = int(my * ds),
            main_score= m_s, main_region_index = m_ri,
            overlap_x = ox, overlap_y = oy,
            overlap_x0= int(ox * ds), overlap_y0= int(oy * ds),
            overlap_score = o_s, overlap_region_index = o_ri,
            scores_by_rotation = scores_by_rot,
        )
        return self.result
