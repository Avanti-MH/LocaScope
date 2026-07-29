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
"""

from __future__ import annotations

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
from PatchingLib          import QueryPatchContainer, WsiTissuesContainer, FeaturesMap    # noqa: E402
from TissuesRegionsMask   import TissuesRegionsMask                          # noqa: E402
from GigaPathSlideWinSim  import SlidingWindowSimilarity                     # noqa: E402


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
    ):
        if isinstance(wsi, str):
            wsi = openslide.OpenSlide(wsi)
        self.wsi       = wsi
        self.encoder   = encoder
        self.mask      = mask
        self.mpp       = mpp
        self.tile_size = tile_size
        self.overlap   = overlap

        self.wsi_container: Optional[WsiTissuesContainer]                    = None
        self.wsi_features:  Optional[list[FeaturesMap]]                      = None
        # Per-rotation state (dict keyed by rotation degree)
        self.qc_by_rot:            Dict[int, QueryPatchContainer]                                = {}
        self.query_features_by_rot: Dict[int, FeaturesMap]                                       = {}
        self.sim_maps_by_rot:      Dict[int, list[tuple[torch.Tensor, torch.Tensor]]]            = {}
        self.result: Optional[SlideWinSimRotResult]                                              = None

    # ── Stage 1: WSI features (single orientation) ───────────────────────────
    def build_wsi_features(self, mpp: Optional[float] = None) -> list[FeaturesMap]:
        mpp = mpp or self.mpp
        if mpp is None:
            raise ValueError('mpp must be provided in __init__ or build_wsi_features()')
        if self.mask is None:
            self.mask = TissuesRegionsMask.from_wsi(self.wsi)
        self.wsi_container = WsiTissuesContainer.from_mpp(
            self.wsi, mpp, tile_size=self.tile_size, overlap=self.overlap, mask=self.mask,
        )
        self.wsi_features = [tp.to_features(self.encoder) for tp in self.wsi_container]
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
    def _find_best_in_grid(
        self,
        sim_maps:    list[tuple[torch.Tensor, torch.Tensor]],
        use_overlap: bool,
    ) -> tuple[int, int, float, int]:
        """(x @ level-n, y @ level-n, score, region_index) of the strongest window."""
        ds   = self.wsi_container.ds
        half = self.tile_size // 2
        best_score = -float('inf')
        best_x = best_y = 0
        best_region_idx = 0
        for ri, (region, (main_sim, overlap_sim)) in enumerate(
            zip(self.mask.tissue_regions, sim_maps)
        ):
            hm = overlap_sim if use_overlap else main_sim
            if hm.numel() == 0:
                continue
            hm_mean = hm.mean(dim=(-2, -1))
            idx = int(hm_mean.argmax())
            r, c = divmod(idx, hm_mean.shape[1])
            score = float(hm_mean[r, c])
            if score > best_score:
                best_score = score
                x_off = half if use_overlap else 0
                y_off = half if use_overlap else 0
                best_x = int(region.x / ds) + c * self.tile_size + x_off
                best_y = int(region.y / ds) + r * self.tile_size + y_off
                best_region_idx = ri
        return best_x, best_y, best_score, best_region_idx

    def find_best(self) -> SlideWinSimRotResult:
        """Best match across (4 rotations) x (main + overlap grids)."""
        if not self.sim_maps_by_rot:
            self.compute_sim_maps()

        ds = self.wsi_container.ds

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
