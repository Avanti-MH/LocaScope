
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union, Optional

import numpy as np
from PIL import Image

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'utilities'))
sys.path.insert(0, str(ROOT / 'aiNNModel'))

import openslide
from PatchingLib import QueryPatchContainer, WsiTissuesContainer, FeaturesMap
from TissuesRegionsMask import TissuesRegionsMask
from GigaPathFunc import gigapath_model, gigapath_encode

def _sim_tensors(q_grid: torch.Tensor, wsi_grid: torch.Tensor) -> torch.Tensor:
    '''
    Core unfold similarity: [R_q, C_q, D] × [R_w, C_w, D] → [H_out, W_out, R_q, C_q].
    Returns empty tensor when wsi is smaller than query.
    '''
    R_q, C_q, _ = q_grid.shape
    R_w, C_w, _ = wsi_grid.shape
    if R_w < R_q or C_w < C_q:
        return torch.empty(0)
    wsi     = wsi_grid.permute(2, 0, 1)
    windows = wsi.unfold(1, R_q, 1).unfold(2, C_q, 1)        # [D, H_out, W_out, R_q, C_q]
    q       = q_grid.permute(2, 0, 1)
    return (windows * q[:, None, None, :, :]).sum(dim=0)      # [H_out, W_out, R_q, C_q]


def SlidingWindowSimilarity(
    qFeatureMap: FeaturesMap,
    WsiFeatureMap: FeaturesMap,
) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Slide qFeatureMap (kernel) over WsiFeatureMap (input), computing per-patch cosine similarity.

    Features must be L2-normalized. Uses combinations 1+3: both searches use the main query
    kernel so scores are directly comparable across grids.

    Returns (main_sim, overlap_sim):
      main_sim    shape [H_out,   W_out,   R_q, C_q]  WSI main grid,    origin (region.x, region.y)
      overlap_sim shape [H_out-1, W_out-1, R_q, C_q]  WSI overlap grid, origin (region.x + tile/2·ds, ...)
                        (empty tensor when WSI has no overlap patches)
      H_out = R_wsi - R_q + 1,  W_out = C_wsi - C_q + 1
    '''
    q_grid  = qFeatureMap.main_feature_grid()       # fixed: always main query kernel
    main_sim = _sim_tensors(q_grid, WsiFeatureMap.main_feature_grid())

    wsi_ov  = WsiFeatureMap.overlap_feature_grid()
    overlap_sim = _sim_tensors(q_grid, wsi_ov) if wsi_ov.numel() > 0 \
                  else torch.empty(0)

    return main_sim, overlap_sim


def GigaPathSlideWinSim(
    query: QueryPatchContainer,
    Wsi: Union[openslide.OpenSlide, WsiTissuesContainer],
    mpp: float,
    tile_size: int = 256,
    overlap: bool = True,
    mask: Optional[TissuesRegionsMask] = None,
    encoder: Optional[callable] = None,
    batch_size: int = 128,
) -> list[tuple[torch.Tensor, torch.Tensor]]:

    if isinstance(Wsi, openslide.OpenSlide):
        WsiTissuesPathes = WsiTissuesContainer.from_mpp(Wsi, mpp, tile_size=tile_size, overlap=overlap, mask=mask)
    elif isinstance(Wsi, WsiTissuesContainer):
        WsiTissuesPathes = Wsi
    else:
        raise ValueError("Wsi must be an openslide.OpenSlide or WsiTissuesContainer")

    if not isinstance(query, QueryPatchContainer):
        raise ValueError("Query must be a QueryPatchContainer")
    if len(query) == 0:
        query.extract_all(tile_size=tile_size, overlap=overlap)

    if encoder is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = gigapath_model(device)
        encoder = lambda patches: gigapath_encode(patches, model, device, batch_size=batch_size)

    WsiRegionsFeatures = [tp.to_features(encoder) for tp in WsiTissuesPathes]
    QueryFeatures = query.to_features(encoder)

    return [SlidingWindowSimilarity(QueryFeatures, wf) for wf in WsiRegionsFeatures]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SlideWinSimResult:
    # best match (union of main and overlap, whichever scored higher)
    best_x: int          # top-left X @ level-n
    best_y: int          # top-left Y @ level-n
    best_x0: int         # top-left X @ level-0
    best_y0: int         # top-left Y @ level-0
    best_score: float
    from_overlap: bool   # True if best came from the overlap grid
    best_region_index: int
    ds: float            # downsample factor (level-n pixel size in level-0 pixels)
    # main grid best (for debug / visualization)
    main_x: int
    main_y: int
    main_x0: int
    main_y0: int
    main_score: float
    main_region_index: int
    # overlap grid best (for debug / visualization)
    overlap_x: int
    overlap_y: int
    overlap_x0: int
    overlap_y0: int
    overlap_score: float
    overlap_region_index: int


# ── Pipeline class ────────────────────────────────────────────────────────────

class GigaPathSlidingWinSim:
    '''
    Staged GigaPath sliding-window similarity search.

    Stages (in order):
        1. build_wsi_features(mpp)  — tile WSI at mpp → encode → FeaturesMap per region
        2. build_query_features()   — extract query patches → encode → FeaturesMap
        3. compute_sim_maps()       — SlidingWindowSimilarity per region → sim_maps
        4. find_best()              — find best match → SlideWinSimResult

    All intermediate state is stored on self for debugging and visualization.
    Stages that depend on earlier ones are built automatically if not called yet.
    '''

    def __init__(
        self,
        wsi: Union[openslide.OpenSlide, str],
        encoder: Callable,
        mask: Optional[TissuesRegionsMask] = None,
        mpp: Optional[float] = None,
        tile_size: int = 256,
        overlap: bool = True,
    ):
        if isinstance(wsi, str):
            wsi = openslide.OpenSlide(wsi)
        self.wsi = wsi
        self.encoder = encoder
        self.mask = mask
        self.mpp = mpp
        self.tile_size = tile_size
        self.overlap = overlap

        # Intermediate state — inspect at any stage for debugging / visualization
        self.wsi_container: Optional[WsiTissuesContainer] = None
        self.wsi_features: Optional[list[FeaturesMap]] = None
        self.qc: Optional[QueryPatchContainer] = None
        self.query_features: Optional[FeaturesMap] = None
        self.sim_maps: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self.result: Optional[SlideWinSimResult] = None

    # ── Stage 1 ──────────────────────────────────────────────────────────────

    def build_wsi_features(self, mpp: Optional[float] = None) -> list[FeaturesMap]:
        '''Tile WSI at the given mpp (falls back to self.mpp), encode each region.'''
        mpp = mpp or self.mpp
        if mpp is None:
            raise ValueError('mpp must be provided in __init__ or build_wsi_features()')
        if self.mask is None:
            self.mask = TissuesRegionsMask.from_wsi(self.wsi)
        self.wsi_container = WsiTissuesContainer.from_mpp(
            self.wsi, mpp, tile_size=self.tile_size, overlap=self.overlap, mask=self.mask
        )
        self.wsi_features = [tp.to_features(self.encoder) for tp in self.wsi_container]
        return self.wsi_features

    # ── Stage 2 ──────────────────────────────────────────────────────────────

    def build_query_features(
        self,
        query: Union[QueryPatchContainer, Image.Image, np.ndarray],
    ) -> FeaturesMap:
        '''Extract query patches then encode.'''
        if isinstance(query, (Image.Image, np.ndarray)):
            query = QueryPatchContainer(query)
        if len(query) == 0:
            query.extract_all(tile_size=self.tile_size, overlap=self.overlap)
        self.qc = query
        self.query_features = self.qc.to_features(self.encoder)
        return self.query_features

    # ── Stage 3a ─────────────────────────────────────────────────────────────

    def compute_sim_maps(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        '''Run SlidingWindowSimilarity for each WSI region.'''
        if self.wsi_features is None:
            raise RuntimeError('call build_wsi_features() first')
        if self.query_features is None:
            raise RuntimeError('call build_query_features() first')
        self.sim_maps = [
            SlidingWindowSimilarity(self.query_features, wf)
            for wf in self.wsi_features
        ]
        return self.sim_maps

    # ── Stage 3b ─────────────────────────────────────────────────────────────

    def _find_best_in_grid(self, use_overlap: bool) -> tuple[int, int, float, int]:
        '''Return (x @ level-n, y @ level-n, score, region_index) of the best window.'''
        ds = self.wsi_container.ds
        half = self.tile_size // 2   # overlap offset in level-n pixels
        best_score = -float('inf')
        best_x = best_y = 0
        best_region_idx = 0

        for ri, (region, (main_sim, overlap_sim)) in enumerate(
            zip(self.mask.tissue_regions, self.sim_maps)
        ):
            hm = overlap_sim if use_overlap else main_sim
            if hm.numel() == 0:
                continue
            hm_mean = hm.mean(dim=(-2, -1))          # [H_out, W_out]
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

    def find_best(self) -> SlideWinSimResult:
        '''Find the overall best match across all regions and both grids.'''
        if self.sim_maps is None:
            self.compute_sim_maps()

        ds = self.wsi_container.ds
        mx, my, m_score, m_ri = self._find_best_in_grid(use_overlap=False)
        ox, oy, o_score, o_ri = self._find_best_in_grid(use_overlap=True)

        from_overlap = o_score > m_score
        bx, by, bs, b_ri = (ox, oy, o_score, o_ri) if from_overlap else (mx, my, m_score, m_ri)

        self.result = SlideWinSimResult(
            best_x=bx,  best_y=by,  best_x0=int(bx * ds), best_y0=int(by * ds),
            best_score=bs, from_overlap=from_overlap, best_region_index=b_ri, ds=ds,
            main_x=mx,  main_y=my,  main_x0=int(mx * ds), main_y0=int(my * ds),
            main_score=m_score, main_region_index=m_ri,
            overlap_x=ox, overlap_y=oy, overlap_x0=int(ox * ds), overlap_y0=int(oy * ds),
            overlap_score=o_score, overlap_region_index=o_ri,
        )
        return self.result