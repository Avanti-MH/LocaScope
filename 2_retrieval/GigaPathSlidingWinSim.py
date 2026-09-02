
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
from PatchingLib import (QueryPatchContainer, WsiTissuesContainer,
                         FeaturesMap, WsiFeaturesMap)
from SafeSlide import SafeSlide
from TissuesRegionsMask import TissuesRegionsMask
from GigaPathFunc import GigaPathEncoderConfig

def _sim_tensors_unfold(q_grid: torch.Tensor, wsi_grid: torch.Tensor) -> torch.Tensor:
    '''The original implementation, kept as the reference the fast one is
    measured against (test_gigapath_slide_win_sim step 0). Not called in
    production -- see `_sim_tensors` for why.'''
    R_q, C_q, _ = q_grid.shape
    R_w, C_w, _ = wsi_grid.shape
    if R_w < R_q or C_w < C_q:
        return torch.empty(0)
    wsi     = wsi_grid.permute(2, 0, 1)
    windows = wsi.unfold(1, R_q, 1).unfold(2, C_q, 1)        # [D, H_out, W_out, R_q, C_q]
    q       = q_grid.permute(2, 0, 1)
    return (windows * q[:, None, None, :, :]).sum(dim=0)      # [H_out, W_out, R_q, C_q]


def _sim_tensors(q_grid: torch.Tensor, wsi_grid: torch.Tensor) -> torch.Tensor:
    '''
    Core unfold similarity: [R_q, C_q, D] × [R_w, C_w, D] → [H_out, W_out, R_q, C_q].
    Returns empty tensor when wsi is smaller than query.

    out[h, w, r, c] = the cosine between query tile (r, c) and WSI tile
    (h + r, w + c). Which is a dot product over D, and D is contracted FIRST
    here -- that is the whole difference from `_sim_tensors_unfold`.

    That version writes `windows * q` before summing. `unfold` is a view and
    costs nothing, but the multiply materialises [D, H, W, R_q, C_q]: each WSI
    tile appears in up to R_q*C_q windows, and every copy still carries all D
    channels. On BRACS_1228 L0 region 0, 145x147 windows against a 4x5 query
    kernel, that is 2.62 GB for an output of 1.71 MB -- 1536x, and 7680x for
    the concatenated multi-slot descriptors bench_slidewin_pooling builds.

    Contracting D first gives every (WSI tile, query tile) dot product once,
    which is R_w*C_w*R_q*C_q numbers -- 1.79 MB for the same case. The windows
    are then pure indexing: no arithmetic, R_q*C_q slice copies.

    NOT bit-identical. einsum dispatches to a matmul, whose reduction order
    differs from an elementwise multiply-then-sum; fp32 puts the gap around
    1e-7. On CUDA it also depends on `torch.backends.cuda.matmul.allow_tf32`,
    which nothing in this project sets: TF32 keeps 10 mantissa bits, so with it
    enabled the gap is ~1e-3 instead. Step 0 of
    test_gigapath_slide_win_sim.py measures both against a decoy rather than a
    tolerance, and prints the flag.
    '''
    R_q, C_q, _ = q_grid.shape
    R_w, C_w, _ = wsi_grid.shape
    if R_w < R_q or C_w < C_q:
        return torch.empty(0)
    H_out, W_out = R_w - R_q + 1, C_w - C_q + 1

    # [R_w, C_w, R_q, C_q]: every WSI tile against every query tile, once.
    sims = torch.einsum('rcd,ijd->rcij', wsi_grid, q_grid)
    out = torch.empty(H_out, W_out, R_q, C_q,
                      dtype=sims.dtype, device=sims.device)
    for r in range(R_q):
        for c in range(C_q):
            # Window (h, w) puts query tile (r, c) over WSI tile (h+r, w+c),
            # so one query tile's whole heat map is a shifted view of `sims`.
            out[:, :, r, c] = sims[r:r + H_out, c:c + W_out, r, c]
    return out


def SlidingWindowSimilarity(
    qFeatureMap: FeaturesMap,
    WsiFeatureMap: FeaturesMap,
    device=None,
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

    `device` moves the three grids before the similarity runs, so the einsum and
    the window slicing happen there. None leaves them where the FeaturesMap put
    them, which is the host -- TileEncoderFunc.features() ends its reduction
    with .cpu() (TileEncoderFunc.py:898) and nothing in this path moves them
    back. That default exists so the four callers which do not pass a device
    keep the behaviour they were measured with; the retrieval path
    (GigaPathSlidingWinSimRot.compute_sim_maps) passes one unconditionally.

    ORDER MATTERS, and not for style. The three grid builders below are Python
    double loops -- one `out[r, c] = self[idx]` per cell, PatchingLib.py:608 and
    :615 -- so they are built on whatever device the features already live on
    and moved AFTERWARDS, in one transfer each. Building them on a GPU instead
    would turn a few thousand memory copies into a few thousand kernel launches,
    per call, 172 calls per photo.

    What this costs, measured against log/BenchMarkV2 (S1104233, uni2): the
    einsum was 4.2 s over the whole 71-photo run and the grids total about
    160 MB per rotation, so moving them per call uploads roughly 45 GB across
    the run. The transfer is expected to exceed what the einsum saves. That is
    the reason the timers below exist rather than an argument against the move.
    '''
    q_grid  = qFeatureMap.main_feature_grid()       # fixed: always main query kernel
    wsi_main = WsiFeatureMap.main_feature_grid()
    wsi_ov  = WsiFeatureMap.overlap_feature_grid()

    if device is not None:
        q_grid   = q_grid.to(device)
        wsi_main = wsi_main.to(device)
        wsi_ov   = wsi_ov.to(device)

    main_sim = _sim_tensors(q_grid, wsi_main)
    overlap_sim = _sim_tensors(q_grid, wsi_ov) if wsi_ov.numel() > 0 \
                  else torch.empty(0)

    return main_sim, overlap_sim


def compute_gigapath_sliding_win_similarity(
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
        # dtype='fp32' and not the GigaPathEncoderConfig default of fp16: the old
        # the free function this replaced defaulted to fp32 (see
        # GigaPathFunc_old) and this caller never overrode it,
        # so fp16 here would be a precision change smuggled in by a refactor.
        encoder = GigaPathEncoderConfig(batch_size=batch_size).with_model(dtype='fp32').build(device)

    wsi_features = WsiTissuesPathes.to_features(encoder)
    QueryFeatures = query.to_features(encoder)

    return [SlidingWindowSimilarity(QueryFeatures, wf) for wf in wsi_features]


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
        1. build_wsi_features(mpp)  — tile WSI at mpp → encode → WsiFeaturesMap
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
        # SafeSlide so a hole in a MIRAX cannot kill the handle mid-run; see
        # utilities/SafeSlide.py. Only reached when constructed from a path --
        # LocaScopePipeline passes an already-open slide.
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi = wsi
        self.encoder = encoder
        self.mask = mask
        self.mpp = mpp
        self.tile_size = tile_size
        self.overlap = overlap

        # Intermediate state — inspect at any stage for debugging / visualization
        self.wsi_container: Optional[WsiTissuesContainer] = None
        self.wsi_features: Optional[WsiFeaturesMap] = None
        self.qc: Optional[QueryPatchContainer] = None
        self.query_features: Optional[FeaturesMap] = None
        self.sim_maps: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
        self.result: Optional[SlideWinSimResult] = None

    # ── Stage 1 ──────────────────────────────────────────────────────────────

    def build_wsi_features(self, mpp: Optional[float] = None) -> WsiFeaturesMap:
        '''Tile WSI at the given mpp (falls back to self.mpp), encode each region.'''
        mpp = mpp or self.mpp
        if mpp is None:
            raise ValueError('mpp must be provided in __init__ or build_wsi_features()')
        if self.mask is None:
            # '' and not a threshold: the retriever scores a window by the
            # mean cosine over the query's tiles, so a window on blank glass
            # loses on its own merits and the mask here is an optimisation, not
            # a correctness requirement. '' skips the read AND the array -- see
            # TissueSegFunc on why that is not the same as a method returning
            # ones.
            from TissueSegFunc import TissueSegConfig
            self.mask = TissuesRegionsMask.from_wsi(
                self.wsi, method=TissueSegConfig('').build())
        self.wsi_container = WsiTissuesContainer.from_mpp(
            self.wsi, mpp, tile_size=self.tile_size, overlap=self.overlap, mask=self.mask
        )
        self.wsi_features = self.wsi_container.to_features(self.encoder)
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

        # The container's regions, not self.mask's. from_mpp filters the mask
        # down to what can host a tile at this ds, so the sim_maps are per
        # SURVIVING region while self.mask still lists them all. Zipping the
        # wrong one lands every window on a neighbouring region's coordinates
        # and raises nothing.
        # .items() and not zip(...): the pairing is the WsiFeaturesMap's, so
        # there is no second list left to zip the wrong one of.
        for ri, ((region, _), (main_sim, overlap_sim)) in enumerate(
            zip(self.wsi_features.items(), self.sim_maps)
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