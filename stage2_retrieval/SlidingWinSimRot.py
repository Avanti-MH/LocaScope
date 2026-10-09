"""Rotation-aware sliding-window retrieval, and the window kernel it runs on.

Tries all 4 cardinal rotations (0, 90, 180, 270 deg) of the query image and
ranks every window of every rotation, lattice and region together. The output
is a CandidateSet (StageInterface.py): candidates by score, in the frame of
the level's grids, which stage 3 reads its crop by.

`SlidingWindowSimilarity` (the per-window cosine kernel) lives here too, so the
window bench, the off-grid bench and this class import the kernel from one
place.

Cost:
    4x query patch extraction + encoding + sim-map computation.
    WSI features are encoded ONCE and shared across the 4 query orientations.

Usage:
    cfg = SLIDEWIN_RECIPES['gigapath']
    r  = SlidingWinSimRot(cfg, device).build(wsi, mask)
    cs = r.retrieve(shot_img, est_mpp_result)   # stage 1's output in
    cs.best, cs.origin_l0(cs.best)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image


import openslide                                                            # noqa: E402
from PatchingLib          import (QueryPatchContainer, PatchGrid, region_grids,  # noqa: E402
                                  FeaturesMap, WsiFeaturesMap)
from SafeSlide            import SafeSlide                                                # noqa: E402
from SlideReader          import SlideReader                                              # noqa: E402
from ConfigIdentity       import IdentifiedBuild, IdentifiedConfig, register              # noqa: E402
from TileEncoderFunc      import TileEncoder, TileEncoderConfig, encoder_config           # noqa: E402
from TissueMask   import TissueRegion, TissueMask            # noqa: E402
from stage2_retrieval.StageInterface import (Candidate, CandidateSet,  # noqa: E402
                                             ROTATIONS)

if TYPE_CHECKING:       # annotations only
    from stage1_estimation.StageInterface import EstMppResult


# ── the window kernel ─────────────────────────────────────────────────────────

def _sim_tensors_unfold(q_grid: torch.Tensor, wsi_grid: torch.Tensor) -> torch.Tensor:
    '''The original implementation, kept as the reference the fast one is
    measured against. Not called in production -- see `_sim_tensors` for why.'''
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
    Core unfold similarity: [R_q, C_q, D] x [R_w, C_w, D] -> [H_out, W_out, R_q, C_q].
    Returns empty tensor when wsi is smaller than query.

    out[h, w, r, c] = the cosine between query tile (r, c) and WSI tile
    (h + r, w + c). Which is a dot product over D, and D is contracted FIRST
    here -- that is the whole difference from `_sim_tensors_unfold`.

    That version writes `windows * q` before summing. `unfold` is a view and
    costs nothing, but the multiply materialises [D, H, W, R_q, C_q]: each WSI
    tile appears in up to R_q*C_q windows, and every copy still carries all D
    channels. On BRACS_1228 L0 region 0, 145x147 windows against a 4x5 query
    kernel, that is 2.62 GB for an output of 1.71 MB -- 1536x, and 7680x for
    the concatenated multi-slot descriptors bench_window_retrieval builds.

    Contracting D first gives every (WSI tile, query tile) dot product once,
    which is R_w*C_w*R_q*C_q numbers -- 1.79 MB for the same case. The windows
    are then pure indexing: no arithmetic, R_q*C_q slice copies.

    NOT bit-identical. einsum dispatches to a matmul, whose reduction order
    differs from an elementwise multiply-then-sum; fp32 puts the gap around
    1e-7. On CUDA it also depends on `torch.backends.cuda.matmul.allow_tf32`,
    which nothing in this project sets: TF32 keeps 10 mantissa bits, so with it
    enabled the gap is ~1e-3 instead.
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


def _grid_sims(q_grid: torch.Tensor, wsi_main: torch.Tensor,
               wsi_ov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''(main_sim, overlap_sim) of one query kernel against one region's two
    lattices, every grid already on the device the work runs on. What
    `SlidingWindowSimilarity` computes once its grids are built, and what the
    retriever computes from the grids it keeps per level.'''
    main_sim = _sim_tensors(q_grid, wsi_main)
    overlap_sim = (_sim_tensors(q_grid, wsi_ov) if wsi_ov.numel() > 0
                   else torch.empty(0))
    return main_sim, overlap_sim


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
      overlap_sim shape [H_out-1, W_out-1, R_q, C_q]  WSI overlap grid, origin (region.x + tile/2*ds, ...)
                        (empty tensor when WSI has no overlap patches)
      H_out = R_wsi - R_q + 1,  W_out = C_wsi - C_q + 1

    `device` moves the three grids before the similarity runs, so the einsum and
    the window slicing happen there. None leaves them where the FeaturesMap put
    them. Each grid is ONE gather on the features' own device
    (`FeaturesMap.main_feature_grid`), so features already on the GPU are
    arranged there and never cross to the host. The retriever does not come
    through here: it keeps each level's two WSI grids, arranged once
    (`SlidingWinSimRot.wsi_grids`), and calls `_grid_sims` per query.
    '''
    q_grid  = qFeatureMap.main_feature_grid()       # fixed: always main query kernel
    wsi_main = WsiFeatureMap.main_feature_grid()
    wsi_ov  = WsiFeatureMap.overlap_feature_grid()

    if device is not None:
        q_grid   = q_grid.to(device)
        wsi_main = wsi_main.to(device)
        wsi_ov   = wsi_ov.to(device)
    return _grid_sims(q_grid, wsi_main, wsi_ov)


# ── the window score ─────────────────────────────────────────────────────────
#
# A window is a query-sized block of reference tiles, and the kernel above gives
# the cosine of every query tile with the reference tile under it. The SCORE is
# how those `R_q x C_q` cosines become one number per window. It is a config
# field of the retriever (`score`), so two scores are two retrievers: two ids,
# two stage-2 entries.

#: Cosines of L2-normalised features live in [-1, 1] and log is undefined at or
#: below zero. Raising this floor pulls geomean towards the arithmetic mean,
#: because it stops the worst tile from dominating -- it is a parameter, not a
#: guard. Same value as bench_offgrid_score, so the benches agree.
GEOMEAN_FLOOR: float = 1e-3


def _score_mean(per_tile: torch.Tensor) -> torch.Tensor:
    """Arithmetic mean over the query's tiles: strong tiles carry a window even
    when the rest disagree -- an OR. Reduced in the maps' own dtype, as the
    retriever always has, so this score changes nothing that was cached."""
    return per_tile.mean(dim=(-2, -1))


def _score_geomean(per_tile: torch.Tensor) -> torch.Tensor:
    """`exp(mean(log(x)))`, each cosine floored at `GEOMEAN_FLOOR`: one tile
    near zero drags the window down however good the others are -- an AND.
    "Use AND instead of OR" and "multiply instead of add" are the same change;
    a linear combiner cannot be an AND."""
    floored: torch.Tensor = per_tile.float().clamp_min(GEOMEAN_FLOOR)
    return torch.exp(torch.log(floored).mean(dim=(-2, -1)))


#: Named window scores: `[..., R_q, C_q]` per-tile cosines -> `[...]`, one per
#: window. A new score is one entry here and nothing else.
WINDOW_SCORES: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'mean': _score_mean,
    'geomean': _score_geomean,
}


def window_score(per_tile: torch.Tensor, kind: str = 'mean') -> torch.Tensor:
    """The score `kind` of every window of `per_tile` (`[..., R_q, C_q]`)."""
    try:
        fn: Callable[[torch.Tensor], torch.Tensor] = WINDOW_SCORES[kind]
    except KeyError:
        raise ValueError(f'no window score {kind!r}; the scores are '
                         f'{", ".join(WINDOW_SCORES)}') from None
    return fn(per_tile)


# ── the retriever ─────────────────────────────────────────────────────────────

@register('sliding_win_sim_rot')
@dataclass(frozen=True)
class SlidingWinSimRotConfig(IdentifiedConfig):
    '''What the retriever is: the encoder it scores with (a TileEncoderConfig,
    precision and batch size included), the tile it cuts, whether it also
    slides the half-tile offset lattice, how many candidates it returns, and
    how far apart two of them at one rotation must be (`min_sep_tiles`, see
    the attribute in `__init__`), and the window `score` (`WINDOW_SCORES`).

    Every field is identity. `k` and `min_sep_tiles` change no score, but they
    change which candidates the CandidateSet holds, and that set is this
    stage's output. The encoder's own NOT_IDENTITY (batch_size) still applies
    inside it.'''
    encoder:       TileEncoderConfig
    tile_size:     int   = 256
    overlap:       bool  = True
    k:             int   = 20
    min_sep_tiles: float = 1.0
    score:         str   = 'mean'

    BASELINE = {'tile_size': 256, 'overlap': True, 'k': 20, 'min_sep_tiles': 1.0,
                'score': 'mean'}

    def __post_init__(self) -> None:
        if self.score not in WINDOW_SCORES:
            raise ValueError(f'no window score {self.score!r}; the scores are '
                             f'{", ".join(WINDOW_SCORES)}')


#: Named retrievers, every field written out (test_config_identity's recipe
#: lint); `--stage2 slidewin:<name>`. The encoder is the registry's own config
#: at fp16 -- one encoder per entry, imported when this module is.
SLIDEWIN_RECIPES: Dict[str, SlidingWinSimRotConfig] = {
    'gigapath': SlidingWinSimRotConfig(
        encoder=encoder_config('gigapath').with_model(dtype='fp16'),
        tile_size=256, overlap=True, k=100, min_sep_tiles=1.0, score='mean'),
}


class SlidingWinSimRot(IdentifiedBuild):
    """Rotation-aware sliding-window retrieval: stage 2, first phase.

        r = SlidingWinSimRot(
            SLIDEWIN_RECIPES['gigapath'], device).build(wsi, mask)
        cs = r.retrieve(query, est_mpp_result)          # CandidateSet

    `build` binds a slide; `retrieve` takes stage 1's output, builds the
    features of the level it routed to on first use (cached per level, and
    through `feature_store` when one is given), scores every window of the
    query at 4 rotations, and returns the best `k` as a CandidateSet.

    The steps stay public for a caller that drives a scale itself (the
    benches): build_wsi_features, build_query_features, compute_sim_maps,
    candidate_set.
    """

    ROTATIONS = ROTATIONS

    def __init__(
        self,
        cfg:          SlidingWinSimRotConfig,
        device:       Union[str, torch.device, None] = None,
        multi_gpu:    bool = False,
        read_workers: int  = 0,
    ) -> None:
        '''Builds its own encoder from `cfg.encoder`, as every stage-1
        estimator does: what the stage computes with is part of the stage.
        `device` and `multi_gpu` are how it is built, not what it is, so they
        are arguments here rather than config fields -- the same split as
        `KnnEstMpp(cfg, device)`.

        The encoder is built on first use (`encoder`, `model`), not here: a
        stage object can be made, and named by its config, without loading a
        model, and a run whose every stage-2 entry hits never loads one.'''
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg: SlidingWinSimRotConfig = cfg
        self.device: torch.device = torch.device(device)
        self._multi_gpu: bool = multi_gpu
        self._encoder: Optional[TileEncoder] = None
        self.tile_size: int = cfg.tile_size
        self.overlap: bool = cfg.overlap
        #: How many candidates `retrieve` returns.
        self.k: int = cfg.k
        #: A candidate whose window origin is within this many tiles (level-0
        #: distance) of one already kept AT THE SAME ROTATION is dropped: a
        #: strong peak otherwise fills the list with itself, seen from the main
        #: lattice and from the offset one, and a nominal k of 5 is two real
        #: places. Per rotation on purpose: the same spot at two orientations
        #: is two hypotheses. 0 keeps the raw ranking.
        self.min_sep_tiles: float = cfg.min_sep_tiles
        #: How a window's per-tile cosines become its score (`WINDOW_SCORES`).
        self.score: str = cfg.score
        #: `read_workers` processes read grid blocks ahead of the encoder; the
        #: CPU budget is the entry point's to decide.
        self.read_workers: int = read_workers

        self.wsi: Optional[SafeSlide] = None
        self.mask: Optional[TissueMask] = None
        self.reader: Optional[SlideReader] = None
        self.feature_store: Optional[Any] = None
        # The scale of the CURRENT build -- what everything downstream reads.
        self.level:   Optional[int]                = None
        self.ds:      Optional[float]              = None
        self.regions: Optional[list[TissueRegion]] = None
        #: One PatchGrid per region of `regions`, at the current scale -- the
        #: frame a CandidateSet is expressed in.
        self.grids:   Optional[list[PatchGrid]]    = None
        self.wsi_features: Optional[WsiFeaturesMap] = None
        #: Per region of `regions`, its (main, offset) feature grids
        #: [rows, cols, D] on the device, arranged ONCE per level -- every
        #: query and every rotation slides over these.
        self.wsi_grids: Optional[list] = None
        #: {ds: (regions, grids, features, wsi_grids)} for every scale built so far.
        self._by_ds: Dict[float, tuple] = {}
        #: {level: (regions, grids)} for every level looked at so far -- geometry
        #: alone, no pixels: what the fit check reads before a level is encoded.
        self._geometry_by_level: Dict[int, Tuple[List[TissueRegion], List[PatchGrid]]] = {}
        #: {level: why} for a level whose build raised; remembered so a level
        #: that failed once is not read again for every query. Failures that
        #: depend on the query (it does not fit) are not kept here.
        self._unusable: Dict[int, str] = {}
        #: Seconds the last `retrieve` spent building levels (0 when cached).
        self.last_level_seconds: float = 0.0
        #: `{level: (seconds, tiles)}` of every level encoded since `build`: the
        #: wall-clock seconds the whole level took and how many tiles it held.
        #: A level that was already built (or read from the feature store) is
        #: not here again.
        self.level_encodes: Dict[int, Tuple[float, int]] = {}

        # Per-rotation state (dict keyed by rotation degree)
        self.qc_by_rot:             Dict[int, QueryPatchContainer] = {}
        self.query_features_by_rot: Dict[int, FeaturesMap]         = {}
        self.sim_maps_by_rot: Dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    @property
    def encoder(self) -> TileEncoder:
        '''The tile encoder, built on first use from `cfg.encoder`.'''
        if self._encoder is None:
            self._encoder = self.cfg.encoder.build(self.device,
                                                   multi_gpu=self._multi_gpu)
        return self._encoder

    @property
    def model(self) -> torch.nn.Module:
        '''What IdentifiedBuild.weights_id hashes: the encoder is this stage's
        only trained part.'''
        return self.encoder.model

    # ── one slide ────────────────────────────────────────────────────────────
    def build(self, wsi: Union[openslide.OpenSlide, str], mask: Optional[TissueMask],
              feature_store: Optional[Any] = None) -> 'SlidingWinSimRot':
        """Bind a slide and its mask. Cheap: no level is built until one is
        asked for. `feature_store` is anything with load(regions, ds=, level=,
        tile_size=, overlap=) -> WsiFeaturesMap | None and save(WsiFeaturesMap);
        this class asks, and rebuilds when the answer is None."""
        # SafeSlide so a hole in a MIRAX cannot kill the handle mid-run.
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        self.wsi, self.mask, self.feature_store = wsi, mask, feature_store
        self.reader = SlideReader(wsi, workers=self.read_workers)
        self.level = self.ds = None
        self.regions = self.grids = self.wsi_features = self.wsi_grids = None
        self._by_ds = {}
        self._geometry_by_level = {}
        self._unusable = {}
        self.level_encodes = {}
        self.qc_by_rot, self.query_features_by_rot, self.sim_maps_by_rot = {}, {}, {}
        return self

    def retrieve(self, query: np.ndarray, estimate: EstMppResult) -> CandidateSet:
        """Stage 2 on stage 1's output: the best `k` windows of `query`, at the
        level `estimate.chosen_level` routed it to when that level can hold the
        query, else at the nearest finer level that can.

        A level cannot hold the query when no tissue region has a tile grid as
        large as the query's, at any of the 4 rotations, on either lattice
        (`_fitting_rotations`). That is read off the geometry before anything is
        encoded. It only gets harder as ds grows -- a region's tile grid shrinks
        and the query's does not -- so a level that does not fit rules out every
        coarser one, which is recorded without being tried. The levels found
        unsearchable are `irretrievable_lvl`; the other levels tried are
        `alter_lvl`, and the last of them is the one searched (`level`). A level
        whose build raises is recorded the same way and remembered. When no
        level can be searched the set has no candidates and `level` is the
        routed one."""
        if self.wsi is None:
            raise RuntimeError('call build(wsi, mask) first')
        routed: int = int(estimate.chosen_level)
        n_levels: int = len(self.wsi.level_downsamples)
        dims: Dict[int, Tuple[int, int]] = self._query_dims(query)
        irretrievable: List[int] = []
        alter: List[int] = []
        found: Optional[CandidateSet] = None
        query_ready: bool = False
        seconds: float = 0.0
        level: int
        for level in range(routed, -1, -1):
            if level != routed:
                alter.append(level)
            why: Optional[str] = self._unusable.get(level)
            if why is None and not self._fitting_rotations(level, dims):
                why = 'no region holds the query'
                coarser: int
                for coarser in range(level + 1, n_levels):
                    if coarser not in irretrievable:
                        irretrievable.append(coarser)
            if why is None:
                fresh: bool = (float(self.wsi.level_downsamples[level])
                               not in self._by_ds)
                started: float = time.perf_counter()
                try:
                    self.build_wsi_features(level=level)
                except Exception as e:                              # noqa: BLE001
                    why = f'{type(e).__name__}: {e}'
                    self._unusable[level] = why
                took: float = time.perf_counter() - started
                seconds += took
                if why is None and fresh:
                    self.level_encodes[level] = (
                        took, sum(len(g) for g in self.grids))
            if why is None:
                if not query_ready:
                    self.build_query_features(query)
                    query_ready = True
                self.compute_sim_maps()
                searched: CandidateSet = self.candidate_set()
                if len(searched):
                    found = searched
                    break
            irretrievable.append(level)
        self.last_level_seconds = seconds
        lost: Tuple[int, ...] = tuple(sorted(set(irretrievable)))
        if found is not None:
            return replace(found, irretrievable_lvl=lost, alter_lvl=tuple(alter))
        self.sim_maps_by_rot = {}
        return CandidateSet(candidates=(), level=routed,
                            ds=float(self.wsi.level_downsamples[routed]), grids=(),
                            irretrievable_lvl=lost, alter_lvl=tuple(alter))

    # ── which levels can hold the query ──────────────────────────────────────
    def _geometry(self, level: int, ds: float
                  ) -> Tuple[List[TissueRegion], List[PatchGrid]]:
        """The regions at `level` that can host a tile, and their grids: geometry
        alone, no pixel is read. Cached per level, shared by the fit check and
        `build_wsi_features`."""
        cached: Optional[Tuple[List[TissueRegion], List[PatchGrid]]] = (
            self._geometry_by_level.get(level))
        if cached is not None:
            return cached
        regions: List[TissueRegion]
        if self.mask is not None:
            regions = self.mask.patchable(self.tile_size * ds).tissue_regions
        else:
            w0: int
            h0: int
            w0, h0 = self.wsi.level_dimensions[0]
            regions = [TissueRegion(x=0, y=0, w=w0, h=h0, index=0)]
        grids: List[PatchGrid] = region_grids(regions, ds=ds, level=level,
                                              tile_size=self.tile_size,
                                              overlap=self.overlap)
        self._geometry_by_level[level] = (regions, grids)
        return regions, grids

    def _query_dims(self, query: Union[QueryPatchContainer, Image.Image, np.ndarray]
                    ) -> Dict[int, Tuple[int, int]]:
        """(rows, cols) of the query's tile grid at each rotation -- from its
        size alone, nothing is cut or encoded."""
        img: np.ndarray = self._as_array(query)
        height: int = int(img.shape[0])
        width: int = int(img.shape[1])
        out: Dict[int, Tuple[int, int]] = {}
        rot: int
        for rot in self.ROTATIONS:
            w: int
            h: int
            w, h = (width, height) if rot in (0, 180) else (height, width)
            grid: PatchGrid = PatchGrid.from_size(w, h, self.tile_size,
                                                  overlap=self.overlap)
            out[rot] = (grid.grid_rows, grid.grid_cols)
        return out

    def _fitting_rotations(self, level: int, dims: Dict[int, Tuple[int, int]]
                           ) -> Tuple[int, ...]:
        """The rotations at which some region of `level` holds at least one
        window: its main tile grid, or the offset lattice's, is as large as the
        query's (rows, cols) there -- the condition `_sim_tensors` slides
        under. Empty when the level cannot be searched for this query."""
        grids: List[PatchGrid] = self._geometry(
            level, float(self.wsi.level_downsamples[level]))[1]
        fit: List[int] = []
        rot: int
        for rot in self.ROTATIONS:
            rows: int
            cols: int
            rows, cols = dims[rot]
            if rows < 1 or cols < 1:
                continue
            if any((g.grid_rows >= rows and g.grid_cols >= cols)
                   or (g.overlap_rows >= rows and g.overlap_cols >= cols)
                   for g in grids):
                fit.append(rot)
        return tuple(fit)

    def frame(self, level: int) -> CandidateSet:
        """An empty CandidateSet in `level`'s frame -- its level, ds and region
        grids -- to put stored candidates in (stage 3 run on candidates read
        back from a cache). Builds the level if it is not."""
        self.build_wsi_features(level=int(level))
        return CandidateSet(candidates=(), level=int(self.level), ds=float(self.ds),
                            grids=tuple(self.grids), irretrievable_lvl=(),
                            alter_lvl=())

    @staticmethod
    def _as_array(query: Union[QueryPatchContainer, Image.Image, np.ndarray]
                  ) -> np.ndarray:
        """The query as an RGB array, whatever it came as."""
        if isinstance(query, QueryPatchContainer):
            return np.array(query.img)
        if isinstance(query, Image.Image):
            return np.array(query.convert('RGB'))
        return np.asarray(query)

    # ── the level's features ─────────────────────────────────────────────────
    def build_wsi_features(self, mpp: Optional[float] = None,
                           ds: Optional[float] = None,
                           level: Optional[int] = None) -> WsiFeaturesMap:
        """Encode every usable tissue region at one scale, given as exactly
        one of `level` (taken as is), `ds` or `mpp` (snapped to the nearest
        level, SlideReader.native_scale). The same scale a second time
        reuses its features.

        `self.mask` is never replaced. Each build takes a `patchable` view of
        it, so every build starts from the whole segmentation -- filtering is
        monotone in ds, and a build that narrowed the mask in place would
        leave a later, finer build unable to see what the coarse pass dropped.
        """
        if self.wsi is None:
            raise RuntimeError('call build(wsi, mask) first')
        if (mpp is None) + (ds is None) + (level is None) != 2:
            raise ValueError('give exactly one of mpp / ds / level')
        if level is not None:
            self.level, self.ds = int(level), float(self.wsi.level_downsamples[level])
        else:
            self.level, self.ds = self.reader.native_scale(mpp=mpp, ds=ds)

        if self.ds in self._by_ds:
            (self.regions, self.grids, self.wsi_features,
             self.wsi_grids) = self._by_ds[self.ds]
        else:
            self.regions, self.grids = self._geometry(self.level, self.ds)
            geo = dict(ds=self.ds, level=self.level, tile_size=self.tile_size,
                       overlap=self.overlap)

            # A cache hit reads nothing off the slide: stage 3 reads its own
            # window, so no pixels have to be resident.
            self.wsi_features = None
            if self.feature_store is not None:
                self.wsi_features = self.feature_store.load(self.regions, **geo)
                if self.wsi_features is not None:
                    # read onto the host; moved to the device once, here
                    self.wsi_features = self.wsi_features.to(self.device)
            if self.wsi_features is None:
                blocks = self.reader.read_grid(
                    self.regions, self.grids, self.ds, tile=self.tile_size,
                    offset=self.overlap, level=self.level)
                self.wsi_features = WsiFeaturesMap.from_grid_read(
                    blocks, self.regions, self.grids, self.encoder, **geo)
                if self.feature_store is not None:
                    self.feature_store.save(self.wsi_features)
            self.wsi_grids = [(fm.main_feature_grid(), fm.overlap_feature_grid())
                              for fm in self.wsi_features]
            self._by_ds[self.ds] = (self.regions, self.grids, self.wsi_features,
                                    self.wsi_grids)

        # Similarity maps belong to the scale that produced them.
        self.sim_maps_by_rot = {}
        return self.wsi_features

    # ── the query at 4 rotations ─────────────────────────────────────────────
    @staticmethod
    def _rotate_np(img: np.ndarray, rot_deg: int) -> np.ndarray:
        """Lossless 90-deg-step rotation of an RGB image (H, W, C)."""
        if rot_deg not in ROTATIONS:
            raise ValueError(f'unsupported rotation {rot_deg}; must be 0/90/180/270')
        return img if rot_deg == 0 else np.rot90(img, k=rot_deg // 90)

    def build_query_features(
        self,
        query: Union[QueryPatchContainer, Image.Image, np.ndarray],
    ) -> Dict[int, FeaturesMap]:
        """Extract + encode query patches at each of 4 cardinal rotations."""
        img_np: np.ndarray = self._as_array(query)

        self.qc_by_rot            = {}
        self.query_features_by_rot = {}
        for rot in self.ROTATIONS:
            qc = QueryPatchContainer(self._rotate_np(img_np, rot))
            qc.extract_all(tile_size=self.tile_size, overlap=self.overlap)
            self.qc_by_rot[rot]             = qc
            self.query_features_by_rot[rot] = qc.to_features(self.encoder)
        return self.query_features_by_rot

    # ── scores ───────────────────────────────────────────────────────────────
    def compute_sim_maps(self) -> Dict[int, list[tuple[torch.Tensor, torch.Tensor]]]:
        if self.wsi_features is None:
            raise RuntimeError('call build_wsi_features() first')
        if not self.query_features_by_rot:
            raise RuntimeError('call build_query_features() first')
        # Everything is on the encoder's device already: the WSI grids were
        # arranged there once per level, the query's features came back there.
        self.sim_maps_by_rot = {}
        for rot, qfm in self.query_features_by_rot.items():
            q_grid = qfm.main_feature_grid()
            self.sim_maps_by_rot[rot] = [_grid_sims(q_grid, main, offset)
                                         for main, offset in self.wsi_grids]
        return self.sim_maps_by_rot

    def _window_scores(self, sim_maps):
        """Yield (region_index, lattice, [rows, cols] score per window).

        A window's score is `self.score` of its per-tile cosines, so the two
        trailing dims collapse. A region smaller than the query has none. The
        maps were computed over the FILTERED regions, which is what
        wsi_features carries, so region_index indexes `self.regions`."""
        for ri, (main_sim, offset_sim) in enumerate(sim_maps):
            for lattice, hm in (('main', main_sim), ('offset', offset_sim)):
                if hm.numel():
                    yield ri, lattice, window_score(hm, self.score)

    def candidate_set(self, k: Optional[int] = None) -> CandidateSet:
        """The `k` best windows of the current maps, across every rotation,
        lattice and region, as a CandidateSet. topk per grid rather than
        over the union: only k can survive from any one grid anyway."""
        if not self.sim_maps_by_rot:
            self.compute_sim_maps()
        k = self.k if k is None else k

        raw = []
        for rot, sim_maps in self.sim_maps_by_rot.items():
            for ri, lattice, scores in self._window_scores(sim_maps):
                flat = scores.reshape(-1)
                vals, idxs = torch.topk(flat, min(max(k, 1), flat.numel()))
                n_cols = scores.shape[1]
                for v, i in zip(vals.tolist(), idxs.tolist()):
                    r, c = divmod(int(i), n_cols)
                    raw.append(Candidate(ri, lattice, r, c, rot, float(v)))

        frame = dict(level=self.level, ds=self.ds, grids=tuple(self.grids),
                     irretrievable_lvl=(), alter_lvl=())
        if not raw:
            # No region holds a single window: the sliding kernel needs a
            # region grid at least as large as the query's. Nothing to rank;
            # `retrieve` checks the fit first and moves to a finer level.
            return CandidateSet(candidates=(), **frame)
        raw.sort(key=lambda c: -c.score)

        probe = CandidateSet(candidates=(), **frame)
        min_sep = self.min_sep_tiles * self.tile_size * self.ds
        kept, origins = [], []
        for c in raw:
            o = probe.origin_l0(c)
            if min_sep > 0 and any(
                    kc.rotation == c.rotation
                    and math.hypot(o[0] - ko[0], o[1] - ko[1]) < min_sep
                    for kc, ko in zip(kept, origins)):
                continue
            kept.append(c)
            origins.append(o)
            if len(kept) == k:
                break
        return CandidateSet(candidates=tuple(kept), **frame)

    # ── what a bench reads off the maps of the current query ─────────────────
    #
    # Diagnostics, optional for a retriever (Retriever's docstring): valid from
    # a `retrieve` until the next one, which replaces the maps they read.
    def window_tile_sims(self, c: Candidate) -> torch.Tensor:
        """[rows, cols] cosine of every query tile against the reference tile
        under it in window `c` -- the numbers `c.score` is the retriever's
        `score` of."""
        main, offset = self.sim_maps_by_rot[c.rotation][c.region_index]
        return (main if c.lattice == 'main' else offset)[c.row, c.col]

    def tile_sim_rows(self, candidates: Sequence[Candidate]
                      ) -> List[Dict[str, Union[int, float]]]:
        """The per-tile cosines of `candidates` as table rows: for each, in
        order (`rank` is its place in `candidates`, 1 the first), every query
        tile (`q_row`, `q_col`, on the query turned to the candidate's
        rotation), the reference tile under it on the window's lattice
        (`ref_row`, `ref_col`) and their `cosine`."""
        out: List[Dict[str, Union[int, float]]] = []
        rank: int
        c: Candidate
        for rank, c in enumerate(candidates, 1):
            grid: np.ndarray = self.window_tile_sims(c).float().cpu().numpy()
            q_row: int
            q_col: int
            for q_row in range(grid.shape[0]):
                for q_col in range(grid.shape[1]):
                    out.append(dict(rank=rank, q_row=q_row, q_col=q_col,
                                    ref_row=c.row + q_row, ref_col=c.col + q_col,
                                    cosine=float(grid[q_row, q_col])))
        return out

    def rotations_searched(self) -> Tuple[int, ...]:
        """The rotations that have at least one window at the level the last
        `retrieve` searched: a region too small for the query at one rotation
        is not too small at the transposed one."""
        out: List[int] = []
        rot: int
        maps: List[Tuple[torch.Tensor, torch.Tensor]]
        for rot, maps in self.sim_maps_by_rot.items():
            if any(hm.numel() for pair in maps for hm in pair):
                out.append(rot)
        return tuple(out)

    def nearest_window(self, x_l0: float, y_l0: float, rotation: int,
                       lattices=('main', 'offset')) -> tuple[Candidate, int, float]:
        """`(window, rank, distance)`: the window at `rotation` whose centre is
        nearest the level-0 point, scored; its rank among every window of the
        current maps, every rotation, lattice and region (1 is the best, ties
        go to it); and the centre's distance in level-0 px. With the shot's
        true centre and rotation, that window is the truth stage 2 should find.
        `lattices` limits the search to those lattices -- `('main',)` is the
        nearest window of the main grid alone."""
        q = self.qc_by_rot[rotation].grid
        best = None
        for ri, (main, offset) in enumerate(self.sim_maps_by_rot[rotation]):
            grid = self.grids[ri]
            for lattice, hm in (('main', main), ('offset', offset)):
                if not hm.numel() or lattice not in lattices:
                    continue
                n_r, n_c = hm.shape[:2]
                xs = np.array([grid.tile_origin_l0(lattice, 0, j)[0] for j in range(n_c)],
                              dtype=np.float64) + q.grid_cols * self.tile_size * self.ds / 2.0
                ys = np.array([grid.tile_origin_l0(lattice, i, 0)[1] for i in range(n_r)],
                              dtype=np.float64) + q.grid_rows * self.tile_size * self.ds / 2.0
                j = int(np.abs(xs - x_l0).argmin())
                i = int(np.abs(ys - y_l0).argmin())
                d = math.hypot(xs[j] - x_l0, ys[i] - y_l0)
                if best is None or d < best[0]:
                    best = (d, ri, lattice, i, j, hm)
        if best is None:
            raise ValueError('no window at this level holds the query')
        d, ri, lattice, i, j, hm = best
        score = float(window_score(hm[i, j], self.score))
        higher = sum(int((s > score).sum())
                     for maps in self.sim_maps_by_rot.values()
                     for _, _, s in self._window_scores(maps))
        return Candidate(ri, lattice, i, j, int(rotation), score), higher + 1, d
