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
    cfg = SlidingWinSimRotConfig(encoder_config('gigapath'), k=20)
    r  = SlidingWinSimRot(cfg, device).build(wsi, mask)
    cs = r.retrieve(shot_img, est_mpp_result)   # stage 1's output in
    cs.best, cs.origin_l0(cs.best)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image

# utilities/ by hand, then _paths for the rest -- the idiom stage1_estimation
# uses. The project root it adds is what resolves `stage2_retrieval.X`.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'utilities'))
from _paths import setup_import_paths                                   # noqa: E402
setup_import_paths()

import openslide                                                            # noqa: E402
from PatchingLib          import (QueryPatchContainer, PatchGrid, region_grids,  # noqa: E402
                                  FeaturesMap, WsiFeaturesMap)
from SafeSlide            import SafeSlide                                                # noqa: E402
from SlideReader          import SlideReader                                              # noqa: E402
from ConfigIdentity       import IdentifiedBuild, IdentifiedConfig, register              # noqa: E402
from TileEncoderFunc      import TileEncoderConfig                                        # noqa: E402
from TissueMask   import TissueRegion, TissueMask            # noqa: E402
from stage2_retrieval.StageInterface import (Candidate, CandidateSet,  # noqa: E402
                                             ROTATIONS)


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


# ── the retriever ─────────────────────────────────────────────────────────────

@register('sliding_win_sim_rot')
@dataclass(frozen=True)
class SlidingWinSimRotConfig(IdentifiedConfig):
    '''What the retriever is: the encoder it scores with (a TileEncoderConfig,
    precision and batch size included), the tile it cuts, whether it also
    slides the half-tile offset lattice, how many candidates it returns, and
    how far apart two of them at one rotation must be (`min_sep_tiles`, see
    the attribute in `__init__`).

    Every field is identity. `k` and `min_sep_tiles` change no score, but they
    change which candidates the CandidateSet holds, and that set is this
    stage's output. The encoder's own NOT_IDENTITY (batch_size) still applies
    inside it.'''
    encoder:       TileEncoderConfig
    tile_size:     int   = 256
    overlap:       bool  = True
    k:             int   = 20
    min_sep_tiles: float = 1.0

    BASELINE = {'tile_size': 256, 'overlap': True, 'k': 20, 'min_sep_tiles': 1.0}


class SlidingWinSimRot(IdentifiedBuild):
    """Rotation-aware sliding-window retrieval: stage 2, first phase.

        r = SlidingWinSimRot(
            SlidingWinSimRotConfig(encoder_config('gigapath'))).build(wsi, mask)
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
    ):
        '''Builds its own encoder from `cfg.encoder`, as every stage-1
        estimator does: what the stage computes with is part of the stage.
        `device` and `multi_gpu` are how it is built, not what it is, so they
        are arguments here rather than config fields -- the same split as
        `KnnEstMpp(cfg, device)`.'''
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg = cfg
        self.device = torch.device(device)
        self.encoder = cfg.encoder.build(self.device, multi_gpu=multi_gpu)
        #: What IdentifiedBuild.weights_id hashes: the encoder is this stage's
        #: only trained part.
        self.model = self.encoder.model
        self.tile_size = cfg.tile_size
        self.overlap = cfg.overlap
        #: How many candidates `retrieve` returns.
        self.k = cfg.k
        #: A candidate whose window origin is within this many tiles (level-0
        #: distance) of one already kept AT THE SAME ROTATION is dropped: a
        #: strong peak otherwise fills the list with itself, seen from the main
        #: lattice and from the offset one, and a nominal k of 5 is two real
        #: places. Per rotation on purpose: the same spot at two orientations
        #: is two hypotheses. 0 keeps the raw ranking.
        self.min_sep_tiles = cfg.min_sep_tiles
        #: `read_workers` processes read grid blocks ahead of the encoder; the
        #: CPU budget is the entry point's to decide.
        self.read_workers = read_workers

        self.wsi = self.mask = self.reader = self.feature_store = None
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

        # Per-rotation state (dict keyed by rotation degree)
        self.qc_by_rot:             Dict[int, QueryPatchContainer] = {}
        self.query_features_by_rot: Dict[int, FeaturesMap]         = {}
        self.sim_maps_by_rot: Dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    # ── one slide ────────────────────────────────────────────────────────────
    def build(self, wsi: Union[openslide.OpenSlide, str], mask: Optional[TissueMask],
              feature_store=None) -> 'SlidingWinSimRot':
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
        self.qc_by_rot, self.query_features_by_rot, self.sim_maps_by_rot = {}, {}, {}
        return self

    def retrieve(self, query, estimate) -> CandidateSet:
        """Stage 2 on stage 1's output: the best `k` windows of `query` at the
        level `estimate.chosen_level` routed it to. Stage 1 chooses the level
        once; this does not choose again."""
        self.build_wsi_features(level=int(estimate.chosen_level))
        self.build_query_features(query)
        self.compute_sim_maps()
        return self.candidate_set()

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
            if self.mask is not None:
                self.regions = self.mask.patchable(self.tile_size * self.ds).tissue_regions
            else:
                w0, h0 = self.wsi.level_dimensions[0]
                self.regions = [TissueRegion(x=0, y=0, w=w0, h=h0, index=0)]
            geo = dict(ds=self.ds, level=self.level, tile_size=self.tile_size,
                       overlap=self.overlap)
            self.grids = region_grids(self.regions, **geo)

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
        if isinstance(query, QueryPatchContainer):
            img_np = np.array(query.img)
        elif isinstance(query, Image.Image):
            img_np = np.array(query.convert('RGB'))
        else:
            img_np = np.asarray(query)

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
        """Yield (region_index, lattice, [rows, cols] mean-cosine per window).

        A window's score is the mean cosine over the query's tiles, so the two
        trailing dims collapse. A region smaller than the query has none. The
        maps were computed over the FILTERED regions, which is what
        wsi_features carries, so region_index indexes `self.regions`."""
        for ri, (main_sim, offset_sim) in enumerate(sim_maps):
            for lattice, hm in (('main', main_sim), ('offset', offset_sim)):
                if hm.numel():
                    yield ri, lattice, hm.mean(dim=(-2, -1))

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

        q_grid = self.qc_by_rot[0].grid
        if not raw:
            # No region holds a single window: the sliding kernel needs a
            # region grid at least as large as the query's.
            biggest = max(((main.shape[0], main.shape[1])
                           for main, _ in self.wsi_grids), default=(0, 0))
            raise ValueError(
                f'query does not fit any tissue region at this level: query grid '
                f'is {q_grid.grid_rows}x{q_grid.grid_cols} tiles '
                f'({self.tile_size}px each), the largest region grid is '
                f'{biggest[0]}x{biggest[1]}. Route to a finer level, lower the '
                f'query sensor, or relax the mask filtering.')
        raw.sort(key=lambda c: -c.score)

        frame = dict(level=self.level, ds=self.ds, grids=tuple(self.grids))
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
