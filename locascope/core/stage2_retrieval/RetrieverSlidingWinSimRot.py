"""The rotation-aware sliding window: a first-phase retriever, and the window kernel it runs on.

    cfg = SLIDEWIN_RECIPES['gigapath']
    r   = RetrieverSlidingWinSimRot(cfg, device).build(wsi, mask)
    out = r.retrieve(est_mpp_result, shot_img)          # stage 1's output in, RetrievalOutput out
    out.candidates.best, out.candidates.origin_l0(out.candidates.best)

The query is turned to all 4 cardinal rotations (0, 90, 180, 270 deg) and every window of every rotation,
lattice and region is scored together: the cosine of each query tile with the reference tile under it,
reduced to one number per window by the config's `score` (`WINDOW_SCORES`). The best `raw_topk` windows,
a window dropped when a better one at its rotation lies within `min_sep_tiles`, are the list.

Cost: the level is encoded ONCE per slide and shared by the 4 rotations of every query; a query costs its
4 encodings and the similarity maps.

What it offers downstream (`RetrievalOutput.extras`):

    level_features   a `GridSource` over the level's and the query's features (`Context.feature_key`)
    window_scores    the score of EVERY window, `{(rotation, region, lattice): [rows, cols]}`: the whole
                     ranking, which a bench ranks the truth in without the top-k in the way
    tile_sims        the per-tile cosines, `{rotation: [(main, offset) per region]}`, each [H, W, R_q, C_q]

`SlidingWindowSimilarity` (the per-window cosine kernel) lives here, so the benches import the kernel from
the place the retriever runs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ConfigIdentity import register
from PatchingLib import FeaturesMap
from SafeSlide import SafeSlide
from TileEncoderFunc import TileEncoder, TileEncoderConfig, encoder_config
from stage1_estimation.StageInterface import EstMppResult
from stage2_retrieval.Context import (LevelFeatures, QueryFeatures, WindowSearch, first_phase_output,
                                      top_candidates)
from stage2_retrieval.Retriever import EncoderPool, Extra, RetrievalOutput, Retriever, RetrieverConfig
from stage2_retrieval.StageInterface import Candidate, CandidateSet


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
    the concatenated multi-slot descriptors bench_stage2_retriever builds.

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
               wsi_ov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    them. The retriever does not come through here: it keeps each level's two
    WSI grids, arranged once (`LevelFeatures.tensors`), and calls `_grid_sims`
    per query.
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
    when the rest disagree -- an OR. Reduced in the maps' own dtype."""
    return per_tile.mean(dim=(-2, -1))


def _score_min(per_tile: torch.Tensor) -> torch.Tensor:
    """The worst tile of the window: one tile that does not match sinks it -- an AND, the sharpest."""
    return per_tile.float().amin(dim=(-2, -1))


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
    'min': _score_min,
}


def window_score(per_tile: torch.Tensor, kind: str = 'mean') -> torch.Tensor:
    """The score `kind` of every window of `per_tile` (`[..., R_q, C_q]`)."""
    try:
        fn: Callable[[torch.Tensor], torch.Tensor] = WINDOW_SCORES[kind]
    except KeyError:
        raise ValueError(f'no window score {kind!r}; the scores are '
                         f'{", ".join(WINDOW_SCORES)}') from None
    return fn(per_tile)


def window_score_maps(sim_maps: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]], kind: str
                      ) -> Dict[Tuple[int, int, str], torch.Tensor]:
    """`{(rotation, region, lattice): [rows, cols]}` the score of every window of the per-tile maps, in the
    order the list is made in (rotation, then region, main before offset). A region smaller than the query
    has no window and no entry."""
    out: Dict[Tuple[int, int, str], torch.Tensor] = {}
    for rot, per_region in sim_maps.items():
        for ri, (main_sim, offset_sim) in enumerate(per_region):
            for lattice, hm in (('main', main_sim), ('offset', offset_sim)):
                if hm.numel():
                    out[(rot, ri, lattice)] = window_score(hm, kind)
    return out


# ── the retriever ─────────────────────────────────────────────────────────────

@register('retriever_slidewin')
@dataclass(frozen=True)
class RetrieverSlidingWinSimRotConfig(RetrieverConfig):
    '''The encoder it scores with (a TileEncoderConfig, precision and pooling included), the tile it cuts,
    whether it also slides the half-tile offset lattice, how many windows it lists (`raw_topk`, also the
    number of windows each score map is first cut to), how far apart two of them at one rotation must be
    (`min_sep_tiles`, in tiles at the level searched; 0 keeps the raw ranking), and the window `score`.

    Every field is identity: `raw_topk` and `min_sep_tiles` change no score, but they change which windows
    the list holds, and the list is the output. The consumer's `topk` (a prefix) is not a field. The
    encoder's own NOT_IDENTITY (batch_size) still applies inside it.'''
    encoder:       TileEncoderConfig
    tile_size:     int   = 256
    overlap:       bool  = True
    raw_topk:      int   = 1000
    min_sep_tiles: float = 1.0
    score:         str   = 'mean'

    BASELINE = {'tile_size': 256, 'overlap': True, 'raw_topk': 1000, 'min_sep_tiles': 1.0, 'score': 'mean'}
    VERSION = 1

    def __post_init__(self) -> None:
        if self.score not in WINDOW_SCORES:
            raise ValueError(f'no window score {self.score!r}; the scores are '
                             f'{", ".join(WINDOW_SCORES)}')
        if self.raw_topk < 1:
            raise ValueError(f'raw_topk must be >= 1, got {self.raw_topk}')


#: How many listed windows the `tile_sims` table holds the per-tile cosines of: the top of a list is what is
#: looked at (a list of 1000 would be 20,000 rows a FoV).
TILE_SIM_ROWS: int = 10


#: Named retrievers, every field written out (test_config_identity's recipe lint); `--stage2 slidewin:<name>`.
#: The encoder is the registry's own config at fp16.
SLIDEWIN_RECIPES: Dict[str, RetrieverSlidingWinSimRotConfig] = {
    'gigapath': RetrieverSlidingWinSimRotConfig(
        encoder=encoder_config('gigapath').with_model(dtype='fp16'),
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, score='mean'),
    'uni2': RetrieverSlidingWinSimRotConfig(
        encoder=encoder_config('uni2').with_model(dtype='fp16'),
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, score='mean'),
    # cls_avg: the CLS and the patch mean as two slots. The window bench's best pooling for uni2, and what
    # the rerankers read their features in.
    'uni2-avg': RetrieverSlidingWinSimRotConfig(
        encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, score='mean'),
}


class RetrieverSlidingWinSimRot(Retriever):
    """See the module docstring. `build` binds a slide; `retrieve` takes stage 1's output, encodes the level
    it routed to on first use (kept for the slide, and through `feature_store` when one is given), scores
    every window of the query at 4 rotations, and lists the best `raw_topk`.

    `read_workers` processes read grid blocks ahead of the encoder (the CPU budget is the caller's to
    decide); `feature_store` is anything with `load` / `save` of a level's features (`Context.encode_level`)."""

    UPSTREAM = (EstMppResult,)
    TABLES = ('tile_sims',)

    def __init__(self, cfg: RetrieverSlidingWinSimRotConfig, device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False, read_workers: int = 0,
                 feature_store: Optional[Any] = None) -> None:
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: RetrieverSlidingWinSimRotConfig = cfg
        self._encoder: Optional[TileEncoder] = None
        #: the slide, its levels, their features (`Context.WindowSearch`)
        self.search: WindowSearch = WindowSearch(lambda: self.encoder, cfg.tile_size, cfg.overlap, self.device,
                                                 read_workers, feature_store)

    @property
    def level_encodes(self) -> Dict[int, Tuple[float, int]]:
        """`{level: (seconds, tiles)}` of every level encoded since `build`."""
        return self.search.level_encodes

    @property
    def last_level_seconds(self) -> float:
        """Seconds the last `retrieve` spent building levels (0 when cached)."""
        return self.search.last_level_seconds

    @property
    def encoder(self) -> TileEncoder:
        """The tile encoder, from the pool on first use (loaded once per trunk)."""
        if self._encoder is None:
            self._encoder = self.encoders.get(self.cfg.encoder)
        return self._encoder

    @property
    def model(self) -> torch.nn.Module:
        '''What IdentifiedBuild.weights_id hashes: the encoder is this retriever's only trained part.'''
        return self.encoder.model

    @model.setter
    def model(self, value: Any) -> None:
        """The base class sets `model` to None; the encoder is the model here."""

    @property
    def tile_size(self) -> int:
        return self.cfg.tile_size

    @property
    def overlap(self) -> bool:
        return self.cfg.overlap

    # ── one slide ────────────────────────────────────────────────────────────
    def build(self, wsi: Union[str, Any], mask: Any) -> 'RetrieverSlidingWinSimRot':
        """Bind a slide and its mask. Cheap: no level is built until one is asked for."""
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)          # so a hole in a MIRAX cannot kill the handle mid-run
        super().build(wsi, mask)
        self.search.build(wsi, mask)
        return self

    def frame(self, level: int) -> CandidateSet:
        """An empty CandidateSet in `level`'s frame (geometry only)."""
        assert self.search.levels is not None
        return self.search.levels.frame(level)

    # ── one query ────────────────────────────────────────────────────────────
    def run(self, upstream: EstMppResult, query: np.ndarray) -> RetrievalOutput:
        made: Dict[str, Any] = {}

        def scorer(level: LevelFeatures, qf: QueryFeatures, frame: CandidateSet) -> CandidateSet:
            cs, made['sims'], made['scores'] = list_windows(level, qf, frame, self.cfg)
            return cs

        cs, level, qf = self.search.run(int(upstream.chosen_level), query, scorer)
        out: RetrievalOutput = first_phase_output(cs, level, qf, self.encoder, self.cfg.tile_size,
                                                  self.cfg.overlap, made.get('scores', {}))
        if out.extras:
            out.extras['tile_sims'] = Extra('tile_sims', out.extras['level_features'].key, made['sims'])
        return out

    # ── what it writes of what it offers ─────────────────────────────────────
    def tables(self, out: RetrievalOutput, qc: Any) -> Dict[str, List[Dict[str, Any]]]:
        """`tile_sims`: for the first `TILE_SIM_ROWS` listed windows, every query tile's cosine with the
        reference tile under it."""
        if 'tile_sims' not in out.extras:
            return {}
        return {'tile_sims': tile_sim_rows(out.extras['tile_sims'].value,
                                           out.candidates.candidates[:TILE_SIM_ROWS])}


def list_windows(level: LevelFeatures, query: QueryFeatures, frame: CandidateSet,
                 cfg: RetrieverSlidingWinSimRotConfig
                 ) -> Tuple[CandidateSet, Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]],
                            Dict[Tuple[int, int, str], torch.Tensor]]:
    """`(list, per-tile maps, window scores)` of one query against one level: the per-tile cosines of every
    window at every rotation, their `cfg.score`, and the best `cfg.raw_topk` windows in `frame`."""
    sims: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {
        rot: [_grid_sims(qfm.main_feature_grid(), main, offset) for main, offset in level.tensors]
        for rot, qfm in query.features.items()}
    scores: Dict[Tuple[int, int, str], torch.Tensor] = window_score_maps(sims, cfg.score)
    return top_candidates(scores, frame, cfg.raw_topk, cfg.min_sep_tiles, Candidate), sims, scores


def window_tile_sims(sims: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]], c: Candidate) -> torch.Tensor:
    """[rows, cols] cosine of every query tile against the reference tile under it in window `c` -- the
    numbers `c.score` is the retriever's `score` of."""
    main, offset = sims[c.rotation][c.region_index]
    return (main if c.lattice == 'main' else offset)[c.row, c.col]


def tile_sim_rows(sims: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]], candidates: Sequence[Candidate]
                  ) -> List[Dict[str, Union[int, float]]]:
    """The per-tile cosines of `candidates` as table rows: for each, in order (`rank` is its place in
    `candidates`, 1 the first), every query tile (`q_row`, `q_col`, on the query turned to the candidate's
    rotation), the reference tile under it on the window's lattice (`ref_row`, `ref_col`) and their `cosine`."""
    out: List[Dict[str, Union[int, float]]] = []
    for rank, c in enumerate(candidates, 1):
        grid: np.ndarray = window_tile_sims(sims, c).float().cpu().numpy()
        for q_row in range(grid.shape[0]):
            for q_col in range(grid.shape[1]):
                out.append(dict(rank=rank, q_row=q_row, q_col=q_col, ref_row=c.row + q_row,
                                ref_col=c.col + q_col, cosine=float(grid[q_row, q_col])))
    return out


__all__ = ['GEOMEAN_FLOOR', 'RetrieverSlidingWinSimRot', 'RetrieverSlidingWinSimRotConfig', 'SLIDEWIN_RECIPES',
           'SlidingWindowSimilarity', 'WINDOW_SCORES', 'list_windows', 'tile_sim_rows', 'window_score',
           'window_score_maps', 'window_tile_sims']
