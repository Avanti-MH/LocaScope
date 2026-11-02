"""F2 -- the query's tiles find their nearest tiles on the slide, and vote: a first-phase retriever.

    r   = RetrieverTileVote(TILEVOTE_RECIPES['uni2-avg-exact'], device).build(wsi, mask)
    out = r.retrieve(est_mpp_result, shot_img)

Each query tile looks up its `m` most similar reference tiles on the whole level. A hit at `(row, col)` of a
lattice says the window would start at `(row - i, col - j)` if query tile `(i, j)` really is that tile; it
votes for that window with the softmax weight of its similarity. A window's score is the share of the votes
it gathered, after a small box filter so a vote for a neighbouring window helps a little (the truth need not
be ON a lattice).

It is the sliding window seen from the other side: the sliding window asks, of each window, how well its
tiles match the query's; this asks, of each query tile, where its matches are, and lets them vote for the
window they would belong to. What it adds is the INDEX behind the look-up -- `exact` (a GPU matrix product),
or a graph (`hnsw`) or an inverted file (`ivf`) over all the tiles of the level, `faiss` on the CPU -- so a
slide too large to slide over could be searched, and the graph's own errors can be measured against `exact`
on the slides that are not. No training.

It encodes its own level and query (`Context.WindowSearch`) and offers them as `level_features`, with the
score of every window as `window_scores`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ConfigIdentity import register
from SafeSlide import SafeSlide
from TileEncoderFunc import TileEncoder, TileEncoderConfig, encoder_config
from stage1_estimation.StageInterface import EstMppResult
from stage2_retrieval.Context import (LATTICES, GridSource, LevelFeatures, QueryFeatures, WindowSearch,
                                      first_phase_output, top_candidates)
from stage2_retrieval.Retriever import EncoderPool, RetrievalOutput, Retriever, RetrieverConfig
from stage2_retrieval.StageInterface import ROTATIONS, Candidate, CandidateSet, VoteCandidate

BACKENDS: Tuple[str, ...] = ('exact', 'hnsw', 'ivf')


@register('retriever_tilevote')
@dataclass(frozen=True)
class RetrieverTileVoteConfig(RetrieverConfig):
    """The encoder (pooling included) and tile cut, the offset lattice, the list's length (`raw_topk`) and
    separation (`min_sep_tiles`); `m` hits per query tile, weighted `softmax(sim / tau)`; the vote map is
    box-filtered over `2 * smooth + 1` windows. `hnsw_m` and `ef_search` are the graph's degree and search
    width, `nlist` and `nprobe` the inverted file's cells and cells searched: they change what is found, so
    they are identity. `chunk` (tiles per matrix product) is memory."""
    encoder:       TileEncoderConfig
    tile_size:     int   = 256
    overlap:       bool  = True
    raw_topk:      int   = 1000
    min_sep_tiles: float = 1.0
    backend:       str   = 'exact'
    m:             int   = 64
    tau:           float = 0.05
    smooth:        int   = 1
    hnsw_m:        int   = 32
    ef_search:     int   = 128
    nlist:         int   = 1024
    nprobe:        int   = 16
    chunk:         int   = 8192

    BASELINE = {'tile_size': 256, 'overlap': True, 'raw_topk': 1000, 'min_sep_tiles': 1.0, 'backend': 'exact',
                'm': 64, 'tau': 0.05, 'smooth': 1, 'hnsw_m': 32, 'ef_search': 128, 'nlist': 1024, 'nprobe': 16}
    NOT_IDENTITY = ('chunk',)
    VERSION = 1

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f'backend must be one of {BACKENDS}, got {self.backend!r}')
        if self.m < 1 or self.smooth < 0 or self.tau <= 0 or self.chunk < 1 or self.raw_topk < 1:
            raise ValueError('m >= 1, smooth >= 0, tau > 0, chunk >= 1 and raw_topk >= 1 are required')


#: Named retrievers, every field written out; `--stage2 tilevote:<name>`.
TILEVOTE_RECIPES: Dict[str, RetrieverTileVoteConfig] = {
    'uni2-avg-exact': RetrieverTileVoteConfig(
        encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, backend='exact', m=64, tau=0.05,
        smooth=1, hnsw_m=32, ef_search=128, nlist=1024, nprobe=16, chunk=8192),
}


class TileIndex:
    """The nearest tiles of the whole level, by inner product (the tiles are unit
    vectors, so by cosine). `search(q, m)` returns `(values, grid, flat)`, each
    `[nq, m]`: the hit's similarity, which grid it is in (index into `grids`) and
    its flat index in that grid."""

    def __init__(self, grids: List[Tuple[int, str, torch.Tensor]], cfg: RetrieverTileVoteConfig) -> None:
        self.grids: List[Tuple[int, str, torch.Tensor]] = grids
        self.cfg: RetrieverTileVoteConfig = cfg
        self.flats: List[torch.Tensor] = [g.reshape(-1, g.shape[-1]) for _, _, g in grids]
        self.sizes: np.ndarray = np.array([int(f.shape[0]) for f in self.flats], dtype=np.int64)
        self.edges: np.ndarray = np.concatenate([[0], np.cumsum(self.sizes)])
        self._faiss: Optional[Any] = None
        if cfg.backend != 'exact':
            self._faiss = self._build()

    def _build(self) -> Any:
        import faiss                                                  # noqa: PLC0415
        data: np.ndarray = np.concatenate([f.float().cpu().numpy() for f in self.flats]).astype(np.float32)
        d: int = int(data.shape[1])
        if self.cfg.backend == 'hnsw':
            index: Any = faiss.IndexHNSWFlat(d, int(self.cfg.hnsw_m), faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = int(self.cfg.ef_search)
        else:
            # faiss cannot train more clusters than it has points: a small level (a few hundred tiles) gets
            # as many as it has; a level with enough is indexed with `nlist` exactly as before
            nlist: int = min(int(self.cfg.nlist), len(data))
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
            rng: np.random.Generator = np.random.default_rng(0)
            index.train(data[rng.choice(len(data), size=min(len(data), 64 * nlist), replace=False)])
            index.nprobe = int(self.cfg.nprobe)
        index.add(data)
        return index

    def search(self, q: torch.Tensor, m: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._faiss is not None:
            vals, ids = self._faiss.search(q.float().cpu().numpy().astype(np.float32), m)
            gid: np.ndarray = np.searchsorted(self.edges, ids, side='right') - 1
            flat: np.ndarray = ids - self.edges[np.clip(gid, 0, len(self.sizes) - 1)]
            bad: np.ndarray = ids < 0                       # fewer than m found
            vals = np.where(bad, -2.0, vals)
            dev: Any = q.device
            return (torch.from_numpy(vals).to(dev), torch.from_numpy(np.where(bad, 0, gid)).to(dev),
                    torch.from_numpy(np.where(bad, 0, flat)).to(dev))
        nq: int = int(q.shape[0])
        best_v: torch.Tensor = torch.full((nq, m), -2.0, device=q.device)
        best_g: torch.Tensor = torch.zeros((nq, m), dtype=torch.long, device=q.device)
        best_f: torch.Tensor = torch.zeros((nq, m), dtype=torch.long, device=q.device)
        qf: torch.Tensor = q.float()
        step: int = int(self.cfg.chunk)
        for gi, flat_g in enumerate(self.flats):
            for s in range(0, int(flat_g.shape[0]), step):
                sims: torch.Tensor = qf @ flat_g[s:s + step].float().T
                v, i = sims.topk(min(m, sims.shape[1]), dim=1)
                cv: torch.Tensor = torch.cat([best_v, v], dim=1)
                cg: torch.Tensor = torch.cat([best_g, torch.full_like(i, gi)], dim=1)
                cf: torch.Tensor = torch.cat([best_f, i + s], dim=1)
                top: torch.Tensor = cv.topk(m, dim=1).indices
                best_v, best_g, best_f = cv.gather(1, top), cg.gather(1, top), cf.gather(1, top)
        return best_v, best_g, best_f


def level_grids(source: GridSource, n_regions: int) -> List[Tuple[int, str, torch.Tensor]]:
    """`(region, lattice, grid [R, C, F])` of every non-empty grid of the level."""
    out: List[Tuple[int, str, torch.Tensor]] = []
    for region in range(n_regions):
        for lattice in LATTICES:
            g: torch.Tensor = source.reference_grid(region, lattice, source.pooling)
            if g.numel():
                out.append((region, lattice, g))
    return out


def window_maps(source: GridSource, index: TileIndex, cfg: RetrieverTileVoteConfig
                ) -> Dict[Tuple[int, int, str], torch.Tensor]:
    """The score of every window of the level, per `(rotation, region, lattice)`: the whole ranking."""
    m: int = int(cfg.m)
    maps: Dict[Tuple[int, int, str], torch.Tensor] = {}
    for rot in ROTATIONS:
        qg: torch.Tensor = source.query_grid(rot, source.pooling)
        a, b = int(qg.shape[0]), int(qg.shape[1])
        vals, gid, flat = index.search(qg.reshape(a * b, -1), m)
        weight: torch.Tensor = torch.softmax(vals / float(cfg.tau), dim=1)                # [nq, m]
        qi: torch.Tensor = torch.arange(a * b, device=vals.device)
        ti, tj = (qi // b)[:, None], (qi % b)[:, None]
        for gi, (region, lattice, grid) in enumerate(index.grids):
            if grid.shape[0] < a or grid.shape[1] < b:
                continue
            ho, wo = int(grid.shape[0]) - a + 1, int(grid.shape[1]) - b + 1
            here: torch.Tensor = gid == gi
            row: torch.Tensor = flat // int(grid.shape[1]) - ti
            col: torch.Tensor = flat % int(grid.shape[1]) - tj
            ok: torch.Tensor = here & (row >= 0) & (row < ho) & (col >= 0) & (col < wo)
            vote: torch.Tensor = torch.zeros(ho * wo, device=vals.device)
            vote.index_add_(0, (row * wo + col)[ok], weight[ok])
            vote = vote.reshape(1, 1, ho, wo) / float(a * b)
            if cfg.smooth:
                r: int = int(cfg.smooth)
                vote = F.avg_pool2d(vote, 2 * r + 1, stride=1, padding=r, count_include_pad=False)
            maps[(rot, region, lattice)] = vote[0, 0]
    return maps


def vote_candidate(region: int, lattice: str, row: int, col: int, rotation: int, score: float) -> Candidate:
    """A listed window, carrying the share of the votes it gathered."""
    return VoteCandidate(region, lattice, row, col, rotation, score, votes=score)


class RetrieverTileVote(Retriever):
    """See the module docstring. The index of a level is built on its first query and kept for the slide."""

    UPSTREAM = (EstMppResult,)

    def __init__(self, cfg: RetrieverTileVoteConfig, device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False, read_workers: int = 0,
                 feature_store: Optional[Any] = None) -> None:
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: RetrieverTileVoteConfig = cfg
        self._encoder: Optional[TileEncoder] = None
        self.search: WindowSearch = WindowSearch(lambda: self.encoder, cfg.tile_size, cfg.overlap, self.device,
                                                 read_workers, feature_store)
        self._index: Dict[int, TileIndex] = {}

    @property
    def encoder(self) -> TileEncoder:
        if self._encoder is None:
            self._encoder = self.encoders.get(self.cfg.encoder)
        return self._encoder

    @property
    def model(self) -> torch.nn.Module:
        return self.encoder.model

    @model.setter
    def model(self, value: Any) -> None:
        """The encoder is the model here."""

    def build(self, wsi: Union[str, Any], mask: Any) -> 'RetrieverTileVote':
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        super().build(wsi, mask)
        self.search.build(wsi, mask)
        self._index = {}
        return self

    def frame(self, level: int) -> CandidateSet:
        assert self.search.levels is not None
        return self.search.levels.frame(level)

    def run(self, upstream: EstMppResult, query: np.ndarray) -> RetrievalOutput:
        made: Dict[str, Any] = {}

        def scorer(level: LevelFeatures, qf: QueryFeatures, frame: CandidateSet) -> CandidateSet:
            source: GridSource = GridSource(level, qf, self.encoder.feature_pooling)
            if level.level not in self._index:
                self._index = {level.level: TileIndex(level_grids(source, len(level.grids)), self.cfg)}
            made['scores'] = window_maps(source, self._index[level.level], self.cfg)
            return top_candidates(made['scores'], frame, self.cfg.raw_topk, self.cfg.min_sep_tiles,
                                  vote_candidate)

        cs, level, qf = self.search.run(int(upstream.chosen_level), query, scorer)
        return first_phase_output(cs, level, qf, self.encoder, self.cfg.tile_size, self.cfg.overlap,
                                  made.get('scores', {}))


__all__ = ['BACKENDS', 'RetrieverTileVote', 'RetrieverTileVoteConfig', 'TILEVOTE_RECIPES', 'TileIndex',
           'level_grids', 'vote_candidate', 'window_maps']
