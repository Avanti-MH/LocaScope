"""F4 -- a window as a bag of tile clusters, and of pairs of neighbouring ones: a first-phase retriever.

    r   = RetrieverClusterBow(CLUSTERBOW_RECIPES['uni2-avg'], device).build(wsi, mask)
    out = r.retrieve(est_mpp_result, shot_img)

The level's tiles are clustered (k-means on the unit vectors, `k1` clusters, then `k2` inside each: the
clusters of clusters), every tile gets the id of its cluster, and a window becomes the histogram of the ids
under it plus the histogram of its horizontally and vertically adjacent pairs of ids (the graph of clusters
that touch). A query is the same two histograms. A window's score is the histogram intersection with the
idf of each bin as its weight, so a rare cluster counts for more than stroma.

There is no learned part and no tile-to-tile cosine, so it fails where the sliding window does not: a query
whose tiles are each near no single reference tile but whose MIX of tissue types is that of one place. The
histograms of every window are box sums of a one-hot grid (`avg_pool2d`), a slice of bins at a time.

It encodes its own level and query (`Context.WindowSearch`) and offers them as `level_features`, with the
score of every window as `window_scores`. A level is clustered on its first query and kept for the slide.
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
from stage2_retrieval.Context import (GridSource, LevelFeatures, QueryFeatures, WindowSearch,
                                      first_phase_output, top_candidates)
from stage2_retrieval.Retriever import EncoderPool, RetrievalOutput, Retriever, RetrieverConfig
from stage2_retrieval.RetrieverTileVote import level_grids
from stage2_retrieval.StageInterface import ROTATIONS, Candidate, CandidateSet


@register('retriever_clusterbow')
@dataclass(frozen=True)
class RetrieverClusterBowConfig(RetrieverConfig):
    """The encoder (pooling included) and tile cut, the offset lattice, the list's length (`raw_topk`) and
    separation (`min_sep_tiles`); `fit_tiles` of the level (seeded) are clustered into `k1` clusters and
    `k2` inside each, `iters` rounds; a pair of adjacent ids is hashed into `buckets` bins; `bigram_weight`
    is the share of the score the pairs have. `channels` is the number of bins per pass (memory)."""
    encoder:       TileEncoderConfig
    tile_size:     int   = 256
    overlap:       bool  = True
    raw_topk:      int   = 1000
    min_sep_tiles: float = 1.0
    k1:            int   = 32
    k2:            int   = 64
    iters:         int   = 10
    fit_tiles:     int   = 200000
    buckets:       int   = 4096
    bigram_weight: float = 0.5
    seed:          int   = 0
    channels:      int   = 256

    BASELINE = {'tile_size': 256, 'overlap': True, 'raw_topk': 1000, 'min_sep_tiles': 1.0, 'k1': 32, 'k2': 64,
                'iters': 10, 'fit_tiles': 200000, 'buckets': 4096, 'bigram_weight': 0.5, 'seed': 0}
    NOT_IDENTITY = ('channels',)
    VERSION = 1

    def __post_init__(self) -> None:
        if min(self.k1, self.k2, self.iters, self.buckets, self.channels, self.raw_topk) < 1:
            raise ValueError('k1, k2, iters, buckets, channels and raw_topk must be >= 1')
        if not 0.0 <= self.bigram_weight <= 1.0:
            raise ValueError(f'bigram_weight must be in [0, 1], got {self.bigram_weight}')


#: Named retrievers, every field written out; `--stage2 clusterbow:<name>`.
CLUSTERBOW_RECIPES: Dict[str, RetrieverClusterBowConfig] = {
    'uni2-avg': RetrieverClusterBowConfig(
        encoder=encoder_config('uni2', pooling='cls_avg').with_model(dtype='fp16'),
        tile_size=256, overlap=True, raw_topk=1000, min_sep_tiles=1.0, k1=32, k2=64, iters=10,
        fit_tiles=200000, buckets=4096, bigram_weight=0.5, seed=0, channels=256),
}


def spherical_kmeans(x: torch.Tensor, k: int, iters: int, gen: torch.Generator) -> torch.Tensor:
    """`[k', F]` unit centroids of the unit vectors `x` `[n, F]` (`k'` = `min(k, n)`):
    assignment by the largest inner product, update by the normalised mean, an
    empty cluster re-seeded from a random point."""
    n: int = int(x.shape[0])
    k = min(k, n)
    cent: torch.Tensor = x[torch.randperm(n, generator=gen)[:k].to(x.device)].clone()
    for _ in range(iters):
        assign: torch.Tensor = (x @ cent.T).argmax(dim=1)
        total: torch.Tensor = torch.zeros_like(cent).index_add_(0, assign, x)
        count: torch.Tensor = torch.bincount(assign, minlength=k)
        empty: torch.Tensor = count == 0
        total[empty] = x[torch.randint(0, n, (int(empty.sum()),), generator=gen).to(x.device)]
        cent = F.normalize(total, dim=1)
    return cent


class LevelBag:
    """The clustering of one level and the idf of its bins."""

    def __init__(self, grids: List[Tuple[int, str, torch.Tensor]], cfg: RetrieverClusterBowConfig) -> None:
        self.cfg: RetrieverClusterBowConfig = cfg
        gen: torch.Generator = torch.Generator().manual_seed(int(cfg.seed))
        flats: List[torch.Tensor] = [g.reshape(-1, g.shape[-1]) for _, _, g in grids]
        total: int = sum(int(f.shape[0]) for f in flats)
        picked: torch.Tensor = torch.randperm(total, generator=gen)[:min(int(cfg.fit_tiles), total)].sort().values
        edges: np.ndarray = np.cumsum([0] + [int(f.shape[0]) for f in flats])
        sample: torch.Tensor = torch.cat([
            flats[g][(picked[(picked >= edges[g]) & (picked < edges[g + 1])] - int(edges[g])).to(flats[g].device)]
            for g in range(len(flats))]).float()
        top: torch.Tensor = spherical_kmeans(sample, int(cfg.k1), int(cfg.iters), gen)
        owner: torch.Tensor = (sample @ top.T).argmax(dim=1)
        leaves: List[torch.Tensor] = []
        for t in range(int(top.shape[0])):
            members: torch.Tensor = sample[owner == t]
            if len(members):
                leaves.append(spherical_kmeans(members, max(1, min(int(cfg.k2), len(members) // 32)),
                                               int(cfg.iters), gen))
        self.centroids: torch.Tensor = torch.cat(leaves)
        self.n_leaf: int = int(self.centroids.shape[0])
        self.ids: Dict[Tuple[int, str], torch.Tensor] = {
            (region, lattice): self.assign(g) for region, lattice, g in grids}
        count: torch.Tensor = torch.zeros(self.n_leaf, device=self.centroids.device)
        pairs: torch.Tensor = torch.zeros(int(cfg.buckets), device=self.centroids.device)
        for ids in self.ids.values():
            count += torch.bincount(ids.reshape(-1), minlength=self.n_leaf).float()
            for p in self.pair_ids(ids):
                pairs += torch.bincount(p.reshape(-1), minlength=int(cfg.buckets)).float()
        self.idf: torch.Tensor = torch.log((count.sum() + 1.0) / (count + 1.0))
        self.idf_pair: torch.Tensor = torch.log((pairs.sum() + 1.0) / (pairs + 1.0))

    def assign(self, grid: torch.Tensor) -> torch.Tensor:
        """`[R, C]` leaf id of every tile of `grid` `[R, C, F]`."""
        flat: torch.Tensor = grid.reshape(-1, grid.shape[-1])
        out: List[torch.Tensor] = [(flat[s:s + 65536].float() @ self.centroids.T).argmax(dim=1)
                                   for s in range(0, int(flat.shape[0]), 65536)]
        return torch.cat(out).reshape(grid.shape[0], grid.shape[1])

    def pair_ids(self, ids: torch.Tensor) -> List[torch.Tensor]:
        """The hashed ids of the horizontal pairs `[R, C-1]` and the vertical
        pairs `[R-1, C]` of an id grid."""
        b: int = int(self.cfg.buckets)
        horizontal: torch.Tensor = (ids[:, :-1] * self.n_leaf + ids[:, 1:]) % b
        vertical: torch.Tensor = (ids[:-1] * self.n_leaf + ids[1:]) % b
        return [horizontal, vertical]


def histogram_scores(ids: torch.Tensor, n_bins: int, window: Tuple[int, int], q_counts: torch.Tensor,
                     idf: torch.Tensor, channels: int) -> torch.Tensor:
    """`[ho, wo]`: for every `window = (a, b)` block of the id grid `ids` `[R, C]`,
    the idf-weighted histogram intersection of its ids with the query's counts
    `q_counts` `[n_bins]`, over the query's own idf-weighted total."""
    a, b = window
    ho, wo = int(ids.shape[0]) - a + 1, int(ids.shape[1]) - b + 1
    score: torch.Tensor = torch.zeros((ho, wo), device=ids.device)
    if ho < 1 or wo < 1:
        return score
    for s in range(0, n_bins, channels):
        bins: torch.Tensor = torch.arange(s, min(s + channels, n_bins), device=ids.device)
        onehot: torch.Tensor = (ids.unsqueeze(0) == bins[:, None, None]).half().unsqueeze(0)    # [1, c, R, C]
        counts: torch.Tensor = F.avg_pool2d(onehot, (a, b), stride=1)[0].float() * float(a * b)   # [c, ho, wo]
        score += (torch.minimum(counts, q_counts[bins][:, None, None]) * idf[bins][:, None, None]).sum(dim=0)
    return score / float((q_counts * idf).sum().clamp(min=1e-6))


@torch.no_grad()
def window_maps(source: GridSource, bag: LevelBag, cfg: RetrieverClusterBowConfig
                ) -> Dict[Tuple[int, int, str], torch.Tensor]:
    """The score of every window of the level, per `(rotation, region, lattice)`: the whole ranking."""
    n_pair: int = int(cfg.buckets)
    ch: int = int(cfg.channels)
    w_bi: float = float(cfg.bigram_weight)
    maps: Dict[Tuple[int, int, str], torch.Tensor] = {}
    for rot in ROTATIONS:
        qg: torch.Tensor = source.query_grid(rot, source.pooling)
        a, b = int(qg.shape[0]), int(qg.shape[1])
        q_ids: torch.Tensor = bag.assign(qg)
        q_uni: torch.Tensor = torch.bincount(q_ids.reshape(-1), minlength=bag.n_leaf).float()
        q_h, q_v = (torch.bincount(p.reshape(-1), minlength=n_pair).float() for p in bag.pair_ids(q_ids))
        for (region, lattice), ids in bag.ids.items():
            if ids.shape[0] < a or ids.shape[1] < b:
                continue
            score: torch.Tensor = (1.0 - w_bi) * histogram_scores(ids, bag.n_leaf, (a, b), q_uni, bag.idf, ch)
            if w_bi > 0 and a > 1 and b > 1:
                horizontal, vertical = bag.pair_ids(ids)
                score = score + w_bi * 0.5 * (
                    histogram_scores(horizontal, n_pair, (a, b - 1), q_h, bag.idf_pair, ch)[:, :score.shape[1]][:score.shape[0]]
                    + histogram_scores(vertical, n_pair, (a - 1, b), q_v, bag.idf_pair, ch)[:score.shape[0], :score.shape[1]])
            maps[(rot, region, lattice)] = score
    return maps


class RetrieverClusterBow(Retriever):
    """See the module docstring."""

    UPSTREAM = (EstMppResult,)

    def __init__(self, cfg: RetrieverClusterBowConfig, device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False, read_workers: int = 0,
                 feature_store: Optional[Any] = None) -> None:
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: RetrieverClusterBowConfig = cfg
        self._encoder: Optional[TileEncoder] = None
        self.search: WindowSearch = WindowSearch(lambda: self.encoder, cfg.tile_size, cfg.overlap, self.device,
                                                 read_workers, feature_store)
        self._bag: Dict[int, LevelBag] = {}

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

    def build(self, wsi: Union[str, Any], mask: Any) -> 'RetrieverClusterBow':
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        super().build(wsi, mask)
        self.search.build(wsi, mask)
        self._bag = {}
        return self

    def frame(self, level: int) -> CandidateSet:
        assert self.search.levels is not None
        return self.search.levels.frame(level)

    def run(self, upstream: EstMppResult, query: np.ndarray) -> RetrievalOutput:
        made: Dict[str, Any] = {}

        def scorer(level: LevelFeatures, qf: QueryFeatures, frame: CandidateSet) -> CandidateSet:
            source: GridSource = GridSource(level, qf, self.encoder.feature_pooling)
            if level.level not in self._bag:
                self._bag = {level.level: LevelBag(level_grids(source, len(level.grids)), self.cfg)}
            made['scores'] = window_maps(source, self._bag[level.level], self.cfg)
            return top_candidates(made['scores'], frame, self.cfg.raw_topk, self.cfg.min_sep_tiles, Candidate)

        cs, level, qf = self.search.run(int(upstream.chosen_level), query, scorer)
        return first_phase_output(cs, level, qf, self.encoder, self.cfg.tile_size, self.cfg.overlap,
                                  made.get('scores', {}))


__all__ = ['CLUSTERBOW_RECIPES', 'LevelBag', 'RetrieverClusterBow', 'RetrieverClusterBowConfig',
           'histogram_scores', 'spherical_kmeans', 'window_maps']
