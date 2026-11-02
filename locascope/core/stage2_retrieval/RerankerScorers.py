"""Rerankers that read features and nothing else: nothing is read from the slide beyond the level the list
is in, and nothing is learned. Each is a `Retriever` whose upstream is a list.

    RerankerDual       the query's half-tile lattice as well as its main one
    RerankerCsls       cosine corrected for tiles that look like everything
    RerankerCorr       the query's tiles against ALL the window's tiles, not only the one under each
    RerankerCandGraph  a candidate against the candidates around it

    r   = RerankerDual(RerankerDualConfig(encoder=..., topk_in=1000, order_by='dual'), device)
    out = r.build(wsi, mask).retrieve(upstream_output, shot_img)

`ColumnRanker` is what they (and `RerankerEncode`, `RerankerNeural`) share: read the first `topk_in` of the
list, make the table of what came in, get the features (`Context.RerankFeatures`: the upstream's when the
config reuses them, else its own), let the method write its columns, and order the list by the column the
config names (`order_by`, or the one the method's mode makes). The columns go downstream as the `columns`
extra and into the entry's `rerank_scores` table. `stage2_retrieval/spec.md` has the reason for each method.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ConfigIdentity import register
from TileEncoderFunc import TileEncoder, TileEncoderConfig
from stage2_retrieval.Context import (LATTICES, RerankFeatures, ScoringInput, WindowBatch, incoming_table,
                                      offset_pair_cosines, reranked_output, score_rows)
from stage2_retrieval.Retriever import EncoderPool, Extra, RetrievalOutput, Retriever, RetrieverConfig, Uses


# ══════════════════════════════════════════════════════════════════════════════
#  what every reranker does around its scoring
# ══════════════════════════════════════════════════════════════════════════════

class ColumnRanker(Retriever):
    """A retriever whose upstream is a list: it scores the first `topk_in` candidates and orders them by one
    column. Subclasses implement `score(inp) -> {column: [K]}` and `column`; `READS_FEATURES` says whether
    the method reads window features (then the config has `encoder`, `tile_size`, `overlap` and `reuse`),
    `READS_TILES` whether it reads and encodes tiles now (a moved window, raw tokens)."""

    UPSTREAM = (RetrievalOutput,)
    TABLES = ('rerank_scores',)
    READS_FEATURES: bool = True
    READS_TILES: bool = False

    def __init__(self, cfg: RetrieverConfig, device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False,
                 feature_store: Optional[Any] = None, **_: Any) -> None:
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self._encoder: Optional[TileEncoder] = None
        self.features: Optional[RerankFeatures] = None
        if self.READS_FEATURES or self.READS_TILES:
            self.features = RerankFeatures(lambda: self.encoder, cfg.encoder, int(cfg.tile_size),
                                           bool(getattr(cfg, 'overlap', True)), self.device,
                                           bool(getattr(cfg, 'reuse', ())), feature_store)

    @property
    def encoder(self) -> TileEncoder:
        if self._encoder is None:
            self._encoder = self.encoders.get(self.cfg.encoder)
        return self._encoder

    @property
    def column(self) -> str:
        """The column the list is ordered by."""
        return str(self.cfg.order_by)

    def uses(self) -> Tuple[Uses, ...]:
        return self.features.uses() if self.features is not None and self.READS_FEATURES else ()

    def build(self, wsi: Any, mask: Any) -> 'ColumnRanker':
        super().build(wsi, mask)
        if self.features is not None:
            self.features.build(wsi, mask)
        return self

    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(f'{type(self).__name__}.score')

    def run(self, upstream: RetrievalOutput, query: np.ndarray) -> RetrievalOutput:
        cs: Any = replace(upstream.candidates, candidates=upstream.candidates.candidates[:int(self.cfg.topk_in)])
        if not len(cs):
            return RetrievalOutput(candidates=cs, parent=upstream)
        table = incoming_table(cs)
        own: Dict[str, Extra] = {}
        source: Optional[Any] = None
        if self.READS_FEATURES:
            assert self.features is not None
            source, made = self.features.source(upstream, query)
            if made:
                own['level_features'] = Extra('level_features', self.features.key, source)
        tiles: Optional[Any] = self.features.tiles(cs) if self.READS_TILES and self.features is not None else None
        tile_size: int = int(getattr(self.cfg, 'tile_size', cs.grids[0].tile_size if cs.grids else 256))
        inp: ScoringInput = ScoringInput(query, cs, table, source, tile_size, tiles)
        for name, values in self.score(inp).items():
            table.put(name, values.to(table['in'].device))
        return reranked_output(upstream, cs, table, self.column, own)

    def tables(self, out: RetrievalOutput, qc: Any) -> Dict[str, List[Dict[str, Any]]]:
        """`rerank_scores`: every candidate read, by its window, and every column written."""
        rows: List[Dict[str, Any]] = score_rows(out, qc)
        return {'rerank_scores': rows} if rows else {}


def _tiles_per_window(batch: WindowBatch) -> torch.Tensor:
    """`[K]` the number of tiles in each window (the shapes differ by rotation)."""
    return batch.scatter([torch.full((len(g.index),), float(g.shape[0] * g.shape[1]),
                                     device=g.q.device) for g in batch.groups])


# ══════════════════════════════════════════════════════════════════════════════
#  S0  the half-tile lattice of the query
# ══════════════════════════════════════════════════════════════════════════════

DUAL_COLUMNS: Tuple[str, ...] = ('dual', 'cos_main', 'cos_offset')


@register('reranker_dual')
@dataclass(frozen=True)
class RerankerDualConfig(RetrieverConfig):
    """The encoder (pooling included) and tile cut its features are, whether the offset lattice is there,
    how many candidates it reads (`topk_in`), the column it orders by (`order_by`, one of `DUAL_COLUMNS`), and
    whether it reuses the upstream's features (`reuse`, not identity: the bridge lock makes them the same)."""
    encoder:  TileEncoderConfig
    tile_size: int = 256
    overlap:  bool = True
    topk_in:  int = 1000
    order_by: str = 'dual'
    reuse:    Tuple[str, ...] = ()

    BASELINE = {'tile_size': 256, 'overlap': True, 'topk_in': 1000, 'order_by': 'dual'}
    NOT_IDENTITY = ('reuse',)
    VERSION = 1

    def __post_init__(self) -> None:
        if self.order_by not in DUAL_COLUMNS:
            raise ValueError(f'order_by must be one of {DUAL_COLUMNS}, got {self.order_by!r}')


class RerankerDual(ColumnRanker):
    """The retriever matches the query's main tiles against the tiles under them (a 5x4 query: 20 pairs).
    The query also has tiles half a tile in, and the reference has a tile at every one of those places: 12
    more pairs for the same window. `dual` is the mean cosine over all of them.

    Columns: `dual`, `cos_main` (what the sliding window scores with) and `cos_offset`."""

    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        pooling: str = inp.pooling()
        batch: WindowBatch = inp.batch(pooling)
        n_main: torch.Tensor = _tiles_per_window(batch)
        main_sum: torch.Tensor = batch.scatter([g.cos.sum(dim=(1, 2)) for g in batch.groups])
        off_sum, n_off = offset_pair_cosines(inp.cs, inp.source, pooling)
        return {'dual': (main_sum + off_sum) / (n_main + n_off),
                'cos_main': main_sum / n_main,
                'cos_offset': off_sum / n_off.clamp(min=1.0)}


# ══════════════════════════════════════════════════════════════════════════════
#  S1  hubness
# ══════════════════════════════════════════════════════════════════════════════

CSLS_MODES: Tuple[str, ...] = ('csls', 'rank')


@register('reranker_csls')
@dataclass(frozen=True)
class RerankerCslsConfig(RetrieverConfig):
    """The encoder, tile cut, lattices, `topk_in` and `reuse` as `RerankerDualConfig`. `mode` 'csls': the pair
    value is `2 cos - rho_i - r_t`, ordered by `csls`; 'rank': one minus the share of the level's tiles that
    query tile `i` likes more than it likes this one, ordered by `rankpct`. `n_ref` tiles of the level are
    sampled (seeded) to measure how much each reference tile looks like everything (`r_t`, the mean of its
    top-`k` cosines to the sample), and each query tile's own `rho_i` is its top-`k` mean over ALL the
    level's tiles -- so this method reads the whole level. `chunk` is memory, not identity."""
    encoder:   TileEncoderConfig
    tile_size: int = 256
    overlap:   bool = True
    topk_in:   int = 1000
    mode:      str = 'csls'
    n_ref:     int = 20000
    k:         int = 10
    seed:      int = 0
    chunk:     int = 8192
    reuse:     Tuple[str, ...] = ()

    BASELINE = {'tile_size': 256, 'overlap': True, 'topk_in': 1000, 'mode': 'csls', 'n_ref': 20000, 'k': 10,
                'seed': 0}
    NOT_IDENTITY = ('chunk', 'reuse')
    VERSION = 1

    def __post_init__(self) -> None:
        if self.mode not in CSLS_MODES:
            raise ValueError(f'mode must be one of {CSLS_MODES}, got {self.mode!r}')
        if self.n_ref < 2 or self.k < 1 or self.chunk < 1:
            raise ValueError('n_ref >= 2, k >= 1 and chunk >= 1 are required')


class RerankerCsls(ColumnRanker):
    """A tile of generic stroma matches a query tile of generic stroma almost as well as the true place does,
    and matches everything else of its kind too. A window of such tiles scores high for any query that has
    some. The correction takes away what a tile would score against anything (`r_t`) and what the query tile
    scores against anything (`rho_i`), so a pair counts for how much more it is than that.

    Columns: `csls` (mode 'csls') or `rankpct` (mode 'rank'), with `rho_mean` and `hub_mean`, the two
    corrections averaged over the window."""

    def __init__(self, cfg: RerankerCslsConfig, device: Optional[Any] = None, **kw: Any) -> None:
        super().__init__(cfg, device, **kw)
        self._sample: Optional[torch.Tensor] = None
        self._sample_key: Optional[Tuple[Any, ...]] = None

    @property
    def column(self) -> str:
        return 'csls' if self.cfg.mode == 'csls' else 'rankpct'

    def build(self, wsi: Any, mask: Any) -> 'RerankerCsls':
        super().build(wsi, mask)
        self._sample, self._sample_key = None, None
        return self

    # the level's tiles, one grid at a time (no copy of the level is made)
    @staticmethod
    def _grids(inp: ScoringInput, pooling: str) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for region in range(len(inp.cs.grids)):
            for lattice in LATTICES:
                g: torch.Tensor = inp.source.reference_grid(region, lattice, pooling)
                if g.numel():
                    out.append(g.reshape(-1, g.shape[-1]))
        return out

    def _sample_of(self, inp: ScoringInput, pooling: str) -> torch.Tensor:
        key: Tuple[Any, ...] = (inp.cs.level, inp.cs.ds, pooling, len(inp.cs.grids))
        if self._sample is None or self._sample_key != key:
            flats: List[torch.Tensor] = self._grids(inp, pooling)
            sizes: List[int] = [int(f.shape[0]) for f in flats]
            total: int = sum(sizes)
            n: int = min(int(self.cfg.n_ref), total)
            gen: torch.Generator = torch.Generator().manual_seed(int(self.cfg.seed))
            picked: torch.Tensor = torch.randperm(total, generator=gen)[:n].sort().values
            edges: np.ndarray = np.cumsum([0] + sizes)
            parts: List[torch.Tensor] = []
            for g, flat in enumerate(flats):
                inside: torch.Tensor = picked[(picked >= edges[g]) & (picked < edges[g + 1])] - int(edges[g])
                if len(inside):
                    parts.append(flat[inside.to(flat.device)])
            self._sample, self._sample_key = torch.cat(parts).float(), key
        return self._sample

    def _hub(self, inp: ScoringInput, batch: WindowBatch, pooling: str) -> List[torch.Tensor]:
        """Per group `[G, qr, qc]`: r_t of every reference tile under each window."""
        sample: torch.Tensor = self._sample_of(inp, pooling)
        k: int = min(int(self.cfg.k) + 1, int(sample.shape[0]))
        out: List[torch.Tensor] = []
        for grp in batch.groups:
            hub: torch.Tensor = torch.zeros(grp.q.shape[:3], device=grp.q.device)
            cands = [inp.cs.candidates[int(i)] for i in grp.index.tolist()]
            qr, qc = grp.shape
            by_grid: Dict[Tuple[int, str], List[int]] = {}
            for g, c in enumerate(cands):
                by_grid.setdefault((c.region_index, c.lattice), []).append(g)
            for (region, lattice), gs in by_grid.items():
                grid: torch.Tensor = inp.source.reference_grid(region, lattice, pooling)
                rows: torch.Tensor = torch.tensor([cands[g].row for g in gs], dtype=torch.long, device=grid.device)
                cols: torch.Tensor = torch.tensor([cands[g].col for g in gs], dtype=torch.long, device=grid.device)
                ri: torch.Tensor = rows[:, None, None] + torch.arange(qr, device=grid.device)[None, :, None]
                ci: torch.Tensor = cols[:, None, None] + torch.arange(qc, device=grid.device)[None, None, :]
                flat_id: torch.Tensor = ri * int(grid.shape[1]) + ci               # [G, qr, qc]
                uniq, inverse = torch.unique(flat_id.reshape(-1), return_inverse=True)
                tiles: torch.Tensor = grid.reshape(-1, grid.shape[-1])[uniq].float()
                r_parts: List[torch.Tensor] = []
                for s in range(0, int(tiles.shape[0]), 2048):
                    # a sampled tile finds itself first, so the nearest is dropped for every tile alike
                    best: torch.Tensor = (tiles[s:s + 2048] @ sample.T).topk(k, dim=1).values
                    r_parts.append(best[:, 1:].mean(dim=1) if k > 1 else best[:, 0])
                r_u: torch.Tensor = torch.cat(r_parts)
                hub[torch.tensor(gs, device=hub.device)] = r_u[inverse].reshape(flat_id.shape)
            out.append(hub)
        return out

    def _rho_and_rank(self, inp: ScoringInput, pooling: str, rotations: List[int],
                      thresholds: Optional[Dict[int, torch.Tensor]]
                      ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Per rotation: `rho` `[qr, qc]` (top-k mean cosine of each query tile over the whole level) and, in
        rank mode, `frac` `[G_r, qr, qc]` (the share of the level's tiles more similar to the query tile than
        the candidate's own tile)."""
        flats: List[torch.Tensor] = self._grids(inp, pooling)
        total: int = sum(int(f.shape[0]) for f in flats)
        rho: Dict[int, torch.Tensor] = {}
        frac: Dict[int, torch.Tensor] = {}
        for rot in rotations:
            qg: torch.Tensor = inp.source.query_grid(rot, pooling)
            nq: int = int(qg.shape[0] * qg.shape[1])
            q: torch.Tensor = qg.reshape(nq, -1).float()
            best: torch.Tensor = torch.full((nq, int(self.cfg.k)), -2.0, device=q.device)
            counts: Optional[torch.Tensor] = None
            thr: Optional[torch.Tensor] = None
            if thresholds is not None:
                thr = thresholds[rot].reshape(thresholds[rot].shape[0], nq, 1)
                counts = torch.zeros(thr.shape[:2], device=q.device)
            for flat in flats:
                for s in range(0, int(flat.shape[0]), int(self.cfg.chunk)):
                    sims: torch.Tensor = q @ flat[s:s + int(self.cfg.chunk)].float().T
                    best = torch.cat([best, sims.topk(min(int(self.cfg.k), sims.shape[1]), dim=1).values],
                                     dim=1).topk(int(self.cfg.k), dim=1).values
                    if counts is not None and thr is not None:
                        counts += (sims.unsqueeze(0) > thr).sum(dim=-1)
            rho[rot] = best.mean(dim=1).reshape(qg.shape[0], qg.shape[1])
            if counts is not None:
                frac[rot] = (counts / float(total)).reshape(-1, qg.shape[0], qg.shape[1])
        return rho, frac

    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        pooling: str = inp.pooling()
        batch: WindowBatch = inp.batch(pooling)
        rotations: List[int] = sorted({c.rotation for c in inp.cs.candidates})
        cos_by_group: List[torch.Tensor] = [g.cos for g in batch.groups]
        rank_mode: bool = self.cfg.mode == 'rank'
        thresholds: Optional[Dict[int, torch.Tensor]] = None
        where: Dict[int, List[Tuple[int, int]]] = {}
        if rank_mode:
            thresholds, where = {}, {}
            for gi, grp in enumerate(batch.groups):
                for g, i in enumerate(grp.index.tolist()):
                    where.setdefault(inp.cs.candidates[i].rotation, []).append((gi, g))
            for rot, pairs in where.items():
                thresholds[rot] = torch.stack([cos_by_group[gi][g] for gi, g in pairs])
        rho, frac = self._rho_and_rank(inp, pooling, rotations, thresholds)
        hub: List[torch.Tensor] = self._hub(inp, batch, pooling)

        csls: List[torch.Tensor] = []
        rho_mean: List[torch.Tensor] = []
        hub_mean: List[torch.Tensor] = []
        rankpct: List[torch.Tensor] = []
        # the rotation of each window in a group, to pick its query tiles' rho
        for gi, grp in enumerate(batch.groups):
            rot_of: List[int] = [inp.cs.candidates[i].rotation for i in grp.index.tolist()]
            rho_g: torch.Tensor = torch.stack([rho[r] for r in rot_of]).to(grp.q.device)   # [G, qr, qc]
            csls.append((2.0 * cos_by_group[gi] - rho_g - hub[gi]).mean(dim=(1, 2)))
            rho_mean.append(rho_g.mean(dim=(1, 2)))
            hub_mean.append(hub[gi].mean(dim=(1, 2)))
            if rank_mode:
                rows: List[torch.Tensor] = []
                for g, r in enumerate(rot_of):
                    idx: int = where[r].index((gi, g))
                    rows.append(1.0 - frac[r][idx])
                rankpct.append(torch.stack(rows).mean(dim=(1, 2)))
        out: Dict[str, torch.Tensor] = {'rho_mean': batch.scatter(rho_mean),
                                        'hub_mean': batch.scatter(hub_mean)}
        out['rankpct' if rank_mode else 'csls'] = batch.scatter(rankpct if rank_mode else csls)
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  S2  the correspondence matrix
# ══════════════════════════════════════════════════════════════════════════════

CORR_COLUMNS: Tuple[str, ...] = ('diag', 'contrast', 'mnn', 'sinkhorn')


@register('reranker_corr')
@dataclass(frozen=True)
class RerankerCorrConfig(RetrieverConfig):
    """The encoder, tile cut, lattices, `topk_in` and `reuse` as `RerankerDualConfig`; `tau` and `iters` the
    Sinkhorn temperature and iteration count; `order_by` one of `CORR_COLUMNS`."""
    encoder:   TileEncoderConfig
    tile_size: int = 256
    overlap:   bool = True
    topk_in:   int = 1000
    tau:       float = 0.05
    iters:     int = 20
    order_by:  str = 'sinkhorn'
    reuse:     Tuple[str, ...] = ()

    BASELINE = {'tile_size': 256, 'overlap': True, 'topk_in': 1000, 'tau': 0.05, 'iters': 20,
                'order_by': 'sinkhorn'}
    NOT_IDENTITY = ('reuse',)
    VERSION = 1

    def __post_init__(self) -> None:
        if self.tau <= 0 or self.iters < 1:
            raise ValueError('tau > 0 and iters >= 1 are required')
        if self.order_by not in CORR_COLUMNS:
            raise ValueError(f'order_by must be one of {CORR_COLUMNS}, got {self.order_by!r}')


class RerankerCorr(ColumnRanker):
    """The retriever compares the query's tile `i` with the window's tile `i` only. The `n x n` matrix
    `C[i, j]` of all of them says more: at the true place each query tile likes ITS tile best, and at a place
    of uniform texture every tile likes every tile about equally.

    Columns: `diag` (the mean of the diagonal: the retriever's score again), `contrast` (mean over `i` of
    `C[i, i]` minus the best other `C[i, j]`), `mnn` (the share of tiles that are each other's nearest
    neighbour at the same place), `sinkhorn` (the mass an entropic optimal-transport plan puts on the
    diagonal, per tile)."""

    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        batch: WindowBatch = inp.batch()
        diag: List[torch.Tensor] = []
        contrast: List[torch.Tensor] = []
        mnn: List[torch.Tensor] = []
        sink: List[torch.Tensor] = []
        for grp in batch.groups:
            n: int = grp.shape[0] * grp.shape[1]
            q: torch.Tensor = grp.q.reshape(len(grp.index), n, -1).float()
            w: torch.Tensor = grp.inner.reshape(len(grp.index), n, -1).float()
            c: torch.Tensor = torch.einsum('gnf,gmf->gnm', q, w)                 # [G, n, n]
            eye: torch.Tensor = torch.eye(n, dtype=torch.bool, device=c.device)
            d: torch.Tensor = c.diagonal(dim1=1, dim2=2)                          # [G, n]
            other: torch.Tensor = c.masked_fill(eye, float('-inf')).amax(dim=2)
            ids: torch.Tensor = torch.arange(n, device=c.device)
            diag.append(d.mean(dim=1))
            contrast.append((d - other).mean(dim=1))
            mnn.append(((c.argmax(dim=2) == ids) & (c.argmax(dim=1) == ids)).float().mean(dim=1))
            logp: torch.Tensor = c / float(self.cfg.tau)
            for _ in range(int(self.cfg.iters)):
                logp = logp - torch.logsumexp(logp, dim=2, keepdim=True)
                logp = logp - torch.logsumexp(logp, dim=1, keepdim=True)
            sink.append(torch.exp(logp).diagonal(dim1=1, dim2=2).sum(dim=1) / n)
        return {'diag': batch.scatter(diag), 'contrast': batch.scatter(contrast),
                'mnn': batch.scatter(mnn), 'sinkhorn': batch.scatter(sink)}


# ══════════════════════════════════════════════════════════════════════════════
#  S6  the candidates around a candidate
# ══════════════════════════════════════════════════════════════════════════════

CANDGRAPH_COLUMNS: Tuple[str, ...] = ('nbr_support', 'prominence', 'n_nbr', 'rot_margin')


@register('reranker_candgraph')
@dataclass(frozen=True)
class RerankerCandGraphConfig(RetrieverConfig):
    """`topk_in` candidates read; `on` the column of the incoming table the graph reads (`in`: the score the
    list came with; `first`: the first phase's); `radius` how far (in tiles) two candidates' centres may be to
    be neighbours; `order_by` one of `CANDGRAPH_COLUMNS`. No features: the tile size is the window's."""
    tile_size: int = 256
    topk_in:   int = 1000
    on:        str = 'in'
    radius:    float = 2.0
    order_by:  str = 'prominence'

    BASELINE = {'tile_size': 256, 'topk_in': 1000, 'on': 'in', 'radius': 2.0, 'order_by': 'prominence'}
    VERSION = 1

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f'radius must be > 0, got {self.radius}')
        if self.on not in ('in', 'first'):
            raise ValueError(f"on must be 'in' or 'first', got {self.on!r}")
        if self.order_by not in CANDGRAPH_COLUMNS:
            raise ValueError(f'order_by must be one of {CANDGRAPH_COLUMNS}, got {self.order_by!r}')


class RerankerCandGraph(ColumnRanker):
    """Candidates are nodes; two are joined when their centres are within `radius` tiles (whatever their
    rotations). The shape of the scores around a candidate says something its own score does not -- and which
    way it says it is not assumed: a true place is a peak, a patch of uniform texture is a plateau, but a true
    place whose neighbours also score (they share tiles with it) is a hill.

    Columns: `nbr_support` (the mean score of the neighbours), `prominence` (own minus that), `n_nbr`,
    `rot_margin` (own minus the best score of the same place at another rotation; own minus the lowest score
    when it has none) -- a true place is one rotation's peak, an isotropic texture is high at all four."""

    READS_FEATURES = False

    def encoder_needs(self) -> Dict[str, Any]:
        return {}

    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        s: torch.Tensor = inp.table[self.cfg.on]
        centres: torch.Tensor = torch.as_tensor(inp.centres(), dtype=torch.float64, device=s.device)
        tile_px: float = float(inp.cs.grids[0].tile_size) * float(inp.cs.ds)
        dist: torch.Tensor = torch.cdist(centres, centres)
        eye: torch.Tensor = torch.eye(len(s), dtype=torch.bool, device=s.device)
        near: torch.Tensor = (dist <= float(self.cfg.radius) * tile_px) & ~eye
        n_nbr: torch.Tensor = near.sum(dim=1).float()
        support: torch.Tensor = (near.float() @ s) / n_nbr.clamp(min=1.0)
        prominence: torch.Tensor = torch.where(n_nbr > 0, s - support, torch.zeros_like(s))
        rotation: torch.Tensor = torch.tensor([c.rotation for c in inp.cs.candidates], device=s.device)
        same_place: torch.Tensor = (dist < 0.5 * tile_px) & (rotation[:, None] != rotation[None, :])
        best_other: torch.Tensor = s[None, :].expand(len(s), -1).masked_fill(~same_place, float('-inf')).amax(dim=1)
        margin: torch.Tensor = torch.where(same_place.any(dim=1), s - best_other, s - s.min())
        return {'nbr_support': support, 'prominence': prominence, 'n_nbr': n_nbr, 'rot_margin': margin}


__all__ = ['CANDGRAPH_COLUMNS', 'CORR_COLUMNS', 'CSLS_MODES', 'ColumnRanker', 'DUAL_COLUMNS', 'RerankerCandGraph',
           'RerankerCandGraphConfig', 'RerankerCorr', 'RerankerCorrConfig', 'RerankerCsls', 'RerankerCslsConfig',
           'RerankerDual', 'RerankerDualConfig']
