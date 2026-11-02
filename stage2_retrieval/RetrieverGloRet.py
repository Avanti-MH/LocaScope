"""GloRet and LogLoRet (`proposal.md`) as stage-2 methods.

    GloRet-{ContextEn}-{GlobalEn}::{TilePool}+{search}     Arch 1, a bi-encoder: a window and the query are each
                                                           encoded alone and meet in one inner product
    LogLoRet-{ContextEn}-{GlobalEn}::{TilePool}+{no-match} Arch 2, a cross-attention scorer of one query and one window

A method takes its windows in the two ways `Retriever.py` names, and each class here says which:

    RetrieverGloRet      both. `run_wsi` (the first phase): the vector of EVERY window of the level is made once and
                         kept, and a query is one matrix-vector product per rotation (`Align`, which has no vector,
                         keeps the contextualised tokens and takes a cosine per tile). `run_candidates` (any later
                         phase): the windows of the list it is handed, one by one.
    RerankerLogLoRet     the list only: its score reads a query and one window together, so nothing about a window
                         can be made before the query is known. As the first phase it is handed every window.

The first phase's config is `RetrieverGloRetConfig` (`raw_topk`, `min_sep_tiles`); a later phase's is
`RerankerGloRetConfig` (`topk_in`, and `reuse`: the features of the phase before). Both build `RetrieverGloRet`.
The two ways give the same number for the same window (`test_neural`).

The checkpoint is what `training/GloRet` writes: the network, which registry name built it (`gloret`, `logloret`) and
the features it reads (`pooling` of `encoder`). It is named in the config (`weights`), which is identity: another file
is another retriever.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ConfigIdentity import register
from SafeSlide import SafeSlide
from TileEncoderFunc import TileEncoder, TileEncoderConfig
from stage1_estimation.StageInterface import EstMppResult, weights_path
from stage2_retrieval.Context import (LATTICES, GridSource, LevelFeatures, QueryFeatures, RerankFeatures,
                                      ScoringInput, WindowBatch, WindowSearch, first_phase_output,
                                      gather_windows, incoming_table, reranked_output, top_candidates)
from stage2_retrieval.Retriever import EncoderPool, Extra, RetrievalOutput, Retriever, RetrieverConfig
from stage2_retrieval.StageInterface import ROTATIONS, Candidate, CandidateSet

SEARCHES: Tuple[str, ...] = ('exact', 'hnsw', 'ivf')


@register('retriever_gloret')
@dataclass(frozen=True)
class RetrieverGloRetConfig(RetrieverConfig):
    """GloRet as the first phase. `weights` the checkpoint (by name, under the results root), `search` how the
    window vectors are searched (only `exact` for now), `band` how many rows of windows are encoded at a time."""
    encoder:       TileEncoderConfig
    weights:       str
    method:        str
    tile_size:     int   = 256
    overlap:       bool  = True
    raw_topk:      int   = 1000
    min_sep_tiles: float = 1.0
    search:        str   = 'exact'
    band:          int   = 16

    BASELINE = {'tile_size': 256, 'overlap': True, 'raw_topk': 1000, 'min_sep_tiles': 1.0, 'search': 'exact'}
    NOT_IDENTITY = ('band',)
    VERSION = 1

    def __post_init__(self) -> None:
        if self.search not in SEARCHES:
            raise ValueError(f'search must be one of {SEARCHES}, got {self.search!r}')
        if self.band < 1 or self.raw_topk < 1:
            raise ValueError('band and raw_topk must be >= 1')


@register('reranker_gloret')
@dataclass(frozen=True)
class RerankerGloRetConfig(RetrieverConfig):
    """GloRet as a later phase: it scores the first `topk_in` windows of the list before it."""
    encoder:   TileEncoderConfig
    weights:   str
    method:    str
    tile_size: int = 256
    overlap:   bool = True
    topk_in:   int = 1000
    reuse:     Tuple[str, ...] = ()
    batch:     int = 256

    BASELINE = {'tile_size': 256, 'overlap': True, 'topk_in': 1000}
    NOT_IDENTITY = ('reuse', 'batch')
    VERSION = 1


@register('reranker_logloret')
@dataclass(frozen=True)
class RerankerLogLoRetConfig(RetrieverConfig):
    """LogLoRet, which scores the first `topk_in` windows of the list before it (a list of every window, as the
    first phase)."""
    encoder:   TileEncoderConfig
    weights:   str
    method:    str
    no_match:  str
    tile_size: int = 256
    overlap:   bool = True
    topk_in:   int = 1000
    reuse:     Tuple[str, ...] = ()
    batch:     int = 64

    BASELINE = {'tile_size': 256, 'overlap': True, 'topk_in': 1000}
    NOT_IDENTITY = ('reuse', 'batch')
    VERSION = 1


def load_net(weights: str, device: Any, pooling: str, arch: str, method: str,
             no_match: str = '') -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """The network of checkpoint `weights`, and its file's record; refused when it is not `arch`, is not the
    `method` the config names (its ContextEn and GlobalEn), has another no-match option than `no_match`, or was
    trained on features of another pooling than the encoder gives."""
    from Checkpoints import build_rerank_from_checkpoint           # noqa: PLC0415
    net, ckpt = build_rerank_from_checkpoint(weights_path(weights), device)
    if ckpt['arch'] != arch:
        raise ValueError(f'{weights} is a {ckpt["arch"]!r} network, this method needs {arch!r}')
    built: str = f'{"GloRet" if arch == "gloret" else "LogLoRet"}-{net.cfg.context}-{net.cfg.global_en}'
    if built != method:
        raise ValueError(f'{weights} is {built}, the config names {method}')
    if no_match and net.cfg.no_match != no_match:
        raise ValueError(f'{weights} was trained with the {net.cfg.no_match!r} no-match option, the config names {no_match!r}')
    if ckpt['pooling'] != pooling:
        raise ValueError(f'{weights} reads {ckpt["pooling"]!r} features and the encoder gives {pooling!r}')
    return net, ckpt


def amp_context(device: Any, on: bool) -> Any:
    """bf16 autocast on a card, nothing else: the networks are trained under it."""
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if on and torch.device(device).type == 'cuda' else contextlib.nullcontext())


class WindowEncodings:
    """What the windows of a slide's level are encoded to by an Arch 1 network, kept for the next query. The
    retriever and the trainer's validation both rank through this, so they give the same ranking."""

    def __init__(self, net: torch.nn.Module, device: Any, band: int = 16, amp: bool = True) -> None:
        self.net: torch.nn.Module = net
        self.device: Any = device
        self.band: int = int(band)
        self.amp: bool = bool(amp)
        self.level: Optional[int] = None
        self.kept: Dict[Tuple[int, str, Tuple[int, int]], torch.Tensor] = {}

    def reset(self) -> None:
        """Forget the encodings (another slide)."""
        self.level, self.kept = None, {}

    @torch.no_grad()
    def encode_grid(self, grid: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        """What every window of `grid` `[R, C, F]` is encoded to: `[Ho, Wo, d_r]` fp16, or for `Align` the unit
        tokens `[Ho, Wo, N, d]`. A window sees only its own tiles, so a band of window rows is encoded from the
        rows of the grid it covers, `band` rows at a time."""
        a, b = shape
        ho, wo = int(grid.shape[0]) - a + 1, int(grid.shape[1]) - b + 1
        parts: List[torch.Tensor] = []
        cols: torch.Tensor = torch.arange(wo)
        for r0 in range(0, ho, self.band):
            rr: torch.Tensor = torch.arange(r0, min(r0 + self.band, ho))
            block, _ = gather_windows(grid, rr.repeat_interleave(wo), cols.repeat(len(rr)), a, b)
            feats: torch.Tensor = block.reshape(len(rr) * wo, a * b, -1).float()
            with amp_context(self.device, self.amp):
                out: torch.Tensor = self.net.encode(feats, shape, 'window')
            if self.net.aligned:
                out = torch.nn.functional.normalize(out.float(), dim=-1)
            parts.append(out.half().reshape(len(rr), wo, *out.shape[1:]))
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def window_maps(self, source: Any, n_regions: int, level: int) -> Dict[Tuple[int, int, str], torch.Tensor]:
        """The score of every window of the level, per `(rotation, region, lattice)`: the whole ranking. The
        encodings of the level's windows are kept for the next query."""
        net: torch.nn.Module = self.net
        if self.level != level:
            self.level, self.kept = level, {}
        maps: Dict[Tuple[int, int, str], torch.Tensor] = {}
        for rot in ROTATIONS:
            qg: torch.Tensor = source.query_grid(rot, source.pooling)
            shape: Tuple[int, int] = (int(qg.shape[0]), int(qg.shape[1]))
            with amp_context(self.device, self.amp):
                q: torch.Tensor = net.encode(qg.reshape(1, -1, qg.shape[-1]).float(), shape, 'query')[0]
            if net.aligned:
                q = torch.nn.functional.normalize(q.float(), dim=-1)
            for region in range(n_regions):
                for lattice in LATTICES:
                    grid: torch.Tensor = source.reference_grid(region, lattice, source.pooling)
                    if grid.numel() == 0 or grid.shape[0] < shape[0] or grid.shape[1] < shape[1]:
                        continue
                    slot: Tuple[int, str, Tuple[int, int]] = (region, lattice, shape)
                    if slot not in self.kept:
                        self.kept[slot] = self.encode_grid(grid, shape)
                    enc: torch.Tensor = self.kept[slot]
                    if net.aligned:      # a cosine per tile, taken one row of windows at a time
                        rows: List[torch.Tensor] = [torch.einsum('hwnd,nd->hw', enc[r:r + 64].float(), q)
                                                    for r in range(0, enc.shape[0], 64)]
                        maps[(rot, region, lattice)] = torch.cat(rows, dim=0) / (int(enc.shape[2]) * float(net.tau))
                    else:
                        maps[(rot, region, lattice)] = (enc.float() @ q.float()) / float(net.tau)
        return maps


@torch.no_grad()
def pair_scores_gloret(net: torch.nn.Module, batch: WindowBatch, size: int, device: Any, amp: bool = True
                       ) -> torch.Tensor:
    """`[K]` the Arch 1 score of each window of `batch` against the query turned to its rotation."""
    per_group: List[torch.Tensor] = []
    for g in batch.groups:
        a, b = g.shape
        q = g.q.reshape(len(g.index), a * b, -1)
        w = g.inner.reshape(len(g.index), a * b, -1)
        out: List[torch.Tensor] = []
        for s in range(0, len(g.index), size):
            with amp_context(device, amp):
                eq = net.encode(q[s:s + size].float(), g.shape, 'query')
                ew = net.encode(w[s:s + size].float(), g.shape, 'window')
            out.append(net.pair_score(eq.float(), ew.float()))
        per_group.append(torch.cat(out))
    return batch.scatter(per_group)


@torch.no_grad()
def pair_scores_logloret(net: torch.nn.Module, batch: WindowBatch, size: int, device: Any, amp: bool = True,
                         s_up: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`[K]` the Arch 2 score of each window of `batch` against the query turned to its rotation (`s_up`: the
    score the list came with, which the network may read)."""
    per_group: List[torch.Tensor] = []
    for g in batch.groups:
        a, b = g.shape
        q = g.q.reshape(len(g.index), a * b, -1)
        w = g.inner.reshape(len(g.index), a * b, -1)
        out: List[torch.Tensor] = []
        for s in range(0, len(g.index), size):
            up = None if s_up is None else s_up[g.index[s:s + size].to(s_up.device)]
            with amp_context(device, amp):
                out.append(net(q[s:s + size].float(), w[s:s + size].float(), g.shape, up)['score'].float())
        per_group.append(torch.cat(out))
    return batch.scatter(per_group)


# ══════════════════════════════════════════════════════════════════════════════
#  GloRet
# ══════════════════════════════════════════════════════════════════════════════

class RetrieverGloRet(Retriever):
    """See the module docstring."""

    UPSTREAM = (EstMppResult, RetrievalOutput)
    TABLES = ('rerank_scores',)

    def __init__(self, cfg: Union[RetrieverGloRetConfig, RerankerGloRetConfig], device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False, read_workers: int = 0,
                 feature_store: Optional[Any] = None, **_: Any) -> None:
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: Union[RetrieverGloRetConfig, RerankerGloRetConfig] = cfg
        self._encoder: Optional[TileEncoder] = None
        self.amp: bool = True
        self.net: Optional[torch.nn.Module] = None
        self.ckpt: Optional[Dict[str, Any]] = None
        #: the first phase's tools: the slide's levels and the vectors of their windows
        self.search: WindowSearch = WindowSearch(lambda: self.encoder, cfg.tile_size, cfg.overlap, self.device,
                                                 read_workers, feature_store)
        self.encodings: Optional[WindowEncodings] = None
        #: a later phase's: the features of the list's level
        self.features: RerankFeatures = RerankFeatures(lambda: self.encoder, cfg.encoder, int(cfg.tile_size),
                                                       bool(cfg.overlap), self.device,
                                                       bool(getattr(cfg, 'reuse', ())), feature_store)

    # ── what it is made of ───────────────────────────────────────────────────
    @property
    def encoder(self) -> TileEncoder:
        if self._encoder is None:
            self._encoder = self.encoders.get(self.cfg.encoder)
        return self._encoder

    def loaded(self, pooling: str) -> torch.nn.Module:
        if self.net is None:
            self.net, self.ckpt = load_net(self.cfg.weights, self.device, pooling, 'gloret', self.cfg.method)
        return self.net

    def uses(self) -> Tuple[Any, ...]:
        return self.features.uses() if isinstance(self.cfg, RerankerGloRetConfig) else ()

    def build(self, wsi: Union[str, Any], mask: Any) -> 'RetrieverGloRet':
        if isinstance(wsi, str):
            wsi = SafeSlide(wsi)
        super().build(wsi, mask)
        self.search.build(wsi, mask)
        self.features.build(wsi, mask)
        if self.encodings is not None:
            self.encodings.reset()
        return self

    def frame(self, level: int) -> CandidateSet:
        assert self.search.levels is not None
        return self.search.levels.frame(level)

    # ── the WSI interface ────────────────────────────────────────────────────
    def window_maps(self, source: Any, n_regions: int, level: int) -> Dict[Tuple[int, int, str], torch.Tensor]:
        """See `WindowEncodings.window_maps`; the encodings are kept for the next query of the slide."""
        if self.encodings is None:
            self.encodings = WindowEncodings(self.loaded(source.pooling), self.device,
                                             int(getattr(self.cfg, 'band', 16)), self.amp)
        return self.encodings.window_maps(source, n_regions, level)

    def run_wsi(self, upstream: EstMppResult, query: np.ndarray) -> RetrievalOutput:
        cfg = self.cfg
        if not isinstance(cfg, RetrieverGloRetConfig):
            raise TypeError('as the first phase GloRet needs a RetrieverGloRetConfig (the config of a later phase '
                            'has no raw_topk)')
        # TODO: search 'hnsw' and 'ivf' (SEARCHES). The window vectors are the same whatever the backend (no retraining):
        # build the index over the slide's window vectors of the rotation-shaped query (WindowEncodings keeps them) and
        # take the top raw_topk from it in place of the exact scan in `top_candidates`. To be done once the heads are trained.
        if cfg.search != 'exact':
            raise NotImplementedError(f'search {cfg.search!r}: only the exact search is made')
        made: Dict[str, Any] = {}

        def scorer(level: LevelFeatures, qf: QueryFeatures, frame: CandidateSet) -> CandidateSet:
            source: GridSource = GridSource(level, qf, self.encoder.feature_pooling)
            made['scores'] = self.window_maps(source, len(level.grids), level.level)
            return top_candidates(made['scores'], frame, cfg.raw_topk, cfg.min_sep_tiles, Candidate)

        cs, level, qf = self.search.run(int(upstream.chosen_level), query, scorer)
        return first_phase_output(cs, level, qf, self.encoder, cfg.tile_size, cfg.overlap, made.get('scores', {}))

    # ── the candidate-set interface ──────────────────────────────────────────
    def scores_of(self, batch: WindowBatch, size: int) -> torch.Tensor:
        """`[K]` the score of each window of `batch` against the query turned to its rotation."""
        return pair_scores_gloret(self.loaded(batch.pooling), batch, size, self.device, self.amp)

    def run_candidates(self, upstream: RetrievalOutput, query: np.ndarray) -> RetrievalOutput:
        cfg = self.cfg
        if not isinstance(cfg, RerankerGloRetConfig):
            raise TypeError('as a later phase GloRet needs a RerankerGloRetConfig (topk_in)')
        cs: CandidateSet = replace(upstream.candidates, candidates=upstream.candidates.candidates[:int(cfg.topk_in)])
        if not len(cs):
            return RetrievalOutput(candidates=cs, parent=upstream)
        table = incoming_table(cs)
        own: Dict[str, Extra] = {}
        source, made = self.features.source(upstream, query)
        if made:
            own['level_features'] = Extra('level_features', self.features.key, source)
        inp: ScoringInput = ScoringInput(query, cs, table, source, int(cfg.tile_size), None)
        table.put('gloret', self.scores_of(inp.batch(source.pooling), int(cfg.batch)).to(table['in'].device))
        return reranked_output(upstream, cs, table, 'gloret', own)

    def tables(self, out: RetrievalOutput, qc: Any) -> Dict[str, List[Dict[str, Any]]]:
        from stage2_retrieval.Context import score_rows                   # noqa: PLC0415
        rows: List[Dict[str, Any]] = score_rows(out, qc) if 'columns' in out.extras else []
        return {'rerank_scores': rows} if rows else {}


# ══════════════════════════════════════════════════════════════════════════════
#  LogLoRet
# ══════════════════════════════════════════════════════════════════════════════

class RerankerLogLoRet(Retriever):
    """See the module docstring."""

    UPSTREAM = (RetrievalOutput,)
    TABLES = ('rerank_scores',)

    def __init__(self, cfg: RerankerLogLoRetConfig, device: Optional[Any] = None, *,
                 encoders: Optional[EncoderPool] = None, multi_gpu: bool = False, feature_store: Optional[Any] = None,
                 **_: Any) -> None:
        super().__init__(cfg, device, encoders=encoders, multi_gpu=multi_gpu)
        self.cfg: RerankerLogLoRetConfig = cfg
        self._encoder: Optional[TileEncoder] = None
        self.amp: bool = True
        self.net: Optional[torch.nn.Module] = None
        self.features: RerankFeatures = RerankFeatures(lambda: self.encoder, cfg.encoder, int(cfg.tile_size),
                                                       bool(cfg.overlap), self.device, bool(cfg.reuse), feature_store)

    @property
    def encoder(self) -> TileEncoder:
        if self._encoder is None:
            self._encoder = self.encoders.get(self.cfg.encoder)
        return self._encoder

    def uses(self) -> Tuple[Any, ...]:
        return self.features.uses()

    def build(self, wsi: Any, mask: Any) -> 'RerankerLogLoRet':
        super().build(wsi, mask)
        self.features.build(wsi, mask)
        return self

    def scores_of(self, batch: WindowBatch, size: int, s_up: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.net is None:
            self.net, _ = load_net(self.cfg.weights, self.device, batch.pooling, 'logloret', self.cfg.method,
                                   self.cfg.no_match)
        return pair_scores_logloret(self.net, batch, size, self.device, self.amp, s_up)

    def run(self, upstream: RetrievalOutput, query: np.ndarray) -> RetrievalOutput:
        cs: CandidateSet = replace(upstream.candidates, candidates=upstream.candidates.candidates[:int(self.cfg.topk_in)])
        if not len(cs):
            return RetrievalOutput(candidates=cs, parent=upstream)
        table = incoming_table(cs)
        own: Dict[str, Extra] = {}
        source, made = self.features.source(upstream, query)
        if made:
            own['level_features'] = Extra('level_features', self.features.key, source)
        inp: ScoringInput = ScoringInput(query, cs, table, source, int(self.cfg.tile_size), None)
        table.put('logloret', self.scores_of(inp.batch(source.pooling), int(self.cfg.batch)).to(table['in'].device))
        return reranked_output(upstream, cs, table, 'logloret', own)

    def tables(self, out: RetrievalOutput, qc: Any) -> Dict[str, List[Dict[str, Any]]]:
        from stage2_retrieval.Context import score_rows                   # noqa: PLC0415
        rows: List[Dict[str, Any]] = score_rows(out, qc) if 'columns' in out.extras else []
        return {'rerank_scores': rows} if rows else {}


__all__ = ['RerankerGloRetConfig', 'RerankerLogLoRet', 'RerankerLogLoRetConfig', 'RetrieverGloRet',
           'RetrieverGloRetConfig', 'WindowEncodings', 'amp_context', 'load_net', 'pair_scores_gloret',
           'pair_scores_logloret']
