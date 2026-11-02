"""Rerankers with learned weights, and the tensors they and their training (`training/RerankHead`) read a
candidate list through.

    RerankerTileGT    `RerankHeads.TileGT` (or `TileGTContext`) on every candidate
    RerankerCorrNet   `RerankHeads.CorrNet` on the correspondence matrix
    RerankerStack     `RerankHeads.StackNet` on the columns the rerankers upstream wrote

`neural_inputs` and `stack_features` are the ONE place a candidate list becomes a network's input, imported by
the training code as well: what a network was trained on and what it is run on cannot differ in how the
tiles are laid out.

`weights` is a file name relative to the results root (`RerankHead/weights/<f>.pt`), as the stage-1
checkpoints are, and is NOT identity by name: its CONTENT is in the stage's id (`BenchCommon.stage_hexid`), so a
retrained file under the same name is another entry and a renamed file with the same bytes the same one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from ConfigIdentity import register
from TileEncoderFunc import TileEncoderConfig
from stage1_estimation.StageInterface import weights_path
from stage2_retrieval.Context import ColumnTable, ScoreTable, ScoringInput, WindowBatch
from stage2_retrieval.Retriever import RetrievalOutput, RetrieverConfig
from stage2_retrieval.RerankerScorers import ColumnRanker

#: The window-level scalars a network is given (`RerankHeads.TileGTCfg.n_global`).
N_GLOBAL: int = 8
#: The per-tile scalars (`TileGTCfg.n_scalar`).
N_SCALAR: int = 1
#: What `pair_features` gives per pair of candidates (`ContextCfg.n_pair`).
N_PAIR: int = 4


@dataclass
class NeuralInputs:
    """A candidate list as the tensors `TileGT` and `CorrNet` take, in candidate
    order. `shapes` is each candidate's (rows, cols), for `CorrNet`."""
    q: torch.Tensor            # [K, n, F]
    w: torch.Tensor            # [K, n, F]
    pos: torch.Tensor          # [K, n, 2] long
    scalars: torch.Tensor      # [K, n, N_SCALAR]
    corr: torch.Tensor         # [K, n, n]
    glob: torch.Tensor         # [K, N_GLOBAL]
    shapes: List[Tuple[int, int]]

    def __len__(self) -> int:
        return int(self.q.shape[0])


def _z(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-6)


def neural_inputs(batch: WindowBatch, first_score: torch.Tensor, first_rank: torch.Tensor,
                  rotation: List[int], lattice: List[str]) -> NeuralInputs:
    """The tensors of `batch`'s windows. `first_score`, `first_rank` `[K]` are what
    the first phase gave each; they enter as the list-relative scalars a network
    may use (a z-score within the query's candidates, a log rank)."""
    k: int = len(batch)
    n: int = batch.groups[0].shape[0] * batch.groups[0].shape[1]
    ref: torch.Tensor = batch.groups[0].q
    q: torch.Tensor = torch.zeros((k, n, ref.shape[-1]), device=ref.device, dtype=ref.dtype)
    w: torch.Tensor = torch.zeros_like(q)
    pos: torch.Tensor = torch.zeros((k, n, 2), dtype=torch.long, device=ref.device)
    shapes: List[Tuple[int, int]] = [(0, 0)] * k
    for grp in batch.groups:
        rows, cols = grp.shape
        if rows * cols != n:
            raise ValueError(f'windows of {rows * cols} tiles and of {n} in one list')
        idx: torch.Tensor = grp.index.to(ref.device)
        q[idx] = grp.q.reshape(len(idx), n, -1)
        w[idx] = grp.inner.reshape(len(idx), n, -1)
        rc: torch.Tensor = torch.stack(torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij'),
                                       dim=-1).reshape(n, 2).to(ref.device)
        pos[idx] = rc.unsqueeze(0).expand(len(idx), -1, -1)
        for i in grp.index.tolist():
            shapes[i] = (rows, cols)
    qf, wf = q.float(), w.float()
    cos_ii: torch.Tensor = (qf * wf).sum(-1, keepdim=True)
    corr: torch.Tensor = torch.bmm(qf, wf.transpose(1, 2))
    rot_onehot: torch.Tensor = torch.tensor([[float(r == v) for v in (0, 90, 180, 270)] for r in rotation],
                                            device=ref.device)
    offset: torch.Tensor = torch.tensor([float(l == 'offset') for l in lattice], device=ref.device)
    glob: torch.Tensor = torch.cat([
        _z(first_score.to(ref.device))[:, None],
        (torch.log1p(first_rank.to(ref.device).float()) / 7.0)[:, None],
        offset[:, None], rot_onehot,
        _z(cos_ii.mean(dim=(1, 2)))[:, None]], dim=1)
    return NeuralInputs(q=q, w=w, pos=pos, scalars=cos_ii, corr=corr, glob=glob, shapes=shapes)


def pair_features(centres: torch.Tensor, rotation: torch.Tensor, tile_px: float) -> torch.Tensor:
    """`[K, K, N_PAIR]`: for every pair of candidates, the offset between their
    centres in tiles (clipped to +-8 and scaled to +-1), whether they share a
    rotation, and whether they are within two tiles."""
    delta: torch.Tensor = (centres[None, :, :] - centres[:, None, :]) / tile_px
    dist: torch.Tensor = delta.norm(dim=-1)
    return torch.stack([(delta[..., 0] / 8.0).clamp(-1, 1), (delta[..., 1] / 8.0).clamp(-1, 1),
                        (rotation[:, None] == rotation[None, :]).float(), (dist <= 2.0).float()], dim=-1)


def stack_features(table: ScoreTable, columns: List[str]) -> torch.Tensor:
    """`[K, 2 * len(columns)]`: each column as its rank within the list (0 to 1)
    and as a z-score. A stacker sees how a candidate compares with the others of
    its own query, never a score's absolute size."""
    k: int = table.n
    feats: List[torch.Tensor] = []
    for name in columns:
        col: torch.Tensor = table[name].float()
        rank: torch.Tensor = torch.argsort(torch.argsort(col)).float() / max(k - 1, 1)
        feats += [rank, _z(col)]
    return torch.stack(feats, dim=1)


def _load(weights: str, device: Any) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    from Checkpoints import build_rerank_from_checkpoint          # noqa: PLC0415
    return build_rerank_from_checkpoint(weights_path(weights), device)


class _Learned(ColumnRanker):
    """What the learned rerankers share: the checkpoint is loaded on first use, on the device the candidates'
    tensors are on, and must have been trained on the features of the pooling it is given."""

    def __init__(self, cfg: RetrieverConfig, device: Optional[Any] = None, **kw: Any) -> None:
        super().__init__(cfg, device, **kw)
        self._model: Optional[torch.nn.Module] = None
        self._ckpt: Optional[Dict[str, Any]] = None

    def _loaded(self, device: Any) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        if self._model is None:
            self._model, self._ckpt = _load(self.cfg.weights, device)
        assert self._ckpt is not None
        return self._model, self._ckpt

    def _check_pooling(self, ckpt: Dict[str, Any], pooling: str) -> None:
        if ckpt['pooling'] != pooling:
            raise ValueError(f'{self.cfg.weights} reads {ckpt["pooling"]!r} features and it is given '
                             f'{pooling!r}: it was trained on something else')

    def _inputs(self, inp: ScoringInput, pooling: str) -> NeuralInputs:
        cands = inp.cs.candidates
        return neural_inputs(inp.batch(pooling), inp.table['first'], inp.table['first_rank'],
                             [c.rotation for c in cands], [c.lattice for c in cands])


# ══════════════════════════════════════════════════════════════════════════════
#  TileGT
# ══════════════════════════════════════════════════════════════════════════════

TILEGT_COLUMNS: Tuple[str, ...] = ('tilegt', 'tilegt_rot')


@register('reranker_tilegt')
@dataclass(frozen=True)
class RerankerTileGTConfig(RetrieverConfig):
    """The encoder, tile cut, lattices, `topk_in` and `reuse` as `RerankerDualConfig`; `weights` a `tilegt` or
    `tilegt_ctx` checkpoint (`RerankHead/weights/...`). `shift_loc`: also move each window by the offset the
    network predicts for the true place (a `ShiftedCandidate`), instead of only scoring it. `batch` is
    memory. Ordered by `order_by`, one of `TILEGT_COLUMNS`."""
    encoder:   TileEncoderConfig
    weights:   str
    tile_size: int = 256
    overlap:   bool = True
    topk_in:   int = 1000
    shift_loc: bool = False
    order_by:  str = 'tilegt'
    batch:     int = 256
    reuse:     Tuple[str, ...] = ()

    BASELINE = {'tile_size': 256, 'overlap': True, 'topk_in': 1000, 'shift_loc': False, 'order_by': 'tilegt'}
    NOT_IDENTITY = ('weights', 'batch', 'reuse')
    VERSION = 1

    def __post_init__(self) -> None:
        if self.order_by not in TILEGT_COLUMNS:
            raise ValueError(f'order_by must be one of {TILEGT_COLUMNS}, got {self.order_by!r}')


class RerankerTileGT(_Learned):
    """The learned verifier. Columns: `tilegt` (the logit), `tilegt_rot` (the logit that the rotation is right)
    and `tilegt_dx` / `tilegt_dy` (where the network puts the true place, in tiles from the window's centre)."""

    @torch.no_grad()
    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        pooling: str = inp.pooling()
        ni: NeuralInputs = self._inputs(inp, pooling)
        model, ckpt = self._loaded(ni.q.device)
        self._check_pooling(ckpt, pooling)
        outs: Dict[str, List[torch.Tensor]] = {'logit': [], 'rot': [], 'loc': []}
        if ckpt['arch'] == 'tilegt_ctx':
            centres: torch.Tensor = torch.as_tensor(inp.centres(), dtype=torch.float32, device=ni.q.device)
            rotation: torch.Tensor = torch.tensor([c.rotation for c in inp.cs.candidates], device=ni.q.device)
            pair: torch.Tensor = pair_features(centres, rotation,
                                               float(inp.cs.grids[0].tile_size) * float(inp.cs.ds))
            out = model(ni.q, ni.w, ni.pos, ni.scalars, ni.corr, ni.glob, pair)
            for key in outs:
                outs[key].append(out[key])
        else:
            step: int = int(self.cfg.batch)
            for s in range(0, len(ni), step):
                sl = slice(s, s + step)
                out = model(ni.q[sl], ni.w[sl], ni.pos[sl], ni.scalars[sl], ni.corr[sl], ni.glob[sl])
                for key in outs:
                    outs[key].append(out[key])
        logit, rot, loc = (torch.cat(outs[k]) for k in ('logit', 'rot', 'loc'))
        cols: Dict[str, torch.Tensor] = {'tilegt': logit, 'tilegt_rot': rot,
                                         'tilegt_dx': loc[:, 0], 'tilegt_dy': loc[:, 1]}
        if self.cfg.shift_loc:
            tile: float = float(inp.cs.grids[0].tile_size)
            cands = inp.cs.candidates
            cols['shift_dx'] = torch.tensor([c.shift_px()[0] for c in cands], device=logit.device) + loc[:, 0] * tile
            cols['shift_dy'] = torch.tensor([c.shift_px()[1] for c in cands], device=logit.device) + loc[:, 1] * tile
        return cols


# ══════════════════════════════════════════════════════════════════════════════
#  CorrNet
# ══════════════════════════════════════════════════════════════════════════════

@register('reranker_corrnet')
@dataclass(frozen=True)
class RerankerCorrNetConfig(RetrieverConfig):
    """The encoder, tile cut, lattices, `topk_in` and `reuse` as `RerankerDualConfig`; `weights` a `corrnet`
    checkpoint; `batch` memory. Ordered by `corrnet`."""
    encoder:   TileEncoderConfig
    weights:   str
    tile_size: int = 256
    overlap:   bool = True
    topk_in:   int = 1000
    batch:     int = 512
    reuse:     Tuple[str, ...] = ()

    BASELINE = {'tile_size': 256, 'overlap': True, 'topk_in': 1000}
    NOT_IDENTITY = ('weights', 'batch', 'reuse')
    VERSION = 1


class RerankerCorrNet(_Learned):
    """The learned reading of the correspondence matrix. Column `corrnet` (logit)."""

    @property
    def column(self) -> str:
        return 'corrnet'

    @torch.no_grad()
    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        pooling: str = inp.pooling()
        ni: NeuralInputs = self._inputs(inp, pooling)
        model, ckpt = self._loaded(ni.q.device)
        self._check_pooling(ckpt, pooling)
        logit: torch.Tensor = torch.zeros(len(ni), device=ni.q.device)
        for shape in sorted(set(ni.shapes)):
            where: torch.Tensor = torch.tensor([i for i, s in enumerate(ni.shapes) if s == shape],
                                               device=ni.q.device)
            for s in range(0, len(where), int(self.cfg.batch)):
                part: torch.Tensor = where[s:s + int(self.cfg.batch)]
                logit[part] = model(ni.corr[part], shape[0], shape[1])
        return {'corrnet': logit}


# ══════════════════════════════════════════════════════════════════════════════
#  the stacker
# ══════════════════════════════════════════════════════════════════════════════

@register('reranker_stack')
@dataclass(frozen=True)
class RerankerStackConfig(RetrieverConfig):
    """`weights` a `stack` checkpoint, which says which columns it reads (`extra['columns']`); `topk_in` read.
    The columns are the ones the rerankers upstream wrote (their `columns` extras, nearest first) and the
    incoming table's own (`in`, `first`, `first_rank`); a chain that does not write them is refused. Ordered
    by `stack`."""
    weights: str
    topk_in: int = 1000

    BASELINE = {'topk_in': 1000}
    NOT_IDENTITY = ('weights',)
    VERSION = 1


class RerankerStack(_Learned):
    """The columns of the chain, combined by a trained network. Column `stack`."""

    READS_FEATURES = False

    def encoder_needs(self) -> Dict[str, Any]:
        return {}

    @property
    def column(self) -> str:
        return 'stack'

    def run(self, upstream: RetrievalOutput, query: Any) -> RetrievalOutput:
        self._upstream: RetrievalOutput = upstream
        return super().run(upstream, query)

    @torch.no_grad()
    def score(self, inp: ScoringInput) -> Dict[str, torch.Tensor]:
        device: Any = inp.table['in'].device
        model, ckpt = self._loaded(device)
        columns: List[str] = list(ckpt['extra']['columns'])
        keys = [c.key() for c in inp.cs.candidates]
        for extra in self._upstream.find('columns'):
            table: ColumnTable = extra.value
            for name in columns:
                if name in inp.table:
                    continue
                values: Optional[List[float]] = table.column(name, keys)
                if values is not None:
                    inp.table.put(name, torch.tensor(values, dtype=torch.float32))
        missing: List[str] = [c for c in columns if c not in inp.table]
        if missing:
            raise KeyError(f'{self.cfg.weights} reads the columns {columns}; the chain wrote no {missing} '
                           f'(it has {inp.table.names()})')
        return {'stack': model(stack_features(inp.table, columns).to(device))}


__all__ = ['N_GLOBAL', 'N_PAIR', 'N_SCALAR', 'NeuralInputs', 'RerankerCorrNet', 'RerankerCorrNetConfig',
           'RerankerStack', 'RerankerStackConfig', 'RerankerTileGT', 'RerankerTileGTConfig', 'TILEGT_COLUMNS',
           'neural_inputs', 'pair_features', 'stack_features']
