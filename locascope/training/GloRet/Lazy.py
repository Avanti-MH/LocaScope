'''Patch tokens on demand (stage B of TilePool `attnpool`): training does not need the whole fine level, only the tiles under
the windows it scores.

A tile's patch tokens are 0.6 MB, so a fine level (ds 2, ds 1: 10^5 tiles) is tens of gigabytes and does not fit
the card; the coarse levels of stage A do. A `LazyLevel` holds the level's geometry only (its grids have no features) and
reads and encodes the tiles under a window WHEN THE WINDOW IS ASKED FOR (`window_features`), so the steps of
`Full.py` and `Rerank.py` and the choosing of `Episodes.py` run on it as on any level, and nothing of the training changes:

    teacher / warm-up    the pool is chosen from the first phase's scores (numbers): no token is read for it
    the steps            the tiles under the pool's windows are read once and KEPT for the visit's steps (`tokens`)
    the network's own    ranking the items reads the tiles under every ranked window, once, and keeps NOTHING (`streaming`):
                         the items are the entries of a list (10^4), not a level's windows (5 * 10^5)
    the visit's end      `release` drops the tokens

    level = lazy_level_of(res, device)        # the level's geometry: no features; level.access reads and encodes a tile
'''
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from training.GloRet.Steps import place_of
from training.GloRet.Supply import Cands, Fov, FovResult, Level, Part, Shape, fov_of

TileKey = Tuple[Part, int, int]          # (region, lattice), row, col of a tile in its grid


@dataclass
class TileAccess:
    """How the tiles of one slide's level are read and encoded: what a `FovResult` of a source that encodes on demand
    carries. `read(points)` yields `(uint8 [B, H, W, 3], index [B])` for tiles at level-0 `points`; `encode` turns
    such a batch into `[B, P * F]` fp16."""
    read: Callable[[List[Tuple[int, int]]], Iterator[Tuple[torch.Tensor, torch.Tensor]]]
    encode: Callable[[Any], torch.Tensor]
    width: int                           # P * F


class QuerySource:
    """The part of a `GridSource` an on-demand source has: the query's tokens. Its level has no grid."""

    def __init__(self, query: Any, pooling: str) -> None:
        self.query: Any = query
        self.pooling: str = pooling

    def query_grid(self, rotation: int, pooling: str, lattice: str = 'main') -> torch.Tensor:
        fm = self.query.features[rotation]
        return fm.main_feature_grid() if lattice == 'main' else fm.overlap_feature_grid()

    def reference_grid(self, region: int, lattice: str, pooling: str) -> torch.Tensor:
        raise RuntimeError('this source encodes the tiles of windows on demand and holds no level grid')


class TileTokens:
    """The tokens of the tiles asked for, encoded once and kept on the card. `fill(keys)` reads and encodes the tiles
    not held yet; `gather(parts, rows, cols, shape)` is the `[K, a * b, P * F]` tokens of windows."""

    def __init__(self, level: Level, access: TileAccess, device: Any) -> None:
        self.level: Level = level
        self.access: TileAccess = access
        self.device: Any = device
        self.index: Dict[TileKey, int] = {}
        self.chunks: List[torch.Tensor] = []
        self._store: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.index)

    def origin(self, key: TileKey) -> Tuple[int, int]:
        part, row, col = key
        return int(self.level.xs[part][col]), int(self.level.ys[part][row])

    def fill(self, keys: Iterable[TileKey]) -> int:
        """Read and encode the tiles of `keys` that are not held; returns how many were."""
        new: List[TileKey] = [k for k in dict.fromkeys(keys) if k not in self.index]
        if not new:
            return 0
        out: torch.Tensor = torch.empty((len(new), self.access.width), dtype=torch.float16, device=self.device)
        seen: int = 0
        for batch, index in self.access.read([self.origin(k) for k in new]):
            out[index.to(self.device)] = self.access.encode(batch).to(self.device)
            seen += int(len(index))
        if seen != len(new):
            raise RuntimeError(f'{len(new)} tiles asked for, {seen} read')
        base: int = len(self.index)
        for i, k in enumerate(new):
            self.index[k] = base + i
        self.chunks.append(out)
        self._store = None
        return len(new)

    @property
    def store(self) -> torch.Tensor:
        if self._store is None:
            self._store = self.chunks[0] if len(self.chunks) == 1 else torch.cat(self.chunks)
            self.chunks = [self._store]
        return self._store

    def gather(self, parts: Sequence[Part], rows: np.ndarray, cols: np.ndarray, shape: Shape) -> torch.Tensor:
        a, b = shape
        ids: np.ndarray = np.empty((len(parts), a * b), dtype=np.int64)
        for k, part in enumerate(parts):
            r0, c0 = int(rows[k]), int(cols[k])
            for i in range(a):
                for j in range(b):
                    ids[k, i * b + j] = self.index[(part, r0 + i, c0 + j)]
        return self.store[torch.as_tensor(ids, device=self.store.device)]



def window_keys(parts: Sequence[Part], rows: np.ndarray, cols: np.ndarray, shape: Shape) -> List[TileKey]:
    """The tiles under the windows whose top-left tile is `(rows[i], cols[i])` of grid `parts[i]`."""
    a, b = shape
    return [(parts[k], int(rows[k]) + i, int(cols[k]) + j) for k in range(len(parts)) for i in range(a) for j in range(b)]


@dataclass
class LazyLevel(Level):
    """A `Level` whose grids hold no features (`grids[...]` is `[R, C, 0]`: the shape of the grid, so that the windows
    can be numbered) and whose windows are gathered from the tokens of the tiles under them, read when asked for.
    `streaming`: the tiles of a window are read for that one call and dropped (a ranking of many windows); else they are
    read once and kept in `tokens` (the steps of a visit)."""
    tokens: Optional[TileTokens] = None
    access: Optional[TileAccess] = None
    device: Any = None
    streaming: bool = False
    lazy = True

    def window_features(self, parts: Sequence[Part], rows: np.ndarray, cols: np.ndarray, shape: Shape) -> torch.Tensor:
        rows, cols = np.asarray(rows), np.asarray(cols)
        if self.streaming:
            if self.access is None:
                raise RuntimeError('no way to read a tile yet: set the level\'s access')
            once: TileTokens = TileTokens(self, self.access, self.device)
            once.fill(window_keys(parts, rows, cols, shape))
            return once.gather(parts, rows, cols, shape)             # a copy: the tiles go with `once`
        if self.tokens is None:
            if self.access is None:
                raise RuntimeError('no tokens yet: set the level\'s access, or fill them')
            self.tokens = TileTokens(self, self.access, self.device)
        self.tokens.fill(window_keys(parts, rows, cols, shape))
        return self.tokens.gather(parts, rows, cols, shape)

    def release(self) -> None:
        """Drop the tokens kept for the steps."""
        self.tokens = None


def lazy_level_of(res: FovResult, device: Any = None) -> LazyLevel:
    """The geometry of the level `res` was searched on: each region's grids by their size and the level-0 origin of
    every row and column, no features; its source's way to read and encode a tile (`res.tiles`) as `access`."""
    grids: Dict[Part, torch.Tensor] = {}
    xs: Dict[Part, np.ndarray] = {}
    ys: Dict[Part, np.ndarray] = {}
    for region, pg in enumerate(res.cs.grids):
        for lattice, rows, cols in (('main', pg.grid_rows, pg.grid_cols), ('offset', pg.overlap_rows, pg.overlap_cols)):
            if not int(rows) or not int(cols):
                continue
            grids[(region, lattice)] = torch.empty((int(rows), int(cols), 0), dtype=torch.float16)
            xs[(region, lattice)] = np.array([pg.tile_origin_l0(lattice, 0, c)[0] for c in range(int(cols))], dtype=np.float64)
            ys[(region, lattice)] = np.array([pg.tile_origin_l0(lattice, r, 0)[1] for r in range(int(rows))], dtype=np.float64)
    return LazyLevel(slide=res.slide, level=res.level, grids=grids, xs=xs, ys=ys, tile_l0=res.tile_l0,
                     access=getattr(res, 'tiles', None), device=device)


def tile_keys(cands: Cands, fov: Fov) -> List[TileKey]:
    """The tiles under the windows `cands` of `fov`."""
    keys: List[TileKey] = []
    for k in range(len(cands)):
        a, b = fov.shape(int(cands.rot[k]))
        part, r0, c0 = cands.part[k], int(cands.row[k]), int(cands.col[k])
        keys += [(part, r0 + i, c0 + j) for i in range(a) for j in range(b)]
    return keys


@torch.no_grad()
def logloret_rank_lazy(net: torch.nn.Module, res: FovResult, access: TileAccess, keep: int, batch: int = 64,
                       device: Optional[Any] = None) -> float:
    """`Steps.logloret_rank` for a source that encodes on demand: the rank of the first answer when the network
    re-orders the first `keep` windows of the list, the tokens of only those windows' tiles read and encoded."""

    from BenchCommon import listed_windows                                        # noqa: PLC0415
    from stage2_retrieval.RetrieverGloRet import amp_context                      # noqa: PLC0415
    cs = replace(res.cs, candidates=res.cs.candidates[:keep])
    if not len(cs):
        return float('inf')
    dev: Any = device if device is not None else next(net.parameters()).device
    level: LazyLevel = lazy_level_of(res)
    fov: Fov = fov_of(replace(res, cs=cs))
    tokens: TileTokens = TileTokens(level, access, dev)
    tokens.fill(tile_keys(fov.listing, fov))
    level.tokens = tokens
    scores: np.ndarray = np.zeros(len(fov.listing), dtype=np.float64)
    for rot in sorted({int(r) for r in fov.listing.rot}):
        idx: np.ndarray = np.flatnonzero(fov.listing.rot == rot)
        shape: Shape = fov.shape(rot)
        fq: torch.Tensor = fov.queries[rot].reshape(1, shape[0] * shape[1], -1).float()
        for s in range(0, len(idx), batch):
            part: np.ndarray = idx[s:s + batch]
            feats: torch.Tensor = level.window_features([fov.listing.part[j] for j in part], fov.listing.row[part],
                                                        fov.listing.col[part], shape)
            with amp_context(dev, True):
                out = net(fq, feats if feats.is_cuda else feats.float(), shape, None)
            scores[part] = out['score'].float().cpu().numpy()
    listed = listed_windows(cs)
    return place_of([listed[i] for i in np.argsort(-scores, kind='stable')], res)


@torch.no_grad()
def gloret_rerank_rank_lazy(net: torch.nn.Module, res: FovResult, access: TileAccess, keep: int, batch: int = 64,
                            device: Optional[Any] = None) -> float:
    """`Steps.gloret_rerank_rank` for a source that encodes on demand: a bi-encoder re-orders the first `keep` windows of
    the list by the score of the query against each window's vector, the tiles of only those windows read and encoded."""
    from BenchCommon import listed_windows                                        # noqa: PLC0415
    from stage2_retrieval.RetrieverGloRet import amp_context                      # noqa: PLC0415
    cs = replace(res.cs, candidates=res.cs.candidates[:keep])
    if not len(cs):
        return float('inf')
    dev: Any = device if device is not None else next(net.parameters()).device
    level: LazyLevel = lazy_level_of(res)
    fov: Fov = fov_of(replace(res, cs=cs))
    tokens: TileTokens = TileTokens(level, access, dev)
    tokens.fill(tile_keys(fov.listing, fov))
    level.tokens = tokens
    scores: np.ndarray = np.zeros(len(fov.listing), dtype=np.float64)
    for rot in sorted({int(r) for r in fov.listing.rot}):
        idx: np.ndarray = np.flatnonzero(fov.listing.rot == rot)
        shape: Shape = fov.shape(rot)
        fq: torch.Tensor = fov.queries[rot].reshape(1, shape[0] * shape[1], -1).float()
        for s in range(0, len(idx), batch):
            part: np.ndarray = idx[s:s + batch]
            feats: torch.Tensor = level.window_features([fov.listing.part[j] for j in part], fov.listing.row[part],
                                                        fov.listing.col[part], shape)
            with amp_context(dev, True):
                eq = net.encode(fq.expand(len(part), -1, -1), shape, 'query')
                ew = net.encode(feats if feats.is_cuda else feats.float(), shape, 'window')
            scores[part] = net.pair_score(eq.float(), ew.float()).float().cpu().numpy()
    listed = listed_windows(cs)
    return place_of([listed[i] for i in np.argsort(-scores, kind='stable')], res)


__all__ = ['LazyLevel', 'QuerySource', 'TileAccess', 'TileKey', 'TileTokens', 'gloret_rerank_rank_lazy', 'lazy_level_of',
           'logloret_rank_lazy', 'tile_keys', 'window_keys']
