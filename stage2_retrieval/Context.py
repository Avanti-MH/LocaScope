"""The shared tools of stage 2 (not a base class: `Retriever.py` is the interface).
For the first phase: a slide's level geometry and the walk from the routed level
to finer ones (`SlideLevels`), a level's and a query's features (`encode_level`,
`encode_query`, `WindowSearch`), the best windows of a set of score maps
(`top_candidates`). For the rerankers: where tile features come from
(`RerankFeatures`), how a list of candidates becomes aligned tensors
(`WindowBatch`), the table of what came in and the order a column gives
(`ScoreTable`, `ordered`), and boxes.

    source = out.extras['level_features'].value       # a GridSource
    batch = WindowBatch.build(out.candidates, source, 'cls_avg')
    cos = batch.groups[0].cos                         # [G, qr, qc]

A source of tile features answers `has(pooling)`, `reference_grid(region,
lattice, pooling)` and `query_grid(rotation, pooling, lattice)`: a first phase's
level and query (`GridSource`), or a level held in memory for training
(`MemorySource`). Tiles read and encoded now are `OnDemandTiles`.

NOTHING HERE DECIDES A SCORE. The window's score is the retriever's; this module
only puts the tiles a score is made of side by side, in the order a `Candidate`
names them, so the sliding window's `score` is reproduced exactly by
`WindowBatch.mean_cos()` (a test defends it) and no reranker has to re-derive
where a window's tiles are.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from PatchingLib import FeaturesMap, PatchGrid, QueryPatchContainer, WsiFeaturesMap, region_grids
from ReadGeometry import ReadSpec
from TissueMask import TissueRegion
from stage2_retrieval.Retriever import Extra, RetrievalOutput, trunk_id
from stage2_retrieval.StageInterface import ROTATIONS, Candidate, CandidateSet

LATTICES: Tuple[str, str] = ('main', 'offset')


# ══════════════════════════════════════════════════════════════════════════════
#  a slide's levels: geometry, and the walk to a level that holds the query
# ══════════════════════════════════════════════════════════════════════════════

def query_array(query: Any) -> np.ndarray:
    """The query as an RGB array, whatever it came as (`QueryPatchContainer`, PIL image, array)."""
    if isinstance(query, QueryPatchContainer):
        return np.array(query.img)
    if isinstance(query, Image.Image):
        return np.array(query.convert('RGB'))
    return np.asarray(query)


def rotate_query(img: np.ndarray, rotation: int) -> np.ndarray:
    """Lossless 90-degree-step turn of an RGB image (H, W, C), counter-clockwise (`np.rot90`)."""
    if rotation not in ROTATIONS:
        raise ValueError(f'unsupported rotation {rotation}; must be 0/90/180/270')
    return img if rotation == 0 else np.rot90(img, k=rotation // 90)


def query_dims(query: Any, tile_size: int, overlap: bool) -> Dict[int, Tuple[int, int]]:
    """(rows, cols) of the query's tile grid at each rotation -- from its size alone, nothing is cut or
    encoded."""
    img: np.ndarray = query_array(query)
    height, width = int(img.shape[0]), int(img.shape[1])
    out: Dict[int, Tuple[int, int]] = {}
    for rot in ROTATIONS:
        w, h = (width, height) if rot in (0, 180) else (height, width)
        grid: PatchGrid = PatchGrid.from_size(w, h, int(tile_size), overlap=bool(overlap))
        out[rot] = (grid.grid_rows, grid.grid_cols)
    return out


class SlideLevels:
    """The geometry of a bound slide for one tile cut, per level: the tissue regions that can host a tile and
    their grids, read off the mask (no pixel is read), cached; the query's tile grid at each rotation; which
    levels can hold the query; and the walk `route` makes from the level stage 1 routed to, one finer level
    at a time, until one can be searched.

    `self.mask` is never replaced. Each level takes a `patchable` view of it, so every level starts from the
    whole segmentation -- filtering is monotone in ds, and a level that narrowed the mask in place would leave
    a later, finer level unable to see what the coarse pass dropped."""

    def __init__(self, wsi: Any, mask: Any, tile_size: int, overlap: bool) -> None:
        self.wsi: Any = wsi
        self.mask: Any = mask
        self.tile_size: int = int(tile_size)
        self.overlap: bool = bool(overlap)
        self._geometry: Dict[int, Tuple[List[TissueRegion], List[PatchGrid]]] = {}
        #: {level: why} for a level whose preparation raised; remembered so a level that failed once is not
        #: read again for every query. Failures that depend on the query (it does not fit) are not kept here.
        self.unusable: Dict[int, str] = {}

    def ds(self, level: int) -> float:
        return float(self.wsi.level_downsamples[int(level)])

    def geometry(self, level: int) -> Tuple[List[TissueRegion], List[PatchGrid]]:
        """The regions at `level` that can host a tile, and their grids."""
        cached: Optional[Tuple[List[TissueRegion], List[PatchGrid]]] = self._geometry.get(int(level))
        if cached is not None:
            return cached
        ds: float = self.ds(level)
        regions: List[TissueRegion]
        if self.mask is not None:
            regions = self.mask.patchable(self.tile_size * ds).tissue_regions
        else:
            w0, h0 = self.wsi.level_dimensions[0]
            regions = [TissueRegion(x=0, y=0, w=w0, h=h0, index=0)]
        grids: List[PatchGrid] = region_grids(regions, ds=ds, level=int(level), tile_size=self.tile_size,
                                              overlap=self.overlap)
        self._geometry[int(level)] = (regions, grids)
        return regions, grids

    def frame(self, level: int) -> CandidateSet:
        """An empty CandidateSet in `level`'s frame: its level, ds and region grids. What stored candidates
        are put back in (stage 3 run on a list read from a cache) -- geometry only, nothing is encoded."""
        return CandidateSet(candidates=(), level=int(level), ds=self.ds(level),
                            grids=tuple(self.geometry(level)[1]), irretrievable_lvl=(), alter_lvl=())

    def query_dims(self, query: Any) -> Dict[int, Tuple[int, int]]:
        return query_dims(query, self.tile_size, self.overlap)

    def fitting_rotations(self, level: int, dims: Dict[int, Tuple[int, int]]) -> Tuple[int, ...]:
        """The rotations at which some region of `level` holds at least one window: its main tile grid, or the
        offset lattice's, is as large as the query's (rows, cols) there -- the condition a window slides under.
        Empty when the level cannot be searched for this query."""
        grids: List[PatchGrid] = self.geometry(level)[1]
        fit: List[int] = []
        for rot in ROTATIONS:
            rows, cols = dims[rot]
            if rows < 1 or cols < 1:
                continue
            if any((g.grid_rows >= rows and g.grid_cols >= cols)
                   or (g.overlap_rows >= rows and g.overlap_cols >= cols) for g in grids):
                fit.append(rot)
        return tuple(fit)

    def route(self, routed: int, dims: Dict[int, Tuple[int, int]], prepare: Callable[[int], None],
              search: Callable[[int], CandidateSet]) -> CandidateSet:
        """The list of the level `routed` (stage 1's `chosen_level`) when that level can hold the query, else of
        the nearest finer level that can. `prepare(level)` makes the level ready (its features); a raise there
        marks the level unusable, remembered. `search(level)` lists its windows; an empty list moves on.

        A level that does not fit rules out every coarser one -- a region's tile grid shrinks with ds and the
        query's does not -- which is recorded without being tried. The levels found unsearchable are
        `irretrievable_lvl`; the other levels tried are `alter_lvl`, the last of them the one searched. When no
        level can be searched the set has no candidates and `level` is the routed one."""
        n_levels: int = len(self.wsi.level_downsamples)
        irretrievable: List[int] = []
        alter: List[int] = []
        found: Optional[CandidateSet] = None
        for level in range(int(routed), -1, -1):
            if level != routed:
                alter.append(level)
            why: Optional[str] = self.unusable.get(level)
            if why is None and not self.fitting_rotations(level, dims):
                why = 'no region holds the query'
                for coarser in range(level + 1, n_levels):
                    if coarser not in irretrievable:
                        irretrievable.append(coarser)
            if why is None:
                try:
                    prepare(level)
                except Exception as e:                              # noqa: BLE001
                    why = f'{type(e).__name__}: {e}'
                    self.unusable[level] = why
            if why is None:
                searched: CandidateSet = search(level)
                if len(searched):
                    found = searched
                    break
            irretrievable.append(level)
        lost: Tuple[int, ...] = tuple(sorted(set(irretrievable)))
        if found is not None:
            return replace(found, irretrievable_lvl=lost, alter_lvl=tuple(alter))
        return CandidateSet(candidates=(), level=int(routed), ds=self.ds(routed), grids=(),
                            irretrievable_lvl=lost, alter_lvl=tuple(alter))


def nearest_window(frame: CandidateSet, dims: Tuple[int, int], x_l0: float, y_l0: float, rotation: int,
                   lattices: Sequence[str] = LATTICES) -> Optional[Tuple[int, str, int, int, float]]:
    """`(region, lattice, row, col, distance)`: the window of a `dims` = (rows, cols) query at `rotation` whose
    centre is nearest the level-0 point, over the frame's regions and `lattices`, and that distance in level-0
    px. Geometry alone -- which window, not how it scored; None when no region holds a window."""
    rows_q, cols_q = int(dims[0]), int(dims[1])
    best: Optional[Tuple[float, int, str, int, int]] = None
    for ri, grid in enumerate(frame.grids):
        tile: float = float(grid.tile_size)
        for lattice in lattices:
            g_rows, g_cols = ((grid.grid_rows, grid.grid_cols) if lattice == 'main'
                              else (grid.overlap_rows, grid.overlap_cols))
            n_r, n_c = int(g_rows) - rows_q + 1, int(g_cols) - cols_q + 1
            if n_r < 1 or n_c < 1:
                continue
            xs: np.ndarray = np.array([grid.tile_origin_l0(lattice, 0, j)[0] for j in range(n_c)],
                                      dtype=np.float64) + cols_q * tile * frame.ds / 2.0
            ys: np.ndarray = np.array([grid.tile_origin_l0(lattice, i, 0)[1] for i in range(n_r)],
                                      dtype=np.float64) + rows_q * tile * frame.ds / 2.0
            j: int = int(np.abs(xs - x_l0).argmin())
            i: int = int(np.abs(ys - y_l0).argmin())
            d: float = math.hypot(xs[j] - x_l0, ys[i] - y_l0)
            if best is None or d < best[0]:
                best = (d, ri, lattice, i, j)
    if best is None:
        return None
    d, ri, lattice, i, j = best
    return ri, lattice, i, j, d


# ══════════════════════════════════════════════════════════════════════════════
#  a level's features and a query's
# ══════════════════════════════════════════════════════════════════════════════

def feature_key(encoder_cfg: Any, tile_size: int, overlap: bool) -> Tuple[Any, ...]:
    """What a level's features are, for the bridge lock: the encoder's trunk, the pooling, the tile size and
    whether the offset lattice is there. The level is the one the candidates are in (their frame)."""
    return (trunk_id(encoder_cfg), str(encoder_cfg.pooling), int(tile_size), bool(overlap))


def first_phase_output(cs: CandidateSet, level: Optional['LevelFeatures'], query: Optional['QueryFeatures'],
                       encoder: Any, tile_size: int, overlap: bool,
                       scores: Dict[Tuple[int, int, str], torch.Tensor]) -> RetrievalOutput:
    """The output of a first-phase retriever that scored every window of a level: the list, and as extras the
    features it was made from (`level_features`, a `GridSource`) and the score of every window
    (`window_scores`, the whole ranking). No extras when no level could be searched."""
    if not len(cs) or level is None or query is None:
        return RetrievalOutput(candidates=cs)
    key: Tuple[Any, ...] = feature_key(encoder.cfg, tile_size, overlap)
    return RetrievalOutput(candidates=cs, extras={
        'level_features': Extra('level_features', key, GridSource(level, query, encoder.feature_pooling)),
        'window_scores': Extra('window_scores', key, scores)})


@dataclass
class LevelFeatures:
    """Every usable tissue region of one level, encoded: the regions, their grids, the features, and per
    region the (main, offset) feature grids `[rows, cols, F]` on the device, arranged once."""
    level: int
    ds: float
    regions: List[TissueRegion]
    grids: List[PatchGrid]
    features: WsiFeaturesMap
    tensors: List[Tuple[torch.Tensor, torch.Tensor]]


def encode_level(reader: Any, levels: SlideLevels, level: int, encoder: Any, device: Any,
                 feature_store: Optional[Any] = None) -> LevelFeatures:
    """`level` of the slide `levels` describes, encoded by `encoder`. `feature_store` is anything with
    load(regions, ds=, level=, tile_size=, overlap=) -> WsiFeaturesMap | None and save(WsiFeaturesMap); it is
    asked first, and the level is read and encoded only when the answer is None. A store that holds the
    encoded levels themselves (`shared_level(level, make)`, the bench's `LevelStore`) hands back the one it
    holds, so every retriever of the same features reads one copy."""
    shared: Optional[Callable[..., LevelFeatures]] = getattr(feature_store, 'shared_level', None)
    if callable(shared):
        return shared(int(level), lambda: encode_level(reader, levels, level, encoder, device, None))
    regions, grids = levels.geometry(level)
    ds: float = levels.ds(level)
    geo: Dict[str, Any] = dict(ds=ds, level=int(level), tile_size=levels.tile_size, overlap=levels.overlap)
    features: Optional[WsiFeaturesMap] = None
    if feature_store is not None:
        features = feature_store.load(regions, **geo)
        if features is not None:
            features = features.to(device)              # read onto the host; moved to the device once, here
    if features is None:
        blocks: Any = reader.read_grid(regions, grids, ds, tile=levels.tile_size, offset=levels.overlap,
                                       level=int(level))
        features = WsiFeaturesMap.from_grid_read(blocks, regions, grids, encoder, **geo)
        if feature_store is not None:
            feature_store.save(features)
    tensors: List[Tuple[torch.Tensor, torch.Tensor]] = [(fm.main_feature_grid(), fm.overlap_feature_grid())
                                                        for fm in features]
    return LevelFeatures(level=int(level), ds=ds, regions=list(regions), grids=list(grids), features=features,
                         tensors=tensors)


@dataclass
class QueryFeatures:
    """The query cut and encoded at each rotation: its tile containers and their features."""
    containers: Dict[int, QueryPatchContainer]
    features: Dict[int, FeaturesMap]


def encode_query(query: Any, encoder: Any, tile_size: int, overlap: bool,
                 rotations: Sequence[int] = ROTATIONS) -> QueryFeatures:
    """The query turned to each rotation, cut into tiles (and the offset lattice when `overlap`), encoded."""
    img: np.ndarray = query_array(query)
    containers: Dict[int, QueryPatchContainer] = {}
    features: Dict[int, FeaturesMap] = {}
    for rot in rotations:
        qc: QueryPatchContainer = QueryPatchContainer(rotate_query(img, rot))
        qc.extract_all(tile_size=tile_size, overlap=overlap)
        containers[rot] = qc
        features[rot] = qc.to_features(encoder)
    return QueryFeatures(containers=containers, features=features)


class WindowSearch:
    """What every first-phase retriever that scores every window of a level does around its scoring: bind a
    slide, walk from the routed level to one that holds the query (`SlideLevels.route`), encode that level
    (once per slide, through `feature_store` when one is given) and the query (once per query), and hand
    both to the retriever's `scorer`. A tool a retriever holds, not a base class: the scoring is the
    retriever's own.

    `encoder` is a callable, so the encoder is loaded only when a level or a query is actually encoded."""

    def __init__(self, encoder: Callable[[], Any], tile_size: int, overlap: bool, device: Any,
                 read_workers: int = 0, feature_store: Optional[Any] = None) -> None:
        self._encoder: Callable[[], Any] = encoder
        self.tile_size: int = int(tile_size)
        self.overlap: bool = bool(overlap)
        self.device: Any = device
        self.read_workers: int = read_workers
        self.feature_store: Optional[Any] = feature_store
        self.reader: Optional[Any] = None
        self.levels: Optional[SlideLevels] = None
        self._by_level: Dict[int, LevelFeatures] = {}
        #: Seconds the last `run` spent building levels (0 when cached).
        self.last_level_seconds: float = 0.0
        #: `{level: (seconds, tiles)}` of every level encoded since `build`.
        self.level_encodes: Dict[int, Tuple[float, int]] = {}

    def build(self, wsi: Any, mask: Any) -> None:
        from SlideReader import SlideReader                           # noqa: PLC0415
        self.reader = SlideReader(wsi, workers=self.read_workers)
        self.levels = SlideLevels(wsi, mask, self.tile_size, self.overlap)
        self._by_level = {}
        self.level_encodes = {}

    def level_features(self, level: int) -> LevelFeatures:
        """`level`'s features, encoded on first use and kept for the slide."""
        if self.levels is None or self.reader is None:
            raise RuntimeError('call build(wsi, mask) first')
        if level not in self._by_level:
            import time                                               # noqa: PLC0415
            started: float = time.perf_counter()
            feats: LevelFeatures = encode_level(self.reader, self.levels, level, self._encoder(), self.device,
                                                self.feature_store)
            self.level_encodes[level] = (time.perf_counter() - started, sum(len(g) for g in feats.grids))
            self._by_level[level] = feats
        return self._by_level[level]

    def run(self, routed: int, query: Any,
            scorer: Callable[[LevelFeatures, QueryFeatures, CandidateSet], CandidateSet]
            ) -> Tuple[CandidateSet, Optional[LevelFeatures], Optional[QueryFeatures]]:
        """`(list, level features, query features)`: the list `scorer(level, query, frame)` makes at the level
        `route` settles on, and the features it was made from (None when no level could be searched)."""
        import time                                                   # noqa: PLC0415
        if self.levels is None:
            raise RuntimeError('call build(wsi, mask) first')
        dims: Dict[int, Tuple[int, int]] = self.levels.query_dims(query)
        state: Dict[str, Any] = {}
        seconds: List[float] = [0.0]

        def prepare(level: int) -> None:
            started: float = time.perf_counter()
            state['level'] = self.level_features(level)
            seconds[0] += time.perf_counter() - started

        def search(level: int) -> CandidateSet:
            if 'query' not in state:
                state['query'] = encode_query(query, self._encoder(), self.tile_size, self.overlap)
            assert self.levels is not None
            return scorer(state['level'], state['query'], self.levels.frame(level))

        cs: CandidateSet = self.levels.route(int(routed), dims, prepare, search)
        self.last_level_seconds = seconds[0]
        if not len(cs):
            return cs, None, None
        return cs, state['level'], state['query']


class GridSource:
    """A level's features and a query's, read in place: what a first-phase retriever offers as its
    `level_features` extra, and what a reranker reads its windows through. One pooling, the encoder's."""

    def __init__(self, level: LevelFeatures, query: QueryFeatures, pooling: str) -> None:
        self.level: LevelFeatures = level
        self.query: QueryFeatures = query
        self.pooling: str = pooling

    def has(self, pooling: str) -> bool:
        return pooling == self.pooling

    def _check(self, pooling: str) -> None:
        if not self.has(pooling):
            raise KeyError(f'this source holds {self.pooling!r}, not {pooling!r}')

    def reference_grid(self, region: int, lattice: str, pooling: str) -> torch.Tensor:
        self._check(pooling)
        return self.level.tensors[region][LATTICES.index(lattice)]

    def query_grid(self, rotation: int, pooling: str, lattice: str = 'main') -> torch.Tensor:
        self._check(pooling)
        fm: FeaturesMap = self.query.features[rotation]
        return fm.main_feature_grid() if lattice == 'main' else fm.overlap_feature_grid()

    def encode_tiles(self, origins_l0: torch.Tensor, pooling: str, raw: bool = False) -> torch.Tensor:
        raise RuntimeError('a GridSource holds features; a reranker that encodes tiles has its own reader')


# ══════════════════════════════════════════════════════════════════════════════
#  tiles as unit vectors
# ══════════════════════════════════════════════════════════════════════════════

def tile_vectors(encoder: Any, images: Any, pooling: str) -> torch.Tensor:
    """`[N, F]` unit vectors of `images` (a uint8 `[N, T, T, 3]` batch or a list
    of tiles), pooled by `pooling`. The encoder's own pooling is `features()`;
    any other is `pooled()` with its slots normalised one by one and the whole
    normalised again, so the cosine of two vectors is the mean of the per-slot
    cosines (`bench_stage2_retriever.concat_slots`)."""
    if pooling == encoder.feature_pooling:
        return encoder.features(images)
    slots: torch.Tensor = encoder.pooled(images, pooling)
    return F.normalize(F.normalize(slots.float(), dim=-1).flatten(1), dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  tiles read and encoded on request
# ══════════════════════════════════════════════════════════════════════════════

class OnDemandTiles:
    """Tiles of one level, named by their level-0 top-left corner, read from the
    slide and encoded now. A tile asked for twice (neighbouring windows share
    them) is encoded once: `capacity` unit vectors are kept, least recently
    used out. Raw tokens are too big to keep and are never cached."""

    def __init__(self, reader: Any, encoder: Any, *, level: int, ds: float,
                 tile_size: int, capacity: int = 40000, batch: int = 64) -> None:
        self.reader: Any = reader
        self.encoder: Any = encoder
        self.level: int = int(level)
        self.ds: float = float(ds)
        self.tile_size: int = int(tile_size)
        self.capacity: int = int(capacity)
        self.batch: int = int(batch)
        self._kept: 'OrderedDict[Tuple[int, int, str], torch.Tensor]' = OrderedDict()
        #: tiles read from the slide and encoded, over this object's life: what a
        #: bench reports as the cost of a method that re-encodes
        self.encoded: int = 0

    def read(self, origins: Sequence[Tuple[int, int]]) -> torch.Tensor:
        """uint8 `[M, T, T, 3]` in the order of `origins`."""
        out: torch.Tensor = torch.empty((len(origins), self.tile_size, self.tile_size, 3),
                                        dtype=torch.uint8)
        spec: ReadSpec = ReadSpec(self.tile_size, self.tile_size)
        images: torch.Tensor
        index: torch.Tensor
        for images, index in self.reader.read_points(
                list(origins), spec, self.ds, level=self.level, batch=self.batch):
            out[index] = images
        return out

    def vectors(self, origins: Sequence[Tuple[int, int]], pooling: str) -> torch.Tensor:
        """`[M, F]` unit vectors, the missing ones read and encoded together."""
        keys: List[Tuple[int, int, str]] = [(int(x), int(y), pooling) for x, y in origins]
        #: what this call returns, held apart from the cache: a call that asks for more tiles than
        #: `capacity` holds (a shift of many candidates) would otherwise find, once the new ones are in,
        #: its own tiles evicted -- the ones it encoded and the ones the cache had
        got: Dict[Tuple[int, int, str], torch.Tensor] = {k: self._kept[k] for k in keys if k in self._kept}
        missing: List[int] = [i for i, k in enumerate(keys) if k not in got]
        if missing:
            fresh: torch.Tensor = tile_vectors(
                self.encoder, self.read([(keys[i][0], keys[i][1]) for i in missing]), pooling)
            self.encoded += len(missing)
            for i, v in zip(missing, fresh):
                got[keys[i]] = v
                self._kept[keys[i]] = v
            while len(self._kept) > self.capacity:
                self._kept.popitem(last=False)
        for k in keys:
            if k in self._kept:
                self._kept.move_to_end(k)
        return torch.stack([got[k] for k in keys])

    def raw(self, origins: Sequence[Tuple[int, int]]) -> torch.Tensor:
        """`[M, T, D]` fp32, the model's own tokens, prefix included."""
        self.encoded += len(origins)
        return self.encoder.tokens(self.read(origins))


# ══════════════════════════════════════════════════════════════════════════════
#  a level held in memory (training)
# ══════════════════════════════════════════════════════════════════════════════

class MemorySource:
    """A level held in memory: the reference grids, the query's grids per
    rotation, one pooling. What training builds from a level it just encoded,
    and what a bench builds to run a reranker on features it keeps.

    `reference[(region, lattice)]` is `[rows, cols, F]`;
    `queries[(rotation, lattice)]` is `[rows, cols, F]`. `tiles`, when given,
    serves `encode_tiles`."""

    def __init__(self, reference: Dict[Tuple[int, str], torch.Tensor],
                 queries: Dict[Tuple[int, str], torch.Tensor], pooling: str,
                 tiles: Optional[OnDemandTiles] = None) -> None:
        self.reference: Dict[Tuple[int, str], torch.Tensor] = reference
        self.queries: Dict[Tuple[int, str], torch.Tensor] = queries
        self.pooling: str = pooling
        self.tiles: Optional[OnDemandTiles] = tiles

    def has(self, pooling: str) -> bool:
        return pooling == self.pooling

    def reference_grid(self, region: int, lattice: str, pooling: str) -> torch.Tensor:
        if not self.has(pooling):
            raise KeyError(f'this source holds {self.pooling!r}, not {pooling!r}')
        return self.reference[(region, lattice)]

    def query_grid(self, rotation: int, pooling: str, lattice: str = 'main') -> torch.Tensor:
        if not self.has(pooling):
            raise KeyError(f'this source holds {self.pooling!r}, not {pooling!r}')
        return self.queries[(rotation, lattice)]

    def encode_tiles(self, origins_l0: torch.Tensor, pooling: str, raw: bool = False) -> torch.Tensor:
        if self.tiles is None:
            raise RuntimeError('this MemorySource has no reader to encode tiles from')
        origins: List[Tuple[int, int]] = [(int(x), int(y)) for x, y in origins_l0.tolist()]
        return self.tiles.raw(origins) if raw else self.tiles.vectors(origins, pooling)


# ══════════════════════════════════════════════════════════════════════════════
#  candidates as tensors
# ══════════════════════════════════════════════════════════════════════════════

def gather_windows(grid: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
                   qr: int, qc: int, margin: int = 0
                   ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The `qr x qc` blocks of `grid` `[R, C, F]` whose top-left tiles are
    `(rows[g], cols[g])`: `[G, qr + 2*margin, qc + 2*margin, F]`. With a margin
    the block extends that many tiles past the window on every side; where the
    grid ends there is nothing, which `valid` `[G, ...]` says (the values there
    are those of the nearest tile -- read `valid`, not the values). Without one,
    `valid` is None and the window is entirely inside the grid."""
    R, C = int(grid.shape[0]), int(grid.shape[1])
    dr: torch.Tensor = torch.arange(-margin, qr + margin, device=grid.device)
    dc: torch.Tensor = torch.arange(-margin, qc + margin, device=grid.device)
    ri: torch.Tensor = rows.to(grid.device)[:, None, None] + dr[None, :, None]
    ci: torch.Tensor = cols.to(grid.device)[:, None, None] + dc[None, None, :]
    if margin == 0:
        return grid[ri, ci], None
    valid: torch.Tensor = (ri >= 0) & (ri < R) & (ci >= 0) & (ci < C)
    return grid[ri.clamp(0, R - 1), ci.clamp(0, C - 1)], valid


@dataclass
class WindowGroup:
    """The candidates of one window shape, side by side. `index` says where each
    sits in the candidate list; rotations 0 and 180 share a shape, 90 and 270
    the transposed one."""
    index: torch.Tensor                    # [G] positions in the candidate list
    shape: Tuple[int, int]                 # (qr, qc)
    q: torch.Tensor                        # [G, qr, qc, F]    the query's tiles, turned to the window's rotation
    w: torch.Tensor                        # [G, qr + 2m, qc + 2m, F]   the reference tiles around the window
    valid: Optional[torch.Tensor]          # [G, qr + 2m, qc + 2m] bool; None when m == 0
    margin: int

    @property
    def inner(self) -> torch.Tensor:
        """The reference tiles under the window itself: `[G, qr, qc, F]`."""
        m: int = self.margin
        qr, qc = self.shape
        return self.w[:, m:m + qr, m:m + qc]

    @property
    def cos(self) -> torch.Tensor:
        """`[G, qr, qc]` cosine of every query tile with the reference tile under
        it, in float32: the numbers a first-phase score is a mean of."""
        return (self.q.float() * self.inner.float()).sum(-1)


@dataclass
class WindowBatch:
    """A candidate list as tensors, grouped by window shape."""
    candidates: Tuple[Candidate, ...]
    groups: List[WindowGroup]
    pooling: str

    @classmethod
    def build(cls, cs: CandidateSet, source: Any, pooling: str, margin: int = 0,
              candidates: Optional[Sequence[Candidate]] = None) -> 'WindowBatch':
        """The windows of `candidates` (default: all of `cs`) read through
        `source`. Each window's query tiles are the query turned to that
        window's rotation, exactly as the retriever matched them."""
        cands: Tuple[Candidate, ...] = tuple(cs.candidates if candidates is None else candidates)
        qgrids: Dict[int, torch.Tensor] = {
            rot: source.query_grid(rot, pooling) for rot in sorted({c.rotation for c in cands})}
        by_shape: Dict[Tuple[int, int], List[int]] = {}
        for k, c in enumerate(cands):
            qr, qc = (int(v) for v in qgrids[c.rotation].shape[:2])
            by_shape.setdefault((qr, qc), []).append(k)
        groups: List[WindowGroup] = []
        for (qr, qc), ks in by_shape.items():
            proto: torch.Tensor = qgrids[cands[ks[0]].rotation]
            dim: int = int(proto.shape[-1])
            q: torch.Tensor = torch.empty((len(ks), qr, qc, dim), device=proto.device, dtype=proto.dtype)
            w: Optional[torch.Tensor] = None
            valid: Optional[torch.Tensor] = None
            parts: Dict[Tuple[int, int, str], List[int]] = {}
            for g, k in enumerate(ks):
                c: Candidate = cands[k]
                parts.setdefault((c.rotation, c.region_index, c.lattice), []).append(g)
            for (rot, region, lattice), gs in parts.items():
                rows: torch.Tensor = torch.tensor([cands[ks[g]].row for g in gs], dtype=torch.long)
                cols: torch.Tensor = torch.tensor([cands[ks[g]].col for g in gs], dtype=torch.long)
                block, ok = gather_windows(source.reference_grid(region, lattice, pooling),
                                           rows, cols, qr, qc, margin)
                if w is None:
                    w = torch.zeros((len(ks),) + tuple(block.shape[1:]), device=block.device,
                                    dtype=block.dtype)
                    if ok is not None:
                        valid = torch.zeros((len(ks),) + tuple(ok.shape[1:]), device=block.device,
                                            dtype=torch.bool)
                gi: torch.Tensor = torch.tensor(gs, dtype=torch.long, device=block.device)
                w[gi] = block
                q[gi.to(q.device)] = qgrids[rot].unsqueeze(0).expand(len(gs), -1, -1, -1)
                if valid is not None and ok is not None:
                    valid[gi] = ok
            assert w is not None
            groups.append(WindowGroup(index=torch.tensor(ks, dtype=torch.long), shape=(qr, qc),
                                      q=q, w=w, valid=valid, margin=margin))
        return cls(candidates=cands, groups=groups, pooling=pooling)

    def __len__(self) -> int:
        return len(self.candidates)

    def scatter(self, per_group: Sequence[torch.Tensor], fill: float = 0.0) -> torch.Tensor:
        """`[K]` from one `[G]` (or `[G, ...]`) tensor per group, put back in
        candidate order."""
        first: torch.Tensor = per_group[0]
        out: torch.Tensor = torch.full((len(self),) + tuple(first.shape[1:]), fill,
                                       dtype=first.dtype, device=first.device)
        for grp, values in zip(self.groups, per_group):
            out[grp.index.to(out.device)] = values
        return out

    def mean_cos(self) -> torch.Tensor:
        """`[K]` the plain mean of every window's tile cosines: what
        `window_score(..., 'mean')` gives, which the first phase's `score` is."""
        return self.scatter([g.cos.mean(dim=(1, 2)) for g in self.groups])


# ══════════════════════════════════════════════════════════════════════════════
#  boxes
# ══════════════════════════════════════════════════════════════════════════════

def window_boxes(cs: CandidateSet, shape_of: Callable[[int], Tuple[int, int]],
                 candidates: Optional[Sequence[Candidate]] = None) -> np.ndarray:
    """`[K, 4]` float64 level-0 boxes `(x0, y0, x1, y1)` of the windows, shift
    included. `shape_of(rotation)` is the window's `(rows, cols)` in tiles at
    that rotation (a query's grid, turned)."""
    cands: Sequence[Candidate] = cs.candidates if candidates is None else candidates
    out: np.ndarray = np.empty((len(cands), 4), dtype=np.float64)
    for k, c in enumerate(cands):
        qr, qc = shape_of(c.rotation)
        ts: int = cs.grids[c.region_index].tile_size
        x, y = cs.origin_l0(c)
        out[k] = (x, y, x + qc * ts * cs.ds, y + qr * ts * cs.ds)
    return out


def box_area(b: np.ndarray) -> np.ndarray:
    """Areas of `[..., 4]` boxes (zero for an empty one)."""
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection areas of `a` `[N, 4]` with `b` `[M, 4]`: `[N, M]`."""
    lo: np.ndarray = np.maximum(a[:, None, :2], b[None, :, :2])
    hi: np.ndarray = np.minimum(a[:, None, 2:], b[None, :, 2:])
    return np.clip(hi - lo, 0, None).prod(-1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of `a` `[N, 4]` with `b` `[M, 4]`: `[N, M]`."""
    inter: np.ndarray = intersection(a, b)
    union: np.ndarray = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  the half-tile lattice of the query
# ══════════════════════════════════════════════════════════════════════════════

def offset_pair_cosines(cs: CandidateSet, source: Any, pooling: str,
                        candidates: Optional[Sequence[Candidate]] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(sum, count)`, each `[K]`: over the query's OFFSET-lattice tiles (half a
    tile down and right of its main ones; one row and one column fewer), the
    sum of their cosines with the reference tile at the same place, and how many
    there are. The window's main tiles are matched by the retriever; these are
    the tiles in between, which it does not use, at no cost: the reference has a
    tile at every one of those places already.

    A window on the MAIN lattice at `(row, col)` has the query's offset tile
    `(i, j)` on the reference's OFFSET tile `(row + i, col + j)`. A window on the
    OFFSET lattice has it on the reference's MAIN tile `(row + i + 1, col + j + 1)`
    (its own origin is already half a tile in)."""
    cands: Tuple[Candidate, ...] = tuple(cs.candidates if candidates is None else candidates)
    parts: Dict[Tuple[int, int, str], List[int]] = {}
    for k, c in enumerate(cands):
        parts.setdefault((c.rotation, c.region_index, c.lattice), []).append(k)
    total: Optional[torch.Tensor] = None
    count: Optional[torch.Tensor] = None
    for (rot, region, lattice), ks in parts.items():
        qo: torch.Tensor = source.query_grid(rot, source.pooling if not pooling else pooling, 'offset')
        other: str = 'offset' if lattice == 'main' else 'main'
        step: int = 0 if lattice == 'main' else 1
        rows: torch.Tensor = torch.tensor([cands[k].row + step for k in ks], dtype=torch.long)
        cols: torch.Tensor = torch.tensor([cands[k].col + step for k in ks], dtype=torch.long)
        block, _ = gather_windows(source.reference_grid(region, other, pooling or source.pooling),
                                  rows, cols, int(qo.shape[0]), int(qo.shape[1]))
        sums: torch.Tensor = (qo.unsqueeze(0).float() * block.float()).sum(-1).sum(dim=(1, 2))
        if total is None:
            total = torch.zeros(len(cands), device=sums.device)
            count = torch.zeros(len(cands), device=sums.device)
        idx: torch.Tensor = torch.tensor(ks, dtype=torch.long, device=sums.device)
        total[idx] = sums
        count[idx] = float(qo.shape[0] * qo.shape[1])
    if total is None or count is None:
        empty: torch.Tensor = torch.zeros(0)
        return empty, empty
    return total, count


# ══════════════════════════════════════════════════════════════════════════════
#  the query as pixels
# ══════════════════════════════════════════════════════════════════════════════

def rotated_query_tiles(query: np.ndarray, rotation: int, tile_size: int
                        ) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """`(tiles, (rows, cols))`: the query turned by `rotation` (as the retriever
    turns it: `np.rot90`, lossless) and cut into its main tiles, row-major,
    uint8 `[rows * cols, T, T, 3]`. For a reranker that encodes the query itself
    -- raw tokens, another encoder -- the tiles are the ones the retriever
    encoded."""
    turned: np.ndarray = query if rotation == 0 else np.rot90(query, k=rotation // 90)
    qc: QueryPatchContainer = QueryPatchContainer(np.ascontiguousarray(turned))
    qc.extract_all(tile_size=tile_size, overlap=False)
    tiles: List[np.ndarray] = list(qc.iter_main())
    return torch.from_numpy(np.stack(tiles)), (int(qc.grid.grid_rows), int(qc.grid.grid_cols))


# ══════════════════════════════════════════════════════════════════════════════
#  the best windows of a set of score maps
# ══════════════════════════════════════════════════════════════════════════════

def separated(origin: np.ndarray, rotation: np.ndarray, min_sep: float, k: int) -> List[int]:
    """Indices of the candidates a best-first list keeps: the greedy pass that
    drops a candidate when one already kept AT ITS ROTATION has its window
    origin nearer than `min_sep` (level-0 px), and stops at `k`. `origin` is
    `[N, 2]` float64 level-0 origins, `rotation` `[N]`, both in rank order.

    One pass over the list with the kept origins in arrays, so a retrieve of
    k = 500 does not scan a Python list 500 long for each of its thousands of
    raw candidates. It keeps what the list scan kept (a test holds it to that)."""
    n_max: int = min(k, len(origin))
    kept_x: np.ndarray = np.empty(max(n_max, 1), dtype=np.float64)
    kept_y: np.ndarray = np.empty(max(n_max, 1), dtype=np.float64)
    kept_rot: np.ndarray = np.empty(max(n_max, 1), dtype=np.int64)
    kept: List[int] = []
    for i in range(len(origin)):
        n: int = len(kept)
        if n:
            same: np.ndarray = kept_rot[:n] == rotation[i]
            if same.any() and (np.hypot(kept_x[:n][same] - origin[i, 0],
                                        kept_y[:n][same] - origin[i, 1]) < min_sep).any():
                continue
        kept_x[n], kept_y[n], kept_rot[n] = origin[i, 0], origin[i, 1], rotation[i]
        kept.append(i)
        if len(kept) == k:
            break
    return kept



def top_candidates(maps: Dict[Tuple[int, int, str], torch.Tensor], frame: CandidateSet, k: int,
                   min_sep_tiles: float, make: Callable[..., Candidate]) -> CandidateSet:
    """The best `k` windows over score maps, as a `CandidateSet` in `frame`.

    `maps[(rotation, region, lattice)]` is `[rows, cols]`, the score of the window
    whose top-left tile is `(row, col)` on that lattice of that region at that
    rotation. The pass is the retriever's own (`SlidingWinSimRot.candidate_set`):
    the top `k` of each map, best first, and a window dropped when a kept one at
    its rotation has its origin nearer than `min_sep_tiles`. `make(region,
    lattice, row, col, rotation, score)` builds the candidate (a subclass for a
    method with something to say)."""
    raw: List[Candidate] = []
    for (rot, region, lattice), scores in maps.items():
        if scores.numel() == 0:
            continue
        flat: torch.Tensor = scores.reshape(-1)
        vals, idxs = torch.topk(flat, min(max(k, 1), flat.numel()))
        n_cols: int = int(scores.shape[1])
        for v, i in zip(vals.tolist(), idxs.tolist()):
            r, c = divmod(int(i), n_cols)
            raw.append(make(region, lattice, r, c, rot, float(v)))
    if not raw:
        return frame
    raw.sort(key=lambda c: -c.score)
    min_sep: float = float(min_sep_tiles) * float(frame.grids[0].tile_size) * float(frame.ds)
    if min_sep <= 0:
        return CandidateSet(candidates=tuple(raw[:k]), level=frame.level, ds=frame.ds, grids=frame.grids,
                            irretrievable_lvl=frame.irretrievable_lvl, alter_lvl=frame.alter_lvl)
    origin: np.ndarray = np.array([frame.origin_l0(c) for c in raw], dtype=np.float64)
    rotation: np.ndarray = np.array([c.rotation for c in raw], dtype=np.int64)
    keep: List[int] = separated(origin, rotation, min_sep, k)
    return CandidateSet(candidates=tuple(raw[i] for i in keep), level=frame.level, ds=frame.ds,
                        grids=frame.grids, irretrievable_lvl=frame.irretrievable_lvl,
                        alter_lvl=frame.alter_lvl)


# ══════════════════════════════════════════════════════════════════════════════
#  a list handed in: its table, its features, its new order
# ══════════════════════════════════════════════════════════════════════════════
#
# A reranker is a retriever whose upstream is a list. What every one of them does
# around its own scoring is here: the table of what came in, where the features
# come from, the order its column gives, and the columns it offers downstream.

class ScoreTable:
    """Named `[K]` columns, one value per candidate in the order the list came in. Always there: `in` (the
    score the candidate came in with), `first` and `first_rank` (what the first phase gave it)."""

    def __init__(self, n: int) -> None:
        self.n: int = int(n)
        self.cols: Dict[str, torch.Tensor] = {}

    def put(self, name: str, values: torch.Tensor) -> None:
        if tuple(values.shape) != (self.n,):
            raise ValueError(f'column {name!r} has shape {tuple(values.shape)}, the table has '
                             f'{self.n} candidates')
        self.cols[name] = values.detach().float()

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self.cols:
            raise KeyError(f'no column {name!r}; the table has {sorted(self.cols)}')
        return self.cols[name]

    def __contains__(self, name: str) -> bool:
        return name in self.cols

    def names(self) -> List[str]:
        return list(self.cols)

    def lists(self) -> Dict[str, List[float]]:
        """Every column as plain floats (one device-to-host copy per column)."""
        return {k: v.cpu().tolist() for k, v in self.cols.items()}


def firsts(cs: CandidateSet) -> Tuple[List[float], List[int]]:
    """`(score, rank)` the first phase gave each candidate of the list: a reranked candidate carries them; a
    first-phase one is its own first phase."""
    from stage2_retrieval.StageInterface import RerankedCandidate       # noqa: PLC0415
    score: List[float] = [float(c.first_score) if isinstance(c, RerankedCandidate) else float(c.score)
                          for c in cs.candidates]
    rank: List[int] = [int(c.first_rank) if isinstance(c, RerankedCandidate) else i + 1
                       for i, c in enumerate(cs.candidates)]
    return score, rank


def incoming_table(cs: CandidateSet) -> ScoreTable:
    """The table of a list as it came in: `in`, and the first phase's `first` and `first_rank`."""
    score, rank = firsts(cs)
    table: ScoreTable = ScoreTable(len(cs))
    table.put('in', torch.tensor([c.score for c in cs.candidates], dtype=torch.float32))
    table.put('first', torch.tensor(score, dtype=torch.float32))
    table.put('first_rank', torch.tensor(rank, dtype=torch.float32))
    return table


class ScoringInput:
    """What a reranker's scoring reads: the query, the list, the table so far, the features (`source`, None
    for a method that reads none), the tile size, and the tiles it may read and encode now (`tiles`). Views of
    the list (window tensors per pooling and margin, boxes) are made once."""

    def __init__(self, query: np.ndarray, cs: CandidateSet, table: ScoreTable, source: Optional[Any],
                 tile_size: int, tiles: Optional['OnDemandTiles'] = None) -> None:
        self.query: np.ndarray = query
        self.cs: CandidateSet = cs
        self.table: ScoreTable = table
        self.source: Optional[Any] = source
        self.tile_size: int = int(tile_size)
        self.tiles: Optional[OnDemandTiles] = tiles
        self._batches: Dict[Tuple[str, int], WindowBatch] = {}
        self._shapes: Dict[int, Tuple[int, int]] = {}
        self._boxes: Optional[np.ndarray] = None

    def pooling(self, name: str = '') -> str:
        """`name`, or the source's own pooling when `name` is empty."""
        if name:
            return name
        if self.source is None:
            raise RuntimeError('this reranker reads no features')
        return self.source.pooling

    def batch(self, pooling: str = '', margin: int = 0) -> WindowBatch:
        """The candidates as window tensors; built once per (pooling, margin)."""
        key: Tuple[str, int] = (self.pooling(pooling), int(margin))
        if key not in self._batches:
            self._batches[key] = WindowBatch.build(self.cs, self.source, key[0], margin=key[1])
        return self._batches[key]

    def shape_of(self, rotation: int) -> Tuple[int, int]:
        """`(rows, cols)` in tiles of a window at `rotation`: the source's query grid, or, with no source, the
        query's size cut as the first phase cuts it."""
        if rotation not in self._shapes:
            if self.source is None:
                height, width = int(self.query.shape[0]), int(self.query.shape[1])
                tile: int = self.tile_size
                self._shapes[rotation] = ((height // tile, width // tile) if rotation in (0, 180)
                                          else (width // tile, height // tile))
            else:
                grid: torch.Tensor = self.source.query_grid(rotation, self.pooling())
                self._shapes[rotation] = (int(grid.shape[0]), int(grid.shape[1]))
        return self._shapes[rotation]

    def boxes(self) -> np.ndarray:
        """`[K, 4]` level-0 boxes of the candidates, shift included."""
        if self._boxes is None:
            self._boxes = window_boxes(self.cs, self.shape_of)
        return self._boxes

    def centres(self) -> np.ndarray:
        b: np.ndarray = self.boxes()
        return np.stack([(b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0], axis=1)

    def encode_tiles(self, origins_l0: torch.Tensor, pooling: str, raw: bool = False) -> torch.Tensor:
        """`[M, F]` (or raw `[M, T, D]`) of the tiles at the level-0 corners `origins_l0`, read now."""
        if self.tiles is None:
            raise RuntimeError('this reranker reads no tiles of its own')
        origins: List[Tuple[int, int]] = [(int(x), int(y)) for x, y in origins_l0.tolist()]
        return self.tiles.raw(origins) if raw else self.tiles.vectors(origins, pooling)


def ordered(cs: CandidateSet, table: ScoreTable, column: str) -> CandidateSet:
    """The list ordered by `column` (descending; ties keep the incoming order), each candidate scored by it,
    remembering the first phase's score and rank, and of the kind the table's columns call for: a
    `ShiftedCandidate` when there is a shift, a `TokenMatchCandidate` when there are inliers."""
    from stage2_retrieval.StageInterface import (RerankedCandidate, ShiftedCandidate,   # noqa: PLC0415
                                                 TokenMatchCandidate)
    final: torch.Tensor = table[column]
    order: List[int] = torch.argsort(final, descending=True, stable=True).tolist()
    cols: Dict[str, List[float]] = table.lists()
    scores: List[float] = final.cpu().tolist()
    first_score, first_rank = firsts(cs)
    out: List[Candidate] = []
    for i in order:
        c: Candidate = cs.candidates[i]
        base: Tuple[int, str, int, int, int] = (c.region_index, c.lattice, c.row, c.col, c.rotation)
        common: Dict[str, Any] = dict(first_score=first_score[i], first_rank=first_rank[i])
        if 'shift_dx' in cols:
            out.append(ShiftedCandidate(*base, scores[i], **common,
                                        shift=(int(round(cols['shift_dx'][i])), int(round(cols['shift_dy'][i])))))
        elif 'n_inliers' in cols:
            out.append(TokenMatchCandidate(*base, scores[i], **common, n_inliers=int(round(cols['n_inliers'][i])),
                                           inlier_ratio=float(cols['inlier_ratio'][i]) if 'inlier_ratio' in cols
                                           else 0.0))
        else:
            out.append(RerankedCandidate(*base, scores[i], **common))
    return replace(cs, candidates=tuple(out))


@dataclass
class ColumnTable:
    """The columns a reranker wrote, by candidate: `keys[i]` is the window (`Candidate.key()`) of row `i`."""
    keys: List[Tuple[int, str, int, int, int]]
    cols: Dict[str, List[float]]

    def column(self, name: str, keys: Sequence[Tuple[int, str, int, int, int]]) -> Optional[List[float]]:
        """`name`'s value for each of `keys`, None when this table lacks the column or one of the windows."""
        if name not in self.cols:
            return None
        where: Dict[Tuple[int, str, int, int, int], int] = {k: i for i, k in enumerate(self.keys)}
        if any(k not in where for k in keys):
            return None
        values: List[float] = self.cols[name]
        return [values[where[k]] for k in keys]


def reranked_output(upstream: RetrievalOutput, cs: CandidateSet, table: ScoreTable, column: str,
                    own: Optional[Dict[str, Extra]] = None) -> RetrievalOutput:
    """The output of a reranker: the list `cs` (what it read) ordered by `column`, its columns offered
    downstream (`columns`, for a stacker and for the entry's `rerank_scores` table), anything else it made
    (`own`, e.g. the features it encoded itself) and the upstream it was made from."""
    extras: Dict[str, Extra] = dict(own or {})
    extras['columns'] = Extra('columns', (column,), ColumnTable(keys=[c.key() for c in cs.candidates],
                                                                cols=table.lists()))
    return RetrievalOutput(candidates=ordered(cs, table, column), extras=extras, parent=upstream)


def reordered(upstream: RetrievalOutput, done: RetrievalOutput, column: str) -> RetrievalOutput:
    """What a reranker that differs from the one that made `done` only in the column it orders by would hand
    back for the same `upstream`: `done`'s columns, the list ordered by `column`. The same numbers as scoring
    again, at none of the cost (a bench runs Corr's four orders off one scoring)."""
    table: ColumnTable = done.extras['columns'].value
    cs: CandidateSet = replace(upstream.candidates, candidates=upstream.candidates.candidates[:len(table.keys)])
    scores: ScoreTable = ScoreTable(len(table.keys))
    for name, values in table.cols.items():
        scores.put(name, torch.tensor(values, dtype=torch.float32))
    own: Dict[str, Extra] = {k: v for k, v in done.extras.items() if k != 'columns'}
    return reranked_output(upstream, cs, scores, column, own)


def score_rows(out: RetrievalOutput, qc: Any) -> List[Dict[str, Any]]:
    """The `rerank_scores` table of a reranker's output: every candidate it read, in the order it read them, by
    its window (`region`, `lattice`, `row`, `col`, `rotation`), and every column it wrote. Where a candidate came
    in and went out, its score and its box are the upstream's list and this phase's (`output`), joined on the
    window; `in`, `first` and `first_rank`, the columns every scoring starts from, are left out for the same
    reason."""
    if 'columns' not in out.extras or out.parent is None:
        return []
    table: ColumnTable = out.extras['columns'].value
    incoming: List[Candidate] = list(out.parent.candidates.candidates[:len(table.keys)])
    own: Dict[str, Any] = {k: v for k, v in table.cols.items() if k not in ('in', 'first', 'first_rank')}
    return [dict(region=c.region_index, lattice=c.lattice, row=c.row, col=c.col, rotation=c.rotation,
                 **{k: v[i] for k, v in own.items()})
            for i, c in enumerate(incoming)]


class RerankFeatures:
    """Where a reranker's features come from. The upstream's, when the config says it reuses them (`reuse`)
    and the upstream offers the very features it would have made (`Retriever.take`: the same trunk, pooling,
    tile and lattices; another key raises). Otherwise its own: the level the list is in, encoded through the
    same `encode_level` the first phase uses (so the numbers are the same either way, a list read back from a
    cache included), and the query. `tiles(cs)` are tiles read and encoded now, for a method that moves a
    window or reads raw tokens."""

    def __init__(self, encoder: Callable[[], Any], encoder_cfg: Any, tile_size: int, overlap: bool,
                 device: Any, reuse: bool, feature_store: Optional[Any] = None) -> None:
        self._encoder: Callable[[], Any] = encoder
        self.key: Tuple[Any, ...] = feature_key(encoder_cfg, tile_size, overlap)
        self.tile_size: int = int(tile_size)
        self.overlap: bool = bool(overlap)
        self.device: Any = device
        self.reuse: bool = bool(reuse)
        self.feature_store: Optional[Any] = feature_store
        self.reader: Optional[Any] = None
        self.levels: Optional[SlideLevels] = None
        self._by_level: Dict[int, LevelFeatures] = {}
        self._tiles: Dict[int, OnDemandTiles] = {}

    def build(self, wsi: Any, mask: Any) -> None:
        from SlideReader import SlideReader                           # noqa: PLC0415
        self.reader = SlideReader(wsi)
        self.levels = SlideLevels(wsi, mask, self.tile_size, self.overlap)
        self._by_level, self._tiles = {}, {}

    def uses(self) -> Tuple[Any, ...]:
        from stage2_retrieval.Retriever import Uses                  # noqa: PLC0415
        return (Uses('level_features', self.key),) if self.reuse else ()

    def source(self, upstream: RetrievalOutput, query: Any) -> Tuple[GridSource, bool]:
        """`(features, made here)`: the upstream's when reused, else this reranker's own."""
        from stage2_retrieval.Retriever import take                  # noqa: PLC0415
        if self.reuse:
            got: Optional[Any] = take(upstream, self.uses()[0])
            if got is not None:
                return got, False
        if self.levels is None or self.reader is None:
            raise RuntimeError('call build(wsi, mask) first')
        level: int = int(upstream.candidates.level)
        if level not in self._by_level:
            self._by_level = {level: encode_level(self.reader, self.levels, level, self._encoder(), self.device,
                                                  self.feature_store)}
        feats: LevelFeatures = self._by_level[level]
        if len(feats.grids) != len(upstream.candidates.grids):
            raise ValueError(f'the list is in {len(upstream.candidates.grids)} regions and this reranker '
                             f'cuts {len(feats.grids)}: another mask or tile')
        qf: QueryFeatures = encode_query(query, self._encoder(), self.tile_size, self.overlap)
        return GridSource(feats, qf, self._encoder().feature_pooling), True

    def tiles(self, cs: CandidateSet) -> OnDemandTiles:
        """Tiles of the list's level, read and encoded now by this reranker's encoder, kept for the level."""
        if cs.level not in self._tiles:
            if self.reader is None:
                raise RuntimeError('call build(wsi, mask) first')
            self._tiles = {cs.level: OnDemandTiles(self.reader, self._encoder(), level=cs.level, ds=cs.ds,
                                                   tile_size=self.tile_size)}
        return self._tiles[cs.level]
