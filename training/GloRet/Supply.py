'''The FoVs GloRet and LogLoRet are trained on, and what each window is for one of them.

Nothing here is stored. A slide is opened, its tissue mask read, a draw of FoV positions made over every level and
-- one LEVEL at a time, a visit -- the level is encoded (by the first phase, the first time it is searched),
photographs are rendered, and each goes through the first phase (`RetrieverSlidingWinSimRot`), whose level
features, query features and top list are handed to the caller one FoV after the next.

    for res in fov_stream(slides, make_retriever, fov, mask_job, device, per_visit, seed, ...):   # a FovResult
        fov = fov_of(res)                                    # BEFORE asking for the next one

THE LABELS (`proposal.md` §9.1; `mark`). A candidate is a window and a rotation of the query. The FoV is searched at
the four rotations of its query; exactly one, `θ*`, is the rotation that sets the photo upright. Against it:

    pos      the window is an answer (`BenchCommon.answers`: the main-lattice window and the offset-lattice window
             nearest the centre of the photo's tile grid) and the rotation is `θ*`. In a LIST (the first phase's
             top, cut by its suppression) an answer may be missing: the windows that kept its place -- the same
             rotation, starting nearer to it than the suppression radius, which the benches count as found
             (`BenchCommon.listed_places`) -- are the positives then
    ignore   the windows around an answer: their origin within `expand` (0.5) of the FoV's window from the answer's,
             half its height up and down and half its width left and right. They share most of its tiles; neither
             a hit nor a miss. At a rotation other than `θ*`, the answer's own place and the windows of that zone
             are ignored too (`wrong_rotation_ignored`; with it off they are negatives)
    hard     every other window, at any rotation. There is no easy negative: far is not easier, near is not harder

The distance of a candidate to the truth, for the soft geometry target (off by default), is its window centre's to
the centre of the query's tile grid, in tiles.
'''
from __future__ import annotations

import gc
import threading
import time
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from SafeSlide import SafeSlide
from SlideReader import SlideReader
from TileSampler import PlanSpec
from TissueMaskConfig import MASK_RECIPES, MaskMaker
from FovSupply import FovRecipe, FovSupply
from camera import Render
from BenchCommon import Answers, Truth, answers, matching_rotation, placed_estimate, query_grid_centre
from stage2_retrieval.Context import gather_windows
from stage2_retrieval.Retriever import RetrievalOutput
from stage2_retrieval.StageInterface import CandidateSet

ROTATIONS: Tuple[int, ...] = (0, 90, 180, 270)
Shape = Tuple[int, int]
Part = Tuple[int, str]                       # (region, lattice): one grid of tiles of a level
Key = Tuple[int, str, int, int, int]         # a candidate's identity (region, lattice, row, col, rotation)
BITS: int = 20


def code_of(region: Any, lattice_is_offset: Any, row: Any, col: Any) -> Any:
    """One integer for the place of a window (not its rotation); works on ints and on arrays."""
    return (((np.asarray(region, dtype=np.int64) * 2 + np.asarray(lattice_is_offset, dtype=np.int64)) << (2 * BITS))
            | (np.asarray(row, dtype=np.int64) << BITS) | np.asarray(col, dtype=np.int64))


def key_code(key: Key) -> int:
    return int(code_of(key[0], key[1] == 'offset', key[2], key[3]))


def decode_code(code: int) -> Tuple[Tuple[int, str], int, int]:
    """`((region, lattice), row, col)` of a place `code`."""
    mask: int = (1 << BITS) - 1
    grid: int = int(code) >> (2 * BITS)
    return (grid // 2, 'offset' if grid % 2 else 'main'), (int(code) >> BITS) & mask, int(code) & mask


# ══════════════════════════════════════════════════════════════════════════════
#  where the time goes, and what is kept between epochs
# ══════════════════════════════════════════════════════════════════════════════

_TIMERS: Dict[str, float] = defaultdict(float)
_TIMERS_LOCK: threading.Lock = threading.Lock()


def tick(name: str, seconds: float) -> None:
    """Add `seconds` to the timer `name` (any thread)."""
    with _TIMERS_LOCK:
        _TIMERS[name] += float(seconds)


def take_timers() -> Dict[str, float]:
    """The timers since the last call, and zero them."""
    with _TIMERS_LOCK:
        out: Dict[str, float] = dict(_TIMERS)
        _TIMERS.clear()
    return out


class CardBudget:
    """The bytes of the card that the levels of the stores may stay on between epochs (`--gpu-levels-gb`); a level that does not
    fit is kept on the host as before."""

    def __init__(self, limit: int) -> None:
        self.limit: int = int(limit)
        self.used: int = 0

    def take(self, n: int) -> bool:
        if self.used + n > self.limit:
            return False
        self.used += n
        return True

    def give(self, n: int) -> None:
        self.used = max(self.used - n, 0)


class MemoryFeatureStore:
    """The encoded levels of ONE slide, kept on the host between epochs: the retriever's `feature_store`
    (`Context.encode_level`). The features of a level do not change from one epoch to the next (the slide, the
    encoder and the pooling are the same; only the FoVs are new), and encoding them is most of the time. `save`
    copies a level's map to the host; `load` hands it back when the level, ds, tile cut, lattices and regions are
    the same, and the retriever moves it to the device."""

    def __init__(self, budget: Optional[CardBudget] = None) -> None:
        self._maps: Dict[Tuple[Any, ...], Any] = {}
        self._card: Dict[Tuple[Any, ...], int] = {}                       # the bytes of each level that stays on the card
        self.budget: Optional[CardBudget] = budget
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def _key(regions: Sequence[Any], ds: float, level: int, tile_size: int, overlap: bool) -> Tuple[Any, ...]:
        return (int(level), float(ds), int(tile_size), bool(overlap),
                tuple((int(r.x), int(r.y), int(r.w), int(r.h)) for r in regions))

    def load(self, regions: Sequence[Any], *, ds: float, level: int, tile_size: int, overlap: bool) -> Optional[Any]:
        found: Optional[Any] = self._maps.get(self._key(regions, ds, level, tile_size, overlap))
        self.hits += found is not None
        self.misses += found is None
        return found

    def save(self, features: Any) -> None:
        key: Tuple[Any, ...] = self._key(features.regions, features.ds, features.level, features.tile_size,
                                         features.overlap)
        size: int = sum(int(m.features.numel()) * int(m.features.element_size()) for m in features.maps)
        if self.budget is not None:
            self.budget.give(self._card.pop(key, 0))
        if self.budget is not None and any(m.features.is_cuda for m in features.maps) and self.budget.take(size):
            self._maps[key] = features                    # stays where it is: no copy back to the host, no move to the card next epoch
            self._card[key] = size
        else:
            self._maps[key] = features.to('cpu')

    def card_bytes(self) -> int:
        return sum(self._card.values())

    def nbytes(self) -> int:
        return sum(int(m.features.numel()) * int(m.features.element_size())
                   for maps in self._maps.values() for m in maps.maps)


# ══════════════════════════════════════════════════════════════════════════════
#  the stream of FoVs through the first phase
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FovResult:
    """One FoV and what the first phase made of it. Valid until the next one: `source` holds this query's
    features."""
    slide: str
    dataset: str
    level: int
    index: int
    cs: CandidateSet                       # the list the candidates are (cut to `topk`), in the features' frame
    source: Any                            # the features' retriever's `level_features`: a `GridSource`
    found: Answers                         # the bench's answers
    rot: int                               # the rotation that sets the photo upright
    centre: Tuple[float, float]            # level-0 point the query's tile grid stands for
    tile_l0: float                         # a tile's side in level-0 px
    tol_px: float                          # the suppression radius, level-0 px
    queries: Dict[int, torch.Tensor]       # rotation -> the query's tile grid, copied
    own: Optional[CandidateSet] = None     # the features' retriever's own list, when another retriever made `cs`
    tiles: Any = None                      # how to read and encode the tiles of this level on demand (Lazy.TileAccess)
    scores: Any = None                     # the score of every window the first phase searched: {(rotation, region, lattice): [rows, cols]}


def fov_stream(slides: Sequence[Tuple[str, str, str]], make_retriever: Callable[[str], Any], fov: FovRecipe,
               mask_job: str, device: Any, per_visit: int, seed: int, epoch: int = 0,
               levels: Optional[Sequence[int]] = None, workers: int = 1, say: Callable[[str], None] = print,
               topk: Optional[int] = None, make_lister: Optional[Callable[[str], Any]] = None,
               own_topk: int = 1, min_ds: float = 0.0) -> Iterator[FovResult]:
    """`slides` is `(dataset, name, path)`. `min_ds`: only the levels whose downsample is at least this (the coarse
    ones, whose tiles are few: the patch tokens of `attnpool`). For each, in the order given: the slide's FoVs of each level (all of
    them in a random order per level, at most `per_visit` each), through a fresh retriever, its list cut to `topk`
    (None: the whole list). A FoV whose level the retriever could not search (it went finer: the query did not
    fit) or that has no answer is passed over, as the bench passes it.

    `make_lister`: another retriever makes the LIST (`cs`), and `make_retriever`'s only gives the features the
    network reads (its own list, cut to `own_topk`, is `FovResult.own`). A candidate is a place and a rotation
    -- region, lattice, row, col, rotation -- which the tissue mask and the tile size decide, not the encoder, so
    a list of one encoder and pooling is a list of windows of the other's features (as a reranker in a chain reads
    the list of a first phase that used other features)."""
    masks: MaskMaker = MaskMaker(MASK_RECIPES['hest'], mask_job, device)
    # the photos of a level are taken by `workers` threads at once (`FovSupply.photo` is the same picture on any
    # thread) while the first of them is searched
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        yield from _fov_stream(slides, make_retriever, fov, masks, pool, per_visit, seed, epoch, levels, say, topk,
                               make_lister, own_topk, min_ds)


def _fov_stream(slides: Sequence[Tuple[str, str, str]], make_retriever: Callable[[str], Any], fov: FovRecipe,
                masks: MaskMaker, pool: ThreadPoolExecutor, per_visit: int, seed: int, epoch: int,
                levels: Optional[Sequence[int]], say: Callable[[str], None],
                topk: Optional[int], make_lister: Optional[Callable[[str], Any]] = None,
                own_topk: int = 1, min_ds: float = 0.0) -> Iterator[FovResult]:
    """`fov_stream` once its tools are made."""
    for dataset, name, path in slides:
        say(f'== {dataset} {name}')
        t0: float = time.perf_counter()
        wsi: SafeSlide = SafeSlide(path)
        # A host that cannot hand out the mask's memory for a moment (the tissue search allocates an array of the
        # slide's size) is not a reason to end a job of hours on every node: try again after the others have let go
        # of theirs, and leave the slide out of this epoch when it still cannot.
        mask: Any = None
        for attempt in range(1, 4):
            try:
                mask = masks.mask(wsi)[0]
                break
            except MemoryError as exc:
                say(f'  the tissue mask could not be allocated ({str(exc)[:90]}) -- try {attempt} of 3')
                gc.collect()
                time.sleep(30.0)
        if mask is None:
            say('  no tissue mask -- slide skipped')
            continue
        reader: SlideReader = SlideReader(wsi)
        tick('open_and_mask', time.perf_counter() - t0)
        microscope: Render = Render(reader, fov.sensor, fov.gap, ds=1.0, seed=fov.sampler.seed)
        plan: PlanSpec = PlanSpec('ladder', fov.rungs_for(reader.level_downsamples), camera=microscope.spec)
        t0 = time.perf_counter()
        try:
            supply: FovSupply = FovSupply(microscope, plan, fov.sampler, mask)
            positions: List[Tuple[int, Any]] = [(i, s.meta) for i, s in enumerate(supply.sampler)]
        except RuntimeError as exc:
            say(f'  no FoV position at any level -- skipped ({str(exc).splitlines()[0]})')
            continue
        tick('draw_positions', time.perf_counter() - t0)
        retriever: Any = make_retriever(name)
        retriever.build(wsi, mask)
        lister: Optional[Any] = make_lister(name) if make_lister is not None else None
        if lister is not None:
            lister.build(wsi, mask)
        encoded: set = set()                                  # the levels whose encoding has been timed
        encoded_list: set = set()
        by_level: Dict[int, List[Tuple[int, Any]]] = {}
        for i, meta in positions:
            by_level.setdefault(int(meta.level), []).append((i, meta))
        order: List[int] = sorted(by_level)
        rng: np.random.Generator = np.random.default_rng([seed, epoch, zlib.crc32(name.encode())])
        rng.shuffle(order)
        for level in order:
            if levels is not None and level not in levels:
                continue
            if min_ds and float(by_level[level][0][1].ds) < min_ds:
                continue
            chosen: List[int] = rng.permutation(len(by_level[level]))[:per_visit].tolist()
            photos: Dict[int, Any] = {j: pool.submit(supply.photo, by_level[level][j][1]) for j in chosen}
            for j in chosen:
                index, meta = by_level[level][j]
                t0 = time.perf_counter()
                img, params = photos[j].result()
                tick('photo_wait', time.perf_counter() - t0)
                geom: Dict[str, float] = supply.geometry(meta)
                truth: Truth = Truth(
                    level=int(geom['level']), ds=float(meta.ds), rot=matching_rotation(int(params['rot_deg'])),
                    centre=partial(query_grid_centre, supply.camera_for(meta.ds), geom['x0'], geom['y0'],
                                   params, img.shape))
                t0 = time.perf_counter()
                try:
                    estimate = placed_estimate(truth.ds, truth.level, wsi.base_mpp)
                    out: RetrievalOutput = retriever.retrieve(estimate, img, topk=own_topk if lister is not None else topk)
                    listed: Optional[RetrievalOutput] = (lister.retrieve(estimate, img, topk=topk)
                                                         if lister is not None else None)
                except Exception as exc:                                  # noqa: BLE001
                    say(f'  [{index}] L{level}: {type(exc).__name__}: {exc} -- skipped')
                    continue
                finally:
                    tick('retrieve', time.perf_counter() - t0)
                for who, seen in ((retriever, encoded), (lister, encoded_list)):
                    for lv, (seconds, tiles) in dict(getattr(who, 'level_encodes', {}) or {}).items():
                        if lv not in seen:                    # the encoding (or the read back) is inside `retrieve`
                            seen.add(lv)
                            tick('level_encode', seconds)
                            tick('level_tiles', tiles)
                cs: CandidateSet = out.candidates
                own: Optional[CandidateSet] = None
                if listed is not None:                         # the list of the lister, in the features' frame
                    if not len(listed.candidates) or listed.candidates.level != cs.level:
                        continue
                    own, cs = cs, replace(cs, candidates=listed.candidates.candidates)
                if not len(cs) or cs.level != truth.level:
                    continue
                made_features = out.find('level_features')
                if not made_features:                          # the first phase could not search this level (no features to read)
                    say(f'  [{index}] L{level}: the first phase made no level features -- skipped')
                    continue
                src: Any = made_features[-1].value
                tiles_extra = out.find('tiles')
                scored = (listed if listed is not None else out).find('window_scores')     # every window's score, the whole ranking
                queries: Dict[int, torch.Tensor] = {
                    rot: src.query_grid(rot, src.pooling).detach().clone() for rot in ROTATIONS}
                tile: int = int(cs.grids[0].tile_size)
                rows, cols = int(queries[truth.rot].shape[0]), int(queries[truth.rot].shape[1])
                centre: Tuple[float, float] = truth.centre(tile)
                found: Optional[Answers] = answers(cs, (rows, cols), centre, truth.rot)
                if found is None:
                    continue
                tile_l0: float = float(tile) * float(cs.ds)
                yield FovResult(slide=name, dataset=dataset, level=level, index=index, cs=cs, source=src,
                                found=found, rot=int(truth.rot), centre=(float(centre[0]), float(centre[1])),
                                tile_l0=tile_l0,
                                tol_px=float(getattr(retriever.cfg, 'min_sep_tiles', 1.0)) * tile_l0, queries=queries,
                                own=own, tiles=(tiles_extra[-1].value if tiles_extra else None),
                                scores=(scored[-1].value if scored else None))


# ══════════════════════════════════════════════════════════════════════════════
#  a level, and the windows on it
# ══════════════════════════════════════════════════════════════════════════════

class Universe:
    """Every window of one shape on a level, numbered: the windows of each grid in row-major order, the grids one
    after the other. `ids` are those numbers."""

    def __init__(self, level: 'Level', shape: Shape) -> None:
        a, b = shape
        self.shape: Shape = shape
        self.parts: List[Part] = []
        self.wide: List[int] = []                            # windows per row of each part
        starts: List[int] = [0]
        for part, grid in level.grids.items():
            ho, wo = int(grid.shape[0]) - a + 1, int(grid.shape[1]) - b + 1
            if ho > 0 and wo > 0:
                self.parts.append(part)
                self.wide.append(wo)
                starts.append(starts[-1] + ho * wo)
        self.starts: np.ndarray = np.array(starts, dtype=np.int64)
        self.total: int = int(starts[-1])
        self.level: 'Level' = level

    def locate(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(part index, row, col)` of each id."""
        part: np.ndarray = np.searchsorted(self.starts, ids, side='right') - 1
        local: np.ndarray = ids - self.starts[part]
        wide: np.ndarray = np.array(self.wide, dtype=np.int64)[part]
        return part, local // wide, local % wide

    def id_of(self, part: Part, row: int, col: int) -> int:
        """The id of the window at `(row, col)` of `part`, -1 when there is none of this shape."""
        if part not in self.parts:
            return -1
        p: int = self.parts.index(part)
        wo: int = self.wide[p]
        ho: int = int(self.starts[p + 1] - self.starts[p]) // wo
        return int(self.starts[p]) + row * wo + col if 0 <= row < ho and 0 <= col < wo else -1

    def origins(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """`(origin [K, 2] level-0 px, code [K])` of each id's window."""
        part, rows, cols = self.locate(ids)
        origin: np.ndarray = np.zeros((len(ids), 2), dtype=np.float64)
        code: np.ndarray = np.zeros(len(ids), dtype=np.int64)
        for p, (region, lattice) in enumerate(self.parts):
            sel: np.ndarray = part == p
            if not sel.any():
                continue
            origin[sel, 0] = self.level.xs[(region, lattice)][cols[sel]]
            origin[sel, 1] = self.level.ys[(region, lattice)][rows[sel]]
            code[sel] = code_of(region, lattice == 'offset', rows[sel], cols[sel])
        return origin, code

    def features(self, ids: np.ndarray) -> torch.Tensor:
        """`[K, a*b, F]` the tile features of each id's window, in the order of `ids`."""
        part, rows, cols = self.locate(ids)
        return self.level.window_features([self.parts[p] for p in part], rows, cols, self.shape)


@dataclass
class Level:
    """A level of one slide: its tile grids (references into the first phase's level features, which stay valid
    after the retriever moves on) and where each row and column of each starts, level-0 px."""
    slide: str
    level: int
    grids: Dict[Part, torch.Tensor]                      # -> [R, C, F]
    xs: Dict[Part, np.ndarray]                           # x of each column
    ys: Dict[Part, np.ndarray]                           # y of each row
    tile_l0: float
    fovs: List['Fov'] = field(default_factory=list)
    _universes: Dict[Shape, Universe] = field(default_factory=dict)
    #: a level whose windows' tokens are read when they are asked for, not held (`Lazy.LazyLevel`); not a field
    lazy = False

    def universe(self, shape: Shape) -> Universe:
        if shape not in self._universes:
            self._universes[shape] = Universe(self, shape)
        return self._universes[shape]

    def window_features_at(self, table: Sequence[Part], at: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                           shape: Shape) -> torch.Tensor:
        """`window_features` for windows whose grid is `table[at[i]]` (`at` an int array, e.g. `Universe.locate`'s part index
        with `Universe.parts` as the table): the grids are told apart by numpy, no Python step per window."""
        a, b = shape
        used: np.ndarray = np.unique(at)
        if len(used) == 1:                                    # one grid: the gathered block is the answer, no second copy
            block, _ = gather_windows(self.grids[table[int(used[0])]], torch.as_tensor(rows), torch.as_tensor(cols), a, b)
            return block.reshape(len(at), a * b, -1)
        out: Optional[torch.Tensor] = None
        for k in used.tolist():
            sel: np.ndarray = np.flatnonzero(at == k)
            block, _ = gather_windows(self.grids[table[k]], torch.as_tensor(rows[sel]), torch.as_tensor(cols[sel]), a, b)
            block = block.reshape(len(sel), a * b, -1)
            if out is None:
                out = torch.empty((len(at), a * b, block.shape[-1]), dtype=block.dtype, device=block.device)
            out[torch.as_tensor(sel, device=block.device)] = block
        assert out is not None
        return out

    def window_features(self, parts: Sequence[Part], rows: np.ndarray, cols: np.ndarray, shape: Shape) -> torch.Tensor:
        """`[K, a*b, F]` the tile features of the windows of `shape` whose top-left tile is `(rows[i], cols[i])`
        of grid `parts[i]`."""
        a, b = shape
        keys: List[Part] = list(dict.fromkeys(parts))
        if len(keys) == 1:                                    # one grid: the gathered block is the answer, no second copy
            block, _ = gather_windows(self.grids[keys[0]], torch.as_tensor(rows), torch.as_tensor(cols), a, b)
            return block.reshape(len(parts), a * b, -1)
        out: Optional[torch.Tensor] = None
        for key in keys:
            sel: np.ndarray = np.flatnonzero(np.array([p == key for p in parts]))
            block, _ = gather_windows(self.grids[key], torch.as_tensor(rows[sel]), torch.as_tensor(cols[sel]), a, b)
            block = block.reshape(len(sel), a * b, -1)
            if out is None:
                out = torch.empty((len(parts), a * b, block.shape[-1]), dtype=block.dtype, device=block.device)
            out[torch.as_tensor(sel, device=block.device)] = block
        assert out is not None
        return out


def level_of(res: FovResult, pooling: str) -> Level:
    """The level the FoV `res` was searched on."""
    grids: Dict[Part, torch.Tensor] = {}
    xs: Dict[Part, np.ndarray] = {}
    ys: Dict[Part, np.ndarray] = {}
    for region, pg in enumerate(res.cs.grids):
        for lattice in ('main', 'offset'):
            g: torch.Tensor = res.source.reference_grid(region, lattice, pooling)
            if not g.numel():
                continue
            grids[(region, lattice)] = g
            xs[(region, lattice)] = np.array([pg.tile_origin_l0(lattice, 0, c)[0] for c in range(int(g.shape[1]))],
                                             dtype=np.float64)
            ys[(region, lattice)] = np.array([pg.tile_origin_l0(lattice, r, 0)[1] for r in range(int(g.shape[0]))],
                                             dtype=np.float64)
    return Level(slide=res.slide, level=res.level, grids=grids, xs=xs, ys=ys, tile_l0=res.tile_l0)


# ══════════════════════════════════════════════════════════════════════════════
#  a FoV as the training sees it
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Placed:
    """Where a FoV was placed: the rotation that sets it upright, the level-0 centre of its query's tile grid,
    its answers `(place code, level-0 origin)`, the radius within which a window stands for an answer, the zone
    around an answer whose windows are ignored -- `(half width, half height)` in level-0 px, by default the radius --
    and whether the windows of that zone at the wrong rotations are ignored too (else they are negatives)."""
    rot: int
    centre: Tuple[float, float]
    answers: List[Tuple[int, Tuple[float, float]]]
    tol_px: float
    zone: Optional[Tuple[float, float]] = None
    wrong_rotation_ignored: bool = True

    def __post_init__(self) -> None:
        if self.zone is None:
            self.zone = (self.tol_px, self.tol_px)


@dataclass
class Cands:
    """Candidates side by side: window `(row, col)` of `part` at `rot`; `origin` level-0 px; `score` what the list
    gave (0 for a candidate made here)."""
    part: List[Part]
    row: np.ndarray
    col: np.ndarray
    rot: np.ndarray
    origin: np.ndarray
    score: np.ndarray

    def __len__(self) -> int:
        return len(self.part)

    def code(self) -> np.ndarray:
        return code_of([p[0] for p in self.part], [p[1] == 'offset' for p in self.part], self.row, self.col)

    def take(self, index: np.ndarray) -> 'Cands':
        return Cands([self.part[i] for i in index], self.row[index], self.col[index], self.rot[index],
                     self.origin[index], self.score[index])

    @staticmethod
    def join(a: 'Cands', b: 'Cands') -> 'Cands':
        return Cands(a.part + b.part, np.concatenate([a.row, b.row]), np.concatenate([a.col, b.col]),
                     np.concatenate([a.rot, b.rot]), np.concatenate([a.origin, b.origin]),
                     np.concatenate([a.score, b.score]))


@dataclass
class Fov:
    """What a training step needs of one FoV, copied out of the first phase: the query's tile grids, where it was
    placed and the first phase's list (its top, in its order)."""
    slide: str
    level: int
    index: int
    queries: Dict[int, torch.Tensor]
    placed: Placed
    listing: Cands
    #: the first phase's score of every window of the level, per rotation (`teacher_scores`); None when it made none
    teacher: Optional[Dict[int, np.ndarray]] = None

    def shape(self, rot: int) -> Shape:
        q: torch.Tensor = self.queries[rot]
        return int(q.shape[0]), int(q.shape[1])

    def size(self, rot: int, tile_l0: float) -> Tuple[float, float]:
        """`(w0, h0)`, the level-0 px a window at `rot` covers."""
        a, b = self.shape(rot)
        return b * tile_l0, a * tile_l0


def teacher_scores(level: 'Level', res: FovResult) -> Optional[Dict[int, np.ndarray]]:
    """The first phase's score of EVERY window of the level (`window_scores`: the whole ranking, no suppression), as data:
    per rotation a float32 array over the windows of the shape the query has there, in the order of `Level.universe`
    (the grids one after the other, the windows of each in row-major order). None when the first phase made none."""
    if res.scores is None:
        return None
    out: Dict[int, np.ndarray] = {}
    for rot in ROTATIONS:
        shape: Shape = (int(res.queries[rot].shape[0]), int(res.queries[rot].shape[1]))
        uni: Universe = level.universe(shape)
        parts: List[np.ndarray] = []
        for p, part in enumerate(uni.parts):
            ho: int = int(uni.starts[p + 1] - uni.starts[p]) // uni.wide[p]
            m = res.scores.get((rot, part[0], part[1]))
            if m is None or tuple(m.shape) != (ho, uni.wide[p]):
                parts.append(np.full(ho * uni.wide[p], -np.inf, dtype=np.float32))
            else:
                parts.append(m.detach().float().cpu().numpy().reshape(-1))
        out[rot] = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)
    return out


def fov_of(res: FovResult, expand: float = 0.5, wrong_rotation_ignored: bool = True, level: Optional['Level'] = None,
           with_list: bool = True) -> Fov:
    """The FoV `res` for training. `listing`: the first phase's list (`res.cs`, already cut to what was asked of the
    first phase; empty when `with_list` is off), `teacher`: the first phase's score of every window of `level` when
    both are there. The windows within `expand` of the FoV's own window (half its height, half its width) of an answer
    are ignored (`mark`), at the wrong rotations too unless `wrong_rotation_ignored` is off."""
    cands = res.cs.candidates if with_list else []
    listing: Cands = Cands(
        part=[(c.region_index, c.lattice) for c in cands],
        row=np.array([c.row for c in cands], dtype=np.int64), col=np.array([c.col for c in cands], dtype=np.int64),
        rot=np.array([c.rotation for c in cands], dtype=np.int64),
        origin=np.array([res.cs.origin_l0(c) for c in cands], dtype=np.float64).reshape(-1, 2),
        score=np.array([c.score for c in cands], dtype=np.float64))
    rows, cols = int(res.queries[res.rot].shape[0]), int(res.queries[res.rot].shape[1])
    placed: Placed = Placed(rot=res.rot, centre=res.centre, tol_px=res.tol_px,
                            answers=[(key_code(a.key), (float(a.origin[0]), float(a.origin[1]))) for a in res.found],
                            zone=(expand * cols * res.tile_l0, expand * rows * res.tile_l0),
                            wrong_rotation_ignored=bool(wrong_rotation_ignored))
    return Fov(slide=res.slide, level=res.level, index=res.index, queries=res.queries, placed=placed, listing=listing,
               teacher=teacher_scores(level, res) if level is not None else None)


# ══════════════════════════════════════════════════════════════════════════════
#  the labels
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Marks:
    """Per candidate, `[K]`. A candidate that is neither `pos` nor `ignore` is a negative."""
    pos: np.ndarray
    ignore: np.ndarray
    dist: np.ndarray             # tiles, the window centre to the truth's
    right: np.ndarray            # the rotation is the upright one

    @property
    def neg(self) -> np.ndarray:
        return ~(self.pos | self.ignore)

    def take(self, index: np.ndarray) -> 'Marks':
        return Marks(self.pos[index], self.ignore[index], self.dist[index], self.right[index])


def mark(code: np.ndarray, rot: np.ndarray, origin: np.ndarray, size: np.ndarray, placed: Placed, tile_l0: float,
         listed: bool = False) -> Marks:
    """The marks of candidates (place `code`, `rot`, level-0 `origin` `[K, 2]` and `size` `[K, 2]`) against a FoV.
    `listed`: the candidates are a suppressed list, in which an answer may be missing and the windows that kept
    its place are the positives (`BenchCommon.listed_places`)."""
    right: np.ndarray = rot == placed.rot
    centres: np.ndarray = origin + size / 2.0
    dist: np.ndarray = np.hypot(centres[:, 0] - placed.centre[0], centres[:, 1] - placed.centre[1]) / tile_l0
    pos: np.ndarray = np.zeros(len(code), dtype=bool)
    around: np.ndarray = np.zeros(len(code), dtype=bool)               # inside the zone of some answer
    assert placed.zone is not None
    half_w, half_h = placed.zone
    for a_code, (ax, ay) in placed.answers:
        itself: np.ndarray = right & (code == a_code)
        if listed and not itself.any():
            pos |= right & (np.hypot(origin[:, 0] - ax, origin[:, 1] - ay) < placed.tol_px)
        else:
            pos |= itself
        around |= (np.abs(origin[:, 0] - ax) <= half_w) & (np.abs(origin[:, 1] - ay) <= half_h)
    ignore: np.ndarray = around & ~pos & (right | placed.wrong_rotation_ignored)
    return Marks(pos=pos, ignore=ignore, dist=dist, right=right)


def marks_of(cands: Cands, fov: Fov, tile_l0: float, listed: bool = False) -> Marks:
    """`mark` for a `Cands`."""
    size: np.ndarray = np.array([fov.size(int(r), tile_l0) for r in cands.rot], dtype=np.float64).reshape(-1, 2)
    return mark(cands.code(), cands.rot, cands.origin, size, fov.placed, tile_l0, listed)


__all__ = ['BITS', 'CardBudget', 'Cands', 'Fov', 'FovResult', 'Key', 'Level', 'Marks', 'MemoryFeatureStore', 'Part', 'Placed',
           'decode_code',
           'ROTATIONS', 'Shape', 'Universe', 'code_of', 'fov_of', 'fov_stream', 'key_code', 'level_of', 'mark',
           'marks_of', 'take_timers', 'tick']
