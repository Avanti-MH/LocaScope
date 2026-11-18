'''The batches of a CoAE run (spec.md §3): places of the training slides' reference (scan) tissue, each as its versions.


    supply = Supply(cfg, rank, world, device)
    for batch in supply.batches(epoch):                 # a Batch a step of the epoch, made by the worker processes ahead of the training
        ...

THE SLIDES: the `stain` slides of `cfg.data.dataset` (`@pure-train`: ki67_pure's train split); a slide registered in a dataset of
`cfg.data.forbidden` (ki67_with_photo, whose photos validate) is refused at start.

THE PLACES are listed once, at start: on each slide and each rung ds 1, 2, 4 .. `max_ds`, every place the TileSampler recipe
`cfg.data.sampler` takes on the DISJOINT lattice of footprints inside the tissue of the mask recipe `cfg.data.mask` (any recipe: the mask
only says where a place may be). A footprint is the tile and the largest shift on both sides (`footprint_px`: 384 px at the rung's ds),
so a version shifted by up to `shift_px` never reaches a neighbour's place, and the sampler scores its background over that footprint.
The draws are TileSampler's, cached in `cfg.data.draw_cache` (a hit opens no slide). THE POOL is what `pool_runs` and `screened_pool`
keep of a (slide, rung) cell's draw, SCREENED AT START by the worker processes: the cell's lattice in Z order is cut into at most
`cfg.data.pool_cap` runs of neighbours (0 = a run a place), spread over the cell's tissue, fixed by (`pool_seed`, slide, rung); of each
run the first place whose reference square lies on the slide and whose centre tile is not blank (grey sd >= `blank_sd`) is kept (the
first candidate is the one an unscreened pool would hold, the rest stand in for it); a cell keeps as many as pass, none is an error.
The screen reads the centre tile with `SCREEN_PAD` px around it, not the whole square. Training lists places of the pool only (`seen`,
`last_seen` still run over the whole draw).

THE CELLS are the (slide, rung) pairs that have places in the pool. AN EPOCH lists at most `budget = min(pool, cfg.data.cell_epoch,
ceil(pool / cfg.data.cell_parts))` of each cell, the least seen so far first (seen = among a step's places, counted across epochs; ties
by the longest unseen, then at random), in a fresh random order (the epoch's seed, the same on every process); a cell with fewer lists
them all. A step lists `places`, all read, and the epoch has sum(budget) // that steps; a cell's budget is spread evenly over them (its
share a step: budget / steps, rounded by a running credit, floor + largest remainder, ties broken by the step's seed), so every step
has the same mix. A step's places of one slide do not overlap (`Place.meets`): the coarse rungs list first, and a place meeting one
already listed goes back to the end of its cell's queue (not used up). A step that cannot be filled ends the epoch; the next epoch picks
anew. Process r takes every world-th of a step's places. So within an epoch a place is used at most once. No two places of a step
overlap: KoLeo over the gathered batch never pushes one tissue from itself.

THE RUNGS are not evened out by the sampling (ds 32 has fewer places than a step holds), and a rung's few places are not repeated to
make up a share. Each place carries a loss weight instead (`Batch.weight`, `Supply.weights_of`): in each process's step it goes as
1 / sqrt(the places of its rung that process read), mean 1 over its places, so a rung's share of the reconstruction loss goes as
sqrt(its places) (`cfg.data.level_weights`; else 1); a rung absent from the step takes nothing.

A PLACE'S VIEWS: `slots` groups (slot 0 canonical: the scan's own pixels; slot 1, unless `cfg.data.level` is '0', perturbed at that
level), a view a source (`ref` and one a simulator of `cfg.data.sims`, photometric only) in each group: V = slots x sources, slot
outside, source inside (`Batch`). A perturbed group is twins (one perturbation drawn for all its sources) under `dep` and `mix`; under
`ind` each source draws its own. A view is a framed canvas (a group's geometry: its FRAME, `plans_of`) and, for a simulated source,
a drawn `Shot`; the worker only draws them (no pixel is made here), the framing and the simulation are made on the GPU
(`train.views_of`: `GpuSims.frames`, `GpuSims.simulate`). Each view is a photo canvas (`photo_px` square, larger than the tile so
that the Co-figure's field and its widened cells have data); the network's input is its centre tile.

THE PROBE (`Supply.probe_set`): the places of a Supply over slides that are never trained on, each as fixed views (`probe_plans`), made
once with a fixed seed on the GPU; the same set every time.

THE BOARDS (`Board`): a Supply holds BOARDS boards, each a uint8 array in SHARED MEMORY (/dev/shm, POSIX `shared_memory`): the
reference reads of a step's share (n x read^2 x 3 bytes: 428 MB at the default config); the framing and the photometric simulation
are made on the GPU (train.views_of, GpuSims). A worker reads the slide itself, attaches by name, writes its place's read in place
and returns only the verdict, the frames' geometries and each view's plan (geometry, source index, frame, shot), so no canvas goes
through a pipe. `land()` copies a batch out, so its board is free for the next flight; two flights are in
the air at once (`BOARDS`). The segments are unlinked when the Supply is closed or collected (`Supply.close`, a `weakref.finalize`, so
at a normal exit and after an exception too); a process killed before that leaves them to the resource tracker, which the spawned
workers share with their parent and which unlinks what is still registered when the parent dies (it warns of the leak). A segment lives
only as long as its process.
'''
from __future__ import annotations

import math
import os
import queue
import random
import threading
import time
import weakref
import zlib
import multiprocessing as mp
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing import shared_memory
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from locascope.core.slides.AccessDatasets import by_source, list_names, locate
from locascope.core.slides.ReadGeometry import ReadSpec
from locascope.core.slides.SafeSlide import SafeSlide
from locascope.core.slides.SlideReader import SlideReader
from locascope.core.tiles.TileSampler import SAMPLER_RECIPES, PlanSpec, SamplerConfig, TileSampler
from locascope.core.tiles.TissueMaskConfig import MASK_RECIPES, MaskMaker
from locascope.training.CoAE.CoFigure import field_maps, n_moments, place_moments
from locascope.training.CoAE.Config import LEVELS, CoaeConfig, DataConfig, Level, ProbeConfig
from locascope.training.CoAE.GpuSims import frames, simulate
from locascope.training.CoAE.Sims import (GREY_WEIGHTS, Geometry, LabGap, Shot, centre, draw_geometry, draw_shot, grey_sd, probe_labs,
                                          read_px, sim_labs)

SOURCES_REF: str = 'ref'
#: every lattice place the sampler's richness gate lets through, not a quota of them: the count TileSampler stops at
ALL_PLACES: int = 10 ** 8
#: the probe view that is the gallery: the ref at level 0, the scan's own pixels
PROBE_ANCHOR: str = 'anchor'


@dataclass
class Probe:
    """The probe: P places, each as K fixed views (`probe_plans`)."""
    tiles: torch.Tensor        # uint8 [P, K, tile, tile, 3]: each view's input (its canvas's centre tile)
    kinds: List[str]           # the K views' names, kinds[0] == PROBE_ANCHOR
    cell: torch.Tensor         # int64 [P]: a place's (slide, rung) cell; a query's wrong places are the other anchors of its cell
    cells: List[str]           # the cells' names, f'{slide}|ds{rung:g}'
    direct: torch.Tensor       # float32 [P, K, M, c, c]: each view's own place descriptor (CoFigure.place_moments of its canvas)
    maps: torch.Tensor         # float16 [P, K, N, t, t]: each view's OWN Co-figure (CoFigure.field_maps of its canvas, after the
                               # simulation); maps[:, 0] is the anchor's


# ── the slide reads and the views' draws in worker PROCESSES (the reads and the draws hold the GIL; threads could not run them side by side) ─
_W: Dict[str, Any] = {}

#: px round the centre tile in the screen's read: beyond the reach of the reader's resampling filter (Lanczos-3 at a 4x reduction
#: reaches 3 output px), so the tile screened is the tile of the full reference read
SCREEN_PAD: int = 16
#: the flights of a Supply in the air at once = its sets of boards
BOARDS: int = 2
USABLE: int = 0           # a screen's verdicts: the place reads and its centre tile is not blank
BLANK: int = 1            # its centre tile is blank (grey sd < blank_sd)
OFF: int = 2              # its reference square runs off the slide


@dataclass(frozen=True)
class Spot:
    """What a worker needs to read a place: the slide's path, the level-0 top-left of its reference square, its ds and level."""
    path: str
    x0: int
    y0: int
    ds: float
    level: int


#: a view as a worker reports it: (geometry, source index, its frame's index within the place, the shot; None = the ref: the frame itself)
ViewPlan = Tuple[Geometry, int, int, Optional[Shot]]


@dataclass
class Made:
    """A worker's report of a place: its verdict (USABLE: the reference read is on the board), the place's frames (its distinct
    geometries, in the order they are drawn), each view's plan (`ViewPlan`), the pid, the wall and CPU seconds of the whole job, and
    of them the wall seconds of the slide read."""
    code: int
    frames: List[Geometry]
    views: List[ViewPlan]
    pid: int
    wall: float
    cpu: float
    read: float


def _init_worker(data: DataConfig, read: int, tile: int, sources: Tuple[str, ...], labs: Dict[str, LabGap]) -> None:
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import cv2
    cv2.setNumThreads(1)
    _W.update(data=data, read=read, tile=tile, sources=sources, labs=labs, boards={}, readers={})


def _reader(path: str) -> SlideReader:
    """This worker's reader of a slide, opened at its first use and kept."""
    if path not in _W['readers']:
        _W['readers'][path] = SlideReader(SafeSlide(path))
    return _W['readers'][path]


def read_spot(reader: SlideReader, s: Spot, read: int) -> Optional[np.ndarray]:
    """The reference square of a place (`read` px at its ds), or None when it runs off the slide."""
    return reader.read(s.x0, s.y0, ReadSpec(read, read), s.ds, level=s.level)


def screen_spot(reader: SlideReader, s: Spot, read: int, tile: int, blank_sd: float) -> int:
    """A place's verdict without reading its whole square: OFF when the square does not fit the slide (no IO); else its centre tile
    with SCREEN_PAD px around it is read (the same lattice of the level: the offset is a whole number of output px) and BLANK when
    the tile's grey sd is below `blank_sd`."""
    if not reader.fits(reader.plan(s.x0, s.y0, ReadSpec(read, read), s.ds, level=s.level).rect):
        return OFF
    side: int = tile + 2 * SCREEN_PAD
    off: int = int(round((read - side) // 2 * s.ds))                # level-0 px; (read - side) is even (Supply checks)
    img: Optional[np.ndarray] = reader.read(s.x0 + off, s.y0 + off, ReadSpec(side, side), s.ds, level=s.level)
    if img is None:
        raise RuntimeError(f'{s.path} ds {s.ds:g} ({s.x0}, {s.y0}): the centre of a square that fits the slide does not')
    return BLANK if grey_sd(centre(img, tile)) < blank_sd else USABLE


def _worker_screen(s: Spot) -> int:
    return screen_spot(_reader(s.path), s, _W['read'], _W['tile'], _W['data'].blank_sd)


def groups_of(data: DataConfig, n_sources: int) -> int:
    """The frames of a place (its distinct geometries): slot 0's one (the scan's own pixels); slot 1's one for its twins, or one a
    source (`ind`)."""
    return 1 if data.slots == 1 else 1 + (1 if data.twins else n_sources)


def plans_of(seed: int, data: DataConfig, sources: Tuple[str, ...], labs: Dict[str, LabGap]) -> Tuple[List[Geometry], List[ViewPlan]]:
    """A place's frames (its distinct geometries, in the order they are drawn) and its views, slot by slot, a source each in
    `sources`' order (V = slots x sources): no pixel is made. Slot 0 is canonical (no perturbation); slot 1 is perturbed at
    `data.level`, one draw for all its sources (twins) unless `data.twins` is False (`ind`: one draw a source). The draws are off
    one `random.Random(seed)`: a group's geometry, then a source at a time (its own geometry under `ind`, then its shot)."""
    rng: random.Random = random.Random(seed)
    geos: List[Geometry] = []
    views: List[ViewPlan] = []
    for slot in range(data.slots):
        level: Level = LEVELS['0'] if slot == 0 else LEVELS[data.level]
        geo: Geometry = draw_geometry(rng, level)
        geos.append(geo)                                                           # one framing for the whole group
        for k, src in enumerate(sources):
            if slot > 0 and not data.twins and k > 0:                              # ind: every source its own perturbation
                geo = draw_geometry(rng, level)
                geos.append(geo)
            views.append((geo, k, len(geos) - 1, None if src == SOURCES_REF else draw_shot(labs[src], rng)))
    return geos, views


BoardRef = Tuple[str, Tuple[int, ...]]


@dataclass
class Board:
    """A uint8 array a Supply's worker processes write in place (shared memory), or a plain array (tests)."""
    arr: np.ndarray
    shm: Optional[shared_memory.SharedMemory] = None      # None: a plain array of this process

    @property
    def ref(self) -> BoardRef:
        """What a worker needs to attach: the segment's name and the array's shape."""
        if self.shm is None:
            raise ValueError('a plain board has no shared segment to attach')
        return (self.shm.name, tuple(self.arr.shape))


def make_board(shape: Tuple[int, ...], shared: bool) -> Board:
    """A board of `shape`: a /dev/shm segment when `shared` (the creator must `release_board` it), else a plain array."""
    if not shared:
        return Board(np.empty(shape, dtype=np.uint8))
    shm: shared_memory.SharedMemory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
    return Board(np.ndarray(shape, dtype=np.uint8, buffer=shm.buf), shm)


def release_board(b: Board) -> None:
    """Unlink a shared board's segment and close it (the creator only; a plain board has nothing to release). The name goes first, so
    the segment is gone even when the close fails; the array is dropped before the close, which a live view of the buffer refuses."""
    if b.shm is not None:
        b.shm.unlink()
        b.arr = np.empty(0, dtype=np.uint8)
        b.shm.close()


def _release_boards(boards: List[Board]) -> None:
    """The `weakref.finalize` callback of a Supply's boards (a module function: a bound method would keep the Supply alive). Each
    board is released in its own `try`: one that fails does not keep the rest."""
    if not boards:
        return
    try:
        release_board(boards[0])
    finally:
        _release_boards(boards[1:])


def _attached(ref: BoardRef) -> np.ndarray:
    """A worker's view of a board, attached once by its name (kept in _W['boards'] for the worker's life; the creator unlinks)."""
    name, shape = ref
    if name not in _W['boards']:
        shm: shared_memory.SharedMemory = shared_memory.SharedMemory(name=name)
        _W['boards'][name] = (shm, np.ndarray(shape, dtype=np.uint8, buffer=shm.buf))
    return _W['boards'][name][1]


def _worker_place(job: Tuple[Spot, int, int, BoardRef]) -> Made:
    """Read a place, write its reference read to the board's row i and draw its views (`plans_of`). BLANK / OFF (the screen let it
    through: Batch.blank counts it) writes nothing."""
    s, i, seed, ref_refs = job
    t: float = time.perf_counter()
    c: float = time.process_time()
    img: Optional[np.ndarray] = read_spot(_reader(s.path), s, _W['read'])
    t_read: float = time.perf_counter() - t
    code: int = OFF if img is None else (BLANK if grey_sd(centre(img, _W['tile'])) < _W['data'].blank_sd else USABLE)
    geos: List[Geometry] = []
    views: List[ViewPlan] = []
    if code == USABLE:
        _attached(ref_refs)[i] = img
        geos, views = plans_of(seed, _W['data'], _W['sources'], _W['labs'])
    return Made(code, geos, views, os.getpid(), time.perf_counter() - t, time.process_time() - c, t_read)


def probe_plans(seed: int, labs: Dict[str, LabGap]) -> Tuple[List[Geometry], List[Tuple[str, int, Optional[Shot]]]]:
    """The probe's frames and views of a place, no pixel made: the frames' geometries (4 + len(labs)) and (kind, frame index, shot) a
    view, always in this order (K = 4 + 2 x len(labs)): the anchor (ref, level 0: the gallery; frame 0); `twin:<source>` the source
    alone (frame 0, a shot); `sib:interp` the framing resampled half a pixel and nothing else; `sib:light`, `sib:large` the geometry
    of that level alone; `both:<source>` both (its own frame at level large, a shot), as a photo meets a search (logs
    before 2026-11-15 call this kind `real:<source>`; it is a simulation, not a real photo)."""
    rng: random.Random = random.Random(seed)
    geos: List[Geometry] = [Geometry()]
    views: List[Tuple[str, int, Optional[Shot]]] = [(PROBE_ANCHOR, 0, None)] + [(f'twin:{s}', 0, draw_shot(lab, rng)) for s, lab in labs.items()]
    geos.append(Geometry(shift=(0.5, 0.5)))
    views.append(('sib:interp', 1, None))
    for lv in ('light', 'large'):
        geos.append(draw_geometry(rng, LEVELS[lv]))
        views.append((f'sib:{lv}', len(geos) - 1, None))
    for s, lab in labs.items():
        geos.append(draw_geometry(rng, LEVELS['large']))                           # the geometry first, then the shot
        views.append((f'both:{s}', len(geos) - 1, draw_shot(lab, rng)))
    return geos, views


@dataclass
class Batch:
    """A process's share of a step. A place's views are contiguous, slot outside and source inside: a view's index within its place is
    `slot x S + k` (S sources, k the source index); V = slots x S views a place, B = places x V:
    refs     uint8 [places, read, read, 3]   the reference reads (the workers', as read)
    frames   List[(int, Geometry)]           a frame = (its place's row of `refs`, its geometry); a place's `Supply.groups` frames contiguous
    base     int64 [B]                       a view's frame (a row of `frames`)
    shots    List[Optional[Shot]], a view    None = the ref: the view is its frame; else the drawn parameters of the simulation
    source   int64 [B]                       0 = ref, k = the k-th simulator of `sources[1:]`
    place    int64 [B]                       0 .. places-1 (this process's)
    weight   float32 [B]                     the place's rung weight (`Supply.weights_of`, from the rungs of this share), for the reconstruction loss
    blank    the places of this process's share that came back blank or off the slide although the pool was screened at start (0 is the rule)
    geos     the views' Geometry
    sources  the source names in index order
    slots    1 (canonical only) or 2 (canonical, perturbed)
    slides   a place's slide name, a place
    feet     a place's footprint (x0, y0, x1, y1) in level-0 px, a place
    index    a place's index into Supply.places, a place
    cell     a place's (slide, rung) cell: an index into Supply.cells, a place
    The views' pixels are made on the GPU: `train.views_of` (the network's input tiles and the canvases the targets read).
    places = the places this process read of its share: at most `data.places / world`, fewer only when a place the screen let through
    came back blank or off the slide (`blank` counts them; 0 is possible)
    timing   seconds the batch took to make, by part: cand (the step's places), wall (take-off to the last view; two flights are in
             the air, so consecutive walls overlap), pack (the arrays), read (the workers' slide-read seconds, summed; part of work),
             work (the workers' wall seconds on this batch, summed), cpu (the workers' CPU seconds, summed; / work ~ 1 when each has a
             core of its own), pids (how many different workers reported)"""
    refs: torch.Tensor
    frames: List[Tuple[int, Geometry]]
    base: torch.Tensor
    shots: List[Optional[Shot]]
    source: torch.Tensor
    place: torch.Tensor
    weight: torch.Tensor
    geos: List[Geometry]
    sources: Tuple[str, ...]
    slots: int
    slides: List[str]
    feet: List[Tuple[float, float, float, float]]
    index: List[int]
    cell: List[int]
    timing: Dict[str, float] = field(default_factory=dict)
    blank: int = 0


@dataclass(frozen=True)
class Place:
    """One lattice place: its slide (index into Supply.slides), the level and ds it is read at, its centre and footprint (level-0 px)."""
    slide: int
    level: int
    ds: float
    cx: float
    cy: float
    foot: Tuple[float, float, float, float]

    def meets(self, other: 'Place') -> bool:
        f, g = self.foot, other.foot
        return self.slide == other.slide and f[0] < g[2] and g[0] < f[2] and f[1] < g[3] and g[1] < f[3]


@dataclass
class Flight:
    """A step's share being made: its places, the workers' futures (place i -> row i of board `board`), when it took off."""
    places: List[Place]
    futs: List['Future[Made]']
    board: int
    t0: float


def training_slides(cfg: DataConfig) -> List[Tuple[str, str]]:
    """(name, path) of the `stain` slides of `dataset` (the first `slides` of them when > 0); raises when one of them is registered in
    a `forbidden` dataset."""
    banned = {n for d in cfg.forbidden for n in list_names(dataset=d)}
    out: List[Tuple[str, str]] = []
    for source, names in by_source(cfg.dataset, stain=cfg.stain):
        for name in names:
            if source in cfg.forbidden or name in banned:
                raise ValueError(f'{name} ({source}) is in a forbidden dataset {cfg.forbidden}: its photos validate, it is never trained on')
            out.append((name, str(locate(name, dataset=source).path)))
    if not out:
        raise RuntimeError(f'no {cfg.stain} slide in {cfg.dataset}')
    return out[:cfg.slides] if cfg.slides > 0 else out


def footprint_px(cfg: CoaeConfig) -> int:
    """A place's side at its rung, in that rung's px: the tile and the largest shift on both sides."""
    return int(cfg.cofig.tile + 2 * np.ceil(cfg.data.shift_px))


def rungs_of(max_ds: int) -> Tuple[float, ...]:
    """ds 1, 2, 4 .. max_ds."""
    return tuple(float(2 ** k) for k in range(int(np.log2(max_ds)) + 1))


def _morton(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """The Z-order code of lattice cells (ix, iy), each < 2**16: their bits interleaved."""
    code: np.ndarray = np.zeros(len(ix), dtype=np.uint64)
    for b in range(16):
        code |= ((ix.astype(np.uint64) >> np.uint64(b)) & np.uint64(1)) << np.uint64(2 * b)
        code |= ((iy.astype(np.uint64) >> np.uint64(b)) & np.uint64(1)) << np.uint64(2 * b + 1)
    return code


def pool_runs(places: List[Place], idx: List[int], cap: int, key: str) -> List[List[int]]:
    """A (slide, rung) cell's RUNS: its lattice in Z order cut into m = min(cap, n) runs of n / m neighbours (cap 0: m = n), each
    run listed from the place the systematic sample picks (offset seeded by `key`) on along the curve to the next run's pick (the
    last run wraps round). A run's first candidate is the one a pool without screening would hold; the rest stand in for it, nearest
    along the curve first. Only the scan's lattice coordinates decide."""
    n: int = len(idx)
    m: int = n if cap <= 0 or n <= cap else cap
    if m == n:
        return [[i] for i in idx]
    pitch: float = places[idx[0]].foot[2] - places[idx[0]].foot[0]                      # the cell's lattice step, level-0 px
    cx: np.ndarray = np.array([places[i].cx for i in idx])
    cy: np.ndarray = np.array([places[i].cy for i in idx])
    ix: np.ndarray = np.floor((cx - cx.min()) / pitch + 0.5).astype(np.int64)
    iy: np.ndarray = np.floor((cy - cy.min()) / pitch + 0.5).astype(np.int64)
    if max(int(ix.max()), int(iy.max())) >= 2 ** 16:
        raise ValueError(f'pool {key}: the lattice ({int(ix.max()) + 1} x {int(iy.max()) + 1} cells) is too wide for the Z-order code')
    order: np.ndarray = np.argsort(_morton(ix, iy), kind='stable')                       # ties keep the draw's order
    u: float = random.Random(zlib.crc32(f'{key}'.encode())).random()                     # [0, 1)
    start: np.ndarray = np.floor((np.arange(m) + u) * n / m).astype(np.int64)            # strictly increasing, < n (n / m > 1)
    runs: List[List[int]] = []
    for j in range(m):
        end: int = int(start[j + 1]) if j + 1 < m else int(start[0]) + n                 # the next run's pick (the last wraps)
        runs.append([idx[int(order[t % n])] for t in range(int(start[j]), end)])
    return runs


def screened_pool(runs: List[List[List[int]]], screen: Callable[[List[int]], List[int]]) -> Tuple[List[List[int]], Dict[int, int], int]:
    """The pool of every cell (`runs[c]`: `pool_runs`): of each run the first candidate `screen` finds USABLE; a run with none adds
    nothing (the cell keeps as many as pass). `screen` takes place indices and returns their verdicts; it is called a ROUND at a
    time -- every unsettled run's next candidate, all cells together -- so a round is one parallel pass. Returns (the cells' pools,
    sorted; how many verdicts of each kind; the rounds)."""
    at: List[List[int]] = [[0] * len(r) for r in runs]
    got: List[List[int]] = [[-1] * len(r) for r in runs]
    tally: Dict[int, int] = {USABLE: 0, BLANK: 0, OFF: 0}
    rounds: int = 0
    while True:
        ask: List[Tuple[int, int, int]] = [(c, j, runs[c][j][at[c][j]]) for c in range(len(runs)) for j in range(len(runs[c]))
                                           if got[c][j] < 0 and at[c][j] < len(runs[c][j])]
        if not ask:
            break
        rounds += 1
        for (c, j, i), code in zip(ask, screen([i for _, _, i in ask])):
            tally[code] += 1
            if code == USABLE:
                got[c][j] = i
            else:
                at[c][j] += 1
    return [sorted(i for i in g if i >= 0) for g in got], tally, rounds


class Supply:
    """The places of the training slides and the steps' batches. A Supply may be built for its places alone (`pool` False: the probe
    lists no epoch); only `steps_an_epoch` refuses a Supply whose epoch cannot fill one step."""

    def __init__(self, cfg: CoaeConfig, rank: int, world: int, device: torch.device, mask_job: str = 'Masks', workers: int = 8,
                 say: Any = print, pool: bool = True) -> None:
        """`pool` False: a Supply for its places alone (the probe): no place is screened, its cells are the whole draw, it lists no
        epoch."""
        self.cfg: CoaeConfig = cfg
        self.rank: int = rank
        self.world: int = world
        self.workers: int = workers
        all_labs: Dict[str, LabGap] = sim_labs()
        self.labs: Dict[str, LabGap] = {s: all_labs[s] for s in cfg.data.sims}
        self.probe_labs: Dict[str, LabGap] = probe_labs(cfg.probe.ood)
        self.sources: Tuple[str, ...] = (SOURCES_REF,) + tuple(cfg.data.sims)
        self.slots: int = cfg.data.slots
        self.versions: int = self.slots * len(self.sources)                       # views a place
        self.photo_px: int = cfg.data.photo_px
        self.read: int = read_px(cfg.data, self.photo_px)
        self.tile: int = cfg.cofig.tile
        self.groups: int = groups_of(cfg.data, len(self.sources))                  # the frames of a place
        if self.read < self.tile + 2 * SCREEN_PAD or (self.read - self.tile) % 2:
            raise ValueError(f'a place is read {self.read} px: the screen reads the centre tile ({self.tile} px) and {SCREEN_PAD} px round it, '
                             f'so the read must be at least {self.tile + 2 * SCREEN_PAD} px and differ from the tile by an even number')
        foot: int = footprint_px(cfg)
        # the camera: the footprint, and what is read around it for the photo canvas (the sampler keeps that reserve on the slide)
        camera: ReadSpec = ReadSpec(foot, foot, margin_out=max(0, (self.read - foot + 1) // 2))
        plan: PlanSpec = PlanSpec('ladder', rungs_of(cfg.data.max_ds), camera)
        scfg: SamplerConfig = replace(SAMPLER_RECIPES[cfg.data.sampler], n_per_rung=ALL_PLACES)
        masks: MaskMaker = MaskMaker(MASK_RECIPES[cfg.data.mask], mask_job, device)
        self.names: List[str] = []
        self.paths: List[str] = []
        self.readers: Dict[int, SlideReader] = {}                                    # the training process's own readers (`_read`, the probe), opened at first use
        self.places: List[Place] = []
        t_draw: float = time.perf_counter()
        for name, path in training_slides(cfg.data):
            draw: TileSampler = TileSampler.cached(path, scfg, plan, cfg.data.draw_cache, masks=masks)
            si: int = len(self.names)
            for s in draw:
                m = s.meta
                cx, cy = m.centre_l0
                h: float = m.footprint_l0 / 2
                self.places.append(Place(si, int(m.level), float(m.ds), cx, cy, (cx - h, cy - h, cx + h, cy + h)))
            self.names.append(name)
            self.paths.append(path)
        draw_s: float = time.perf_counter() - t_draw
        if not self.places:
            raise RuntimeError(f'no place on any training slide ({cfg.data.mask} mask, {cfg.data.sampler} sampler)')
        self.say: Any = say
        by_cell: Dict[Tuple[int, float], List[int]] = {}
        for i, p in enumerate(self.places):
            by_cell.setdefault((p.slide, p.ds), []).append(i)
        keys: List[Tuple[int, float]] = sorted(by_cell, key=lambda k: (-k[1], k[0]))                 # coarse rungs first, then slide
        self.procs: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(     # always: before the pool is screened
            max_workers=cfg.data.processes, mp_context=mp.get_context('spawn'), initializer=_init_worker,
            initargs=(cfg.data, self.read, self.tile, self.sources, self.labs))
        self.pooled: bool = pool
        pools: List[List[int]]
        tally: Dict[int, int] = {USABLE: 0, BLANK: 0, OFF: 0}
        rounds: int = 0
        screen_s: float = 0.0
        if pool:
            t_screen: float = time.perf_counter()
            runs: List[List[List[int]]] = [pool_runs(self.places, by_cell[k], cfg.data.pool_cap,
                                                     f'{cfg.data.pool_seed}|pool|{self.names[k[0]]}|{k[1]:g}') for k in keys]
            pools, tally, rounds = screened_pool(runs, self.screen)
            screen_s = time.perf_counter() - t_screen
        else:
            pools = [list(by_cell[k]) for k in keys]
        self.cell_keys: List[Tuple[int, float]] = [k for k, c in zip(keys, pools) if c]              # a cell with no usable place is no cell
        self.cells: List[List[int]] = [c for c in pools if c]                                        # indices into self.places: the pool
        self.index_of: Dict[Place, int] = {p: i for i, p in enumerate(self.places)}
        if len(self.index_of) != len(self.places):
            raise RuntimeError(f'{len(self.places)} places but {len(self.index_of)} different ones: Supply.index_of needs every place once')
        self.cell_of: np.ndarray = np.full(len(self.places), -1, dtype=np.int64)      # a place's cell (an index into self.cells); -1 = not in the pool
        for c, members in enumerate(self.cells):
            self.cell_of[members] = c
        self.budget: List[int] = [min(len(c), cfg.data.cell_epoch, -(-len(c) // cfg.data.cell_parts)) for c in self.cells]   # a cell's places listed an epoch, at most: ceil(pool / cell_parts)
        self.listed: int = cfg.data.places                                                           # a step lists this many: every one is read
        self._steps: int = sum(self.budget) // self.listed if pool else 0                            # 0: no epoch of this Supply fills a step
        self.pool_rung: Dict[float, int] = dict(sorted(Counter(self.places[i].ds for c in self.cells for i in c).items()))   # the pool, a rung
        self.draw_rung: Dict[float, int] = dict(sorted(Counter(p.ds for p in self.places).items()))   # the draw, a rung
        self.pool_crc: int = zlib.crc32(np.asarray([i for c in self.cells for i in c], dtype=np.int64).tobytes())   # which pool: checked across processes and by a resumed run
        self.seen: np.ndarray = np.zeros(len(self.places), dtype=np.int32)          # times a place was among a step's places (every process the same)
        self.last_seen: np.ndarray = np.full(len(self.places), -1, dtype=np.int32)  # the epoch it was last seen in
        self._boards: Optional[List[Board]] = None                                  # made by the first `launch()`
        self._free: Optional[weakref.finalize] = None
        self._flights: int = 0                                                      # launched so far: picks the board set
        if not pool:
            say(f'supply (places alone, unscreened): {len(self.names)} training slides ({cfg.data.dataset}, {cfg.data.stain}), '
                f'{len(self.places)} places in {len(self.cells)} (slide, rung) cells ({cfg.data.mask} mask, {cfg.data.sampler} sampler, '
                f'footprint {foot} px; draw {draw_s:.0f} s); draw by rung '
                + ', '.join(f'ds {ds:g}: {n}' for ds, n in self.draw_rung.items()))
            return
        epoch_rung: Counter = Counter()
        for (slide, ds), b in zip(self.cell_keys, self.budget):
            epoch_rung[ds] += b
        say(f'supply: {len(self.names)} training slides ({cfg.data.dataset}, {cfg.data.stain}), {len(self.places)} places in '
            f'{len(keys)} (slide, rung) cells ({cfg.data.mask} mask, {cfg.data.sampler} sampler, footprint {foot} px; '
            f'draw {draw_s:.0f} s); draw by rung '
            + ', '.join(f'ds {ds:g}: {n}' for ds, n in self.draw_rung.items())
            + f'; screened {sum(tally.values())} places in {rounds} rounds, {screen_s:.0f} s: {tally[BLANK]} blank, {tally[OFF]} off the '
            f'slide, {len(keys) - len(self.cells)} cells left without a place; pool (pool_cap {cfg.data.pool_cap}, seed {cfg.data.pool_seed}) '
            f'by rung '
            + ', '.join(f'ds {ds:g}: {n}' for ds, n in self.pool_rung.items())
            + f'; an epoch lists {sum(self.budget)} of them (1/{cfg.data.cell_parts} of a cell\'s pool, at most cell_epoch '
            f'{cfg.data.cell_epoch}, the least seen first): '
            + ', '.join(f'ds {ds:g}: {n}' for ds, n in sorted(epoch_rung.items()))
            + f'; {self._steps} steps of {self.listed} places; rung weights from each step\'s own rungs; '
            f'read {self.read} px a place, level {cfg.data.level}, mechanism {cfg.data.mechanism}, {self.versions} views a place '
            f'({self.slots} slots x {len(self.sources)} sources), reads and the draws of the views by {cfg.data.processes} processes, '
            f'{cfg.canvases} canvas(es) a slot read by the targets, {len(os.sched_getaffinity(0))} cpus allowed')

    def spot(self, p: Place) -> Spot:
        """Where a worker reads the place: the level-0 top-left of its reference square, its ds and level."""
        return Spot(self.paths[p.slide], int(round(p.cx - self.read / 2 * p.ds)), int(round(p.cy - self.read / 2 * p.ds)), p.ds, p.level)

    def screen(self, idx: List[int]) -> List[int]:
        """The verdicts (USABLE, BLANK, OFF) of places, by the worker processes."""
        spots: List[Spot] = [self.spot(self.places[i]) for i in idx]
        return list(self.procs.map(_worker_screen, spots, chunksize=max(1, len(spots) // (8 * self.cfg.data.processes))))

    @property
    def steps_an_epoch(self) -> int:
        """The steps of an epoch; raises when the places an epoch lists do not fill one step, or for a Supply built for its places alone."""
        if not self.pooled:
            raise RuntimeError('a Supply built for its places alone lists no epoch')
        if self._steps == 0:
            raise RuntimeError(f'{len(self.cells)} cells (pool_cap {self.cfg.data.pool_cap}, cell_parts {self.cfg.data.cell_parts}, '
                               f'cell_epoch {self.cfg.data.cell_epoch}) list {sum(self.budget)} places an epoch, fewer than one step '
                               f'of {self.listed}: raise data.pool_cap or lower data.cell_parts, data.places')
        return self._steps

    def weights_of(self, counts: Dict[float, int]) -> Dict[float, float]:
        """A step's loss weight a rung, from the places this process read of each (`counts`): M / (sqrt(its places) x the sum of the
        rungs' sqrt(places)), M the places -- as 1 / sqrt(its rung's places), mean 1 over the step's places; 1 when `level_weights`
        is off. A rung absent from the step is absent."""
        if not self.cfg.data.level_weights:
            return {ds: 1.0 for ds in counts}
        m: int = sum(counts.values())
        roots: float = float(sum(math.sqrt(n) for n in counts.values()))
        return {ds: m / (math.sqrt(n) * roots) for ds, n in counts.items()}

    # ── an epoch's places, the same listing on every process ────────────────────────────────────────────────────────
    def epoch_listing(self, epoch: int) -> Iterator[List[Place]]:
        """The epoch's steps, `listed` places each, the same on every process. Each (slide, rung) cell lists at most its `budget`
        this epoch, the least seen first (`seen`, then `last_seen`, then at random), spread evenly over `steps_an_epoch` steps (a credit
        a step, floor + largest remainder, ties broken by the step's seed); a place once at most. Coarse rungs list first; a place
        meeting one already listed on its slide goes to the back of its cell's queue and comes again later in the epoch. A step's
        places count as seen. Ends when a step cannot be filled."""
        seed: int = self.cfg.train.seed
        C: int = len(self.cells)
        E: int = self.steps_an_epoch
        want: int = self.listed
        lines: List[Deque[int]] = []
        for c, idx in enumerate(self.cells):
            arr: np.ndarray = np.asarray(idx, dtype=np.int64)
            g: np.random.Generator = np.random.default_rng(zlib.crc32(f'{seed}|{epoch}|cell{c}|pick'.encode()))
            if len(arr) > self.budget[c]:
                order: np.ndarray = np.lexsort((g.random(len(arr)), self.last_seen[arr], self.seen[arr]))   # keys: seen, then last_seen, then random
                arr = arr[order[:self.budget[c]]]
            arr = arr.copy()
            g.shuffle(arr)
            lines.append(deque(int(i) for i in arr))
        used: List[int] = [0] * C                      # listed this epoch, a cell
        credit: List[float] = [0.0] * C
        rate: List[float] = [b / E for b in self.budget]

        for step in range(E):
            rng: random.Random = random.Random(zlib.crc32(f'{seed}|{epoch}|{step}|quota'.encode()))
            listed: List[int] = []
            by_slide: Dict[int, List[Place]] = {}                       # the step's places a slide: the meets check scans one slide

            def pull(c: int, n: int) -> int:
                """Up to n places of cell c into `listed`; one meeting a listed place of its slide goes to the back of the queue
                (each place looked at once a pull). Returns how many were listed."""
                got: int = 0
                tries: int = len(lines[c])
                while got < n and used[c] < self.budget[c] and tries > 0:
                    i: int = lines[c].popleft()
                    tries -= 1
                    p: Place = self.places[i]
                    if any(p.meets(q) for q in by_slide.get(p.slide, ())):
                        lines[c].append(i)
                        continue
                    listed.append(i)
                    by_slide.setdefault(p.slide, []).append(p)
                    used[c] += 1
                    got += 1
                return got

            for c in range(C):
                credit[c] += rate[c]
            ask: List[int] = [max(0, min(int(math.floor(credit[c])), self.budget[c] - used[c])) for c in range(C)]
            short: int = want - sum(ask)
            pri: List[int] = sorted(range(C), key=lambda c: (-(credit[c] - ask[c]), rng.random()))   # largest remainder first
            for c in pri:
                if short <= 0:
                    break
                if used[c] + ask[c] < self.budget[c]:
                    ask[c] += 1
                    short -= 1
            while short < 0:                                             # the floors can add up to more than a step: the smallest remainders wait
                for c in reversed(pri):
                    if short >= 0:
                        break
                    if ask[c] > 0:
                        ask[c] -= 1
                        short += 1
            for c in range(C):                                           # self.cells is coarse-first
                if ask[c] > 0:
                    credit[c] -= pull(c, ask[c])
            while len(listed) < want:                                    # the places handed back left the step short: a place a cell, by remainder
                progressed: bool = False
                for c in pri:
                    if len(listed) >= want:
                        break
                    if used[c] < self.budget[c]:
                        got: int = pull(c, 1)
                        credit[c] -= got
                        progressed = progressed or got > 0
                if not progressed:
                    break
            if len(listed) < want:
                self.say(f'epoch {epoch}: {len(listed)} of {want} places listable at step {step} of {E}: next epoch')
                return
            random.Random(zlib.crc32(f'{seed}|{epoch}|{step}|listed'.encode())).shuffle(listed)
            head: np.ndarray = np.asarray(listed, dtype=np.int64)                            # every listed place is read
            self.seen[head] += 1
            self.last_seen[head] = epoch
            yield [self.places[i] for i in listed]

    def _read(self, p: Place) -> Optional[np.ndarray]:
        """The reference square round a place (the probe's read, in this process), or None when it runs off the slide or its centre tile
        is blank."""
        r: Optional[SlideReader] = self.readers.get(p.slide) or self.readers.setdefault(p.slide, SlideReader(SafeSlide(self.paths[p.slide])))
        img: Optional[np.ndarray] = read_spot(r, self.spot(p), self.read)
        if img is None or self._blank(centre(img, self.cfg.cofig.tile)):
            return None
        return img

    def _blank(self, tile: np.ndarray) -> bool:
        """The blank rule (ExpCountBlankTiles): a tile whose grey sd is below `blank_sd` is background."""
        return grey_sd(tile) < self.cfg.data.blank_sd

    def probe_set(self, cfg: CoaeConfig, field_px: int, device: torch.device) -> Probe:
        """The probe (spec.md §6) of this Supply's places: each (slide, rung) cell with `cfg.probe.rungs`' ds takes the first
        `cfg.probe.places` of its places (a shuffle seeded by the cell) that read and are not blank; a cell with fewer is left out. Each
        place is `probe_plans` (seeded by the cell and the place's index); its canvases (`GpuSims.frames`, `GpuSims.simulate`), its
        descriptor and every view's own Co-figure are made on the GPU. A cell at a time, the canvases dropped after: the same config gives
        the same probe."""
        pc: ProbeConfig = cfg.probe
        tile: int = cfg.cofig.tile
        weights: torch.Tensor = torch.tensor(GREY_WEIGHTS, dtype=torch.float32, device=device)
        tiles: List[torch.Tensor] = []
        cell_of: List[int] = []
        cell_names: List[str] = []
        direct: List[torch.Tensor] = []
        maps: List[torch.Tensor] = []
        kinds: List[str] = []
        t0: float = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for c, (slide, ds) in enumerate(self.cell_keys):
                if ds not in pc.rungs:
                    continue
                name: str = self.names[slide]
                idx: List[int] = list(self.cells[c])
                random.Random(zlib.crc32(f'{pc.seed}|probe|{name}|{ds:g}'.encode())).shuffle(idx)
                reads: List[np.ndarray] = []
                at: int = 0
                while len(reads) < pc.places and at < len(idx):
                    chunk: List[int] = idx[at:at + 2 * pc.places]
                    at += len(chunk)
                    for img in pool.map(lambda i: self._read(self.places[i]), chunk):
                        if img is not None and len(reads) < pc.places:
                            reads.append(img)
                if len(reads) < pc.places:
                    self.say(f'probe: {name} ds {ds:g}: {len(reads)} of {pc.places} places, left out')
                    continue
                seeds: List[int] = [zlib.crc32(f'{pc.seed}|probe|{name}|{ds:g}|{j}'.encode()) & 0x7fffffff for j in range(pc.places)]
                made: List[Tuple[List[Geometry], List[Tuple[str, int, Optional[Shot]]]]] = [probe_plans(seed, self.probe_labs) for seed in seeds]
                kinds = [k for k, _, _ in made[0][1]]
                k_n: int = len(kinds)
                step: int = max(1, 128 // k_n)                                                   # never more than 128 canvases on the GPU
                cell_tiles: List[torch.Tensor] = []
                dir_c: List[torch.Tensor] = []
                maps_c: List[torch.Tensor] = []
                for s in range(0, pc.places, step):
                    part: List[Tuple[List[Geometry], List[Tuple[str, int, Optional[Shot]]]]] = made[s:s + step]
                    refs_d: torch.Tensor = torch.from_numpy(np.stack(reads[s:s + step])).to(device)          # [len(part), read, read, 3]
                    fr: List[Tuple[int, Geometry]] = [(j, g) for j, (geos_, _) in enumerate(part) for g in geos_]       # every frame of the part
                    nb: int = len(part[0][0])
                    bases_d: torch.Tensor = frames(refs_d, [j for j, _ in fr], [g for _, g in fr], self.photo_px)      # [len(part) x nb, H, W, 3]
                    rows: List[int] = [j * nb + g for j, (_, vs) in enumerate(part) for _, g, _ in vs]
                    shots: List[Optional[Shot]] = [sh for _, vs in part for _, _, sh in vs]
                    canv: torch.Tensor = bases_d[torch.tensor(rows, device=device)]                           # uint8 [len(part) x K, H, W, 3]
                    sim: List[int] = [j for j, sh in enumerate(shots) if sh is not None]
                    if sim:
                        at_d: torch.Tensor = torch.tensor(sim, device=device)
                        canv[at_d] = simulate(canv[at_d], [shots[j] for j in sim])
                    a: int = self.photo_px // 2 - tile // 2
                    cell_tiles.append(canv[:, a:a + tile, a:a + tile].contiguous().cpu().view(len(part), k_n, tile, tile, 3))
                    grey: torch.Tensor = canv.float() @ weights
                    dir_c.append(place_moments(grey, cfg.cofig, field_px, cfg.head.pool_cells, cfg.data.dilate_px).cpu())
                    maps_c.append(field_maps(grey, cfg.cofig, field_px).half().cpu())                         # every view's own Co-figure [len(part) x K, N, t, t]
                tiles.append(torch.cat(cell_tiles))
                m_n, c_n = dir_c[0].shape[1], dir_c[0].shape[2]
                direct.append(torch.cat(dir_c).view(pc.places, k_n, m_n, c_n, c_n))
                maps.append(torch.cat(maps_c).view(pc.places, k_n, *maps_c[0].shape[1:]))
                cell_of += [len(cell_names)] * pc.places
                cell_names.append(f'{name}|ds{ds:g}')
        if not cell_names:
            raise RuntimeError(f'probe: no (slide, rung) cell of {pc.dataset} has {pc.places} places at ds {pc.rungs}')
        self.say(f'probe: {len(cell_of)} places in {len(cell_names)} cells, {len(kinds)} views each, {time.perf_counter() - t0:.0f} s')
        return Probe(torch.cat(tiles), kinds, torch.tensor(cell_of, dtype=torch.int64), cell_names, torch.cat(direct), torch.cat(maps))

    def _get_boards(self, n_places: int) -> List[Board]:
        """BOARDS boards of reference reads [n_places, read, read, 3], shared memory, made at the first call; the segments are unlinked
        by `close` or, failing that, when the Supply is collected or the interpreter exits. The boards' size is fixed by that first call:
        a later call with another `n_places` (a Supply's rank and world changed after it made a batch) raises."""
        if self._boards is None:
            self._boards = [make_board((n_places, self.read, self.read, 3), True) for _ in range(BOARDS)]
            self._free = weakref.finalize(self, _release_boards, list(self._boards))
        elif self._boards[0].arr.shape[0] != n_places:
            raise ValueError(f'the boards hold {self._boards[0].arr.shape[0]} places a step, asked for {n_places}: a '
                             f'Supply\'s rank and world are fixed once it made a batch')
        return self._boards

    def close(self) -> None:
        """Shut down the worker processes, then release the boards (unlink their shared memory)."""
        if self.procs is not None:
            self.procs.shutdown()
            self.procs = None
        if self._free is not None:
            self._free()
            self._free = None
            self._boards = None

    def state(self) -> Dict[str, torch.Tensor]:
        """What a resumed run needs of this Supply: how often and when each place was seen (the same on every process), and a checksum of
        the pool the screen at start gave. The rest of an epoch's listing and of a batch is a function of (train.seed, epoch, step,
        rank)."""
        return {'seen': torch.from_numpy(self.seen.copy()), 'last_seen': torch.from_numpy(self.last_seen.copy()),
                'pool': torch.tensor([self.pool_crc], dtype=torch.int64)}

    def load_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Take over a `state()` of a Supply of the same draw and pool; raises when it holds another number of places or was made on
        another pool."""
        for name in ('seen', 'last_seen', 'pool'):
            if name not in state:
                raise ValueError(f'the saved state has no {name!r}')
        for name in ('seen', 'last_seen'):
            n: int = int(state[name].shape[0]) if state[name].dim() == 1 else -1
            if n != len(self.places):
                raise ValueError(f'the saved state has {name} of {tuple(state[name].shape)}, this Supply has {len(self.places)} places')
        if int(state['pool'][0]) != self.pool_crc:
            raise ValueError(f'the saved state is of another pool (the screen at start gave other places): checksum {int(state["pool"][0])}, '
                             f'this Supply\'s {self.pool_crc}')
        self.seen = state['seen'].numpy().astype(np.int32).copy()
        self.last_seen = state['last_seen'].numpy().astype(np.int32).copy()

    def launch(self, listed: List[Place], step: int, epoch: int) -> Flight:
        """Send this process's share of a step (every world-th place) to the workers: a job a place (read, screen, draws)."""
        cfg: CoaeConfig = self.cfg
        if cfg.data.places % self.world:
            raise ValueError(f'places {cfg.data.places} is not a multiple of the {self.world} processes: the remainder would be dropped')
        n_places: int = cfg.data.places // self.world
        mine: List[Place] = listed[self.rank::self.world]
        if len(mine) != n_places:
            raise ValueError(f'a step of {len(listed)} places gives rank {self.rank} {len(mine)}, not {n_places}')
        k: int = self._flights % BOARDS
        self._flights += 1
        refs_b: Board = self._get_boards(n_places)[k]
        rng: random.Random = random.Random(zlib.crc32(f'{cfg.train.seed}|{epoch}|{step}|{self.rank}|versions'.encode()))
        seeds: List[int] = [rng.getrandbits(31) for _ in range(n_places)]
        futs: List['Future[Made]'] = [self.procs.submit(_worker_place, (self.spot(p), i, seeds[i], refs_b.ref))
                                      for i, p in enumerate(mine)]
        return Flight(mine, futs, k, time.perf_counter())

    def land(self, f: Flight) -> Batch:
        """Wait for a flight's workers and take its batch off the boards (a copy: the batch owns its arrays, the board set is free for
        the next flight). The places that came back BLANK or OFF are left out and counted (`Batch.blank`): the screen at start should
        leave none."""
        made: List[Made] = [x.result() for x in f.futs]                  # in place order; a worker's exception is raised here
        t_made: float = time.perf_counter()
        keep: List[int] = [i for i, m in enumerate(made) if m.code == USABLE]
        kept_p: List[Place] = [f.places[i] for i in keep]
        rows: np.ndarray = np.asarray(keep, dtype=np.int64)
        refs: torch.Tensor = torch.from_numpy(self._boards[f.board].arr[rows])                                  # fancy index = a copy
        w: Dict[float, float] = self.weights_of(dict(Counter(p.ds for p in kept_p)))
        fr: List[Tuple[int, Geometry]] = []
        geos: List[Geometry] = []
        shots: List[Optional[Shot]] = []
        base: List[int] = []
        src: List[int] = []
        place: List[int] = []
        weight: List[float] = []
        for n_, i in enumerate(keep):
            if len(made[i].frames) != self.groups:
                raise RuntimeError(f'a place came back with {len(made[i].frames)} frames, {self.groups} expected (groups_of)')
            first: int = len(fr)
            fr += [(n_, g) for g in made[i].frames]
            for geo, k, g, shot in made[i].views:
                geos.append(geo)
                src.append(k)
                place.append(n_)
                weight.append(w[kept_p[n_].ds])
                base.append(first + g)
                shots.append(shot)
        t_pack: float = time.perf_counter()
        index: List[int] = [self.index_of[p] for p in kept_p]
        return Batch(refs, fr, torch.tensor(base, dtype=torch.int64), shots, torch.tensor(src, dtype=torch.int64),
                     torch.tensor(place, dtype=torch.int64), torch.tensor(weight, dtype=torch.float32), geos, self.sources, self.slots,
                     [self.names[p.slide] for p in kept_p], [p.foot for p in kept_p],
                     index, [int(self.cell_of[i]) for i in index],
                     dict(cand=0.0, wall=t_made - f.t0, pack=t_pack - t_made, read=float(sum(m.read for m in made)),
                          work=float(sum(m.wall for m in made)), cpu=float(sum(m.cpu for m in made)),
                          pids=float(len({m.pid for m in made}))),
                     blank=len(made) - len(keep))

    def batch(self, listed: List[Place], step: int, epoch: int) -> Batch:
        """`land(launch(...))`: one batch alone (tests); not while a `batches()` runs."""
        return self.land(self.launch(listed, step, epoch))

    def batches(self, epoch: int, ahead: int = 2) -> Iterator[Batch]:
        """The epoch's batches (`epoch_listing`): at most BOARDS flights in the air and `ahead` batches waiting for the training, made by
        a thread that only waits for the workers; a board set is launched again only after its flight landed (the flights land in
        order). `step` counts within the epoch (the versions' seed carries the epoch too). A Supply has one unfinished `batches()` at a
        time, and `batch()` is not called directly while one runs (both use the boards)."""
        q: 'queue.Queue[Any]' = queue.Queue(maxsize=ahead)

        def work() -> None:
            try:
                listing: Iterator[List[Place]] = self.epoch_listing(epoch)
                flying: Deque[Tuple[Flight, float]] = deque()
                s: int = 0
                more: bool = True
                while more or flying:
                    if more and len(flying) < BOARDS:
                        t: float = time.perf_counter()
                        step_places: Optional[List[Place]] = next(listing, None)
                        if step_places is None:
                            more = False
                        else:
                            flying.append((self.launch(step_places, s, epoch), time.perf_counter() - t))
                            s += 1
                        continue
                    f, cand = flying.popleft()
                    b: Batch = self.land(f)
                    b.timing['cand'] = cand
                    q.put(b)
            except BaseException as exc:                         # noqa: BLE001  (handed to the consumer)
                q.put(exc)
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
