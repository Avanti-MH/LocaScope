#!/usr/bin/env python3
"""Tests for what every stage-2 retriever stands on: `stage2_retrieval/Retriever.py`
(the interface, the encoder pool, the bridge lock), `stage2_retrieval/StageInterface.py`
(the candidate kinds, `origin_l0`), `stage2_retrieval/Context.py` (the levels, the
sources, `WindowBatch`, boxes, on-demand tiles, `separated`, the greedy pass behind
`min_sep_tiles`) and the sliding window's scores (`RetrieverSlidingWinSimRot.list_windows`).

    python utilities/test_modules/TestRerank/test_context.py

No slide, no model, no GPU: grids are random unit vectors, the "slide" is a
function of the pixel position, the "encoder" a function of the pixels.

WHAT THIS DEFENDS
-----------------
    candidates   a shifted window is read where its shift says, in every method
                 that turns a candidate into a position; a stored row comes back
                 as the kind of candidate it was written from
    WindowBatch  the tensors of a window are the tiles the retriever matched:
                 their cosines average to the score the first phase gave, for
                 every rotation, lattice and region, on both window shapes
    gather       a block with a margin shows what is there and says where nothing is
    separated    the array pass keeps exactly what the list scan kept
    tiles        a tile asked for twice is encoded once
    retriever    (`Retriever.py`) an unbuilt retriever, a wrong upstream and a topk longer than the
                 config's list are refused; topk is a prefix and keeps the extras; fewer windows than
                 raw_topk come back as they are
    bridge       the matching extra is taken (the nearest, or a farther one); none of the kind: the
                 retriever makes its own; the kind under another key only: BridgeError
    encoders     one loaded model per trunk, read through any pooling; another trunk (encoder or
                 precision) and other weights under one trunk are refused
    levels       (`SlideLevels`) a level too small rules out the coarser ones, a level that raised is
                 remembered and not read again, the walk ends at the first level that lists windows;
                 `nearest_window` is the window centred on the point; a `GridSource` reads the same
                 windows as a `MemorySource` holding the same tensors
    slidewin     every window's score is its score of the cosines of the tiles under it (mean, geomean,
                 min, against the loops); the list is the best first, `raw_topk` long, kept apart at
                 one rotation by `min_sep_tiles`
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'bench_modules'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
import torch.nn.functional as F                                  # noqa: E402

from BenchCommon import stored_candidates                        # noqa: E402
from stage2_retrieval.Context import (GridSource, LevelFeatures, MemorySource,   # noqa: E402
                                      OnDemandTiles, QueryFeatures, SlideLevels, WindowBatch,
                                      box_area, gather_windows, iou_matrix, nearest_window,
                                      separated, window_boxes)
from stage2_retrieval.Retriever import (BridgeError, EncoderPool, Extra, RetrievalOutput,   # noqa: E402
                                        Retriever, Uses, check_encoder, take, trunk_id)
from stage2_retrieval.StageInterface import (Candidate, CandidateSet,  # noqa: E402
                                             RerankedCandidate, ShiftedCandidate,
                                             TokenMatchCandidate, VoteCandidate)

_RESULTS: List[Tuple[str, Optional[BaseException]]] = []


def check(name: str, fn: Callable[[], Optional[str]]) -> None:
    try:
        out: Optional[str] = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── stand-ins ─────────────────────────────────────────────────────────────────

TILE: int = 256


class _Grid:
    """What `CandidateSet` asks of a `PatchGrid`: a tile size, the offsets and
    two origin formulas. The offset lattice sits half a tile down and right."""
    tile_size: int = TILE
    x_offset: int = 0
    y_offset: int = 0

    def __init__(self, ds: float, origin: Tuple[int, int] = (0, 0)) -> None:
        self.ds: float = ds
        self.origin: Tuple[int, int] = origin

    def tile_origin(self, lattice: str, row: int, col: int) -> Tuple[int, int]:
        half: int = TILE // 2 if lattice == 'offset' else 0
        return col * TILE + half, row * TILE + half

    def tile_origin_l0(self, lattice: str, row: int, col: int) -> Tuple[int, int]:
        x, y = self.tile_origin(lattice, row, col)
        return self.origin[0] + int(round(x * self.ds)), self.origin[1] + int(round(y * self.ds))


def cset(cands: List[Candidate], ds: float = 2.0, n_regions: int = 2) -> CandidateSet:
    return CandidateSet(candidates=tuple(cands), level=1, ds=ds,
                        grids=tuple(_Grid(ds, (1000 * i, 0)) for i in range(n_regions)),
                        irretrievable_lvl=(), alter_lvl=())


def unit(*shape: int, seed: int = 0) -> torch.Tensor:
    g: torch.Generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(*shape, generator=g), dim=-1)


def make_source(dim: int = 8, qshape: Tuple[int, int] = (4, 5), n_regions: int = 2
                ) -> Tuple[MemorySource, Dict[Tuple[int, str], torch.Tensor]]:
    ref: Dict[Tuple[int, str], torch.Tensor] = {}
    for r in range(n_regions):
        ref[(r, 'main')] = unit(12 + r, 14, dim, seed=10 + r)
        ref[(r, 'offset')] = unit(11 + r, 13, dim, seed=20 + r)
    qs: Dict[Tuple[int, str], torch.Tensor] = {}
    for rot in (0, 90, 180, 270):
        shape: Tuple[int, int] = qshape if rot in (0, 180) else (qshape[1], qshape[0])
        qs[(rot, 'main')] = unit(*shape, dim, seed=30 + rot)
        qs[(rot, 'offset')] = unit(shape[0] - 1, shape[1] - 1, dim, seed=40 + rot)
    return MemorySource(ref, qs, 'cls'), ref


# ── candidates ────────────────────────────────────────────────────────────────

def t_the_base_candidate_is_unchanged() -> str:
    c: Candidate = Candidate(0, 'main', 3, 4, 90, 0.5)
    assert c.key() == (0, 'main', 3, 4, 90) and c.shift_px() == (0, 0) and c.extra() == {}
    assert list(c.__dataclass_fields__) == ['region_index', 'lattice', 'row', 'col', 'rotation', 'score']
    return 'six fields, no extras'


def t_candidate_kinds_carry_their_own_columns() -> str:
    base = (0, 'offset', 1, 2, 180, 0.4)
    r: RerankedCandidate = RerankedCandidate(*base, first_score=0.3, first_rank=7)
    assert r.extra() == dict(first_score=0.3, first_rank=7) and r.key() == Candidate(*base).key()
    s: ShiftedCandidate = ShiftedCandidate(*base, first_score=0.3, first_rank=7, shift=(-64, 32))
    assert s.shift_px() == (-64, 32) and s.extra()['dx'] == -64 and s.extra()['first_rank'] == 7
    assert s.key() == r.key(), 'a shift is not another hypothesis'
    t: TokenMatchCandidate = TokenMatchCandidate(*base, n_inliers=5, inlier_ratio=0.25)
    assert t.extra()['n_inliers'] == 5 and t.extra()['first_rank'] == 0
    assert VoteCandidate(*base, votes=2.5).extra() == dict(votes=2.5)
    return 'four kinds'


def t_a_shift_moves_the_one_position_formula() -> str:
    cs: CandidateSet = cset([Candidate(1, 'offset', 2, 3, 0, 1.0),
                             ShiftedCandidate(1, 'offset', 2, 3, 0, 1.0, shift=(-64, 32))], ds=2.0)
    plain, moved = cs.candidates
    x, y = cs.origin_l0(plain)
    assert (x, y) == (1000 + int(round((3 * TILE + 128) * 2.0)), int(round((2 * TILE + 128) * 2.0)))
    assert cs.origin_l0(moved) == (x - 128, y + 64), (cs.origin_l0(moved), (x, y))
    assert cs.window_local(moved) == (cs.window_local(plain)[0] - 64, cs.window_local(plain)[1] + 32)
    return 'level px times ds'


def t_rows_carry_the_extras_and_stored_rows_come_back() -> str:
    q = SimpleNamespace(grid=SimpleNamespace(tile_size=TILE, grid_rows=4, grid_cols=5))
    cs: CandidateSet = cset([
        ShiftedCandidate(0, 'main', 1, 1, 0, 0.9, first_score=0.8, first_rank=5, shift=(32, -32)),
        RerankedCandidate(0, 'main', 2, 2, 90, 0.7, first_score=0.7, first_rank=2),
        Candidate(0, 'main', 3, 3, 180, 0.5)])
    rows = cs.rows(q)
    assert rows[0]['dx'] == 32 and rows[0]['first_rank'] == 5 and 'dx' not in rows[1]
    # a CSV cell is text, and a cell a row does not have is empty
    cols = {k for r in rows for k in r}
    text = [{k: ('' if r.get(k) is None else str(r.get(k))) for k in cols} for r in rows]
    back = stored_candidates(text)
    assert isinstance(back[0], ShiftedCandidate) and back[0].shift == (32, -32)
    assert isinstance(back[1], RerankedCandidate) and not isinstance(back[1], ShiftedCandidate)
    assert back[1].first_rank == 2 and type(back[2]) is Candidate
    assert [b.key() for b in back] == [c.key() for c in cs.candidates]
    return 'shifted / reranked / plain'


# ── gather and WindowBatch ────────────────────────────────────────────────────

def t_gather_windows_is_the_slice() -> str:
    grid: torch.Tensor = unit(9, 11, 6, seed=1)
    rows, cols = torch.tensor([0, 3, 4]), torch.tensor([0, 5, 6])
    got, valid = gather_windows(grid, rows, cols, 4, 5)
    assert valid is None and got.shape == (3, 4, 5, 6)
    for g in range(3):
        assert torch.equal(got[g], grid[rows[g]:rows[g] + 4, cols[g]:cols[g] + 5])
    return 'three windows'


def t_gather_windows_margin_says_where_nothing_is() -> str:
    grid: torch.Tensor = unit(9, 11, 6, seed=2)
    rows, cols = torch.tensor([0, 5]), torch.tensor([0, 6])
    got, valid = gather_windows(grid, rows, cols, 4, 5, margin=1)
    assert got.shape == (2, 6, 7, 6) and valid is not None and valid.shape == (2, 6, 7)
    assert not valid[0, 0].any() and not valid[0, :, 0].any(), 'row -1 and col -1 are off the grid'
    assert valid[0, 1:, 1:].all()
    assert torch.equal(got[0, 1, 1], grid[0, 0])
    # the second window ends on the grid's last row and column: the margin
    # past them is nothing, the rest is the grid
    assert valid[1, :5, :6].all() and not valid[1, 5].any() and not valid[1, :, 6].any()
    assert torch.equal(got[1, :5, :6], grid[4:9, 5:11])
    return 'margin 1'


def t_window_batch_reproduces_the_first_phase_score() -> str:
    src, ref = make_source()
    rng: np.random.Generator = np.random.default_rng(0)
    cands: List[Candidate] = []
    for _ in range(60):
        rot: int = int(rng.choice([0, 90, 180, 270]))
        qr, qc = (4, 5) if rot in (0, 180) else (5, 4)
        region: int = int(rng.integers(0, 2))
        lattice: str = str(rng.choice(['main', 'offset']))
        R, C = ref[(region, lattice)].shape[:2]
        cands.append(Candidate(region, lattice, int(rng.integers(0, R - qr + 1)),
                               int(rng.integers(0, C - qc + 1)), rot, 0.0))
    cs: CandidateSet = cset(cands)
    batch: WindowBatch = WindowBatch.build(cs, src, 'cls')
    assert sorted(g.shape for g in batch.groups) == [(4, 5), (5, 4)]
    got: torch.Tensor = batch.mean_cos()
    for k, c in enumerate(cands):
        qg: torch.Tensor = src.queries[(c.rotation, 'main')]
        g: torch.Tensor = ref[(c.region_index, c.lattice)]
        want: float = float(np.mean([float(qg[i, j] @ g[c.row + i, c.col + j])
                                     for i in range(qg.shape[0]) for j in range(qg.shape[1])]))
        assert abs(float(got[k]) - want) < 1e-5, (k, float(got[k]), want)
    return '60 windows, 4 rotations, 2 lattices, 2 regions, 2 shapes'


def t_window_batch_on_a_subset_and_with_a_margin() -> str:
    src, ref = make_source()
    cands: List[Candidate] = [Candidate(0, 'main', 2, 3, 0, 0.0), Candidate(1, 'offset', 1, 1, 90, 0.0),
                              Candidate(0, 'main', 0, 0, 180, 0.0)]
    cs: CandidateSet = cset(cands)
    whole: WindowBatch = WindowBatch.build(cs, src, 'cls')
    part: WindowBatch = WindowBatch.build(cs, src, 'cls', candidates=cands[1:])
    assert len(part) == 2 and torch.allclose(part.mean_cos(), whole.mean_cos()[1:], atol=1e-6)
    ctx: WindowBatch = WindowBatch.build(cs, src, 'cls', margin=1)
    assert torch.allclose(ctx.mean_cos(), whole.mean_cos(), atol=1e-6), 'a margin is around the window, not in it'
    first: object = next(g for g in ctx.groups if g.shape == (4, 5))
    assert first.w.shape[1:3] == (6, 7) and first.valid is not None and first.inner.shape[1:3] == (4, 5)
    return 'subset and margin leave the score alone'


# ── sources ───────────────────────────────────────────────────────────────────

class _Reader:
    """`read_points` as SlideReader gives it: batches of uint8 tiles and the
    index of each, here a constant image whose value is a function of the
    position, so what the encoder sees says where the tile was read."""

    def __init__(self) -> None:
        self.reads: int = 0

    def read_points(self, points, spec, ds, *, level=None, batch=64):
        for s in range(0, len(points), batch):
            chunk = points[s:s + batch]
            self.reads += len(chunk)
            imgs = torch.stack([torch.full((spec.sensor_h, spec.sensor_w, 3),
                                           (x // 7 + y // 11) % 251, dtype=torch.uint8)
                                for x, y in chunk])
            yield imgs, torch.arange(s, s + len(chunk))


class _Encoder:
    feature_pooling: str = 'cls'

    def features(self, images: torch.Tensor) -> torch.Tensor:
        v: torch.Tensor = images[:, 0, 0, 0].float()
        return F.normalize(torch.stack([v, v * v, torch.ones_like(v)], dim=1), dim=-1)

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(images), 5, 3)


def t_a_tile_asked_for_twice_is_encoded_once() -> str:
    reader, enc = _Reader(), _Encoder()
    tiles: OnDemandTiles = OnDemandTiles(reader, enc, level=1, ds=2.0, tile_size=TILE, capacity=100)
    a: torch.Tensor = tiles.vectors([(0, 0), (700, 0), (0, 900)], 'cls')
    assert tiles.encoded == 3 and reader.reads == 3 and a.shape == (3, 3)
    b: torch.Tensor = tiles.vectors([(700, 0), (0, 0), (333, 333)], 'cls')
    assert tiles.encoded == 4 and reader.reads == 4, (tiles.encoded, reader.reads)
    assert torch.equal(b[0], a[1]) and torch.equal(b[1], a[0])
    assert tiles.raw([(0, 0)]).shape == (1, 5, 3) and tiles.encoded == 5
    return '4 encodes for 6 asked'


def t_the_tile_cache_forgets_the_oldest() -> str:
    reader, enc = _Reader(), _Encoder()
    tiles: OnDemandTiles = OnDemandTiles(reader, enc, level=0, ds=1.0, tile_size=TILE, capacity=2)
    tiles.vectors([(0, 0)], 'cls')
    tiles.vectors([(100, 0)], 'cls')
    tiles.vectors([(200, 0)], 'cls')
    tiles.vectors([(0, 0)], 'cls')
    assert tiles.encoded == 4, 'the first was out when it came back'
    return 'capacity 2'


def t_a_call_for_more_tiles_than_the_cache_holds_still_gets_them_all() -> str:
    reader, enc = _Reader(), _Encoder()
    tiles: OnDemandTiles = OnDemandTiles(reader, enc, level=0, ds=1.0, tile_size=TILE, capacity=3)
    origins: List[Tuple[int, int]] = [(100 * i, 0) for i in range(10)]
    got: torch.Tensor = tiles.vectors(origins, 'cls')                       # used to raise KeyError
    assert got.shape == (10, 3) and tiles.encoded == 10
    again: torch.Tensor = tiles.vectors(origins[-2:] + origins[:1], 'cls')  # the last two are kept, the first is not
    assert torch.equal(again[0], got[8]) and torch.equal(again[1], got[9]) and torch.equal(again[2], got[0])
    assert tiles.encoded == 11, tiles.encoded
    return '10 tiles through a cache of 3'


def t_tiles_the_cache_held_are_not_lost_to_the_new_ones_of_the_same_call() -> str:
    reader, enc = _Reader(), _Encoder()
    tiles: OnDemandTiles = OnDemandTiles(reader, enc, level=0, ds=1.0, tile_size=TILE, capacity=3)
    old: List[Tuple[int, int]] = [(0, 0), (100, 0)]
    first: torch.Tensor = tiles.vectors(old, 'cls')                          # both kept, the oldest first
    new: List[Tuple[int, int]] = [(200 + 100 * i, 0) for i in range(5)]
    got: torch.Tensor = tiles.vectors(old + new, 'cls')                      # five new tiles push the two out
    assert got.shape == (7, 3) and torch.equal(got[0], first[0]) and torch.equal(got[1], first[1])
    assert tiles.encoded == 7, tiles.encoded
    return 'hits of a call survive the call'


def t_memory_source_with_no_reader_says_so() -> str:
    src, _ = make_source()
    try:
        src.encode_tiles(torch.tensor([[0, 0]]), 'cls')
    except RuntimeError:
        return 'refuses'
    raise AssertionError('a source with no reader encoded tiles')


# ── boxes ─────────────────────────────────────────────────────────────────────

def t_window_boxes_and_iou() -> str:
    cs: CandidateSet = cset([Candidate(0, 'main', 2, 3, 0, 1.0), Candidate(0, 'main', 2, 3, 90, 1.0),
                             ShiftedCandidate(0, 'main', 2, 3, 0, 1.0, shift=(64, 0))], ds=2.0)
    shape = lambda rot: (4, 5) if rot in (0, 180) else (5, 4)          # noqa: E731
    b: np.ndarray = window_boxes(cs, shape)
    x0, y0 = 3 * TILE * 2.0, 2 * TILE * 2.0
    assert np.allclose(b[0], [x0, y0, x0 + 5 * TILE * 2, y0 + 4 * TILE * 2])
    assert np.allclose(b[1], [x0, y0, x0 + 4 * TILE * 2, y0 + 5 * TILE * 2]), 'turned, the sides swap'
    assert np.allclose(b[2][:2], [x0 + 128, y0])
    iou: np.ndarray = iou_matrix(b, b)
    assert np.allclose(np.diag(iou), 1.0) and iou[0, 1] < 1.0
    assert abs(iou[0, 2] - (5 * 512 - 128) / (5 * 512 + 128)) < 1e-9, iou[0, 2]
    far: np.ndarray = np.array([[1e6, 1e6, 1e6 + 10, 1e6 + 10]])
    assert iou_matrix(b, far).sum() == 0 and box_area(far)[0] == 100
    return 'identical 1, shifted by 128, disjoint 0'


# ── the greedy pass ───────────────────────────────────────────────────────────

def _list_scan(origin: np.ndarray, rotation: np.ndarray, min_sep: float, k: int) -> List[int]:
    """The pass as it was written before: a list of kept candidates scanned in
    Python for every candidate."""
    kept: List[int] = []
    for i in range(len(origin)):
        if any(rotation[j] == rotation[i]
               and math.hypot(origin[i, 0] - origin[j, 0], origin[i, 1] - origin[j, 1]) < min_sep
               for j in kept):
            continue
        kept.append(i)
        if len(kept) == k:
            break
    return kept


def t_separated_keeps_what_the_list_scan_kept() -> str:
    rng: np.random.Generator = np.random.default_rng(3)
    n_cases: int = 0
    for trial in range(40):
        n: int = int(rng.integers(1, 400))
        origin: np.ndarray = rng.integers(0, 30, size=(n, 2)).astype(np.float64) * rng.choice([64.0, 128.0, 256.0])
        rotation: np.ndarray = rng.choice([0, 90, 180, 270], size=n)
        for min_sep, k in ((256.0, 20), (512.0, 500), (1.0, 5), (0.5, 10_000)):
            assert separated(origin, rotation, min_sep, k) == _list_scan(origin, rotation, min_sep, k), \
                (trial, min_sep, k)
            n_cases += 1
    return f'{n_cases} cases, ties and repeats included'


def t_separated_is_per_rotation() -> str:
    origin: np.ndarray = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [5000.0, 0.0]])
    rotation: np.ndarray = np.array([0, 0, 90, 0])
    assert separated(origin, rotation, 256.0, 10) == [0, 2, 3]
    assert separated(origin, rotation, 256.0, 2) == [0, 2]
    assert separated(np.zeros((0, 2)), np.zeros(0, dtype=np.int64), 256.0, 5) == []
    return 'the same spot at another rotation is another hypothesis'


# ── Retriever.py: the interface ───────────────────────────────────────────────

class _Est:
    """Stands in for stage 1's output."""
    chosen_level: int = 0


class _Lister(Retriever):
    """A first-phase retriever that lists `n` fixed windows and offers one extra."""
    UPSTREAM = (_Est,)

    def __init__(self, n: int, raw_topk: int, key: Tuple[Any, ...] = ('a',)) -> None:
        super().__init__(SimpleNamespace(raw_topk=raw_topk), 'cpu')
        self.n: int = n
        self.key: Tuple[Any, ...] = key

    def run(self, upstream: Any, query: Any) -> RetrievalOutput:
        cands: List[Candidate] = [Candidate(0, 'main', i, 0, 0, 1.0 - 0.01 * i) for i in range(self.n)]
        return RetrievalOutput(candidates=cset(cands), extras={'feat': Extra('feat', self.key, 'payload')})


def t_retriever_checks_its_upstream_and_the_consumers_topk() -> str:
    r: _Lister = _Lister(n=30, raw_topk=50)
    try:
        r.retrieve(_Est(), None)
        raise AssertionError('a retriever that was never built ran')
    except RuntimeError:
        pass
    r.build(SimpleNamespace(), None)
    out: RetrievalOutput = r.retrieve(_Est(), None)
    assert len(out.candidates) == 30, 'fewer windows than raw_topk: all of them, the config unchanged'
    ten: RetrievalOutput = r.retrieve(_Est(), None, topk=10)
    assert [c.key() for c in ten.candidates] == [c.key() for c in out.candidates][:10], 'topk is a prefix'
    assert ten.extras == out.extras, 'a cut keeps the extras'
    for bad, kind in ((lambda: r.retrieve(out, None), TypeError),
                      (lambda: r.retrieve(_Est(), None, topk=51), ValueError)):
        try:
            bad()
            raise AssertionError(f'expected {kind.__name__}')
        except kind:
            pass
    return 'unbuilt, wrong upstream, topk > raw_topk all refused; topk a prefix'


def t_the_bridge_lock_takes_the_matching_extra_or_nothing_and_refuses_another() -> str:
    first: RetrievalOutput = RetrievalOutput(candidates=cset([]), extras={'feat': Extra('feat', ('uni2', 'cls'), 1)})
    second: RetrievalOutput = RetrievalOutput(candidates=cset([]), parent=first,
                                              extras={'feat': Extra('feat', ('uni2', 'cls_avg'), 2)})
    assert [e.value for e in second.find('feat')] == [2, 1], 'nearest first'
    assert take(second, Uses('feat', ('uni2', 'cls'))) == 1, 'a farther match is found'
    assert take(second, Uses('feat', ('uni2', 'cls_avg'))) == 2
    assert take(second, Uses('cols', ('x',))) is None, 'nothing of that kind: the retriever makes its own'
    assert take(_Est(), Uses('feat', ('uni2', 'cls'))) is None, 'stage 1 offers nothing'
    read_back: RetrievalOutput = RetrievalOutput(candidates=cset([]))
    assert take(read_back, Uses('feat', ('uni2', 'cls'))) is None, 'a list read back from a cache offers nothing'
    try:
        take(second, Uses('feat', ('gigapath', 'cls')))
        raise AssertionError('another key was taken')
    except BridgeError:
        pass
    return 'match: taken; none offered: made by itself; another key: BridgeError'


def _enc_cfg(name: str, pooling: str = '', dtype: str = 'fp16') -> Any:
    from TileEncoderFunc import encoder_config                   # noqa: PLC0415
    return encoder_config(name, pooling=pooling).with_model(dtype=dtype)


class _Built:
    """A loaded encoder: its config, its model, the id of its weights."""

    def __init__(self, cfg: Any, weights_id: str) -> None:
        self.cfg: Any = cfg
        self.model: object = object()
        self.weights_id: str = weights_id


def t_an_encoder_is_shared_by_trunk_and_never_substituted() -> str:
    uni_cls, uni_avg = _enc_cfg('uni2', 'cls'), _enc_cfg('uni2', 'cls_avg')
    assert trunk_id(uni_cls) == trunk_id(uni_avg), 'the pooling is not the trunk'
    assert trunk_id(uni_cls) != trunk_id(_enc_cfg('uni2', 'cls', 'fp32')), 'the precision is'
    built: _Built = _Built(uni_cls, 'w1')
    check_encoder(built, uni_avg)
    for other in (_enc_cfg('gigapath'), _enc_cfg('uni2', 'cls', 'fp32')):
        try:
            check_encoder(built, other)
            raise AssertionError('another trunk was accepted')
        except ValueError:
            pass
    pool: EncoderPool = EncoderPool('cpu')
    pool.offer(built)
    view: Any = pool.get(uni_avg)
    assert view.model is built.model and view.cfg.pooling == 'cls_avg', 'one loaded model, read through cls_avg'
    assert pool.get(uni_cls) is built
    try:
        pool.offer(_Built(uni_cls, 'w2'))
        raise AssertionError('a second model under one trunk was accepted')
    except ValueError:
        pass
    return 'cls and cls_avg share one model; another trunk or other weights refused'


class _Slide:
    """`level_downsamples` and a level-0 size: what the geometry reads of a slide with no mask."""
    level_downsamples: Tuple[float, ...] = (1.0, 4.0, 16.0)
    level_dimensions: Tuple[Tuple[int, int], ...] = ((4096, 4096), (1024, 1024), (256, 256))


def t_route_walks_to_a_finer_level_and_remembers_an_unusable_one() -> str:
    levels: SlideLevels = SlideLevels(_Slide(), None, TILE, True)
    query: np.ndarray = np.zeros((768, 768, 3), dtype=np.uint8)            # 3 x 3 tiles
    dims: Dict[int, Tuple[int, int]] = levels.query_dims(query)
    assert dims[0] == (3, 3) and not levels.fitting_rotations(2, dims) and levels.fitting_rotations(1, dims)
    prepared: List[int] = []

    def prepare(level: int) -> None:
        prepared.append(level)
        if level == 1:
            raise OSError('unreadable')

    def search(level: int) -> CandidateSet:
        return replace(levels.frame(level), candidates=(Candidate(0, 'main', 0, 0, 0, 1.0),))

    got: CandidateSet = levels.route(2, dims, prepare, search)
    assert got.level == 0 and got.irretrievable_lvl == (1, 2) and got.alter_lvl == (1, 0), got
    levels.route(2, dims, prepare, search)
    assert prepared == [1, 0, 0], 'a level that raised is not prepared again'
    none: CandidateSet = SlideLevels(_Slide(), None, TILE, True).route(
        2, dims, lambda level: None, lambda level: levels.frame(level))
    assert not len(none) and none.level == 2 and none.irretrievable_lvl == (0, 1, 2)
    return 'L2 too small, L1 unreadable, L0 searched; L1 not read again'


def t_nearest_window_is_geometry() -> str:
    levels: SlideLevels = SlideLevels(_Slide(), None, TILE, True)
    frame: CandidateSet = levels.frame(0)
    grid: Any = frame.grids[0]
    rows_q, cols_q = 3, 4
    for lattice, (i, j) in (('main', (2, 3)), ('offset', (5, 1))):
        x, y = grid.tile_origin_l0(lattice, i, j)
        got = nearest_window(frame, (rows_q, cols_q), x + cols_q * TILE / 2.0, y + rows_q * TILE / 2.0, 0)
        assert got is not None and got[:4] == (0, lattice, i, j) and got[4] == 0.0, got
    main_only = nearest_window(frame, (rows_q, cols_q), *grid.tile_origin_l0('offset', 5, 1), 0, ('main',))
    assert main_only is not None and main_only[1] == 'main'
    assert nearest_window(levels.frame(2), (3, 3), 0.0, 0.0, 0) is None, 'no region holds the window'
    return 'the window whose centre is the point, on either lattice'


def t_a_grid_source_reads_what_a_memory_source_holding_the_same_tensors_reads() -> str:
    mem, ref = make_source()
    level: LevelFeatures = LevelFeatures(level=1, ds=2.0, regions=[], grids=[], features=None,
                                         tensors=[(ref[(r, 'main')], ref[(r, 'offset')]) for r in range(2)])
    qmaps: Dict[int, Any] = {rot: SimpleNamespace(main_feature_grid=lambda rot=rot: mem.queries[(rot, 'main')],
                                                  overlap_feature_grid=lambda rot=rot: mem.queries[(rot, 'offset')])
                             for rot in (0, 90, 180, 270)}
    grid: GridSource = GridSource(level, QueryFeatures(containers={}, features=qmaps), 'cls')
    cands: List[Candidate] = [Candidate(0, 'main', 1, 2, 0, 0.0), Candidate(1, 'offset', 3, 1, 90, 0.0)]
    a: WindowBatch = WindowBatch.build(cset(cands), grid, 'cls', margin=1)
    b: WindowBatch = WindowBatch.build(cset(cands), mem, 'cls', margin=1)
    assert torch.equal(a.mean_cos(), b.mean_cos())
    assert torch.equal(a.groups[0].w, b.groups[0].w)
    try:
        grid.reference_grid(0, 'main', 'cls_avg')
        raise AssertionError('another pooling was read')
    except KeyError:
        pass
    return 'same windows, same cosines; another pooling refused'


# ── the sliding window ────────────────────────────────────────────────────────

def t_the_sliding_window_scores_every_window_by_its_tile_cosines_and_lists_the_best() -> str:
    from TileEncoderFunc import encoder_config                   # noqa: PLC0415
    from stage2_retrieval.RetrieverSlidingWinSimRot import (GEOMEAN_FLOOR,   # noqa: PLC0415
                                                            RetrieverSlidingWinSimRotConfig, list_windows)
    mem, ref = make_source(dim=16)
    tensors: List[Tuple[torch.Tensor, torch.Tensor]] = [(ref[(r, 'main')], ref[(r, 'offset')]) for r in range(2)]
    qmaps: Dict[int, Any] = {rot: SimpleNamespace(main_feature_grid=lambda rot=rot: mem.queries[(rot, 'main')])
                             for rot in (0, 90, 180, 270)}
    frame: CandidateSet = cset([])
    enc: Any = encoder_config('uni2').with_model(dtype='fp16')
    reduce = {'mean': lambda c: c.mean(), 'min': lambda c: c.min(),
              'geomean': lambda c: torch.exp(torch.log(c.clamp_min(GEOMEAN_FLOOR)).mean())}
    checked: int = 0
    for score, fn in reduce.items():
        for k, min_sep in ((7, 1.0), (40, 2.0), (25, 0.0)):
            cs, sims, scores = list_windows(
                LevelFeatures(level=1, ds=frame.ds, regions=[], grids=list(frame.grids), features=None,
                              tensors=tensors),
                QueryFeatures(containers={}, features=qmaps), frame,
                RetrieverSlidingWinSimRotConfig(encoder=enc, tile_size=TILE, overlap=True, raw_topk=k,
                                                min_sep_tiles=min_sep, score=score))
            assert len(scores) == 16, 'every rotation x region x lattice has its map'
            # a few windows' scores, against the loop over their tiles
            for (rot, region, lattice), m in list(scores.items())[::5]:
                q: torch.Tensor = mem.queries[(rot, 'main')]
                grid: torch.Tensor = ref[(region, lattice)]
                for (i, j) in ((0, 0), (m.shape[0] - 1, m.shape[1] - 1)):
                    cos: torch.Tensor = (grid[i:i + q.shape[0], j:j + q.shape[1]] * q).sum(-1)
                    assert abs(float(fn(cos)) - float(m[i, j])) < 1e-5, (score, rot, region, lattice, i, j)
                    checked += 1
            listed: List[Candidate] = list(cs.candidates)
            assert len(listed) <= k and (len(listed) == k or min_sep > 0)
            assert listed[0].score == max(float(m.max()) for m in scores.values())
            assert [c.score for c in listed] == sorted((c.score for c in listed), reverse=True)
            sep: float = min_sep * TILE * frame.ds
            for a in range(len(listed)):
                for b in range(a):
                    if listed[a].rotation == listed[b].rotation and sep > 0:
                        pa, pb = cset([]).origin_l0(listed[a]), cset([]).origin_l0(listed[b])
                        assert math.hypot(pa[0] - pb[0], pa[1] - pb[1]) >= sep, 'kept apart at one rotation'
    return f'{checked} window scores against the loops; lists best first, separated per rotation'


_TESTS = [t for n, t in sorted(globals().items(), key=lambda kv: kv[0])
          if n.startswith('t_')]


def main() -> int:
    for fn in _TESTS:
        check(fn.__name__[2:].replace('_', ' '), fn)
    failed: List[str] = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
