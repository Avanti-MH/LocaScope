'''The items of a FoV: everything its softmax could be over, whatever scores them (`stage2_retrieval/proposal.md` §9.5).

An item is a window of the level at one rotation. Where the items come from is `--items` (or a head's `@items=`):

    level   every window of the level at the four rotations (of each rotation's shape), flat: the windows of the rotations
            one after the other, each in the order of `Level.universe`. The retrieval's candidates.
    list    the entries of the list another retriever gave (`Fov.listing`, `--raw-topk` of them), in its order: the
            first phase's, where the windows an answer needs have already been picked out. The reranking's candidates.

The labels are `Supply.mark`. On a LIST (`listed`) an answer may be missing and the windows that kept its place are the
positives; when NO window of the list is a positive, the answers' own windows are ADDED to the items (`injected`): a FoV
is not left out for what its list lacks, the scorer is still trained to put the answer ahead of the list's hard
negatives. (A level has every window, so its answers are always there.)

`Items` is the one shape both kinds of scorer read: a bi-encoder (GloRet, `Full.py`) takes a pool of them as the window
ids of each rotation (`pool_of`), a pair scorer (LogLoRet, `Rerank.py`) as `Cands` (`cands_of`).
'''
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from training.GloRet.Supply import (ROTATIONS, Cands, Fov, Level, Marks, Part, Universe, decode_code, mark,
                                    marks_of)

#: a FoV's pool for a bi-encoder: per rotation the sorted ids of the windows (`Universe` numbering) of that rotation's shape
Pool = Dict[int, np.ndarray]

SOURCES: Tuple[str, ...] = ('level', 'list')


def window_marks(level: Level, f: Fov, rot: int, cache: Dict[Any, Any]) -> Optional[Marks]:
    """The marks of every window of the shape the FoV has at `rot`, in the order of the universe; None when the level
    has no window of that shape. Kept in `cache` (by FoV and rotation) for the steps of the visit."""
    key: Tuple[int, int] = (id(f), rot)
    if key in cache:
        return cache[key]
    uni: Universe = level.universe(f.shape(rot))
    if not uni.total:
        return None
    origin, code = uni.origins(np.arange(uni.total, dtype=np.int64))
    a, b = uni.shape
    size: np.ndarray = np.tile(np.array([b * level.tile_l0, a * level.tile_l0], dtype=np.float64), (uni.total, 1))
    cache[key] = mark(code, np.full(uni.total, rot, dtype=np.int64), origin, size, f.placed, level.tile_l0)
    return cache[key]


@dataclass(eq=False)
class Items:
    """The items of one FoV, flat, with their marks and the teacher's score of each (the score another retriever gave
    them: the first phase's window scores for a level, the list's own order for a list; None when it made none)."""
    source: str
    level: Level
    fov: Fov
    rot: np.ndarray                 # [n] rotation of each item
    pos: np.ndarray
    ignore: np.ndarray
    right: np.ndarray
    dist: np.ndarray
    teacher: Optional[np.ndarray]
    injected: int = 0               # answers added to a list that held none (the last `injected` items)
    cands: Optional[Cands] = None   # a list: its entries (and the answers added), in item order
    wid: Optional[np.ndarray] = None            # [n] the item's id in the universe of its rotation's shape
    dense: Dict[int, Optional[Marks]] = field(default_factory=dict)    # per rotation, over every window of the universe

    @property
    def n(self) -> int:
        return int(len(self.pos))

    # ── for a bi-encoder ────────────────────────────────────────────────────────────────────────────────────────────
    def window_ids(self) -> np.ndarray:
        """`[n]` the universe id of each item's window (-1: the level has no such window)."""
        if self.wid is None:
            assert self.cands is not None
            wid: np.ndarray = np.full(self.n, -1, dtype=np.int64)
            for rot in ROTATIONS:
                at: np.ndarray = np.flatnonzero(self.rot == rot)
                if not len(at):
                    continue
                uni: Universe = self.level.universe(self.fov.shape(rot))
                wid[at] = [uni.id_of(self.cands.part[j], int(self.cands.row[j]), int(self.cands.col[j])) for j in at]
            self.wid = wid
        return self.wid

    def pool_of(self, flat: np.ndarray) -> Pool:
        """The pool whose items are the flat indices `flat`: per rotation the sorted window ids."""
        wid: np.ndarray = self.window_ids()
        flat = np.asarray(flat, dtype=np.int64)
        out: Pool = {}
        for rot in ROTATIONS:
            ids: np.ndarray = wid[flat[self.rot[flat] == rot]]
            out[rot] = np.unique(ids[ids >= 0]).astype(np.int64)
        return out

    def everything(self) -> Pool:
        return self.pool_of(np.arange(self.n, dtype=np.int64))

    def dense_marks(self, rot: int) -> Optional[Marks]:
        """The marks over EVERY window of the universe of the shape at `rot` (those that are not items are never read)."""
        if rot not in self.dense:
            uni: Universe = self.level.universe(self.fov.shape(rot))
            if not uni.total:
                self.dense[rot] = None
            else:
                at: np.ndarray = np.flatnonzero(self.rot == rot)
                w: np.ndarray = self.window_ids()[at]
                at, w = at[w >= 0], w[w >= 0]
                arrays: Dict[str, np.ndarray] = {
                    'pos': np.zeros(uni.total, dtype=bool), 'ignore': np.zeros(uni.total, dtype=bool),
                    'right': np.zeros(uni.total, dtype=bool), 'dist': np.zeros(uni.total, dtype=np.float32)}
                for name in arrays:
                    arrays[name][w] = getattr(self, name)[at]
                self.dense[rot] = Marks(**arrays)
        return self.dense[rot]

    # ── for a pair scorer ───────────────────────────────────────────────────────────────────────────────────────────
    def cands_of(self, flat: np.ndarray) -> Cands:
        """The items `flat` as `Cands` (window, rotation, place), in that order."""
        flat = np.asarray(flat, dtype=np.int64)
        if self.cands is not None:
            return self.cands.take(flat)
        rot: np.ndarray = self.rot[flat]
        wid: np.ndarray = self.window_ids()[flat]
        parts: np.ndarray = np.empty(len(flat), dtype=object)                # a (region, lattice) tuple each
        row: np.ndarray = np.zeros(len(flat), dtype=np.int64)
        col: np.ndarray = np.zeros(len(flat), dtype=np.int64)
        origin: np.ndarray = np.zeros((len(flat), 2), dtype=np.float64)
        for r in ROTATIONS:
            at: np.ndarray = np.flatnonzero(rot == r)
            if not len(at):
                continue
            uni: Universe = self.level.universe(self.fov.shape(r))
            p, rr, cc = uni.locate(wid[at])
            org, _ = uni.origins(wid[at])
            table: np.ndarray = np.empty(len(uni.parts), dtype=object)       # the universe's parts as an object array
            for k, part in enumerate(uni.parts):
                table[k] = part
            parts[at] = table[p]
            row[at], col[at], origin[at] = rr, cc, org
        return Cands(parts.tolist(), row, col, rot.astype(np.int64), origin, np.zeros(len(flat)))      # type: ignore[arg-type]

    def take(self, flat: np.ndarray) -> Marks:
        flat = np.asarray(flat, dtype=np.int64)
        return Marks(self.pos[flat], self.ignore[flat], self.dist[flat], self.right[flat])


# ══════════════════════════════════════════════════════════════════════════════
#  making them
# ══════════════════════════════════════════════════════════════════════════════

def level_items(level: Level, f: Fov, cache: Dict[Any, Any]) -> Optional[Items]:
    """Every window of the level at the four rotations; kept in `cache`; None when none of them is an answer."""
    key: Tuple[int, str] = (id(f), 'level-items')
    if key in cache:
        return cache[key]
    rots: List[np.ndarray] = []
    wids: List[np.ndarray] = []
    parts: Dict[str, List[np.ndarray]] = {'pos': [], 'ignore': [], 'right': [], 'dist': []}
    sizes: Dict[int, int] = {}
    dense: Dict[int, Optional[Marks]] = {}
    for rot in ROTATIONS:
        mk: Optional[Marks] = window_marks(level, f, rot, cache)
        dense[rot] = mk
        size: int = 0 if mk is None else int(len(mk.pos))
        sizes[rot] = size
        rots.append(np.full(size, rot, dtype=np.int64))
        wids.append(np.arange(size, dtype=np.int64))
        if mk is not None:
            parts['pos'].append(mk.pos)
            parts['ignore'].append(mk.ignore)
            parts['right'].append(mk.right)
            parts['dist'].append(mk.dist.astype(np.float32))
    cat: Dict[str, np.ndarray] = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in parts.items()}
    teacher: Optional[np.ndarray] = None
    if f.teacher is not None and all(len(f.teacher[rot]) == sizes[rot] for rot in ROTATIONS):
        teacher = np.concatenate([f.teacher[rot] for rot in ROTATIONS]).astype(np.float32)
    items: Optional[Items] = Items(
        source='level', level=level, fov=f, rot=np.concatenate(rots), pos=cat['pos'].astype(bool),
        ignore=cat['ignore'].astype(bool), right=cat['right'].astype(bool), dist=cat['dist'].astype(np.float32),
        teacher=teacher, wid=np.concatenate(wids), dense=dense)
    if not items.pos.any():
        items = None
    cache[key] = items
    return items


def answer_cands(level: Level, f: Fov) -> Cands:
    """The answers' own windows (at the right rotation) that the level has, as `Cands`."""
    shape = f.shape(f.placed.rot)
    uni: Universe = level.universe(shape)
    parts: List[Part] = []
    rows: List[int] = []
    cols: List[int] = []
    origins: List[Tuple[float, float]] = []
    for code, (ax, ay) in f.placed.answers:
        part, row, col = decode_code(code)
        if uni.id_of(part, row, col) < 0:
            continue
        parts.append(part)
        rows.append(row)
        cols.append(col)
        origins.append((ax, ay))
    return Cands(parts, np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
                 np.full(len(parts), f.placed.rot, dtype=np.int64), np.array(origins, dtype=np.float64).reshape(-1, 2),
                 np.zeros(len(parts)))


def list_items(level: Level, f: Fov, cache: Dict[Any, Any], limit: Optional[int] = None,
               inject: bool = True) -> Optional[Items]:
    """The entries of a FoV's list (its first `limit`), kept in `cache`. When no entry is a positive the answers' own
    windows are added (`inject`; a level whose windows cannot be named, a lazy one, cannot); None when there is still no
    positive."""
    key: Tuple[int, str] = (id(f), 'list-items')
    if key in cache:
        return cache[key]
    cands: Cands = f.listing if limit is None or len(f.listing) <= limit else f.listing.take(np.arange(limit))
    marks: Optional[Marks] = marks_of(cands, f, level.tile_l0, listed=True) if len(cands) else None
    injected: int = 0
    if (marks is None or not marks.pos.any()) and inject:
        extra: Cands = answer_cands(level, f)
        if len(extra):
            cands = Cands.join(cands, extra) if len(cands) else extra
            marks = marks_of(cands, f, level.tile_l0, listed=True)
            injected = len(extra)
    items: Optional[Items] = None
    if marks is not None and marks.pos.any():
        n: int = len(cands)
        items = Items(source='list', level=level, fov=f, rot=cands.rot.astype(np.int64), pos=marks.pos, ignore=marks.ignore,
                      right=marks.right, dist=marks.dist.astype(np.float32), teacher=-np.arange(n, dtype=np.float64),
                      injected=injected, cands=cands)
    cache[key] = items
    return items


def items_of(source: str, level: Level, f: Fov, cache: Dict[Any, Any], limit: Optional[int] = None,
             inject: bool = True) -> Optional[Items]:
    """`level_items` or `list_items`, by `source`."""
    if source == 'level':
        return level_items(level, f, cache)
    if source == 'list':
        return list_items(level, f, cache, limit, inject)
    raise ValueError(f'items {source!r}: one of {SOURCES}')


__all__ = ['Items', 'Pool', 'SOURCES', 'answer_cands', 'items_of', 'level_items', 'list_items', 'window_marks']
