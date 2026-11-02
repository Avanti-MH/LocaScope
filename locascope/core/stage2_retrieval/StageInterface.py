'''Stage 2 (retrieval)'s candidates: what every retriever's list is made of, and
the frame it is expressed in. The retriever itself is `Retriever.py`; the
stage-1 counterpart is `stage1_estimation/StageInterface.py`; spec.md in this
directory is the design.

Each stage eats the previous stage's output:

    stage 1   est.estimate(query)                  -> EstMppResult
    stage 2   ret.retrieve(EstMppResult, query)    -> RetrievalOutput, its .candidates a CandidateSet
    stage 3   loc.localize(query, CandidateSet, topk) -> LocalizationResultSet
              (stage3_localization/StageInterface.py)

A `Candidate` holds only what retrieval found -- which window, at which
rotation, how strongly. Everything derivable (pixel position, window size,
rank) is left out and derived through the `CandidateSet` it came in, which
carries the frame those indices mean something in: the regions' grids, the
level and the ds. One formula each, so the bench, the figures, the pipeline
and stage 3 cannot disagree about where a candidate is.

POSITIONS ARE LEVEL-0, AND FRACTIONAL WHERE THEY ARE NOT A READ ORIGIN.
openslide samples level n at x / ds with the fraction interpolated (measured
in diag_read_exp.py's phase flow), so a level-n position written as
int(region.x / ds) is off by that fraction of a level pixel -- about 1 um at
BRACS level 1, 4 um at level 2. `origin_l0` is the level-0 integer a read of
the window starts at, `PatchGrid.tile_origin_l0`; nothing here truncates.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:       # annotations only: nothing here is imported at run time
    from PatchingLib import PatchGrid


#: The rotations a retriever tries, in degrees; the query is turned by these
#: before matching, so a 90/270 window has the query's rows and cols swapped.
ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class Candidate:
    '''One scored window. Identity is the first five fields: two methods
    found the same window exactly when those five are equal.

    `row`, `col` are the window's top-left tile on `lattice` of region
    `region_index`'s grid; `lattice` is 'main' or 'offset' (the main lattice
    shifted half a tile on both axes).'''
    region_index: int
    lattice: str
    row: int
    col: int
    rotation: int
    score: float

    def key(self) -> Tuple[int, str, int, int, int]:
        return (self.region_index, self.lattice, self.row, self.col, self.rotation)

    def shift_px(self) -> Tuple[int, int]:
        '''(dx, dy) in LEVEL px that a refining method moved the window off its
        lattice origin; (0, 0) for a window that sits on its lattice.
        `CandidateSet.origin_l0` adds it, so the one formula for a position
        stays one formula.'''
        return (0, 0)

    def extra(self) -> Dict[str, Any]:
        '''What this kind of candidate carries beyond the six fields, as table
        columns. A subclass adds fields for what only its method knows and
        names them here; the base class has none.'''
        return {}


@dataclass(frozen=True)
class RerankedCandidate(Candidate):
    '''A candidate a reranker handed back. `score` is the last phase's;
    `first_score` and `first_rank` are what the FIRST phase (the retriever,
    a union included) gave this window -- not an intermediate reranker's -- so
    a caller can tell a window the reranker pulled up from one the retriever
    already had on top. Defaults only so subclasses can add fields after them.'''
    first_score: float = 0.0
    first_rank: int = 0

    def extra(self) -> Dict[str, Any]:
        return dict(first_score=self.first_score, first_rank=self.first_rank)


@dataclass(frozen=True)
class ShiftedCandidate(RerankedCandidate):
    '''A window moved off its lattice by `shift` = (dx, dy) level px (a
    sub-tile refinement). Identity (`key`) stays the lattice window it came
    from: two shifts of one window are one hypothesis.'''
    shift: Tuple[int, int] = (0, 0)

    def shift_px(self) -> Tuple[int, int]:
        return (int(self.shift[0]), int(self.shift[1]))

    def extra(self) -> Dict[str, Any]:
        return dict(super().extra(), dx=int(self.shift[0]), dy=int(self.shift[1]))


@dataclass(frozen=True)
class TokenMatchCandidate(RerankedCandidate):
    '''A window verified on patch tokens: how many of the token matches agree on
    one similarity transform, and their share of the tokens compared.'''
    n_inliers: int = 0
    inlier_ratio: float = 0.0

    def extra(self) -> Dict[str, Any]:
        return dict(super().extra(), n_inliers=self.n_inliers,
                    inlier_ratio=self.inlier_ratio)


@dataclass(frozen=True)
class VoteCandidate(Candidate):
    '''A window found by tile voting: the weight the tiles voted for it with.'''
    votes: float = 0.0

    def extra(self) -> Dict[str, Any]:
        return dict(votes=self.votes)


@dataclass(frozen=True)
class CandidateSet:
    '''Candidates by descending score -- position in `candidates` is the rank
    -- and the frame they are expressed in.

    `grids[i]` is region i's PatchGrid (made by `PatchGrid.for_region`, so it
    knows its level-0 origin). Nothing the next stage can get from its own
    input is here: the tile size is each grid's own, and a window's size is
    the query's tile grid, which travels beside the set as the query itself
    -- the methods that need it take the UNROTATED QueryPatchContainer.

    `level` and `ds` are the level the candidates were found at. It is the
    level stage 1 routed to unless that level could not hold the query: the
    retriever then goes finer, one level at a time. `irretrievable_lvl` are the
    levels it found unsearchable (the routed one included, and every coarser
    level that fit-monotonicity rules out with it) and `alter_lvl` the other
    levels it tried, in order; the last of them is `level`. Comparing `level`
    with stage 1's `chosen_level` tells whether it moved. A set with no
    candidates means no level could be searched.'''
    candidates: Tuple[Candidate, ...]
    level: int
    ds: float
    grids: Tuple[PatchGrid, ...]
    irretrievable_lvl: Tuple[int, ...]
    alter_lvl: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, rank: int) -> Candidate:
        return self.candidates[rank]

    def __iter__(self):
        return iter(self.candidates)

    @property
    def best(self) -> Candidate:
        return self.candidates[0]

    # ── the one place a candidate becomes a position ─────────────────────────

    def window_tiles(self, c: Candidate, query) -> Tuple[int, int]:
        '''(rows, cols) of the window: `query`'s tile grid (the UNROTATED
        QueryPatchContainer), transposed at 90 and 270 degrees because the
        query was turned before matching.

        The grid is the query's own cut, so its tile size has to be the one
        the region grids were cut at -- otherwise the count is of other tiles
        and the window comes out the wrong size without any error.'''
        q, ts = query.grid, self.grids[c.region_index].tile_size
        if q.tile_size != ts:
            raise ValueError(f'query cut at {q.tile_size} px tiles, the '
                             f"candidates' grid at {ts}")
        if c.rotation in (90, 270):
            return q.grid_cols, q.grid_rows
        return q.grid_rows, q.grid_cols

    def window_local(self, c: Candidate) -> Tuple[int, int]:
        '''Level px of the window's top-left from its region's own origin,
        shift included.'''
        grid = self.grids[c.region_index]
        x, y = grid.tile_origin(c.lattice, c.row, c.col)
        dx, dy = c.shift_px()
        return x - grid.x_offset + dx, y - grid.y_offset + dy

    def origin_l0(self, c: Candidate) -> Tuple[int, int]:
        '''Level-0 integer a read of the window starts at: its lattice tile's
        origin, plus the shift a refining method gave it (level px times ds,
        rounded; none for a window on its lattice).'''
        x, y = self.grids[c.region_index].tile_origin_l0(c.lattice, c.row, c.col)
        dx, dy = c.shift_px()
        if dx or dy:
            return x + int(round(dx * self.ds)), y + int(round(dy * self.ds))
        return x, y

    def window_l0(self, c: Candidate, query) -> Tuple[float, float]:
        '''(w, h) of the window in level-0 px.'''
        rows, cols = self.window_tiles(c, query)
        tile = self.grids[c.region_index].tile_size
        return cols * tile * self.ds, rows * tile * self.ds

    def centre_l0(self, c: Candidate, query) -> Tuple[float, float]:
        '''Centre of the window, level-0, fractional.'''
        (x, y), (w, h) = self.origin_l0(c), self.window_l0(c, query)
        return x + w / 2.0, y + h / 2.0

    def rows(self, query) -> list:
        '''The set as table rows, rank first (1 is the best): each window's
        identity, its score and its level-0 box. `query` is the UNROTATED
        QueryPatchContainer, as for `window_l0`.'''
        out = []
        for rank, c in enumerate(self.candidates, 1):
            (x, y), (w, h) = self.origin_l0(c), self.window_l0(c, query)
            out.append(dict(rank=rank, region=c.region_index, lattice=c.lattice,
                            row=c.row, col=c.col, rotation=c.rotation,
                            score=c.score, x0=x, y0=y, w0=w, h0=h, **c.extra()))
        return out


