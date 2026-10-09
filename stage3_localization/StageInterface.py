'''Stage 3 (localization)'s shared interface: what every localizer hands back
and the shape of the call that produces it. The stage-2 counterpart is
`stage2_retrieval/StageInterface.py`, whose `Candidate` / `CandidateSet` this
mirrors: a `LocalizationResult` is one verified candidate, a
`LocalizationResultSet` the verified candidates of a query.

    stage 2   ret.retrieve(query, EstMppResult)          -> CandidateSet
    stage 3   loc.localize(query, CandidateSet, topk)    -> LocalizationResultSet

A method's own result subclasses `LocalizationResult` and adds what is its
own (`SiftRansacResult`: the homography, the inlier counts, the matches). The
pipeline, the bench and locate_photo see only the fields here; a method's extras
are asked for with `getattr` and done without when absent (`pair_rows()`).

CONTRACT. `confidence` is 0.0 when the method did not localize the query, and
`x0`, `y0`, `center_x0`, `center_y0` are then the candidate window's own
position, so a caller never has to ask whether a fit "succeeded": the largest
confidence wins, and where all are 0.0 the answer is the retrieval's.
Positions are level-0 and fractional (see stage2_retrieval/StageInterface.py).
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Protocol, Tuple, Union

import numpy as np

from stage2_retrieval.StageInterface import Candidate, CandidateSet

if TYPE_CHECKING:       # annotations only
    from SafeSlide import SafeSlide


@dataclass(frozen=True)
class LocalizationResult:
    '''One verified candidate: where the query is, and how sure the method is.

    `x0, y0` is where the query's own (0, 0) landed and `center_x0, center_y0`
    its centre; the centre is the rotation-invariant anchor, so prefer it when
    the orientation is unknown. `rank` is the candidate's place in the
    CandidateSet (0 is the best); `ds` and `level` are the frame the candidate
    was found in. `confidence` grows with the evidence, 0.0 meaning none.'''
    x0: float
    y0: float
    center_x0: float
    center_y0: float
    confidence: float
    rank: int
    candidate: Candidate
    ds: float
    level: int

    def row(self) -> Dict[str, Optional[Union[int, float, bool, str]]]:
        '''This result as one table row, rank first and 1-based. A method that
        has more to say overrides this and keeps these columns.'''
        c: Candidate = self.candidate
        return dict(rank=self.rank + 1, region=c.region_index, lattice=c.lattice,
                    row=c.row, col=c.col, rotation=c.rotation,
                    confidence=self.confidence, x0=self.x0, y0=self.y0,
                    center_x0=self.center_x0, center_y0=self.center_y0)


@dataclass(frozen=True)
class LocalizationResultSet:
    '''The verified candidates, by candidate rank (position is the rank).'''
    results: Tuple[LocalizationResult, ...]

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, rank: int) -> LocalizationResult:
        return self.results[rank]

    def __iter__(self) -> Iterator[LocalizationResult]:
        return iter(self.results)

    @property
    def best(self) -> LocalizationResult:
        '''The answer: the largest confidence, the better-ranked candidate on a
        tie. Where every confidence is 0.0 that is the first candidate's own
        position (the CONTRACT above).'''
        return max(self.results, key=lambda r: (r.confidence, -r.rank))


class Localizer(Protocol):
    '''Stage 3 (spec.md in stage2_retrieval): the candidates -> a position.
    Every method that replaces `SiftRansacLocalizer` honours this and nothing
    else is assumed of it. `query` is the query image; a method that wants it
    cut does the cutting itself.'''

    def build(self, wsi: Union[str, SafeSlide]) -> 'Localizer': ...

    def localize(self, query: np.ndarray, cs: CandidateSet,
                 topk: int) -> LocalizationResultSet: ...
