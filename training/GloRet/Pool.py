'''Which candidates a step's softmax is over: `--top-p`, the pool of a FoV, and the memory of its hard negatives
(`stage2_retrieval/proposal.md` §9.5).

THE ITEMS of a FoV are everything its softmax could be over, flat: for a retriever (GloRet) every window of the level at
the four rotations, for a reranker (LogLoRet) the entries of the list another retriever gave. Their marks
(`Supply.Marks`: positive, ignored, right rotation, distance) and a score for each come from outside; this file only
chooses. `--top-p` says how many:

    a number with a point (0.05, 1.0)  a share of the items: 1.0 is all of them
    an integer (2000)                  that many

`--min-candidates` raises a share that is too few for a small level (not an integer: that is what was asked). The
negatives of the pool are K; the answers come on top (K + 1), and so do the near windows of the soft target.

THREE REGIMES, by the share (an integer is read as K over the items):

    all        1.0 (or K reaches every item)   every item; no ranking, no episode
    each step  above 0.5                       the top K by the score of the network as it is NOW, made again at every
                                               step; no teacher, no episode, no memory
    episodes   0.5 or below                    the pool is made at the start of each EPISODE and holds for its steps: from
                                               the teacher's scores in the first epochs (`--warmup-epochs`), from the
                                               network's own after; with the memory

THE POOL, from the scores of the items and the best score of an answer: the negatives ahead of the answer (the ones
that beat it), the nearest first, are all hard negatives (N' of them); the pool is the first K of them. When fewer
than K beat the answer, the places after it are filled with the MEMORY's members that are not ahead (it holds the
negatives that beat the answer when last ranked), then with the next best-scoring items.

THE MEMORY of a FoV (`Memory`): the negatives that beat its answer, with how far; a negative the answer now beats is
struck off, and it is entered again when it beats the answer again. When every item is ranked at the start of an
episode its members are exactly the negatives ahead of the answer, and the pool does not need them; it matters when a
ranking leaves some items out.
'''
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TopP:
    """`--top-p`: a share of the items (`share`) or a number of them (`count`)."""
    share: Optional[float] = None
    count: Optional[int] = None

    def __post_init__(self) -> None:
        if (self.share is None) == (self.count is None):
            raise ValueError('a TopP is a share or a count')
        if self.share is not None and not 0.0 < self.share <= 1.0:
            raise ValueError(f'--top-p {self.share}: a share is above 0 and at most 1.0')
        if self.count is not None and self.count < 1:
            raise ValueError(f'--top-p {self.count}: a count is at least 1')

    def size(self, items: int, min_candidates: int) -> int:
        """K, the negatives of the pool, for a FoV with `items` of them: never more than there are."""
        if self.count is not None:
            return min(self.count, items)
        assert self.share is not None
        return min(items, max(math.ceil(self.share * items), min_candidates))

    def regime(self, items: int, k: int) -> str:
        """`all`, `each step` or `episodes` (see the module docstring)."""
        share: float = self.share if self.share is not None else k / max(items, 1)
        if k >= items or share >= 1.0:
            return 'all'
        return 'each step' if share > 0.5 else 'episodes'


def parse_top_p(values: Sequence[str]) -> Tuple[Optional[TopP], Optional[TopP]]:
    """`(the share, the count)` that `--top-p` gave: at most one of each, at least one. A token with a point (or an
    exponent) is a share, any other is a count; the share is for the retrievers, the count for the rerankers."""
    share: Optional[TopP] = None
    count: Optional[TopP] = None
    for token in values:
        text: str = str(token).strip().lower()
        if '.' in text or 'e' in text:
            if share is not None:
                raise ValueError('--top-p: two shares')
            share = TopP(share=float(text))
        else:
            if count is not None:
                raise ValueError('--top-p: two counts')
            count = TopP(count=int(text))
    if share is None and count is None:
        raise ValueError('--top-p: give a share (0.05, 1.0), a count (2000) or one of each')
    return share, count


@dataclass
class Memory:
    """The negatives that beat a FoV's answer, as indices into its items, with how far they did."""
    idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    margin: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def __len__(self) -> int:
        return int(len(self.idx))

    def fork(self) -> 'Memory':
        return Memory(self.idx.copy(), self.margin.copy())


def beat_the_answer(scores: np.ndarray, pos: np.ndarray, eligible: np.ndarray) -> Tuple[np.ndarray, float]:
    """`(the eligible negatives that score above the best answer, nearest first, that score)`."""
    best: float = float(scores[pos].max())
    cand: np.ndarray = np.flatnonzero(eligible)
    ahead: np.ndarray = cand[scores[cand] > best]
    return ahead[np.argsort(-scores[ahead], kind='stable')], best


def top_by_score(scores: np.ndarray, candidates: np.ndarray, k: int, skip: Optional[np.ndarray] = None) -> np.ndarray:
    """The `k` best-scoring of `candidates` that are not in `skip`, best first."""
    if skip is not None and len(skip):
        candidates = candidates[~np.isin(candidates, skip)]
    if k <= 0 or not len(candidates):
        return np.zeros(0, dtype=np.int64)
    if k >= len(candidates):
        return candidates[np.argsort(-scores[candidates], kind='stable')]
    part: np.ndarray = candidates[np.argpartition(-scores[candidates], k)[:k]]
    return part[np.argsort(-scores[part], kind='stable')]


def make_pool(scores: np.ndarray, pos: np.ndarray, eligible: np.ndarray, always: np.ndarray, k: int,
              memory: Optional[Memory]) -> Tuple[np.ndarray, int]:
    """The pool of one FoV: `(its sorted indices, N')`, N' the negatives that beat the answer (all of them, whatever
    K). `scores` over the items, `pos` the answers, `eligible` the negatives that may be chosen, `always` the items that
    are in the pool whatever (the answers, the near windows of the soft target), `k` the negatives to choose.
    `memory` is updated: the negatives that beat the answer now, and no other. Needs an answer among the items."""
    ahead, best = beat_the_answer(scores, pos, eligible)
    head: np.ndarray = ahead[:k]
    chosen: List[np.ndarray] = [head]
    taken: int = len(head)
    if memory is not None:
        # a member that no longer beats the answer is struck off; the others are, by what was just said, among `ahead`
        keep: np.ndarray = memory.idx[np.isin(memory.idx, ahead)] if len(memory) else np.zeros(0, dtype=np.int64)
        if taken < k and len(keep):
            extra: np.ndarray = keep[~np.isin(keep, head)][:k - taken]
            chosen.append(extra)
            taken += len(extra)
        memory.idx = ahead.astype(np.int64)
        memory.margin = (scores[ahead] - best).astype(np.float64)
    if taken < k:
        chosen.append(top_by_score(scores, np.flatnonzero(eligible), k - taken, skip=np.concatenate(chosen)))
    pool: np.ndarray = np.unique(np.concatenate(chosen + [np.flatnonzero(always)]).astype(np.int64))
    return pool, int(len(ahead))


__all__ = ['Memory', 'TopP', 'beat_the_answer', 'make_pool', 'parse_top_p', 'top_by_score']
