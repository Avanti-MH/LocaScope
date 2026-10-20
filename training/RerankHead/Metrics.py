'''Recall of a ranked candidate list, the figure the second phase is for.

`first_hit_rank`: where the first hit stands when the candidates are ordered by a
score (1 = on top); `inf` when the list holds none. `recall_at` over many queries
counts a query with no hit in its list as a miss at every k -- the denominator is
every query asked, so the ceiling (what the first phase's K allows) shows.
'''
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

KS: tuple = (1, 5, 10, 30, 100)


def first_hit_rank(score: np.ndarray, hit: np.ndarray) -> float:
    """1 + the number of candidates scoring strictly above the best hit (a tie is
    resolved in the hit's favour, as `nearest_window`'s rank does), or `inf`."""
    if not hit.any():
        return float('inf')
    return 1.0 + float((score > score[hit].max()).sum())


def recall_at(ranks: Sequence[float], ks: Sequence[int] = KS) -> Dict[str, float]:
    """`{'R@k': share of queries whose first hit is within the top k}` and
    `ceiling`, the share that has any hit in the list at all."""
    r: np.ndarray = np.asarray(ranks, dtype=np.float64)
    if r.size == 0:
        return {**{f'R@{k}': float('nan') for k in ks}, 'ceiling': float('nan'), 'n': 0}
    out: Dict[str, float] = {f'R@{k}': float((r <= k).mean()) for k in ks}
    out['ceiling'] = float(np.isfinite(r).mean())
    out['n'] = int(r.size)
    return out
