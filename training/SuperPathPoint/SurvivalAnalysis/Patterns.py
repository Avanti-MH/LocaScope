"""A survival vector -> one of the six patterns. spec.md 3.2.

    pattern, j_lo, j_hi = classify(alive)      # alive is bool[L], finest first

PURE, AND THAT IS THE POINT
=============================
Nothing here imports torch, opens a slide or touches a store. Every input is a
short boolean vector that can be typed out by hand, which matters because this
is the one place in Stage B that CANNOT FAIL LOUDLY: a classifier that calls
`[alive, dead, alive]` a contiguous band raises nothing, and the number it
corrupts -- what fraction of points are bands -- is the number spec.md 3.3 uses
to decide whether Stage C's head can be simplified to two outputs. 0.97 says
yes, 0.6 says the simplification is a silent error. So the classifier that
produces it must be checkable without a GPU.

THE VOCABULARY (spec.md 3.2 詞彙表)
=====================================
    存活向量   `alive[L]`, finest rung first
    連續帶     the alive entries form one contiguous run
    j_lo/j_hi  the finest and coarsest index of that run

    一直存活   j_lo == 0 and j_hi == L-1     carries POSITION information
    細部存活   j_lo == 0, j_hi < L-1
    晚生型     j_lo > 0, j_hi == L-1         the large structures
    只在一階   j_lo == j_hi                  carries SCALE information
    中間帶     0 < j_lo <= j_hi < L-1
    不連續     not a band at all

WHY `只在一階` IS ITS OWN CLASS AND NOT A DEGENERATE BAND
==========================================================
Arithmetically it is a band of width 1, and four of the other five names are
also just (j_lo, j_hi) read out. It is named separately because it is the class
the pipeline has a USE for that the others do not: a point alive at exactly one
rung says which rung you are looking at, which is stage 1's question. A point
alive everywhere says where you are, which is stage 2 and 3's.

It is also the class most sensitive to the detection threshold, and that is not
a coincidence -- it is defined by TWO negative decisions, one on each side, so
either one flipping moves it into a different class. `一直存活` survives one
flip as `細部存活`; `只在一階` does not survive one as anything adjacent.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

#: The six, in the order a bar chart should draw them: by where the band sits,
#: with the non-band last. A caller iterating this gets a stable column order.
PATTERNS = ('一直存活', '細部存活', '晚生型', '只在一階', '中間帶', '不連續')

#: ASCII for figures. matplotlib's default font has no CJK glyphs, so a
#: Chinese axis label renders as a row of empty boxes -- and the pattern names
#: ARE the axis labels, which made all three of Stage B's figures unreadable on
#: 2026-09-01. Kept beside the names rather than in the plotting code so that
#: adding a pattern cannot leave one without a label.
ASCII_NAMES = {
    '一直存活': 'alive-everywhere',
    '細部存活': 'fine-only',
    '晚生型': 'late-born',
    '只在一階': 'one-rung-only',
    '中間帶': 'mid-band',
    '不連續': 'flicker',
}

#: A point that is alive nowhere. It cannot occur in a table built from
#: detections -- a row exists because the point was detected somewhere -- so it
#: is a corruption check rather than a category, and it is named so that a
#: caller who sees it knows the table is wrong rather than interesting.
EMPTY = '空'


def band_of(alive: Sequence[bool]) -> Tuple[Optional[int], Optional[int], bool]:
    """`(j_lo, j_hi, is_band)`. `j_lo`/`j_hi` are None only when nothing is alive.

    `is_band` is False when the alive entries are not one contiguous run. The
    indices are still returned in that case -- they are the first and last alive
    rung -- because a flickering point still has a range, and reporting it lets
    "flickers across a wide span" be told from "flickers between two adjacent
    rungs". Callers that need a band must read the flag; a caller that reads
    only the indices sees a band that is not there, which is why the flag is
    returned rather than signalled by returning None.
    """
    flags = np.asarray(alive, dtype=bool)
    where = np.flatnonzero(flags)
    if not len(where):
        return None, None, False
    lo, hi = int(where[0]), int(where[-1])
    return lo, hi, bool(flags[lo:hi + 1].all())


def classify(alive: Sequence[bool]) -> Tuple[str, Optional[int], Optional[int]]:
    """One of `PATTERNS` (or `EMPTY`), with the band's ends.

    The order of the tests is forced by the definitions overlapping: `只在一階`
    is a band with `j_lo == j_hi`, which would also satisfy `晚生型`'s test when
    the single rung is the coarsest. Width one is checked FIRST because it is
    the narrower claim -- a point alive only at the coarsest rung says "this
    rung and no other", which is a stronger statement than "born late and
    stayed", and reporting the weaker one would put it in a class whose members
    are alive at several rungs.
    """
    length = len(alive)
    lo, hi, is_band = band_of(alive)
    if lo is None:
        return EMPTY, None, None
    if not is_band:
        return '不連續', lo, hi
    if lo == hi:
        return '只在一階', lo, hi
    if lo == 0 and hi == length - 1:
        return '一直存活', lo, hi
    if lo == 0:
        return '細部存活', lo, hi
    if hi == length - 1:
        return '晚生型', lo, hi
    return '中間帶', lo, hi


def alive_from(score: np.ndarray, dist: np.ndarray, *,
               score_threshold: float, tau: np.ndarray) -> np.ndarray:
    """`alive` derived from the stored columns. NOT a stored column itself.

    spec.md 3.2: `alive` is `score` over a threshold AND `dist` within `tau`,
    and storing it would freeze both thresholds into the data. This function is
    where the two get applied, so that a threshold sweep is a re-read of the
    same table rather than a re-run of the whole analysis.

    `dist < 0` is "no partner found at that rung", which is dead. It is a
    separate sentinel from "found but far" because the two mean different
    things to a caller trying to decide whether `tau` was too tight, and
    collapsing them would hide exactly that.
    """
    score = np.asarray(score, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    return (score > float(score_threshold)) & (dist >= 0.0) & (dist <= tau)


def summarise(alive_rows: np.ndarray) -> dict:
    """`{pattern: count}` over `[N, L]`, every key present even at zero.

    Every key present because a missing key and a zero mean the same thing to a
    reader and different things to a plot: a bar chart built from a dict that
    dropped its empty classes silently renumbers its own axis.
    """
    out = {name: 0 for name in PATTERNS}
    out[EMPTY] = 0
    for row in np.asarray(alive_rows, dtype=bool):
        out[classify(row)[0]] += 1
    return out


def band_fraction(alive_rows: np.ndarray, *, multi_only: bool = False
                  ) -> float:
    """What share of points are contiguous bands. spec.md 3.3 reads this.

    Computed over points that are alive somewhere. `EMPTY` rows are excluded
    rather than counted as non-bands -- they are a corruption of the table, and
    letting them push the fraction down would make a broken table look like a
    finding about scale structure.

    `multi_only` EXCLUDES POINTS ALIVE AT EXACTLY ONE RUNG, AND THAT IS THE
    NUMBER TO QUOTE. A width-one band is a band by arithmetic and says nothing
    about whether a head can be `(j_lo, j_hi)`: there is no interval to get
    wrong. On the 2026-09-01 corpus 69 per cent of points were single-rung, so
    the unfiltered fraction read 0.944 while the fraction among points that
    actually span a range read 0.82 -- and 0.97 against 0.6 is the decision
    spec.md 3.3 makes with it.

    Both are returned by `Report.pattern_table` because the difference between
    them is itself a fact about the corpus: it is how singleton-heavy it is.
    """
    rows = np.asarray(alive_rows, dtype=bool)
    kept = rows[rows.any(axis=1)]
    if multi_only:
        kept = kept[kept.sum(axis=1) > 1]
    if not len(kept):
        return float('nan')
    return float(np.mean([band_of(r)[2] for r in kept]))
