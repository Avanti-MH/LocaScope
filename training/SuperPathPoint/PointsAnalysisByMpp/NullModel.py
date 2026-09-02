"""What the six patterns would be if the rungs were independent coin flips.

    expected = null_patterns(alive_rate_per_rung)

WHY A CLOSED FORM AND NOT A DECOY
===================================
Every criterion in this project is a margin over a decoy (spec.md 1), and the
usual decoy is a shifted point set. For the pattern distribution there is
something better available, and better in two specific ways.

    EXACT      L rungs is 2**L survival vectors -- 64 at L=6. Enumerating them
               gives the expected fraction of each pattern with no sampling
               error at all, where a shifted-set decoy would have one and would
               need a run to produce it.
    CONTROLLED The null keeps the MEASURED per-rung alive rates. So "bands are
               more common than chance" is a statement about STRUCTURE and not
               about density -- a slide whose points mostly survive everywhere
               has a high band fraction for a reason that has nothing to do with
               scale structure, and a null that did not hold the rates fixed
               would credit that to the finding.

THE DECOY IS NOT REPLACED, IT MOVES
=====================================
Shifting the point set is still the right instrument for MATCHING -- whether a
set displaced past tau matches anyway -- because that has no closed form: it
depends on how the points are laid out in the tile, which is exactly what is
being asked. So: the null model for patterns, the shifted decoy for tau.

WHAT THE NULL DOES NOT MODEL
==============================
Independence between rungs. That is the point -- it is the hypothesis being
tested -- but it also means the null is wrong in a KNOWN direction for one
reason that has nothing to do with scale: neighbouring rungs read overlapping
tissue at similar detail, so their detections correlate even for a structureless
image. The null therefore UNDERSTATES the band fraction a null-ish corpus would
show, and a small excess over it is not evidence. Quote the excess with that
said, or measure the residual correlation on shuffled tiles before leaning on a
small one.
"""

from __future__ import annotations

import itertools
from typing import Dict, Sequence

import numpy as np

from PointsAnalysisByMpp.Patterns import EMPTY, PATTERNS, classify


def null_patterns(alive_rate: Sequence[float]) -> Dict[str, float]:
    """`{pattern: expected fraction}` over the 2**L vectors, given per-rung rates.

    Conditioned on being alive somewhere, because the table is: a row exists
    because the point was detected at some rung, so the all-dead vector cannot
    occur and leaving it in the denominator would deflate every fraction by the
    same factor -- invisibly, and by more the sparser the corpus.
    """
    rate = np.asarray(alive_rate, dtype=np.float64)
    if rate.ndim != 1 or not len(rate):
        raise ValueError(f'alive_rate must be a 1-d rate per rung, got {rate}')
    if np.any((rate < 0.0) | (rate > 1.0)):
        raise ValueError(f'alive_rate holds values outside [0, 1]: {rate}')

    out = {name: 0.0 for name in PATTERNS}
    out[EMPTY] = 0.0
    for bits in itertools.product((False, True), repeat=len(rate)):
        vector = np.asarray(bits, bool)
        probability = float(np.prod(np.where(vector, rate, 1.0 - rate)))
        out[classify(vector)[0]] += probability

    alive_somewhere = 1.0 - out[EMPTY]
    if alive_somewhere <= 0.0:
        return {name: float('nan') for name in out}
    scaled = {name: value / alive_somewhere for name, value in out.items()}
    scaled[EMPTY] = 0.0
    return scaled


def alive_rate_of(alive_rows: np.ndarray) -> np.ndarray:
    """`[L]` -- the measured per-rung alive rate, which is the null's input.

    Over EVERY row including the all-dead ones, deliberately. The rate is a
    property of the rungs and not of the surviving subset; conditioning it on
    "alive somewhere" would feed the null a rate inflated by the very selection
    the null is then asked to model.
    """
    rows = np.asarray(alive_rows, dtype=bool)
    if not len(rows):
        return np.zeros(0, np.float64)
    return rows.mean(axis=0).astype(np.float64)


def excess(measured: Dict[str, float], null: Dict[str, float]
           ) -> Dict[str, float]:
    """`measured - null` per pattern. The number a finding is made of.

    A ratio would be worse here and it is worth saying why: several patterns
    have null fractions near zero on a sparse corpus, and a ratio against a
    near-zero denominator turns a rounding difference into a factor of ten. The
    difference is on the same scale as the thing being reported, which is what
    a reader can hold in their head.
    """
    return {name: float(measured.get(name, 0.0) - null.get(name, 0.0))
            for name in set(measured) | set(null)}
