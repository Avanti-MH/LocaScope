"""Why a keypoint was born at the rung it was born at. spec.md 3.2.

    cause = attribute(f_alive, f_score, r_alive, suppressed, born)

新生歸因（birth attribution）: a point has `born_rung = d` on the 'F' axis, and
this assigns that birth a cause. IT IS AN INFERENCE, NOT A MEASUREMENT, so every
branch below has to name the column it reads. A branch with no column behind it
is a story.

WHAT CHANGES BETWEEN ds=1 AND ds=d, AND WHAT DOES NOT
======================================================
Two things move on the 'F' axis and only one moves on 'R':

    detail degrades          d level-0 px per sample     BOTH axes
    the receptive field
    covers d x more tissue   84 -> 84d level-0 px        'F' ONLY

That asymmetry is the whole reason both axes exist. A point born late on BOTH
axes was born by blur; a point born late only on 'F' was born by the widened
neighbourhood. Neither is a guess -- each is one column against another.

THE FOUR CAUSES
=================
    模糊新生              'R' is born at the same rung. Detail loss made it.
    鄰域新生（分數）      'F' only, and its own score rose. The receptive field
                          now holds something it did not hold before.
    鄰域新生（壓制解除）  'F' only, score flat, and it was OUTRANKED at the rung
                          below and is not at the birth rung -- the response
                          that beat it inside the NMS radius is gone.
    未定                  none of the above fits.

WHAT CANNOT BE SEPARATED, AND WHY IT IS NOT LISTED
====================================================
"a large structure came into scale" and "surrounding tissue changed the
response" are both `鄰域新生（分數）`. Geometrically they are one event -- the
receptive field's contents changed and the score rose -- and telling them apart
needs to know WHAT is in the field, which is a semantic question no stacking
geometry answers. They were two labels in an earlier draft and merging them is
the correction, not a simplification: two labels no measurement can separate is
a taxonomy that always reports whichever one the reader already believed.

`max_keypoints` MUST BE OFF UPSTREAM OF THIS
==============================================
With a cap, a point that drops out of the top N looks exactly like one killed by
its neighbourhood, and the table cannot tell them apart -- the cap is global
competition that we imposed. With the cap off, competition is only NMS-local
(radius 4 output px = 4*ds level-0 px) and it is recorded in `suppressed_by`.
`SurvivalProcess` sets it off; this module assumes it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

#: WHAT `dist` AND `suppressed_by_*` HOLD WHEN THE THING THEY DESCRIBE DID NOT
#: HAPPEN. Negative because every real value is a distance or a probability and
#: both are non-negative, so the sentinel cannot be mistaken for one -- and
#: because 0 would read as "matched exactly", the strongest possible claim.
#:
#: IT IS DEFINED HERE, IN THE PURE MODULE, AND `SurvivalProcess` IMPORTS IT.
#: The writer is the natural owner of a format constant and that is the wrong
#: direction here: `SurvivalProcess` imports torch, and a pure reader that
#: depended on it would stop being testable in a second on a login node. So the
#: heavy module depends on the light one. What must not happen is what this
#: file did until 2026-09-01 -- write `>= 0.0` inline and reference nothing. A
#: sentinel with one definition and a reader that spells it out by hand keeps
#: working after the definition changes, for the wrong reason.
NONE = -1.0

BLUR = '模糊新生'
NEIGHBOURHOOD_SCORE = '鄰域新生（分數）'
NEIGHBOURHOOD_RELEASE = '鄰域新生（壓制解除）'
UNDECIDED = '未定'
NOT_LATE = '非新生'

#: Drawing order, and the order the branches are tried in `attribute`.
CAUSES = (BLUR, NEIGHBOURHOOD_SCORE, NEIGHBOURHOOD_RELEASE, UNDECIDED)

#: How much of a rise in the point's own detector probability counts as "its
#: score rose". <!-- PENDING-MEASUREMENT: this is a THRESHOLD and it is a guess
#: (ClaudeRules 8). The first run is a calibration run: plot the distribution of
#: `score[born] - score[born-1]` for late-born points and take the knee, or
#: report the attribution at three values and record how much it moves. Do not
#: quote an attribution split produced by this default as a result. -->
DEFAULT_SCORE_RISE = 0.005


def born_rung_of(alive: Sequence[bool]) -> Optional[int]:
    """The finest rung index where the point is alive, or None if never.

    `born` is the FIRST alive index and not the first detection in time: the
    table has no time. Reading it off `alive` rather than storing it separately
    means it moves with the threshold, which is required -- a point whose birth
    rung changes when the threshold changes has to be re-attributed, and a
    stored `born_rung` would quietly keep the old answer.
    """
    where = np.flatnonzero(np.asarray(alive, dtype=bool))
    return int(where[0]) if len(where) else None


def outranked(score: Sequence[float], rival: Sequence[float]) -> np.ndarray:
    """`[L]` -- at which rungs a stronger response sits inside the NMS radius.

    `rival` is the highest raw probability within `nms_radius` of this
    location's own peak, excluding it (`SurvivalProcess.probe`). Bigger than the
    peak means the detector's point near here is NOT here -- which is the only
    form global competition takes once `max_keypoints` is off, and therefore the
    only thing 鄰域新生（壓制解除） can be read from.

    Compared against `NONE` first, because a rung that could not be probed at
    all (the location falls outside that rung's tile) records the sentinel, and
    a sentinel of -1 would otherwise read as "not outranked" -- a measurement
    where there is none.
    """
    score = np.asarray(score, dtype=np.float64)
    rival = np.asarray(rival, dtype=np.float64)
    return (rival > NONE) & (rival > score)


def attribute(f_alive: Sequence[bool], f_score: Sequence[float],
              r_alive: Sequence[bool],
              suppressed_score: Sequence[float],
              *, score_rise: float = DEFAULT_SCORE_RISE) -> str:
    """One of `CAUSES`, or `NOT_LATE`.

    All five arrays are indexed by rung, finest first, and the 'R' arrays are
    the SAME PHYSICAL POINT on the other axis -- which is only true because the
    'R' stack is derived from the chain's ds 1 tile, so the two axes share a
    centre by construction rather than by a spatial join that could mismatch.
    """
    born = born_rung_of(f_alive)
    if born is None or born == 0:
        # Born at the finest rung is not a birth to explain: there is no
        # coarser-than-it rung it failed at. Returned as its own value rather
        # than as `未定` so that "nothing to explain" and "explanation failed"
        # stay countable apart.
        return NOT_LATE

    r_born = born_rung_of(r_alive)
    if r_born is not None and r_born == born:
        return BLUR

    score = np.asarray(f_score, dtype=np.float64)
    if score[born] - score[born - 1] >= float(score_rise):
        return NEIGHBOURHOOD_SCORE

    beaten = outranked(f_score, suppressed_score)
    # Outranked at the rung below and not at the birth rung is the release.
    if beaten[born - 1] and not beaten[born]:
        return NEIGHBOURHOOD_RELEASE

    return UNDECIDED


def summarise(causes: Sequence[str]) -> dict:
    """`{cause: count}` with every key present, `NOT_LATE` included.

    `NOT_LATE` is in the output because the denominator matters: an attribution
    split over late-born points alone reads as a statement about all points
    unless the number that were not late is beside it.
    """
    out = {name: 0 for name in CAUSES}
    out[NOT_LATE] = 0
    for cause in causes:
        out[cause] = out.get(cause, 0) + 1
    return out
