'''Stage 1 (mpp estimation)'s shared interface: what every estimator's
`.estimate()` hands back, and the shape `.estimate()` itself has to take.

TREE, NOT A LIST OF ALTERNATIVES. This module is the STEM — the one thing
every mpp-estimation method shares (an `EstMppResult` to return, an
`estimate(query)` to return it from). `ClassifierEstMpp.py` and
`KnnEstMpp.py` are BRANCHES: each owns its own config, its own
build-time setup (a KNN reference bank is not a loaded checkpoint, and
forcing them through one shared `build()` signature would hide that they
really do different work), and its own `<Method>EstMppResult(EstMppResult)`
subclass for whatever is specific to how it arrived at an answer.

WHY BOTH `estimated_*` AND `chosen_*`. `estimated_ds`/`estimated_mpp` is the
method's own raw answer -- what it thinks the query's scale is, in whichever
unit is more natural to report it in. `chosen_ds`/`chosen_mpp`/`chosen_level`
is that answer SNAPPED to a pyramid level this particular WSI actually has --
what a caller reads a tile at. The two can legitimately differ (a WSI has no
level at exactly the estimated scale), and a caller that wants the raw signal
rather than the snapped one -- e.g. to score estimation accuracy against
ground truth, where snapping to whatever levels one slide happens to have
would bias the metric -- needs both on hand rather than having to reverse the
snap. Storing both also means a caller never needs the WSI handle just to
convert one into the other.

`estimated_ds`/`chosen_ds` are stored ALONGSIDE the mpp versions, not derived
from them on demand, for the same reason -- `ds = mpp / base_mpp`, so the two
are a matched pair only because `base_mpp` does not change between when this
Result was built and when it is read, and a Result that stored one and made
the reader supply `base_mpp` again to get the other is a second place that
number could disagree with itself.

NO `method: str` FIELD. Which method produced a Result is the Result's own
TYPE (`ClassifierEstMppResult` vs. `KnnEstMppResult`) -- `type(result).__name__`
already answers it, and
a string field alongside the type is a second spelling of the same fact, able
to disagree with it.

ALL FIELDS REQUIRED, NO DEFAULTS -- on this class and on every subclass.
Python dataclass field ordering requires that once one field in the MRO has a
default, every field after it (including a subclass's own, added later) must
have one too, or it is a `TypeError` at class-definition time, not at
construction time. Keeping every field on every level required sidesteps that
trap entirely rather than relying on remembering the ordering rule correctly
in every subclass, forever.

NO `snap_to_wsi_level()` HERE. The operation exists already and is exactly
right for this: `SafeSlide.coarser_level_for_downsample(downsample)` is the
repo's own measured, purpose-built "route a query to a pyramid level" method
(1398-shot comparison against the finer-biased alternative, in its own
docstring). With exactly one caller today (`ClassifierEstMpp.estimate()`),
wrapping that call plus the two-line `chosen_ds`/`chosen_mpp` derivation in a
function of its own here would be a second name for a three-line composition
with no second caller yet to justify sharing it — see the estimator for the
inline version. Extract it the day a second caller needs the identical
composition, not before.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EstMppResult:
    '''What every stage-1 estimator's `.estimate()` returns, at minimum.

    `chosen_level` indexes `wsi.level_downsamples` on the SAME `wsi` the
    estimator was built against -- it is only meaningful together with that
    handle, which is why this Result does not also carry the handle itself
    (a Result that outlives its WSI object should not look like it still
    refers to one).
    '''
    estimated_ds: float
    estimated_mpp: float
    chosen_ds: float
    chosen_mpp: float
    chosen_level: int


@runtime_checkable
class MppEstimator(Protocol):
    '''The one call shape every stage-1 estimator's `.estimate()` honours.

    Deliberately THIN: build-time setup (what a KNN reference bank needs vs.
    what loading a trained checkpoint needs) is not part of this Protocol on
    purpose -- see this module's own docstring for why forcing that through
    one signature would hide a real difference between methods rather than
    unify a superficial one. `query` is a raw image array on purpose too --
    "the input interface is just the query image" is the one thing every
    estimator this Protocol describes was asked to agree on.
    '''

    def estimate(self, query: np.ndarray) -> EstMppResult:
        ...
