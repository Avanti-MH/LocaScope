'''Stage 1 (mpp estimation)'s shared interface: what every estimator's
`.estimate()` hands back, and the shape `.estimate()` itself has to take.

TREE, NOT A LIST OF ALTERNATIVES. This module is the STEM — the one thing
every mpp-estimation method shares (an `EstMppResult` to return, an
`estimate(query)` to return it from). `ClassifierEstMpp.py` and
`KnnEstMpp.py` are BRANCHES: each owns its own config, its own
build-time WORK (a KNN reference bank is sampled and encoded, a classifier
only binds the slide) and its own `<Method>EstMppResult(EstMppResult)`
subclass for whatever is specific to how it arrived at an answer. The CALL
that starts that work is shared, `build(wsi, mask=None)`: what a method does
inside it differs, how it is called does not, and a method that samples
nothing from the slide takes the mask and ignores it.

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

`routed_level` is the one place an estimate becomes a pyramid level:
`ReadGeometry.coarser_level`, the measured routing rule, and the level's own
ds and mpp. Every estimator returns its `chosen_*` fields through it.
'''
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from ReadGeometry import coarser_level


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

    def row(self) -> dict:
        '''This result as one table row: the five fields, and whatever a
        method's subclass adds.'''
        return asdict(self)


@runtime_checkable
class MppEstimator(Protocol):
    '''The call shape every stage-1 estimator honours: `build(wsi, mask=None)`
    once per slide, then `estimate(query)` per query.

    `mask` is the TissueMask the caller already built for the slide, or None
    for the method to make its own. A method that samples nothing from the
    slide (ClassifierEstMpp) takes it and ignores it, so a caller never has to
    ask which kind it holds -- the pipeline and the stage-1 bench each had a
    workaround for that (a signature probe, a `needs_mask` flag) while the
    classifier's build took no mask. Named, not `**kwargs`: a misspelt keyword
    is an error, not silently dropped. `query` is a raw image array -- "the
    input interface is just the query image" is the one thing every estimator
    was asked to agree on.

    `masks` and `cache_job` are where a method that samples a reference bank
    keeps it (`reference_bank`): the MaskMaker `mask` came from and the job
    whose cache holds the draw and its features. Without them the bank is made
    in memory; a method with no bank ignores both.
    '''

    def build(self, wsi, mask=None, *, masks=None,
              cache_job=None) -> 'MppEstimator':
        ...


    def estimate(self, query: np.ndarray) -> EstMppResult:
        ...


def routed_level(level_downsamples: Sequence[float], base_mpp: float,
                 estimated_ds: float) -> Tuple[int, float, float]:
    """`(chosen_level, chosen_ds, chosen_mpp)`: the level a query estimated at
    `estimated_ds` is routed to (`ReadGeometry.coarser_level`), its own
    downsample, and that downsample's mpp."""
    level = coarser_level(level_downsamples, estimated_ds)
    chosen_ds = float(level_downsamples[level])
    return level, chosen_ds, float(base_mpp) * chosen_ds


def reference_bank(wsi, mask, sampler_cfg, tile_size: int, *, masks=None,
                   cache_job=None):
    """`(sampler, plan)`: the TileSampler draw a reference bank is cut from,
    one rung per pyramid level (`PlanSpec('native')` for a plain
    `tile_size` tile), on `mask`.

    With `cache_job` the draw is `TileSampler.cached` in that job's tree --
    `masks` is the MaskMaker whose recipe `mask` was made by, which names the
    draw's address and makes the mask on a miss -- so `index_<sampler>.csv`
    there says which tiles the bank holds, and a later build reads them back.
    Without, it is drawn in memory: the same plan and config, the same tiles."""
    from ReadGeometry import ReadSpec                             # noqa: PLC0415
    from TileSampler import PlanSpec, TileSampler                 # noqa: PLC0415
    plan = PlanSpec('native', camera=ReadSpec(int(tile_size), int(tile_size)))
    if cache_job:
        if masks is None:
            raise ValueError('a cached reference bank needs the MaskMaker its '
                             'mask came from (masks=)')
        path = getattr(wsi, '_filename', None) or str(wsi)
        return TileSampler.cached(path, sampler_cfg, plan, cache_job,
                                  masks=masks), plan
    return TileSampler(wsi, mask, sampler_cfg).sample(plan.plans_for(wsi)), plan


def weights_path(weights: str) -> str:
    """A checkpoint named in a recipe: relative to the results root
    (`MppRoutingHead/weights/<file>`), so a recipe holds no mount; an absolute
    path is used as it is."""
    import os                                                     # noqa: PLC0415
    from _paths import RESULT_DIR                                 # noqa: PLC0415
    return weights if os.path.isabs(weights) else os.path.join(RESULT_DIR, weights)
