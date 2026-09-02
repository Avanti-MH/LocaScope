"""A fixed ds ladder, resolved to a per-slide reading plan.

    ladder = DsLadder()                         # (1, 2, 4, 8, 16, 32)
    for plan in ladder.plan(wsi, tile_size=256):
        tile = wsi.read_region_rgb((x, y), plan.level, plan.read_size)
        tile = cv2.resize(tile, (256, 256), interpolation=cv2.INTER_AREA)

WHY A FIXED LADDER AND NOT THE SLIDE'S OWN LEVELS
--------------------------------------------------
BRACS SVS steps 4x per level and has four of them: ds 1, 4, 16, 32. Ki67 MRXS
steps 2x and has ten: ds 1, 2, 4, 8, ... Sampling at native levels therefore
leaves ds 2 and ds 8 supplied by Ki67 ALONE -- and Ki67 is DAB brown while BRACS
is H&E pink. Resolution and stain become correlated in the training set, and
nothing raises. spec.md 6.5 has the measured pyramids.

The relative-rung labels of Stage C make the same point from the other side: a
rung step has to mean the same thing on both datasets, and "the next pyramid
level" means 4x on one and 2x on the other.

WHY THIS NEEDS ITS OWN LEVEL RESOLVER
--------------------------------------
SafeSlide already has two, and neither answers this question:

    nearest_level_for_downsample   (SafeSlide.py:333) closest by ratio, either
                                   side -- so it can land coarser than asked
    coarser_level_for_downsample   (SafeSlide.py:349) the FINEST level at least
                                   as coarse as the request, i.e. deliberately
                                   the coarse side. Its docstring carries the
                                   1398-shot measurement that justifies that
                                   direction for routing a query to a level.

Asking `coarser_level_for_downsample(2)` on a 4x pyramid returns level 1, whose
ds is 4 -- coarser than requested, so reaching ds 2 from it means UPSAMPLING.
Upsampling does not create resolution; it creates interpolation texture, and a
keypoint detector will happily learn to fire on it.

What a ladder needs is the opposite rounding: the COARSEST level whose native
downsample is at most the target, then downsample the rest of the way in
software. On a 4x pyramid ds 2 becomes "read level 0, shrink by 2".

So `finer_level_for_downsample` below is a third resolver, on purpose, and this
paragraph is why it is not a duplicate of the other two.

NOT AN IdentifiedConfig YET
----------------------------
`utilities/ConfigIdentity.py` imports torch at module scope. This file does
arithmetic on `level_downsamples` and nothing else, and `test_ds_ladder` runs on
a login node in under a second because of that. It grows an identity when the
keypoint label store lands (spec.md 6.3) and needs to hash which ladder produced
which labels -- not before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

#: Same slack, same reason as `SafeSlide.py:80-83`: pyramid downsamples are
#: derived from rounded level dimensions, so a "4x" level reports 4.00003 as
#: readily as 4.0, and an exact comparison lands a level away over a part in
#: 1e5. Not imported from SafeSlide because importing it would pull in openslide
#: for one float.
LEVEL_REL_TOL = 1e-3

#: spec.md 6.5. ds 32 is the coarsest rung with measured evidence that tiles
#: exist at it (Ki67 level 5 and BRACS level 3 both sampled 100/100 at tile 256);
#: ds 64 measured 0/100 after 500 tries. The probe of spec.md 12 step 3b is what
#: turns that into per-slide counts.
DEFAULT_RUNGS: Tuple[float, ...] = (1., 2., 4., 8., 16., 32.)


def finer_level_for_downsample(level_downsamples: Sequence[float],
                               downsample: float) -> int:
    """Coarsest level whose native downsample is at most `downsample`.

    The level to READ so that the target can be reached by shrinking. Rounds
    DOWN in downsample, which is the opposite of
    `SafeSlide.coarser_level_for_downsample` -- see the module docstring for why
    both exist.

    Raises when the request is finer than level 0, because the only way to
    satisfy it would be to upsample and the caller almost certainly meant
    something else.
    """
    if downsample <= 0:
        raise ValueError(f'downsample must be positive, got {downsample}')
    downsamples = [float(d) for d in level_downsamples]
    threshold = float(downsample) * (1.0 + LEVEL_REL_TOL)
    candidates = [i for i, d in enumerate(downsamples) if d <= threshold]
    if not candidates:
        raise ValueError(
            f'ds {downsample} is finer than level 0 (ds {downsamples[0]:.6g}); '
            f'reaching it would mean upsampling, which this ladder refuses. '
            f'Available: {[round(d, 4) for d in downsamples]}')
    return max(candidates)


@dataclass(frozen=True)
class RungPlan:
    """How to read one rung of one slide.

    `read_size` is in LEVEL pixels and is what `read_region` wants; `origin` is
    always in level-0 pixels, because that is what openslide takes regardless of
    level (`utilities/README.md`, coordinate table).
    """
    rung_ds:      float     # the requested downsample
    level:        int       # WSI level to read
    level_ds:     float     # that level's native downsample
    shrink:       float     # extra downsample after reading; >= 1 always
    tile_size:    int       # output tile side, in pixels
    read_size:    int       # level-`level` pixels to read, before shrinking
    footprint_l0: float     # what one tile actually covers at level 0

    #: What must FIT, which is not always what the tile covers: SuperPathPoint
    #: gates on the tile and then reads a pre-tile three times its side
    #: (spec.md 6.6). A sampler that reserves this never offers a position
    #: whose pre-tile runs off the region, which is a refusal instead of the
    #: clipping that would otherwise repair it afterwards. 0 means "the
    #: footprint", so every existing caller is unchanged.
    reserve_l0:   float = 0.0

    #: 'F' (FoV stack: tile_size held, footprint grows with ds) or 'R'
    #: (resolution stack: footprint held, the tile degraded and restored).
    #: The ladder builds 'F' -- reading a coarser level IS the wider field --
    #: and 'R' is constructed by a caller that wants the other question.
    #: In `TileSampler`'s identity, because a survival number that does not say
    #: which stack it measured is not a number anyone can use.
    stack_kind:   str = 'F'

    @property
    def reserve(self) -> float:
        """`reserve_l0`, defaulted to the footprint. Read this, not the field."""
        return float(self.reserve_l0 or self.footprint_l0)

    @property
    def requested_footprint_l0(self) -> float:
        """What the caller asked for, before `read_size` was rounded to whole
        pixels. Kept separate from `footprint_l0` so the rounding stays visible:
        they differ by up to `level_ds` px, which at ds 32 is 32 level-0 px."""
        return self.tile_size * self.rung_ds

    @property
    def is_native(self) -> bool:
        """True when the pyramid has this rung and no shrinking is needed."""
        return abs(self.shrink - 1.0) <= LEVEL_REL_TOL

    def summary(self) -> str:
        native = 'native' if self.is_native else f'shrink {self.shrink:g}x'
        return (f'ds {self.rung_ds:>4g}  level {self.level}  '
                f'(ds {self.level_ds:>8.4g})  read {self.read_size:>5d} px  '
                f'-> {self.tile_size} px  footprint {self.footprint_l0:>8.0f} L0 px'
                f'  [{native}]')


@dataclass(frozen=True)
class DsLadder:
    """A ladder of downsample rungs, shared by every slide.

    The rungs are relative to each slide's own level 0, so they are NOT
    comparable across slides in physical units -- BRACS level 0 is 0.252 um/px
    and Ki67 is 0.243, and other datasets differ far more. That is deliberate and
    it is why Stage C's labels are RELATIVE (spec.md 3.3): a rung step is a
    factor, and a factor has no units.
    """
    rungs: Tuple[float, ...] = DEFAULT_RUNGS

    def __post_init__(self):
        if not self.rungs:
            raise ValueError('a ladder needs at least one rung')
        if any(r <= 0 for r in self.rungs):
            raise ValueError(f'rungs must be positive, got {self.rungs}')
        if list(self.rungs) != sorted(self.rungs):
            raise ValueError(
                f'rungs must be ascending so that a rung index is also an '
                f'ordering; got {self.rungs}')

    def plan(self, level_downsamples: Sequence[float],
             tile_size: int) -> List[RungPlan]:
        """One RungPlan per rung.

        Takes `level_downsamples` rather than a slide handle so that this is
        testable against a made-up 4x pyramid and a made-up 2x pyramid without
        opening a file -- which is the whole reason `test_ds_ladder` runs in a
        second. `plan_for(wsi, ...)` is the convenience that reads it off a
        handle.
        """
        if tile_size <= 0:
            raise ValueError(f'tile_size must be positive, got {tile_size}')
        downsamples = [float(d) for d in level_downsamples]
        plans: List[RungPlan] = []
        for rung in self.rungs:
            level = finer_level_for_downsample(downsamples, rung)
            level_ds = downsamples[level]
            shrink = float(rung) / level_ds
            read_size = int(round(tile_size * shrink))
            plans.append(RungPlan(
                rung_ds=float(rung), level=level, level_ds=level_ds,
                shrink=shrink, tile_size=int(tile_size), read_size=read_size,
                footprint_l0=read_size * level_ds))
        return plans

    def plan_for(self, wsi, tile_size: int) -> List[RungPlan]:
        """`plan()` against an open slide's own pyramid."""
        return self.plan(wsi.level_downsamples, tile_size)

    def reachable(self, level_downsamples: Sequence[float],
                  tile_size: int, max_footprint_l0: float) -> List[RungPlan]:
        """The rungs whose level-0 footprint fits inside `max_footprint_l0`.

        The footprint is `tile_size * rung_ds`, and it is what the tissue mask
        has to accommodate: `TileSampler` rejects any position whose window is
        not `tissue_ratio` tissue, and a window that spans a millimetre of slide
        rarely is. Measured on the existing runs (spec.md 6.5): 8192 level-0 px
        sampled 100/100, 16384 sampled 0/100 after 500 tries, 32768 had no
        region that could even hold it.

        This is a HINT, not the answer. Which rungs actually yield tiles is what
        the probe of spec.md 12 step 3b measures, per slide and per
        `tissue_ratio`; this function only says which ones are worth asking
        about.
        """
        return [p for p in self.plan(level_downsamples, tile_size)
                if p.requested_footprint_l0 <= max_footprint_l0]
