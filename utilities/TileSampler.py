"""Choose which tiles a slide contributes, on three controlled axes, and carry
them as objects that know what they are.

    cfg = SamplerConfig(
        tile=256, n_per_rung=500,
        richness=RichnessConfig(scorer='background'),
        overlap=OverlapConfig(grid_step=256, max_overlap_ratio=0.0),
        inherit=InheritConfig(stack_kind='F', share=0.4, source_rung=1.0),
    )
    sampler = TileSampler(wsi, mask, cfg).sample(rungs)

    sampler.where(bucket='bg70_85')     # richness  -> a filtered container
    sampler.neighbours_of(12)           # overlap   -> indices that overlap it
    sampler.stacks()                    # inherit   -> complete chains
    sampler[7]                          # random    -> one Sample

=============================== SPEC ===============================

THREE AXES, BOTH CONTROLS AND INDEXES
=======================================
Every axis is set when sampling and queried afterwards, because those are the
same information read twice. All three go into `sampler_id`: a corpus cut at
one setting is not the corpus cut at another, and `FeatureStore.cfg_hash`
covers the encoder and the mask but nothing about sampling -- so without this,
two different corpora write the same filename.

    richness   what is IN a tile. Candidates are scored, bucketed, and each
               bucket carries a FLOOR and a CAP -- two constraints of different
               natures, which is why they are two tuples. A cap is always
               achievable; a floor demands that the slide actually has the
               tiles. `RichnessConfig` holds the settled seven-bucket contract
               and the arithmetic guards on it. The scorer is pluggable:
               background fraction (mask only, free), stain saturation,
               entropy.

               THERE IS NO TISSUE GATE. There was, and it scored the same
               quantity as the buckets -- so the gate could empty a bucket the
               quota had reserved, and the rung came back short with nothing
               saying why. A cap of zero on the top buckets is that gate,
               stated once. See `RichnessConfig` for the 475/500 episode.
    overlap    how much two tiles of the SAME rung may share. Candidates come
               off a lattice whose step is a config field, so grid_step=128
               against tile=256 is a deliberate 50 per cent lattice and
               grid_step=256 is a disjoint one. Two further bounds: how much
               any pair may overlap, and what share of the set may overlap at
               all.
    inherit    a set of level-0 centres present at EVERY rung, so the same
               physical tissue appears at every magnification. `share` may be
               anything from 0 to 1.

THE ORDER IS FORCED
====================
    1. the inheritance set, fixed across all rungs and validated at each
    2. richness floors, then targets, then the spill -- per rung, with the
       inherited ones already counted
    3. the overlap bound, per rung, as candidates are taken

Any other order breaks, silently:

    richness first  a centre that is 90 per cent tissue at ds 1 reaches into
                    glass at ds 32, where its footprint is 32x larger, and
                    lands in a different bucket. Fill the buckets first and the
                    inheritance set has nowhere to go.

                    Within richness the three passes are themselves ordered,
                    and for the same class of reason: floors are not
                    independent the way caps are, so a single shuffled pass
                    hands positions to whichever bucket it reaches first and
                    the bucket with the floor finds its share already spent.
    overlap first   at ds 32 the footprint is 8192 level-0 px; a disjoint
                    lattice fills up long before the inherited centres are
                    placed.

`ReferenceSampler` states the same constraint about its own inheritance set:
"it must be fixed before the per-level quotas are filled, since it consumes
them."

TWO CONFLICTS, RESOLVED RATHER THAN HIDDEN
============================================
INHERITED POSITIONS ARE EXEMPT FROM THE OVERLAP BOUND. The set is chosen at one
rung -- the one with the most candidates -- and carried. At a coarser rung the
same centres can be a few hundred px apart while the footprint is 8192, i.e.
almost entirely overlapping. Enforcing the bound on them would drop members and
leave `inherit_id`s that do not resolve, which reads as "the keypoint did not
survive" when it means "the tile was never sampled". They are exempt, counted
separately, and `preflight` reports how many breach the bound.

A BUCKET IS NOT THE SAME BUCKET AT TWO RUNGS, and this one is worth reading
twice because the two requirements are not both satisfiable -- not as an
implementation limit, but by definition.

One centre. At ds 1 its footprint is 256 level-0 px and lands wholly inside a
gland: 90 per cent tissue, bucket `lt15` (little background). The SAME centre
at ds 32 has a footprint of 8192 px, reaches out past the tissue edge into
glass, and is 40 per cent background: bucket `mid`. Nothing moved. The tile
grew.

So "every rung's buckets hit their targets" and "a chain is one thing across
rungs" cannot both hold. `bucket_frame` says which one is given up:

    'per_rung'    the bucket is recomputed at each rung. Each rung's
                  distribution is exactly what the contract says. A chain's
                  members sit in whatever buckets their footprints put them in,
                  so a chain has no single bucket and cannot be grouped by one.

    'at_inherit'  the bucket is fixed when the centre is chosen, at
                  `source_rung`, and carried unchanged to every rung. A chain
                  has ONE bucket. The contract then acts only on the
                  non-inherited remainder, so the per-rung distribution is the
                  carried set plus whatever the contract makes of what is
                  left -- which is not what the floors asked for, and the gap grows
                  with `inherit.share`.

THE DEFAULT IS 'per_rung', because the floors exist to control what each rung
CONTAINS, and that is the thing every consumer of a single rung depends on. It
is the same choice `ReferenceSampler` made.

BUT STAGE B WANTS 'at_inherit'. A survival analysis stratified by bucket --
"do keypoints in tissue-dense tiles survive the ladder better than ones at the
edge?" -- needs the stratum to mean one thing along the whole chain. Under
'per_rung' a chain drifts between buckets as it climbs, and grouping by bucket
at rung k groups a different set than at rung k+1: the question is not
answerable, and it is not answerable in a way that produces a number rather
than an error.

So: 'per_rung' for a corpus that trains, 'at_inherit' for a corpus that is
analysed by stratum. It is in `sampler_id` because the two are different
corpora, and the failure of picking wrong is a table of survival rates whose
rows are not comparable.

TWO STACK KINDS, AND THE DIFFERENCE IS WHICH QUANTITY IS HELD
===============================================================
With the centre fixed, `footprint_l0 = tile * ds` leaves one free choice:

    'F'  FoV stack.        tile_size held, footprint grows with ds. Read at the
                           rung's own level. The ds 32 tile CONTAINS the ds 1
                           tile: same pixel count, more tissue, coarser detail.
    'R'  resolution stack. footprint held at `tile` level-0 px, tile_size held.
                           Read at level 0, downsampled by ds, upsampled back
                           to tile_size. Same tissue, same output size, less
                           real detail.

Both are trainable -- 'R' returns `tile` px, so a fixed-input student eats it.
They answer different questions, and a survival number that does not say which
is meaningless: 'F' asks whether a keypoint survives a wider field, 'R' asks
whether it survives losing resolution. That is why `stack_kind` is in the
identity rather than being a read-time flag.

WHAT EACH TILE CARRIES
=======================
    bucket, score      richness: which bucket, and the raw score
    origin, parent     'grid' | 'jitter' | 'inherit', and a jittered
                       coordinate's parent
    overlap_max        the largest overlap ratio with any other tile of the
                       same rung. 0.0 for a lattice position
    inherit_id         index of the chain; -1 when not inherited
    stack_kind         'F' or 'R'

FOUR WAYS IN, AND THEY ARE THREE DIFFERENT SHAPES
===================================================
    richness   a FILTER   -- one scalar per row     where(bucket='bg70_85')
    overlap    a RELATION -- pairwise               neighbours_of(i)
    inherit    a GROUPING -- by chain               stacks()
    random     an INDEX   -- by position            sampler[i]

Filter and grouping return a container of the same type, so they compose:
`sampler.where(bucket='gt80').stacks()`.

CROSS-RUNG OVERLAP IS INHERITANCE, REGISTERED OR NOT
======================================================
Two tiles of different rungs sharing tissue is the same relation as
inheritance. `inherit_id >= 0` is the registered case. `inherit_id == -1` with
a cross-rung overlap is the UNREGISTERED case, and that is the one to look for
before splitting train from validation -- it is content appearing on both
sides. `unregistered_overlaps()` is that query, and it costs nothing extra
because the index is built anyway.

A CHAIN IS COMPLETE OR IT SAYS SO
===================================
`ReferenceSampler`: "a correspondence with holes in it is not a
correspondence." `stacks()` returns complete chains only; `stacks(
complete_only=False)` returns the rest with the missing rungs named. A
four-rung chain returned as if it were six reads as "the keypoint died at
ds 16" when it means "ds 16 never sampled it", and those two are the whole of
Stage B's conclusion.

A Sample CARRIES COORDINATES, NEVER A HANDLE
==============================================
An openslide handle cannot be pickled, so a `Sample` holding one kills a
DataLoader the moment `num_workers > 0`. `SampleMeta` is therefore plain data
and `materialise(reader)` takes the reader as an argument: the Dataset opens
one handle per worker and hands it in. That constraint is what makes the
streaming mode usable at all, and it is why `Sample.image` is optional rather
than there being two classes.

    resident   materialise() every sample, then decide whether to persist
    streaming  materialise -> read the meta -> release; only metadata is
               carried from end to end

COST
=====
Overlap is O(n^2), so the index is built PER SLIDE and never globally: tiles of
different slides cannot overlap. Within a slide both same-rung and cross-rung
pairs are computed -- 3000 tiles is 4.5M pairs, once, on demand. `neighbours`
is never stored: it is derived from the coordinates, and a stored copy is a
second thing to keep in step with them.

PERSISTENCE
============
`save(with_images=False)` writes the metadata table alone. `save(with_images=
True)` writes `utilities/PreTileStore.py`'s format -- PNG per record,
`index.csv`, `meta.json` -- rather than inventing a second one. The axis
columns join `index.csv`; `stack_kind` and the config belong to the batch and
go in `meta.json`.

WHAT REPLACED WHAT
====================
The previous sampler drew a region uniformly, drew (x, y) uniformly inside it,
and kept the draw if `has_tissue_l0` passed. It controlled none of the three
axes and deduplicated nothing. Measured on the corpus it produced -- 17,784
pre-tiles, 36 cells -- there were 0 exact duplicates but 202,420 overlapping
PAIRS, and 69.2 per cent of tiles overlapped another. At ds 32 it was every
tile: a 8192 px footprint sampled 500 times cannot avoid it by luck.
`RandomStrategy` keeps that behaviour, not as a fallback but as the control
arm: "how much less overlap does the lattice buy" has to be a measurement.
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

# ── the rung a tile is cut at ────────────────────────────────────────────────
#
# `RungPlan` is `utilities/DsLadder.py`'s, not a second one. That file already
# resolves a rung to (level, read_size, shrink, footprint) for a given slide,
# which is the whole of what a sampler needs to know about the pyramid -- and
# it is the only place that knows ds 2 is native on a 2x pyramid and a shrink
# on a 4x one. A local copy would be a second class with the same name and
# overlapping fields, which `extract_pretiles` (which imports both) would have
# had to disambiguate at every use.
#
# Two fields were added there rather than here, for the same reason:
# `reserve_l0` (what must FIT, when a caller reads a pre-tile around the tile)
# and `stack_kind`.

from DsLadder import RungPlan                                    # noqa: E402


def native_plans(wsi, tile: int, factor: int = 1) -> List[RungPlan]:
    """One 'F' rung per PYRAMID level, read natively. No shrinking.

    The ds ladder and the pyramid are different questions and `DsLadder`
    answers the first. This answers the second, which is what a reference bank
    wants: "one set of tiles from each magnification this slide actually has",
    where the magnifications are the slide's own rather than a fixed ladder.

    `rung_ds` is therefore the level's own downsample, and `shrink` is 1 by
    construction -- a native plan that needed shrinking would not be native.
    """
    out = []
    for lv, level_ds in enumerate(wsi.level_downsamples):
        fp = float(tile) * float(level_ds)
        out.append(RungPlan(rung_ds=float(level_ds), level=int(lv),
                            level_ds=float(level_ds), shrink=1.0,
                            tile_size=int(tile), read_size=int(tile),
                            footprint_l0=fp, reserve_l0=fp * int(factor),
                            stack_kind='F'))
    return out


def _scanned_rect(mask) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) level-0: the part of the slide that HAS image data.

    Not the canvas. On a MIRAX the canvas around `openslide.bounds-*` holds no
    pixels at all, so the mask records where it starts and how far it reaches
    and this reads both back off it.
    """
    rows, cols = mask.main_mask.shape
    return (int(mask.origin_x), int(mask.origin_y),
            int(mask.origin_x + round(cols * mask.mask_ds_x)),
            int(mask.origin_y + round(rows * mask.mask_ds_y)))


def _margin_of(plan: RungPlan) -> int:
    """Level-0 px reserved on each side of the tile. ONE definition.

    `SampleMeta.margin` computes the same thing from the meta's own integer
    fields, and the two agreeing is what makes the lattice's bound and the
    read's origin the same geometry. Everything downstream reads
    `footprint + 2*margin` and never `plan.reserve_l0`, which is what was
    ASKED for rather than what can be centred.
    """
    return (int(plan.reserve) - int(plan.footprint_l0)) // 2


def resolution_plan(ds: float, tile: int, factor: int = 1) -> RungPlan:
    """An 'R' rung: footprint held at `tile` level-0 px, read at level 0.

    `DsLadder` builds 'F' rungs -- reading a coarser level IS the wider field --
    and cannot build this one, because here `ds` is not a magnification to read
    at. It is how far the tile is degraded and restored, so `level` is 0 and
    `read_size` is `tile` at every rung, and the only thing that varies is the
    resampling `Sample.materialise` applies afterwards.

    That asymmetry is why this is a separate constructor and not a flag on the
    ladder: a ladder that returned level 0 for every rung would be a ladder
    that had stopped being about levels.
    """
    return RungPlan(rung_ds=float(ds), level=0, level_ds=1.0,
                    shrink=float(ds), tile_size=int(tile),
                    read_size=int(tile), footprint_l0=float(tile),
                    reserve_l0=float(tile) * int(factor), stack_kind='R')


# ── richness: what is in a tile ──────────────────────────────────────────────

def score_background(mask, xy: np.ndarray, plan: RungPlan) -> np.ndarray:
    """Fraction of each footprint the mask calls background. Mask only, free.

    `white_fractions` takes a LEVEL and a tile count, so the footprint has to
    be in LEVEL pixels -- which is `read_size` and not `footprint_l0 / ds`. On
    an 'R' rung `ds` is a degradation factor and not a level downsample, so
    dividing by it would ask about a footprint that does not exist.
    """
    return np.asarray(
        mask.white_fractions(xy, plan.level, max(1, int(plan.read_size))),
        dtype=np.float32)


def score_saturation(mask, xy: np.ndarray, plan: RungPlan) -> np.ndarray:
    """Stain saturation. READS PIXELS, so it is not free and says so.

    Left unimplemented rather than approximated: the mask cannot answer it, and
    a stand-in built from the mask would be `score_background` under a second
    name -- two scorers, one behaviour, and a `sampler_id` that separates two
    identical corpora.
    """
    raise NotImplementedError(
        "score_saturation reads pixels and is not written yet. It cannot be "
        "derived from the mask; a mask-derived stand-in would be "
        "score_background wearing another name, and the two would produce "
        "different sampler_ids for identical corpora")


def score_entropy(mask, xy: np.ndarray, plan: RungPlan) -> np.ndarray:
    """Shannon entropy of the tile. Reads pixels. See score_saturation."""
    raise NotImplementedError(
        'score_entropy reads pixels and is not written yet')


#: Registered scorers. A name here is part of `sampler_id`, so adding one is
#: additive and renaming one re-hashes every corpus cut with it.
SCORERS = {
    'background': score_background,
    'saturation': score_saturation,
    'entropy':    score_entropy,
}

def bucket_names(edges: Sequence[float]) -> Tuple[str, ...]:
    """Bucket names, DERIVED from the edges rather than written down.

    A written-down name drifts from the edge it describes. `BUCKETS` used to be
    the literal `('lt15', 'mid', 'gt70', 'gt80', 'full')`, and the moment the
    cuts moved to 0.15/0.30/0.50/0.70/0.85/0.95 three of those five names were
    lying about which interval they stood for -- while still being written into
    `index.csv` as the column a corpus is filtered by.

    So the name IS the interval: `bg30_50` is background in [0.30, 0.50).

    THE COST, STATED. The old tuple's comment promised that these were "the
    same five and the same order as `ReferenceSampler.BUCKETS`, so a corpus cut
    here and a reference bank cut there can be compared bucket for bucket".
    Seven derived names break that. `ReferenceSampler` is already on the
    retirement list (its own TODO), and a comparison against names that lie is
    not a comparison worth keeping -- but it is a real loss and not a free one.
    """
    cuts = [0.0] + [float(e) for e in edges] + [1.0]
    return tuple(f'bg{int(round(cuts[i] * 100)):02d}_{int(round(cuts[i + 1] * 100)):02d}'
                 for i in range(len(cuts) - 1))


def assign_buckets(score: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Score -> bucket index. Half-open upward: `[cut_i, cut_{i+1})`, top closed.

    `side='right'` puts a score EQUAL to an edge in the upper bucket, which is
    the reading of "背景比高於 15% ~ 低於 30%": 0.15 belongs to the second
    bucket, not the first. A score of exactly 1.0 lands in the last one.

    THE `full` SPECIAL CASE IS GONE. It was a CLOSED bucket at [1.0, 1.0], on
    the argument that "a footprint the mask calls entirely background is a
    different thing from one it calls 99 per cent background -- the first is off
    the tissue altogether and the second is an edge". That argument earned its
    keep while the two carried different quotas. Under the settled contract both
    `bg85_95` and `bg95_100` are capped at zero and neither is reachable, so the
    distinction no longer changes any behaviour -- and an unreachable special
    case is a branch nothing tests.
    """
    return np.searchsorted(np.asarray(edges, dtype=np.float64),
                           np.asarray(score, dtype=np.float64),
                           side='right').astype(np.int8)


def allocate_targets(floors: Sequence[float],
                     caps: Sequence[float]) -> Tuple[float, ...]:
    """Floors plus the unassigned remainder, split evenly among the ASKERS.

    Two redistribution rules act on two DIFFERENT sets, and the difference is
    the whole design:

        the unassigned remainder  ->  buckets with a positive FLOOR
        a bucket's shortfall      ->  buckets with a non-zero CAP and headroom

    This function is the first rule; `spill_order` below is the second.

    A floor asks for tiles. A cap only forbids them -- `bg50_70` at 20 per cent
    says "no more than a fifth", not "give me a fifth" -- so topping a cap-only
    bucket up out of the remainder would put tiles somewhere nothing requested.
    Splitting the remainder over all five non-zero-cap buckets instead of the
    three askers also breaks the contract arithmetically: 30/5 is 6, and
    `bg50_70` would land at 26 per cent against a stated ceiling of 20.

    With the settled numbers: floors 5 + 15 + 50 = 70, remainder 30, three
    askers, so each gains 10 and the targets are 15 / 25 / 60 / 0 / 0 / 0 / 0.
    They sum to exactly 1.0, which is why `bg50_70` and `bg70_85` normally
    receive NOTHING -- their ceilings exist only as somewhere for a shortfall
    to go.

    NOBODY ASKED is a real configuration and not a degenerate one to reject.
    `GigaPathKnnEstiMpp`'s reference bank wants "any admissible tile, no
    preference between buckets" -- all floors zero, caps 1 on the buckets it
    admits. Splitting the remainder evenly there would turn "no preference"
    into "equal thirds", which is a different bank. So with no askers the
    target IS the cap, and the fill is first-come over the shuffle.
    """
    floors = [float(f) for f in floors]
    caps = [float(c) for c in caps]
    askers = [i for i, f in enumerate(floors) if f > 0.0]
    if not askers:
        return tuple(caps)
    remainder = 1.0 - sum(floors)
    out = list(floors)
    if remainder > 0.0:
        share = remainder / len(askers)
        for i in askers:
            out[i] = min(caps[i], out[i] + share)
    return tuple(out)


def caps_for_tissue_ratio(tissue_ratio: float,
                          edges: Optional[Sequence[float]] = None
                          ) -> Tuple[float, ...]:
    """The retired `tissue_ratio` gate, expressed as a cap per bucket.

    `tissue >= r` is `background <= 1 - r`, so every bucket whose interval lies
    wholly at or below `1 - r` is admitted at cap 1 and the rest at 0. Two
    callers wanted exactly their old behaviour back and neither should own the
    translation: `GigaPathKnnEstiMpp`'s reference bank and
    `bench_gigapath_accuracy`, both at 0.5.

    A ratio that does not land on a bucket edge is REFUSED rather than rounded.
    Rounding would silently widen or narrow the corpus, and `edges` is a config
    field the caller can move -- so the refusal is what keeps a moved edge from
    quietly changing a bank that was meant to be unchanged.
    """
    edges = tuple(RichnessConfig().edges if edges is None else edges)
    limit = 1.0 - float(tissue_ratio)
    if not any(abs(float(e) - limit) < 1e-9 for e in edges):
        raise ValueError(
            f'tissue_ratio {tissue_ratio} means background <= {limit:g}, which '
            f'is not one of the richness edges {edges}. Pick a ratio that '
            f'lands on an edge, or pass an explicit RichnessConfig')
    return tuple(1.0 if float(e) <= limit + 1e-9 else 0.0
                 for e in edges) + (0.0,)


def spill_order(caps: Sequence[float], target: Sequence[float]) -> Tuple[int, ...]:
    """Which buckets a shortfall may flow into, in index order.

    "填不滿就去填非 0 上限的桶子" -- so the set is every bucket whose cap is
    non-zero AND whose target has not already consumed that cap. Under the
    settled contract the first three have target == cap and therefore no
    headroom, which leaves `bg50_70` and `bg70_85`; a zero cap is never in the
    set, which is what makes `bg85_95` and `bg95_100` hard rather than merely
    unpopular.
    """
    return tuple(i for i, c in enumerate(caps)
                 if float(c) > 0.0 and float(c) > float(target[i]))



# ── the three axes, as config ────────────────────────────────────────────────

@dataclass(frozen=True)
class RichnessConfig:
    """What is in a tile, and how much of each kind is wanted.

    THE CONTRACT, SETTLED 2026-08-27. Seven buckets on the background fraction,
    each carrying a FLOOR and a CAP -- two constraints of different natures and
    therefore two tuples, not one:

        a CAP is always achievable -- stop taking and it holds
        a FLOOR is not -- it demands that the slide actually HAS that many

        bucket      background     floor    cap     target
        bg00_15     < 15 %            5 %    15 %     15 %
        bg15_30     15 - 30 %        15 %    25 %     25 %
        bg30_50     30 - 50 %        50 %    60 %     60 %
        bg50_70     50 - 70 %         -      20 %      0
        bg70_85     70 - 85 %         -      20 %      0
        bg85_95     85 - 95 %         -       0        0
        bg95_100    > 95 %            -       0        0

    `target` is `allocate_targets(floors, caps)`: the floors sum to 70 per cent,
    and the unassigned 30 is split evenly over the three buckets that ASKED.
    The targets then sum to exactly 1.0, so `bg50_70` and `bg70_85` normally
    receive nothing -- their ceilings exist as somewhere a SHORTFALL can go, and
    a shortfall is the only thing that ever reaches them.

    THREE ARITHMETIC GUARDS, and the middle one is the 475/500 bug itself:

        sum(floors) <= 1      or no rung can satisfy every floor at once
        sum(caps)   >= 1      or the rung is short BY CONSTRUCTION
        floors <= caps        elementwise

    The old contract passed the middle guard by exactly nothing to spare --
    0.85 + 0.15 = 1.00 -- which is why the fine rungs stopped dead on 85/15 and
    read as a supply measurement when they were the caps themselves.

    THE TISSUE GATE IS GONE, and its removal is what makes `bg50_70` and
    `bg70_85` reachable at all. The gate and the buckets scored the SAME
    quantity: `score_background` is `white_fractions`, and `tissue >=
    tissue_ratio` is `background <= 1 - tissue_ratio`. At the settled 0.5 the
    gate deleted every candidate above 50 per cent background before a bucket
    ever saw it, so reserving share for those buckets did not produce tiles --
    it produced a SHORT RUNG. The 2026-08-26 corpus came back 475/500 and the
    number was read as a property of the slides.

    A zero cap now says everything the gate said, and says it once:
    `bg85_95` and `bg95_100` at 0 is exactly a gate at 85 per cent background.
    """
    scorer: str = 'background'

    #: Cuts on the score, ascending, strictly inside (0, 1). Six cuts, seven
    #: buckets. `bucket_names` derives the names from these so the two cannot
    #: drift apart, which they had.
    edges: Tuple[float, ...] = (0.15, 0.30, 0.50, 0.70, 0.85, 0.95)

    #: Smallest share of a rung each bucket must receive. NOT achievable by
    #: filtering -- see `shortfall_policy` for what happens when the supply is
    #: not there.
    floors: Tuple[float, ...] = (0.05, 0.15, 0.50, 0.0, 0.0, 0.0, 0.0)

    #: Largest share of a rung each bucket may receive. A zero is HARD: it binds
    #: the inherited set too, which is the one place a cap used to be advisory.
    caps: Tuple[float, ...] = (0.15, 0.25, 0.60, 0.20, 0.20, 0.0, 0.0)

    #: 'per_rung' | 'at_inherit'. See the spec above -- the two requirements
    #: are not both satisfiable and this says which is given up.
    bucket_frame: str = 'per_rung'

    #: 'ask' | 'taken'. WHAT THE FLOORS ARE A SHARE OF, and the 3b probe of
    #: 2026-08-27 is why it is a switch rather than a constant.
    #:
    #:   'ask'    a share of `n_per_rung`. What was asked for is the frame, so
    #:            a rung that cannot supply the mix takes everything it has and
    #:            the mix drifts. At ds 32 over 12 slides that is 578 tiles at
    #:            13/12/26/33/16 against a target of 15/25/60/0/0 -- almost half
    #:            of them above 50 per cent background.
    #:
    #:   'taken'  a share of what is ACHIEVABLE. The rung is scaled down to the
    #:            largest count whose mix the supply can actually hold, so the
    #:            same ds 32 becomes 250 tiles at 15/25/60. Fewer tiles, and the
    #:            proportions are the ones that were asked for.
    #:
    #: THE DEFAULT IS 'ask' because that is what the fine rungs want and they
    #: are where the corpus lives: at ds 1 and 2 the supply is two orders of
    #: magnitude past the ask, the two frames agree exactly, and 'ask' is the
    #: one that does not need the supply histogram to be right.
    #:
    #: The frames diverge only where a floor cannot be met, which is exactly
    #: where `n_below_floor` is non-zero -- so the report says which rungs the
    #: switch would move before anyone has to change it.
    floor_frame: str = 'ask'

    @property
    def names(self) -> Tuple[str, ...]:
        return bucket_names(self.edges)

    @property
    def targets(self) -> Tuple[float, ...]:
        return allocate_targets(self.floors, self.caps)

    def __post_init__(self):
        if self.scorer not in SCORERS:
            raise ValueError(
                f'no scorer {self.scorer!r}. Known: {", ".join(SCORERS)}')
        if self.bucket_frame not in ('per_rung', 'at_inherit'):
            raise ValueError(
                f"bucket_frame must be 'per_rung' or 'at_inherit', got "
                f"{self.bucket_frame!r}")
        if self.floor_frame not in ('ask', 'taken'):
            raise ValueError(
                f"floor_frame must be 'ask' or 'taken', got "
                f"{self.floor_frame!r}")
        n = len(self.edges) + 1
        if len(self.floors) != n or len(self.caps) != n:
            raise ValueError(
                f'{len(self.edges)} edges make {n} buckets, but there are '
                f'{len(self.floors)} floors and {len(self.caps)} caps')
        prev = 0.0
        for e in self.edges:
            if not (prev < float(e) < 1.0):
                raise ValueError(
                    f'edges must ascend strictly inside (0, 1), got '
                    f'{self.edges}')
            prev = float(e)
        for i, (f, c) in enumerate(zip(self.floors, self.caps)):
            if not (0.0 <= float(f) <= float(c) <= 1.0):
                raise ValueError(
                    f'bucket {self.names[i]}: floor {f} and cap {c} must '
                    f'satisfy 0 <= floor <= cap <= 1')
        if sum(self.floors) > 1.0 + 1e-9:
            raise ValueError(
                f'floors sum to {sum(self.floors):.3f} > 1. No rung can '
                f'satisfy every floor at once, whatever the slide holds')
        if sum(self.caps) < 1.0 - 1e-9:
            raise ValueError(
                f'caps sum to {sum(self.caps):.3f} < 1, so every rung is short '
                f'BY CONSTRUCTION and the shortfall will be read as a property '
                f'of the slides. This is the 475/500 bug of 2026-08-26: the '
                f'reachable caps were 0.85 + 0.15 and the rest of the quota '
                f'sat in buckets the tissue gate had already emptied')



@dataclass(frozen=True)
class OverlapConfig:
    """How much two tiles of the same rung may share.

    THREE KNOBS, AND THEY CAN CONTRADICT EACH OTHER. `grid_step` sets the
    lattice, which fixes the overlap between ADJACENT positions before any
    bound is applied:

        adjacent overlap along one axis = 1 - grid_step / tile
        adjacent overlap on the diagonal = (1 - grid_step / tile) ** 2

    So `grid_step=128, tile=256` means every neighbour overlaps 50 per cent,
    and setting `max_overlap_ratio=0.3` on top of it makes every adjacent pair
    illegal -- the lattice silently degenerates to `grid_step=256` while the
    identity still records 128. `check()` refuses that combination instead,
    with the arithmetic in the message.
    """
    #: Lattice step in OUTPUT pixels -- the same units as `tile`. **0 means the
    #: tile**, i.e. disjoint, and 0 is the ONLY spelling of that: writing the
    #: tile size out is a second spelling of one lattice, and two spellings of
    #: one thing are two `sampler_id`s over one corpus. `check()` refuses it,
    #: the same way `TileEncoderConfig` collapses its head aliases at the door.
    #:
    #: `_lattice` converts to level-0 by `footprint_l0 / tile`, which is `ds` on
    #: an 'F' rung and 1 on an 'R' one. As a LEVEL-0 constant the step would be
    #: disjoint at ds 1 and 87 per cent overlapping at ds 8 -- the trap
    #: `jitter_offsets` below records for its own values.
    grid_step: int = 0

    #: Largest area fraction any two tiles of a rung may share. 0.0 admits
    #: only positions that touch at most at the border.
    max_overlap_ratio: float = 0.0

    #: Largest share of a rung's tiles that may overlap ANYTHING at all. 0.0
    #: forbids it outright; 1.0 leaves only `max_overlap_ratio` binding.
    overlapping_share: float = 0.0

    #: Displacements offered when a bucket runs out of lattice, **as fractions
    #: of the tile**. `ReferenceSampler` writes the same five as absolute
    #: pixels -- (64, 256), (256, 64), (192, 256), (256, 192), (320, 320) --
    #: which are those numbers for a 256 px tile and are FOUR TIMES the tile at
    #: 64. Its own docstring argues that the units matter ("as level-0
    #: constants they would be 87% overlap at ds=8") and then picks one that
    #: holds for a single tile size. Fractions hold for every one.
    #:
    #: Two properties, and an entry must have both:
    #:     disjoint from the parent   max(|dx|, |dy|) >= 1
    #:     not a lattice position     dx or dy not a multiple of 1/2
    #: Three of the five originally there were multiples of 128 px -- half of a
    #: 256 tile -- and failed the second.
    jitter_offsets: Tuple[Tuple[float, float], ...] = (
        (0.25, 1.0), (1.0, 0.25), (0.75, 1.0), (1.0, 0.75), (1.25, 1.25))

    #: Largest share of a rung that may come from jitter rather than lattice.
    #:
    #: ZERO BY DEFAULT, because the default lattice is disjoint and under a
    #: disjoint lattice the top-up is provably dead. `grid_step == tile` TILES
    #: the plane, so every position that is not on the lattice overlaps two to
    #: four lattice tiles -- 75 per cent for four of the five offsets, 56 for
    #: the fifth -- and `max_overlap_ratio = 0` rejects all of them. The lattice
    #: IS the maximum set. A non-zero cap there promises a top-up that cannot
    #: happen, and the bucket stays short with nothing saying why.
    #:
    #: It means something as soon as overlap is allowed, which is what
    #: `ReferenceSampler` assumes: it has these offsets and no overlap bound at
    #: all.
    jitter_cap: float = 0.0

    def step_for(self, tile: int) -> int:
        """`grid_step`, with 0 meaning the tile. The one place that resolves it."""
        return int(self.grid_step or tile)

    def check(self, tile: int) -> None:
        """Refuse a lattice whose own adjacency breaks the bound it is under."""
        if self.grid_step == tile:
            raise ValueError(
                f'grid_step {tile} is the tile, and 0 already means that. Two '
                f'spellings of one lattice are two sampler_ids over one '
                f'corpus, so the synonym is refused at the door rather than '
                f'collapsed silently -- write grid_step=0')
        if self.grid_step < 0 or self.grid_step > tile:
            raise ValueError(
                f'grid_step {self.grid_step} must be 0 (the tile) or in '
                f'1..{tile - 1}; a step larger than the tile leaves gaps the '
                f'sampler cannot see into, which is a mask decision and not a '
                f'lattice one')
        step = self.step_for(tile)
        along = 1.0 - step / float(tile)
        if along > 0 and self.max_overlap_ratio < along:
            raise ValueError(
                f'grid_step {step} on a {tile} px tile makes every adjacent '
                f'pair overlap {along:.0%} along an axis, and '
                f'max_overlap_ratio is {self.max_overlap_ratio:.0%}. Every '
                f'adjacent position is therefore illegal and the lattice '
                f'degenerates to the disjoint one -- while sampler_id still '
                f'records {self.grid_step}. Set grid_step=0 and mean it, or '
                f'raise max_overlap_ratio to at least {along:.2f}')
        if self.jitter_cap > 0 and self.max_overlap_ratio <= 0.0:
            raise ValueError(
                f'jitter_cap is {self.jitter_cap:.0%} and max_overlap_ratio is '
                f'0, and those cannot both hold. A lattice of step {step} on a '
                f'{tile} px tile covers the plane, so every offer the top-up '
                f'can make overlaps a lattice position by 56 to 75 per cent '
                f'and is rejected -- the lattice is already the largest '
                f'disjoint set there is. Set jitter_cap=0 and mean it, or '
                f'raise max_overlap_ratio to at least 0.75')
        for dx, dy in self.jitter_offsets:
            if max(abs(dx), abs(dy)) < 1.0:
                raise ValueError(
                    f'jitter offset ({dx}, {dy}) is under a whole tile in both '
                    f'axes, so it overlaps its parent. The offsets are '
                    f'FRACTIONS of the tile, not pixels')
            if (abs(dx * 2 - round(dx * 2)) < 1e-9
                    and abs(dy * 2 - round(dy * 2)) < 1e-9):
                raise ValueError(
                    f'jitter offset ({dx}, {dy}) is a multiple of half a tile '
                    f'in both axes, so it lands back on a lattice position -- '
                    f'and the bucket was short precisely because the lattice '
                    f'had run out there')


@dataclass(frozen=True)
class InheritConfig:
    """A set of level-0 centres present at every rung.

    `share` is a fraction of a rung, not a count, so one number holds across
    rungs whose sizes differ. 0.0 turns inheritance off entirely and the whole
    `inherit_id` column is -1.
    """
    #: 'F' (tile_size held, footprint grows) or 'R' (footprint held at `tile`,
    #: resolution degraded). See the spec.
    stack_kind: str = 'F'

    share: float = 0.0

    #: Which rung the centres are chosen at. The finest rung has the most
    #: candidates, so it is the natural source -- but it is a field because
    #: choosing at the COARSEST rung guarantees every centre fits at every
    #: rung, which the finest does not.
    source_rung: Optional[float] = None

    #: 'drop' | 'keep'. What `stacks()` does by default with a chain that is
    #: missing a rung. Not in the identity: it decides what a READER is shown,
    #: not which tiles were cut.
    on_incomplete: str = 'drop'

    _NOT_IDENTITY = frozenset({'on_incomplete'})

    def __post_init__(self):
        if self.stack_kind not in ('F', 'R'):
            raise ValueError(
                f"stack_kind must be 'F' (FoV) or 'R' (resolution), got "
                f"{self.stack_kind!r}")
        if not 0.0 <= self.share <= 1.0:
            raise ValueError(f'share must be in [0, 1], got {self.share}')
        if self.on_incomplete not in ('drop', 'keep'):
            raise ValueError(
                f"on_incomplete must be 'drop' or 'keep', got "
                f"{self.on_incomplete!r}")


def _config_parts(cfg, prefix: str = '') -> List[str]:
    """`name=value` for every identity field, nested configs recursed.

    Recursion by hand rather than by `repr(cfg)`: a dataclass repr includes
    fields listed in `_NOT_IDENTITY`, so hashing it would fork the corpus on
    `on_incomplete` -- a field that decides what a reader is shown and not
    which tiles were cut.
    """
    skip = getattr(type(cfg), '_NOT_IDENTITY', frozenset())
    parts = []
    for f in sorted(dataclasses.fields(cfg), key=lambda f: f.name):
        if f.name in skip:
            continue
        value = getattr(cfg, f.name)
        if dataclasses.is_dataclass(value):
            parts += _config_parts(value, f'{prefix}{f.name}.')
        else:
            parts.append(f'{prefix}{f.name}={value!r}')
    return parts


@dataclass(frozen=True)
class SamplerConfig:
    """Everything that decides WHICH tiles are chosen. Hashed into `sampler_id`.

    The hash is the point, and `FeatureStore.py`'s own comment says why: its
    `cfg_hash` covers the encoder and the mask and nothing about sampling, so
    two runs with different quotas or a different seed produce the same
    filename. Two corpora, one name, and the reader gets whichever ran first.
    """
    tile: int = 256
    n_per_rung: int = 500
    seed: int = 0

    richness: RichnessConfig = field(default_factory=RichnessConfig)
    overlap:  OverlapConfig  = field(default_factory=OverlapConfig)
    inherit:  InheritConfig  = field(default_factory=InheritConfig)

    #: 'lattice' | 'random'. 'random' reproduces the sampler this file
    #: replaced -- a region drawn uniformly, a position drawn uniformly inside
    #: it, kept if the tissue gate passes -- and it is here as the CONTROL ARM,
    #: not as a fallback. "How much overlap does the lattice remove" has to be
    #: a measurement against something, and the something is the thing that was
    #: actually run: 202,420 overlapping pairs over 17,784 tiles, 69.2 per cent
    #: of them touching another, and every single tile at ds 32.
    candidates: str = 'lattice'

    #: Rejection budget for candidates='random'. Meaningless for a lattice,
    #: which enumerates rather than draws, and in the identity anyway because
    #: it changes which tiles come out of the random arm.
    max_tries_per_tile: int = 5

    #: Region preparation, run per rung and undone afterwards. Was hard-coded
    #: at 0.01 inside the sampling loop, which meant the figure that drew the
    #: ops pipeline at 0.05 was not showing what the corpus went through.
    min_region_ratio: float = 0.01
    merge_regions: bool = True

    def __post_init__(self):
        if hasattr(self, 'tissue_ratio'):
            raise TypeError(
                'tissue_ratio is gone. It scored the same quantity as the '
                'richness buckets -- background fraction -- and the two '
                'disagreeing is the 475/500 corpus of 2026-08-26. A zero cap '
                'on the top buckets IS the gate: richness.caps[-2:] == (0, 0) '
                'is a gate at 85 per cent background, stated once.')
        if self.candidates not in ('lattice', 'random'):
            raise ValueError(
                f"candidates must be 'lattice' or 'random', got "
                f"{self.candidates!r}")
        self.overlap.check(self.tile)

    def sampler_id(self) -> str:
        return hashlib.sha256(
            '|'.join(_config_parts(self)).encode()).hexdigest()[:8]

    def provenance(self) -> Dict[str, object]:
        """The fields deliberately NOT in the identity, so a run can still
        record them. Two corpora that differ only here are the same corpus."""
        out = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if dataclasses.is_dataclass(value):
                for name in getattr(type(value), '_NOT_IDENTITY', ()):
                    out[f'{f.name}.{name}'] = getattr(value, name)
        return out


# ── one tile ─────────────────────────────────────────────────────────────────

@dataclass
class SampleMeta:
    """Where a tile is and what is known about it. PLAIN DATA, NO HANDLE.

    An openslide handle is not picklable, so a meta carrying one cannot cross
    into a DataLoader worker -- and the failure is a pickling error a long way
    from the cause. Everything here survives `pickle`, `csv` and `json`, and
    `Sample.materialise(reader)` is where a handle enters, from the caller who
    owns it.
    """
    slide: str
    ds: float
    level: int
    x: int                  # top-left, LEVEL-0 coordinates
    y: int
    tile_size: int          # the OUTPUT side, in pixels
    read_size: int          # what is read at `level`, before any resize
    footprint_l0: int       # what this tile covers at level 0

    #: What was RESERVED around the tile, centred on it. 0 means "the
    #: footprint". A caller that reads the reserve rather than the tile -- the
    #: pre-tile corpus does, because a warp of a bare tile is a third black
    #: (spec.md 6.6) -- needs the number here rather than recomputing it from a
    #: config, because it is the number the LATTICE honoured. Recomputed
    #: elsewhere it can disagree with the geometry that placed the tile, and a
    #: read that runs off the region is then repaired by clipping instead of
    #: being impossible.
    reserve_l0: int = 0

    # ── the three axes ──
    bucket: str = 'mid'
    score: float = 0.0
    overlap_max: float = 0.0
    inherit_id: int = -1
    stack_kind: str = 'F'
    origin: str = 'grid'            # 'grid' | 'jitter' | 'inherit'
    parent_x: int = -1              # a jittered tile's parent, else -1
    parent_y: int = -1

    @property
    def centre_l0(self) -> Tuple[float, float]:
        half = self.footprint_l0 / 2.0
        return (self.x + half, self.y + half)

    @property
    def margin(self) -> int:
        """Level-0 px reserved on EACH side. The primary quantity."""
        return (int(self.reserve_l0 or self.footprint_l0)
                - int(self.footprint_l0)) // 2

    @property
    def reserve(self) -> int:
        """The reserve's side. DERIVED from the margin, never read raw.

        `reserve_l0` is what was ASKED for and `footprint + 2*margin` is what
        the geometry can actually centre, and they differ by one whenever
        `reserve_l0 - footprint` is odd -- which it is whenever
        `DsLadder.footprint_l0` (a float: `read_size * level_ds`) times the
        factor lands off a whole number. `int(4096.4 * 3)` is 12289, the pad is
        `(12289 - 4096) // 2 = 4096`, and `4096 + 2*4096` is 12288. One px.
        A region with room to spare absorbs it; a region whose far edge IS the
        mask's -- which at a coarse rung is the only kind left, because
        `filter_patchable` has removed every smaller one -- has nothing to
        absorb it with, and the read runs one px off the scanned rectangle.
        Deriving it here means the two can no longer be different numbers.
        """
        return int(self.footprint_l0) + 2 * self.margin

    @property
    def reserve_origin_l0(self) -> Tuple[int, int]:
        """Top-left of the reserve. The tile sits centred inside it."""
        return (self.x - self.margin, self.y - self.margin)

    def overlap_with(self, other: 'SampleMeta') -> float:
        """Shared area as a fraction of the SMALLER footprint.

        The smaller, not either one and not the union: a ds 1 tile lying wholly
        inside a ds 32 tile shares 100 per cent of ITSELF and 0.1 per cent of
        the other. Dividing by the larger would report that containment as
        almost no overlap, which is exactly the cross-rung case this has to
        catch. `overlap_max` within a rung is unaffected -- there both
        footprints are equal.
        """
        if self.slide != other.slide:
            return 0.0
        dx = min(self.x + self.footprint_l0, other.x + other.footprint_l0) \
            - max(self.x, other.x)
        dy = min(self.y + self.footprint_l0, other.y + other.footprint_l0) \
            - max(self.y, other.y)
        if dx <= 0 or dy <= 0:
            return 0.0
        smaller = float(min(self.footprint_l0, other.footprint_l0) ** 2)
        return float(dx * dy) / max(smaller, 1.0)


def degrade_resolution(img, ds: float, out_side: Optional[int] = None):
    """An 'R' rung's degradation: shrink by `ds`, grow back. THE ONE DEFINITION.

    INTER_AREA down, INTER_LINEAR up. Area-averaging is the non-aliasing
    downsample; coming back up with it would be a second box filter rather than
    the interpolation a real coarser level would have gone through.

    IT LIVES HERE AND NOT IN TWO PLACES ON PURPOSE. `Sample.materialise` applies
    it when a sampler reads an 'R' tile, and Stage B applies it to derive an 'R'
    stack from a chain's ds 1 tile without a second extraction -- the ds 1 tile
    IS `tile` level-0 px read at level 0, which is exactly what an 'R' rung
    starts from. Two spellings of this would make a survival number a statement
    about which resampling filter each half used, and nothing would say so.
    """
    import cv2                                              # noqa: PLC0415

    side = int(out_side or img.shape[0])
    if float(ds) <= 1.0:
        return (img if img.shape[0] == side else
                cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA))
    small = max(1, int(round(side / float(ds))))
    img = cv2.resize(img, (small, small), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (side, side), interpolation=cv2.INTER_LINEAR)


class Sample:
    """One tile: its metadata always, its pixels only if asked for.

    Not two classes and not two modes. Whether the image is resident is a
    property of the instance, so the resident and streaming uses are the same
    object used differently -- and there is no second code path to keep in step.
    """

    __slots__ = ('meta', 'image')

    def __init__(self, meta: SampleMeta, image: Optional[np.ndarray] = None):
        self.meta = meta
        self.image = image

    def __repr__(self) -> str:
        state = 'resident' if self.image is not None else 'meta only'
        return (f'Sample({self.meta.slide} ds{self.meta.ds:g} '
                f'({self.meta.x}, {self.meta.y}) {self.meta.bucket}, {state})')

    def materialise(self, reader, extent: str = 'tile') -> 'Sample':
        """Read the pixels. `reader` is the caller's handle, never ours.

        For an 'F' tile this is one read at the rung's level, resized to
        `tile_size` when the level did not land exactly on the rung. For an
        'R' tile it is a level-0 read, shrunk by `ds` and grown back -- the
        degradation IS the rung, so it happens here rather than at sampling
        time where nothing would record it.

        `extent='reserve'` reads what was RESERVED around the tile instead of
        the tile: the pre-tile corpus stores that, because a warp of a bare
        tile is a third pure black and a black wedge is a straight
        maximum-contrast edge with two right angles (spec.md 6.6). The tile is
        then its centre crop. Reading it here rather than in the caller is what
        keeps the read and the LATTICE honouring the same number -- a reserve
        recomputed at the call site can disagree with the geometry that placed
        the tile, and the read then runs off the region and gets clipped.
        """
        import cv2                                          # noqa: PLC0415

        m = self.meta
        if extent not in ('tile', 'reserve'):
            raise ValueError(
                f"extent must be 'tile' or 'reserve', got {extent!r}")
        if extent == 'reserve':
            scale = m.reserve / float(m.footprint_l0)
            origin = m.reserve_origin_l0
            side = int(round(m.read_size * scale))
            out_side = int(round(m.tile_size * scale))
        else:
            origin, side, out_side = (m.x, m.y), m.read_size, m.tile_size
        img = reader.read_region_rgb(origin, m.level, (side, side))
        if m.stack_kind == 'R' and m.ds > 1.0:
            img = degrade_resolution(img, m.ds, out_side)
        elif img.shape[0] != out_side:
            # INTER_AREA is the ladder's own downsampling filter. Anything else
            # invents high-frequency texture, and a keypoint detector will
            # learn to fire on it.
            img = cv2.resize(img, (out_side, out_side),
                             interpolation=cv2.INTER_AREA)
        self.image = img
        return self

    def release(self) -> 'Sample':
        """Drop the pixels, keep the metadata. The streaming half."""
        self.image = None
        return self


# ── the sampler, which is also the container ─────────────────────────────────

@dataclass
class RungReport:
    """What one rung could offer and what it gave. Knowable before any read.

    `supply` against `per_bucket` is the pair worth reading: the first is the
    candidate pool's histogram, the second is what was taken. A cap shows up as
    the two disagreeing; a FLOOR that could not be met shows up as
    `n_below_floor`, and it is a different fact -- the first says a bucket was
    held back, the second says the slide did not have it.

    `n_after_gate` is GONE along with the gate it counted. The tissue gate and
    the buckets scored the same quantity, so the gate could empty a bucket the
    quota had reserved and the rung came back short with nothing saying why.
    What survives is `n_admissible`: candidates whose bucket has a non-zero
    cap, which is the same filter expressed once instead of twice.
    """
    ds: float
    n_candidates: int = 0
    n_admissible: int = 0
    supply: Dict[str, int] = field(default_factory=dict)
    per_bucket: Dict[str, int] = field(default_factory=dict)
    n_inherited: int = 0
    n_inherit_breaching: int = 0     # exempt from the bound, and how many used it
    n_inherit_refused: int = 0       # chain truncated: bucket capped at zero
    n_goal: int = 0                  # what the mix was scaled to: n_asked under
                                     # floor_frame='ask', less under 'taken'
    n_below_floor: int = 0           # tiles a floor asked for and could not get
    n_spilled: int = 0               # tiles that reached a cap-only bucket
    n_jitter: int = 0
    n_taken: int = 0
    n_asked: int = 0

    @property
    def short(self) -> int:
        return max(0, self.n_asked - self.n_taken)

    def line(self) -> str:
        buckets = ' '.join(f'{k}:{v}' for k, v in self.per_bucket.items())
        tail = f'  SHORT {self.short}' if self.short else ''
        if self.n_below_floor:
            tail += f'  BELOW-FLOOR {self.n_below_floor}'
        if self.n_spilled:
            tail += f'  spill {self.n_spilled}'
        if self.n_goal and self.n_goal != self.n_asked:
            tail += f'  scaled to {self.n_goal}'
        return (f'  ds {self.ds:<5g} cand {self.n_candidates:6d} -> adm '
                f'{self.n_admissible:6d} -> took {self.n_taken:5d}/'
                f'{self.n_asked:<5d} (inherit {self.n_inherited}, refused '
                f'{self.n_inherit_refused}, jitter {self.n_jitter})  '
                f'[{buckets}]{tail}')



class TileSampler:
    """Sampler and container. See the module docstring for the spec."""

    def __init__(self, wsi, mask, cfg: Optional[SamplerConfig] = None,
                 slide: str = '', **old_kwargs):
        _refuse_old_kwargs(old_kwargs)
        # SafeSlide only. Tiles read here are handed downstream -- to the
        # encoders, to query_sim, to the pre-tile store -- and a plain handle
        # returns unphotographed pixels as transparent, which every RGB
        # conversion then paints pure black. A black rectangle's border is a
        # straight maximum-contrast edge with two right angles, which is what a
        # corner detector fires on.
        if not hasattr(wsi, 'read_region_rgb'):
            raise TypeError(
                f'TileSampler takes a utilities/SafeSlide.SafeSlide, not a '
                f'{type(wsi).__name__}. A plain OpenSlide paints every scanner '
                f'hole black, and these tiles are read and used, not only drawn')
        self.wsi = wsi
        self.mask = mask
        self.cfg = cfg or SamplerConfig()
        self.slide = slide or getattr(wsi, 'stem', '') or ''
        self.samples: List[Sample] = []
        self.reports: Dict[float, RungReport] = {}
        self._rng = np.random.default_rng(self.cfg.seed)
        #: chain id -> (bucket, score) as scored at `source_rung`. Filled by
        #: `_choose_centres` and read only under bucket_frame='at_inherit'.
        self._inherit_bucket: Dict[int, Tuple[str, float]] = {}

    # ── candidates ──────────────────────────────────────────────────────────

    def _prepare_regions(self, plan: RungPlan) -> None:
        """filter_regions -> merge -> filter_patchable, for THIS rung.

        Undone by `_restore_regions`. The pair has to bracket every rung or the
        rung's candidates depend on which rungs ran before it -- silently, and
        differently depending on the order the caller asked for.

        `filter_patchable` is given the TILE's footprint, not the reserve, and
        that is the whole of the two-rectangle rule below: the TILE is the
        training sample and has to be in tissue; the RESERVE is only context
        for a warp and has to be READABLE. `extract_pretiles`' own docstring
        says it -- "The pre-tile has to be READABLE, not tissue."

        Requiring the reserve to fit a single region was the earlier rule and
        it cost the coarse rungs almost everything: at ds 32 it demanded a
        region 24576 px wide, and BRACS_1228 came back with 21 tiles of 500
        while S1104233 came back with 0. It also threw away exactly the
        positions a warp wants -- a tile at the edge of a region whose pre-tile
        reaches into the glass beside it, which is real glass a microscope
        would also see.
        """
        self.mask.filter_regions(min_ratio=self.cfg.min_region_ratio)
        if self.cfg.merge_regions:
            self.mask.merge_overlapping()
        self.mask.filter_patchable(tile_size=int(plan.footprint_l0), ds=1.0)

    def _restore_regions(self) -> None:
        self.mask.regions_undo()                      # filter_patchable
        if self.cfg.merge_regions:
            self.mask.regions_undo()                  # merge_overlapping
        self.mask.regions_undo()                      # filter_regions

    def _lattice(self, plan: RungPlan) -> np.ndarray:
        """Level-0 top-left corners on the lattice, inside the regions.

        The step is in LEVEL pixels and is converted here, which is the whole
        reason `grid_step` is not a level-0 constant: as a level-0 number it
        would be a disjoint lattice at ds 1 and an 87 per cent overlapping one
        at ds 8. `ReferenceSampler.JITTER_OFFSETS` records the same trap for
        its own offsets.
        """
        # footprint_l0 / tile is level-0 px per OUTPUT px. It equals ds on an
        # 'F' rung and 1 on an 'R' one, where ds degrades rather than
        # magnifies -- using ds directly would space an R lattice 32x too far
        # apart at ds 32 and the rung would come back nearly empty.
        per_px = int(plan.footprint_l0) / max(self.cfg.tile, 1)
        step = max(1, int(round(
            self.cfg.overlap.step_for(self.cfg.tile) * per_px)))
        pad = _margin_of(plan)
        fp = int(plan.footprint_l0)
        sx0, sy0, sx1, sy1 = _scanned_rect(self.mask)
        out = []
        for region in self.mask.tissue_regions:
            # TWO RECTANGLES, INTERSECTED, and they are two requirements on two
            # different things:
            #
            #   the TILE must be in the REGION      it is the training sample,
            #                                       so it has to be on tissue
            #   the RESERVE must be in the SCANNED  it is only context for a
            #   RECTANGLE                           warp, so it only has to be
            #                                       READABLE
            #
            # A pre-tile reaching out of its region into the glass beside it is
            # fine -- that is real glass and a microscope sees it too. A
            # pre-tile reaching past the scanned rectangle is not: those pixels
            # do not exist, SafeSlide fills them with the background colour, and
            # the straight edge between tissue and that flat fill is exactly
            # what a corner detector fires on. One is data; the other is an
            # artefact of where the scanner stopped.
            x0 = max(region.x, sx0 + pad)
            y0 = max(region.y, sy0 + pad)
            x1 = min(region.x + region.w - fp, sx1 - pad - fp)
            y1 = min(region.y + region.h - fp, sy1 - pad - fp)
            if x1 < x0 or y1 < y0:
                continue
            xs = np.arange(x0, x1 + 1, step, dtype=np.int64)
            ys = np.arange(y0, y1 + 1, step, dtype=np.int64)
            if not len(xs) or not len(ys):
                continue
            gx, gy = np.meshgrid(xs, ys, indexing='xy')
            out.append(np.stack([gx.ravel(), gy.ravel()], axis=1))
        if not out:
            return np.zeros((0, 2), dtype=np.int64)
        return np.concatenate(out, axis=0)

    def _random_candidates(self, plan: RungPlan) -> np.ndarray:
        """The control arm: uniform draws inside a uniformly drawn region.

        This is what the previous sampler did, kept so that the lattice's gain
        is measured against the thing that actually ran rather than against
        zero. It shares the gate and the selection with the lattice, so the
        only difference between the two arms is where the candidates came
        from -- which is what makes the comparison mean anything.

        Draws `n_per_rung * max_tries_per_tile` positions and hands them all
        to the same gate. The old code interleaved drawing and accepting,
        which made `max_tries` a budget on ACCEPTED tiles; separating them
        costs a few thousand wasted draws and makes the two arms comparable.
        """
        regions = self.mask.tissue_regions
        if not regions:
            return np.zeros((0, 2), dtype=np.int64)
        n = int(self.cfg.n_per_rung * max(1, self.cfg.max_tries_per_tile))
        fp = int(plan.footprint_l0)
        pad = _margin_of(plan)
        sx0, sy0, sx1, sy1 = _scanned_rect(self.mask)
        out = []
        for _ in range(n):
            region = regions[int(self._rng.integers(0, len(regions)))]
            # The same two rectangles as `_lattice`, or the control arm would
            # be measuring a different corpus and the comparison would be about
            # the bounds rather than about the candidates.
            lo_x, lo_y = max(region.x, sx0 + pad), max(region.y, sy0 + pad)
            hi_x = min(region.x + region.w - fp, sx1 - pad - fp)
            hi_y = min(region.y + region.h - fp, sy1 - pad - fp)
            if hi_x < lo_x or hi_y < lo_y:
                continue
            out.append([int(self._rng.integers(lo_x, hi_x + 1)),
                        int(self._rng.integers(lo_y, hi_y + 1))])
        if not out:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array(out, dtype=np.int64)

    def _candidates(self, plan: RungPlan) -> np.ndarray:
        """Whichever arm the config asked for. One door, so the gate, the
        scorer and the selection cannot diverge between them."""
        if self.cfg.candidates == 'random':
            return self._random_candidates(plan)
        return self._lattice(plan)

    def _reserve_fits(self, x: int, y: int, plan: RungPlan) -> bool:
        """Is the RESERVE around (x, y) wholly inside the SCANNED RECTANGLE?

        `_lattice` bakes this into its range bounds, so a lattice position
        cannot fail it. The two paths that place a position NOT from the
        lattice -- `_top_up` and `_place_inherited` -- have to ask, and until
        2026-08-27 neither did.

        What that cost, and it is a good illustration of why the assertion in
        `extract_pretiles` is an assertion and not a repair: at ds 8 and above
        the richness quotas run the buckets short, `_top_up` starts displacing,
        and a displacement of up to 1.25 tiles walks straight past the region
        edge. The pre-tile then ran off the scanned rectangle by 94 to 2560 px
        -- while every FINE rung passed, because there the lattice filled the
        quotas and the top-up never ran. A repair would have written those
        tiles with a `clip_px` and nobody would have looked.
        """
        fp = int(plan.footprint_l0)
        pad = _margin_of(plan)
        reserve = fp + 2 * pad          # never `plan.reserve`: see SampleMeta
        x0, y0 = x - pad, y - pad
        sx0, sy0, sx1, sy1 = _scanned_rect(self.mask)
        return (x0 >= sx0 and y0 >= sy0
                and x0 + reserve <= sx1 and y0 + reserve <= sy1)

    def _rate(self, xy: np.ndarray, plan: RungPlan):
        """Score and bucket a set of positions. One door for every caller.

        `sample`, `preflight`, `_choose_centres` and `_place_inherited` all
        need the same two arrays, and until 2026-08-27 three of them got the
        bucket a different way -- `_place_inherited` borrowed the NEAREST
        lattice candidate's via `_nearest`, which was an approximation that
        nothing downstream knew about. Now that a zero cap REFUSES an inherited
        tile, an approximate bucket would decide whether a chain lives, so the
        approximation had to go.
        """
        if not len(xy):
            return np.zeros(0, np.float32), np.zeros(0, np.int8)
        score = SCORERS[self.cfg.richness.scorer](self.mask, xy, plan)
        return score, assign_buckets(score, self.cfg.richness.edges)

    def _admissible(self, bucket: np.ndarray) -> np.ndarray:
        """Positions whose bucket has a non-zero cap.

        THIS IS THE TISSUE GATE, ONCE. `tissue_ratio` used to say
        `background <= 1 - ratio` and the caps said it again per bucket, and
        the two disagreeing is the whole 475/500 episode. A zero cap on
        `bg85_95` and `bg95_100` is a gate at 85 per cent background, stated in
        the only place that also decides what happens to everything below it.
        """
        caps = np.asarray(self.cfg.richness.caps, dtype=np.float64)
        if not len(bucket):
            return np.zeros(0, dtype=bool)
        return caps[np.asarray(bucket, dtype=np.int64)] > 0.0

    # ── selection ───────────────────────────────────────────────────────────


    @staticmethod
    def _overlap_against(x: int, y: int, fp: int,
                         taken: np.ndarray) -> float:
        """Largest overlap ratio of one candidate against everything taken.

        Same-rung only, so both footprints are `fp` and the ratio is the shared
        area over `fp**2`. Vectorised because this runs once per candidate
        against a growing set: 500 tiles is 125k comparisons per rung, which is
        nothing in numpy and minutes in a python loop.
        """
        if not len(taken):
            return 0.0
        dx = np.minimum(x + fp, taken[:, 0] + fp) - np.maximum(x, taken[:, 0])
        dy = np.minimum(y + fp, taken[:, 1] + fp) - np.maximum(y, taken[:, 1])
        area = np.clip(dx, 0, None) * np.clip(dy, 0, None)
        return float(area.max()) / float(fp * fp)

    def _select(self, xy: np.ndarray, bucket: np.ndarray, score: np.ndarray,
                plan: RungPlan, inherited: List[SampleMeta],
                report: RungReport) -> List[SampleMeta]:
        """Phases 2 and 3: the floor/cap contract, then the overlap bound.

        THREE PASSES, AND THE ORDER IS THE CONTRACT. A single shuffled pass was
        enough while every bucket carried only a ceiling -- ceilings are
        independent, so whoever the permutation reached first could not take
        anything another bucket was owed. A FLOOR is not independent: one pass
        hands positions to whichever bucket the shuffle reaches, and the bucket
        with the floor discovers at the end that its share is gone.

            pass 1   fill each bucket to its FLOOR, floors only
            pass 2   fill on to its TARGET (floor + its share of the
                     unassigned remainder)
            pass 3   whatever is still missing SPILLS into the buckets with a
                     non-zero cap and headroom, evenly, and a zero cap is
                     never in that set

        Pass 3 is the answer to "填不滿就去填非 0 上限的桶子". Under the settled
        contract the first three buckets have target == cap, so the only
        headroom is `bg50_70` and `bg70_85` -- which is what gives their
        ceilings a role at all. If they cannot supply it either, the rung is
        SHORT, and `RungReport.short` is where that shows up rather than in a
        substitution nobody asked for.

        `inherited` is already placed and already counted -- it consumed quota
        before this was called, and it is EXEMPT from the overlap bound. The
        exemption is why `taken` starts populated but `overlap_budget` does
        not: an inherited tile may overlap freely, and it does not spend the
        budget the free tiles compete for.
        """
        cfg = self.cfg
        rich = cfg.richness
        ov = cfg.overlap
        fp = int(plan.footprint_l0)
        n_ask = cfg.n_per_rung
        names = rich.names

        taken_xy = [[m.x, m.y] for m in inherited]
        out: List[SampleMeta] = []
        state = {'overlapping': 0, 'budget': 0}   # set once n_goal is known

        # WHAT THE SHARES ARE A SHARE OF. Under floor_frame='ask' it is
        # `n_per_rung`, and a rung that cannot supply the mix takes what it has
        # -- the mix drifts and `n_below_floor` says by how much. Under 'taken'
        # the rung is scaled down to the largest count whose mix the supply can
        # hold, so the proportions survive and the count does not.
        #
        # The scale is `min(supply_b / target_b)` over the buckets that have a
        # target, because that is the largest N for which every one of them can
        # still contribute its share. Inherited tiles count as supply: they are
        # already placed and already in a bucket.
        #
        # SUPPLY IS AN UPPER BOUND, NOT A COUNT OF USABLE POSITIONS. It is the
        # candidate histogram before the overlap bound rejects anything, so
        # under a non-disjoint lattice the scale comes out slightly high and
        # the mix drifts anyway -- less than under 'ask', but not to zero.
        # Under the disjoint default every candidate is usable and it is exact.
        n_goal = n_ask
        if rich.floor_frame == 'taken':
            pool = collections.Counter(names[int(b)] for b in bucket)
            for m in inherited:
                pool[m.bucket] += 1
            for i, t in enumerate(rich.targets):
                if t > 0.0:
                    n_goal = min(n_goal, int(pool.get(names[i], 0) / t))
            n_goal = max(1, min(n_ask, n_goal))
        report.n_goal = n_goal
        # A share of the RUNG, and under 'taken' the rung is n_goal -- keeping
        # it on n_ask would let a scaled-down rung spend a budget sized for a
        # full one, which is the overlap bound quietly loosening exactly where
        # the positions are scarcest.
        state['budget'] = int(round(ov.overlapping_share * n_goal))

        # Per-bucket ceilings and the two staged goals, all in TILES not shares.
        cap_n = {names[i]: int(round(c * n_goal)) for i, c in enumerate(rich.caps)}
        floor_n = {names[i]: int(round(f * n_goal)) for i, f in enumerate(rich.floors)}
        target_n = {names[i]: int(round(t * n_goal))
                    for i, t in enumerate(rich.targets)}

        # The inherited ones already sit in buckets, and they count against
        # every goal -- including a cap of zero, which `_place_inherited`
        # refused to place into, so this can only subtract from a live bucket.
        have = {n: 0 for n in names}
        for m in inherited:
            have[m.bucket] = have.get(m.bucket, 0) + 1

        # ONE SHUFFLED STREAM, walked once per pass, not one pass per bucket.
        # Per-bucket pools would fill bucket 0 to its goal before bucket 1 was
        # offered anything, which is invisible while every goal is tight -- the
        # floors are -- and a spatial bias the moment one is loose. A config
        # with no floors at all (the KNN reference bank) has ALL its goals
        # loose, and per-bucket order would have handed it bucket 0's corner of
        # the slide.
        stream = [int(i) for i in self._rng.permutation(len(xy))]
        used = set()

        def fill_to(goal: Dict[str, int]) -> None:
            for idx in stream:
                if len(out) + len(inherited) >= n_goal:
                    return
                if idx in used:
                    continue
                name = names[int(bucket[idx])]
                if have[name] >= min(goal[name], cap_n[name]):
                    continue
                x, y = int(xy[idx, 0]), int(xy[idx, 1])
                arr = np.asarray(taken_xy, dtype=np.int64) if taken_xy \
                    else np.zeros((0, 2), dtype=np.int64)
                ratio = self._overlap_against(x, y, fp, arr)
                if ratio > ov.max_overlap_ratio:
                    continue
                if ratio > 0.0:
                    if state['overlapping'] >= state['budget']:
                        continue
                    state['overlapping'] += 1
                have[name] += 1
                used.add(idx)
                taken_xy.append([x, y])
                out.append(SampleMeta(
                    slide=self.slide, ds=plan.rung_ds, level=plan.level,
                    x=x, y=y, tile_size=cfg.tile, read_size=plan.read_size,
                    footprint_l0=fp, reserve_l0=fp + 2 * _margin_of(plan),
                    bucket=name, score=float(score[idx]),
                    overlap_max=ratio, inherit_id=-1,
                    stack_kind=plan.stack_kind, origin='grid'))

        fill_to(floor_n)                                          # pass 1
        report.n_below_floor = sum(max(0, floor_n[n] - have[n]) for n in names)
        fill_to(target_n)                                         # pass 2

        # Pass 3. Re-deal the deficit over the buckets that still have room,
        # evenly, and repeat -- a bucket that cannot take its share hands it
        # back rather than stranding it. Bounded by the number of buckets: each
        # round either places a tile or removes a bucket from the set.
        spillable = [names[i] for i in spill_order(rich.caps, rich.targets)]
        goal = dict(target_n)
        for _ in range(len(names) + 1):
            deficit = n_goal - (len(out) + len(inherited))
            room = [n for n in spillable if have[n] < cap_n[n]]
            if deficit <= 0 or not room:
                break
            share = -(-deficit // len(room))          # ceil, so it converges
            for name in room:
                goal[name] = min(cap_n[name], have[name] + share)
            before = len(out)
            fill_to(goal)
            report.n_spilled += len(out) - before
            if len(out) == before:
                break

        report.n_jitter += self._top_up(
            out, taken_xy, plan,
            {n: max(0, min(goal[n], cap_n[n]) - have[n]) for n in names},
            n_goal, len(inherited))
        return out

    def _top_up(self, out: List[SampleMeta], taken_xy: List[List[int]],
                plan: RungPlan, want: Dict[str, int], n_ask: int,
                n_inherited: int) -> int:
        """Displace an existing tile when the lattice has run out.

        Every offer moves a full tile in one axis, so a displaced tile shares
        NO pixels with its parent; and none is a multiple of half a tile, so
        none lands back on a lattice position -- which could not help, because
        the bucket was short precisely where the lattice ran out.

        The displaced tile keeps its PARENT's bucket rather than being
        rescored. Rescoring would need the mask again per offer, and the offer
        is disjoint from the parent, so its own score is a different number
        that no quota asked for.
        """
        ov = self.cfg.overlap
        cap = int(round(ov.jitter_cap * n_ask))
        added = 0
        if cap <= 0 or not out:
            return 0
        fp = int(plan.footprint_l0)
        for parent in list(out):
            if added >= cap or len(out) + n_inherited >= n_ask:
                break
            if want.get(parent.bucket, 0) <= 0:
                continue
            for dx, dy in ov.jitter_offsets:
                if added >= cap or want.get(parent.bucket, 0) <= 0:
                    break
                # The offsets are fractions of the tile, and the tile covers
                # `fp` level-0 px, so this is the displacement in level-0 -- at
                # every rung, without the constant having to know which.
                x = int(parent.x + round(dx * fp))
                y = int(parent.y + round(dy * fp))
                # Scored on its OWN position rather than inheriting the
                # parent's bucket. The old code kept the parent's on the
                # argument that rescoring "would need the mask again per
                # offer" -- but the offer is a FULL TILE away and disjoint, so
                # the parent's bucket was a claim about different pixels. One
                # `white_fractions` call on one position is the cost.
                offer = np.array([[x, y]], dtype=np.int64)
                _, ob = self._rate(offer, plan)
                if not bool(self._admissible(ob)[0]):
                    continue
                if not self._reserve_fits(x, y, plan):
                    continue
                arr = np.asarray(taken_xy, dtype=np.int64)
                if self._overlap_against(x, y, fp, arr) > ov.max_overlap_ratio:
                    continue
                want[parent.bucket] -= 1
                taken_xy.append([x, y])
                out.append(SampleMeta(
                    slide=self.slide, ds=plan.rung_ds, level=plan.level, x=x, y=y,
                    tile_size=self.cfg.tile, read_size=plan.read_size,
                    footprint_l0=fp, reserve_l0=fp + 2 * _margin_of(plan), bucket=parent.bucket, score=parent.score,
                    overlap_max=0.0, inherit_id=-1,
                    stack_kind=plan.stack_kind, origin='jitter',
                    parent_x=parent.x, parent_y=parent.y))
                added += 1
        return added

    # ── the inheritance set: phase 1, before any rung is filled ─────────────

    def _choose_centres(self, plans: Sequence[RungPlan]) -> np.ndarray:
        """Level-0 centres carried to every rung. Chosen ONCE, before phase 2.

        Fixed first because it consumes the quotas -- fill them first and the
        set has nowhere to go -- and because it must be validated at every rung
        before any rung is committed to. `source_rung` defaults to the FINEST,
        which has the most candidates; the coarsest would instead guarantee
        every centre fits everywhere, which is why it is a field and not a
        constant.
        """
        cfg = self.cfg
        if cfg.inherit.share <= 0.0 or not plans:
            return np.zeros((0, 2), dtype=np.int64)

        want = cfg.inherit.source_rung
        source = (min(plans, key=lambda q: q.rung_ds) if want is None else
                  min(plans, key=lambda q: abs(q.rung_ds - want)))
        self._prepare_regions(source)
        try:
            xy = self._candidates(source)
            if len(xy):
                _, b = self._rate(xy, source)
                xy = xy[self._admissible(b)]
        finally:
            self._restore_regions()
        if not len(xy):
            return np.zeros((0, 2), dtype=np.int64)

        n = min(len(xy), int(round(cfg.inherit.share * cfg.n_per_rung)))
        pick = self._rng.choice(len(xy), size=n, replace=False)

        # The chain's bucket, decided HERE and not in the rung loop. Under
        # bucket_frame='at_inherit' this is the value carried to every rung, so
        # it has to be settled before any rung is filled -- the same reason the
        # centres themselves are.
        score, bucket = self._rate(xy[pick], source)
        names = cfg.richness.names
        self._inherit_bucket = {
            i: (names[int(bucket[i])], float(score[i])) for i in range(n)}

        half = int(source.footprint_l0) / 2.0
        return (xy[pick] + half).astype(np.int64)          # centres, level 0

    def _place_inherited(self, centres: np.ndarray, plan: RungPlan,
                         report: RungReport) -> List[SampleMeta]:
        """The carried centres, as tiles of THIS rung. Exempt from the bound.

        A centre chosen at ds 1 and carried to ds 32 has a footprint 32x
        larger, so two centres a few hundred px apart are now almost the same
        tile. Enforcing the overlap bound here would drop members and leave
        `inherit_id`s that do not resolve -- which a survival analysis reads as
        "the keypoint died", when it means "the tile was never cut". So they
        are exempt, and how many of them breach the bound is REPORTED instead
        of being silently allowed.

        A ZERO CAP IS NOT EXEMPT, and that is deliberate. Every other quota is
        advisory for an inherited tile -- `_select` subtracts it from `have`
        and moves on -- because a chain that has to break to satisfy a ceiling
        costs more than the ceiling is worth. A cap of zero is a different
        statement: `bg85_95` and `bg95_100` at 0 is where the tissue gate went,
        and a gate that the inheritance set walks through is not a gate.

        EXPECT THE BREAKS TO CLUSTER AT THE COARSE RUNGS. `bucket_frame` is
        'per_rung', so the bucket is recomputed against THIS rung's footprint,
        and the footprint grows 32x from ds 1 to ds 32 -- a centre at 10 per
        cent background down there is easily at 90 up here. So the rungs where
        the corpus is already thinnest are the ones that lose chain members,
        and `n_inherit_refused` is the column that says how many.

        THE CHAIN TRUNCATES, it does not develop a hole. A refusal at ds 16
        stops the chain there and ds 32 is not attempted, because "the same
        physical tissue at every magnification" is what an `inherit_id` claims
        -- and a chain missing its middle rung does not support that claim on
        either side of the gap. `_truncated` carries the refusal forward.
        """
        fp = int(plan.footprint_l0)
        half = fp // 2
        out: List[SampleMeta] = []
        taken: List[List[int]] = []
        names = self.cfg.richness.names
        caps = self.cfg.richness.caps
        for chain, (cx, cy) in enumerate(centres):
            if chain in self._truncated:
                continue                       # broke at a finer rung already
            x, y = int(cx) - half, int(cy) - half
            if not self._reserve_fits(x, y, plan):
                self._truncated.add(chain)
                continue                       # a reserve that runs off the scan
            pos = np.array([[x, y]], dtype=np.int64)
            score, b = self._rate(pos, plan)
            bi = int(b[0])
            if caps[bi] <= 0.0:
                report.n_inherit_refused += 1
                self._truncated.add(chain)
                continue                       # this rung breaks the chain
            arr = np.asarray(taken, dtype=np.int64) if taken \
                else np.zeros((0, 2), dtype=np.int64)
            ratio = self._overlap_against(x, y, fp, arr)
            if ratio > self.cfg.overlap.max_overlap_ratio:
                report.n_inherit_breaching += 1
            taken.append([x, y])
            out.append(SampleMeta(
                slide=self.slide, ds=plan.rung_ds, level=plan.level, x=x, y=y,
                tile_size=self.cfg.tile, read_size=plan.read_size,
                footprint_l0=fp, reserve_l0=fp + 2 * _margin_of(plan),
                bucket=names[bi], score=float(score[0]),
                overlap_max=ratio, inherit_id=int(chain),
                stack_kind=plan.stack_kind, origin='inherit'))
        report.n_inherited = len(out)
        return out


    # ── the loop ────────────────────────────────────────────────────────────

    def sample(self, plans: Sequence[RungPlan] = (), **old_kwargs
               ) -> 'TileSampler':
        """Fill the container. The three phases, in the order they must run.

        Takes RUNG PLANS, not a level and a count. A plan says which level to
        read, what the footprint is and what must fit -- which is what lets one
        sampler serve a 4x pyramid and a 2x one without knowing which it is on.
        """
        if old_kwargs or not plans:
            raise _migration_error(
                'sample(n=..., level=..., tissue_ratio=..., max_tries=...)',
                'sample() now takes a sequence of RungPlan. n_per_rung is a '
                'SamplerConfig field; tissue_ratio is GONE (richness.caps of '
                'zero is the gate now); max_tries is '
                'max_tries_per_tile and only the random arm reads it. Build '
                'the plans with RungPlan.fov(...) or RungPlan.resolution(...), '
                'or from common/DsLadder.py.')
        cfg = self.cfg
        # FINE TO COARSE, and the truncation is why. A chain breaks at the
        # rung where its footprint first reaches into a zero-capped bucket,
        # and every COARSER rung must then be skipped -- which is only
        # expressible if the coarser ones have not been filled yet. The
        # centres are chosen at the finest rung by the same logic.
        ds_order = [q.rung_ds for q in plans]
        if ds_order != sorted(ds_order):
            raise ValueError(
                f'plans must ascend in rung_ds, got {ds_order}. A chain '
                f'truncates at the rung where it first lands in a zero-capped '
                f'bucket and skips every coarser one, so the coarser rungs '
                f'have to come later')
        # THE PLANS DECIDE WHAT IS CUT; THE CONFIG DECIDES WHAT THE STORE SAYS
        # WAS CUT. `inherit.stack_kind` reaches exactly two places -- the
        # `sampler_id` hash and the `stack_kind` field of the written meta --
        # and NOTHING reads it while choosing tiles (`SampleMeta.stack_kind`
        # comes from `plan.stack_kind`). So the two disagreeing writes a corpus
        # of 'F' tiles whose meta.json says 'R', silently, and a survival table
        # built on that meta is about the other question entirely (spec.md 3.2:
        # a survival number that does not say which axis is meaningless).
        kinds = {q.stack_kind for q in plans}
        if kinds != {cfg.inherit.stack_kind}:
            raise ValueError(
                f'inherit.stack_kind is {cfg.inherit.stack_kind!r} and the '
                f'plans are {sorted(kinds)}. Those are the label and the thing '
                f'labelled: the plans cut the tiles and the config names them '
                f'in meta.json, so a mismatch stores one axis under the other '
                f"axis's name. Build the plans with DsLadder (which makes 'F') "
                f"or resolution_plan (which makes 'R'), and set the config to "
                f'match')

        self._truncated = set()
        centres = self._choose_centres(plans)               # phase 1
        samples: List[SampleMeta] = []

        for plan in plans:
            # EVERY RUNG STARTS FROM THE SAME SEED. `_select` draws a
            # permutation of the candidates, so a shared stream makes ds 2's
            # tiles depend on how many draws ds 1 happened to make -- and when
            # one sampler began serving every rung (2026-09-01, so that
            # inheritance could fix one set of centres across them) that is
            # exactly what happened. `sampler_id` hashes the CONFIG, not the
            # code, so the corpus changed underneath an unchanged id: the one
            # failure that field exists to prevent.
            #
            # Reset here rather than seeding per rung, because that also
            # restores the behaviour of the per-rung samplers this replaced: at
            # `inherit.share = 0` the tiles are bit-identical to the corpus of
            # 2026-08-27, so it can still be regenerated. The centres are drawn
            # ABOVE this loop and keep their own draw, so a rung's tiles do not
            # depend on whether inheritance ran either.
            self._rng = np.random.default_rng(cfg.seed)

            report = RungReport(ds=plan.rung_ds, n_asked=cfg.n_per_rung)
            self._prepare_regions(plan)
            try:
                inherited = self._place_inherited(centres, plan, report)
                xy = self._candidates(plan)
                report.n_candidates = len(xy)
                score, bucket = self._rate(xy, plan)
                names = cfg.richness.names
                for i, name in enumerate(names):
                    report.supply[name] = int((bucket == i).sum())
                keep = self._admissible(bucket)
                xy, score, bucket = xy[keep], score[keep], bucket[keep]
                report.n_admissible = len(xy)

                if cfg.richness.bucket_frame == 'at_inherit':
                    # The bucket decided at `source_rung`, carried unchanged.
                    # A chain then has ONE bucket -- what a per-bucket survival
                    # analysis needs -- and the quotas act only on the
                    # remainder, which is what that costs.
                    for m in inherited:
                        got = self._inherit_bucket.get(m.inherit_id)
                        if got:
                            m.bucket, m.score = got
                else:
                    # Recomputed at this rung. The footprint is this rung's, so
                    # the bucket is too; a chain drifts between buckets as it
                    # climbs, and that is the price of each rung's distribution
                    # being what the quotas asked for.
                    pass          # _place_inherited already rated them

                chosen = self._select(xy, bucket, score, plan, inherited,
                                      report)
                rung = inherited + chosen
                report.n_taken = len(rung)
                for name in names:
                    report.per_bucket[name] = sum(
                        1 for m in rung if m.bucket == name)
                samples.extend(rung)
            finally:
                self._restore_regions()
            self.reports[plan.rung_ds] = report

        self.samples = [Sample(m) for m in samples]
        return self

    def preflight(self, plans: Sequence[RungPlan]) -> List[RungReport]:
        """What every rung can offer, before a single pixel is read.

        The whole plan is geometry over the mask, so a corpus that cannot be
        cut is knowable in seconds rather than after hours of reads. Uses the
        same lattice and the same gate as `sample`, so a disagreement between
        the two is a bug and not a tolerance.
        """
        out = []
        for plan in plans:
            report = RungReport(ds=plan.rung_ds, n_asked=self.cfg.n_per_rung)
            self._prepare_regions(plan)
            try:
                xy = self._candidates(plan)
                report.n_candidates = len(xy)
                _, bucket = self._rate(xy, plan)
                names = self.cfg.richness.names
                for i, name in enumerate(names):
                    report.supply[name] = int((bucket == i).sum())
                    report.per_bucket[name] = report.supply[name]
                report.n_admissible = int(self._admissible(bucket).sum())
            finally:
                self._restore_regions()
            out.append(report)
        return out

    # ── four ways in ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> Union[Sample, 'TileSampler']:
        """Random access. A slice returns a container, an int returns a Sample."""
        if isinstance(index, slice):
            return self._view(self.samples[index])
        return self.samples[index]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def _view(self, samples: List[Sample]) -> 'TileSampler':
        """A container over a subset, sharing this one's slide and config.

        A view rather than a copy so `where(...).stacks()` composes, and
        sharing the config so a subset's `sampler_id` still says which corpus
        it came out of -- a filtered set is a QUESTION about a corpus, not a
        different corpus.
        """
        other = TileSampler.__new__(TileSampler)
        other.wsi, other.mask, other.cfg = self.wsi, self.mask, self.cfg
        other.slide, other.reports = self.slide, self.reports
        other._rng = self._rng
        other._inherit_bucket = getattr(self, '_inherit_bucket', {})
        other.samples = samples
        return other

    def where(self, **eq) -> 'TileSampler':
        """RICHNESS and everything else scalar: a filter over the rows.

            sampler.where(bucket='gt80')
            sampler.where(ds=32.0, origin='grid')

        Returns a container, so it composes with `stacks()` and `__getitem__`.
        """
        keep = []
        for s in self.samples:
            if all(getattr(s.meta, k, None) == v for k, v in eq.items()):
                keep.append(s)
        return self._view(keep)

    def neighbours_of(self, index: int, min_ratio: float = 0.0,
                      same_rung: Optional[bool] = None) -> List[int]:
        """OVERLAP: the pairwise relation. Indices whose footprint overlaps.

        `same_rung=True` is the diversity question -- how much of this rung
        repeats itself. `same_rung=False` is the leakage question, and it is
        the same relation inheritance registers: content on two rungs at once.
        `None` asks both.
        """
        me = self.samples[index].meta
        out = []
        for i, s in enumerate(self.samples):
            if i == index:
                continue
            if same_rung is not None and (s.meta.ds == me.ds) != same_rung:
                continue
            if me.overlap_with(s.meta) > min_ratio:
                out.append(i)
        return out

    def stack(self, inherit_id: int) -> List[Sample]:
        """INHERITANCE: one chain, finest rung first."""
        members = [s for s in self.samples if s.meta.inherit_id == inherit_id]
        return sorted(members, key=lambda s: s.meta.ds)

    def stacks(self, complete_only: Optional[bool] = None
               ) -> Dict[int, List[Sample]]:
        """INHERITANCE: every chain, grouped.

        Complete only by default, because the alternative is a number that
        lies: a four-rung chain handed over as if it were six reads as "the
        keypoint died at the two missing rungs" when it means "those rungs
        never cut a tile there", and telling those apart is the whole of a
        survival measurement. `complete_only=False` returns the rest as well;
        `incomplete()` says what each is missing.
        """
        if complete_only is None:
            complete_only = self.cfg.inherit.on_incomplete == 'drop'
        chains: Dict[int, List[Sample]] = {}
        for s in self.samples:
            if s.meta.inherit_id >= 0:
                chains.setdefault(s.meta.inherit_id, []).append(s)
        want = len(self.reports) or len({s.meta.ds for s in self.samples})
        out = {}
        for cid, members in chains.items():
            if complete_only and len(members) < want:
                continue
            out[cid] = sorted(members, key=lambda s: s.meta.ds)
        return out

    def incomplete(self) -> Dict[int, List[float]]:
        """Which rungs each broken chain is missing. The other half of `stacks`."""
        rungs = sorted(self.reports) or sorted({s.meta.ds for s in self.samples})
        chains: Dict[int, set] = {}
        for s in self.samples:
            if s.meta.inherit_id >= 0:
                chains.setdefault(s.meta.inherit_id, set()).add(s.meta.ds)
        return {cid: [d for d in rungs if d not in got]
                for cid, got in chains.items() if len(got) < len(rungs)}

    def unregistered_overlaps(self, min_ratio: float = 0.0
                              ) -> List[Tuple[int, int, float]]:
        """Cross-rung content sharing that inheritance did NOT register.

        The same relation as a chain, minus the registration. This is the
        query to run before splitting train from validation: a ds 1 tile lying
        inside a ds 32 tile is the same tissue on both sides of the split, and
        nothing else in the pipeline would notice. Costs nothing extra -- the
        pairwise index has to exist for `neighbours_of` anyway.
        """
        out = []
        for i in range(len(self.samples)):
            a = self.samples[i].meta
            for j in range(i + 1, len(self.samples)):
                b = self.samples[j].meta
                if a.ds == b.ds:
                    continue
                if a.inherit_id >= 0 and a.inherit_id == b.inherit_id:
                    continue                          # registered: a chain
                r = a.overlap_with(b)
                if r > min_ratio:
                    out.append((i, j, r))
        return out

    # ── materialise / release ───────────────────────────────────────────────

    def materialise(self, reader=None) -> 'TileSampler':
        """Read every sample's pixels. The resident half.

        `reader` defaults to this sampler's own handle, which is right for a
        single process and wrong inside a DataLoader worker -- there the worker
        opens its own and passes it, because an openslide handle does not
        survive being pickled across the fork.
        """
        r = reader if reader is not None else self.wsi
        for s in self.samples:
            s.materialise(r)
        return self

    def release(self) -> 'TileSampler':
        """Drop every image, keep every meta. The streaming half."""
        for s in self.samples:
            s.release()
        return self

    def images(self) -> List[np.ndarray]:
        """Resident pixels, in order. Raises rather than reading silently.

        A `read_all()` that quietly materialised would make the streaming mode
        impossible to hold: the whole point is that pixels are read once and
        dropped, and a helper that re-reads them turns a bounded loop into an
        unbounded one without saying so.
        """
        missing = [i for i, s in enumerate(self.samples) if s.image is None]
        if missing:
            raise RuntimeError(
                f'{len(missing)} of {len(self.samples)} samples hold no '
                f'pixels; call materialise() first. This does not read for '
                f'you, because a helper that did would silently undo the '
                f'streaming mode')
        return [s.image for s in self.samples]

    def drop_holes(self, reader=None, min_valid: float = 0.95
                   ) -> Dict[float, int]:
        """Discard tiles the scanner never photographed. READS PIXELS.

        A separate pass and not part of `sample()`, deliberately: everything
        `sample` does is geometry over the mask, which is what makes
        `preflight` able to say whether a corpus can be cut before a single
        pixel is read. Folding a read into it would cost that property for a
        check most callers do not need.

        Whether a tile was photographed is a property of (location, LEVEL) and
        cannot be answered from a mask or shared between rungs -- a corrupt
        stored tile at level 0 says nothing about level 3. So this runs per
        tile, and a chain is re-checked at every rung: `ReferenceSampler` calls
        that out for the same reason ("a correspondence with holes in it is not
        a correspondence").

        Returns how many went, per rung. Chains that lose a member become
        incomplete rather than silently short, which `stacks()` then drops and
        `incomplete()` names.
        """
        r = reader if reader is not None else self.wsi
        if not hasattr(r, 'read_region_valid'):
            raise TypeError(
                f'{type(r).__name__} has no read_region_valid, so it cannot '
                f'say which pixels were photographed. SafeSlide does; a plain '
                f'OpenSlide reports a hole as transparent and every RGB '
                f'conversion then paints it black, which is indistinguishable '
                f'from densely stained tissue by area alone')
        gone: Dict[float, int] = {}
        keep = []
        for sample in self.samples:
            m = sample.meta
            valid = r.read_region_valid((m.x, m.y), m.level,
                                        (m.read_size, m.read_size))
            if float(np.asarray(valid).mean()) < min_valid:
                gone[m.ds] = gone.get(m.ds, 0) + 1
                continue
            keep.append(sample)
        self.samples = keep
        return gone

    # ── persistence ─────────────────────────────────────────────────────────

    #: `index.csv` columns. The three axes join the coordinates, so every
    #: question the container answers survives to disk. `neighbours` is NOT
    #: here: it is derived from the coordinates, and a stored copy is a second
    #: thing to keep in step with them.
    COLUMNS = ('index', 'slide', 'ds', 'level', 'x', 'y', 'tile_size',
               'read_size', 'footprint_l0', 'reserve_l0', 'bucket', 'score',
               'overlap_max',
               'inherit_id', 'stack_kind', 'origin', 'parent_x', 'parent_y')

    def save(self, folder: Union[str, Path], with_images: bool = False
             ) -> Path:
        """`index.csv` + `meta.json`, and the PNGs when asked for.

        The format is `utilities/PreTileStore.py`'s, not a second one: an
        `index.csv` beside a `meta.json` beside one PNG per record. What this
        adds is the axis columns.

        `with_images=False` is the streaming corpus -- coordinates and
        metadata, with the pixels read on demand by whoever loads it. That is
        what a Dataset wants, and it is why the two are one method with a flag
        rather than two formats.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        if with_images:
            import cv2                                      # noqa: PLC0415
            for i, s in enumerate(self.samples):
                if s.image is None:
                    s.materialise(self.wsi)
                cv2.imwrite(str(folder / f'{i:06d}.png'),
                            cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR))

        with open(folder / 'index.csv', 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(self.COLUMNS + (('file',) if with_images else ()))
            for i, s in enumerate(self.samples):
                m = s.meta
                row = [i] + [getattr(m, c) for c in self.COLUMNS[1:]]
                if with_images:
                    row.append(f'{i:06d}.png')
                writer.writerow(row)

        meta = {
            'sampler_id': self.cfg.sampler_id(),
            'slide': self.slide,
            'n_samples': len(self.samples),
            'with_images': bool(with_images),
            'stack_kind': self.cfg.inherit.stack_kind,
            'config': _config_parts(self.cfg),
            'provenance': self.cfg.provenance(),
            'rungs': {f'{d:g}': dataclasses.asdict(r)
                      for d, r in self.reports.items()},
        }
        with open(folder / 'meta.json', 'w') as handle:
            json.dump(meta, handle, indent=2)
        return folder

    @classmethod
    def load(cls, folder: Union[str, Path], wsi=None, mask=None,
             cfg: Optional[SamplerConfig] = None) -> 'TileSampler':
        """Offline: metadata alone, or metadata plus a handle to read with.

        `wsi=None` is legitimate and is the point of the streaming corpus -- a
        Dataset loads the table in the parent process, forks, and each worker
        opens its own handle. So this cannot require one, and `materialise`
        takes the reader as an argument for the same reason.
        """
        folder = Path(folder)
        with open(folder / 'meta.json') as handle:
            meta = json.load(handle)
        rows = []
        with open(folder / 'index.csv', newline='') as handle:
            for row in csv.DictReader(handle):
                rows.append(SampleMeta(
                    slide=row['slide'], ds=float(row['ds']),
                    level=int(row['level']), x=int(row['x']), y=int(row['y']),
                    tile_size=int(row['tile_size']),
                    read_size=int(row['read_size']),
                    footprint_l0=int(row['footprint_l0']),
                    reserve_l0=int(row.get('reserve_l0', 0) or 0),
                    bucket=row['bucket'], score=float(row['score']),
                    overlap_max=float(row['overlap_max']),
                    inherit_id=int(row['inherit_id']),
                    stack_kind=row['stack_kind'], origin=row['origin'],
                    parent_x=int(row['parent_x']),
                    parent_y=int(row['parent_y'])))

        out = cls.__new__(cls)
        out.wsi, out.mask = wsi, mask
        out.cfg = cfg or SamplerConfig()
        out.slide = meta.get('slide', '')
        out.samples = [Sample(m) for m in rows]
        out.reports = {}
        out._rng = np.random.default_rng(out.cfg.seed)
        out._inherit_bucket = {}

        stored = meta.get('sampler_id', '')
        if cfg is not None and stored and stored != cfg.sampler_id():
            raise ValueError(
                f'{folder} was cut with sampler_id {stored} and the config '
                f'passed in hashes to {cfg.sampler_id()}. Loading it under the '
                f'wrong config would report the wrong axes for every row -- '
                f'pass the right config, or none, and read the stored one')
        return out

    # ── summary ─────────────────────────────────────────────────────────────

    def summary(self) -> 'TileSampler':
        print(f'slide      : {self.slide}')
        print(f'sampler_id : {self.cfg.sampler_id()}')
        print(f'samples    : {len(self.samples)}')
        for ds in sorted(self.reports):
            print(self.reports[ds].line())
        chains = self.stacks(complete_only=True)
        broken = self.incomplete()
        if chains or broken:
            print(f'chains     : {len(chains)} complete, {len(broken)} broken')
        return self


# ── what the previous sampler's callers hit ──────────────────────────────────

class TileInfo:
    """The old per-tile record. Refuses, and says what replaced it.

    Not an alias for `SampleMeta`: the old record carried `mpp` and no axis at
    all, so code that reads `info.mpp` off a `SampleMeta` would get an
    AttributeError somewhere unrelated. An explicit refusal here puts the
    message at the line that needs changing.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            'TileInfo is gone; TileSampler now yields Sample objects whose '
            '.meta is a SampleMeta. The fields moved: level/x/y are the same, '
            'tile_size is the OUTPUT side and read_size is what is read, and '
            'mpp is no longer carried (ask the slide -- it was a copy of '
            'wsi.base_mpp * level_downsample and could go stale). See the '
            'module docstring.')


def _migration_error(what: str, instead: str) -> TypeError:
    return TypeError(
        f'{what} is gone. {instead}\n'
        f'  There is deliberately NO default sampling behaviour: the three '
        f'axes -- richness, overlap, inheritance -- all change which tiles '
        f'come out, all go into sampler_id, and a default would be a value '
        f'that was set and that nobody noticed. Say what you want.\n'
        f'  The old behaviour is SamplerConfig(candidates="random"), kept as '
        f'the control arm rather than as a fallback.')


def _refuse_old_kwargs(kwargs: Dict[str, object]) -> None:
    old = {'tile_size', 'seed'} & set(kwargs)
    if old:
        raise _migration_error(
            f'TileSampler(..., {", ".join(sorted(old))}=...)',
            'Both moved into SamplerConfig: '
            'TileSampler(wsi, mask, SamplerConfig(tile=256, seed=42)).')
