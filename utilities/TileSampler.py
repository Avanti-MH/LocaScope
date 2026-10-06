"""Choose which tiles a slide contributes, on three controlled axes, and carry
them as objects that know what they are.

    cfg = SamplerConfig(
        n_per_rung=500,
        richness=RichnessConfig(scorer='background'),
        overlap=OverlapConfig(step=1.0, max_overlap_ratio=0.0),
        inherit=InheritConfig(stack_kind='F', share=0.4, source_rung=1.0),
    )
    camera  = ReadSpec(256, 256)                   # ReadGeometry: how big, what is read
    sampler = TileSampler(wsi, mask, cfg).sample(PlanSpec("ladder", rungs, camera).plans_for(wsi))

    sampler.where(bucket='bg70_85')     # richness  -> a filtered container
    sampler.neighbours_of(12)           # overlap   -> indices that overlap it
    sampler.stacks()                    # inherit   -> complete chains
    sampler[7]                          # random    -> one Sample

=============================== SPEC ===============================

THREE AXES, BOTH CONTROLS AND INDEXES
=======================================
Every axis is set when sampling and queried afterwards, because those are the
same information read twice. All three go into `sampler_id`: a corpus cut at
one setting is not the corpus cut at another.

    richness   what is IN a tile. Candidates are scored, bucketed, and each
               bucket carries a FLOOR and a CAP -- two constraints of different
               natures, which is why they are two tuples. A cap is always
               achievable; a floor demands that the slide actually has the
               tiles. `RichnessConfig` holds the settled seven-bucket contract
               and the arithmetic guards on it. The scorer is pluggable:
               background fraction (mask only, free), stain saturation,
               entropy.

               There is no separate tissue gate: it would score the same
               quantity as the buckets and could empty a bucket the quota had
               reserved. A cap of zero on the top buckets is that gate.
    overlap    how much two tiles of the SAME rung may share. Candidates come
               off a lattice whose step is a config field, a fraction of the
               footprint: step=0.5 is a deliberate 50 per cent lattice and
               step=1.0 a disjoint one. Two further bounds: how much
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
CONTAINS, and that is the thing every consumer of a single rung depends on.

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
A correspondence with holes in it is not a correspondence. `stacks()` returns
complete chains only; `stacks(complete_only=False)` returns the rest with the
missing rungs named. A four-rung chain returned as if it were six reads as
"the keypoint died at ds 16" when it means "ds 16 never sampled it", and those
two are the whole of Stage B's conclusion.

A Sample CARRIES COORDINATES, NEVER A HANDLE, NEVER PIXELS
============================================================
An openslide handle cannot be pickled, so a `Sample` holding one kills a
DataLoader the moment `num_workers > 0`. `SampleMeta` is therefore plain data.

This module decides WHERE; how big a read is (`ReadSpec`) and the read itself
(`SlideReader`) are not its.
`SlideReader(wsi, resize='area').read_samples(sampler, ReadSpec(tile, tile))`
reads them, with the handle the caller owns: a Dataset opens one per worker
and builds its reader on it.

COST
=====
Overlap is O(n^2), so the index is built PER SLIDE and never globally: tiles of
different slides cannot overlap. Within a slide both same-rung and cross-rung
pairs are computed -- 3000 tiles is 4.5M pairs, once, on demand. `neighbours`
is never stored: it is derived from the coordinates, and a stored copy is a
second thing to keep in step with them.

PERSISTENCE
============
`save()` writes the metadata table: `index.csv` and `meta.json`, the layout
`Store.PreTileStore` uses, whose PNGs `extract_pretiles` writes through a
camera. The axis columns join `index.csv`; `stack_kind` and the config belong
to the batch and go in `meta.json`.

THE CONTROL ARM
================
`candidates='random'` draws a region uniformly, a position uniformly inside it,
and keeps it if the tissue gate passes. It controls none of the three axes and
deduplicates nothing; it is there so that what the lattice buys is a
measurement.
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

# `RungPlan` is `DsLadder`'s: it resolves a rung to (level, read_size, shrink,
# footprint) for a slide, and is the one place that knows ds 2 is native on a
# 2x pyramid and a shrink on a 4x one.

from ConfigIdentity import (IdentifiedConfig, record,          # noqa: E402
                            record_diff, register)
from Cache import (atomic_dir, check_source, source_key,         # noqa: E402
                   wsi_stem_of)
from DsLadder import DsLadder, RungPlan                          # noqa: E402


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


@dataclass(frozen=True)
class PlanSpec:
    """Which rungs to cut, as a VALUE -- before any slide is open.

    `RungPlan`s depend on the slide's pyramid, so they cannot exist until the
    slide is opened; a cache hit exists to avoid opening it. This is the part
    of the plan that is known up front: it names a cache directory (`key`) and
    turns into the slide's own plans on a miss (`plans_for`).

        PlanSpec('ladder', (1, 2, 4, 8, 16, 32), camera)   DsLadder's rungs
        PlanSpec('native', camera=camera)                  the slide's own levels

    WHICH RUNGS × WHICH CAMERA. The rungs say at what scales; the camera
    (`ReadGeometry.ReadSpec`) says how big a footprint is and what is read
    around it, so the sampler never decides either. The tile size is the
    camera's sensor; the reserve is what the camera reads beyond its
    footprint, computed by `ReadSpec.place` from the numbers the camera
    itself will use.

    Both halves are in `key`, so a draw made for one camera is never read back
    for another.
    """
    kind: str = 'ladder'
    rungs: Tuple[float, ...] = ()
    camera: object = None                  # ReadGeometry.ReadSpec

    def __post_init__(self):
        from ReadGeometry import ReadSpec                         # noqa: PLC0415
        if self.kind not in ('ladder', 'native'):
            raise ValueError(f"kind must be 'ladder' or 'native', got {self.kind!r}")
        if self.kind == 'ladder' and not self.rungs:
            raise ValueError('a ladder PlanSpec needs rungs')
        if self.camera is None:
            raise ValueError('a PlanSpec needs the camera it places for -- '
                             'ReadGeometry.ReadSpec(256, 256) is a plain tile')
        if not isinstance(self.camera, ReadSpec):
            raise TypeError(f'camera must be a ReadGeometry.ReadSpec, got '
                            f'{type(self.camera).__name__}')
        object.__setattr__(self, 'rungs',
                           tuple(sorted(float(r) for r in self.rungs)))

    def key(self) -> str:
        if self.kind == 'native':
            base = 'native'
        else:
            base = 'ladder-' + '-'.join(f'{r:g}' for r in self.rungs)
        # '-', never '_': a cache directory is `<region_id>_<sampler_id>_<plan>`
        # and `PreTileCorpus` splits it on the first two underscores
        return f'{base}-{self.camera.key()}'

    def plans_for(self, wsi) -> List[RungPlan]:
        tile = self.camera.long_side
        if self.kind == 'native':
            plans = native_plans(wsi, tile)
        else:
            plans = DsLadder(rungs=self.rungs).plan_for(wsi, tile)
        return [with_camera(p, self.camera) for p in plans]


def with_camera(plan: RungPlan, camera) -> RungPlan:
    """`plan` placed for `camera`: its tile size is the camera's footprint,
    its FoV rectangle and its reserve come from `ReadSpec.place`.

    'F' rungs only. On an 'R' rung `ds` degrades a level-0 read rather than
    choosing a scale, so the camera's rectangle at `ds` is not what is read
    there; an 'R' plan keeps the reserve it was built with
    (`resolution_plan`)."""
    if plan.stack_kind != 'F':
        return plan
    if plan.tile_size != camera.long_side:
        raise ValueError(f'plan is a {plan.tile_size} px tile and the camera\'s '
                         f'footprint is {camera.long_side} px')
    fw, fh, reserve = camera.place(plan.footprint_l0, plan.rung_ds)
    rect = not camera.square
    return dataclasses.replace(plan, reserve_l0=float(reserve),
                               fov_w_l0=fw if rect else 0,
                               fov_h_l0=fh if rect else 0)


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
    degradation the reader applies afterwards (`SlideReader.read(stack='R')`).

    That asymmetry is why this is a separate constructor and not a flag on the
    ladder: a ladder that returned level 0 for every rung would be a ladder
    that had stopped being about levels.
    """
    return RungPlan(rung_ds=float(ds), level=0, level_ds=1.0,
                    shrink=float(ds), tile_size=int(tile),
                    read_size=int(tile), footprint_l0=float(tile),
                    reserve_l0=float(tile) * int(factor), stack_kind='R')


# ── the pre-tile: the reserve, read back ─────────────────────────────────────
#
# A pre-tile is what a tile camera reads with the pre-tile margin
# (`SlideReader.read` with `ReadSpec(tile, tile, margin_out=centre_margin(...))`): the tile and
# the context reserved around it, centred on it, so a homography of the tile
# samples real tissue instead of a black wedge (SuperPathPoint spec.md 6.6).
# These are the reserve's own arithmetic -- how big, how far in, and the crop
# that gives the tile back -- so they live beside the reserve that the lattice
# honours, not in the store that happens to keep the pixels.

#: spec.md 6.6. The derived bound is 2.49; 3 is that rounded up, and the slack
#: pays for the one approximation in the derivation (a projective map is not
#: exactly a scaling about the centre, so the 1/patch_ratio term is nominal).
PRE_TILE_FACTOR = 3


def pre_tile_px(tile: int, factor: int = PRE_TILE_FACTOR) -> int:
    """The pre-tile side, in tile-resolution pixels.

    Refuses an odd margin. `centre_crop` has to remove `(pre - tile) / 2` from
    each side, and a half-pixel there would put the tile off centre by a
    different amount on the two sides. An odd factor times an even tile is
    always fine; the check is here for the day one of those stops being true.
    """
    pre = int(tile) * int(factor)
    if (pre - int(tile)) % 2:
        raise ValueError(
            f'tile {tile} x factor {factor} = {pre} leaves an odd margin of '
            f'{pre - tile} px, so the tile cannot sit exactly in the centre. '
            f'Use an even tile size or an odd factor')
    return pre


def centre_margin(tile: int, factor: int = PRE_TILE_FACTOR) -> int:
    """Pixels removed from each side to get the tile back out of the pre-tile."""
    return (pre_tile_px(tile, factor) - int(tile)) // 2


def centre_crop(image: np.ndarray, tile: int) -> np.ndarray:
    """The central `tile x tile` of a pre-tile. The ONE definition of the crop.

    Strict on purpose: square, at least `tile`, and an even margin -- a crop
    that silently shifted by half a pixel would move the labels against the
    pixels and look like a slightly worse model. (query_sim's `_centre_crop`
    is lenient by design, for a rectangular FoV; it is a different contract,
    not a second copy of this one.)
    """
    h, w = np.asarray(image).shape[:2]
    if h != w:
        raise ValueError(f'pre-tile must be square, got {h}x{w}')
    if h < tile:
        raise ValueError(f'pre-tile is {h} px, smaller than the {tile} px tile')
    if (h - tile) % 2:
        raise ValueError(
            f'{h} px pre-tile and {tile} px tile leave an odd margin; see '
            f'pre_tile_px()')
    off = (h - tile) // 2
    return image[off:off + tile, off:off + tile]


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


#: Side, in OUTPUT px, of the cells a rectangular FoV's background fraction is
#: averaged over (`TileSampler._fov_background`). 256 is the tile the encoders
#: see, so a cell is one tile of the FoV.
FOV_CELL = 256

#: Registered scorers. A name here is part of `sampler_id`, so adding one is
#: additive and renaming one re-hashes every corpus cut with it.
SCORERS = {
    'background': score_background,
    'saturation': score_saturation,
    'entropy':    score_entropy,
}

def bucket_names(edges: Sequence[float]) -> Tuple[str, ...]:
    """Bucket names, DERIVED from the edges rather than written down, so a
    name cannot drift from the interval it stands for while still being the
    `index.csv` column a corpus is filtered by: `bg30_50` is background in
    [0.30, 0.50).
    """
    cuts = [0.0] + [float(e) for e in edges] + [1.0]
    return tuple(f'bg{int(round(cuts[i] * 100)):02d}_{int(round(cuts[i + 1] * 100)):02d}'
                 for i in range(len(cuts) - 1))


def assign_buckets(score: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Score -> bucket index. Half-open upward: `[cut_i, cut_{i+1})`, top closed.

    `side='right'` puts a score EQUAL to an edge in the upper bucket, which is
    the reading of "背景比高於 15% ~ 低於 30%": 0.15 belongs to the second
    bucket, not the first. A score of exactly 1.0 lands in the last one.
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
    `KnnEstMpp`'s reference bank wants "any admissible tile, no
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
class RichnessConfig(IdentifiedConfig):
    """What is in a tile, and how much of each kind is wanted.

    THE CONTRACT. Seven buckets on the background fraction,
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

    THREE ARITHMETIC GUARDS:

        sum(floors) <= 1      or no rung can satisfy every floor at once
        sum(caps)   >= 1      or the rung is short BY CONSTRUCTION
        floors <= caps        elementwise

    There is no tissue gate besides the caps: `bg85_95` and `bg95_100` at 0 is
    exactly a gate at 85 per cent background.
    """
    scorer: str = 'background'

    #: Cuts on the score, ascending, strictly inside (0, 1). Six cuts, seven
    #: buckets. `bucket_names` derives the names from these.
    edges: Tuple[float, ...] = (0.15, 0.30, 0.50, 0.70, 0.85, 0.95)

    #: Smallest share of a rung each bucket must receive. NOT achievable by
    #: filtering -- see `shortfall_policy` for what happens when the supply is
    #: not there.
    floors: Tuple[float, ...] = (0.05, 0.15, 0.50, 0.0, 0.0, 0.0, 0.0)

    #: Largest share of a rung each bucket may receive. A zero is HARD: it binds
    #: the inherited set too.
    caps: Tuple[float, ...] = (0.15, 0.25, 0.60, 0.20, 0.20, 0.0, 0.0)

    #: 'per_rung' | 'at_inherit'. See the spec above -- the two requirements
    #: are not both satisfiable and this says which is given up.
    bucket_frame: str = 'per_rung'

    #: 'ask' | 'taken'. WHAT THE FLOORS ARE A SHARE OF.
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

    BASELINE = {'scorer': 'background',
                'edges': (0.15, 0.30, 0.50, 0.70, 0.85, 0.95),
                'floors': (0.05, 0.15, 0.50, 0.0, 0.0, 0.0, 0.0),
                'caps': (0.15, 0.25, 0.60, 0.20, 0.20, 0.0, 0.0),
                'bucket_frame': 'per_rung', 'floor_frame': 'ask'}

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
class OverlapConfig(IdentifiedConfig):
    """How much two tiles of the same rung may share.

    THREE KNOBS, AND THEY CAN CONTRADICT EACH OTHER. `step` sets the lattice,
    which fixes the overlap between ADJACENT positions before any bound is
    applied:

        adjacent overlap along one axis = 1 - step
        adjacent overlap on the diagonal = (1 - step) ** 2

    So `step=0.5` means every neighbour overlaps 50 per cent, and setting
    `max_overlap_ratio=0.3` on top of it makes every adjacent pair illegal --
    the lattice silently degenerates to the disjoint one while the identity
    still records 0.5. `check()` refuses that combination instead, with the
    arithmetic in the message.

    Every quantity here is a FRACTION OF THE FOOTPRINT, so the sampler needs
    no tile size: the footprint is the camera's (`PlanSpec.camera`), and the
    same config means the same lattice for a 256 px tile and a 1440 px FoV.
    """
    #: Lattice step as a fraction of the footprint's side. 1.0 is disjoint --
    #: the default, and the only spelling of it. `_lattice` multiplies by the
    #: rung's `footprint_l0`, so the step is the same fraction at every rung:
    #: as a LEVEL-0 constant it would be disjoint at ds 1 and 87 per cent
    #: overlapping at ds 8.
    step: float = 1.0

    #: Largest area fraction any two tiles of a rung may share. 0.0 admits
    #: only positions that touch at most at the border.
    max_overlap_ratio: float = 0.0

    #: Largest share of a rung's tiles that may overlap ANYTHING at all. 0.0
    #: forbids it outright; 1.0 leaves only `max_overlap_ratio` binding.
    overlapping_share: float = 0.0

    #: Displacements offered when a bucket runs out of lattice, **as fractions
    #: of the tile**, so they mean the same overlap at every tile size.
    #:
    #: Two properties, and an entry must have both:
    #:     disjoint from the parent   max(|dx|, |dy|) >= 1
    #:     not a lattice position     dx or dy not a multiple of 1/2
    jitter_offsets: Tuple[Tuple[float, float], ...] = (
        (0.25, 1.0), (1.0, 0.25), (0.75, 1.0), (1.0, 0.75), (1.25, 1.25))

    #: Largest share of a rung that may come from jitter rather than lattice.
    #:
    #: ZERO BY DEFAULT, because the default lattice is disjoint and under a
    #: disjoint lattice the top-up is provably dead. `step == 1.0` TILES
    #: the plane, so every position that is not on the lattice overlaps two to
    #: four lattice tiles -- 75 per cent for four of the five offsets, 56 for
    #: the fifth -- and `max_overlap_ratio = 0` rejects all of them. The lattice
    #: IS the maximum set. A non-zero cap there promises a top-up that cannot
    #: happen, and the bucket stays short with nothing saying why. It means
    #: something as soon as overlap is allowed.
    jitter_cap: float = 0.0

    BASELINE = {'step': 1.0, 'max_overlap_ratio': 0.0, 'overlapping_share': 0.0,
                'jitter_offsets': ((0.25, 1.0), (1.0, 0.25), (0.75, 1.0),
                                   (1.0, 0.75), (1.25, 1.25)),
                'jitter_cap': 0.0}

    def __post_init__(self):
        self.check()

    def check(self) -> None:
        """Refuse a lattice whose own adjacency breaks the bound it is under."""
        if not 0.0 < self.step <= 1.0:
            raise ValueError(
                f'step {self.step} must be in (0, 1]: a fraction of the '
                f'footprint, 1.0 disjoint. A step larger than the footprint '
                f'leaves gaps the sampler cannot see into, which is a mask '
                f'decision and not a lattice one')
        along = 1.0 - self.step
        if along > 0 and self.max_overlap_ratio < along:
            raise ValueError(
                f'step {self.step} makes every adjacent pair overlap '
                f'{along:.0%} along an axis, and max_overlap_ratio is '
                f'{self.max_overlap_ratio:.0%}. Every adjacent position is '
                f'therefore illegal and the lattice degenerates to the disjoint '
                f'one -- while the id still records {self.step}. Set '
                f'step=1.0 and mean it, or raise max_overlap_ratio to at least '
                f'{along:.2f}')
        if self.jitter_cap > 0 and self.max_overlap_ratio <= 0.0:
            raise ValueError(
                f'jitter_cap is {self.jitter_cap:.0%} and max_overlap_ratio is '
                f'0, and those cannot both hold. A lattice of step '
                f'{self.step} covers the plane, so every offer the top-up can '
                f'make overlaps a lattice position by 56 to 75 per cent and is '
                f'rejected -- the lattice is already the largest disjoint set '
                f'there is. Set jitter_cap=0 and mean it, or raise '
                f'max_overlap_ratio to at least 0.75')
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
class InheritConfig(IdentifiedConfig):
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

    BASELINE = {'stack_kind': 'F', 'share': 0.0, 'source_rung': None}
    NOT_IDENTITY = ('on_incomplete',)

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


@register('sampler')
@dataclass(frozen=True)
class SamplerConfig(IdentifiedConfig):
    """Everything that decides WHICH tiles are chosen; its `identity_id` names
    the draw.

    Two runs with different quotas or a different seed are two corpora, and
    the hash gives them two names.

    NO TILE SIZE. How big a footprint is belongs to the camera
    (`PlanSpec.camera`), and every quantity here is relative to the footprint,
    so a sampler config and the camera cannot disagree about it.
    """
    n_per_rung: int = 500
    seed: int = 0

    richness: RichnessConfig = field(default_factory=RichnessConfig)
    overlap:  OverlapConfig  = field(default_factory=OverlapConfig)
    inherit:  InheritConfig  = field(default_factory=InheritConfig)

    #: 'lattice' | 'random'. 'random' is the CONTROL ARM, not a fallback: a
    #: region drawn uniformly, a position drawn uniformly inside it, kept if
    #: the tissue gate passes -- what the lattice's overlap is measured against.
    candidates: str = 'lattice'

    #: Rejection budget for candidates='random'. Meaningless for a lattice,
    #: which enumerates rather than draws, and in the identity anyway because
    #: it changes which tiles come out of the random arm.
    max_tries_per_tile: int = 5

    # No region prep here: it belongs to the mask's recipe
    # (`TissueMaskConfig.min_region_ratio` / `merge`, hashed into `region_id`).

    BASELINE = {'n_per_rung': 500, 'seed': 0, 'richness': 'RichnessConfig',
                'overlap': 'OverlapConfig', 'inherit': 'InheritConfig',
                'candidates': 'lattice', 'max_tries_per_tile': 5}

    def __post_init__(self):
        if self.candidates not in ('lattice', 'random'):
            raise ValueError(
                f"candidates must be 'lattice' or 'random', got "
                f"{self.candidates!r}")


# ── one tile ─────────────────────────────────────────────────────────────────

@dataclass
class SampleMeta:
    """Where a tile is and what is known about it. PLAIN DATA, NO HANDLE.

    An openslide handle is not picklable, so a meta carrying one cannot cross
    into a DataLoader worker -- and the failure is a pickling error a long way
    from the cause. Everything here survives `pickle`, `csv` and `json`; a
    handle enters at the reader that reads it (`SlideReader.read_samples`), from the
    caller who owns it.
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

    #: The camera's FoV when it is a rectangle inside the square footprint
    #: (`RungPlan.fov_w_l0`); 0 for a square camera, whose FoV IS the footprint.
    fov_w_l0: int = 0
    fov_h_l0: int = 0

    @property
    def fov_rect(self) -> Tuple[int, int, int, int]:
        """`(x0, y0, w, h)` level-0: what the camera photographs for this
        sample. The footprint itself for a square camera; for a rectangular
        one, the rectangle centred in the footprint -- the same offset
        `ReadSpec.fov_offset` used when the reserve was computed."""
        fp = int(self.footprint_l0)
        if not (self.fov_w_l0 and self.fov_h_l0):
            return self.x, self.y, fp, fp
        return (self.x + (fp - self.fov_w_l0) // 2,
                self.y + (fp - self.fov_h_l0) // 2, self.fov_w_l0, self.fov_h_l0)

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
        `reserve_l0 - footprint` is odd: `int(4096.4 * 3)` is 12289, the pad is
        `(12289 - 4096) // 2 = 4096`, and `4096 + 2*4096` is 12288. A region
        whose far edge is the mask's has no room to absorb that px, so the
        reserve is the one the geometry centres.
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




class Sample:
    """One tile's place: its `SampleMeta`. No pixels -- a reader reads those
    (`SlideReader.read_samples`)."""

    __slots__ = ('meta',)

    def __init__(self, meta: SampleMeta):
        self.meta = meta

    def __repr__(self) -> str:
        return (f'Sample({self.meta.slide} ds{self.meta.ds:g} '
                f'({self.meta.x}, {self.meta.y}) {self.meta.bucket})')


# ── the sampler, which is also the container ─────────────────────────────────

@dataclass
class RungReport:
    """What one rung could offer and what it gave. Knowable before any read.

    `supply` against `per_bucket` is the pair worth reading: the first is the
    candidate pool's histogram, the second is what was taken. A cap shows up as
    the two disagreeing; a FLOOR that could not be met shows up as
    `n_below_floor`, and it is a different fact -- the first says a bucket was
    held back, the second says the slide did not have it.

    `n_admissible` counts candidates whose bucket has a non-zero cap -- the
    tissue gate, expressed through the caps.
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
                 slide: str = ''):
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
        #: Filled by `cached`: where this draw lives and what was reused.
        self.cache_info: Dict[str, object] = {}

    # ── candidates ──────────────────────────────────────────────────────────

    def _regions(self, plan: RungPlan):
        """The regions that can host THIS rung's tile: a `patchable` view of
        the mask, taken fresh per rung and never written back.

        The mask arrives with its recipe's region prep already applied
        (`TissueMaskConfig.regions`), so all that is left is the one step that
        depends on the rung.

        The TILE's footprint, not the reserve, and that is the whole of the
        two-rectangle rule in `_lattice`: the TILE is the training sample and
        has to be in tissue; the RESERVE is only context for a warp and has to
        be READABLE.
        """
        if plan.is_rect:
            # a region hosts a rectangular FoV if it holds the RECTANGLE;
            # `patchable` takes one side and would ask for the long one twice
            fw, fh = self._dims(plan)
            return [r for r in self.mask.tissue_regions if r.w >= fw and r.h >= fh]
        return self.mask.patchable(int(plan.footprint_l0)).tissue_regions

    @staticmethod
    def _dims(plan: RungPlan) -> Tuple[int, int]:
        """What must lie in tissue: the camera's FoV -- the footprint square,
        or the rectangle a rectangular camera photographs."""
        if plan.is_rect:
            return int(plan.fov_w_l0), int(plan.fov_h_l0)
        fp = int(plan.footprint_l0)
        return fp, fp

    @staticmethod
    def _offset(plan: RungPlan) -> Tuple[int, int]:
        """The FoV's top-left inside the footprint square: (0, 0) when they
        are the same, centred for a rectangle (`SampleMeta.fov_rect`)."""
        if not plan.is_rect:
            return 0, 0
        fp = int(plan.footprint_l0)
        return (fp - int(plan.fov_w_l0)) // 2, (fp - int(plan.fov_h_l0)) // 2

    def _meta(self, plan: RungPlan, x: int, y: int, **axes) -> SampleMeta:
        """A sample of `plan` at (x, y). The one place a SampleMeta's geometry
        is written, so the four ways a tile is placed cannot disagree on it."""
        fp = int(plan.footprint_l0)
        return SampleMeta(
            slide=self.slide, ds=plan.rung_ds, level=plan.level, x=int(x), y=int(y),
            tile_size=plan.tile_size, read_size=plan.read_size, footprint_l0=fp,
            reserve_l0=fp + 2 * _margin_of(plan), stack_kind=plan.stack_kind,
            fov_w_l0=int(plan.fov_w_l0), fov_h_l0=int(plan.fov_h_l0), **axes)

    def _lattice(self, plan: RungPlan) -> np.ndarray:
        """Level-0 top-left corners on the lattice, inside the regions.

        The step is a fraction of the footprint and is converted here, which
        is the whole reason it is not a level-0 constant: as a level-0 number it
        would be a disjoint lattice at ds 1 and an 87 per cent overlapping one
        at ds 8.
        """
        # A fraction of the footprint, so the same step at every rung and on
        # either stack: the footprint grows with ds on an 'F' rung and is held
        # on an 'R' one, and using ds directly would space an R lattice 32x
        # too far apart at ds 32.
        fp = int(plan.footprint_l0)
        step = max(1, int(round(self.cfg.overlap.step * fp)))
        pad = _margin_of(plan)
        fw, fh = self._dims(plan)
        offx, offy = self._offset(plan)
        sx0, sy0, sx1, sy1 = _scanned_rect(self.mask)
        out = []
        for region in self._regions(plan):
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
            #
            # "The tile" is the camera's FoV: the footprint square, or the
            # rectangle centred in it (`_dims`, `_offset`), which is what a
            # rectangular sensor actually photographs.
            x0 = max(region.x - offx, sx0 + pad)
            y0 = max(region.y - offy, sy0 + pad)
            x1 = min(region.x + region.w - fw - offx, sx1 - pad - fp)
            y1 = min(region.y + region.h - fh - offy, sy1 - pad - fp)
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

        It shares the gate and the selection with the lattice, so the only
        difference between the two arms is where the candidates came from --
        which is what makes the comparison mean anything.

        Draws `n_per_rung * max_tries_per_tile` positions and hands them all
        to the same gate, so the budget is on draws, not on accepted tiles.
        """
        regions = self._regions(plan)
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
        lattice -- `_top_up` and `_place_inherited` -- have to ask: a top-up
        displacement of up to 1.25 tiles can walk past the region edge.
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
        need the same two arrays. An inherited tile is scored at its own
        position, not borrowed from a nearby candidate: a zero cap refuses it,
        so its bucket decides whether a chain lives.
        """
        if not len(xy):
            return np.zeros(0, np.float32), np.zeros(0, np.int8)
        if plan.is_rect:
            score = self._fov_background(xy, plan)
        else:
            score = SCORERS[self.cfg.richness.scorer](self.mask, xy, plan)
        return score, assign_buckets(score, self.cfg.richness.edges)

    def _fov_background(self, xy: np.ndarray, plan: RungPlan) -> np.ndarray:
        """Background fraction of a RECTANGULAR FoV at each footprint origin.

        `white_fractions` answers for square cells, so the rectangle is covered
        by a centred grid of `FOV_CELL`-output-px cells and their fractions
        averaged. A 1440 px width is 5.6 cells, so the grid covers 1280 of it;
        `test_tile_sampler` measures that against the exact fraction."""
        xy = np.asarray(xy, dtype=np.int64).reshape(-1, 2)
        fw, fh = self._dims(plan)
        offx, offy = self._offset(plan)
        per_px = float(plan.footprint_l0) / plan.tile_size
        level_ds = float(plan.level_ds)
        cell_lvl = max(1, int(round(FOV_CELL * per_px / level_ds)))
        cell_l0 = cell_lvl * level_ds
        n_cols = max(1, int(fw // cell_l0))
        n_rows = max(1, int(fh // cell_l0))
        margin_x = (fw - n_cols * cell_l0) / 2.0
        margin_y = (fh - n_rows * cell_l0) / 2.0
        gx, gy = np.meshgrid(offx + margin_x + np.arange(n_cols) * cell_l0,
                             offy + margin_y + np.arange(n_rows) * cell_l0,
                             indexing='xy')
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        origins = np.rint(xy[:, None, :] + grid[None, :, :]).astype(np.int64)
        frac = self.mask.white_fractions(origins.reshape(-1, 2), plan.level, cell_lvl)
        return np.asarray(frac, dtype=np.float32).reshape(len(xy), -1).mean(axis=1)

    def _admissible(self, bucket: np.ndarray) -> np.ndarray:
        """Positions whose bucket has a non-zero cap.

        THIS IS THE TISSUE GATE. A zero cap on `bg85_95` and `bg95_100` is a
        gate at 85 per cent background, stated in the only place that also
        decides what happens to everything below it.
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
                out.append(self._meta(plan, x, y, bucket=name,
                                      score=float(score[idx]), overlap_max=ratio,
                                      inherit_id=-1, origin='grid'))

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

        The displaced tile is scored on its OWN position and records its own
        bucket and score: it is disjoint from the parent, so the parent's
        number is a claim about different pixels. The parent's bucket only
        decides which quota the offer is drawn against.
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
                # parent's bucket: the offer is a FULL TILE away and disjoint, so
                # the parent's bucket would be a claim about different pixels. One
                # `white_fractions` call on one position is the cost.
                offer = np.array([[x, y]], dtype=np.int64)
                oscore, ob = self._rate(offer, plan)
                if not bool(self._admissible(ob)[0]):
                    continue
                if not self._reserve_fits(x, y, plan):
                    continue
                arr = np.asarray(taken_xy, dtype=np.int64)
                if self._overlap_against(x, y, fp, arr) > ov.max_overlap_ratio:
                    continue
                want[parent.bucket] -= 1
                taken_xy.append([x, y])
                out.append(self._meta(
                    plan, x, y, bucket=self.cfg.richness.names[int(ob[0])],
                    score=float(oscore[0]), overlap_max=0.0, inherit_id=-1,
                    origin='jitter', parent_x=parent.x, parent_y=parent.y))
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
        xy = self._candidates(source)
        if len(xy):
            _, b = self._rate(xy, source)
            xy = xy[self._admissible(b)]
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
            out.append(self._meta(plan, x, y, bucket=names[bi],
                                  score=float(score[0]), overlap_max=ratio,
                                  inherit_id=int(chain), origin='inherit'))
        report.n_inherited = len(out)
        return out


    # ── the loop ────────────────────────────────────────────────────────────

    def sample(self, plans: Sequence[RungPlan]) -> 'TileSampler':
        """Fill the container. The three phases, in the order they must run.

        Takes RUNG PLANS, not a level and a count. A plan says which level to
        read, what the footprint is and what must fit -- which is what lets one
        sampler serve a 4x pyramid and a 2x one without knowing which it is on.
        """
        if not plans:
            raise ValueError('sample() needs at least one RungPlan')
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
        # A RECTANGULAR FoV is placed by the square of its long side, and three
        # things built on that square are not written for a rectangle: an
        # inheritance chain (one centre across rungs; a camera has one rung),
        # the random control arm, and a scorer that reads pixels.
        if any(q.is_rect for q in plans):
            if cfg.inherit.share > 0.0:
                raise ValueError(
                    f'inherit.share is {cfg.inherit.share} with a rectangular '
                    f'FoV: a camera serves one rung, and inheritance is one '
                    f'centre at several')
            if cfg.candidates != 'lattice':
                raise ValueError(
                    f"candidates must be 'lattice' for a rectangular FoV, got "
                    f'{cfg.candidates!r}')
            if cfg.richness.scorer != 'background':
                raise ValueError(
                    f"a rectangular FoV is scored by 'background' only, got "
                    f'{cfg.richness.scorer!r}')

        self._truncated = set()
        centres = self._choose_centres(plans)               # phase 1
        samples: List[SampleMeta] = []

        for plan in plans:
            # EVERY RUNG STARTS FROM THE SAME SEED. `_select` draws a
            # permutation of the candidates, so a shared stream would make
            # ds 2's tiles depend on how many draws ds 1 happened to make, and
            # `sampler_id` hashes the config, not the draw count. The centres
            # are drawn ABOVE this loop and keep their own draw, so a rung's
            # tiles do not depend on whether inheritance ran either.
            self._rng = np.random.default_rng(cfg.seed)

            report = RungReport(ds=plan.rung_ds, n_asked=cfg.n_per_rung)
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
            xy = self._candidates(plan)
            report.n_candidates = len(xy)
            _, bucket = self._rate(xy, plan)
            names = self.cfg.richness.names
            for i, name in enumerate(names):
                report.supply[name] = int((bucket == i).sum())
                report.per_bucket[name] = report.supply[name]
            report.n_admissible = int(self._admissible(bucket).sum())
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

    # ── persistence ─────────────────────────────────────────────────────────

    #: `index.csv` columns. The three axes join the coordinates, so every
    #: question the container answers survives to disk. `neighbours` is NOT
    #: here: it is derived from the coordinates, and a stored copy is a second
    #: thing to keep in step with them.
    COLUMNS = ('index', 'slide', 'ds', 'level', 'x', 'y', 'tile_size',
               'read_size', 'footprint_l0', 'reserve_l0', 'bucket', 'score',
               'overlap_max',
               'inherit_id', 'stack_kind', 'origin', 'parent_x', 'parent_y',
               'fov_w_l0', 'fov_h_l0')

    def save(self, folder: Union[str, Path],
             extra_meta: Optional[Dict[str, object]] = None) -> Path:
        """`index.csv` + `meta.json`: coordinates and metadata, with the
        pixels read on demand by whoever loads it, through a camera.

        The layout is `Store.PreTileStore`'s, not a second one. What this adds
        is the axis columns. Pixels are not written here: `extract_pretiles`
        is the one writer of them.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        with open(folder / 'index.csv', 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(self.COLUMNS)
            for i, s in enumerate(self.samples):
                m = s.meta
                writer.writerow([i] + [getattr(m, c) for c in self.COLUMNS[1:]])

        meta = {
            'sampler_id': self.cfg.identity_id(),
            'slide': self.slide,
            'n_samples': len(self.samples),
            'stack_kind': self.cfg.inherit.stack_kind,
            'config': self.cfg.identity_parts(),
            'provenance': self.cfg.provenance(),
            'rungs': {f'{d:g}': dataclasses.asdict(r)
                      for d, r in self.reports.items()},
            **(extra_meta or {}),
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
        opens its own handle. So this cannot require one, and the camera that
        reads a sample is built on the reader its caller owns.
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
                    parent_y=int(row['parent_y']),
                    fov_w_l0=int(row['fov_w_l0']),
                    fov_h_l0=int(row['fov_h_l0'])))

        out = cls.__new__(cls)
        out.wsi, out.mask = wsi, mask
        out.cfg = cfg or SamplerConfig()
        out.slide = meta.get('slide', '')
        out.samples = [Sample(m) for m in rows]
        # The per-rung reports ride in meta.json (`save` writes them), and a
        # reload that dropped them could not say what the draw looked like --
        # which is the whole of `write_report`.
        out.reports = {float(d): RungReport(**r)
                       for d, r in meta.get('rungs', {}).items()}
        out.cache_info = {}
        out._rng = np.random.default_rng(out.cfg.seed)
        out._inherit_bucket = {}

        stored = meta.get('sampler_id', '')
        if cfg is not None and stored and stored != cfg.identity_id():
            raise ValueError(
                f'{folder} was cut with sampler_id {stored} and the config '
                f'passed in hashes to {cfg.identity_id()}. Loading it under the '
                f'wrong config would report the wrong axes for every row -- '
                f'pass the right config, or none, and read the stored one')
        return out

    # ── the cache ───────────────────────────────────────────────────────────

    @classmethod
    def cached(cls, wsi_path: Union[str, Path], cfg: SamplerConfig,
               plan: PlanSpec, sampler_root: Union[str, Path], *, masks,
               report_dir: Optional[Union[str, Path]] = None) -> 'TileSampler':
        """The draw for one slide, from the cache when it is there and made
        -- segmentation included -- when it is not.

            <sampler_root>/<seg_id>/<slide>/<region_id>_<sampler_id>_<plan>/
                                            index.csv + meta.json

        `masks` is the caller's `TissueMaskConfig.MaskMaker`: the recipe, the
        device and the MASK cache, which is its own object under its own root
        (`<made_by>_mask`). The draw's path repeats the mask's keys, upstream
        above downstream, so dropping a recipe is one `rm -rf` of `<seg_id>/`
        in each root.

        A hit opens NOTHING: no slide, no segmenter. A miss opens the slide,
        asks `masks` for the mask (itself a hit or a segmentation), samples,
        and writes the draw atomically. Either way the sampler comes back in
        one state -- `wsi=None, mask=None` -- so a caller cannot tell which
        happened by what it holds, only by `cache_info`; a caller that wants
        pixels reads them through a camera on its own reader.

        `masks` is taken as an object and never imported: the mask recipes
        import every segmenter, and this module stays without them.

        `report_dir`, when given, receives `sampler_report_<slide>.md` and
        `samples_<slide>.csv` -- on a hit exactly as on a miss, because the
        report is written from what this sampler holds, not from how it got it.
        """
        mask_cfg = masks.cfg
        slide = wsi_stem_of(wsi_path)
        folder = (Path(sampler_root) / mask_cfg.seg_id() / slide
                  / f'{mask_cfg.region_id()}_{cfg.identity_id()}_{plan.key()}')
        info = dict(folder=str(folder), seg_id=mask_cfg.seg_id(),
                    region_id=mask_cfg.region_id(), sampler_id=cfg.identity_id(),
                    plan=plan.key(),
                    mask_parts=list(mask_cfg.identity_parts()))
        # The draw depends on the mask's code as well as on its id, so the
        # mask recipe's versions ride in this record too.
        want = record(cfg, also=(mask_cfg,), seg_id=info['seg_id'],
                      region_id=info['region_id'], plan=info['plan'])
        if (folder / 'meta.json').exists():
            with open(folder / 'meta.json') as handle:
                stored = json.load(handle)
            check_source(stored, wsi_path, folder)
            stale = record_diff(stored.get('identity'), want)
            if stale:
                print(f'  [sampler] {folder} is stale, drawing again: '
                      + '; '.join(stale), flush=True)
                shutil.rmtree(folder)
        if (folder / 'meta.json').exists():
            out = cls.load(folder, cfg=cfg)
            info.update(mask_hit=True, samples_hit=True)
        else:
            from SafeSlide import SafeSlide                     # noqa: PLC0415
            with SafeSlide(str(wsi_path)) as wsi:
                mask, mask_hit = masks.mask(wsi)
                out = cls(wsi, mask, cfg, slide=slide)
                out.sample(plan.plans_for(wsi))
                with atomic_dir(folder) as tmp:
                    out.save(tmp, extra_meta=dict(
                        source=source_key(wsi_path), wsi_path=str(wsi_path),
                        seg_id=info['seg_id'], region_id=info['region_id'],
                        plan=info['plan'], mask_parts=info['mask_parts'],
                        identity=want))
            out.wsi, out.mask = None, None
            info.update(mask_hit=bool(mask_hit), samples_hit=False)
        out.slide = slide
        out.cache_info = info
        if report_dir is not None:
            out.write_report(report_dir)
        return out

    def write_report(self, report_dir: Union[str, Path]) -> Path:
        """`sampler_report_<slide>.md` and `samples_<slide>.csv` into
        `report_dir` -- the result directory of whoever asked for this draw.

        The report is the whole of what decided the draw and what came of it:
        the sampler config and the mask recipe, the cache entry used and
        whether the mask and the draw were reused or made, and per rung what
        the candidate pool offered (`supply`) against what was taken
        (`per_bucket`) -- a cap shows as the two disagreeing, a slide that did
        not have a bucket as `below floor`. The CSV is every sample's identity,
        `save`'s own columns.
        """
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        slide = self.slide or 'slide'
        info = getattr(self, 'cache_info', {}) or {}

        csv_path = report_dir / f'samples_{slide}.csv'
        with open(csv_path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(self.COLUMNS)
            for i, s in enumerate(self.samples):
                writer.writerow([i] + [getattr(s.meta, c) for c in self.COLUMNS[1:]])

        names = list(self.cfg.richness.names)
        lines = [f'# Sampler report -- {slide}', '']
        if info:
            state = lambda hit: 'reused' if hit else 'computed'   # noqa: E731
            lines += ['## Cache', '',
                      f'- entry: `{info.get("folder", "")}`',
                      f'- mask: {state(info.get("mask_hit"))}, '
                      f'draw: {state(info.get("samples_hit"))}',
                      f'- seg_id `{info.get("seg_id")}`, region_id '
                      f'`{info.get("region_id")}`, sampler_id '
                      f'`{info.get("sampler_id")}`, plan `{info.get("plan")}`', '',
                      '## Mask recipe (fields that differ from the baseline)', '']
            lines += [f'- `{p}`' for p in info.get('mask_parts', [])] or ['- (baseline)']
            lines.append('')
        lines += ['## Sampler config', '']
        lines += [f'- `{p}`' for p in self.cfg.identity_parts()]
        for k, v in self.cfg.provenance().items():
            lines.append(f'- `{k}={v!r}` (provenance, not identity)')
        lines += ['', f'## Distribution ({len(self.samples)} samples)', '',
                  '| rung | asked | taken | short | below floor | spilled | '
                  + ' | '.join(names) + ' |',
                  '|' + '---|' * (6 + len(names))]
        totals = collections.Counter()
        for ds in sorted(self.reports):
            r = self.reports[ds]
            totals.update(r.per_bucket)
            lines.append(f'| {ds:g} | {r.n_asked} | {r.n_taken} | {r.short} | '
                         f'{r.n_below_floor} | {r.n_spilled} | '
                         + ' | '.join(str(r.per_bucket.get(n, 0)) for n in names)
                         + ' |')
        lines.append('| all | | ' + str(len(self.samples)) + ' | | | | '
                     + ' | '.join(str(totals.get(n, 0)) for n in names) + ' |')
        lines += ['', '## Candidate supply per bucket (what the pool offered)', '',
                  '| rung | candidates | admissible | ' + ' | '.join(names) + ' |',
                  '|' + '---|' * (3 + len(names))]
        for ds in sorted(self.reports):
            r = self.reports[ds]
            lines.append(f'| {ds:g} | {r.n_candidates} | {r.n_admissible} | '
                         + ' | '.join(str(r.supply.get(n, 0)) for n in names)
                         + ' |')
        lines += ['', f'Sample identities: `{csv_path.name}`', '']

        md_path = report_dir / f'sampler_report_{slide}.md'
        md_path.write_text('\n'.join(lines))
        return md_path

    # ── summary ─────────────────────────────────────────────────────────────

    def summary(self) -> 'TileSampler':
        print(f'slide      : {self.slide}')
        print(f'sampler_id : {self.cfg.identity_id()}')
        print(f'samples    : {len(self.samples)}')
        for ds in sorted(self.reports):
            print(self.reports[ds].line())
        chains = self.stacks(complete_only=True)
        broken = self.incomplete()
        if chains or broken:
            print(f'chains     : {len(chains)} complete, {len(broken)} broken')
        return self
