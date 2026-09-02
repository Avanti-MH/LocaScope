#!/usr/bin/env python3
"""Tests for utilities/TileSampler.py -- the three axes, and the units.

    python utilities/test_modules/test_tile_sampler.py
    python utilities/test_modules/test_tile_sampler.py --only overlap

No slide, no model, no data. The mask is a real `TissuesRegionsMask` built by
`from_mask` over a synthetic array, and the slide is the four attributes
`from_mask` and `read_region_rgb` need. That is deliberate: the sampler's whole
job is arithmetic over a mask, so a fake mask tests the thing itself rather
than a slide reader.

WHAT WOULD RUN AND BE WRONG
=============================
1. A lattice step computed as `grid_step * ds`. It equals the right answer on
   an 'F' rung and is 32x too coarse on an 'R' one at ds 32, where `ds` is a
   degradation factor and not a magnification. The rung comes back nearly
   empty and reads as "there was no tissue there".
2. Inherited positions run through the overlap bound. At ds 32 the footprint
   is 8192 px, so carried centres a few hundred px apart almost coincide and
   most of them are dropped -- leaving `inherit_id`s that do not resolve, which
   a survival analysis reads as "the keypoint died".
3. `bucket_frame` doing nothing. The two values are supposed to give different
   corpora; if the carried bucket is silently recomputed, `at_inherit` becomes
   `per_rung` while `sampler_id` still separates them, and two identical
   corpora sit under two names.
4. A chain returned four rungs long as if it were six. "Died at ds 16" and
   "ds 16 never sampled it" are the whole of Stage B's conclusion.
5. A `SampleMeta` that cannot be pickled. It works in one process and dies the
   moment a DataLoader forks, a long way from the cause.

Every load-bearing check scores against a DECOY -- the random arm, the other
`bucket_frame`, the un-exempted bound -- rather than against a tolerance.

Sections:
  1. units      -- 'F' against 'R', which is where the lattice step lives
  2. overlap    -- the lattice, the bound, the share, the jitter properties
  3. richness   -- bucket cuts and quota caps
  4. inherit    -- chains, the exemption, completeness, bucket_frame
  5. identity   -- what moves sampler_id and what must not
  6. access     -- the four ways in, and that they compose
  7. carry      -- Sample, pickling, persistence
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))

from _paths import setup_import_paths                            # noqa: E402

setup_import_paths()

import numpy as np                                               # noqa: E402

from DsLadder import RungPlan                                    # noqa: E402
from TileSampler import (InheritConfig, OverlapConfig,           # noqa: E402
                         allocate_targets, bucket_names, spill_order,
                         caps_for_tissue_ratio,
                         RichnessConfig, Sample, SampleMeta,
                         SamplerConfig, TileSampler, assign_buckets,
                         resolution_plan)
from TissuesRegionsMask import TissuesRegionsMask                # noqa: E402

_RESULTS = []

TILE = 64            # small enough that a whole rung fits in a page of numbers
MASK_DS = 4.0        # level-0 px per mask px


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ══════════════════════════════════════════════════════════════════════════════
#  a slide and a mask, both made up
# ══════════════════════════════════════════════════════════════════════════════

class _Slide:
    """What `from_mask` reads, plus a reader that returns something recognisable.

    `read_region_rgb` and not `read_region`: the sampler refuses a handle
    without it, and that refusal is one of the checks below.
    """

    def __init__(self, width, height, mpp=0.25, downsamples=(1., 2., 4., 8.)):
        self.level_dimensions = [(int(width), int(height))]
        self.level_downsamples = list(downsamples)
        self.level_count = len(downsamples)
        self.properties = {'openslide.mpp-x': str(mpp),
                           'openslide.mpp-y': str(mpp)}

    def read_region_rgb(self, location, level, size):
        # The pixel values encode the request, so a materialise() check can
        # tell WHERE it read from rather than only that it read something.
        w, h = size
        out = np.zeros((h, w, 3), np.uint8)
        out[..., 0] = location[0] % 256
        out[..., 1] = location[1] % 256
        out[..., 2] = level
        return out


def _mask(tissue_blocks, rows=200, cols=200, ds=MASK_DS, slide=None):
    """A mask with rectangular tissue blocks, in MASK pixels."""
    arr = np.zeros((rows, cols), dtype=bool)
    for r, c, h, w in tissue_blocks:
        arr[r:r + h, c:c + w] = True
    wsi = slide or _Slide(int(cols * ds), int(rows * ds))
    return wsi, TissuesRegionsMask.from_mask(
        wsi, arr, span=(int(cols * ds), int(rows * ds)))


def _one_big_block():
    """One region covering most of the mask: enough candidates at every rung."""
    return _mask([(10, 10, 180, 180)])


def _wide_block():
    """A region big enough that a COARSE rung still has room to place several.

    `_one_big_block` is 720 level-0 px on a side, which at ds 8 holds one
    512 px footprint and no second one -- so any check that needs two coarse
    tiles to interact passes vacuously there. Four times the area is what makes
    the coarse rung a real rung rather than a single position.
    """
    return _mask([(20, 20, 360, 360)], rows=400, cols=400)


def _permissive(**over):
    """A richness config that constrains NOTHING, for the tests of other axes.

    THIS IS NOT LAZINESS, IT IS THE POINT OF HAVING THREE AXES. The production
    contract caps `bg00_15` at 15 per cent of a rung, and every synthetic mask
    here is a solid rectangle -- so every candidate scores background 0, lands
    in that one bucket, and a rung of 40 comes back with SIX tiles. The overlap
    tests then measure an almost empty rung: "0/6 overlapping, budget 4" passes
    while proving nothing, the top-up never runs because its bucket has no
    quota left, and the random arm is too sparse to produce a single
    overlapping pair. Two of those failed outright on 2026-08-27 and three more
    passed vacuously.

    All floors zero and all caps 1 means first-come over the shuffle, which is
    what these fixtures assumed before richness had floors. The tests that ARE
    about richness pass `RichnessConfig()` explicitly and use `_graded_mask`.
    """
    base = dict(floors=(0.0,) * 7, caps=(1.0,) * 7)
    base.update(over)
    return RichnessConfig(**base)


def _graded_mask(tile=TILE, ds=1.0):
    """A mask whose tiles span the background range, not just bucket zero.

    Every other fixture is solid tissue, so `white_fractions` returns 0 for all
    of them and the richness axis has exactly one bucket to work with -- which
    cannot show a floor being met, a cap biting, or a chain truncating.

    HOLES, NOT STRIPES. The tissue stays one connected region with one bbox, so
    `filter_regions` -> `merge_overlapping` -> `filter_patchable` see the same
    single region they see everywhere else; only what is INSIDE it varies. A
    hole of side h in a tile of side t reads as h^2/t^2 background, and the
    side is chosen per tile COLUMN so the spread is a property of position and
    not of the rng.
    """
    cell = int(tile * ds / MASK_DS)              # tile side in mask px
    arr = np.zeros((200, 200), dtype=bool)
    arr[10:190, 10:190] = True
    # bg per column band: 0, 0.06, 0.19, 0.39, 0.56, 0.77, 0.88, and repeat.
    sides = [0, 4, 7, 10, 12, 14, 15]
    for j in range(11):
        h = sides[j % len(sides)]
        if not h:
            continue
        h = max(1, int(round(h * cell / 16.0)))
        for i in range(11):
            r0 = 10 + i * cell + (cell - h) // 2
            c0 = 10 + j * cell + (cell - h) // 2
            arr[r0:r0 + h, c0:c0 + h] = False
    wsi = _Slide(int(200 * MASK_DS), int(200 * MASK_DS))
    return wsi, TissuesRegionsMask.from_mask(
        wsi, arr, span=(int(200 * MASK_DS), int(200 * MASK_DS)))


def _plans(dss=(1.0, 2.0), tile=TILE, factor=1):
    out = []
    for ds in dss:
        fp = float(tile * ds)
        out.append(RungPlan(rung_ds=float(ds), level=0, level_ds=1.0,
                            shrink=float(ds), tile_size=tile,
                            read_size=tile, footprint_l0=fp,
                            reserve_l0=fp * factor, stack_kind='F'))
    return out


def _cfg(**over):
    base = dict(tile=TILE, n_per_rung=40, seed=0,
                richness=_permissive(),
                overlap=OverlapConfig(),          # 0 == the tile, disjoint
                inherit=InheritConfig())
    base.update(over)
    return SamplerConfig(**base)


def _overlapping_pairs(sampler, ds=None):
    metas = [s.meta for s in sampler
             if ds is None or abs(s.meta.ds - ds) < 1e-9]
    n = 0
    for i in range(len(metas)):
        for j in range(i + 1, len(metas)):
            if metas[i].overlap_with(metas[j]) > 0:
                n += 1
    return n


# ══════════════════════════════════════════════════════════════════════════════
#  1. units -- 'F' against 'R'
# ══════════════════════════════════════════════════════════════════════════════

def t_the_lattice_step_is_output_pixels_and_not_ds():
    """THE ONE THIS FILE EXISTS FOR.

    The step converts LEVEL pixels to level-0 px, and the conversion is
    `footprint_l0 / tile` -- which equals `ds` on an 'F' rung and 1 on an 'R'
    one. Using `ds` directly is right for F and 32x too coarse for R at ds 32:
    the rung comes back with a handful of candidates and reads as "there was
    no tissue at that magnification".

    So: an R rung at ds 8 must offer the SAME number of lattice positions as
    an R rung at ds 1, because their footprints are identical. The decoy is
    the F rung at ds 8, which must offer far fewer.
    """
    wsi, mask = _one_big_block()
    cfg = _cfg()
    s = TileSampler(wsi, mask, cfg)

    r1 = s.preflight([resolution_plan(1.0, TILE)])[0]
    r8 = s.preflight([resolution_plan(8.0, TILE)])[0]
    f8 = s.preflight(_plans((8.0,)))[0]

    assert r1.n_candidates == r8.n_candidates, (
        f'R rungs at ds 1 and ds 8 hold the same footprint and must offer the '
        f'same lattice: {r1.n_candidates} vs {r8.n_candidates}. A step '
        f'computed as grid_step*ds gives exactly this failure')
    assert f8.n_candidates < r8.n_candidates, (
        f'the F rung at ds 8 has a footprint 8x wider and must offer FEWER '
        f'positions, not {f8.n_candidates} against {r8.n_candidates} -- if '
        f'they match, the F step is not scaling either')
    return (f'R ds1 {r1.n_candidates} == R ds8 {r8.n_candidates}, '
            f'F ds8 {f8.n_candidates}')


def t_an_R_rung_reads_level_0_and_an_F_rung_grows():
    r = resolution_plan(32.0, TILE)
    assert (r.level, r.read_size, r.footprint_l0) == (0, TILE, float(TILE))
    assert r.stack_kind == 'R'
    f = _plans((32.0,))[0]
    assert f.footprint_l0 == float(TILE * 32) and f.stack_kind == 'F'
    return f'R footprint {r.footprint_l0:g}, F footprint {f.footprint_l0:g}'


def t_reserve_is_what_must_fit_not_what_the_tile_covers():
    """A 3x reserve shrinks the usable lattice -- against the SCANNED
    RECTANGLE, not against the region.

    The tile has to be in the region because it is the training sample; the
    reserve only has to be readable, so it is allowed to reach into the glass
    beside the region and is not allowed to reach past the edge of the imaged
    area. Requiring it to fit a single region was the earlier rule and it
    emptied the coarse rungs -- at ds 32 it demanded a region 24576 px wide,
    and one slide came back with 0 tiles of 500.
    """
    wsi, mask = _one_big_block()
    s = TileSampler(wsi, mask, _cfg())
    plain = s.preflight(_plans((1.0,), factor=1))[0]
    reserved = s.preflight(_plans((1.0,), factor=3))[0]
    assert reserved.n_candidates < plain.n_candidates, (
        f'reserving 3x the tile did not shrink the lattice '
        f'({reserved.n_candidates} against {plain.n_candidates}); the reserve '
        f'is being ignored and every position will clip')
    return f'{plain.n_candidates} -> {reserved.n_candidates} with a 3x reserve'


# ══════════════════════════════════════════════════════════════════════════════
#  2. overlap
# ══════════════════════════════════════════════════════════════════════════════

def t_every_placed_tile_keeps_its_reserve_inside_the_scanned_rectangle():
    """THE ONE THAT WAS ACTUALLY WRONG.

    `_lattice` bakes the reserve into its range bounds, so a lattice position
    cannot run off. The two paths that place a position from somewhere ELSE --
    `_top_up`, which displaces a parent by up to 1.25 tiles, and
    `_place_inherited`, which carries a centre from another rung -- both
    checked the tissue gate and the overlap bound and neither checked the
    reserve.

    It only showed at the COARSE rungs, and that is why it took a real run to
    find: at ds 1 the lattice fills the quotas and the top-up never runs, so
    every fine rung passed. At ds 8 and above the buckets go short, the top-up
    starts, and `extract_pretiles` stopped with the pre-tile 94 to 2560 px off
    the scanned rectangle.

    The fixture forces the top-up: tiny quotas on a big region leave the
    buckets short at once, so almost everything placed here comes through the
    path that was broken.
    """
    # TWO things have to be true for the top-up to run at all, and each of
    # them cost a run to find:
    #
    #   the quota must still be SHORT after the lattice -- so ask for more than
    #   the lattice has (200 against about 100 positions), not for less
    #   overlap must be ALLOWED -- under a disjoint lattice every offer the
    #   top-up can make lands on a taken position, so `jitter_cap` is dead and
    #   `OverlapConfig.check` now refuses that combination outright
    #   the RICHNESS caps must not bite first -- the production contract caps
    #   bg00_15 at 15 per cent, this mask is solid tissue so every candidate is
    #   in that bucket, and the rung stops at 30 of 200 with the top-up's quota
    #   already spent. `_permissive` is what makes the top-up the binding
    #   constraint again, which is the thing under test here.
    wsi, mask = _one_big_block()
    cfg = _cfg(n_per_rung=200, inherit=InheritConfig(share=0.1),
               richness=_permissive(),
               overlap=OverlapConfig(max_overlap_ratio=0.9,
                                     overlapping_share=1.0,
                                     jitter_cap=0.20))
    plans = _plans((1.0, 4.0), factor=3)
    s = TileSampler(wsi, mask, cfg).sample(plans)

    by_origin = {}
    for x in s:
        by_origin[x.meta.origin] = by_origin.get(x.meta.origin, 0) + 1
    assert by_origin.get('jitter'), (
        'no tile came from the top-up, so this fixture cannot show the bug -- '
        'lower the quotas or raise n_per_rung')

    plan_of = {p.rung_ds: p for p in plans}
    for i, sample in enumerate(s):
        m = sample.meta
        plan = plan_of[m.ds]
        ok = s._reserve_fits(m.x, m.y, plan)
        assert ok, (
            f'tile {i} (origin={m.origin}, ds={m.ds:g}) reserves '
            f'{m.reserve} px around ({m.x}, {m.y}) and that runs past the '
            f'scanned rectangle. A lattice position cannot do this; '
            f'{m.origin} could')
    return f'{len(s)} tiles, ' + ' '.join(f'{k}:{v}' for k, v in by_origin.items())


def t_the_reserve_is_derived_from_the_margin_and_not_asked_for():
    """An ODD reserve, which is what a float footprint times a factor gives.

    `DsLadder.footprint_l0` is `read_size * level_ds` and is a float, so
    `int(4096.4 * 3)` is 12289. The pad is `(12289 - 4096) // 2 = 4096` and
    `4096 + 2*4096` is 12288 -- the lattice reserves 12288 of room and the
    meta claimed 12289. One px, absorbed by any region with slack and by none
    at a coarse rung, where `filter_patchable` has left only the regions whose
    far edge IS the mask's.

    So `reserve` is derived from `margin` rather than read raw, and this pins
    that: `reserve == footprint + 2*margin` for an odd request as well as an
    even one, and `reserve_origin + reserve` lands exactly `margin` past the
    tile on both sides.
    """
    for asked, want in ((3 * TILE, 3 * TILE), (3 * TILE + 1, 3 * TILE)):
        m = SampleMeta(slide='s', ds=1.0, level=0, x=1000, y=2000,
                       tile_size=TILE, read_size=TILE, footprint_l0=TILE,
                       reserve_l0=asked)
        assert m.reserve == want, (
            f'asked {asked}, footprint {TILE}, margin {m.margin} -> reserve '
            f'{m.reserve}; it must be footprint + 2*margin = {want}, or the '
            f'read and the lattice bound are two different rectangles')
        ox, oy = m.reserve_origin_l0
        assert (ox, oy) == (1000 - m.margin, 2000 - m.margin)
        assert ox + m.reserve == m.x + TILE + m.margin, (
            'the reserve is not centred on the tile')
    return 'even and odd requests both centre exactly'


def t_a_disjoint_lattice_overlaps_nothing_and_the_random_arm_does():
    """The measurement the module claims. Not a tolerance -- a decoy.

    The lattice at `grid_step == tile` cannot produce an overlapping pair; the
    sampler it replaced produced 202,420 of them over the real corpus. Both
    arms run the same gate and the same selection here, so the only difference
    is where the candidates came from.
    """
    wsi, mask = _one_big_block()
    plans = _plans((1.0,))
    grid = TileSampler(wsi, mask, _cfg()).sample(plans)
    rand = TileSampler(wsi, mask, _cfg(candidates='random',
                                       overlap=OverlapConfig(
                                           max_overlap_ratio=1.0,
                                           overlapping_share=1.0))).sample(plans)
    n_grid, n_rand = _overlapping_pairs(grid), _overlapping_pairs(rand)
    assert n_grid == 0, f'the lattice produced {n_grid} overlapping pairs'
    assert n_rand > 0, (
        'the random arm produced none either, so this fixture cannot tell the '
        'two apart -- make the region smaller or ask for more tiles')
    return f'lattice 0 pairs, random {n_rand}'


def t_a_half_step_lattice_under_a_tight_bound_is_refused():
    """grid_step 128 on a 256 tile means every neighbour overlaps 50 per cent.
    Accepting that with max_overlap_ratio=0.3 would degenerate the lattice to
    grid_step=256 while `sampler_id` still recorded 128."""
    try:
        OverlapConfig(grid_step=TILE // 2, max_overlap_ratio=0.3).check(TILE)
    except ValueError as e:
        msg = str(e)
        assert 'degenerate' in msg and str(TILE) in msg
        return 'refused, with the arithmetic'
    raise AssertionError('a lattice that breaks its own bound was accepted')


def t_the_overlapping_share_is_a_budget_and_binds():
    wsi, mask = _one_big_block()
    cfg = _cfg(overlap=OverlapConfig(grid_step=TILE // 2,
                                     max_overlap_ratio=0.9,
                                     overlapping_share=0.10))
    s = TileSampler(wsi, mask, cfg).sample(_plans((1.0,)))
    over = sum(1 for x in s if x.meta.overlap_max > 0)
    cap = int(round(0.10 * cfg.n_per_rung))
    assert over <= cap, f'{over} tiles overlap against a budget of {cap}'
    return f'{over}/{len(s)} overlapping, budget {cap}'


def t_every_jitter_offset_is_disjoint_and_off_lattice():
    """Both properties, per offset, AT EVERY TILE SIZE.

    The offsets are fractions of the tile. `ReferenceSampler` writes the same
    five as absolute pixels -- (64, 256) and so on -- which are those numbers
    only at tile 256 and are four times the tile at 64. Its own docstring
    argues that the units matter and then picks one that holds for one size.
    So the check runs at three tile sizes: a pixel constant passes at 256 and
    fails at the others.
    """
    ok = OverlapConfig()
    for tile in (64, 256, 1024):
        ok.check(tile)
    for dx, dy in ok.jitter_offsets:
        assert max(abs(dx), abs(dy)) >= 1.0, (dx, dy)
    try:
        OverlapConfig(jitter_offsets=((0.5, 1.0),)).check(TILE)
    except ValueError as e:
        assert 'lattice position' in str(e)
    else:
        raise AssertionError('an on-lattice offset was accepted')
    try:
        OverlapConfig(jitter_offsets=((0.125, 0.125),)).check(TILE)
    except ValueError as e:
        assert 'overlaps its parent' in str(e)
    else:
        raise AssertionError('an offset that overlaps its parent was accepted')
    return f'{len(ok.jitter_offsets)} offsets, three tile sizes'


def t_a_top_up_under_a_disjoint_lattice_is_refused():
    """`jitter_cap > 0` with `max_overlap_ratio = 0` promises what cannot happen.

    A lattice of step `tile` covers the plane. Every position that is not on
    it overlaps two to four lattice tiles -- 75 per cent for four of the five
    offsets and 56 for the fifth -- so with the bound at 0 every offer is
    rejected and the lattice is already the largest disjoint set there is.
    The bucket then stays short with nothing saying why, which is the failure
    this refusal replaces.
    """
    try:
        OverlapConfig(jitter_cap=0.2, max_overlap_ratio=0.0).check(TILE)
    except ValueError as e:
        assert 'largest' in str(e) and 'jitter_cap=0' in str(e)
    else:
        raise AssertionError('a dead top-up was accepted')
    # and the two coherent spellings both build
    OverlapConfig(jitter_cap=0.0).check(TILE)
    OverlapConfig(jitter_cap=0.2, max_overlap_ratio=0.9).check(TILE)
    return 'refused, and both coherent settings build'


def t_grid_step_equal_to_the_tile_is_refused_as_a_synonym_of_zero():
    """Two spellings of one lattice are two sampler_ids over one corpus."""
    try:
        OverlapConfig(grid_step=TILE).check(TILE)
    except ValueError as e:
        assert 'grid_step=0' in str(e)
        return 'refused'
    raise AssertionError('the synonym was accepted')


# ══════════════════════════════════════════════════════════════════════════════
#  3. richness
# ══════════════════════════════════════════════════════════════════════════════

def t_buckets_cut_where_the_edges_say():
    """An edge belongs to the bucket ABOVE it, and the name says the interval.

    `side='right'` is the reading of "背景比高於 15% ~ 低於 30%": 0.15 is in the
    second bucket, not the first. The DECOY is the boundary itself -- a
    `side='left'` implementation passes every interior point and fails only
    here, so testing 0.10 and 0.20 would not have separated them.
    """
    edges = (0.15, 0.30, 0.50, 0.70, 0.85, 0.95)
    names = bucket_names(edges)
    assert names == ('bg00_15', 'bg15_30', 'bg30_50', 'bg50_70', 'bg70_85',
                     'bg85_95', 'bg95_100'), names
    # float64 on purpose: this is the FUNCTION's contract, and 0.15, 0.30,
    # 0.85 and 0.95 are none of them representable in float32. The first run of
    # this test used float32 and 0.95 arrived as 0.9499999881, one bucket low
    # -- the test was wrong and the code was right, which is the same shape as
    # the DELTA_MAX episode in CLAUDE.md.
    score = np.array([0.0, 0.1499, 0.15, 0.2999, 0.30, 0.4999, 0.50,
                      0.6999, 0.70, 0.85, 0.9499, 0.95, 0.999, 1.0], np.float64)
    got = [names[i] for i in assign_buckets(score, edges)]
    want = ['bg00_15', 'bg00_15', 'bg15_30', 'bg15_30', 'bg30_50', 'bg30_50',
            'bg50_70', 'bg50_70', 'bg70_85', 'bg85_95', 'bg85_95',
            'bg95_100', 'bg95_100', 'bg95_100']
    assert got == want, f'{got}\n{want}'
    # THE DECOY. `side='left'` agrees everywhere except on the edges
    # themselves, so an interior-only test cannot separate the two.
    left = [names[i] for i in
            np.searchsorted(np.asarray(edges), score, side='left')]
    assert left != got, (
        'side=left and side=right agree on this input, so it does not pin '
        'which one is implemented -- put a score exactly ON an edge')
    # And what a float32 score does at an edge, stated rather than asserted
    # away: `white_fractions` returns a ratio of pixel counts, so a value
    # within 1e-7 of a cut is arbitrary on either side and no caller can tell.
    f32 = [names[i] for i in assign_buckets(score.astype(np.float32), edges)]
    drift = sum(1 for a, b in zip(f32, got) if a != b)
    return f'{len(names)} buckets, edges land upward; {drift}/{len(got)} f32 drift'


def t_targets_split_the_remainder_over_the_askers_only():
    """The settled contract, recomputed rather than copied.

    Two wrong readings are the decoys, and both are refuted by arithmetic
    rather than by a tolerance:

      * splitting over all five non-zero CAPS gives 30/5 = 6 points each and
        puts bg50_70 at 26 per cent, over its stated ceiling of 20
      * not splitting at all leaves the targets summing to 0.70, so every rung
        is 30 per cent short by construction
    """
    r = RichnessConfig()
    got = allocate_targets(r.floors, r.caps)
    want = (0.15, 0.25, 0.60, 0.0, 0.0, 0.0, 0.0)
    assert all(abs(a - b) < 1e-9 for a, b in zip(got, want)), f'{got} != {want}'
    assert abs(sum(got) - 1.0) < 1e-9, f'targets sum to {sum(got)}, not 1'
    assert got[3] <= r.caps[3] + 1e-9, (
        f'bg50_70 target {got[3]} over its cap {r.caps[3]} -- the remainder '
        f'went to the cap-only buckets, which is the reading that breaks the '
        f'stated 20 per cent ceiling')
    # Nobody asked: the target IS the cap, not an even split.
    loose = allocate_targets((0.0,) * 7, (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    assert loose == (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0), loose
    return f'targets {[round(t, 2) for t in got]}, spill into ' \
           f'{[r.names[i] for i in spill_order(r.caps, got)]}'


def t_caps_must_be_able_to_reach_a_full_rung():
    bad = [
        (dict(caps=(0.3, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0),
              floors=(0.0,) * 7), 'caps sum'),
        (dict(floors=(0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
              caps=(1.0,) * 7), 'floors sum'),
        (dict(floors=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
              caps=(0.2, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)), 'floor'),
    ]
    for kwargs, expect in bad:
        try:
            RichnessConfig(**kwargs)
        except ValueError as exc:
            assert expect in str(exc), f'{expect!r} not in {exc}'
        else:
            raise AssertionError(f'RichnessConfig({kwargs}) was accepted')
    # And the settled contract passes all three.
    RichnessConfig()
    return '3 refusals, default accepted'


def t_floors_are_met_before_the_free_fill():
    """A floor bucket gets its share even when the shuffle favours another.

    THE DECOY IS THE OLD IMPLEMENTATION. A single shuffled pass with per-bucket
    caps is what this replaced, and on this fixture it leaves bg30_50 under its
    floor -- so a regression to one pass fails here rather than passing quietly
    with a different mix.
    """
    wsi, mask = _graded_mask()
    cfg = _cfg(richness=RichnessConfig(), n_per_rung=60)
    s = TileSampler(wsi, mask, cfg).sample(_plans((1.0,)))
    rich = cfg.richness
    rep = s.reports[1.0]
    live = [b for b in rich.names if rep.supply.get(b, 0)]
    assert len(live) >= 4, (
        f'the fixture supplies only {live}; a floor cannot be shown to bind '
        f'when one bucket holds everything -- that is the solid-tissue mask '
        f'this one exists to replace')
    for i, name in enumerate(rich.names):
        got = len(s.where(bucket=name))
        cap = int(round(rich.caps[i] * cfg.n_per_rung))
        assert got <= cap, f'{name}: {got} over its cap of {cap}'
        floor = int(round(rich.floors[i] * cfg.n_per_rung))
        supply = rep.supply.get(name, 0)
        if supply >= floor:
            assert got >= floor, (
                f'{name}: floor asked {floor}, supply had {supply}, got {got}')
    assert rep.supply, 'the report carries no supply histogram to check against'
    checked = [b for i, b in enumerate(rich.names)
               if rich.floors[i] > 0
               and rep.supply.get(b, 0) >= round(rich.floors[i] * cfg.n_per_rung)]
    assert checked, (
        'no floor bucket had the supply to meet its floor, so every floor '
        'check above was skipped and this test passed on an empty set')
    return (' '.join(f'{b}:{len(s.where(bucket=b))}' for b in rich.names)
            + f'   floors verified: {len(checked)}')


def t_floor_frame_trades_count_for_mix():
    """'ask' keeps the tiles and lets the mix drift; 'taken' keeps the mix.

    THE FIXTURE HAS TO BE SUPPLY-STARVED OR THE TWO FRAMES ARE THE SAME THING.
    Where the supply is two orders of magnitude past the ask -- ds 1 and 2 on
    a real slide -- both frames return `n_per_rung` with the target mix, and a
    test written there would pass under either implementation. So this asks for
    more than `_graded_mask` can supply at the target proportions, which is the
    ds 32 situation the 3b probe of 2026-08-27 measured.

    The assertion is a COMPARISON between the two frames, not a threshold on
    either: 'taken' must hold the mix closer AND return no more tiles. A
    tolerance on the mix alone would pass for an implementation that simply
    took fewer tiles at random.
    """
    wsi, mask = _graded_mask()
    plans = _plans((1.0,))
    out = {}
    for frame in ('ask', 'taken'):
        cfg = _cfg(richness=RichnessConfig(floor_frame=frame), n_per_rung=300)
        s = TileSampler(wsi, mask, cfg).sample(plans)
        rep = s.reports[1.0]
        got = [len(s.where(bucket=b)) for b in cfg.richness.names]
        total = sum(got)
        assert total, f'{frame} placed nothing; the fixture cannot compare'
        share = [g / total for g in got]
        drift = sum(abs(a - b) for a, b in zip(share, cfg.richness.targets))
        out[frame] = (total, drift, rep.n_goal, got)

    (n_ask, d_ask, g_ask, got_ask) = out['ask']
    (n_tk, d_tk, g_tk, got_tk) = out['taken']
    assert g_ask == 300, f"'ask' scaled the rung to {g_ask}; it must not"
    assert g_tk < 300, (
        f"'taken' left the goal at {g_tk}, so the supply was not short and "
        f'this fixture cannot tell the frames apart -- raise n_per_rung')
    assert n_tk <= n_ask, (
        f"'taken' returned {n_tk} against 'ask' {n_ask}; scaling the rung down "
        f'cannot return more tiles')
    assert d_tk < d_ask, (
        f"'taken' drifted {d_tk:.3f} from the target mix and 'ask' {d_ask:.3f} "
        f'-- the frame that scales the rung is the one that is supposed to '
        f'hold the proportions')
    return (f'ask {n_ask} tiles drift {d_ask:.2f}   '
            f'taken {n_tk} tiles drift {d_tk:.2f} (goal {g_tk})')


def t_a_zero_cap_is_never_filled():
    """Including by the inheritance set, which every other quota is advisory
    for. A zero cap is where the tissue gate went, and a gate the carried
    centres walk through is not a gate."""
    wsi, mask = _graded_mask()
    cfg = _cfg(richness=RichnessConfig(), inherit=InheritConfig(share=0.5))
    s = TileSampler(wsi, mask, cfg).sample(_plans((1.0, 2.0)))
    zero = [n for n, c in zip(cfg.richness.names, cfg.richness.caps) if c == 0]
    assert zero, 'the fixture has no zero-capped bucket to test'
    # A zero-capped bucket the mask never produced is not evidence of a cap
    # holding -- it is evidence of nothing. At least one of them has to be in
    # the candidate pool for the emptiness below to mean anything.
    supplied = [n for n in zero
                if any(r.supply.get(n, 0) for r in s.reports.values())]
    assert supplied, (
        f'none of the zero-capped buckets {zero} appears in the candidate '
        f'pool, so their being empty says nothing about the cap')
    for name in zero:
        assert not len(s.where(bucket=name)), (
            f'{name} is capped at 0 and got {len(s.where(bucket=name))}')
    inherited = [x for x in s if x.meta.inherit_id >= 0]
    assert inherited, 'no inherited tiles placed; the fixture cannot show this'
    return (f'{len(supplied)} of {len(zero)} zero-capped buckets were supplied and stayed empty, {len(inherited)} inherited')


def t_a_scorer_that_reads_pixels_refuses_rather_than_approximating():
    """`saturation` and `entropy` are registered but unwritten. A mask-derived
    stand-in would BE `background` under a second name, and the two would hand
    identical corpora two different sampler_ids."""
    from TileSampler import SCORERS
    for name in ('saturation', 'entropy'):
        try:
            SCORERS[name](None, np.zeros((1, 2)), _plans((1.0,))[0])
        except NotImplementedError:
            continue
        raise AssertionError(f'{name} returned something')
    return 'both refuse'


# ══════════════════════════════════════════════════════════════════════════════
#  4. inherit
# ══════════════════════════════════════════════════════════════════════════════

def t_a_chain_is_the_same_level_0_centre_at_every_rung():
    wsi, mask = _one_big_block()
    cfg = _cfg(inherit=InheritConfig(share=0.5, stack_kind='F'))
    s = TileSampler(wsi, mask, cfg).sample(_plans((1.0, 2.0, 4.0)))
    chains = s.stacks()
    assert chains, 'no complete chain at all'
    for cid, members in chains.items():
        cx = {round(m.meta.centre_l0[0]) for m in members}
        cy = {round(m.meta.centre_l0[1]) for m in members}
        assert len(cx) == 1 and len(cy) == 1, (
            f'chain {cid} is not one centre: {sorted(cx)} {sorted(cy)}')
        assert len({m.meta.ds for m in members}) == len(members)
    return f'{len(chains)} chains, {len(next(iter(chains.values())))} rungs each'


def t_inherited_positions_are_exempt_from_the_overlap_bound():
    """Carried centres at a coarse rung almost coincide -- the footprint grew
    and they did not move. Running them through the bound drops members and
    leaves `inherit_id`s that do not resolve, which a survival analysis reads
    as "the keypoint died" when it means "the tile was never cut".

    The decoy is the count: every centre that passes the tissue gate must be
    present at the coarse rung, breaching or not, and the breaches must be
    REPORTED rather than acted on.
    """
    # `_wide_block`, not `_one_big_block`: at ds 8 the footprint is 512 level-0
    # px and the smaller region holds exactly one, so no two inherited tiles
    # could overlap there and the check would pass without testing anything.
    wsi, mask = _wide_block()
    cfg = _cfg(inherit=InheritConfig(share=1.0), n_per_rung=20,
               overlap=OverlapConfig())        # 0 == the tile, disjoint
    plans = _plans((1.0, 8.0))
    s = TileSampler(wsi, mask, cfg).sample(plans)
    coarse = s.where(ds=8.0, origin='inherit')
    breaching = sum(1 for x in coarse if x.meta.overlap_max > 0)
    assert breaching > 0, (
        'no inherited tile overlaps another at ds 8, so this fixture cannot '
        'show the exemption -- raise inherit.share or the ds gap')
    assert s.reports[8.0].n_inherit_breaching == breaching, (
        f'the report says {s.reports[8.0].n_inherit_breaching} breaches and '
        f'{breaching} tiles carry one; the exemption has to be counted, not '
        f'only allowed')
    return f'{breaching}/{len(coarse)} inherited tiles breach, all kept'


def t_stacks_returns_complete_chains_only_and_names_what_is_missing():
    """A four-rung chain handed over as six reads as "died at the two missing
    rungs". `incomplete()` is the other half."""
    wsi, mask = _one_big_block()
    cfg = _cfg(inherit=InheritConfig(share=0.6))
    s = TileSampler(wsi, mask, cfg).sample(_plans((1.0, 2.0)))
    assert s.stacks(), 'nothing complete to break'

    # Break one chain by hand: drop its coarse member, as a failed gate would.
    victim = next(iter(s.stacks()))
    s.samples = [x for x in s.samples
                 if not (x.meta.inherit_id == victim and x.meta.ds == 2.0)]
    assert victim not in s.stacks(), 'a broken chain came back as complete'
    assert victim in s.stacks(complete_only=False), 'and then vanished entirely'
    assert s.incomplete().get(victim) == [2.0], s.incomplete().get(victim)
    return f'chain {victim} missing ds 2, named'


def t_bucket_frame_changes_what_a_chain_carries():
    """The two values must give different corpora. If the carried bucket were
    silently recomputed, `at_inherit` would BE `per_rung` while sampler_id
    still separated them -- two identical corpora under two names."""
    wsi, mask = _wide_block()
    plans = _plans((1.0, 4.0))
    made = {}
    for frame in ('per_rung', 'at_inherit'):
        cfg = _cfg(richness=_permissive(bucket_frame=frame),
                   inherit=InheritConfig(share=0.8))
        s = TileSampler(wsi, mask, cfg).sample(plans)
        drift = 0
        for members in s.stacks().values():
            if len({m.meta.bucket for m in members}) > 1:
                drift += 1
        made[frame] = (drift, len(s.stacks()))
        # WITHOUT THIS the check passes on nothing. `drift == 0` is trivially
        # true when there are no chains, and the first run of this file
        # reported `per_rung drift 0/0, at_inherit drift 0/0` -- a green line
        # over an empty set. A vacuous pass is worse than a failure: it is a
        # failure that nobody will look at.
        assert made[frame][1] > 0, (
            f'{frame}: no complete chain, so the comparison is over an empty '
            f'set. Raise inherit.share, or use a region the coarse rung can '
            f'place several tiles in')
    assert made['at_inherit'][0] == 0, (
        f"at_inherit let {made['at_inherit'][0]} chains drift between buckets; "
        f'a carried bucket that is recomputed is not carried')
    return (f"per_rung drift {made['per_rung'][0]}/{made['per_rung'][1]}, "
            f"at_inherit drift 0/{made['at_inherit'][1]}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. identity
# ══════════════════════════════════════════════════════════════════════════════

def t_sampler_id_moves_with_every_axis():
    base = _cfg()
    same = _cfg()
    assert base.sampler_id() == same.sampler_id(), 'the hash is not stable'
    moves = {
        'tile':      _cfg(tile=TILE * 2),
        'n_per_rung': _cfg(n_per_rung=41),
        'seed':      _cfg(seed=1),
        'candidates': _cfg(candidates='random'),
        'richness.scorer': _cfg(richness=RichnessConfig(scorer='entropy')),
        'richness.bucket_frame':
            _cfg(richness=RichnessConfig(bucket_frame='at_inherit')),
        'richness.floors':
            _cfg(richness=RichnessConfig(
                floors=(0.10, 0.15, 0.50, 0.0, 0.0, 0.0, 0.0))),
        'richness.caps':
            _cfg(richness=RichnessConfig(
                caps=(0.20, 0.25, 0.60, 0.20, 0.20, 0.0, 0.0))),
        'richness.floor_frame':
            _cfg(richness=RichnessConfig(floor_frame='taken')),
        'overlap.grid_step': _cfg(overlap=OverlapConfig(grid_step=TILE // 2,
                                                        max_overlap_ratio=1.0)),
        'overlap.max_overlap_ratio':
            _cfg(overlap=OverlapConfig(max_overlap_ratio=0.5)),
        'overlap.overlapping_share':
            _cfg(overlap=OverlapConfig(overlapping_share=0.5)),
        'inherit.stack_kind': _cfg(inherit=InheritConfig(stack_kind='R')),
        'inherit.share': _cfg(inherit=InheritConfig(share=0.5)),
        'inherit.source_rung': _cfg(inherit=InheritConfig(source_rung=4.0)),
    }
    for name, cfg in moves.items():
        assert cfg.sampler_id() != base.sampler_id(), (
            f'{name} changed and sampler_id did not; two corpora would share '
            f'a filename')
    return f'{len(moves)} fields, all move it'


def t_sampler_id_does_not_move_with_provenance():
    """`on_incomplete` decides what a READER is shown, not which tiles were
    cut. Forking the corpus on it would be a store that split on a reporting
    choice."""
    a = _cfg(inherit=InheritConfig(share=0.5, on_incomplete='drop'))
    b = _cfg(inherit=InheritConfig(share=0.5, on_incomplete='keep'))
    assert a.sampler_id() == b.sampler_id(), (
        'on_incomplete moved sampler_id; it is in _NOT_IDENTITY for a reason')
    assert 'inherit.on_incomplete' in a.provenance()
    return 'on_incomplete recorded, not hashed'


# ══════════════════════════════════════════════════════════════════════════════
#  6. access
# ══════════════════════════════════════════════════════════════════════════════

def t_where_returns_a_container_and_composes_with_stacks():
    wsi, mask = _one_big_block()
    s = TileSampler(wsi, mask,
                    _cfg(inherit=InheritConfig(share=0.6))).sample(
        _plans((1.0, 2.0)))
    fine = s.where(ds=1.0)
    assert isinstance(fine, TileSampler) and len(fine) < len(s)
    assert all(x.meta.ds == 1.0 for x in fine)
    composed = s.where(origin='inherit').stacks(complete_only=False)
    assert composed, 'filter then group returned nothing'
    return f'{len(s)} -> where(ds=1) {len(fine)} -> stacks {len(composed)}'


def t_neighbours_separates_same_rung_from_cross_rung():
    """The two questions are diversity and leakage, and they are the same
    relation asked of different pairs."""
    wsi, mask = _one_big_block()
    s = TileSampler(wsi, mask,
                    _cfg(inherit=InheritConfig(share=1.0))).sample(
        _plans((1.0, 4.0)))
    i = next(k for k, x in enumerate(s) if x.meta.ds == 1.0
             and x.meta.inherit_id >= 0)
    cross = s.neighbours_of(i, same_rung=False)
    assert cross, 'an inherited fine tile overlaps nothing coarse -- impossible'
    assert all(s[j].meta.ds != 1.0 for j in cross)
    same = s.neighbours_of(i, same_rung=True)
    assert all(s[j].meta.ds == 1.0 for j in same)
    return f'{len(same)} same-rung, {len(cross)} cross-rung'


def t_unregistered_overlaps_finds_content_that_is_not_a_chain():
    """A ds 1 tile inside a ds 32 tile is the same tissue on both sides of a
    train/val split, and nothing else in the pipeline would notice. Registered
    chains are excluded, because those are deliberate."""
    wsi, mask = _one_big_block()
    s = TileSampler(wsi, mask,
                    _cfg(inherit=InheritConfig(share=0.3))).sample(
        _plans((1.0, 4.0)))
    found = s.unregistered_overlaps()
    assert found, 'no cross-rung overlap at all, which cannot be right'
    for i, j, _ in found:
        a, b = s[i].meta, s[j].meta
        assert a.ds != b.ds
        assert not (a.inherit_id >= 0 and a.inherit_id == b.inherit_id), (
            'a registered chain was reported as unregistered')
    return f'{len(found)} unregistered cross-rung pairs'


def t_overlap_with_divides_by_the_smaller_footprint():
    """A ds 1 tile lying wholly inside a ds 32 one shares 100 per cent of
    ITSELF and 0.1 per cent of the other. Dividing by the larger would report
    containment as almost no overlap -- which is the cross-rung case this has
    to catch."""
    small = SampleMeta(slide='s', ds=1.0, level=0, x=1000, y=1000,
                       tile_size=TILE, read_size=TILE, footprint_l0=TILE)
    big = SampleMeta(slide='s', ds=32.0, level=0, x=0, y=0,
                     tile_size=TILE, read_size=TILE, footprint_l0=TILE * 32)
    assert abs(small.overlap_with(big) - 1.0) < 1e-9, small.overlap_with(big)
    assert abs(big.overlap_with(small) - 1.0) < 1e-9, 'not symmetric'
    other = SampleMeta(slide='other', ds=1.0, level=0, x=1000, y=1000,
                       tile_size=TILE, read_size=TILE, footprint_l0=TILE)
    assert small.overlap_with(other) == 0.0, 'two slides cannot overlap'
    return 'containment reads 1.0, and slides do not mix'


# ══════════════════════════════════════════════════════════════════════════════
#  7. carry
# ══════════════════════════════════════════════════════════════════════════════

def t_a_sample_meta_survives_pickling():
    """THE DATALOADER CONSTRAINT.

    An openslide handle is not picklable, so a meta holding one works in one
    process and dies the moment `num_workers > 0` -- with a pickling error a
    long way from the cause. This is the check that keeps the streaming mode
    usable, and it is cheap enough to have no excuse.
    """
    wsi, mask = _one_big_block()
    s = TileSampler(wsi, mask, _cfg()).sample(_plans((1.0,)))
    meta = s[0].meta
    back = pickle.loads(pickle.dumps(meta))
    assert back == meta, 'a SampleMeta did not survive a round trip'
    assert not any(hasattr(getattr(meta, f), 'read_region')
                   for f in vars(meta)), 'a meta is carrying a slide handle'
    return 'pickled and back, no handle'


def t_materialise_takes_the_reader_and_release_keeps_the_meta():
    wsi, mask = _one_big_block()
    s = TileSampler(wsi, mask, _cfg()).sample(_plans((1.0,)))
    one = s[0]
    assert one.image is None
    one.materialise(wsi)
    assert one.image is not None and one.image.shape == (TILE, TILE, 3)
    # the fake reader encodes the request, so this says WHERE it read
    assert int(one.image[0, 0, 0]) == one.meta.x % 256
    before = one.meta
    one.release()
    assert one.image is None and one.meta is before
    return f'read at ({one.meta.x}, {one.meta.y}), released'


def t_an_R_sample_is_degraded_and_restored_to_the_tile_size():
    wsi, mask = _one_big_block()
    s = TileSampler(wsi, mask, _cfg()).sample([resolution_plan(4.0, TILE)])
    img = s[0].materialise(wsi).image
    assert img.shape == (TILE, TILE, 3), (
        f'an R tile came back {img.shape}; the whole point is that the output '
        f'side is held so a fixed-input student can eat it')
    return f'{img.shape} at ds 4'


def t_images_refuses_rather_than_reading_for_you():
    """A helper that quietly materialised would undo the streaming mode: the
    point is that pixels are read once and dropped."""
    wsi, mask = _one_big_block()
    s = TileSampler(wsi, mask, _cfg()).sample(_plans((1.0,)))
    try:
        s.images()
    except RuntimeError as e:
        assert 'materialise' in str(e)
        s.materialise(wsi)
        assert len(s.images()) == len(s)
        return 'refused, then served after materialise()'
    raise AssertionError('images() read for us')


def t_save_and_load_round_trip_every_axis():
    wsi, mask = _one_big_block()
    cfg = _cfg(inherit=InheritConfig(share=0.5))
    s = TileSampler(wsi, mask, cfg).sample(_plans((1.0, 2.0)))
    with tempfile.TemporaryDirectory() as tmp:
        s.save(tmp, with_images=False)
        back = TileSampler.load(tmp)
        assert len(back) == len(s)
        for a, b in zip(s, back):
            assert a.meta == b.meta, f'{a.meta}\n{b.meta}'
        assert back.stacks(complete_only=False), 'chains did not survive'
    return f'{len(s)} rows, every column'


def t_load_refuses_a_config_that_is_not_the_one_it_was_cut_with():
    wsi, mask = _one_big_block()
    cfg = _cfg()
    s = TileSampler(wsi, mask, cfg).sample(_plans((1.0,)))
    with tempfile.TemporaryDirectory() as tmp:
        s.save(tmp)
        TileSampler.load(tmp, cfg=cfg)                # the right one is fine
        try:
            TileSampler.load(tmp, cfg=_cfg(seed=99))
        except ValueError as e:
            assert 'sampler_id' in str(e)
            return 'refused'
    raise AssertionError('a mismatched config was accepted')


def t_a_plain_openslide_handle_is_refused():
    class _Plain:
        level_count = 1
        level_downsamples = [1.0]

    wsi, mask = _one_big_block()
    try:
        TileSampler(_Plain(), mask, _cfg())
    except TypeError as e:
        assert 'SafeSlide' in str(e)
        return 'refused'
    raise AssertionError('a handle with no read_region_rgb was accepted')


def t_the_old_api_refuses_with_the_replacement():
    wsi, mask = _one_big_block()
    for call, needle in (
        (lambda: TileSampler(wsi, mask, tile_size=256), 'SamplerConfig'),
        (lambda: TileSampler(wsi, mask, _cfg()).sample(n=10), 'RungPlan'),
    ):
        try:
            call()
        except TypeError as e:
            assert needle in str(e), str(e)
            assert 'NO default' in str(e)
        else:
            raise AssertionError('the old API still works, silently')
    return 'both entrances name their replacement'


# ══════════════════════════════════════════════════════════════════════════════
#  8. equivalence -- against PatchGrid, which is the other implementation
# ══════════════════════════════════════════════════════════════════════════════

def _patchgrid_positions(region, tile, ds, overlap):
    """PatchGrid's positions for one region, converted to level-0."""
    from PatchingLib import PatchGrid
    w_n, h_n = int(region.w / ds), int(region.h / ds)
    if w_n < tile or h_n < tile:
        return set()
    grid = PatchGrid.from_size(w_n, h_n, tile, overlap=overlap,
                               x_offset=int(region.x / ds),
                               y_offset=int(region.y / ds), ds=ds)
    return {(int(round(i.x * ds)), int(round(i.y * ds)))
            for i in grid.iter_infos()}


def _lattice_positions(sampler, plan):
    sampler._prepare_regions(plan)
    try:
        return {(int(x), int(y)) for x, y in sampler._lattice(plan)}
    finally:
        sampler._restore_regions()


def t_the_main_grid_is_exactly_patchgrids():
    """`grid_step == tile` against `PatchGrid(overlap=False)`.

    Set equality, both ways. This is the check that would let `_lattice` be
    deleted in favour of the one implementation -- and it is the reason to run
    it rather than to assume it: `PatchGrid` is what
    `QueryPatchContainer.extract_all` places QUERY patches with, so a sampler
    that disagreed would put reference tiles on a different lattice than the
    queries they are matched against.
    """
    wsi, mask = _one_big_block()
    s = TileSampler(wsi, mask, _cfg())
    for ds in (1.0, 2.0):
        plan = _plans((ds,))[0]
        s._prepare_regions(plan)
        regions = list(s.mask.tissue_regions)
        s._restore_regions()
        want = set()
        for r in regions:
            want |= _patchgrid_positions(r, TILE, ds, overlap=False)
        got = _lattice_positions(s, plan)
        assert got == want, (
            f'ds {ds}: {len(got - want)} positions the lattice invents, '
            f'{len(want - got)} PatchGrid has and it does not')
    return 'identical at ds 1 and ds 2, both directions'


def t_the_half_step_lattice_is_a_different_definition_from_overlap_true():
    """AND IT IS NOT A BUG IN EITHER. Two questions, two answers.

        PatchGrid(overlap=True)   half positions only BETWEEN two main ones,
                                  so `len(main) - 1` of them. That is right for
                                  a QUERY image, where the overlap patch exists
                                  to cover the seam between two main patches.
        grid_step = tile // 2     every half position that FITS. That is right
                                  for a SAMPLER, where a legal position is a
                                  candidate whether or not it has a neighbour.

    They differ by at most one per axis, at the far edge: at length 100 the
    lattice offers 32 and PatchGrid offers nothing; at 160 it offers 96 and
    PatchGrid stops at 32. Pinned here so that "use PatchGrid for everything"
    is a decision about which definition is wanted rather than an assumption
    that there is only one.
    """
    wsi, mask = _one_big_block()
    cfg = _cfg(overlap=OverlapConfig(grid_step=TILE // 2,
                                     max_overlap_ratio=1.0,
                                     overlapping_share=1.0))
    s = TileSampler(wsi, mask, cfg)
    plan = _plans((1.0,))[0]
    s._prepare_regions(plan)
    regions = list(s.mask.tissue_regions)
    s._restore_regions()
    want = set()
    for r in regions:
        want |= _patchgrid_positions(r, TILE, 1.0, overlap=True)
    got = _lattice_positions(s, plan)

    assert want <= got, (
        f'{len(want - got)} PatchGrid positions the lattice does not offer; '
        f'the lattice is supposed to be the SUPERSET -- every position that '
        f'fits, against only the ones between two mains')
    extra = got - want
    assert extra, (
        'the two agreed exactly, so this fixture cannot show the difference '
        '-- pick a region whose side is not a whole multiple of the tile')
    return f'{len(got)} lattice, {len(want)} PatchGrid, {len(extra)} extra'


def t_neither_generator_places_a_partial_tile():
    """The thing I wrongly expected to differ.

    A common lattice writes one last tile FLUSH against the far edge so the
    edge is covered, and that tile overlaps its neighbour. `PatchGrid` does
    not: `_full_grid_starts` (PatchingLib.py:197) is `range(0, length, tile)`
    filtered to the starts that fit, and drops the partial. So does `arange`.
    Pinned in both, because a flush-right tile would put an overlapping pair
    into a lattice whose whole claim is that it has none.
    """
    wsi, mask = _one_big_block()
    s = TileSampler(wsi, mask, _cfg())
    plan = _plans((1.0,))[0]
    s._prepare_regions(plan)
    regions = list(s.mask.tissue_regions)
    s._restore_regions()
    fp = int(plan.footprint_l0)
    for r in regions:
        for x, y in _patchgrid_positions(r, TILE, 1.0, overlap=False):
            assert x + fp <= r.x + r.w and y + fp <= r.y + r.h
    for x, y in _lattice_positions(s, plan):
        inside = any(x + fp <= r.x + r.w and y + fp <= r.y + r.h
                     and x >= r.x and y >= r.y for r in regions)
        assert inside, f'({x}, {y}) is not wholly inside any region'
    return 'no flush-right tile in either'


_SECTIONS = {
    'units':    ['t_the_lattice_step_is_output_pixels_and_not_ds',
                 't_every_placed_tile_keeps_its_reserve_inside_the_scanned_rectangle',
                 't_the_reserve_is_derived_from_the_margin_and_not_asked_for',
                 't_an_R_rung_reads_level_0_and_an_F_rung_grows',
                 't_reserve_is_what_must_fit_not_what_the_tile_covers'],
    'overlap':  ['t_a_disjoint_lattice_overlaps_nothing_and_the_random_arm_does',
                 't_a_half_step_lattice_under_a_tight_bound_is_refused',
                 't_the_overlapping_share_is_a_budget_and_binds',
                 't_every_jitter_offset_is_disjoint_and_off_lattice',
                 't_a_top_up_under_a_disjoint_lattice_is_refused',
                 't_grid_step_equal_to_the_tile_is_refused_as_a_synonym_of_zero'],
    'richness': ['t_buckets_cut_where_the_edges_say',
                 't_targets_split_the_remainder_over_the_askers_only',
                 't_caps_must_be_able_to_reach_a_full_rung',
                 't_floors_are_met_before_the_free_fill',
                 't_a_zero_cap_is_never_filled',
                 't_floor_frame_trades_count_for_mix',
                 't_a_scorer_that_reads_pixels_refuses_rather_than_approximating'],
    'inherit':  ['t_a_chain_is_the_same_level_0_centre_at_every_rung',
                 't_inherited_positions_are_exempt_from_the_overlap_bound',
                 't_stacks_returns_complete_chains_only_and_names_what_is_missing',
                 't_bucket_frame_changes_what_a_chain_carries'],
    'identity': ['t_sampler_id_moves_with_every_axis',
                 't_sampler_id_does_not_move_with_provenance'],
    'access':   ['t_where_returns_a_container_and_composes_with_stacks',
                 't_neighbours_separates_same_rung_from_cross_rung',
                 't_unregistered_overlaps_finds_content_that_is_not_a_chain',
                 't_overlap_with_divides_by_the_smaller_footprint'],
    'equivalence': ['t_the_main_grid_is_exactly_patchgrids',
                    't_the_half_step_lattice_is_a_different_definition_from_overlap_true',
                    't_neither_generator_places_a_partial_tile'],
    'carry':    ['t_a_sample_meta_survives_pickling',
                 't_materialise_takes_the_reader_and_release_keeps_the_meta',
                 't_an_R_sample_is_degraded_and_restored_to_the_tile_size',
                 't_images_refuses_rather_than_reading_for_you',
                 't_save_and_load_round_trip_every_axis',
                 't_load_refuses_a_config_that_is_not_the_one_it_was_cut_with',
                 't_a_plain_openslide_handle_is_refused',
                 't_the_old_api_refuses_with_the_replacement'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    args = ap.parse_args()

    for section in (args.only or list(_SECTIONS)):
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            check(name[2:].replace('_', ' '), globals()[name])

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
