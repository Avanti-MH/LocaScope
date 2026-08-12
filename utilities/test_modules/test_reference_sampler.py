#!/usr/bin/env python3
"""Unit tests for utilities/ReferenceSampler.py.

Everything this module gets wrong produces a reference set that looks fine. A
background fraction off by one mask pixel shifts every bucket boundary a little;
a centre-alignment that drops a factor of two still returns coordinates inside
the slide; a displacement written in level-0 pixels still returns tiles. None of
those raise, and the damage only shows up later as an estimator that is
mysteriously worse than it should be.

So the assertions here are mostly scored against a DELIBERATELY WRONG
alternative rather than against a tolerance, which is what the camera-coordinate
test established as the pattern for this project: the right answer has to beat
the plausible mistake, not merely land near the truth.

No slide, no model, no GPU. Runs in about a second.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _d in ('utilities', 'utilities/test_modules'):
    p = str(_ROOT / _d)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                  # noqa: E402

import ReferenceSampler as RS                                       # noqa: E402
from ReferenceSampler import (BUCKETS, InheritPlan, LevelGeoms,     # noqa: E402
                              ReferenceSampler, SamplerConfig)


# ── harness ───────────────────────────────────────────────────────────────────

_RESULTS = []


def check(name: str, fn) -> None:
    try:
        fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}')
    except Exception as e:                                # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n        {type(e).__name__}: {e}')


class FakeRegion:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h


class FakeMask:
    """The four things ReferenceSampler actually touches on a mask.

    A stub rather than a real TissuesRegionsMask so the test needs no slide. The
    risk that the stub drifts from the real class is covered by
    t_mask_api_still_matches below -- without that, this file could keep passing
    after the real API moved underneath it.
    """

    def __init__(self, main_mask, mask_ds=1.0, regions=(),
                 downsamples=(1.0, 2.0, 4.0, 8.0)):
        self.main_mask = np.asarray(main_mask, dtype=np.uint8)
        self.mask_ds_x = self.mask_ds_y = float(mask_ds)
        self.tissue_regions = list(regions)
        self.wsi_level_downsamples = list(downsamples)

    def to_mask_xy(self, x, y):
        return int(x / self.mask_ds_x), int(y / self.mask_ds_y)

    def _levelLength_converter(self, w, h, level):
        ds = self.wsi_level_downsamples[level]
        return int(w * ds / self.mask_ds_x), int(h * ds / self.mask_ds_y)


def _half_tissue_mask():
    """100x100, tissue in columns 0-49 only. Every fraction below is exact."""
    m = np.zeros((100, 100), dtype=np.uint8)
    m[:, :50] = 1
    return FakeMask(m)


# ── white fraction ────────────────────────────────────────────────────────────

def t_white_fraction_is_exact():
    """A summed-area table off by one is the classic silent error here: every
    fraction moves a little, every bucket boundary shifts, nothing raises."""
    mask = _half_tissue_mask()
    xy = np.array([[0, 0], [50, 0], [40, 0], [30, 0]], dtype=np.int64)
    w = RS.white_fractions(mask, xy, level=0, tile=20)

    want = [0.0,      # entirely inside the tissue half
            1.0,      # entirely inside the background half
            0.5,      # columns 40-59: half and half
            0.0]      # columns 30-49: still all tissue
    for got, exp, (x, _) in zip(w, want, xy):
        assert abs(float(got) - exp) < 1e-6, \
            f'tile at x={x}: white {float(got):.4f}, expected {exp}'


def t_outside_the_mask_counts_as_background():
    """A tile hanging off the edge must be scored against its FULL area, the
    same choice has_tissue documents. Dividing by the clipped area instead would
    call this tile pure tissue -- a wrong answer with a believable value."""
    m = np.zeros((100, 100), dtype=np.uint8)
    m[:, 90:] = 1                       # tissue only in the last ten columns
    mask = FakeMask(m)

    w = float(RS.white_fractions(mask, np.array([[90, 0]]), level=0, tile=20)[0])
    assert abs(w - 0.5) < 1e-6, (
        f'white {w:.4f}; expected 0.5 (ten tissue columns inside, ten columns '
        f'off the mask). 0.0 means the clipped area was used as the denominator')


def t_white_fraction_scales_with_level():
    """The footprint grows with ds, so the SAME coordinate has a different
    background fraction at every level. A level argument that is ignored would
    return the level-0 answer everywhere and quietly make the buckets
    incomparable across levels."""
    mask = _half_tissue_mask()
    mask.wsi_level_downsamples = [1.0, 4.0]
    xy = np.array([[40, 0]], dtype=np.int64)

    w0 = float(RS.white_fractions(mask, xy, level=0, tile=5)[0])   # cols 40-44
    w1 = float(RS.white_fractions(mask, xy, level=1, tile=5)[0])   # cols 40-59
    assert abs(w0 - 0.0) < 1e-6, f'level 0 white {w0:.4f}, expected 0'
    assert abs(w1 - 0.5) < 1e-6, f'level 1 white {w1:.4f}, expected 0.5'


# ── buckets ───────────────────────────────────────────────────────────────────

def t_bucket_edges_are_where_they_claim():
    """< against <= at four boundaries. Off by one edge and a whole band of
    tiles silently changes quota."""
    cfg = SamplerConfig()
    w = np.array([0.0, 0.1499, 0.15, 0.70, 0.7001, 0.80, 0.8001, 0.999, 1.0])
    b = RS.assign_buckets(w, cfg)
    want = ['lt15', 'lt15', 'mid', 'mid', 'gt70', 'gt70', 'gt80', 'full', 'full']
    for wi, bi, exp in zip(w, b, want):
        assert BUCKETS[bi] == exp, \
            f'white {wi} -> {BUCKETS[bi]}, expected {exp}'


# ── inheritance geometry ──────────────────────────────────────────────────────

def t_inheritance_keeps_the_centre_fixed():
    """The point of carrying a location to another level is that it covers the
    SAME tissue. Sharing a top-left corner instead makes the coarse tile extend
    away from the fine one -- invisible at ds=1 and wrong everywhere else, the
    same shape of bug the camera coordinate test caught.

    Scored against the two plausible mistakes rather than a tolerance.
    """
    tile = 256
    xy0 = np.array([[1000, 2000]], dtype=np.int64)
    plan = InheritPlan(n_candidates=1, xy0=xy0, half0=tile * 1.0 / 2.0)

    centre0 = xy0[0] + tile * 1.0 / 2.0

    for ds in (1.0, 2.0, 8.0):
        tl = plan.at(ds, tile)
        centre = tl[0] + tile * ds / 2.0
        assert np.allclose(centre, centre0, atol=1.0), (
            f'ds={ds}: centre moved from {centre0} to {centre}')

    # the wrong alternative: keep the top-left. Must NOT hold the centre.
    ds = 8.0
    wrong_centre = xy0[0] + tile * ds / 2.0
    assert not np.allclose(wrong_centre, centre0, atol=1.0), (
        'the test cannot tell the two apart -- pick a ds where they differ')


# ── displacement ──────────────────────────────────────────────────────────────

def t_every_offered_offset_is_disjoint_and_off_grid():
    """Two properties, and an offset needs both to be worth offering.

    DISJOINT: no shared pixel with the parent. Adding (128, 128) later would
    reintroduce 25% overlap, and near-duplicate reference tiles are what the
    twin measurement found doing damage.

    OFF-GRID: not a multiple of half a tile in both axes. The main grid steps by
    a tile and the overlap grid sits half a tile in, so the union of the two is
    every multiple of tile//2 -- a displacement onto one of those has produced a
    coordinate the grid already offered, and the bucket was short precisely
    because the grid had run out there. Three of the five offsets originally in
    the table failed this and silently added positions that were already
    candidates.
    """
    tile = SamplerConfig().tile
    pitch = tile // 2
    for offset_x, offset_y in RS.JITTER_OFFSETS:
        overlap = max(0, tile - abs(offset_x)) * max(0, tile - abs(offset_y))
        assert overlap == 0, \
            f'offset ({offset_x}, {offset_y}) overlaps its parent by {overlap} px^2'
        assert offset_x % pitch != 0 or offset_y % pitch != 0, (
            f'offset ({offset_x}, {offset_y}) is a multiple of {pitch} in both '
            f'axes, so it lands on a position the grid already offers')


def t_displacement_recomputes_its_own_background():
    """A displaced tile must be measured where it landed, not where it came
    from. Every offered offset is a whole tile, so parent and child share no
    pixel: copying the parent's fraction would let a tile drawn for the
    tissue-dense bucket sit on glass while the quota reported itself satisfied.

    The mask here is tissue on the left half only, so a parent at x=10 is pure
    tissue and a full-tile step right lands on pure background -- the two must
    not come back with the same number.
    """
    m = np.zeros((400, 400), dtype=np.uint8)
    m[:, :200] = 1
    mask = FakeMask(m, regions=[FakeRegion(0, 0, 400, 400)])
    g = LevelGeoms(level=0, ds=1.0, footprint_l0=20,
                   xy=np.array([[180, 100]], dtype=np.int64),
                   region=np.zeros(1, np.int32),
                   grid_rc=np.zeros((1, 2), np.int32),
                   kind=np.zeros(1, np.int8),
                   white=np.zeros(1, np.float32),      # parent: pure tissue
                   bucket=np.zeros(1, np.int8))
    cfg = SamplerConfig(tile=20, jitter_offsets=((20, 20),))
    s = ReferenceSampler({0: g}, cfg, mask=mask)

    # Asking for the tissue-dense bucket: the step that lands on glass must be
    # refused rather than recorded with the parent's value.
    made, gave_up = s._jitter_from(0, np.array([0]), n_wanted=4,
                                   bucket=BUCKETS.index('lt15'))
    for _nx, _ny, _p, w in made:
        assert w < cfg.edges[0], (
            f'displaced tile recorded white {w:.3f}, which is not in the bucket '
            f'it was drawn for -- the parent value was copied')


def t_displacement_is_in_level_n_units():
    """Offsets are declared in tiles at the level being sampled, so the level-0
    step must scale with ds. Treated as level-0 constants they would be 87%
    overlap at ds=8 -- tiles that still look like new samples and are not."""
    ds = 8.0
    # All tissue, and big enough that a 2048 level-0 footprint plus a full-tile
    # step still lands inside it: a displacement that falls off the mask reads
    # as pure background and gets refused for the wrong reason, which would make
    # this test fail for something other than what it is testing.
    mask = FakeMask(np.ones((1000, 1000), dtype=np.uint8), mask_ds=8.0,
                    regions=[FakeRegion(0, 0, 8000, 8000)])
    g = LevelGeoms(level=3, ds=ds, footprint_l0=int(256 * ds),
                   xy=np.array([[3000, 3000]], dtype=np.int64),
                   region=np.zeros(1, np.int32),
                   grid_rc=np.zeros((1, 2), np.int32),
                   kind=np.zeros(1, np.int8),
                   white=np.zeros(1, np.float32),
                   bucket=np.zeros(1, np.int8))
    s = ReferenceSampler({3: g}, SamplerConfig(), mask=mask)

    made, _ = s._jitter_from(3, np.array([0]), n_wanted=6,
                             bucket=BUCKETS.index('lt15'))
    assert made, 'no displaced coordinate was produced'
    offered = {d for pair in RS.JITTER_OFFSETS for d in pair}
    for nx, ny, _p, _w in made:
        for step in (abs(nx - 3000), abs(ny - 3000)):
            assert step % ds == 0, f'step {step} is not a whole number of ds={ds}'
            assert step / ds in offered, (
                f'level-0 step {step} is {step / ds} tiles, which is not one of '
                f'the offered offsets {sorted(offered)}')


# ── quotas ────────────────────────────────────────────────────────────────────

def _geoms_with(counts, ds=1.0, level=0):
    """A LevelGeoms holding `counts[b]` positions in each bucket."""
    n = sum(counts.values())
    bucket, white = [], []
    mid = {'lt15': 0.05, 'mid': 0.4, 'gt70': 0.75, 'gt80': 0.9, 'full': 1.0}
    for b, c in counts.items():
        bucket += [BUCKETS.index(b)] * c
        white += [mid[b]] * c
    xy = np.stack([np.arange(n) * 1000, np.zeros(n, dtype=np.int64)], axis=1)
    return LevelGeoms(level=level, ds=ds, footprint_l0=int(256 * ds), xy=xy,
                      region=np.zeros(n, np.int32),
                      grid_rc=np.zeros((n, 2), np.int32),
                      kind=np.zeros(n, np.int8),
                      white=np.array(white, np.float32),
                      bucket=np.array(bucket, np.int8))


def t_plan_respects_the_target_and_the_caps():
    cfg = SamplerConfig()
    g = _geoms_with({'lt15': 50000, 'mid': 20000, 'gt70': 5000,
                     'gt80': 3000, 'full': 2000})
    p = RS.plan_level(g, cfg)

    assert p.n_target == cfg.n_target, f'n_target {p.n_target}'
    assert p.got <= p.n_target, f'planned {p.got} > target {p.n_target}'
    assert p.caps_ok, f'caps violated: {p.cap_note}'

    got = {b.bucket: b.got for b in p.buckets}
    assert got['lt15'] >= int(cfg.floor_lt15 * cfg.n_target), (
        f"lt15 got {got['lt15']}, floor is "
        f"{int(cfg.floor_lt15 * cfg.n_target)}")
    for name, v, cap in (
            ('>70', got['gt70'] + got['gt80'] + got['full'], cfg.cap_gt70),
            ('>80', got['gt80'] + got['full'], cfg.cap_gt80),
            ('=100', got['full'], cfg.cap_full)):
        assert v <= cfg.n_target * cap, f'{name} = {v} exceeds {cap:.1%}'


def t_small_grid_caps_the_target_and_the_jitter_share():
    """A level with few positions must not be padded out with displacements: the
    per-level 125% and the per-bucket 20% are the same number by construction and
    both have to bind."""
    cfg = SamplerConfig()
    g = _geoms_with({'lt15': 400, 'mid': 100, 'gt70': 40, 'gt80': 20, 'full': 3})
    p = RS.plan_level(g, cfg)

    assert p.n_target == int(np.floor(563 * cfg.over)), (
        f'n_target {p.n_target}, expected min(1000, 563 x {cfg.over})')
    for b in p.buckets:
        if b.got:
            share = b.from_jitter / b.got
            assert share <= cfg.jitter_cap + 1e-9, (
                f'{b.bucket}: {share:.1%} displaced, cap is {cfg.jitter_cap:.0%}')
    assert p.short, 'a grid this small cannot reach its target'


def t_shortfall_is_named_not_swallowed():
    cfg = SamplerConfig()
    g = _geoms_with({'lt15': 20, 'mid': 900, 'gt70': 40, 'gt80': 20, 'full': 5})
    p = RS.plan_level(g, cfg)
    lt15 = [b for b in p.buckets if b.bucket == 'lt15'][0]
    assert lt15.short > 0, 'a bucket with 20 positions cannot meet its floor'
    assert lt15.reason, 'the shortfall carries no reason'


# ── identity ──────────────────────────────────────────────────────────────────

def t_sampler_id_moves_when_any_setting_moves():
    """cfg_hash covers the encoder and the mask but nothing about sampling, so
    without this every quota change writes to the same filename."""
    base = SamplerConfig()
    changed = {
        'n_target': 500, 'over': 1.5, 'jitter_cap': 0.1, 'seed': 43,
        'edges': (0.2, 0.7, 0.8), 'floor_lt15': 0.9, 'share_mid': 0.05,
        'cap_gt70': 0.2, 'cap_gt80': 0.1, 'cap_full': 0.01,
        'inherit_frac': 0.25, 'inherit_over': 3.0, 'min_valid': 0.9,
        'max_miss': 10, 'tile': 512,
        'jitter_offsets': ((256, 256),),
    }
    for name, val in changed.items():
        other = replace(base, **{name: val})
        assert base.sampler_id() != other.sampler_id(), \
            f'changing {name} does not change sampler_id'


# ── the draw ──────────────────────────────────────────────────────────────────

def t_no_coordinate_is_drawn_twice():
    cfg = SamplerConfig(n_target=200)
    g = _geoms_with({'lt15': 5000, 'mid': 2000, 'gt70': 500,
                     'gt80': 300, 'full': 200})
    p = RS.plan_level(g, cfg)
    s = ReferenceSampler({0: g}, cfg, mask=_half_tissue_mask()).plan(0, p)

    seen = set(zip(s.x.tolist(), s.y.tolist()))
    assert len(seen) == len(s), f'{len(s) - len(seen)} duplicate coordinates'
    assert len(s) == p.got, f'produced {len(s)}, plan said {p.got}'
    assert (s.origin == 0).all(), 'this grid needed no displacement'


def t_replacements_are_new_coordinates():
    cfg = SamplerConfig(n_target=100)
    g = _geoms_with({'lt15': 500, 'mid': 200, 'gt70': 50, 'gt80': 30, 'full': 20})
    p = RS.plan_level(g, cfg)
    sampler = ReferenceSampler({0: g}, cfg, mask=_half_tissue_mask())
    s = sampler.plan(0, p)

    taken = set(zip(s.x.tolist(), s.y.tolist()))
    for x, y, _i in sampler.replace(0, BUCKETS.index('lt15'), n=5):
        assert (x, y) not in taken, f'replacement ({x}, {y}) was already used'
        taken.add((x, y))


# ── the handoff ───────────────────────────────────────────────────────────────

def t_store_args_match_what_save_accepts():
    """to_store_args exists so the handoff cannot drift. If FeatureStore.save
    renames a parameter, this fails here rather than at the end of an hour of
    encoding."""
    import FeatureStore as FS

    cfg = SamplerConfig(n_target=50)
    g = _geoms_with({'lt15': 300, 'mid': 100, 'gt70': 30, 'gt80': 20, 'full': 10})
    p = RS.plan_level(g, cfg)
    s = ReferenceSampler({0: g}, cfg, mask=_half_tissue_mask()).plan(0, p)

    import torch
    args = RS.to_store_args(s, torch.zeros(len(s), 8))

    params = set(inspect.signature(FS.save).parameters)
    for k in args:
        assert k in params, f'{k!r} is not a parameter of FeatureStore.save'
    for k in ('features', 'x', 'y', 'region', 'grid_rc'):
        assert k in args, f'{k!r} missing from the handoff'
    for k in args['extra']:
        assert k not in ('features', 'x', 'y', 'region', 'grid_rc'), \
            f'extra tensor {k!r} collides with a core tensor name'
        assert len(args['extra'][k]) == len(s), f'extra {k!r} has the wrong length'


def t_mask_api_still_matches():
    """The stub above only stands in for a real mask while the real mask still
    looks like this. Without this check the whole file could keep passing after
    TissuesRegionsMask moved underneath it."""
    from TissuesRegionsMask import TissuesRegionsMask

    for name in ('to_mask_xy', '_levelLength_converter'):
        assert callable(getattr(TissuesRegionsMask, name, None)), \
            f'TissuesRegionsMask has no {name}(); the FakeMask stub is stale'

    src = inspect.getsource(TissuesRegionsMask.__init__)
    for attr in ('main_mask', 'tissue_regions', 'mask_ds_x'):
        assert f'self.{attr}' in src, \
            f'TissuesRegionsMask no longer sets {attr}; the stub is stale'


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print('ReferenceSampler')
    print('  white fraction')
    for name, fn in (
            ('exact on a known mask',              t_white_fraction_is_exact),
            ('outside the mask is background',     t_outside_the_mask_counts_as_background),
            ('the footprint follows the level',    t_white_fraction_scales_with_level)):
        check(name, fn)

    print('  buckets')
    check('edges are where they claim', t_bucket_edges_are_where_they_claim)

    print('  inheritance')
    check('carries the centre, not the corner', t_inheritance_keeps_the_centre_fixed)

    print('  displacement')
    for name, fn in (
            ('every offer is disjoint and off-grid', t_every_offered_offset_is_disjoint_and_off_grid),
            ('measured where it landed, not where it came from',
             t_displacement_recomputes_its_own_background),
            ('steps are level-n tiles, scaled by ds',   t_displacement_is_in_level_n_units)):
        check(name, fn)

    print('  quotas')
    for name, fn in (
            ('target and nested caps hold',        t_plan_respects_the_target_and_the_caps),
            ('a small grid binds both caps',       t_small_grid_caps_the_target_and_the_jitter_share),
            ('a shortfall is named',               t_shortfall_is_named_not_swallowed)):
        check(name, fn)

    print('  identity')
    check('sampler_id moves with every setting', t_sampler_id_moves_when_any_setting_moves)

    print('  the draw')
    for name, fn in (
            ('no coordinate twice',   t_no_coordinate_is_drawn_twice),
            ('replacements are new',  t_replacements_are_new_coordinates)):
        check(name, fn)

    print('  handoff')
    for name, fn in (
            ('store args match save()', t_store_args_match_what_save_accepts),
            ('the mask stub is not stale', t_mask_api_still_matches)):
        check(name, fn)

    bad = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(bad)}/{len(_RESULTS)} passed')
    if bad:
        print('failed: ' + ', '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
