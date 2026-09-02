#!/usr/bin/env python3
"""Tests for training/SuperPathPoint/common/DsLadder.py.

    python utilities/test_modules/test_ds_ladder.py
    python utilities/test_modules/test_ds_ladder.py --wsi /path/to/slide.svs

No model, no GPU, and by default no WSI: the two pyramids it checks against are
the ones already measured from this project's own runs (spec.md 6.5), written
out as plain lists. That is the reason `DsLadder.plan` takes `level_downsamples`
rather than a slide handle -- a resolver that can only be tested by opening a
40 GB MIRAX is a resolver nobody tests.

Pass `--wsi` to additionally print the plan for a real slide. That path is a
print, not an assertion: it exists so a human can look at a real pyramid, and
the assertions stay runnable everywhere.

WHAT THIS IS DEFENDING AGAINST
------------------------------
One thing, mostly: **silently upsampling**. If the resolver picks a level
coarser than the requested rung, the reader still returns an image of the right
size and everything downstream works. What it contains is interpolation
texture -- and a keypoint detector trained on it will learn to fire on
interpolation. Nothing raises, and the failure looks like a model that is
slightly worse.

The check that carries the weight is therefore not a tolerance but a decoy:
`SafeSlide.coarser_level_for_downsample` rounds the OTHER way, and on a 4x
pyramid the two resolvers disagree at ds 2. Section 2 pins that disagreement, so
that anyone who later "simplifies" this module by reusing the SafeSlide helper
gets told what they broke.

Sections:
  1. resolve  -- which level, on both real pyramid shapes
  2. decoy    -- why SafeSlide's existing resolvers are not this one
  3. plan     -- footprint arithmetic and the read size
  4. reach    -- which rungs each tile_size can have at all
  5. guards   -- what has to raise
"""

from __future__ import annotations

import argparse
import os
import sys

# `utilities/` holds the one definition of the output roots (`_paths.py`), and
# `setup_import_paths` -- which puts every other package on the path -- is
# inside it, so it has to be reachable before anything else is imported.
#
# BOTH parents are inserted, and that is deliberate rather than sloppy: this
# file runs from `utilities/test_modules/` and from
# `utilities/test_modules/TestSuperPathPoint/`, one level deeper, and inserting
# both means the move needs no edit here. The one that is not `utilities/` is
# either the repo root or `test_modules/`; neither holds a `_paths.py`, and
# `setup_import_paths` puts the repo root on the path anyway.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))


from _paths import setup_import_paths                           # noqa: E402

setup_import_paths()

from DsLadder import (DEFAULT_RUNGS, DsLadder,           # noqa: E402
                             LEVEL_REL_TOL, finer_level_for_downsample)


#: The two pyramid shapes this project actually has, measured from the runs
#: quoted in spec.md 6.5 rather than assumed:
#:
#:   BRACS_1228   base_mpp 0.2519, 4 levels, mpp 0.252 / 1.008 / 4.031 / 8.062
#:   Ki67 MRXS    base_mpp 0.243, 10 levels, mpp 0.243 / 0.485 / 0.970 / ...
#:
#: The trailing 4.00003-style noise is deliberate: openslide derives downsamples
#: from rounded level dimensions, so a "4x" level reports 4.00003 as readily as
#: 4.0 (SafeSlide.py:80-83). A resolver that compares exactly lands a level away
#: over a part in 1e5, and these values are what make that regression visible.
BRACS_PYRAMID = [1.0, 4.00003, 16.00012, 32.00048]
KI67_PYRAMID = [1.0, 1.99998, 4.00001, 8.00004, 16.00009, 32.00021,
                64.00043, 128.0009, 256.0018, 512.0036]

_RESULTS = []


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ══════════════════════════════════════════════════════════════════════════════
#  1. resolve
# ══════════════════════════════════════════════════════════════════════════════

def t_picks_the_coarsest_level_that_does_not_upsample():
    """For every rung on both pyramids: chosen level ds <= rung ds.

    The invariant that makes the whole ladder honest. Stated as `<=` with the
    same relative slack the pyramid noise needs, not as `==`, because most rungs
    are not native on a 4x pyramid and never will be.
    """
    for name, pyramid in (('BRACS', BRACS_PYRAMID), ('Ki67', KI67_PYRAMID)):
        for rung in DEFAULT_RUNGS:
            level = finer_level_for_downsample(pyramid, rung)
            level_ds = pyramid[level]
            assert level_ds <= rung * (1 + LEVEL_REL_TOL), (
                f'{name} rung {rung}: picked level {level} with ds {level_ds}, '
                f'which is COARSER than the rung -- reaching it needs upsampling')
    return f'{len(DEFAULT_RUNGS)} rungs x 2 pyramids, none upsample'


def t_picks_the_coarsest_such_level_not_just_any():
    """Level 0 always satisfies "does not upsample"; the point is to read less.

    Without the `max()` this would return level 0 for every rung: correct, and
    catastrophically slow, because ds 32 would then read 32x32 the pixels and
    throw 1023 of every 1024 away. A test that only checked the invariant above
    would pass on that implementation.
    """
    table = {
        'BRACS': (BRACS_PYRAMID, {1.: 0, 2.: 0, 4.: 1, 8.: 1, 16.: 2, 32.: 3}),
        'Ki67':  (KI67_PYRAMID, {1.: 0, 2.: 1, 4.: 2, 8.: 3, 16.: 4, 32.: 5}),
    }
    for name, (pyramid, expected) in table.items():
        for rung, want in expected.items():
            got = finer_level_for_downsample(pyramid, rung)
            assert got == want, (
                f'{name} rung {rung}: picked level {got}, expected {want} '
                f'(pyramid {[round(d, 3) for d in pyramid]})')
    return 'BRACS 0,0,1,1,2,3 and Ki67 0,1,2,3,4,5 for rungs 1..32'


def t_pyramid_noise_does_not_move_the_choice():
    """4.00003 must still count as ds 4 for a rung of 4.

    This is `SafeSlide.py:80-83`'s hazard from the other direction: with an
    exact comparison, `4.00003 <= 4` is False and rung 4 falls back to level 0,
    reading 16x the pixels for the same tile. Silent, and only visible as the
    job being four times slower than it should be.
    """
    exact = finer_level_for_downsample([1.0, 4.0, 16.0], 4.0)
    noisy = finer_level_for_downsample([1.0, 4.00003, 16.00012], 4.0)
    assert exact == noisy == 1, (
        f'exact pyramid gave level {exact}, noisy gave {noisy}; both should be 1')
    # And the slack must not be so wide that a genuinely coarser level sneaks in.
    assert finer_level_for_downsample([1.0, 4.4, 16.0], 4.0) == 0, (
        'a level 10% coarser than the rung was accepted -- the tolerance is '
        'meant for rounding noise, not for a different level')
    return 'noise of 1e-5 tolerated, a 10% gap is not'


# ══════════════════════════════════════════════════════════════════════════════
#  2. decoy -- why SafeSlide's resolvers are not this one
# ══════════════════════════════════════════════════════════════════════════════

def _coarser_level_decoy(downsamples, downsample):
    """`SafeSlide.coarser_level_for_downsample`, transcribed (SafeSlide.py:367-369).

    Copied rather than imported so this test needs no openslide, and so the
    comparison stays fixed even if that method is later changed -- the point of
    the check is the ROUNDING DIRECTION, and it should not silently start
    passing because the other function moved.
    """
    threshold = float(downsample) * (1.0 - LEVEL_REL_TOL)
    return next((i for i, d in enumerate(downsamples) if d >= threshold),
                len(downsamples) - 1)


def t_disagrees_with_the_coarser_resolver_where_it_matters():
    """On a 4x pyramid at ds 2 the two resolvers differ, and only one is usable.

    `coarser_level_for_downsample` exists for routing a QUERY to a level, and its
    docstring carries a 1398-shot measurement showing that rounding coarse is
    six times better than rounding fine for that job. Both facts are true and
    neither makes it the right resolver for a ladder.

    If this check ever fails because the two now agree, that is the signal to go
    read both docstrings before deleting one of them.
    """
    ours = finer_level_for_downsample(BRACS_PYRAMID, 2.0)
    theirs = _coarser_level_decoy(BRACS_PYRAMID, 2.0)
    assert ours != theirs, (
        'the two resolvers agree at BRACS ds 2, so one of them has changed '
        'direction; DsLadder is only justified while they differ')
    assert BRACS_PYRAMID[ours] <= 2.0, \
        f'ours picked ds {BRACS_PYRAMID[ours]} for rung 2'
    assert BRACS_PYRAMID[theirs] > 2.0, (
        f'the decoy picked ds {BRACS_PYRAMID[theirs]}, which does not upsample '
        f'after all -- the premise of this test is gone')
    return (f'BRACS rung 2: ours level {ours} (ds {BRACS_PYRAMID[ours]:.0f}), '
            f'coarser-resolver level {theirs} (ds {BRACS_PYRAMID[theirs]:.0f}, '
            f'would upsample {BRACS_PYRAMID[theirs] / 2:.0f}x)')


# ══════════════════════════════════════════════════════════════════════════════
#  3. plan
# ══════════════════════════════════════════════════════════════════════════════

def t_read_size_and_footprint_agree():
    """read_size x level_ds is the footprint, and it is what was asked for.

    Two numbers written by different lines: `read_size` comes from
    `tile_size * shrink` and `footprint_l0` from `read_size * level_ds`. They
    agree only if `shrink` is `rung_ds / level_ds`, so agreement is evidence
    rather than a restatement.
    """
    ladder = DsLadder()
    worst = 0.0
    for pyramid in (BRACS_PYRAMID, KI67_PYRAMID):
        for tile in (256, 512, 1024):
            for plan in ladder.plan(pyramid, tile):
                assert plan.shrink >= 1.0 - LEVEL_REL_TOL, (
                    f'shrink {plan.shrink} < 1 at rung {plan.rung_ds} -- '
                    f'that is upsampling')
                assert plan.read_size >= 1
                rel = abs(plan.footprint_l0 - plan.requested_footprint_l0) \
                    / plan.requested_footprint_l0
                worst = max(worst, rel)
    assert worst < 2e-3, (
        f'footprint drifts from the request by {100 * worst:.3f}% -- more than '
        f'whole-pixel rounding of read_size can explain')
    return f'2 pyramids x 3 tile sizes x 6 rungs, max drift {100 * worst:.4f}%'


def t_native_rungs_read_without_shrinking():
    """Where the pyramid already has the rung, nothing is resampled.

    On Ki67 every rung of the default ladder is native. On BRACS only 1, 4, 16
    and 32 are -- ds 2 and ds 8 do not exist on a 4x pyramid and have to be made
    by shrinking level 0 and level 1. That asymmetry IS the reason the ladder is
    fixed rather than per-slide (spec.md 6.5), so it is worth pinning.
    """
    ladder = DsLadder()
    ki67 = {p.rung_ds: p.is_native for p in ladder.plan(KI67_PYRAMID, 256)}
    assert all(ki67.values()), f'Ki67 should be native at every rung, got {ki67}'

    bracs = {p.rung_ds: p.is_native for p in ladder.plan(BRACS_PYRAMID, 256)}
    expected = {1.: True, 2.: False, 4.: True, 8.: False, 16.: True, 32.: True}
    assert bracs == expected, f'BRACS native rungs {bracs}, expected {expected}'

    shrink_2 = next(p for p in ladder.plan(BRACS_PYRAMID, 256) if p.rung_ds == 2.)
    assert shrink_2.level == 0 and shrink_2.read_size == 512, (
        f'BRACS ds 2 should read 512 px of level 0, got level '
        f'{shrink_2.level} read {shrink_2.read_size}')
    return 'Ki67 native at 6/6, BRACS at 4/6; BRACS ds 2 reads 512 px of level 0'


# ══════════════════════════════════════════════════════════════════════════════
#  4. reach
# ══════════════════════════════════════════════════════════════════════════════

def t_reachable_reproduces_the_spec_table():
    """Which rungs each tile_size can have, against spec.md 6.5's table.

    The measured wall is 8192 level-0 px: at that footprint both datasets
    sampled 100/100, at 16384 Ki67 sampled 0/100 after 500 tries, and at 32768
    no region could hold a tile at all. So `max_footprint_l0=8192` should
    reproduce the spec's row counts exactly -- 6 rungs for tile 256, 5 for 512,
    4 for 1024.

    If this fails, the spec table and the code have drifted apart, and the spec
    is the thing a human reads before spending a GPU-week.
    """
    ladder = DsLadder()
    got = {tile: len(ladder.reachable(KI67_PYRAMID, tile, 8192))
           for tile in (256, 512, 1024)}
    expected = {256: 6, 512: 5, 1024: 4}
    assert got == expected, f'reachable rung counts {got}, spec.md 6.5 says {expected}'

    top = {tile: ladder.reachable(KI67_PYRAMID, tile, 8192)[-1].rung_ds
           for tile in (256, 512, 1024)}
    assert top == {256: 32., 512: 16., 1024: 8.}, \
        f'coarsest reachable rung per tile size is {top}'
    return 'tile 256/512/1024 -> 6/5/4 rungs, topping out at ds 32/16/8'


def t_footprint_is_tile_times_ds():
    """The quantity the wall is measured in, spelled once.

    tile 1024 at ds 16 is 16384 level-0 px -- the same footprint as tile 256 at
    ds 64, which is the configuration measured at 0/100. The two are the same
    ask of the tissue mask, and that equivalence is why the wall transfers
    between tile sizes at all.
    """
    ladder = DsLadder(rungs=(16.,))
    big = ladder.plan(KI67_PYRAMID, 1024)[0]
    small = DsLadder(rungs=(64.,)).plan(KI67_PYRAMID, 256)[0]
    assert abs(big.requested_footprint_l0 - small.requested_footprint_l0) < 1e-6, (
        f'tile 1024 x ds 16 = {big.requested_footprint_l0} but '
        f'tile 256 x ds 64 = {small.requested_footprint_l0}')
    assert big.requested_footprint_l0 == 16384
    return 'tile 1024 x ds 16 == tile 256 x ds 64 == 16384 L0 px'


# ══════════════════════════════════════════════════════════════════════════════
#  5. guards
# ══════════════════════════════════════════════════════════════════════════════

def t_finer_than_level_zero_raises():
    """Asking for a rung the slide does not have must not quietly upsample."""
    try:
        finer_level_for_downsample(BRACS_PYRAMID, 0.5)
    except ValueError as e:
        assert 'upsampl' in str(e).lower(), f'unhelpful message: {e}'
        return 'ds 0.5 on a level-0-is-1.0 pyramid raises and says why'
    raise AssertionError('a sub-level-0 downsample was accepted')


def t_ladder_guards_its_own_shape():
    """Empty, non-positive, or unsorted rungs are all mistakes worth a raise.

    Ascending order is required because a rung's INDEX is used as an ordering
    everywhere downstream -- Stage B's survival vector is indexed by it, and
    Stage C's relative rungs are differences of those indices. An unsorted
    ladder would make `j = +1` mean "one step coarser" for some pairs and "one
    step finer" for others, with nothing to signal it.
    """
    for bad, why in ((), 'empty'), ((1., -2.), 'negative'), ((4., 1.), 'unsorted'):
        try:
            DsLadder(rungs=bad)
        except ValueError:
            continue
        raise AssertionError(f'{why} rungs {bad} were accepted')
    return 'empty / negative / unsorted all raise'


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = {
    'resolve': ['t_picks_the_coarsest_level_that_does_not_upsample',
                't_picks_the_coarsest_such_level_not_just_any',
                't_pyramid_noise_does_not_move_the_choice'],
    'decoy':   ['t_disagrees_with_the_coarser_resolver_where_it_matters'],
    'plan':    ['t_read_size_and_footprint_agree',
                't_native_rungs_read_without_shrinking'],
    'reach':   ['t_reachable_reproduces_the_spec_table',
                't_footprint_is_tile_times_ds'],
    'guards':  ['t_finer_than_level_zero_raises', 't_ladder_guards_its_own_shape'],
}


def _print_real_slide(path, tile_size):
    """Not an assertion. A look at a real pyramid, for a human."""
    from SafeSlide import SafeSlide

    wsi = SafeSlide(path)
    try:
        print(f'\n[wsi] {os.path.basename(path)}')
        print(f'  base_mpp {wsi.base_mpp:.4f}  levels {wsi.level_count}')
        print('  native downsamples: '
              f'{[round(float(d), 3) for d in wsi.level_downsamples]}')
        print(f'  plan at tile_size {tile_size}:')
        for plan in DsLadder().plan_for(wsi, tile_size):
            print(f'    {plan.summary()}')
    finally:
        wsi.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS),
                    help='run only these sections (default: all)')
    ap.add_argument('--wsi', help='also print the plan for a real slide')
    ap.add_argument('--tile-size', type=int, default=256)
    args = ap.parse_args()

    for section in (args.only or sorted(_SECTIONS)):
        print(f'\n[{section}]')
        for name in _SECTIONS[section]:
            check(name[2:].replace('_', ' '), globals()[name])

    if args.wsi:
        _print_real_slide(args.wsi, args.tile_size)

    failed = [n for n, e in _RESULTS if e is not None]
    print(f'\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} passed')
    if failed:
        print('failed: ' + ', '.join(failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
