#!/usr/bin/env python3
"""Tests for SafeSlide against a slide with real holes.

There is no synthetic path here: the behaviour under test is how the openslide C
library reacts to a MIRAX cell the scanner never wrote, which cannot be faked.
The tests therefore need a slide that actually fails, and default to the one this
was diagnosed on.

  0. control      — a BARE OpenSlide really is destroyed by one failed read, so
                    the rest of the suite is testing something that matters
  1. good read    — an ordinary read is untouched, no reopen, no hole recorded
  2. bad read     — subdivision recovers tissue the whole-rect blank threw away,
                    scored against min_chunk=0, which IS the old behaviour
  3. survival     — metadata and further reads still work afterwards. This is
                    the whole point: a bare handle fails here
  4. sharing      — a second holder of the SAME object recovers too, which is
                    why LocaScopePipeline can hand one slide to every stage
  5. rgb          — the part with no image composites to the background colour
                    (white), not the black that .convert('RGB') produces
  6. valid        — the alpha channel separates real pixels from holes inside
                    ONE read, and agrees with the recorded hole list

Checks 2 to 6 were written before cfbef2d, when a failed read blanked the whole
requested rectangle, and they asserted that: reopens == 1, one hole covering the
whole request, every pixel transparent. Subdivision made all of that false and
nothing failed, because nothing ran this file. SafeSlide's own Attributes block
had drifted the same way and is why `reopens == 1` looked right.

Usage:
    python utilities/test_modules/test_safe_slide.py
    python utilities/test_modules/test_safe_slide.py \\
        --wsi /path/to/other.mrxs --bad-x .. --bad-y .. --bad-w .. --bad-h ..

Defaults describe S1103037: the whole bbox of tissue region index=2 fails,
while its top-left 256 px reads fine.
"""

import argparse
import os
import sys

# _paths holds the one definition of OUTPUT_ROOT for every package, so it lives
# in utilities/ rather than beside this file. That directory goes on sys.path
# here, because setup_import_paths -- which puts the rest there -- is inside it.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import openslide

from _paths import setup_import_paths
setup_import_paths()

from SafeSlide import SafeSlide          # noqa: E402


# ── 0. control: a bare handle really does die ────────────────────────────────

def validate_bare_handle_dies(path, bad):
    """Without SafeSlide, one failed read takes the whole object with it.

    If this ever stops failing, SafeSlide is solving a problem that no longer
    exists and the rest of this file is theatre -- so assert the breakage
    rather than assuming it.
    """
    w = openslide.OpenSlide(path)
    n_before = w.level_count                      # fine before

    try:
        w.read_region((bad[0], bad[1]), 0, (bad[2], bad[3]))
        raise AssertionError(
            f'expected a hole at {bad} but the read succeeded; '
            f'point --bad-* at a position that actually fails'
        )
    except openslide.lowlevel.OpenSlideError:
        pass

    # The killer detail: level_count reads no pixels, yet it is now dead too.
    try:
        w.level_count
        raise AssertionError('bare handle survived a failed read -- openslide '
                             'no longer latches its error state?')
    except openslide.lowlevel.OpenSlideError:
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass

    print(f'[PASS] control: bare OpenSlide dies on a failed read '
          f'(level_count was {n_before}, now raises)')


# ── 1. an ordinary read is unaffected ────────────────────────────────────────

def validate_good_read(path, good):
    w = SafeSlide(path)
    img = w.read_region((good[0], good[1]), 0, (good[2], good[3]))

    assert img.size == (good[2], good[3]), f'size {img.size}'
    assert w.reopens == 0, f'a good read should not reopen, got {w.reopens}'
    assert w.holes == [], f'a good read should record nothing, got {w.holes}'
    alpha = np.array(img.getchannel('A'))
    assert alpha.max() > 0, 'expected real pixels; is --good-* over empty canvas?'
    w.close()
    print('[PASS] good read: passthrough, no reopen, no hole recorded')


# ── 2. a failed read becomes a blank ─────────────────────────────────────────

def validate_bad_read(path, bad):
    """Subdivision recovers tissue that the whole-rect blank threw away.

    Scored against a deliberately wrong alternative rather than a threshold:
    min_chunk=0 disables subdivision and IS the pre-cfbef2d behaviour, by
    SafeSlide's own documentation. So the same rect is read twice and the two
    results are compared. A margin over a decoy is evidence; a tolerance on one
    number would only be a guess about how much this particular hole costs.

    What each side must show is forced by the geometry, not chosen:

        min_chunk=0    every pixel transparent, exactly one rect abandoned, and
                       that rect is the whole request
        min_chunk=64   some pixels real (that is the recovery) AND some still
                       transparent (there IS a hole, or the comparison is
                       measuring nothing), many small rects abandoned, each
                       inside the request

    reopens is NOT asserted equal to the number of holes. Subdivision fails at
    every node of the descent and heals at each one, so reopens counts the cost
    of finding the hole while holes counts the damage. Asserting one from the
    other is what the previous version of this test did.
    """
    # ── the decoy: subdivision off, which is what cfbef2d replaced ──────────
    w0 = SafeSlide(path, min_chunk=0, warn=False)
    img0 = w0.read_region((bad[0], bad[1]), 0, (bad[2], bad[3]))
    alpha0 = np.array(img0.getchannel('A'))

    assert img0.size == (bad[2], bad[3]), f'size {img0.size}'
    assert img0.mode == 'RGBA', f'mode {img0.mode}'
    assert alpha0.max() == 0, (
        'with min_chunk=0 the whole rect must be blank; if it is not, this rect '
        'no longer fails and --bad-* needs pointing somewhere that does')
    assert len(w0.holes) == 1, f'expected one abandoned rect, got {w0.holes}'
    assert w0.holes[0] == (bad[0], bad[1], 0, bad[2], bad[3]), \
        f'abandoned rect recorded as {w0.holes[0]}'
    w0.close()

    # ── production: subdivision on ──────────────────────────────────────────
    w = SafeSlide(path)
    img = w.read_region((bad[0], bad[1]), 0, (bad[2], bad[3]))
    alpha = np.array(img.getchannel('A'))

    assert img.size == (bad[2], bad[3]), f'size {img.size}'
    assert img.mode == 'RGBA', f'mode {img.mode}'

    real = int((alpha > 0).sum())
    total = int(alpha.size)
    assert real > 0, (
        f'subdivision recovered nothing: all {total} px are still blank, the '
        f'same answer min_chunk=0 gave. Either this rect has genuinely no '
        f'image anywhere -- in which case it cannot demonstrate subdivision, '
        f'and --bad-* should straddle the edge of a hole -- or the descent is '
        f'not reaching readable tiles.')
    assert real < total, (
        f'every pixel came back real, so this read never hit a hole and the '
        f'comparison above measured nothing. Point --bad-* at a rect that '
        f'actually fails.')

    assert len(w.holes) > 1, (
        f'subdivision recovered pixels but abandoned {len(w.holes)} rect(s); '
        f'recovery means the request was split, so more than one leaf should '
        f'have been given up on')
    x0, y0, bw, bh = bad
    for hx, hy, hlv, hw, hh in w.holes:
        assert hlv == 0, f'level {hlv} recorded for a level-0 read'
        assert x0 <= hx and hx + hw <= x0 + bw, f'abandoned rect {hx}+{hw} escapes x'
        assert y0 <= hy and hy + hh <= y0 + bh, f'abandoned rect {hy}+{hh} escapes y'
        assert max(hw, hh) <= w._min_chunk, \
            f'abandoned {hw}x{hh}, above the min_chunk={w._min_chunk} floor'
    assert w.reopens >= len(w.holes), \
        f'{w.reopens} reopens for {len(w.holes)} abandoned rects; every ' \
        f'abandoned rect failed at least once'

    lost = sum(hw * hh for _, _, _, hw, hh in w.holes)
    print(f'[PASS] bad read: min_chunk=0 blanks all {total} px; min_chunk='
          f'{w._min_chunk} recovers {real} ({100.0 * real / total:.1f}%), '
          f'abandoning {len(w.holes)} rects = {lost} px, {w.reopens} reopens')
    w.close()


# ── 3. the object survives ───────────────────────────────────────────────────

def validate_survives(path, good, bad):
    w = SafeSlide(path)
    n_levels = w.level_count
    dims = w.dimensions

    w.read_region((bad[0], bad[1]), 0, (bad[2], bad[3]))      # kill it

    # Metadata: the exact calls that raise on a bare handle.
    assert w.level_count == n_levels, 'level_count changed after a heal'
    assert w.dimensions == dims, 'dimensions changed after a heal'
    assert len(w.level_downsamples) == n_levels
    assert w.properties.get('openslide.mpp-x') is not None, \
        'properties map is bound to the dead handle'

    # And reading still works. What matters is that the GOOD read adds nothing,
    # so the count is compared against itself rather than against 1: the failed
    # read above reopened once per node of the subdivision descent.
    before = w.reopens
    assert before > 0, 'the bad read should have reopened at least once'
    img = w.read_region((good[0], good[1]), 0, (good[2], good[3]))
    assert np.array(img.getchannel('A')).max() > 0, \
        'a good read after a heal came back blank'
    assert w.reopens == before, \
        f'the good read reopened again ({before} -> {w.reopens})'
    w.close()
    print('[PASS] survives: metadata and reads both work after a failure')


# ── 4. every holder of the object recovers ───────────────────────────────────

def validate_shared(path, good, bad):
    """Healing replaces _osr INSIDE the shared object, so aliases recover.

    This is what a helper function returning a fresh OpenSlide could not do,
    and it is what lets LocaScopePipeline pass one slide to TissuesRegionsMask,
    TileSampler and WsiTissuesContainer at once.
    """
    w = SafeSlide(path)
    alias = w                       # stands in for another stage holding it

    w.read_region((bad[0], bad[1]), 0, (bad[2], bad[3]))
    before = w.reopens

    assert alias.level_count == w.level_count
    img = alias.read_region((good[0], good[1]), 0, (good[2], good[3]))
    assert np.array(img.getchannel('A')).max() > 0, \
        'the second holder did not recover'
    assert alias is w, 'the alias stopped being the same object'
    assert alias.reopens == before, \
        f'reading through the alias reopened ({before} -> {alias.reopens})'
    w.close()
    print('[PASS] shared: a second holder of the same object recovers too')


# ── 5. holes come back white, not black ──────────────────────────────────────

def validate_rgb_background(path, bad):
    """The hole composites to background; the recovered tissue does not.

    Located by the alpha mask rather than by assuming the whole rect is a hole.
    That assumption was true before subdivision and is what the old version
    asserted -- `(rgb == expected).all()` -- which now fails on the tissue
    cfbef2d recovers. The proposition worth testing was never `the whole rect
    is white`; it is `the part with no image is background rather than black`.
    """
    # The FULL bad rect, not a small read at its corner: the hole is somewhere
    # inside the region, so a 256 px read anchored at the same origin succeeds
    # and would be testing real tissue instead. `good` deliberately IS that
    # corner, which makes the mistake easy to make.
    rgb_shape = (bad[3], bad[2], 3)
    w = SafeSlide(path, warn=False)
    rgb, valid = w.read_region_valid((bad[0], bad[1]), 0, (bad[2], bad[3]))

    assert w.reopens > 0, f'expected the read to fail, reopens={w.reopens}'
    assert rgb.shape == rgb_shape, f'shape {rgb.shape}'
    assert rgb.dtype == np.uint8
    assert not valid.all(), 'no hole in this rect; --bad-* must point at one'

    # White is the openslide default background; a slide declaring another
    # colour would give that instead, so compare against what the slide says.
    expected = np.array(
        [int(w.background_color[1:][i:i+2], 16) for i in (0, 2, 4)],
        dtype=np.uint8)
    holes = rgb[~valid]
    assert (holes == expected).all(), (
        f'{int((holes != expected).any(axis=1).sum())} hole pixels are not the '
        f'background {tuple(int(v) for v in expected)}; first is '
        f'{tuple(int(v) for v in holes[(holes != expected).any(axis=1)][0])}. '
        f'This is the .convert("RGB") black-hole bug.')
    w.close()
    print(f'[PASS] rgb: {len(holes)} hole px composited to background '
          f'{tuple(int(v) for v in expected)}, not black')


# ── 6. alpha marks the hole ──────────────────────────────────────────────────

def validate_valid_mask(path, good, bad):
    """alpha separates real pixels from holes, WITHIN one read.

    `not valid_bad.any()` held only while a failure blanked the whole rect. The
    mask has to distinguish now, which is a stronger claim than the one it
    replaced: both values must appear in the same array, and the invalid part
    must agree with what read_region recorded as abandoned.
    """
    w = SafeSlide(path, warn=False)

    # Again the full rect -- see the note in validate_rgb_background.
    _, valid_bad = w.read_region_valid((bad[0], bad[1]), 0, (bad[2], bad[3]))
    assert valid_bad.shape == (bad[3], bad[2])
    assert w.reopens > 0, f'expected the read to fail, reopens={w.reopens}'
    assert not valid_bad.all(), 'a rect with a hole cannot be entirely valid'
    assert valid_bad.any(), (
        'the whole rect is invalid, so the mask is not separating anything; '
        'either subdivision recovered nothing here or it is not running')

    # The two are written by different code -- alpha comes from the image
    # openslide returned, holes from the recursion that gave up -- so their
    # agreement is evidence rather than a restatement.
    lost = sum(hw * hh for _, _, _, hw, hh in w.holes)
    invalid = int((~valid_bad).sum())
    assert invalid == lost, (
        f'{invalid} px are transparent but {lost} px were recorded as '
        f'abandoned; the mask and the hole list disagree about the same read')

    _, valid_good = w.read_region_valid((good[0], good[1]), 0, (256, 256))
    assert valid_good.all(), 'the good read must be entirely valid'
    w.close()
    print(f'[PASS] valid: alpha separates {invalid} hole px from '
          f'{int(valid_bad.sum())} real px, and agrees with holes')


def main():
    KI67 = '/work/u26130998/datasets/Ki67'
    ap = argparse.ArgumentParser()
    ap.add_argument('--wsi', default=f'{KI67}/S1103037,G7E,110122.mrxs')
    # tissue region index=2, whose full bbox read is the failure that blocked
    # the retriever build
    ap.add_argument('--bad-x', type=int, default=59264)
    ap.add_argument('--bad-y', type=int, default=30928)
    ap.add_argument('--bad-w', type=int, default=2816)
    ap.add_argument('--bad-h', type=int, default=2608)
    # its top-left corner, which reads fine
    ap.add_argument('--good-x', type=int, default=59264)
    ap.add_argument('--good-y', type=int, default=30928)
    ap.add_argument('--good-w', type=int, default=256)
    ap.add_argument('--good-h', type=int, default=256)
    args = ap.parse_args()

    if not os.path.exists(args.wsi):
        print(f'[SKIP] no such slide: {args.wsi}')
        return 1

    bad = (args.bad_x, args.bad_y, args.bad_w, args.bad_h)
    good = (args.good_x, args.good_y, args.good_w, args.good_h)
    print(f'slide : {args.wsi}')
    print(f'  hole: ({bad[0]}, {bad[1]}) {bad[2]}x{bad[3]}')
    print(f'  good: ({good[0]}, {good[1]}) {good[2]}x{good[3]}\n')

    assert issubclass(SafeSlide, openslide.OpenSlide), \
        'SafeSlide must stay an OpenSlide for isinstance checks downstream'

    validate_bare_handle_dies(args.wsi, bad)
    validate_good_read(args.wsi, good)
    validate_bad_read(args.wsi, bad)
    validate_survives(args.wsi, good, bad)
    validate_shared(args.wsi, good, bad)
    validate_rgb_background(args.wsi, bad)
    validate_valid_mask(args.wsi, good, bad)

    print('\nAll checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
