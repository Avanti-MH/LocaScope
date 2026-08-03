#!/usr/bin/env python3
"""Tests for SafeSlide against a slide with real holes.

There is no synthetic path here: the behaviour under test is how the openslide C
library reacts to a MIRAX cell the scanner never wrote, which cannot be faked.
The tests therefore need a slide that actually fails, and default to the one this
was diagnosed on.

  0. control      — a BARE OpenSlide really is destroyed by one failed read, so
                    the rest of the suite is testing something that matters
  1. good read    — an ordinary read is untouched, no reopen, no hole recorded
  2. bad read     — a failed read returns a fully transparent blank of the
                    requested size, and records the position
  3. survival     — metadata and further reads still work afterwards. This is
                    the whole point: a bare handle fails here
  4. sharing      — a second holder of the SAME object recovers too, which is
                    why LocaScopePipeline can hand one slide to every stage
  5. rgb          — a hole composites to the background colour (white), not the
                    black that .convert('RGB') produces
  6. valid        — the alpha channel marks the hole as unreal

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
    w = SafeSlide(path)
    img = w.read_region((bad[0], bad[1]), 0, (bad[2], bad[3]))

    assert img.size == (bad[2], bad[3]), \
        f'blank must match the requested size, got {img.size}'
    assert img.mode == 'RGBA', f'mode {img.mode}'
    alpha = np.array(img.getchannel('A'))
    assert alpha.max() == 0, 'blank must be fully transparent'

    assert w.reopens == 1, f'expected exactly one reopen, got {w.reopens}'
    assert len(w.holes) == 1, f'expected one hole, got {w.holes}'
    assert w.holes[0] == (bad[0], bad[1], 0, bad[2], bad[3]), \
        f'hole recorded as {w.holes[0]}'
    w.close()
    print('[PASS] bad read: transparent blank of the right size, position logged')


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

    # And reading still works.
    img = w.read_region((good[0], good[1]), 0, (good[2], good[3]))
    assert np.array(img.getchannel('A')).max() > 0, \
        'a good read after a heal came back blank'
    assert w.reopens == 1, f'the good read should not have reopened again ({w.reopens})'
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

    assert alias.level_count == w.level_count
    img = alias.read_region((good[0], good[1]), 0, (good[2], good[3]))
    assert np.array(img.getchannel('A')).max() > 0, \
        'the second holder did not recover'
    assert alias.reopens == 1 and alias is w
    w.close()
    print('[PASS] shared: a second holder of the same object recovers too')


# ── 5. holes come back white, not black ──────────────────────────────────────

def validate_rgb_background(path, bad):
    # The FULL bad rect, not a small read at its corner: the hole is somewhere
    # inside the region, so a 256 px read anchored at the same origin succeeds
    # and would be testing real tissue instead. `good` deliberately IS that
    # corner, which makes the mistake easy to make.
    rgb_shape = (bad[3], bad[2], 3)
    w = SafeSlide(path)
    rgb = w.read_region_rgb((bad[0], bad[1]), 0, (bad[2], bad[3]))

    assert w.reopens == 1, f'expected the read to fail, reopens={w.reopens}'
    assert rgb.shape == rgb_shape, f'shape {rgb.shape}'
    assert rgb.dtype == np.uint8
    # White is the openslide default background; a slide declaring another
    # colour would give that instead, so compare against what the slide says.
    expected = tuple(int(w.background_color[1:][i:i+2], 16) for i in (0, 2, 4))
    got = tuple(int(v) for v in rgb.reshape(-1, 3)[0])
    assert got == expected, \
        f'hole came back {got}, expected background {expected}; ' \
        f'this is the .convert("RGB") black-hole bug'
    assert (rgb == np.array(expected, dtype=np.uint8)).all(), \
        'hole should be uniformly the background colour'
    w.close()
    print(f'[PASS] rgb: hole composited to background {expected}, not black')


# ── 6. alpha marks the hole ──────────────────────────────────────────────────

def validate_valid_mask(path, good, bad):
    w = SafeSlide(path)

    # Again the full rect -- see the note in validate_rgb_background.
    _, valid_bad = w.read_region_valid((bad[0], bad[1]), 0, (bad[2], bad[3]))
    assert valid_bad.shape == (bad[3], bad[2])
    assert not valid_bad.any(), 'a hole must be entirely invalid'
    assert w.reopens == 1, f'expected the read to fail, reopens={w.reopens}'

    _, valid_good = w.read_region_valid((good[0], good[1]), 0, (256, 256))
    assert valid_good.any(), 'a good read must be at least partly valid'
    w.close()
    print('[PASS] valid: alpha separates real pixels from holes')


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
