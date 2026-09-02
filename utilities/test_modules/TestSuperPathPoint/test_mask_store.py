#!/usr/bin/env python3
"""Tests for utilities/MaskStore.py.

    python utilities/test_modules/test_mask_store.py

No weights, no GPU, no WSI: the segmenter and the slide are both a handful of
lines at the bottom of this file. Temporary directories only.

WHAT THIS IS DEFENDING AGAINST
-------------------------------
The mask costs 3.5 to 6 minutes of GPU per slide and three later steps read it
(the sampling probe, the pre-tile extraction, and any bench that wants the same
regions the training tiles came from). So the failures worth pinning are the
ones that make a reader get the WRONG mask rather than no mask:

    a metadata field that comes back as a string
        `from __future__ import annotations` makes every annotation lazy, so
        `field.type is float` compares against the STRING 'float' and never
        matches. Every field then decodes as str, and the first caller to write
        `f'{meta.mask_ds:.0f}'` raises -- which is what happened, in the probe,
        after the store itself had written six correct files
    two segmenters sharing a filename
        `segmenter_id` folds the config, the encoder's weights and the encoder's
        configuration. If it stopped separating files, a rebuilt mask would
        overwrite one that a probe had already been run against
    a cache invalidated by a moved slide
        the opposite error, and the one that looks safe: hashing `wsi_path`
        would make every remount a full rebuild of six slides
    components read when nobody asked
        they are 30 to 40x the mask on disk. `with_components=False` has to read
        the header and one tensor, not the file

Sections:
  1. meta      -- identity, provenance, and the typed round trip
  2. store     -- write, read, refuse
  3. components -- the second tensor and what it costs to ignore
  4. build     -- what build_one does and what it declines to do
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

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

import numpy as np                                               # noqa: E402

import MaskStore                                                 # noqa: E402
from MaskStore import MaskMeta, MaskMismatch, SlideMask          # noqa: E402

_RESULTS = []
ROWS, COLS, K = 12, 20, 4


def check(name, fn):
    try:
        out = fn()
        _RESULTS.append((name, None))
        print(f'  ok    {name}' + (f'   {out}' if out else ''))
    except Exception as e:                                       # noqa: BLE001
        _RESULTS.append((name, e))
        print(f'  FAIL  {name}\n          {type(e).__name__}: {e}')


# ── stand-ins ────────────────────────────────────────────────────────────────

class FakeSlide:
    """Just enough of a slide for `MaskMeta.of`: a filename.

    `_filename` is openslide's own attribute, which is why `wsi_stem_of` reads
    it -- a stand-in that invented a different name would be testing a
    convention nothing else uses.
    """
    def __init__(self, path='/data/SLIDE_A.svs'):
        self._filename = path


class FakeSegmenter:
    """A slide segmenter: it has `mask_wsi`, so `build_one` will drive it."""
    def __init__(self, identity='seg1111', fraction=0.25, components=True):
        self.cfg = type('Cfg', (), {'method': 'fake-seg'})()
        self._identity = identity
        self._fraction = fraction
        self._components = components

    def identity_id(self):
        return self._identity

    def mask_wsi(self, wsi):
        rng = np.random.default_rng(0)
        mask = (rng.random((ROWS, COLS)) < self._fraction).astype(np.uint8)
        components = (rng.random((ROWS, COLS, K)).astype(np.float16)
                      if self._components else None)
        return SlideMask(mask=mask, origin=(1000, 2000), span=(ROWS * 14,
                                                               COLS * 14),
                         mask_ds=14.0, report={'cells': 7},
                         components=components)


class ImageSegmenter:
    """No `mask_wsi`. `build_one` must decline it and say where to go instead."""
    def identity_id(self):
        return 'img1111'


def _made() -> tuple:
    wsi, seg = FakeSlide(), FakeSegmenter()
    slide_mask = MaskStore.build_one(wsi, seg)
    return slide_mask, MaskMeta.of(slide_mask, wsi, seg)


# ══════════════════════════════════════════════════════════════════════════════
#  1. meta
# ══════════════════════════════════════════════════════════════════════════════

def t_the_round_trip_keeps_the_types():
    """THE REGRESSION. `mask_ds` must come back a float, not the string '14.0'.

    Asserted by formatting it the way the probe does, because that is where it
    surfaced: `f'{meta.mask_ds:.0f}'` on a str raises `Unknown format code 'f'`,
    and everything upstream of that print had already written correct files.
    """
    _, meta = _made()
    back = MaskMeta.from_strings(meta.to_strings())
    assert isinstance(back.mask_ds, float), type(back.mask_ds)
    assert isinstance(back.rows, int), type(back.rows)
    assert isinstance(back.fraction, float), type(back.fraction)
    assert isinstance(back.method, str), type(back.method)
    assert f'{back.mask_ds:.0f}' == '14'
    assert back.cfg_hash() == meta.cfg_hash(), 'the hash moved across a round trip'
    return f'mask_ds {back.mask_ds!r}, rows {back.rows!r}'

def t_every_identity_field_changes_the_filename():
    _, base = _made()
    moved = {'wsi_stem': 'SLIDE_B', 'method': 'hsv', 'segmenter_id': 'seg2222'}
    assert set(moved) == set(MaskStore._IDENTITY_FIELDS), (
        f'this test covers {sorted(moved)} but the module identifies on '
        f'{sorted(MaskStore._IDENTITY_FIELDS)}')
    import dataclasses
    for field, value in moved.items():
        other = dataclasses.replace(base, **{field: value})
        assert other.cfg_hash() != base.cfg_hash(), field
    return f'{len(moved)} fields'


def t_geometry_and_provenance_are_not_identity():
    """`mask_ds`, `origin` and `span` are DETERMINED by the segmenter and the
    slide. Hashing them as well would hide a disagreement between the two rather
    than catch it -- and `wsi_path` changing is a remount, not a new mask."""
    import dataclasses
    _, base = _made()
    for field, value in (('wsi_path', '/other/mount/SLIDE_A.svs'),
                         ('mask_ds', 32.0), ('origin_x', 0), ('span_w', 5),
                         ('rows', 99), ('fraction', 0.9),
                         ('n_components', 0),
                         ('created_at', '2020-01-01T00:00:00')):
        other = dataclasses.replace(base, **{field: value})
        assert other.cfg_hash() == base.cfg_hash(), (
            f'{field} moved the hash, so a remounted slide or a re-measured '
            f'geometry would invalidate six slides of GPU time')
    return 'wsi_path, geometry, counts, created_at'


# ══════════════════════════════════════════════════════════════════════════════
#  2. store
# ══════════════════════════════════════════════════════════════════════════════

def t_write_then_read_returns_the_same_mask():
    with tempfile.TemporaryDirectory() as root:
        slide_mask, meta = _made()
        path = MaskStore.save(root, slide_mask, meta)
        back, got = MaskStore.load(path)
        assert np.array_equal(back.mask, slide_mask.mask)
        assert back.origin == slide_mask.origin, (back.origin, slide_mask.origin)
        assert back.span == slide_mask.span
        assert back.mask_ds == slide_mask.mask_ds
        assert back.report == slide_mask.report, (back.report, slide_mask.report)
        assert got.cfg_hash() == meta.cfg_hash()
    return f'{ROWS}x{COLS}, origin and report survive'


def t_require_refuses_rather_than_falls_back():
    with tempfile.TemporaryDirectory() as root:
        slide_mask, meta = _made()
        path = MaskStore.save(root, slide_mask, meta)
        MaskStore.load(path, require={'method': 'fake-seg'})
        try:
            MaskStore.load(path, require={'method': 'hsv'})
        except MaskMismatch as e:
            assert 'method' in str(e), str(e)
            return 'refused, and named the field'
    raise AssertionError('a mismatched require was accepted')


def t_find_one_refuses_two_segmenters_of_one_slide():
    """A root legitimately holds two masks of the same slide from two
    segmenters. Picking whichever sorted first would sample tiles from a mask
    nobody chose."""
    with tempfile.TemporaryDirectory() as root:
        wsi = FakeSlide()
        for identity in ('seg1111', 'seg2222'):
            seg = FakeSegmenter(identity=identity)
            slide_mask = MaskStore.build_one(wsi, seg)
            MaskStore.save(root, slide_mask, MaskMeta.of(slide_mask, wsi, seg))
        try:
            MaskStore.find_one(root, wsi_stem='SLIDE_A')
        except MaskMismatch as e:
            assert '2 masks' in str(e), str(e)
        else:
            raise AssertionError('find_one picked one of two segmenters')
        assert MaskStore.find_one(root, segmenter_id='seg2222').exists()
    return 'ambiguous refused, segmenter_id resolves it'


def t_save_validates_the_meta_against_the_array():
    import dataclasses
    with tempfile.TemporaryDirectory() as root:
        slide_mask, meta = _made()
        for bad, why in ((dataclasses.replace(meta, rows=99), 'wrong rows'),
                         (dataclasses.replace(meta, segmenter_id=''),
                          'empty segmenter_id')):
            try:
                MaskStore.save(root, slide_mask, bad)
            except ValueError:
                continue
            raise AssertionError(f'{why} was accepted')
    return 'rows and segmenter_id both checked'


# ══════════════════════════════════════════════════════════════════════════════
#  3. components
# ══════════════════════════════════════════════════════════════════════════════

def t_components_ride_along_and_come_back_unchanged():
    """The field `background_threshold` is a question about, and the reason a
    threshold sweep costs seconds instead of 3.5 to 6 minutes of GPU a slide."""
    with tempfile.TemporaryDirectory() as root:
        slide_mask, meta = _made()
        assert meta.n_components == K, meta.n_components
        path = MaskStore.save(root, slide_mask, meta)
        back, _ = MaskStore.load(path, with_components=True)
        assert back.components is not None
        assert back.components.shape == (ROWS, COLS, K), back.components.shape
        assert np.array_equal(back.components, slide_mask.components)
    return f'{ROWS}x{COLS}x{K} float16, exact'


def t_the_default_read_does_not_return_them():
    """Off by default because they are 30 to 40x the mask, and almost every
    reader wants the bit."""
    with tempfile.TemporaryDirectory() as root:
        slide_mask, meta = _made()
        back, _ = MaskStore.load(MaskStore.save(root, slide_mask, meta))
        assert back.components is None, 'components were read unasked'
    return 'components is None'


def t_asking_for_absent_components_raises():
    """Not a None. A sweep that silently found nothing to sweep would report a
    flat curve, which reads as "the threshold does not matter"."""
    with tempfile.TemporaryDirectory() as root:
        wsi, seg = FakeSlide(), FakeSegmenter(components=False)
        slide_mask = MaskStore.build_one(wsi, seg)
        meta = MaskMeta.of(slide_mask, wsi, seg)
        assert meta.n_components == 0, meta.n_components
        path = MaskStore.save(root, slide_mask, meta)
        try:
            MaskStore.load(path, with_components=True)
        except MaskMismatch as e:
            assert 'no components' in str(e), str(e)
            return 'refused'
    raise AssertionError('a componentless store answered with_components')


def t_components_of_the_wrong_shape_are_refused():
    """They are two views of the same cell grid. A disagreement means one of
    them was cropped or transposed, and the threshold sweep would then be of a
    field that does not line up with the mask it produced."""
    import dataclasses
    with tempfile.TemporaryDirectory() as root:
        slide_mask, meta = _made()
        wrong = dataclasses.replace(
            slide_mask, components=np.zeros((COLS, ROWS, K), np.float16))
        try:
            MaskStore.save(root, wrong, meta)
        except ValueError as e:
            assert 'components' in str(e), str(e)
            return 'transposed components refused'
    raise AssertionError('a transposed components array was accepted')


# ══════════════════════════════════════════════════════════════════════════════
#  4. build
# ══════════════════════════════════════════════════════════════════════════════

def t_build_one_drives_a_slide_segmenter():
    slide_mask, meta = _made()
    assert slide_mask.shape == (ROWS, COLS), slide_mask.shape
    assert meta.wsi_stem == 'SLIDE_A', meta.wsi_stem
    assert meta.method == 'fake-seg', meta.method
    assert meta.segmenter_id == 'seg1111', meta.segmenter_id
    assert abs(meta.fraction - slide_mask.fraction) < 1e-9
    return f'fraction {meta.fraction:.2f}'


def t_build_one_declines_an_image_segmenter_and_says_where_to_go():
    """`from_wsi(wsi, method=...)` is the other path, and the error has to name
    it: "has no mask_wsi" tells a caller what is missing, not what to do."""
    try:
        MaskStore.build_one(FakeSlide(), ImageSegmenter())
    except TypeError as e:
        assert 'from_wsi' in str(e), str(e)
        return 'declined, and named the other path'
    raise AssertionError('build_one accepted an image segmenter')


# ══════════════════════════════════════════════════════════════════════════════

_SECTIONS = {
    'meta':       ['t_the_round_trip_keeps_the_types',
                   't_every_identity_field_changes_the_filename',
                   't_geometry_and_provenance_are_not_identity'],
    'store':      ['t_write_then_read_returns_the_same_mask',
                   't_require_refuses_rather_than_falls_back',
                   't_find_one_refuses_two_segmenters_of_one_slide',
                   't_save_validates_the_meta_against_the_array'],
    'components': ['t_components_ride_along_and_come_back_unchanged',
                   't_the_default_read_does_not_return_them',
                   't_asking_for_absent_components_raises',
                   't_components_of_the_wrong_shape_are_refused'],
    'build':      ['t_build_one_drives_a_slide_segmenter',
                   't_build_one_declines_an_image_segmenter_and_says_where_to_go'],
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='+', choices=sorted(_SECTIONS))
    args = ap.parse_args()

    for section in (args.only or sorted(_SECTIONS)):
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
